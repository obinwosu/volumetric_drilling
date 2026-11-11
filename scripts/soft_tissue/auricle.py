"""Auricle: cartilage framework under a thin skin-fat + perichondrium cover.

The ear mesh (ear.obj) is the outer skin surface. The cartilage framework is
the same closed mesh moved inward along each vertex normal by the cover
thickness (auricular skin with its thin fat, plus perichondrium), so the cover
is even everywhere. Where the ear is thin (helix rim, antihelix folds) the
inward move is capped so at least ``KEEP_SHARE`` of the local thickness is left
as cartilage, and the two sides never cross. The local thickness is the
distance along the inward normal to the opposite side of the mesh.

Same vertices and triangles as the input, so skin and framework correspond
one to one.

ear.obj is a solid block: a thin auricle on its face side and a filled bulge
behind it (no posterior surface). ``cut`` trims the bulge off along a gently
curved surface and closes the opening, so the kept part is the auricle; its
cut face is the auricle's posterior skin. The cut is set in the placement
viewer (depth, two tilts, two bends) and saved with the placement.

Not yet: the lobule has no cartilage. It needs its own region (picked later).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import sparse

AURICULAR_SKIN_MM = 0.8  # tissue_params skin "low": auricular skin, Danter et al. 1996
PERICHONDRIUM_MM = 0.15  # tissue_params perichondrium nominal
COVER_MM = AURICULAR_SKIN_MM + PERICHONDRIUM_MM
KEEP_SHARE = 0.3  # cartilage keeps at least this share of the local ear thickness


@dataclasses.dataclass
class Cut:
    """Keep the side of the surface z < depth + bend, in the ear's own axes.

    The axes are the ear mesh's principal axes: ``n`` the thin one, pointing to
    the bulge; ``e1`` the long one. Tilts turn ``n`` about e1 and e2; bends are
    the surface's sag (mm) 20 mm from the centre along e1 and e2 (positive bows
    toward the bulge).
    """

    depth_mm: float = 2.5
    tilt1_deg: float = 0.0
    tilt2_deg: float = 0.0
    bend1_mm: float = 0.0
    bend2_mm: float = 0.0
    enabled: bool = True


BEND_REACH_MM = 20.0


def ear_axes(verts, tris):
    """Centre, long axis e1, and thin axis n pointing to the side with less surface relief (the bulge)."""
    v = np.asarray(verts, np.float64)
    c = v.mean(axis=0)
    _u, _s, vt = np.linalg.svd(v - c, full_matrices=False)
    e1, n = vt[0], vt[2]
    t = (v - c) @ n
    # the face side (concha, helix) is the thin slab; the bulge reaches further out
    if abs(t.max()) < abs(t.min()):
        n = -n
    return c, e1, n


def ear_landmarks(verts, tris):
    """Where the ear model's parts are, in its own frame.

    superior   the long axis, toward the wider end (helix; the lobule end is narrow)
    face       out of the ear's face (the concha side), opposite the bulge behind it
    concha     floor of the cavum conchae: the deepest point of a depth map of the face,
               away from the outline. The meatus opens here.
    anterior   across the ear, toward the concha (tragus side)
    """
    import open3d as o3d
    from scipy import ndimage

    v = np.asarray(verts, np.float64)
    c, e1, n = ear_axes(v, tris)
    e2 = np.cross(n, e1)
    x = (v - c) @ e1
    span = x.max() - x.min()
    top = (v - c)[x > x.max() - 0.2 * span] @ e2
    bot = (v - c)[x < x.min() + 0.2 * span] @ e2
    superior = e1 if np.ptp(top) >= np.ptp(bot) else -e1
    face = -n
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(np.asarray(tris, np.uint32)))
    step = 0.4
    y = (v - c) @ e2
    A, B = np.meshgrid(np.arange(x.min(), x.max(), step), np.arange(y.min(), y.max(), step), indexing="ij")
    start = c + A[..., None] * e1 + B[..., None] * e2 + face * 40.0
    rays = np.concatenate([start.reshape(-1, 3), np.tile(-face, (start.size // 3, 1))], axis=1).astype(np.float32)
    th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(A.shape)
    hit = np.isfinite(th)
    inner = ndimage.binary_erosion(hit, iterations=int(4.0 / step))
    depth = np.where(hit, th, 0.0)
    smooth = ndimage.gaussian_filter(depth, 2.0) / np.maximum(ndimage.gaussian_filter(hit.astype(float), 2.0), 1e-6)
    smooth[~inner] = -np.inf
    i, j = np.unravel_index(np.argmax(smooth), smooth.shape)
    concha = start[i, j] - face * th[i, j]
    across = (concha - c) - ((concha - c) @ superior) * superior - ((concha - c) @ face) * face
    anterior = across / np.linalg.norm(across)
    return dict(center=c, superior=superior, face=face, anterior=anterior, concha=concha, length=float(span))


def _cut_frame(center, e1, n, cut: Cut):
    from placement_viewer import rot  # same Rodrigues helper

    e2 = np.cross(n, e1)
    nn = rot(e2, cut.tilt2_deg) @ rot(e1, cut.tilt1_deg) @ n
    f1 = e1 - (e1 @ nn) * nn
    f1 /= np.linalg.norm(f1)
    f2 = np.cross(nn, f1)
    return center, f1, f2, nn


def cut_value(p, frame, cut: Cut):
    o, f1, f2, nn = frame
    r = np.asarray(p, np.float64) - o
    x, y, z = r @ f1, r @ f2, r @ nn
    k1 = cut.bend1_mm / BEND_REACH_MM**2
    k2 = cut.bend2_mm / BEND_REACH_MM**2
    return z - cut.depth_mm - k1 * x * x - k2 * y * y


@dataclasses.dataclass
class CutResult:
    verts: np.ndarray
    tris: np.ndarray
    cap_tris: np.ndarray  # rows of tris that are the cut face
    loops: int  # cut outlines; more than one means the cut broke through the auricle
    removed_share: float  # share of the input surface area cut away


def _project(p, frame, cut, iters=3):
    nn = frame[3]
    for _ in range(iters):
        p = p - cut_value(p, frame, cut)[:, None] * nn
    return p


def cut(verts, tris, cut: Cut, axes=None) -> CutResult:
    """Trim the ear mesh to the kept side and close it with a cap on the cut surface."""
    v = np.asarray(verts, np.float64)
    t = np.asarray(tris, np.int64)
    if signed_volume(v, t) < 0:
        t = t[:, [0, 2, 1]]
    axes = axes or ear_axes(v, t)
    frame = _cut_frame(*axes, cut)
    s = cut_value(v, frame, cut)
    neg = s < 0
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    kept_tris = [t[neg[t].all(axis=1)]]
    crossing = t[neg[t].any(axis=1) & ~neg[t].all(axis=1)]
    new_pts, edge_id = [], {}

    def edge_point(i, j):
        key = (i, j) if i < j else (j, i)
        if key not in edge_id:
            a = s[i] / (s[i] - s[j])
            edge_id[key] = len(v) + len(new_pts)
            new_pts.append(v[i] + a * (v[j] - v[i]))
        return edge_id[key]

    extra = []
    for tri in crossing:
        poly = []
        for k in range(3):
            i, j = tri[k], tri[(k + 1) % 3]
            if neg[i]:
                poly.append(i)
            if neg[i] != neg[j]:
                poly.append(edge_point(i, j))
        for k in range(1, len(poly) - 1):
            extra.append((poly[0], poly[k], poly[k + 1]))
    pts = np.concatenate([v, _project(np.asarray(new_pts).reshape(-1, 3), frame, cut)]) if new_pts else v
    body = np.concatenate(kept_tris + [np.asarray(extra, np.int64).reshape(-1, 3)])
    cap, loops, interior = _cap(pts, body, frame, cut)
    pts = np.concatenate([pts, interior])
    all_tris = np.concatenate([body, cap])
    used = np.unique(all_tris)
    remap = np.full(len(pts), -1, np.int64)
    remap[used] = np.arange(len(used))
    out_t = remap[all_tris].astype(np.int32)
    removed = 1.0 - float(area[neg[t].all(axis=1)].sum() + 0.5 * area[neg[t].any(axis=1) & ~neg[t].all(axis=1)].sum()) / float(area.sum())
    return CutResult(pts[used], out_t, np.arange(len(body), len(all_tris)), loops, removed)


def _boundary_loops(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = e[cnt[inv.ravel()] == 1]  # directed boundary edges
    nxt = {int(a): int(c) for a, c in b}
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur)
            loop.append(cur)
            cur = nxt.get(cur)
            if cur is None:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _cap(pts, body, frame, cut_: Cut, spacing=0.8):
    """Triangulate the cut outline(s) on the cut surface, with interior points for an even mesh.

    Returns (faces, number of outlines, new interior points). Interior point k is
    vertex len(pts) + k.
    """
    import shapely
    from scipy.spatial import Delaunay

    o, f1, f2, nn = frame
    loops = _boundary_loops(body)
    if not loops:
        return np.zeros((0, 3), np.int64), 0, np.zeros((0, 3))

    def xy(idx):
        r = pts[idx] - o
        return np.stack([r @ f1, r @ f2], axis=1)

    rings = sorted(((shapely.Polygon(xy(np.array(lp))), lp) for lp in loops), key=lambda r: -abs(r[0].area))
    outers = []
    for poly, loop in rings:
        host = next((h for h in outers if h[0].contains(poly.representative_point())), None)
        if host is None:
            outers.append([poly, loop, []])
        else:
            host[2].append((poly, loop))
    faces, interior = [], []
    n_new = 0
    for poly, loop, holes in outers:
        shape = shapely.Polygon(poly.exterior.coords, [h[0].exterior.coords for h in holes]).buffer(0)
        bidx = np.array(loop + [i for h in holes for i in h[1]])
        bxy = xy(bidx)
        lo, hi = bxy.min(axis=0), bxy.max(axis=0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], spacing), np.arange(lo[1], hi[1], spacing), indexing="ij")
        g = np.stack([gx.ravel(), gy.ravel()], axis=1)
        inner = shape.buffer(-0.6 * spacing)
        g = g[shapely.contains_xy(inner, g[:, 0], g[:, 1])] if not inner.is_empty else g[:0]
        allxy = np.concatenate([bxy, g])
        tri = Delaunay(allxy).simplices
        cen = allxy[tri].mean(axis=1)
        tri = tri[shapely.contains_xy(shape, cen[:, 0], cen[:, 1])]
        ids = np.concatenate([bidx, len(pts) + n_new + np.arange(len(g))])
        faces.append(ids[tri])
        gp = o + g[:, :1] * f1 + g[:, 1:2] * f2 + nn * cut_.depth_mm
        interior.append(_project(gp, frame, cut_))
        n_new += len(g)
    new = np.concatenate(interior) if interior else np.zeros((0, 3))
    faces = np.concatenate(faces).astype(np.int64)
    allp = np.concatenate([pts, new])
    fn = np.cross(allp[faces[:, 1]] - allp[faces[:, 0]], allp[faces[:, 2]] - allp[faces[:, 0]])
    flip = fn @ nn < 0  # the cap faces +nn, away from the kept side
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces, len(loops), new


def clip_mesh(verts, tris, values):
    """Keep the part of a mesh where ``values`` (per vertex) < 0, cutting edges at the zero crossing.

    The cut edge is left open (no cap). Returns (verts, tris).
    """
    v = np.asarray(verts, np.float64)
    t = np.asarray(tris, np.int64)
    s = np.asarray(values, np.float64)
    neg = s < 0
    keep = [t[neg[t].all(axis=1)]]
    crossing = t[neg[t].any(axis=1) & ~neg[t].all(axis=1)]
    new_pts, edge_id, extra = [], {}, []
    for tri in crossing:
        poly = []
        for k in range(3):
            i, j = tri[k], tri[(k + 1) % 3]
            if neg[i]:
                poly.append(i)
            if neg[i] != neg[j]:
                key = (i, j) if i < j else (j, i)
                if key not in edge_id:
                    a = s[i] / (s[i] - s[j])
                    edge_id[key] = len(v) + len(new_pts)
                    new_pts.append(v[i] + a * (v[j] - v[i]))
                poly.append(edge_id[key])
        for k in range(1, len(poly) - 1):
            extra.append((poly[0], poly[k], poly[k + 1]))
    pts = np.concatenate([v, np.asarray(new_pts).reshape(-1, 3)])
    body = np.concatenate(keep + [np.asarray(extra, np.int64).reshape(-1, 3)])
    used = np.unique(body)
    remap = np.full(len(pts), -1, np.int64)
    remap[used] = np.arange(len(used))
    return pts[used], remap[body].astype(np.int32)


@dataclasses.dataclass
class Framework:
    verts: np.ndarray
    tris: np.ndarray
    offset_mm: np.ndarray  # how far each vertex moved in
    thickness_mm: np.ndarray  # local ear thickness along the inward normal (inf if none)
    capped_share: float  # share of vertices whose cover was thinned to fit


def vertex_normals(verts, tris):
    v = np.asarray(verts, np.float64)
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n


def signed_volume(verts, tris):
    v = np.asarray(verts, np.float64)
    return float(np.einsum("ij,ij->i", v[tris[:, 0]], np.cross(v[tris[:, 1]], v[tris[:, 2]])).sum() / 6.0)


def local_thickness(verts, tris, normals):
    """Distance along the inward normal to the far side of the closed mesh."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(verts, np.float32)), o3d.core.Tensor(np.asarray(tris, np.uint32)))
    origin = verts - normals * 1e-3
    rays = np.concatenate([origin, -normals], axis=1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    return hit["t_hit"].numpy().astype(np.float64) + 1e-3


def _smooth_scalar(values, tris, n, iterations):
    a = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    b = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    m = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    m = m + m.T
    deg = np.maximum(np.asarray(m.sum(axis=1)).ravel(), 1.0)
    out = values.copy()
    for _ in range(iterations):
        out = 0.5 * out + 0.5 * (m @ out) / deg
    return out


def framework(verts, tris, cover_mm=COVER_MM, keep_share=KEEP_SHARE, smooth=8) -> Framework:
    verts = np.asarray(verts, np.float64)
    tris = np.asarray(tris, np.int64)
    if signed_volume(verts, tris) < 0:  # make normals point out
        tris = tris[:, [0, 2, 1]]
    normals = vertex_normals(verts, tris)
    w = local_thickness(verts, tris, normals)
    cap = np.where(np.isfinite(w), 0.5 * (1.0 - keep_share) * w, cover_mm)
    offset = np.minimum(cover_mm, cap)
    # smooth the offset so the cover thins gradually, never past the cap
    offset = np.minimum(_smooth_scalar(offset, tris, len(verts), smooth), cap)
    return Framework(verts - normals * offset[:, None], tris.astype(np.int32), offset, w, float(np.mean(cap < cover_mm)))
