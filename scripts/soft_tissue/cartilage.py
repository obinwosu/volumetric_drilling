"""Auricular cartilage framework from the auricle's skin sheet (edited in Blender).

The skin sheet (``blender_edit``: the CT ear cut down in Blender) is one open surface, normals
outward. Over the free auricle it is two skins facing each other (front and back), over the
concha floor it may be one skin lying on the head. The "auricle" weight marks where cartilage
belongs (not in the lobule).

``clean_sheet`` keeps the main piece, pulls jutting vertices back into the surface and closes
pinholes, leaving the big openings (the base where the auricle meets the head, the meatus).

``framework`` builds the cartilage as a solid on a grid, from each point's nearest place on the
sheet (distance d, which side, that place's paint and cover):

- inside the sheet (behind its outward normal), and not off its edge;
- at least ``cover`` deep: skin + perichondrium (``auricle.COVER_MM``), thinned where the ear is
  thin so at least ``keep_share`` of its thickness stays cartilage (as ``auricle.framework``);
- at most ``cover + max_mm`` deep: where the sheet is two skins this never binds (the far skin's
  cover comes first, so the cartilage fills between them); where it is one skin it sets the
  cartilage's thickness;
- where the paint is auricle.

Holes under about 2 x ``CLOSE_MM`` in the result are closed (the CT skin's pits would
otherwise punch through thin cartilage), and its surface is smoothed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import auricle as au
import envelope as ev

GRID_MM = 0.2
MAX_MM = 1.5  # cartilage thickness under a single skin (auricular cartilage ~1-2 mm)
KEEP_SHARE = au.KEEP_SHARE  # where the ear is thin, cartilage keeps at least this share of it
PINHOLE_MM = 4.0  # holes smaller than this across are closed
CLOSE_MM = 1.0  # holes in the cartilage up to about twice this across are closed
SPIKE_MM = 0.8  # vertices further than this from their neighbours' average are pulled in


def _edges(t):
    return np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])


def _neighbour_mean(v, t):
    e = _edges(t)
    e = np.concatenate([e, e[:, ::-1]])
    n = len(v)
    m = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(m.sum(axis=1)).ravel(), 1)
    return (m @ v) / deg[:, None], deg


def boundary_loops(t):
    """Open edges' loops, as lists of vertex indices (unordered)."""
    es = np.sort(_edges(t), axis=1)
    u, c = np.unique(es, axis=0, return_counts=True)
    b = u[c == 1]
    if not len(b):
        return []
    n = int(b.max()) + 1
    k, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n)), directed=False)
    verts = np.unique(b)
    return [verts[lab[verts] == i].tolist() for i in np.unique(lab[verts])]


def _ordered_loop(t, loop):
    """The loop's vertices in order round it (None if it is not a simple cycle)."""
    es = np.sort(_edges(t), axis=1)
    u, c = np.unique(es, axis=0, return_counts=True)
    s = set(loop)
    b = [tuple(x) for x in u[c == 1] if x[0] in s]
    nb = {}
    for a, b_ in b:
        nb.setdefault(a, []).append(b_)
        nb.setdefault(b_, []).append(a)
    if any(len(x) != 2 for x in nb.values()):
        return None
    order = [loop[0]]
    prev = None
    while True:
        nxt = [x for x in nb[order[-1]] if x != prev]
        prev = order[-1]
        if nxt[0] == order[0]:
            return order
        order.append(nxt[0])
        if len(order) > len(loop):
            return None


def clean_sheet(v, t, w, pinhole_mm=PINHOLE_MM, spike_mm=SPIKE_MM, log=print):
    """The main piece, spikes pulled in, pinholes closed. Returns (v, t, w)."""
    v, t, w = np.asarray(v, float), np.asarray(t, np.int64), np.asarray(w, float)
    n = len(v)
    k, lab = connected_components(coo_matrix((np.ones(3 * len(t)), (_edges(t)[:, 0], _edges(t)[:, 1])), shape=(n, n)), directed=False)
    main = np.argmax(np.bincount(lab[t[:, 0]], minlength=k))
    t = t[lab[t[:, 0]] == main]
    used = np.unique(t)
    remap = np.full(n, -1)
    remap[used] = np.arange(len(used))
    v, w, t = v[used], w[used], remap[t]
    log(f"sheet: kept the main piece ({len(v)} of {n} vertices)")
    pulled = 0
    for _ in range(5):  # jutting vertices: back to their neighbours' average
        mean, _deg = _neighbour_mean(v, t)
        off = np.linalg.norm(v - mean, axis=1) > spike_mm
        if not off.any():
            break
        v[off] = mean[off]
        pulled += int(off.sum())
    log(f"sheet: pulled in {pulled} jutting vertices")
    closed = 0
    extra_v, extra_w, extra_t = [], [], []
    for loop in boundary_loops(t):
        pts = v[loop]
        if np.ptp(pts, axis=0).max() >= pinhole_mm:
            continue  # the base, the meatus: stay open
        order = _ordered_loop(t, loop)
        if order is None:
            continue
        c = len(v) + len(extra_v)
        extra_v.append(v[order].mean(axis=0))
        extra_w.append(float(w[order].mean()))
        for a, b in zip(order, order[1:] + order[:1]):
            extra_t.append((a, b, c))
        closed += 1
    if extra_v:
        v = np.concatenate([v, np.asarray(extra_v)])
        w = np.concatenate([w, np.asarray(extra_w)])
        ft = np.asarray(extra_t)
        # orient the fans like the faces around them: an edge used a->b by the sheet is used b->a by the fan
        es = {(int(a), int(b)) for a, b in _edges(t)}
        flip = np.array([(a, b) in es for a, b, _c in ft])
        ft[flip] = ft[flip][:, [1, 0, 2]]
        t = np.concatenate([t, ft])
    log(f"sheet: closed {closed} pinholes")
    area = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    t = t[area > 1e-8]  # no zero-area faces (they have no normal)
    return v, t.astype(np.int32), w


def framework(v, t, w, grid_mm=GRID_MM, cover_mm=au.COVER_MM, max_mm=MAX_MM, keep_share=KEEP_SHARE, log=print):
    """Cartilage solid (verts, tris) inside the painted part of the skin sheet. See the module doc."""
    import open3d as o3d

    import surface

    v, t, w = np.asarray(v, float), np.asarray(t, np.int32), np.asarray(w, float)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t))
    m.compute_vertex_normals()
    nrm = np.asarray(m.vertex_normals)
    # local thickness: along the inward normal to the other skin (none: a single skin)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(t.astype(np.uint32)))
    q = (v - 0.02 * nrm).astype(np.float32)
    th = sc.cast_rays(o3d.core.Tensor(np.concatenate([q, -nrm.astype(np.float32)], axis=1)))["t_hit"].numpy() + 0.02
    cover_v = np.where(np.isfinite(th), np.minimum(cover_mm, 0.5 * (1 - keep_share) * th), cover_mm)
    painted = w > 0.5
    lo = v[painted].min(axis=0) - cover_mm - max_mm - 1.0
    hi = v[painted].max(axis=0) + cover_mm + max_mm + 1.0
    shape = np.ceil((hi - lo) / grid_mm).astype(int) + 1
    log(f"cartilage grid {tuple(shape)} at {grid_mm} mm")
    axes = [lo[k] + grid_mm * np.arange(shape[k]) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    field = np.full(len(pts), 1.0, np.float32)
    reach = cover_mm + max_mm + 2 * grid_mm
    step = 2_000_000
    for i in range(0, len(pts), step):
        p = pts[i : i + step]
        r = sc.compute_closest_points(o3d.core.Tensor(p))
        cp = r["points"].numpy()
        fid = r["primitive_ids"].numpy().astype(np.int64)
        fn = r["primitive_normals"].numpy()
        off = p - cp
        d = np.linalg.norm(off, axis=1)
        near = d < reach
        if not near.any():
            continue
        tri = t[fid[near]]
        bary = r["primitive_uvs"].numpy()[near]
        bw = np.stack([1 - bary[:, 0] - bary[:, 1], bary[:, 0], bary[:, 1]], axis=1)
        cov = (cover_v[tri] * bw).sum(axis=1)
        paint = (w[tri] * bw).sum(axis=1)
        dn = d[near]
        side = np.einsum("ij,ij->i", off[near], fn[near]) / np.maximum(dn, 1e-6)  # -1: straight in behind the skin
        f = np.maximum.reduce([cov - dn, dn - (cover_mm + max_mm), side + 0.5, (0.5 - paint) * 4.0])
        sub = field[i : i + step]
        sub[near] = f
        field[i : i + step] = sub
    field = np.nan_to_num(field, nan=1.0).reshape(shape)
    # the CT skin's pits (segmentation noise) would punch the cover through thin cartilage:
    # close holes under ``CLOSE_MM`` in the solid, then smooth it
    solid = field < 0
    r = max(1, int(round(CLOSE_MM / grid_mm)))
    solid = ndimage.binary_closing(np.pad(solid, r + 1), iterations=r)[r + 1 : -r - 1, r + 1 : -r - 1, r + 1 : -r - 1]
    field = 0.5 - ndimage.gaussian_filter(solid.astype(np.float32), 1.0)
    fv, ft = ev.marching_cubes_field(field, np.full(3, grid_mm), lo)
    if not len(ft):
        return fv, ft
    fv = surface.smooth_surface(fv, ft, iterations=4, lam=0.5, max_step=0.5 * grid_mm)
    fv, ft = ev._largest_component(fv, ft, share=0.02)[:2]
    fv, n = keep_under_skin(fv, ft, v, t)
    log(f"cartilage: {len(ft)} tris ({n} vertices pulled back under the skin)")
    return fv, ft


def keep_under_skin(fv, ft, sheet_v, sheet_t, min_mm=0.3, iterations=5):
    """Cartilage vertices that poke out of the skin sheet (or come within ``min_mm`` of it) are
    pulled back under it, ``min_mm`` deep along the skin's normal. Returns (verts, count moved)."""
    import open3d as o3d

    fv = np.asarray(fv, float).copy()
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(sheet_v, np.float32)), o3d.core.Tensor(np.asarray(sheet_t, np.uint32)))
    moved = np.zeros(len(fv), bool)
    for _ in range(iterations):
        r = sc.compute_closest_points(o3d.core.Tensor(fv.astype(np.float32)))
        cp, n = r["points"].numpy().astype(float), r["primitive_normals"].numpy().astype(float)
        off = fv - cp
        d = np.linalg.norm(off, axis=1)
        side = np.einsum("ij,ij->i", off, n)
        under_face = np.abs(side) > 0.3 * d  # not off the sheet's edge ...
        bad = (under_face | (side > 0)) & (side > -min_mm) & (d < 8.0)  # ... unless past it on its outer side
        if not bad.any():
            break
        fv[bad] = cp[bad] - min_mm * n[bad]
        moved |= bad
    return fv, int(moved.sum())


def clip_under_skin(fv, ft, skin_v, skin_t, min_mm=0.5, reach_mm=8.0):
    """The cartilage cut (not moved) where it comes out of the skin or within ``min_mm`` of it,
    over the skin's face or past its open edge on its outer side (as ``keep_under_skin``).
    Returns (verts, tris)."""
    import open3d as o3d

    fv, ft = np.asarray(fv, float), np.asarray(ft)
    if not len(ft) or not len(skin_t):
        return fv, ft
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(skin_v, np.float32)), o3d.core.Tensor(np.asarray(skin_t, np.uint32)))
    r = sc.compute_closest_points(o3d.core.Tensor(fv.astype(np.float32)))
    off = fv - r["points"].numpy()
    d = np.linalg.norm(off, axis=1)
    side = np.einsum("ij,ij->i", off, r["primitive_normals"].numpy())
    near = ((np.abs(side) > 0.3 * d) | (side > 0)) & (d < reach_mm)
    val = np.where(near, side + min_mm, -1.0)
    if not (val > 0).any():
        return fv, ft
    import auricle as au_

    cv, ct = au_.clip_mesh(fv, ft, val)
    return ev._largest_component(cv, ct, share=0.005)[:2]
