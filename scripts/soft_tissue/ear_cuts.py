"""Hand-drawn cuts on the ear's skin and cartilage meshes.

A cut is an outline clicked on a mesh, a direction (the view direction when it
was made) and a depth. Everything inside the outline, seen along that
direction, is removed: either all the way through, or from the outline's
nearest point down to ``depth_mm``. The mesh stays closed: the cut's walls
become new surface (for the skin, the lining of the opening).

Cuts live in the ear model's own frame (ear.obj millimetres), so they move with
the ear when it is placed again. Skin cuts apply to the auricle (ear.obj after
the bulge trim); cartilage cuts apply to the framework. Cuts are applied in
order, on a signed-distance grid, and the result is meshed once.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage

import envelope as ev
import surface

GRID_MM = 0.25
TARGETS = ("skin", "cartilage")


@dataclasses.dataclass
class MeshCut:
    target: str  # "skin" or "cartilage"
    outline_obj: list  # clicked points, ear.obj mm
    direction_obj: list  # view direction (into the ear), ear.obj frame
    depth_mm: float = 0.0  # 0 = all the way through


def _sdf(verts, tris, pts):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(verts, np.float32)), o3d.core.Tensor(np.asarray(tris, np.uint32)))
    out = np.empty(len(pts), np.float32)
    step = 2_000_000
    for i in range(0, len(pts), step):
        out[i : i + step] = scene.compute_signed_distance(o3d.core.Tensor(pts[i : i + step].astype(np.float32))).numpy()
    return out


def _region(pts, cut: MeshCut):
    """Signed distance-like field, negative inside the cut region."""
    from PIL import Image, ImageDraw

    d = np.asarray(cut.direction_obj, float)
    d /= np.linalg.norm(d)
    a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    outline = np.asarray(cut.outline_obj, float)
    uv_o = np.stack([outline @ a, outline @ b], axis=1)
    uv_p = np.stack([pts @ a, pts @ b], axis=1)
    res = 0.1
    lo = uv_o.min(axis=0) - 2.0
    n = np.ceil((uv_o.max(axis=0) + 2.0 - lo) / res).astype(int) + 1
    img = Image.new("L", (int(n[0]), int(n[1])), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in (uv_o - lo) / res], fill=1)
    inside = np.asarray(img, dtype=bool).T
    sd2 = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * res
    field = ndimage.map_coordinates(sd2.astype(np.float32), ((uv_p - lo) / res).T, order=1, mode="constant", cval=1e3)
    if cut.depth_mm > 0:
        t = pts @ d
        t0 = float((outline @ d).min()) - 0.5  # start just outside the surface where it was drawn
        field = np.maximum.reduce([field, t0 - t, t - (t0 + 0.5 + cut.depth_mm)])
    return field


def apply_cuts(verts, tris, cuts, smooth=3, remove=None):
    """The mesh with the cuts removed. Returns (verts, tris); unchanged if there is nothing to cut.

    ``remove``: optional f(points) < 0 where material is also removed (e.g. deep to the scalp skin).
    """
    if not cuts and remove is None:
        return np.asarray(verts), np.asarray(tris)
    v = np.asarray(verts, np.float64)
    lo = v.min(axis=0) - 1.5
    hi = v.max(axis=0) + 1.5
    shape = np.ceil((hi - lo) / GRID_MM).astype(int) + 1
    axes = [lo[k] + GRID_MM * np.arange(shape[k]) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    field = _sdf(v, tris, pts)
    for c in cuts:
        field = np.maximum(field, -_region(pts, c))
    if remove is not None:
        field = np.maximum(field, -np.asarray(remove(pts), np.float32))
    out_v, out_t = ev.marching_cubes_field(field.reshape(shape), np.full(3, GRID_MM), lo)
    if len(out_t) and smooth:
        out_v = surface.smooth_surface(out_v, out_t, iterations=smooth, lam=0.5, max_step=0.5 * GRID_MM)
    return out_v, out_t


def deep_to_skin(scalp_skin, vol):
    """f(points): < 0 deep to the scalp's outer skin, > 0 outside; magnitude = distance to the skin.

    A point is deep if a ray from it going lateral (outward) meets the scalp skin, or if it is
    inside the bone. Over the canal opening and past the envelope's edge the ray escapes, so
    those points count as outside.
    """
    import open3d as o3d

    sv, st = (np.asarray(x) for x in scalp_skin)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(sv.astype(np.float32)), o3d.core.Tensor(st.astype(np.uint32)))
    lat = np.asarray(vol.frame.lateral, np.float32)

    def f(points):
        p = np.asarray(points, np.float64)
        out = np.empty(len(p), np.float32)
        step = 1_000_000
        for i in range(0, len(p), step):
            q = p[i : i + step].astype(np.float32)
            dist = scene.compute_distance(o3d.core.Tensor(q)).numpy()
            rays = np.concatenate([q, np.broadcast_to(lat, q.shape)], axis=1)
            behind = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())
            idx = np.round(p[i : i + step] / vol.mm).astype(int)
            ok = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
            bone = np.zeros(len(q), bool)
            bone[ok] = vol.solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
            out[i : i + step] = np.where(bone, -np.maximum(dist, 0.5), np.where(behind, -dist, dist))
        return out

    return f


def join_to_scalp(ear_skin, scalp_skin, deep_field):
    """Make the ear's skin rise out of the scalp's.

    ``ear_skin``: the placed ear's closed skin mesh (volume mm). ``deep_field``: f(points) < 0
    deep to the scalp's outer skin (``deep_to_skin``). The ear keeps only what
    stands outside the scalp; the scalp drops what lies inside the ear. Both are cut at the
    same line (where the two surfaces cross), so the edges meet. Returns (ear, scalp), open.
    """
    import open3d as o3d

    import auricle as au

    ev_, et = (np.asarray(x) for x in ear_skin)
    sv, st = (np.asarray(x) for x in scalp_skin)
    ear = au.clip_mesh(ev_, et, -deep_field(ev_))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(ev_.astype(np.float32)), o3d.core.Tensor(et.astype(np.uint32)))
    sd = scene.compute_signed_distance(o3d.core.Tensor(sv.astype(np.float32))).numpy()
    scalp = au.clip_mesh(sv, st, -sd)
    return ear, scalp


def widen(cut: MeshCut, mean_radius_mm: float) -> MeshCut:
    """The same cut, its outline scaled about its centre (across the cut direction) to a mean radius."""
    d = np.asarray(cut.direction_obj, float)
    d /= np.linalg.norm(d)
    o = np.asarray(cut.outline_obj, float)
    c = o.mean(axis=0)
    r = o - c
    along = np.outer(r @ d, d)
    across = r - along
    k = mean_radius_mm / max(np.linalg.norm(across, axis=1).mean(), 1e-9)
    return MeshCut(cut.target, (c + along + across * k).tolist(), cut.direction_obj, cut.depth_mm)


FUNNEL_RINGS = 24
FUNNEL_POINTS = 128


def _ring(points, centre, normal, n=FUNNEL_POINTS, ref=None):
    """Resample a closed outline to n points, ordered by angle about ``centre`` across ``normal``."""
    normal = normal / np.linalg.norm(normal)
    ref = ref if ref is not None else np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    ref = ref - (ref @ normal) * normal
    ref /= np.linalg.norm(ref)
    other = np.cross(normal, ref)
    r = np.asarray(points, float) - centre
    ang = np.arctan2(r @ other, r @ ref)
    order = np.argsort(ang)
    p = np.asarray(points, float)[order]
    a = ang[order]
    p = np.concatenate([p, p[:1]])
    a = np.concatenate([a, a[:1] + 2 * np.pi])
    want = np.linspace(-np.pi, np.pi, n, endpoint=False)
    want = np.where(want < a[0], want + 2 * np.pi, want)
    return np.stack([np.interp(want, a, p[:, k]) for k in range(3)], axis=1)


def eac_rim(vol, env):
    """The EAC's lateral opening as a ring of points at the canal's lateral end (volume mm)."""
    from scipy import ndimage as ndi

    f = vol.frame
    g = env.canal_grid
    edge = g & ~ndi.binary_erosion(g)
    ij = np.argwhere(edge)
    q = env.canal_lo + ij * 0.25
    t_end = (f.canal_lateral_end - f.canal_center) @ f.lateral
    return f.canal_center + q[:, :1] * f.anterior + q[:, 1:2] * f.superior + t_end * f.lateral


def funnel(vol, env, hole_outline, hole_dir, ear_closed):
    """Skin lining from the EAC's lateral rim to where the ear's meatus hole leaves the ear's back.

    ``hole_outline``: the meatus skin cut's outline (volume mm, on the ear's front surface);
    ``hole_dir``: its cut direction (into the ear); ``ear_closed``: the placed ear before cuts.
    The hole is a straight prism through the ear, so where it leaves the back its rim is the
    outline carried along ``hole_dir``. The funnel is lofted between that rim and the canal's,
    easing in and out, so it meets the hole's wall and the canal end edge to edge.
    Returns (open tube (verts, tris), closed tube (verts, tris)) for trimming the scalp skin.
    """
    import open3d as o3d

    f = vol.frame
    d = np.asarray(hole_dir, float)
    d /= np.linalg.norm(d)
    o = np.asarray(hole_outline, float)
    c_front = o.mean(axis=0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(ear_closed[0], np.float32)), o3d.core.Tensor(np.asarray(ear_closed[1], np.uint32)))
    start = c_front - d * 0.5
    hits = scene.list_intersections(o3d.core.Tensor(np.concatenate([start, d])[None].astype(np.float32)))
    ts = np.sort(hits["t_hit"].numpy())
    depth = float(ts[-1] - 0.5) if len(ts) else 10.0  # the last surface crossed is the ear's back
    c_back = c_front + d * depth
    top = o + np.outer((c_back - o) @ d, d)  # outline carried to the back, across the cut
    rim = eac_rim(vol, env)
    c_rim = rim.mean(axis=0)
    axis = c_back - c_rim
    ref = f.superior
    # order both rings by angle about the same axis, so point k faces the same way all along
    common = axis / max(np.linalg.norm(axis), 1e-9)
    ring_a = _ring(rim, c_rim, common, ref=ref)
    ring_b = _ring(top, c_back, common, ref=ref)
    # a gentle bow, posterosuperior, so the canal is not a straight loft
    tilt = np.radians(SLEEVE_GAP_TILT_DEG)
    bow = np.cos(tilt) * f.superior - np.sin(tilt) * f.anterior
    ax = axis / max(np.linalg.norm(axis), 1e-9)
    bow -= (bow @ ax) * ax
    bow /= max(np.linalg.norm(bow), 1e-9)
    rings = []
    for s in np.linspace(0.0, 1.0, FUNNEL_RINGS):
        w = s * s * (3 - 2 * s)
        rings.append((1 - w) * ring_a + w * ring_b + bow * FUNNEL_BOW_MM * np.sin(np.pi * s))
    V = np.concatenate(rings)
    n = FUNNEL_POINTS
    tris = []
    for j in range(FUNNEL_RINGS - 1):
        for k in range(n):
            a, b = j * n + k, j * n + (k + 1) % n
            c, e = a + n, b + n
            tris += [(a, b, c), (b, e, c)]
    T = np.asarray(tris, np.int32)
    # orient so normals face into the lumen? keep outward from the tube axis
    mid = V[: n * FUNNEL_RINGS].reshape(FUNNEL_RINGS, n, 3).mean(axis=1)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ring_of = T[:, 0] // n
    if np.mean(np.einsum("ij,ij->i", fn, V[T[:, 0]] - mid[ring_of])) < 0:
        T = T[:, [0, 2, 1]]
    # closed version (capped ends) for inside tests
    ca, cb = len(V), len(V) + 1
    Vc = np.concatenate([V, mid[:1], mid[-1:]])
    caps = [(ca, (k + 1) % n, k) for k in range(n)] + [(cb, (FUNNEL_RINGS - 1) * n + k, (FUNNEL_RINGS - 1) * n + (k + 1) % n) for k in range(n)]
    Tc = np.concatenate([T, np.asarray(caps, np.int32)])
    return (V, T), (Vc, Tc), float(np.linalg.norm(axis))


def meatus_cut(cuts, T, frame):
    """The skin cut nearest the canal axis (the meatus), or None."""
    best, best_d = None, np.inf
    for c in cuts:
        if c.target != "skin":
            continue
        o = np.asarray(c.outline_obj, float) @ T[:3, :3].T + T[:3, 3]
        r = o.mean(axis=0) - frame.canal_lateral_end
        dist = np.linalg.norm(r - (r @ frame.lateral) * frame.lateral)
        if dist < best_d:
            best, best_d = c, dist
    return best


def assemble(vol, env, auricle_base, framework, cuts, T, scalp_skin=None, join=True, make_funnel=True, meatus=None):
    """The ear as placed: cuts, joined to the scalp, cartilage trimmed, meatus and canal lining.

    auricle_base, framework: (verts, tris) in ear.obj mm. T: ear.obj mm -> volume mm.
    ``meatus``: a MeatusShape; when enabled, the concha is opened automatically into the canal
    (else a hand-drawn skin cut nearest the canal axis is lofted to the canal, if there is one).
    Returns dict: skin, cartilage (ear.obj mm), scalp, funnel, sleeve (volume mm; None when not
    made), funnel_skin_mm, meatus_depth_mm.
    """
    import open3d as o3d

    import auricle as au

    flip = (lambda t: np.asarray(t)[:, [0, 2, 1]]) if np.linalg.det(T[:3, :3]) < 0 else (lambda t: np.asarray(t))
    to_w = lambda v: np.asarray(v) @ T[:3, :3].T + T[:3, 3]  # noqa: E731
    Tinv = np.linalg.inv(T)
    to_obj = lambda v: np.asarray(v) @ Tinv[:3, :3].T + Tinv[:3, 3]  # noqa: E731
    skin_cuts = [c for c in cuts if c.target == "skin"]
    cart_cuts = [c for c in cuts if c.target == "cartilage"]
    join = join and scalp_skin is not None
    deep = deep_to_skin(scalp_skin, vol) if join else None
    base_w = (to_w(auricle_base[0]), flip(auricle_base[1]))
    region, depth, rings = None, None, None
    if meatus is not None and meatus.enabled:
        depth = meatus.depth_mm if meatus.depth_mm > 0 else concha_depth(vol, base_w) + 2.0
        region = meatus_region(vol, env, meatus, depth)
        rings = meatus_rings(vol, env, meatus, depth)
    skin_remove = (lambda P: region(to_w(P))) if region is not None else None
    skin = apply_cuts(*auricle_base, skin_cuts, remove=skin_remove)

    def cart_remove(P):
        w = to_w(P)
        parts = []
        if deep is not None:
            parts.append(deep(w))
        if region is not None:
            parts.append(region(w, au.COVER_MM))  # cartilage opens wider: skin lines its edge
        return np.minimum.reduce(parts)

    cart = apply_cuts(*framework, cart_cuts, remove=cart_remove if (deep is not None or region is not None) else None)
    skin_w = (to_w(skin[0]), flip(skin[1]))
    scalp = scalp_skin
    ear_w = join_to_scalp(skin_w, scalp_skin, deep)[0] if join else skin_w
    if join:
        scalp = join_to_scalp(skin_w, scalp_skin, deep)[1]
    if rings is None and make_funnel:
        mc = meatus_cut(cuts, T, vol.frame)
        if mc is not None:
            (fv, ft), _closed, _len = funnel(vol, env, to_w(mc.outline_obj), T[:3, :3] @ np.asarray(mc.direction_obj), base_w)
            rings = fv.reshape(FUNNEL_RINGS, FUNNEL_POINTS, 3)
    tube = sl = canal_skin = None
    if rings is not None and make_funnel:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(skin_w[0].astype(np.float32)), o3d.core.Tensor(np.asarray(skin_w[1], np.uint32)))
        # the lining runs from the canal until it enters the ear's flesh; from there the ear's
        # carved wall lines the meatus and its cartilage takes over from the sleeve
        base_scene = o3d.t.geometry.RaycastingScene()
        base_scene.add_triangles(o3d.core.Tensor(base_w[0].astype(np.float32)), o3d.core.Tensor(np.asarray(base_w[1], np.uint32)))
        nr, npts, _ = rings.shape
        sd_r = base_scene.compute_signed_distance(o3d.core.Tensor(rings.reshape(-1, 3).astype(np.float32))).numpy().reshape(nr, npts)
        inside = np.flatnonzero((sd_r < 0).mean(axis=1) > 0.5)
        if len(inside):
            rings = rings[: max(int(inside[0]) + 2, 2)]
        (fv, ft), closed = _loft(rings)
        sd = scene.compute_signed_distance(o3d.core.Tensor(fv.astype(np.float32))).numpy()
        tube = au.clip_mesh(fv, ft, -sd)
        if scalp is not None:  # open the scalp skin where the meatus passes through it
            cs = o3d.t.geometry.RaycastingScene()
            cs.add_triangles(o3d.core.Tensor(closed[0].astype(np.float32)), o3d.core.Tensor(closed[1].astype(np.uint32)))
            inside = cs.compute_signed_distance(o3d.core.Tensor(np.asarray(scalp[0], np.float32))).numpy()
            scalp = au.clip_mesh(scalp[0], scalp[1], -inside)
        sl, canal_skin = sleeve(vol, (fv, ft), rings=len(rings), n=rings.shape[1])
    return dict(skin=(to_obj(ear_w[0]), flip(ear_w[1])), cartilage=cart, scalp=scalp, funnel=tube,
                sleeve=sl, funnel_skin_mm=canal_skin, meatus_depth_mm=depth)


# Cartilaginous canal: an incomplete sleeve around the funnel (anterior and inferior walls,
# thinner posteriorly, missing posterosuperiorly), between the canal skin and the concha.
CANAL_SKIN_CART_MM = 1.0  # skin over the cartilaginous canal: thicker and more compliant than over bone
CANAL_SKIN_BONE_MM = 0.1  # very thin, tightly adherent skin over the bony canal (tissue_params bony_canal_skin)
SLEEVE_MM = 1.0  # cartilage thickness, before the gentle variation
SLEEVE_GAP_DEG = 130.0  # width of the posterosuperior deficiency
SLEEVE_GAP_TILT_DEG = 35.0  # the gap's centre: superior turned this far toward posterior
FUNNEL_BOW_MM = 1.5  # the canal's gentle curve


def _smooth_noise(n_rings, n_pts, seed=7, amp=0.25):
    """Gentle, irregular variation (1 +- amp), smooth along and around the canal."""
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, n_rings)[:, None]
    a = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)[None, :]
    out = np.zeros((n_rings, n_pts))
    for k in (1, 2, 3):
        ph = rng.uniform(0, 2 * np.pi, 2)
        out += np.sin(k * a + ph[0]) * np.cos(np.pi * k * s + ph[1]) / k
    return 1.0 + amp * out / np.abs(out).max()


def sleeve(vol, funnel_vt, rings=None, n=None):
    """The cartilaginous canal as a closed, incomplete sleeve around the (unclipped) funnel loft.

    Returns (verts, tris) in volume mm, and a per-funnel-vertex skin thickness (mm) that is
    thick over the sleeve and thins to the bony-canal value at the canal's rim.
    """
    f = vol.frame
    rings = rings or FUNNEL_RINGS
    n = n or FUNNEL_POINTS
    V = np.asarray(funnel_vt[0], float).reshape(rings, n, 3)
    centre = V.mean(axis=1)
    radial = V - centre[:, None, :]
    radial /= np.linalg.norm(radial, axis=2, keepdims=True)
    tilt = np.radians(SLEEVE_GAP_TILT_DEG)
    gap_dir = np.cos(tilt) * f.superior - np.sin(tilt) * f.anterior  # posterosuperior
    # canal walls are named about the canal's own axis (lateral): measure the gap there
    L = f.lateral
    across = radial - (radial @ L)[..., None] * L
    across /= np.maximum(np.linalg.norm(across, axis=2, keepdims=True), 1e-9)
    g = gap_dir - (gap_dir @ L) * L
    g /= np.linalg.norm(g)
    ang = np.degrees(np.arccos(np.clip(across @ g, -1, 1)))
    keep = np.median(ang, axis=0) > SLEEVE_GAP_DEG / 2  # one contiguous arc, the same for every ring
    # thinner over the posterior wall, with gentle irregular variation
    post = np.clip(-(radial @ f.anterior), 0, 1)
    thick = SLEEVE_MM * (1.0 - 0.4 * post) * _smooth_noise(rings, n)
    # skin: thick over the cartilaginous canal, easing to bony-canal skin at the rim (ring 0)
    s = np.linspace(0, 1, rings)[:, None]
    w = np.clip(s / 0.3, 0, 1)
    skin = CANAL_SKIN_BONE_MM + (CANAL_SKIN_CART_MM - CANAL_SKIN_BONE_MM) * (w * w * (3 - 2 * w))
    skin = np.broadcast_to(skin, (rings, n))
    inner = V + radial * skin[..., None]
    outer = V + radial * (skin + thick)[..., None]
    # rotate the angle index so the kept arc is contiguous from 0
    k0 = int(np.flatnonzero(~keep)[-1] + 1) % n if (~keep).any() else 0
    order = (np.arange(n) + k0) % n
    order = order[keep[order]]
    m = len(order)
    I, O = inner[:, order], outer[:, order]
    verts = np.concatenate([I.reshape(-1, 3), O.reshape(-1, 3)])
    off = rings * m
    idx = lambda j, k, outer_=False: j * m + k + (off if outer_ else 0)  # noqa: E731
    tris = []
    for j in range(rings - 1):
        for k in range(m - 1):
            for o_ in (False, True):
                a, b, c, d = idx(j, k, o_), idx(j, k + 1, o_), idx(j + 1, k, o_), idx(j + 1, k + 1, o_)
                tris += [(a, c, b), (b, c, d)] if not o_ else [(a, b, c), (b, d, c)]
    for j in (0, rings - 1):  # ends of the sleeve along the canal
        for k in range(m - 1):
            a, b, c, d = idx(j, k), idx(j, k + 1), idx(j, k, True), idx(j, k + 1, True)
            tris += [(a, b, c), (b, d, c)] if j == 0 else [(a, c, b), (b, c, d)]
    for k in (0, m - 1):  # the edges of the posterosuperior gap
        for j in range(rings - 1):
            a, b, c, d = idx(j, k), idx(j + 1, k), idx(j, k, True), idx(j + 1, k, True)
            tris += [(a, c, b), (b, c, d)] if k == 0 else [(a, b, c), (b, d, c)]
    T = np.asarray(tris, np.int32)
    import auricle as au

    if au.signed_volume(verts, T) < 0:
        T = T[:, [0, 2, 1]]
    return (verts, T), skin.reshape(-1)


# -- automatic meatus: the concha opens into the canal as a flared, sloping funnel ----------
@dataclasses.dataclass
class MeatusShape:
    """The concha's opening into the canal, anchored to the canal (not drawn).

    It starts as the EAC's lateral opening and flares outward going laterally, up to the
    concha floor. The posterior wall flares more than the anterior (tragal) wall, so the
    concha slopes down into the canal and funnels sound toward it.
    """

    enabled: bool = True
    flare_anterior_deg: float = 10.0
    flare_posterior_deg: float = 45.0
    flare_len_mm: float = 6.0  # the flare happens over this last stretch, at the concha
    # S-shaped course: going out from the EAC's opening the canal swings down and anterior to
    # the meatus (so, going in, the cartilaginous canal bends up and back)
    drop_inferior_mm: float = 3.0
    drop_anterior_mm: float = 2.0
    widen_mm: float = 0.5  # beyond the EAC's own outline
    depth_mm: float = 0.0  # lateral reach from the canal's end; 0 = up to the concha floor + 2 mm


def _swing(shape: MeatusShape, s):
    """Offset of the canal's centre (anterior, superior mm) at fraction s of the way out to the meatus."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    w = s * s * (3 - 2 * s)  # leaves the bony canal straight, arrives at the meatus straight
    return np.stack([shape.drop_anterior_mm * w, -shape.drop_inferior_mm * w], axis=-1)


def _canal_maps(env):
    from scipy import ndimage as ndi

    g = env.canal_grid
    sd2 = (ndi.distance_transform_edt(~g) - ndi.distance_transform_edt(g)) * 0.25
    centroid = env.canal_lo + np.argwhere(g).mean(axis=0) * 0.25
    return sd2.astype(np.float32), centroid


def _flare_tan(shape: MeatusShape, q, centroid, ant2):
    """tan(flare) by direction around the canal: anterior wall steep, posterior wall sloping."""
    r = q - centroid
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)
    w = 0.5 * (1.0 - r @ ant2)  # 0 facing anterior, 1 facing posterior
    ta, tp = np.tan(np.radians(shape.flare_anterior_deg)), np.tan(np.radians(shape.flare_posterior_deg))
    return ta * (1 - w) + tp * w


def meatus_region(vol, env, shape: MeatusShape, depth_mm):
    """f(points, extra_mm=0) < 0 inside the flared meatus (grown by ``extra_mm``), volume mm."""
    f = vol.frame
    sd2, centroid = _canal_maps(env)
    t_end = float((f.canal_lateral_end - f.canal_center) @ f.lateral)
    ant2 = np.array([1.0, 0.0])  # the canal maps use (anterior, superior) coordinates

    def region(points, extra_mm=0.0):
        p = np.asarray(points, float) - f.canal_center
        t = p @ f.lateral
        q = np.stack([p @ f.anterior, p @ f.superior], axis=1)
        q = q - _swing(shape, (t - t_end) / max(depth_mm, 1e-6))  # follow the canal's S-curve
        d2 = ndimage.map_coordinates(sd2, ((q - env.canal_lo) / 0.25).T, order=1, mode="constant", cval=1e3)
        start = t_end + max(depth_mm - shape.flare_len_mm, 0.0)  # the canal is near a tube until the concha
        grow = np.maximum(t - start, 0.0) * _flare_tan(shape, q, centroid, ant2)
        field = d2 - shape.widen_mm - extra_mm - grow
        return np.maximum.reduce([field, (t_end - 3.0) - t, t - (t_end + depth_mm)])

    return region


def meatus_rings(vol, env, shape: MeatusShape, depth_mm, n_rings=FUNNEL_RINGS, n=FUNNEL_POINTS):
    """The meatus wall as rings from the canal's rim out to ``depth_mm``: its lining (volume mm)."""
    f = vol.frame
    _sd2, centroid = _canal_maps(env)
    rim = eac_rim(vol, env)
    c_rim = rim.mean(axis=0)
    ring0 = _ring(rim, c_rim, f.lateral, n=n, ref=f.superior)
    q0 = np.stack([(ring0 - f.canal_center) @ f.anterior, (ring0 - f.canal_center) @ f.superior], axis=1)
    out_dir = q0 - centroid
    out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True)
    tan = _flare_tan(shape, q0, centroid, np.array([1.0, 0.0]))
    rings = []
    start = max(depth_mm - shape.flare_len_mm, 0.0)
    for s in np.linspace(0.0, 1.0, n_rings):
        t = s * depth_mm
        q = q0 + out_dir * (shape.widen_mm + max(t - start, 0.0) * tan)[:, None] + _swing(shape, s)
        rings.append(f.canal_center + q[:, :1] * f.anterior + q[:, 1:2] * f.superior
                     + ((ring0 - f.canal_center) @ f.lateral + t)[:, None] * f.lateral)
    return np.stack(rings)


def _loft(rings):
    """Tube through rings (n_rings, n, 3): (verts, tris) open, and a capped closed copy."""
    nr, n, _ = rings.shape
    V = rings.reshape(-1, 3)
    tris = []
    for j in range(nr - 1):
        for k in range(n):
            a, b = j * n + k, j * n + (k + 1) % n
            tris += [(a, b, a + n), (b, b + n, a + n)]
    T = np.asarray(tris, np.int32)
    mid = rings.mean(axis=1)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    if np.mean(np.einsum("ij,ij->i", fn, V[T[:, 0]] - mid[T[:, 0] // n])) < 0:
        T = T[:, [0, 2, 1]]
    ca, cb = len(V), len(V) + 1
    caps = [(ca, (k + 1) % n, k) for k in range(n)] + [(cb, (nr - 1) * n + k, (nr - 1) * n + (k + 1) % n) for k in range(n)]
    Vc = np.concatenate([V, mid[:1], mid[-1:]])
    Tc = np.concatenate([T, np.asarray(caps, np.int32)])
    import auricle as au

    if au.signed_volume(Vc, Tc) < 0:
        Tc = Tc[:, [0, 2, 1]]
    return (V, T), (Vc, Tc)


def concha_depth(vol, ear_closed_w):
    """How far lateral of the canal's end the ear's concha floor lies (along the canal axis)."""
    import open3d as o3d

    f = vol.frame
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(ear_closed_w[0], np.float32)), o3d.core.Tensor(np.asarray(ear_closed_w[1], np.uint32)))
    ray = np.concatenate([f.canal_lateral_end, f.lateral])[None].astype(np.float32)
    hits = np.sort(scene.list_intersections(o3d.core.Tensor(ray))["t_hit"].numpy())
    # the concha floor is the last surface crossed going out (the ear's face), not its back
    return float(hits[-1]) if len(hits) else 10.0
