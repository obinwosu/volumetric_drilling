"""The auricle's skin sheet (from Blender) joined to the scalp skin, the canal and the cartilage.

The sheet (``blender_edit``) is the auricle's skin as one open surface, normals outward: two
skins over the free auricle, one over parts of the concha, open at its base (where it meets the
head) and at the meatus (a short tube that ends inside the EAC). ``join_sheet`` makes it the
ear's skin in the volume:

1. The envelope meets the auricle at its base (``EnvelopeMeet``): where the sheet's base edge
   stands lateral of the envelope, the envelope (its skin and the layers under it) rises gently
   to it, the full rise at the base and fading out over ``RISE_MM``; the auricle's own contours
   and the auriculocephalic angle are left alone (no skirt).
2. The envelope's skin gives way under the auricle (inside the base outline, with the auricle's
   skin over it: the auricle's own skin is there, over cartilage, thin fascia and periosteum). Nothing of the envelope is left on
   the auricle's outer side (in the concha's bowl, or poking through it): the auricle is a
   projection of the side of the head.
3. The canal's skin is grown from the sheet's meatus tube (whose end sits inside the EAC): ring
   after ring down the canal's centreline, pulled sideways onto the EAC's wall on the lumen's
   side of the bone, easing in over the first few mm, smoothed, and open at the eardrum at the
   canal's full width (the middle ear shows); the sheet near the meatus is smoothed too. Its first
   ring is the tube's own edge, so the meatus runs on into the canal with no gap and no bone
   showing; water poured into the auricle leaves only down the canal (``meatus_leaks``).
4. The cartilage framework is built from the sheet (``cartilage``), inside the painted auricle,
   or is the one edited in Blender.
5. Where the cartilage carries on past the sheet's base edge (the back and top of the auricle,
   where the sheet is open underneath), skin closes the auricle (``medial_skin``): from the
   sheet's base edge down to where the cartilage reaches the envelope (the postauricular crease,
   the root of the helix), stretched smooth across the opening (``stretch``, no lumps), with the
   cartilage cut back under it. There the envelope does not rise to the sheet's base edge, and
   its skin stays over the fascia wherever an auricle skin faces down onto it: the cleft between
   the auricle and the head (skin, cartilage, skin, cleft, skin, fascia, periosteum, bone).
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

import auricle as au
import cartilage as ca
import ear_cuts as ec
import envelope as ev

SKIRT_TOWARD_HEAD = 0.7  # how much the skirt heads medially (0: straight on along the sheet)
SKIRT_RINGS = 8
SKIRT_MAX_MM = 30.0  # furthest the skirt reaches to the scalp skin
FRONT_REACH_MM = 15.0  # scalp skin this close in front of the auricle's outward-facing skin is not landed on
SKIRT_FRONT_MM = 3.0  # the skirt runs behind this line (mm anterior of the canal's end) ...
SKIRT_ABOVE_MM = 10.0  # ... or above this (mm superior of it): behind and above the ear only
STRETCH_MM = 2.0  # neighbouring skirt strips may land at most this much further apart than they start
CONCHA_AIR_MM = 8.0  # no skirt where it would hang this close in front of outward-facing skin (the concha)
BASE_TRIM_MM = 0.3  # the cartilage stops this far short of the sheet's base edge
TOUCH_MM = 0.8  # envelope points this close outside the auricle's skin count as meeting it (kept)
RISE_MM = 12.0  # the envelope rises to the auricle's base over this distance outside it
RISE_NEIGHBOURS = 16  # base points blended for the base's height (no stripes)
LANDING_OUT_MM = 1.5  # base points already under the scalp skin land this far out along it
LINING_OVERLAP_MM = 1.0
BONE_CLEAR_MM = 0.3  # the skin near the meatus and the canal's skin keep this far off the bone
ANNULUS_SHARE = 0.15  # the canal ends when this share of directions is still on bony wall (the rest reached the eardrum)
TM_MARGIN_MM = 1.0  # the canal's skin stops this far short of the EAC's medial end (open there)
CANAL_SMOOTH = 12  # smoothing passes over the grown canal
MEATUS_SMOOTH_MM = 6.0  # the sheet is smoothed this close to the meatus tube's end ...
MEATUS_SMOOTH = 8  # ... with this many passes
CANAL_STEP_MM = 0.5  # ring spacing down the grown canal
CANAL_BLEND_MM = 3.0  # the grown canal eases onto the EAC's wall over this length
BRIDGE_GAP_MM = 1.5
BRIDGE_RINGS = 4  # rings between the meatus tube's end and the canal lining
BRIDGE_CLEAR_MM = 0.25  # ... kept this far off the bone  # the lining stops this far short of the meatus tube's end; a strip joins them
LINING_MM = 0.2
MEDIAL_GRID_MM = 0.25  # grid for the skin over the cartilage's medial face
MEDIAL_COVERED_MM = 0.5  # cartilage this much further than the cover from the sheet is under the sheet already
CONTINUED_MM = 2.5  # base edge points this close to the medial skin are carried on by it (no rise there)
SEAM_MM = 2.0  # the medial skin's edge this close to the sheet's base edge is moved onto it
DRAPE_MM = 1.5  # the medial skin drapes over the cartilage like a ball this big rolled over it (no beads over its knobs)
EDGE_SMOOTH = 6  # passes easing the sheet's base edge (no sawtooth where the medial skin meets it)
FRONT_SLOPE_MM = 8.0  # the skin slopes off the auricle's front edge onto the face over at least this distance
FRONT_ROWS = 14
FRONT_BELOW_MM = -12.0  # the front slope stops this far below the auricle's centre (the lobule is left as it is)
FRONT_RUN = 2.5  # mm forward per mm down: a gentle slope (about 20 degrees on average)
FRONT_HOLLOW = 2.0  # the slope's profile: height over the face falls off as (1 - s)^this (a gentle hollow)
FRONT_MAX_MM = 30.0  # rows of the front slope
CANAL_CART_MM = 0.5  # cartilage keeps this far outside the ear canal's wall
CANAL_CLEAR_MM = 3.0  # the medial skin keeps this far from the ear canal
SKIN_OVER_BONE_MM = 1.0  # the auricle's skin keeps this far off bone (skin, thin tissue, periosteum)
FILLET_MM = 4.0  # the face's skin rounds up onto the auricle's front edge over this distance
FILLET_BEHIND_MM = 3.0  # the envelope gives way this far behind the front edge (under the auricle)
FILLET_MAX_SLOPE = 1.5  # the steepest the fillet starts (rise per run)
PINHOLE_COUNT = 150  # removed patches of the envelope smaller than this (vertices) are kept: no holes punched at the junction
PIT_MM = 1.5  # sheet points this little under envelope tissue are a pit dipping through it (lifted out)
CLOSED_MM = 4.5  # cartilage over the envelope is enclosed (the concha's floor) when the auricle's skin closes over it this close


def _boundary_dirs(v, t, loop):
    """Unit vectors in the surface at each boundary vertex, pointing out past the edge."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t))
    m.compute_vertex_normals()
    n = np.asarray(m.vertex_normals)[loop]
    order = ca._ordered_loop(t, loop)
    if order is None:
        order = loop
    idx = {k: i for i, k in enumerate(order)}
    pts = v[order]
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    out = np.cross(tang, np.asarray(m.vertex_normals)[order])
    # point it away from the mesh: compare with the step from the neighbouring interior centroid
    nbr = np.zeros_like(pts)
    cnt = np.zeros(len(pts))
    for tri in t[np.isin(t, order).any(axis=1)]:
        for a in tri:
            if a in idx:
                c = v[tri].mean(axis=0)
                nbr[idx[a]] += pts[idx[a]] - c
                cnt[idx[a]] += 1
    nbr /= np.maximum(cnt, 1)[:, None]
    out *= np.sign(np.einsum("ij,ij->i", out, nbr))[:, None]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    # smooth along the loop
    for _ in range(5):
        out = 0.5 * out + 0.25 * (np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0))
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return order, out


def in_front_of(sheet_scene, points, lateral, reach_mm=None):
    """Points in front of the sheet's outward- and lateral-facing skin (the concha's air)."""
    import open3d as o3d

    p = np.asarray(points, np.float32)
    r = sheet_scene.compute_closest_points(o3d.core.Tensor(p))
    off = p - r["points"].numpy()
    d = np.linalg.norm(off, axis=1)
    fn = r["primitive_normals"].numpy()
    out = (np.einsum("ij,ij->i", off, fn) > 0.3 * d) & (fn @ np.asarray(lateral, float) > 0.3)
    return out & (d < (reach_mm or FRONT_REACH_MM))


def landing_skirt(v, t, loop, scalp, lateral, rings=SKIRT_RINGS, allow=None, drape=None):
    """Smooth strip from the base loop down onto the scalp skin. Returns (new verts, tris (indexing
    ``v`` then the new verts), landing points in loop order)."""
    import open3d as o3d

    order, t0 = _boundary_dirs(v, t, loop)
    p0 = v[order]
    lat = np.asarray(lateral, float)
    d = t0 - SKIRT_TOWARD_HEAD * lat
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    own = o3d.t.geometry.RaycastingScene()
    own.add_triangles(o3d.core.Tensor(np.asarray(v, np.float32)), o3d.core.Tensor(np.asarray(t, np.uint32)))
    # land only on scalp skin that is not in front of the auricle's outward-facing skin: the scalp
    # lying over the concha's floor is removed anyway, and landing on it would span the bowl
    sv_, st_ = (np.asarray(x) for x in scalp)
    sv_, st_ = au.clip_mesh(sv_, st_, np.where(in_front_of(own, sv_, lat), 1.0, -1.0))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(sv_, np.float32)), o3d.core.Tensor(np.asarray(st_, np.uint32)))
    hit = sc.cast_rays(o3d.core.Tensor(np.concatenate([p0, d], axis=1).astype(np.float32)))["t_hit"].numpy()
    ok = np.isfinite(hit) & (hit < SKIRT_MAX_MM)
    if drape is not None:  # in front: drape down onto the nearest scalp skin (never a long reach)
        ok &= ~np.asarray(drape(p0), bool)
    p1 = np.where(ok[:, None], p0 + np.nan_to_num(hit)[:, None] * d, 0.0)
    # base points under the scalp already (or whose ray misses it): land just outside them on it
    if (~ok).any():
        r = sc.compute_closest_points(o3d.core.Tensor(p0[~ok].astype(np.float32)))
        cp, cn = r["points"].numpy(), r["primitive_normals"].numpy()
        tt = t0[~ok] - np.einsum("ij,ij->i", t0[~ok], cn)[:, None] * cn
        tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
        q = sc.compute_closest_points(o3d.core.Tensor((cp + LANDING_OUT_MM * tt).astype(np.float32)))
        p1[~ok] = q["points"].numpy()
    for _ in range(6):  # no zigzag along the landing line
        p1 = 0.5 * p1 + 0.25 * (np.roll(p1, 1, axis=0) + np.roll(p1, -1, axis=0))
        p1 = sc.compute_closest_points(o3d.core.Tensor(p1.astype(np.float32)))["points"].numpy().astype(float)
    n1 = sc.compute_closest_points(o3d.core.Tensor(p1.astype(np.float32)))["primitive_normals"].numpy()
    t1 = d - np.einsum("ij,ij->i", d, n1)[:, None] * n1  # arrive along the scalp, heading away from the ear
    bad = np.linalg.norm(t1, axis=1) < 0.2
    t1[bad] = (t0 - np.einsum("ij,ij->i", t0, n1)[:, None] * n1)[bad]
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9)
    span = np.linalg.norm(p1 - p0, axis=1, keepdims=True)
    rows = []
    for k in range(1, rings + 1):  # cubic Hermite from (p0, t0) to (p1, t1)
        s_ = k / rings
        h00, h10, h01, h11 = 2 * s_**3 - 3 * s_**2 + 1, s_**3 - 2 * s_**2 + s_, -2 * s_**3 + 3 * s_**2, s_**3 - s_**2
        rows.append(h00 * p0 + h10 * span * t0 + h01 * p1 + h11 * span * t1)
    n = len(order)
    sv = np.concatenate(rows)
    off = len(v)
    ring = lambda r: np.asarray(order) if r == 0 else off + (r - 1) * n + np.arange(n)  # noqa: E731
    # no skirt where it would hang in the concha's air (in front of outward-facing skin: the
    # opening toward the canal): the edge stays open there, as cut in Blender
    in_air = np.zeros(len(order), bool) if allow is None else ~np.asarray(allow(p0), bool)  # only where allowed
    draped = np.zeros(len(order), bool) if drape is None else np.asarray(drape(p0), bool)
    for row in rows[1:-1]:  # anywhere along the strip (not its ends, which sit on the sheet and the scalp)
        # hanging in the concha's open air: in front of outward-facing skin with nothing of the
        # auricle on its outer side (a drape tucked under the tragus has the tragus outside it)
        r_ = row.astype(np.float32)
        out_ray = own.cast_rays(o3d.core.Tensor(np.concatenate([r_ + 0.05 * lat.astype(np.float32), np.broadcast_to(lat.astype(np.float32), r_.shape)], axis=1)))
        open_above = ~np.isfinite(out_ray["t_hit"].numpy())
        in_air |= in_front_of(own, row, lat, CONCHA_AIR_MM) & open_above
    in_air = ndimage.binary_dilation(in_air, iterations=2) if in_air.any() else in_air
    keep = ~in_air & ~np.roll(in_air, -1)  # quads between two kept points
    # no stretched pieces: neighbouring strips that land far apart (or far from where they start)
    # would make a flat sheet across whatever lies between
    gap0 = np.linalg.norm(np.roll(p0, -1, axis=0) - p0, axis=1)
    gap1 = np.linalg.norm(np.roll(p1, -1, axis=0) - p1, axis=1)
    keep &= (gap1 < 3.0 * gap0 + STRETCH_MM) | (draped & np.roll(draped, -1))
    tris = []
    for r in range(rings):
        a_, a1 = ring(r), ring(r + 1)
        b_, b1 = np.roll(a_, -1), np.roll(a1, -1)
        tris.append(np.stack([a_, b_, b1], axis=1)[keep])
        tris.append(np.stack([a_, b1, a1], axis=1)[keep])
    st = np.concatenate(tris)
    es = {(int(x), int(y)) for x, y in ca._edges(np.asarray(t))}
    if np.mean([(int(x), int(y)) in es for x, y in st[:n, :2]]) > 0.5:
        st = st[:, [0, 2, 1]]
    outline = p1.copy()
    if in_air.any():  # no skirt here: the scalp is cut along the edge's own footprint, under it
        outline[in_air] = sc.compute_closest_points(o3d.core.Tensor(p0[in_air].astype(np.float32)))["points"].numpy()
    return sv, st, outline, int(in_air.sum())


def _behind_or_above(points, frame, eac):
    """Base points where the skirt goes: behind the canal's line, or well above it (not in front of
    the tragus, where the base edge meets the face at an awkward angle)."""
    r = np.asarray(points, float) - eac.end_centre
    return (r @ np.asarray(frame.anterior, float) < SKIRT_FRONT_MM) | (r @ np.asarray(frame.superior, float) > SKIRT_ABOVE_MM)


def _past_tube_end(eac, cv, loop_pts, bins=36):
    """> 0 for canal-lining points lateral of the meatus tube's end (+ overlap), going round the
    canal: the tube's end is oblique, so the cut follows it angle by angle."""
    def angle(p):
        r = np.asarray(p, float) - eac.end_centre
        f_a = np.cross(eac.lateral, [0.0, 0.0, 1.0])
        f_a /= np.linalg.norm(f_a)
        f_s = np.cross(f_a, eac.lateral)
        return np.arctan2(r @ f_s, r @ f_a)

    tl, _ = eac.axial(loop_pts)
    k = ((angle(loop_pts) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    tmax = np.full(bins, np.nan)
    for b in range(bins):
        if (k == b).any():
            tmax[b] = np.median(tl[k == b])
    ok = np.isfinite(tmax)
    idx = np.arange(bins)
    tmax = np.interp(idx, np.concatenate([idx[ok] - bins, idx[ok], idx[ok] + bins]), np.tile(tmax[ok], 3))
    tmax = np.minimum.reduce([tmax, np.roll(tmax, 1), np.roll(tmax, -1)])  # never past the tube's end
    tc, _ = eac.axial(cv)
    kc = ((angle(cv) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    return tc - (tmax[kc] - BRIDGE_GAP_MM)


def _zipper(pa, ang_a, pb, ang_b):
    """Triangles joining two rings that go round the same centre, walked together by angle.
    ``pa``/``pb``: vertex indices, ``ang_*``: their angles. Returns (n, 3) indices."""
    oa, ob = np.argsort(ang_a), np.argsort(ang_b)
    pa, aa = np.asarray(pa)[oa], np.asarray(ang_a)[oa]
    pb, ab = np.asarray(pb)[ob], np.asarray(ang_b)[ob]
    na, nb = len(pa), len(pb)
    j0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (ab - aa[0]))))))
    ua = np.r_[aa, aa[0] + 2 * np.pi]
    ub = np.r_[ab[np.arange(j0, j0 + nb) % nb], ab[j0] + 2 * np.pi]
    ub = np.unwrap(ub)
    ub += 2 * np.pi * np.round((ua[0] - ub[0]) / (2 * np.pi))
    i = j = 0
    tris = []
    while i < na or j < nb:
        ai, ai1 = pa[i % na], pa[(i + 1) % na]
        bj, bj1 = pb[(j0 + j) % nb], pb[(j0 + j + 1) % nb]
        if j >= nb or (i < na and ua[i + 1] <= ub[j + 1]):
            tris.append((ai, ai1, bj))
            i += 1
        else:
            tris.append((ai, bj1, bj))
            j += 1
    return np.asarray(tris, np.int64)


def _bridge_to_tube(cv, ct, sv, tube_loop, eac, vol, rings=BRIDGE_RINGS, samples=96):
    """Stitch the lining's lateral edge to the sheet's meatus-tube edge (the tube edge's vertices
    are copied into the lining mesh, so the two meet exactly). In between, ``rings`` rings go round
    the canal; any point of them in or against bone moves toward the canal's centre until it is
    just off it, so the join follows the bone's contour instead of cutting behind it."""
    cv, ct = np.asarray(cv, float), np.asarray(ct, np.int64)
    loops = ca.boundary_loops(ct)
    if not loops:
        return cv, ct, 0
    tl, _ = eac.axial(cv)
    lin = np.asarray(max(loops, key=lambda lp: float(np.mean(tl[lp]))))  # the lateral edge (not the open far end)
    tp = np.asarray(sv, float)[tube_loop]
    c = 0.5 * (tp.mean(axis=0) + cv[lin].mean(axis=0))
    _u, _s, vt = np.linalg.svd(np.concatenate([tp, cv[lin]]) - c)
    ax = vt[2] if vt[2] @ eac.lateral > 0 else -vt[2]
    e1 = np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    ang = lambda p: np.arctan2((np.asarray(p) - c) @ e2, (np.asarray(p) - c) @ e1)  # noqa: E731

    def resample(pts):  # the ring's position at evenly spaced angles
        a_ = ang(pts)
        o = np.argsort(a_)
        a_, q = a_[o], pts[o]
        th = np.linspace(-np.pi, np.pi, samples, endpoint=False)
        ext_a = np.r_[a_ - 2 * np.pi, a_, a_ + 2 * np.pi]
        ext_q = np.concatenate([q, q, q])
        return np.stack([np.interp(th, ext_a, ext_q[:, k]) for k in range(3)], axis=1), th

    ra, th = resample(tp)
    rb, _ = resample(cv[lin])
    mids = []
    for k in range(1, rings + 1):
        s_ = k / (rings + 1)
        m = (1 - s_) * ra + s_ * rb
        for _ in range(40):  # off the bone, toward the canal's centre
            idx = np.round(m / vol.mm).astype(int)
            ok = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
            near = np.zeros(len(m), bool)
            near[ok] = vol.solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
            gd = ndimage.map_coordinates(env_dist_cache[0], ((m - env_dist_cache[1]) / env_dist_cache[2]).T, order=1, mode="nearest") if env_dist_cache else None
            if gd is not None:
                near |= gd < BRIDGE_CLEAR_MM
            if not near.any():
                break
            rad = m - c
            rad -= np.outer(rad @ ax, ax)
            m[near] -= 0.1 * rad[near] / np.maximum(np.linalg.norm(rad[near], axis=1, keepdims=True), 1e-9)
        mids.append(m)
    base = len(cv)
    cv = np.concatenate([cv, tp] + mids)
    tube_ids = base + np.arange(len(tp))
    ring_ids = [base + len(tp) + k * samples + np.arange(samples) for k in range(rings)]
    tris = [_zipper(tube_ids, ang(tp), ring_ids[0], th)]
    for k in range(rings - 1):
        a_, b_ = ring_ids[k], ring_ids[k + 1]
        tris.append(np.stack([a_, np.roll(a_, -1), np.roll(b_, -1)], axis=1))
        tris.append(np.stack([a_, np.roll(b_, -1), b_], axis=1))
    tris.append(_zipper(ring_ids[-1], th, lin, ang(cv[lin])))
    tri = np.concatenate(tris)
    fn = np.cross(cv[tri[:, 1]] - cv[tri[:, 0]], cv[tri[:, 2]] - cv[tri[:, 0]])
    toward = c - cv[tri].mean(axis=1)
    toward -= np.outer(toward @ ax, ax)  # radially in, toward the lumen
    flip = np.einsum("ij,ij->i", fn, toward) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return cv, np.concatenate([ct, tri]).astype(np.int32), len(tri)


env_dist_cache = []  # (dist grid, origin, spacing): the envelope's distance to bone, set by join_sheet


def boundary_cycle(t, loop):
    """The open edge through ``loop``'s vertices in order, following the faces' own edge
    directions (a pinched edge, touching itself at a vertex, visits that vertex twice)."""
    t = np.asarray(t, np.int64)
    s_ = set(int(x) for x in loop)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    und = np.sort(e, axis=1)
    _u, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    b = e[cnt[inv.ravel()] == 1]
    b = [(int(x), int(y)) for x, y in b if x in s_]
    nxt = {}
    for x, y in b:
        nxt.setdefault(x, []).append(y)
    start = b[0][0]
    stack, cyc = [start], []
    while stack:  # Hierholzer: every open edge once
        x = stack[-1]
        if nxt.get(x):
            stack.append(nxt[x].pop())
        else:
            cyc.append(stack.pop())
    return cyc[::-1][:-1]


def canal_centreline(eac, step=0.5):
    """Points along the EAC's middle, medial to lateral (centroid of its cross-sections)."""
    pts = np.argwhere(eac.sd < -0.2) * eac.mm + eac.origin
    tt = (pts - eac.end_centre) @ eac.lateral
    edges = np.arange(tt.min(), tt.max() + step, step)
    k = np.digitize(tt, edges)
    cl = np.array([pts[k == i].mean(axis=0) for i in np.unique(k) if (k == i).sum() > 5])
    for _ in range(4):
        cl[1:-1] = 0.25 * cl[:-2] + 0.5 * cl[1:-1] + 0.25 * cl[2:]
    return cl


def _push_off_bone(q, vol, env, centre_fn, clear=BONE_CLEAR_MM, step=0.1, n=60):
    """Points in or against bone moved toward the canal's centre (``centre_fn(points)``: the
    centreline point beside each) until ``clear`` mm off it."""
    q = np.asarray(q, float).copy()
    for _ in range(n):
        idx = np.round(q / vol.mm).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
        bad = np.zeros(len(q), bool)
        bad[ok] = vol.solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        d = ndimage.map_coordinates(env.dist, ((q - env.origin) / env.h).T, order=1, mode="nearest")
        bad |= d < clear
        if not bad.any():
            break
        r = centre_fn(q[bad]) - q[bad]
        q[bad] += step * r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
    return q


def clear_meatus_of_bone(v, t, tube_loop, eac, vol, env, reach_mm=10.0):
    """The sheet near the meatus moved off the bone where it runs inside it (the CT ear is another
    person's: its canal does not follow this bone), toward the canal's centre, the move smoothed
    into the neighbouring skin. Returns (verts, count moved)."""
    from scipy.spatial import cKDTree

    v = np.asarray(v, float)
    cl = canal_centreline(eac)
    tree = cKDTree(cl)
    near = cKDTree(v[tube_loop]).query(v)[0] < reach_mm
    if not near.any():
        return v, 0
    centre_fn = lambda q: cl[tree.query(q)[1]]  # noqa: E731
    moved = v.copy()
    moved[near] = _push_off_bone(v[near], vol, env, centre_fn)
    disp = moved - v
    n_moved = int((np.linalg.norm(disp, axis=1) > 1e-6).sum())
    if n_moved:
        mean, _deg = ca._neighbour_mean(disp, np.asarray(t))  # spread the move a little: no kink at its edge
        for _ in range(3):
            disp = np.where((np.linalg.norm(disp, axis=1, keepdims=True) > 1e-6), disp, 0.5 * mean)
            mean, _deg = ca._neighbour_mean(disp, np.asarray(t))
        disp *= near[:, None]
    return v + disp, n_moved


def off_bone(v, t, vol, env, clear=SKIN_OVER_BONE_MM, skip=None, step=0.1, n=40):
    """Sheet vertices closer to bone than ``clear`` (the CT skin lying on or in the bone, e.g. the
    concha's floor) moved straight away from the bone until ``clear`` off it, the move eased into
    the neighbouring skin. ``skip``: vertices left alone (the meatus has its own). Returns
    (verts, count moved)."""
    v = np.asarray(v, float)
    q = v.copy()

    def dist(p):
        return ndimage.map_coordinates(env.dist, ((p - env.origin) / env.h).T, order=1, mode="nearest")

    moved = np.zeros(len(v), bool)
    for _ in range(n):
        d = dist(q)
        bad = d < clear
        if skip is not None:
            bad &= ~skip
        if not bad.any():
            break
        p = q[bad]
        h = 0.5 * float(np.min(env.h))
        g = np.stack([dist(p + h * e) - dist(p - h * e) for e in np.eye(3)], axis=1)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        q[bad] += step * g
        moved |= bad
    disp = q - v
    if moved.any():
        for _ in range(4):  # ease it into the skin around
            mean, _deg = ca._neighbour_mean(disp, np.asarray(t))
            disp = np.where(moved[:, None], disp, 0.5 * mean)
        # the bone's voxel steps would show in the lifted skin: smoothed over the lifted patch and
        # a ring round it, then pushed out once more where the smoothing brought it back too close
        q = v + disp
        e = ca._edges(np.asarray(t))
        region = moved.copy()
        for _ in range(3):
            grow = region.copy()
            grow[e[region[e[:, 0]], 1]] = True
            grow[e[region[e[:, 1]], 0]] = True
            region = grow
        for _ in range(10):
            mean, _deg = ca._neighbour_mean(q, np.asarray(t))
            q[region] = 0.5 * q[region] + 0.5 * mean[region]
        for _ in range(n):
            bad = (dist(q) < 0.8 * clear) & region
            if not bad.any():
                break
            p = q[bad]
            h = 0.5 * float(np.min(env.h))
            g = np.stack([dist(p + h * e_) - dist(p - h * e_) for e_ in np.eye(3)], axis=1)
            q[bad] += step * g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        disp = q - v
    return v + disp, int(moved.sum())


def grow_canal(sv, st, tube_loop, eac, step=CANAL_STEP_MM, blend_mm=CANAL_BLEND_MM, vol=None, env=None):
    """The canal's skin, from the sheet's meatus-tube end to the eardrum. At each depth along the
    canal's centreline, rays go out at fixed angles (in a frame carried down the centreline
    without twisting) to the EAC's wall, ``LINING_MM`` inside it: one ring. Over the first
    ``blend_mm`` the rings blend from the tube end's own shape to the wall's. One direction per
    vertex of the tube's end, so each column joins straight down from it: the meatus runs on into
    the canal with no gap; the
    last ring is where the canal ends, open at its full width (the tympanic ring). Smoothed, and
    kept off the bone. Returns (verts, tris) facing the lumen."""
    ring0 = np.asarray(boundary_cycle(st, tube_loop))
    p0 = np.asarray(sv, float)[ring0]
    cl = canal_centreline(eac)
    tc = (cl - eac.end_centre) @ eac.lateral
    tang = np.gradient(cl, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    t_start = float(np.median((p0 - eac.end_centre) @ eac.lateral))

    def at(t_, arr):
        return np.array([np.interp(t_, tc, arr[:, k]) for k in range(3)])

    # a frame carried down the centreline without twisting (parallel transport)
    ts = np.arange(t_start, tc.min(), -step)
    T0 = at(t_start, tang)
    T0 /= np.linalg.norm(T0)
    e1 = np.cross(T0, [0.0, 0.0, 1.0] if abs(T0[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    c0 = at(t_start, cl)

    def ring_at(c, e1_, e2_):
        """The wall round centre c: the first point out along each angle at sd = -LINING_MM."""
        dirs = np.cos(th)[:, None] * e1_ + np.sin(th)[:, None] * e2_
        rs = np.arange(0.0, 15.0, 0.1)
        q = c + rs[None, :, None] * dirs[:, None, :]
        sd = eac(q.reshape(-1, 3)).reshape(len(th), len(rs))
        out = np.zeros((len(th), 3))
        ok = np.ones(len(th), bool)
        for j in range(len(th)):
            k = np.nonzero(sd[j] > -LINING_MM)[0]
            if len(k) == 0 or k[0] == 0:
                ok[j] = False
                continue
            i = k[0]
            f = (-LINING_MM - sd[j, i - 1]) / max(sd[j, i] - sd[j, i - 1], 1e-9)
            out[j] = c + (rs[i - 1] + f * (rs[i] - rs[i - 1])) * dirs[j]
        if vol is not None:  # the canal's wall is bone behind it; the eardrum's face is not
            beyond = out[:, None, :] + np.array([0.5, 1.0, 1.5])[None, :, None] * dirs[:, None, :]
            idx = np.round(beyond.reshape(-1, 3) / vol.mm).astype(int)
            inb = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
            solid = np.zeros(len(idx), bool)
            solid[inb] = vol.solid[idx[inb, 0], idx[inb, 1], idx[inb, 2]]
            ok &= solid.reshape(len(th), 3).any(axis=1)
        return out, ok

    # one direction per vertex of the tube's end, in its own order round the edge (each column
    # of the canal is joined straight down from that vertex: nothing to stitch, no holes)
    e2 = np.cross(T0, e1)
    rel = p0 - c0
    th = np.arctan2(rel @ e2, rel @ e1)
    samples = len(th)
    tube_local = rel @ np.stack([e1, e2, T0]).T  # the tube end, in the first frame's coordinates
    e1_first, e2_0 = e1.copy(), e2.copy()
    rings = []
    pins = []
    pinned = np.zeros(len(th), bool)
    for k, t_ in enumerate(ts[1:], start=1):
        T_ = at(t_, tang)
        T_ /= np.linalg.norm(T_)
        e1 = e1 - (e1 @ T_) * T_  # transport the frame
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(T_, e1)
        c = at(t_, cl)
        wall, ok = ring_at(c, e1, e2)
        if k * step > blend_mm:  # each direction stops at its own annulus point (the eardrum is oblique)
            pinned |= ~ok  # no bone behind: the eardrum's face
            if pinned.mean() > 1.0 - ANNULUS_SHARE:
                break
        elif not ok.all():  # near the meatus: an odd ray gets its neighbour's radius
            idx = np.nonzero(ok)[0]
            if len(idx) == 0:
                break
            for j in np.nonzero(~ok)[0]:
                nb = idx[np.argmin(np.abs(np.angle(np.exp(1j * (th[idx] - th[j])))))]
                wall[j] = c + (wall[nb] - c)
        w = min(1.0, k * step / blend_mm)
        w = w * w * (3 - 2 * w)
        tube_here = c + tube_local @ np.stack([e1, e2, T_])  # the tube end's shape, carried into this frame
        ring = (1 - w) * tube_here + w * wall
        if rings:
            ring[pinned] = rings[-1][pinned]  # stopped at its annulus point
        rings.append(ring)
        pins.append(pinned.copy())
    if not rings:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    R = np.stack(rings)
    P = np.stack(pins)  # directions past their annulus point: they stay where they stopped
    for _ in range(CANAL_SMOOTH):
        sm = R.copy()
        sm[1:-1] = 0.5 * R[1:-1] + 0.125 * (R[:-2] + R[2:] + np.roll(R[1:-1], 1, axis=1) + np.roll(R[1:-1], -1, axis=1))
        sm[0] = 0.5 * R[0] + 0.25 * (np.roll(R[0], 1, axis=0) + np.roll(R[0], -1, axis=0))
        sm[-1] = 0.5 * R[-1] + 0.25 * (np.roll(R[-1], 1, axis=0) + np.roll(R[-1], -1, axis=0))
        for r in range(1, len(R)):
            sm[r][P[r]] = sm[r - 1][P[r]]
        R = sm
    if vol is not None:
        def centre_of(x):
            tt = (x - eac.end_centre) @ eac.lateral
            return np.stack([np.interp(tt, tc, cl[:, k]) for k in range(3)], axis=-1)

        for r in range(len(R)):
            R[r] = _push_off_bone(R[r], vol, env, centre_of)
    n0 = len(ring0)
    allv = np.concatenate([p0, R.reshape(-1, 3)])
    ids = [np.arange(n0)] + [n0 + r * samples + np.arange(samples) for r in range(len(R))]
    tris = []
    for r in range(len(ids) - 1):
        a_, b_ = ids[r], ids[r + 1]
        tris.append(np.stack([a_, np.roll(a_, -1), np.roll(b_, -1)], axis=1))
        tris.append(np.stack([a_, np.roll(b_, -1), b_], axis=1))
    tri = np.concatenate(tris)
    cen = allv[tri].mean(axis=1)  # face the lumen: toward the centreline
    tt = (cen - eac.end_centre) @ eac.lateral
    toward = np.stack([np.interp(tt, tc, cl[:, k]) for k in range(3)], axis=-1) - cen
    fn = np.cross(allv[tri[:, 1]] - allv[tri[:, 0]], allv[tri[:, 2]] - allv[tri[:, 0]])
    flip = np.einsum("ij,ij->i", fn, toward) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return allv, tri.astype(np.int32)


def smooth_near(v, t, seeds, reach_mm, iterations, vol=None, env=None, centre_fn=None):
    """Laplacian smoothing of the sheet within ``reach_mm`` of ``seeds`` (fading out), then kept
    off the bone. The meatus's tube is rough (another person's CT, pushed off this bone)."""
    from scipy.spatial import cKDTree

    v = np.asarray(v, float).copy()
    d = cKDTree(v[seeds]).query(v)[0]
    w = np.clip(1.0 - d / reach_mm, 0.0, 1.0)[:, None] * 0.5
    for _ in range(iterations):
        mean, _deg = ca._neighbour_mean(v, np.asarray(t))
        v = v + w * (mean - v)
    if vol is not None:
        near = d < reach_mm
        v[near] = _push_off_bone(v[near], vol, env, centre_fn)
    return v


def base_outline(sheet_v, sheet_t, eac, env=None, thickness=None, muscle=None):
    """The auricle's base: the sheet's largest open edge that is not the meatus (volume mm, in
    order round it), for ``envelope.AuricleRamp.base_outline``."""
    v, t = np.asarray(sheet_v, float), np.asarray(sheet_t)
    loops = sorted(ca.boundary_loops(t), key=len, reverse=True)
    for lp in loops:
        if np.median(eac(v[lp])) > 1.0:
            order = ca._ordered_loop(t, lp)
            return v[order if order is not None else lp]
    return None


def meatus_leaks(out, eac, reach_mm=8.0):
    """The pour test for a sheet ear: water poured into the auricle may leave only down the canal.
    The ear's skin and the canal lining (with its strip to the meatus tube) are merged where they
    share vertices; any open edge left near the meatus (within ``reach_mm`` of where the tube meets
    the lining) is a leak. Returns a list of (vertex count, extent mm) per leaking loop."""
    ev_, et = (np.asarray(x) for x in out["ear"])
    cv, ct = (np.asarray(x) for x in out["canal"])
    pts = np.concatenate([ev_, cv])
    tri = np.concatenate([et, ct + len(ev_)])
    key = np.round(pts / 1e-4).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    tri = inv.ravel()[tri]
    pts_u = np.zeros((inv.max() + 1, 3))
    pts_u[inv.ravel()] = pts
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    t_all, _ = eac.axial(cv)
    centre = cv[np.abs(t_all - np.percentile(t_all, 90)) < 1.0].mean(axis=0) if len(cv) else eac.end_centre
    leaks = []
    for lp in ca.boundary_loops(tri):
        p = pts_u[lp]
        d = np.linalg.norm(p - centre, axis=1)
        tl, _ = eac.axial(p)
        far_end = np.median(tl) < np.percentile(t_all, 30)  # the open end at the eardrum
        if not far_end and d.min() < reach_mm:
            leaks.append((len(lp), float(np.ptp(p, axis=0).max())))
    return leaks


def _ordered(t, loop):
    o = ca._ordered_loop(t, loop)
    return o if o is not None else boundary_cycle(t, loop)


def medial_skin(fv, ft, sheet_v, sheet_t, scalp, lateral, cover=au.COVER_MM, grid=MEDIAL_GRID_MM, drape=DRAPE_MM, eac=None, log=print):
    """Skin over the cartilage where the sheet does not cover it (its medial face past the sheet's
    base edge): the surface ``cover`` out from the cartilage, draped over its knobs and notches
    (closed with a ball of radius ``drape``), kept where the sheet is not already over it and
    where it is outside the envelope's skin (it ends where it meets the envelope: the crease), and
    (with ``eac``) not within ``CANAL_CLEAR_MM`` of the ear canal (the meatus is the sheet's).
    Returns (verts, tris), normals outward."""
    import open3d as o3d

    import surface

    fv, ft = np.asarray(fv, float), np.asarray(ft)
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    if not len(ft):
        return empty
    pad = cover + drape + 3 * grid
    lo = fv.min(axis=0) - pad
    shape = np.ceil((fv.max(axis=0) + pad - lo) / grid).astype(int) + 1
    cs = o3d.t.geometry.RaycastingScene()
    cs.add_triangles(o3d.core.Tensor(fv.astype(np.float32)), o3d.core.Tensor(np.asarray(ft, np.uint32)))
    axes = [lo[k] + grid * np.arange(shape[k]) for k in range(3)]
    field = np.empty(int(np.prod(shape)), np.float32)
    step = shape[1] * shape[2] * max(1, 4_000_000 // int(shape[1] * shape[2]))
    xs = np.arange(len(field))
    for i in range(0, len(field), step):
        k = xs[i : i + step]
        ix, rem = np.divmod(k, shape[1] * shape[2])
        iy, iz = np.divmod(rem, shape[2])
        p = np.stack([axes[0][ix], axes[1][iy], axes[2][iz]], axis=1).astype(np.float32)
        field[i : i + step] = cs.compute_distance(o3d.core.Tensor(p)).numpy() - cover
    field = field.reshape(shape)
    if drape > 0:  # closing: grown by ``drape``, then shrunk back by it (the skin bridges the cartilage's knobs)
        grown = field < drape
        out_d = ndimage.distance_transform_edt(grown) * grid  # inside the grown solid: distance out of it
        field = np.where(grown, drape - out_d, field - drape + 0.0)
        field = ndimage.gaussian_filter(field, 0.6).astype(np.float32)
    solid = ndimage.binary_fill_holes(field < 0)  # no pockets inside the cartilage
    field[solid & (field > 0)] = -grid
    mv, mt = ev.marching_cubes_field(field, np.full(3, grid), lo)
    if not len(mt):
        return empty
    mv = surface.smooth_surface(mv, mt, iterations=3, lam=0.5, max_step=0.5 * grid)
    # keep: the sheet is not over it (it is not within the cover of the sheet), and it is outside the envelope's skin
    ss_ = o3d.t.geometry.RaycastingScene()
    ss_.add_triangles(o3d.core.Tensor(np.asarray(sheet_v, np.float32)), o3d.core.Tensor(np.asarray(sheet_t, np.uint32)))
    d_sheet = ss_.compute_distance(o3d.core.Tensor(mv.astype(np.float32))).numpy()
    es = o3d.t.geometry.RaycastingScene()
    es.add_triangles(o3d.core.Tensor(np.asarray(scalp[0], np.float32)), o3d.core.Tensor(np.asarray(scalp[1], np.uint32)))
    r = es.compute_closest_points(o3d.core.Tensor(mv.astype(np.float32)))
    n = r["primitive_normals"].numpy().astype(float)
    bad = ~np.isfinite(n).all(axis=1) | (np.linalg.norm(np.nan_to_num(n), axis=1) < 0.5)  # zero-area scalp triangles
    n[bad] = np.asarray(lateral, float)
    n *= np.where(n @ np.asarray(lateral, float) < 0, -1.0, 1.0)[:, None]  # the envelope's outer side: lateral
    above = np.einsum("ij,ij->i", mv - r["points"].numpy(), n)  # > 0: outside the envelope's skin
    val = np.maximum(cover + MEDIAL_COVERED_MM - d_sheet, -above)
    if eac is not None:
        val = np.maximum(val, CANAL_CLEAR_MM - eac(mv))
    mv, mt = au.clip_mesh(mv, mt, val)
    if not len(mt):
        return empty
    mv, mt = ev._largest_component(mv, mt, share=0.01)[:2]
    area = 0.5 * np.linalg.norm(np.cross(mv[mt[:, 1]] - mv[mt[:, 0]], mv[mt[:, 2]] - mv[mt[:, 0]]), axis=1).sum()
    log(f"medial skin: {len(mt)} tris, {area:.0f} mm2 over the cartilage's medial face")
    return mv, np.asarray(mt, np.int32)


def smooth_edge(v, t, loop, passes=EDGE_SMOOTH):
    """The open edge's vertices eased toward their neighbours along it (a lasso cut in Blender
    leaves a sawtooth), with the row inside it following halfway. Returns new verts."""
    v = np.asarray(v, float).copy()
    order = np.asarray(_ordered(t, loop))
    if len(order) < 8:
        return v
    inner = np.zeros(len(v), bool)
    e = ca._edges(np.asarray(t))
    on = np.zeros(len(v), bool)
    on[order] = True
    inner[e[on[e[:, 0]], 1]] = True
    inner[e[on[e[:, 1]], 0]] = True
    inner &= ~on
    start = v[order].copy()
    for _ in range(passes):
        p = v[order]
        v[order] = 0.5 * p + 0.25 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
    shift = v[order] - start
    # the row inside: half the shift of its nearest edge vertex (no thin slivers at the edge)
    from scipy.spatial import cKDTree

    idx = np.nonzero(inner)[0]
    if len(idx):
        _d, k = cKDTree(start).query(v[idx])
        v[idx] += 0.5 * shift[k]
    return v


def front_edge(sheet_v, sheet_t, frame, eac, full=False):
    """The auricle's front edge (toward the face: the helix root and the tragus), for
    ``envelope.Envelope(face=...)``: (points, outward directions), volume mm, or None."""
    v, t = np.asarray(sheet_v, float), np.asarray(sheet_t)
    loops = sorted(ca.boundary_loops(t), key=len, reverse=True)
    base = next((lp for lp in loops if np.median(eac(v[lp])) > 1.0), None)
    if base is None:
        return None
    L, A, S = (np.asarray(x, float) for x in (frame.lateral, frame.anterior, frame.superior))
    order, out = _boundary_dirs(v, t, base)
    p = v[np.asarray(order)]
    oh = out - (out @ L)[:, None] * L
    oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    c = v.mean(axis=0)
    front = (oh @ A > 0.5) & ((p - c) @ A > 8.0) & ((p - c) @ S > FRONT_BELOW_MM)
    k = 4
    f_ = np.r_[front[-k:], front, front[:k]].astype(float)
    front = np.convolve(f_, np.ones(2 * k + 1) / (2 * k + 1), "valid") > 0.5
    if not front.any():
        return None
    return (p[front], oh[front], out[front]) if full else (p[front], oh[front])


def front_slope(v, t, loop, scalp, frame, reach_mm=FRONT_SLOPE_MM, rows=FRONT_ROWS):
    """Skin sloping gently off the auricle's front edge onto the side of the face (over the TMJ),
    instead of closing round the cartilage there. From each point of the base edge that faces
    forward, a strip carries on the way the skin was going and eases down, over ``reach_mm``, onto
    the level of the face: a smooth fit to the envelope's skin around the front of the ear (the
    envelope does not reach over the TMJ in the volume). The strips' first row is the edge itself.

    Returns (front, verts, tris): ``front`` per sheet vertex (the front edge), the new vertices,
    and triangles indexing the sheet's vertices (< len(v)) and then the new ones (len(v) + i)."""
    from scipy.spatial import cKDTree

    v = np.asarray(v, float)
    n = len(v)
    L, A, S = (np.asarray(x, float) for x in (frame.lateral, frame.anterior, frame.superior))
    empty = (np.zeros(n, bool), np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    order, out = _boundary_dirs(v, t, loop)
    order = np.asarray(order)
    p = v[order]
    oh = out - (out @ L)[:, None] * L
    oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    c = v.mean(axis=0)
    front = (oh @ A > 0.5) & ((p - c) @ A > 8.0) & ((p - c) @ S > FRONT_BELOW_MM)  # the upper front (helix root, tragus), not the lobule
    k = 4  # clean the mask round the loop (no single points in or out)
    f_ = np.r_[front[-k:], front, front[:k]].astype(float)
    front = np.convolve(f_, np.ones(2 * k + 1) / (2 * k + 1), "valid") > 0.5
    if not front.any() or front.all():
        return empty
    shift = int(np.argmin(front))  # start the loop outside the front
    order, out, oh, p, front = (np.roll(x, -shift, axis=0) for x in (order, out, oh, p, front))
    runs, i = [], 0
    while i < len(front):
        if front[i]:
            j = i
            while j < len(front) and front[j]:
                j += 1
            if j - i >= 4:
                runs.append(np.arange(i, j))
            i = j
        else:
            i += 1
    if not runs:
        return empty
    # the face's level: a plane fitted to the envelope's skin around the front of the ear
    sv = np.asarray(scalp[0], float)
    uv = lambda q: np.stack([q @ A, q @ S], axis=1)  # noqa: E731
    fp = p[np.concatenate(runs)]
    d, _i = cKDTree(uv(fp)).query(uv(sv))
    sel = (d > 3.0) & (d < 25.0) & ((sv - c) @ A > 0.0)
    if sel.sum() < 20:
        sel = d < 40.0
    X = np.column_stack([np.ones(sel.sum()), sv[sel] @ A, sv[sel] @ S])
    coef = np.linalg.lstsq(X, sv[sel] @ L, rcond=None)[0]

    def level(q):
        return coef[0] + coef[1] * (q @ A) + coef[2] * (q @ S)

    verts, tris, fmask = [], [], np.zeros(n, bool)
    for run in runs:
        idx = order[run]
        fmask[idx] = True
        P, O, OH = p[run], out[run], oh[run].copy()
        for _ in range(4):  # strips side by side, not crossing
            OH[1:-1] = 0.5 * OH[1:-1] + 0.25 * (OH[:-2] + OH[2:])
            OH /= np.maximum(np.linalg.norm(OH, axis=1, keepdims=True), 1e-9)
        # how far forward it reaches: gentle (``FRONT_RUN`` mm forward per mm down), smoothed along the edge
        drop = np.maximum(P @ L - level(P), 0.0)
        reach = np.clip(FRONT_RUN * drop, reach_mm, FRONT_MAX_MM)
        for _ in range(10):
            reach[1:-1] = 0.5 * reach[1:-1] + 0.25 * (reach[:-2] + reach[2:])
        # each strip: forward along the face, its height over the face falling off as (1 - s)^2:
        # quicker off the auricle, easing out flat onto the face (a gentle hollow, not a flat plate)
        H = P @ L - level(P)  # the edge's height over the face
        m = len(run)
        first = n + sum(len(x) for x in verts)
        rows_v = []
        for r in range(1, rows + 1):
            s_ = r / rows
            R = P - (P @ L)[:, None] * L + (s_ * reach)[:, None] * OH  # in the face's plane
            R += (level(R) + H * (1.0 - s_) ** FRONT_HOLLOW)[:, None] * L
            for _ in range(2 + 2 * r):  # smooth along the edge (it zig-zags), more further out
                R[1:-1] = 0.5 * R[1:-1] + 0.25 * (R[:-2] + R[2:])
            rows_v.append(R)
        verts.append(np.concatenate(rows_v))

        def vid(r, j, idx=idx, first=first, m=m):
            return int(idx[j]) if r == 0 else first + (r - 1) * m + j

        for r in range(rows):
            for j in range(m - 1):
                a, b, c_, d_ = vid(r, j), vid(r, j + 1), vid(r + 1, j + 1), vid(r + 1, j)
                tris += [(a, b, c_), (a, c_, d_)]
    nv = np.concatenate(verts)
    tris = np.asarray(tris, np.int64)
    allv = np.concatenate([v, nv])
    nrm = np.cross(allv[tris[:, 1]] - allv[tris[:, 0]], allv[tris[:, 2]] - allv[tris[:, 0]])
    if (nrm @ L).sum() < 0:  # facing out (lateral), like the skin
        tris = tris[:, [0, 2, 1]]
    return fmask, nv, tris


def stretch(mv, mt):
    """The skin stretched smooth across the opening, like a membrane: its open edges (on the
    sheet's base edge, at the crease) held, everything inside harmonic (no lumps). Returns verts."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    mv = np.asarray(mv, float)
    mt = np.asarray(mt)
    loops = ca.boundary_loops(mt)
    if not loops:
        return mv
    fixed = np.zeros(len(mv), bool)
    fixed[np.concatenate(loops)] = True
    e = ca._edges(mt)
    e = np.unique(np.sort(np.concatenate([e, e[:, ::-1]]), axis=1), axis=0)
    e = np.concatenate([e, e[:, ::-1]])
    n = len(mv)
    a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    lap = diags(np.asarray(a.sum(axis=1)).ravel()) - a
    free = np.nonzero(~fixed)[0]
    fx = np.nonzero(fixed)[0]
    if not len(free):
        return mv
    lff = lap[free][:, free].tocsc()
    rhs = -(lap[free][:, fx] @ mv[fx])
    out = mv.copy()
    out[free] = np.column_stack([spsolve(lff, rhs[:, k]) for k in range(3)])
    return out


def _seam_to_base(mv, mt, base_pts, reach_mm=SEAM_MM):
    """The medial skin's open edge, where it comes within ``reach_mm`` of the sheet's base edge,
    moved onto that edge (so the two skins meet with no gap)."""
    from scipy.spatial import cKDTree

    if not len(mt) or not len(base_pts):
        return mv
    dense = []
    for b in base_pts:  # the edge as closely spaced points
        seg = np.roll(b, -1, axis=0) - b
        k = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / 0.05).astype(int))
        for p, d_, n in zip(b, seg, k):
            dense.append(p + np.outer(np.arange(n) / n, d_))
    dense = np.concatenate(dense)
    tree = cKDTree(dense)
    mv = np.asarray(mv, float).copy()
    edge = np.unique(np.concatenate(ca.boundary_loops(mt))) if ca.boundary_loops(mt) else np.zeros(0, int)
    d, i = tree.query(mv[edge])
    near = d < reach_mm
    mv[edge[near]] = dense[i[near]]
    # ease the rows behind the moved edge so it does not fold
    moved = np.zeros(len(mv), bool)
    moved[edge[near]] = True
    for _ in range(4):
        mean, _deg = ca._neighbour_mean(mv, mt)
        free = ~moved
        near_moved = np.zeros(len(mv), bool)
        e = ca._edges(mt)
        near_moved[e[moved[e[:, 0]], 1]] = True
        near_moved[e[moved[e[:, 1]], 0]] = True
        sel = free & near_moved
        mv[sel] = 0.5 * mv[sel] + 0.5 * mean[sel]
    return mv


def _no_pinholes(gone, tris, min_count=PINHOLE_COUNT):
    """Removed patches of fewer than ``min_count`` vertices are kept instead (no pinholes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    gone = np.asarray(gone, bool).copy()
    e = ca._edges(np.asarray(tris))
    e = e[gone[e[:, 0]] & gone[e[:, 1]]]
    n = len(gone)
    _k, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    size = np.bincount(lab[gone], minlength=lab.max() + 1)
    gone[gone & (size[lab] < min_count)] = False
    return gone


def above_tissue(v, meshes, lateral, reach_mm=PIT_MM, clear_mm=0.2):
    """Sheet vertices with envelope tissue just lateral of them (a pit in the CT skin dipping
    under the layers it lies on, so they show through) moved out to ``clear_mm`` above it.
    Returns (verts, count moved)."""
    import open3d as o3d

    v = np.asarray(v, float).copy()
    sc = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        if m is not None and len(m[1]):
            sc.add_triangles(o3d.core.Tensor(np.asarray(m[0], np.float32)), o3d.core.Tensor(np.asarray(m[1], np.uint32)))
    L = np.asarray(lateral, float)
    hit = sc.cast_rays(o3d.core.Tensor(np.concatenate([v, np.broadcast_to(L, v.shape)], axis=1).astype(np.float32)))["t_hit"].numpy()
    under = hit < reach_mm
    v[under] += (hit[under] + clear_mm)[:, None] * L
    return v, int(under.sum())


class EnvelopeMeet:
    """How the envelope meets the auricle (see ``join_sheet``). ``bases``: the sheet's base edges
    (ordered points), ``skin``: the envelope's skin (verts, tris), ``sheet``: (verts, tris).
    ``free``: per base point (all bases in turn), 1 where the envelope may rise to it, 0 where the
    medial skin carries the auricle on down to the envelope (the base there is the crease)."""

    def __init__(self, frame, bases, skin, sheet, res=0.2, free=None, front=None):
        import open3d as o3d
        from PIL import Image, ImageDraw
        from scipy.spatial import cKDTree

        self.L, self.A, self.S = (np.asarray(x, float) for x in (frame.lateral, frame.anterior, frame.superior))
        pts = np.concatenate(bases) if bases else np.zeros((0, 3))
        self.base_h = pts @ self.L
        self.free = np.ones(len(pts)) if free is None else np.asarray(free, float)
        # the auricle's front edge: (points, forward dirs, 3D outward dirs) for the fillet onto the face
        self.front = None
        if front is not None and len(front[0]):
            from scipy.spatial import cKDTree as _T

            fp, foh, fo = (np.asarray(x, float) for x in front)
            steep = np.clip(-(fo @ self.L) / np.maximum(np.linalg.norm(fo - (fo @ self.L)[:, None] * self.L, axis=1), 0.2), 0.0, FILLET_MAX_SLOPE)
            for _ in range(6):  # smooth along the edge
                steep[1:-1] = 0.5 * steep[1:-1] + 0.25 * (steep[:-2] + steep[2:])
            self.front = (fp, foh, steep, _T(self._uv(fp)))
        self.over_scene = None
        uv = self._uv(pts)
        self.tree = cKDTree(uv) if len(uv) else None
        allp = np.concatenate([np.asarray(skin[0]), np.asarray(sheet[0])])
        auv = self._uv(allp)
        self.lo = auv.min(axis=0) - 2.0
        n = np.ceil((auv.max(axis=0) + 2.0 - self.lo) / res).astype(int) + 1
        img = Image.new("L", (int(n[0]), int(n[1])), 0)
        for b in bases:
            ImageDraw.Draw(img).polygon([tuple(q) for q in (self._uv(b) - self.lo) / res], fill=1)
        inside = np.asarray(img, bool).T
        self.sd = ((ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * res).astype(np.float32)
        self.res = res
        self.skin_scene = o3d.t.geometry.RaycastingScene()
        self.skin_scene.add_triangles(o3d.core.Tensor(np.asarray(skin[0], np.float32)), o3d.core.Tensor(np.asarray(skin[1], np.uint32)))
        self.sheet_mesh = sheet
        self.sheet_scene = o3d.t.geometry.RaycastingScene()
        self.sheet_scene.add_triangles(o3d.core.Tensor(np.asarray(sheet[0], np.float32)), o3d.core.Tensor(np.asarray(sheet[1], np.uint32)))

    def set_over(self, medial, cartilage, slope=None):
        """What lies over the envelope under the auricle: the sheet, the medial skin, the cartilage
        (for ``gone``: a skin facing down across a gap is a cleft; cartilage over it is not)."""
        import open3d as o3d

        sc = o3d.t.geometry.RaycastingScene()
        self.over_ids = {}
        for name, m in (("sheet", self.sheet_mesh), ("medial", medial), ("cartilage", cartilage), ("slope", slope)):
            if m is not None and len(m[1]):
                self.over_ids[name] = sc.add_triangles(o3d.core.Tensor(np.asarray(m[0], np.float32)), o3d.core.Tensor(np.asarray(m[1], np.uint32)))
        self.over_scene = sc
        sk = o3d.t.geometry.RaycastingScene()  # the auricle's skins only
        for m in (self.sheet_mesh, medial):
            if m is not None and len(m[1]):
                sk.add_triangles(o3d.core.Tensor(np.asarray(m[0], np.float32)), o3d.core.Tensor(np.asarray(m[1], np.uint32)))
        self.skins_scene = sk
        if slope is not None and len(slope[1]):  # the skin sloping onto the face counts as the auricle's outer skin too
            ds = o3d.t.geometry.RaycastingScene()
            for m in (self.sheet_mesh, slope):
                ds.add_triangles(o3d.core.Tensor(np.asarray(m[0], np.float32)), o3d.core.Tensor(np.asarray(m[1], np.uint32)))
            self.sheet_scene = ds

    def fillet(self, p):
        """(raise mm, in band): just in front of the auricle's front edge the face rises in a short
        concave curve that leaves the edge at its height and with the slope of the auricle's skin
        coming down there, and lands tangent on the face (within ``FILLET_MM``): no crease."""
        p = np.asarray(p, float)
        self._behind = np.zeros(len(p), bool)
        if self.front is None:
            return np.zeros(len(p)), np.zeros(len(p), bool)
        fp, foh, steep, tree = self.front
        r = FILLET_MM
        d, k = tree.query(self._uv(p), distance_upper_bound=r + 2.0)
        hit = np.isfinite(d)
        out, band = np.zeros(len(p)), np.zeros(len(p), bool)
        if not hit.any():
            return out, band
        kk = k[hit]
        x = np.einsum("ij,ij->i", p[hit] - fp[kk], foh[kk])
        side = np.linalg.norm((p[hit] - fp[kk]) - x[:, None] * foh[kk] - ((p[hit] - fp[kk]) @ self.L)[:, None] * self.L, axis=1)
        inb = (x >= 0.0) & (x < r) & (side < 2.0)
        self._behind = np.zeros(len(p), bool)
        self._behind[hit] = (x < 0.0) & (x > -FILLET_BEHIND_MM) & (side < 2.0)  # just behind the edge: under the auricle
        # the curve leaves the edge at its height and slope m and lands on the face (height hp) at
        # x = r' = 2 dh / m, tangent to both: h(x) = he - m x + m x^2 / (2 r')
        m = np.maximum(steep[kk], 0.05)
        he = fp[kk] @ self.L
        hp = p[hit] @ self.L
        dh = np.maximum(he - hp, 0.2)
        rr = np.clip(2 * dh / m, 0.8, r)
        xc = np.clip(x, 0.0, None)
        prof = he - m * xc + m * xc**2 / (2 * rr)
        fade = np.clip(1.0 - xc / rr, 0.0, 1.0)
        out[hit] = np.where(inb & (xc < rr), np.maximum(prof - hp, 0.0) * fade * fade * (3 - 2 * fade), 0.0)
        band[hit] = inb
        return out, band

    def _uv(self, p):
        p = np.asarray(p, float)
        return np.stack([p @ self.A, p @ self.S], axis=1)

    def outline_sd(self, p):
        """Signed distance (mm) from the base outline seen from lateral, < 0 inside (under the auricle)."""
        return ndimage.map_coordinates(self.sd, ((self._uv(p) - self.lo) / self.res).T, order=1, mode="constant", cval=1e3)

    def lift(self, p, skin_p):
        """Envelope points raised (along lateral) toward the auricle's base where the base stands
        lateral of the skin: the full rise at the base outline, fading out over ``RISE_MM``.
        ``skin_p``: the skin's own points (their heights), or None to read the skin below."""
        p = np.asarray(p, float)
        if self.tree is None:
            return self._with_fillet(p)
        sd = self.outline_sd(p)
        near = (sd > -1.0) & (sd < RISE_MM)
        if not near.any():
            return self._with_fillet(p)
        import open3d as o3d

        q = p[near]
        dk, ik = self.tree.query(self._uv(q), k=min(RISE_NEIGHBOURS, len(self.base_h)))
        wk = 1.0 / np.maximum(dk, 0.5) ** 2  # the base's height here: blended from nearby base points
        base_here = (self.base_h[ik] * wk).sum(axis=1) / wk.sum(axis=1)
        free_here = (self.free[ik] * wk).sum(axis=1) / wk.sum(axis=1)
        if skin_p is not None:
            h_skin = np.asarray(skin_p, float)[near] @ self.L
        else:  # the skin above this point: a ray going lateral
            hit = self.skin_scene.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(self.L, q.shape)], axis=1).astype(np.float32)))["t_hit"].numpy()
            h_skin = q @ self.L + np.where(np.isfinite(hit), hit, 0.0)
        rise = np.maximum(0.0, base_here - h_skin) * free_here
        s = np.clip(1.0 - np.maximum(sd[near], 0.0) / RISE_MM, 0.0, 1.0)
        w = s * s * (3 - 2 * s)
        out = p.copy()
        out[near] = q + (w * rise)[:, None] * self.L
        return self._with_fillet(out)

    def _with_fillet(self, p):
        up, _band = self.fillet(p)
        return p + up[:, None] * self.L

    def gone(self, p, under_auricle):
        """Envelope points that must go: on the auricle's outer side (the auricle's skin lies medial
        of them: in the concha's bowl, or through the auricle), and, with ``under_auricle``, inside
        the base outline (the envelope's skin: the auricle's own skin drapes there)."""
        import open3d as o3d

        p = np.asarray(p, float)
        out = np.zeros(len(p), bool)
        near = self.outline_sd(p) < 25.0
        if near.any():
            q = p[near].astype(np.float32)
            rd = self.sheet_scene.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(-self.L, q.shape).astype(np.float32)], axis=1)))
            down = rd["t_hit"].numpy()
            sd_ = self.outline_sd(p[near])
            at_join = (sd_ >= 0) & (sd_ < 2.0)  # just outside the base: touching counts as meeting (inside it, nothing lies over the auricle's skin)
            # outside the base outline only the auricle's outer side counts (its skin below faces up, as
            # in the concha's bowl); skin facing down there is an edge tucked in under the face
            outer = (sd_ < 0) | (rd["primitive_normals"].numpy() @ self.L > 0)
            _up, fil = self.fillet(p[near])
            out[near] |= np.isfinite(down) & (down < 30.0) & (~at_join | (down > TOUCH_MM)) & outer & ~fil  # the auricle's skin lies medial of it
            out[near] |= self._behind  # just behind the front edge: under the auricle (no step where the fillet starts)
            if under_auricle and self.over_scene is None:  # under the auricle: inside the base outline, with the auricle's skin over it
                up = self.sheet_scene.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(self.L, q.shape).astype(np.float32)], axis=1)))["t_hit"].numpy()
                out[near] |= (self.outline_sd(p[near]) < 0) & np.isfinite(up)
            elif under_auricle:  # under the auricle, unless what is over it is a skin facing down across a gap (a cleft)
                r = self.over_scene.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(self.L, q.shape)], axis=1).astype(np.float32)))
                up = r["t_hit"].numpy()
                gid = r["geometry_ids"].numpy().astype(np.int64)
                facing_down = r["primitive_normals"].numpy() @ self.L < 0
                cart = gid == self.over_ids.get("cartilage", -1)
                cleft = np.isfinite(up) & ~cart & facing_down  # however narrow: at the crease the two skins meet
                # cartilage over it counts only where the auricle's skin closes over it from outside
                # (the concha's floor); at the crease the skin above faces the head, and the
                # envelope covers the cartilage going into it
                rs = self.skins_scene.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(self.L, q.shape)], axis=1).astype(np.float32)))
                closed = (rs["t_hit"].numpy() < CLOSED_MM) & (rs["primitive_normals"].numpy() @ self.L > 0)
                covered = np.isfinite(up) & (up < 30.0) & ~cleft & (~cart | closed)
                out[near] |= covered & (self.outline_sd(p[near]) < 0)
                out[near] |= np.isfinite(up) & (gid == self.over_ids.get("slope", -1))  # under the skin sloping onto the face
        return out


def base_trimmer(v, t, bases, margin_mm=BASE_TRIM_MM, reach_mm=6.0, carried=None):
    """f(cart_v, cart_t) -> (verts, tris): the cartilage trimmed back from the sheet's base edge
    (it may not reach past the edge, where the envelope meets the auricle and nothing covers it).
    ``carried``: per sheet vertex, True where the medial skin carries the auricle on past the edge
    (the cartilage may go on there). Further than ``reach_mm`` from the edge, cartilage is cut
    where it is past the edge and off the sheet (its nearest place on the sheet is the open edge,
    not the skin over it): out in the open behind the auricle."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    pts, outs = [], []
    for lp in bases:
        order, tout = _boundary_dirs(v, t, lp)
        keep_ = np.ones(len(order), bool) if carried is None else ~np.asarray(carried)[order]
        pts.append(np.asarray(v)[order][keep_])
        outs.append(tout[keep_])
    if not pts or not sum(len(x) for x in pts):
        return lambda fv, ft: (np.asarray(fv), np.asarray(ft))
    bp, bo = np.concatenate(pts), np.concatenate(outs)
    tree = cKDTree(bp)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(v, np.float32)), o3d.core.Tensor(np.asarray(t, np.uint32)))

    def trim(fv, ft):
        fv = np.asarray(fv, float)
        d, i = tree.query(fv)
        past = np.einsum("ij,ij->i", fv - bp[i], bo[i]) + margin_mm  # > 0: past the edge (or too close)
        r = sc.compute_closest_points(o3d.core.Tensor(fv.astype(np.float32)))
        off = fv - r["points"].numpy()
        ds = np.linalg.norm(off, axis=1)
        off_sheet = np.abs(np.einsum("ij,ij->i", off, r["primitive_normals"].numpy())) < 0.5 * ds  # nearest the open edge, not over the skin
        val = np.where(d < reach_mm, past, np.where(off_sheet & (past > 0), 1.0, -1.0))
        if not (val > 0).any():
            return fv, np.asarray(ft)
        cv, ct = au.clip_mesh(fv, ft, val)
        return ev._largest_component(cv, ct, share=0.02)[:2]

    return trim


def _inside_outline(side, outline, points, res=0.2):
    """Signed distance (mm) from the outline seen from lateral: < 0 inside it."""
    from PIL import Image, ImageDraw

    ab, _h = side.coords(outline)
    xy = ab * side.pix / res
    shape = (np.ceil(np.array(side.shape) * side.pix / res)).astype(int) + 1
    img = Image.new("L", (int(shape[0]), int(shape[1])), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in xy], fill=1)
    inside = np.asarray(img, bool).T
    sd = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * res
    abp, _ = side.coords(points)
    return ndimage.map_coordinates(sd.astype(np.float32), (abp * side.pix / res).T, order=1, mode="constant", cval=1e3)


def join_sheet(vol, env, sheet, scalp_skin, thickness, muscle=None, eac=None, others=None,
               framework_max_mm=ca.MAX_MM, cartilage=None, back_skin=True, front_skin=False, log=print):
    """See the module doc. ``sheet``: (verts, tris, auricle weight), volume mm. ``cartilage``:
    (verts, tris), volume mm, the cartilage as edited in Blender (None: built from the sheet).
    ``back_skin``: also show the skin over the cartilage's medial face past the base edge (the
    ``medial`` skin) as part of the ear. Off: the ear is the sheet as edited, the cartilage stays
    under it and short of its base edge, and the medial skin only tells the envelope where the
    auricle's back is (the cleft and the crease).

    Returns dict like ``ct_meatus.open_meatus``: ear (skin), scalp, canal, layers, framework, skin_mm.
    """
    import open3d as o3d

    import ct_meatus as cm

    t0 = time.time()
    eac = eac or cm.EacField(vol)
    v, t, w = ca.clean_sheet(*sheet, log=log)
    hfn = lambda p: cm.height_over_envelope(env, thickness, muscle, p)  # noqa: E731
    loops = sorted(ca.boundary_loops(t), key=len, reverse=True)
    meatus = [lp for lp in loops if np.median(eac(v[lp])) < 1.0 and np.median(hfn(v[lp])) < 2.0]
    bases = [lp for lp in loops if lp not in meatus and len(lp) > 20]
    if meatus:
        v, n_moved = clear_meatus_of_bone(v, t, meatus[0], eac, vol, env)
        if n_moved:
            log(f"sheet: {n_moved} vertices near the meatus moved off the bone")
        from scipy.spatial import cKDTree

        cl_ = canal_centreline(eac)
        tree_ = cKDTree(cl_)
        v = smooth_near(v, t, meatus[0], MEATUS_SMOOTH_MM, MEATUS_SMOOTH, vol, env, lambda q: cl_[tree_.query(q)[1]])
    # elsewhere the CT skin may lie on (or in) this bone, e.g. the concha's floor: lifted off it
    from scipy.spatial import cKDTree as _T

    skip = (_T(v[meatus[0]]).query(v)[0] < 12.0) if meatus else None
    v, n_bone = off_bone(v, t, vol, env, skip=skip)
    if n_bone:
        log(f"sheet: {n_bone} vertices lifted off the bone")
    scalp0 = tuple(np.asarray(x) for x in scalp_skin)
    L = np.asarray(vol.frame.lateral, float)
    if back_skin:  # a lasso cut leaves a sawtooth edge (the medial skin joins it)
        for lp in bases:
            v = smooth_edge(v, t, lp)
    # the cartilage: as edited, or built under the sheet's painted auricle
    if cartilage is not None:
        fw = tuple(np.asarray(x) for x in cartilage)
    elif (w > 0.5).any():
        fw = ca.framework(v, t, w, max_mm=framework_max_mm, log=log)
    else:
        fw = None
    # skin over the cartilage's medial face, where the cartilage goes on past the sheet's base edge
    base_pts = [v[_ordered(t, lp)] for lp in bases]
    medial = medial_skin(*fw, v, t, scalp0, L, eac=eac, log=log) if fw is not None else (np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    # the front edge (toward the face): skin slopes off it onto the face, it does not close round the cartilage
    front = np.zeros(len(v), bool)
    slope_v, slope_t = np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    if bases and front_skin:
        front, slope_v, slope_t = front_slope(v, t, bases[0], scalp0, vol.frame)
        if len(slope_t):
            log(f"front: {int(front.sum())} edge points slope onto the face ({len(slope_t)} tris)")
    carried = np.zeros(len(v), bool)
    if len(medial[1]):
        from scipy.spatial import cKDTree

        if back_skin:  # joined to the sheet's base edge, stretched smooth across the opening
            medial = (stretch(_seam_to_base(*medial, base_pts), medial[1]), medial[1])
        d_, _i = cKDTree(medial[0]).query(v[np.concatenate(bases)] if bases else np.zeros((0, 3)))
        if bases:
            carried[np.concatenate(bases)[d_ < CONTINUED_MM]] = True
        carried &= ~front
        log(f"medial skin carries {int(carried.sum())} of {sum(len(b) for b in bases)} base edge points on to the crease")
    if not back_skin:
        medial_shown = (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    else:
        medial_shown = medial
    nm = len(medial_shown[0])
    st_ = np.where(slope_t >= len(v), slope_t + nm, slope_t)  # the slope's new vertices go after the medial skin's
    ear_v = np.concatenate([v, medial_shown[0], slope_v])
    ear_t = np.concatenate([np.asarray(t, np.int64), np.asarray(medial_shown[1], np.int64) + len(v), st_])
    slope = (np.concatenate([v, slope_v]), slope_t) if len(slope_t) else None
    # the envelope meets the auricle at its base: where the base stands lateral of the envelope,
    # the envelope (skin and the layers under it) rises gently to it, except where the medial skin
    # carries the auricle on down to the envelope (the crease); under the auricle it gives way,
    # except in the cleft; and nothing of it is left on the auricle's outer side (the auricle is a
    # projection of the side of the head: nothing cuts through it)
    free = np.concatenate([~(carried | front)[_ordered(t, lp)] for lp in bases]).astype(float) if bases else None
    fr_ = front_edge(v, t, vol.frame, eac, full=True)
    meet = EnvelopeMeet(vol.frame, base_pts, scalp0, (v, t), free=free, front=fr_)
    if meet.front is not None:
        _up, _band = meet.fillet(scalp0[0])
        log(f"front: fillet onto the face along {len(meet.front[0])} edge points, slope {np.percentile(meet.front[2], [10, 50, 90]).round(2)}, "
            f"{int(_band.sum())} envelope skin points raised up to {_up.max():.1f} mm")
    meet.set_over(medial, fw, slope)
    log(f"sheet: {len(bases)} base edge(s), {len(meatus)} meatus edge(s), {time.time() - t0:.1f} s")
    sv_ = meet.lift(scalp0[0], scalp0[0])
    scalp = au.clip_mesh(sv_, scalp0[1], np.where(_no_pinholes(meet.gone(sv_, under_auricle=True), scalp0[1]), 1.0, -1.0))
    layers = {}
    for name, mesh in (others or {}).items():
        lv, lt = (np.asarray(x) for x in mesh)
        lv_ = meet.lift(lv, None)
        layers[name] = au.clip_mesh(lv_, lt, np.where(_no_pinholes(meet.gone(lv_, under_auricle=False), lt), 1.0, -1.0))
    # pits in the CT skin that dip under the layers they lie on: lifted out (no fascia through the concha)
    v, n_pit = above_tissue(v, [scalp, *layers.values()], L)
    if n_pit:
        log(f"sheet: {n_pit} vertices in pits lifted out of the tissue under them")
        ear_v = np.concatenate([v, ear_v[len(v):]])
        if slope is not None:
            slope = (np.concatenate([v, slope_v]), slope_t)
    keep = lambda pts: np.where(meet.gone(np.asarray(pts, float), under_auricle=True), 1.0, -1.0)  # noqa: E731
    trim_base = base_trimmer(v, t, bases, carried=carried if back_skin else None)
    log(f"scalp: {len(scalp[1])} tris kept, {time.time() - t0:.1f} s")
    # the canal: grown from the meatus tube's own end, down the canal onto the EAC's wall
    if meatus:
        cv, ct = grow_canal(v, t, meatus[0], eac, vol=vol, env=env)
    else:  # no tube: the EAC's wall, open at the eardrum
        cv, ct = eac.mesh(-LINING_MM)
        cv, ct = cm.open_far_end(cv, ct[:, [0, 2, 1]], eac)
    log(f"canal lining: {len(ct)} tris, {time.time() - t0:.1f} s")
    framework = None
    if fw is not None:  # never poking out of the skin, nor past the base edge (unless the back skin carries it on)
        fv_, n_ = ca.keep_under_skin(fw[0], fw[1], v, t, iterations=12)  # under the sheet as edited
        if n_:
            log(f"cartilage: {n_} vertices pulled back under the skin")
        fv_, ft_ = fv_, fw[1]
        if back_skin and len(medial[1]):  # and cut where it comes out through the skin across the opening
            fv_, ft_ = ca.clip_under_skin(fv_, ft_, *medial)
        if slope is not None:  # and under the skin sloping onto the face
            fv_, ft_ = ca.clip_under_skin(fv_, ft_, *slope)
            fv_, n2 = ca.keep_under_skin(fv_, ft_, ear_v, ear_t, iterations=12)  # the few left near where the two skins meet
            if n2:
                log(f"cartilage: {n2} more vertices pulled under where the skins meet")
        fv_, ft_ = trim_base(fv_, ft_)
        out_of_canal = CANAL_CART_MM - eac(fv_)  # the canal is air: no cartilage in it
        if (out_of_canal > 0).any():
            fv_, ft_ = au.clip_mesh(fv_, ft_, out_of_canal)
            fv_, ft_ = ev._largest_component(fv_, ft_, share=0.005)[:2]
        framework = (fv_, ft_)
    return dict(ear=(ear_v, ear_t.astype(np.int32)), scalp=scalp, canal=(cv, ct), layers=layers, framework=framework,
                skin_mm=cm.skin_depth_at_canal(env, thickness), sheet=(v, t, w), bowl=None, medial=medial,
                cartilage_done=cartilage is not None,  # the edited cartilage is already kept under the skin and trimmed
                skin_keep=keep,  # f(points) < 0 where the envelope's skin stays (for the layer solids)
                trim_cartilage=trim_base)  # f(verts, tris): cartilage trimmed back from the base edge
