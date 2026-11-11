"""Open the CT ear's meatus onto the volume's ear canal, and trim what is buried.

The CT ear (``ct_ear.py``) has a natural meatus: a pit in the concha. Placed on a volume, the
volume's EAC segmentation (a solid plug of canal air) runs up into that pit and overlaps the
ear's tissue around it: part of the segmentation is really the cartilaginous canal. This module
builds one soft-tissue solid around the canal's mouth and carves the canal out of it:

1. the ear (the CT mesh, as placed), draped over the bone it meets: the CT comes from another
   head, so its back surface and crease run into this volume's bone and bone pokes through it
   in places; there the ear's skin passes 1 mm over the bone (within 3 mm of the ear), blended
   into the ear's shape. Bone further from the ear is left to the envelope;
2. mouth tissue: around the canal's mouth, the tissue the envelope would have there, out to the
   skin's depth over bone, tapering away at the edge of the zone. The envelope's skin stops short
   of the canal (its hole is the canal's whole S-shaped outline seen from the side, and it does
   not reach the tympanic bone at all), and the CT meatus comes from another head, so without
   this the ear's meatus would open onto bare bone and the hollow under the scalp skin;
3. the lumen carved out: the EAC, and past its lateral end a short, narrowing continuation, so
   the segmentation's end runs smoothly into the natural meatus beyond it. The edge is rounded.

Before 1, holes right through the CT ear are plugged (its own canal stub runs through its
block, off the volume's canal); the volume's canal is carved afterwards.

Then the scalp skin gives way wherever it passes through that solid, and the solid gives way
wherever it lies in bone or under the scalp skin that is left (the periauricular tissue in the
CT that was buried under the envelope). The two meet at one line. Except in the auricle's
bowls: the envelope's thickness is a model (``envelope.AuricleRamp`` thins it around the ear),
and the CT ear is another head's, so the concha floor can dip under the scalp skin, and cutting
it there would hole the bowl. Seen from lateral, a bowl is where the ear dips
under the scalp skin in a hollow its raised parts enclose all round (``bowl_mask``); there the
scalp skin and every layer under it give way, and the ear stays whole. Held canal down, water
poured into the auricle leaves only over the rim or down the canal. The canal is lined with the
EAC's own surface where nothing else backs it (in bone, under the skin), facing the lumen, so
the ear's carved meatus runs on into an open canal down to the eardrum.

Everything is in volume millimetres. The ear stays a separate mesh (its cartilage framework is
built from it later).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

import auricle as au
import ear_cuts as ec
import envelope as ev

GRID_MM = 0.25
BLEND_MM = 1.5  # radius of the rounded edge where the canal is carved out of the ear
UNION_MM = 1.0  # radius of the fillet where the mouth tissue meets the ear
EAC_SMOOTH_MM = 0.4  # smooths the segmentation's voxel steps
PAD_MM = 6.0
FLARE_LEN_MM = 12.0  # the lumen carries on past the EAC's lateral end this far ...
FLARE_TAN = 0.4  # ... narrowing by this much per mm, so the canal's end runs into the natural meatus
LINING_MM = 0.2  # the canal lining sits this far inside the EAC (thin canal skin over bone; clear of the bone mesh)
PROBE_MM = 0.5  # what backs the canal wall is read this far outside it
MOUTH_MM = 6.0  # mouth tissue reaches this far past the radius of the canal's end ...
MOUTH_TAPER_MM = 3.0  # ... thinning to nothing over the last few mm
MOUTH_MEDIAL_MM = 10.0  # ... and from this far medial of the canal's end outward
MOUTH_OVER_MM = 0.3  # it stands this far above the skin's depth, so the scalp skin lies inside it
DRAPE_MM = 1.0  # where bone meets the ear, the ear's skin passes this far over it ...
DRAPE_REACH_MM = 3.0  # ... within this distance of the ear, thinning away from it
DRAPE_BLEND_MM = 1.5  # radius of the blend between the drape and the ear's own surface
BOWL_PIX_MM = 0.3  # the side-on map that finds the auricle's bowls
BOWL_CLOSE_MM = 1.0  # gaps this narrow in the auricle's raised parts still enclose a bowl
BOWL_MARGIN_MM = 1.0  # the layers give way this far past a bowl's floor
FLATTEN_EDGE_MM = 2.0  # a flatten loop's edge is a smooth slope this wide, straddling the loop
FLATTEN_UNDER_MM = 0.5  # inside it the CT tissue is pressed this far under the envelope's skin
FLATTEN_DEPTH_MM = 6.0  # default reach of a loop past the first surface it was drawn over
BEHIND_BONE_MM = 40.0  # CT tissue with bone this close lateral of it lies behind the bone: cut
PATCH_MM = 0.75  # holes right through the CT ear (its own canal stub) are plugged, blended over this ...
PATCH_REACH_MM = 1.0  # ... reaching this far past the hole's outline seen from lateral


def smooth_max(a, b, k):
    """Polynomial smooth maximum: max(a, b) with the crease rounded over about k."""
    if k <= 0:
        return np.maximum(a, b)
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + k * h * (1.0 - h)


def smooth_min(a, b, k):
    return -smooth_max(-a, -b, k)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class EacField:
    """Signed distance to the EAC segmentation (mm, negative inside), smoothed, on its voxel grid.

    ``lumen`` adds the flare: the EAC's lateral end carried further lateral, narrowing, so where
    the segmentation stops inside the ear the canal still opens into the meatus beyond it.
    """

    def __init__(self, vol, pad_mm=PAD_MM, smooth_mm=EAC_SMOOTH_MM, flare_len_mm=FLARE_LEN_MM, flare_tan=FLARE_TAN):
        mm = np.asarray(vol.mm, float)
        idx = np.argwhere(vol.eac)
        pad = np.ceil(pad_mm / mm).astype(int)
        lo = np.maximum(idx.min(axis=0) - pad, 0)
        hi = np.minimum(idx.max(axis=0) + pad + 1, np.array(vol.shape))
        m = vol.eac[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        sd = ndimage.distance_transform_edt(~m, sampling=mm) - ndimage.distance_transform_edt(m, sampling=mm)
        sd = sd - np.sign(sd) * 0.5 * mm.mean()  # the boundary lies between voxel centres
        if smooth_mm > 0:
            sd = ndimage.gaussian_filter(sd, smooth_mm / mm)
        self.sd = sd.astype(np.float32)
        self.mm = mm
        self.origin = lo * mm
        self.lateral = np.asarray(vol.frame.lateral, float)
        pts = idx * mm
        t = pts @ self.lateral
        self.t_end = float(np.percentile(t, 99.5))
        end = pts[t > self.t_end - 2.0]
        self.end_centre = end.mean(axis=0)
        self.end_radius = float(np.linalg.norm(end - self.end_centre, axis=1).max())
        self.flare_len, self.flare_tan = flare_len_mm, flare_tan

    def __call__(self, points):
        g = (np.asarray(points, float) - self.origin) / self.mm
        return ndimage.map_coordinates(self.sd, g.T, order=1, mode="constant", cval=1e3)

    def axial(self, points):
        """(t, radius): along the lateral axis from the canal's end, and distance from that axis."""
        r = np.asarray(points, float) - self.end_centre
        t = r @ self.lateral
        return t, np.linalg.norm(r - np.outer(t, self.lateral), axis=1)

    def lumen(self, points):
        """The canal's air: the EAC, and past its lateral end a narrowing continuation."""
        p = np.asarray(points, float)
        out = self(p)
        t, rad = self.axial(p)
        near = (t > -3.0) & (t < self.flare_len + 1.0) & (rad < self.end_radius + 3.0)
        if near.any():
            q = p[near]
            best = out[near]
            for s in np.arange(0.5, self.flare_len + 1e-6, 0.5):
                best = np.minimum(best, self(q - s * self.lateral) + s * self.flare_tan)
            out[near] = best
        return out

    def mesh(self, level=0.0):
        """The EAC's surface at ``level`` (normals outward, into the tissue)."""
        return ev.marching_cubes_field(self.sd, self.mm, self.origin, level)


def skin_depth_at_canal(env, thickness):
    """Total soft tissue over bone at the canal's opening, read from the envelope's own thickness
    fields there (with its ``auricle_ramp``, if set): no fat or temporalis at the canal."""
    tf = env.thickness_fields(ev.Muscle(), thickness)
    total = sum(np.broadcast_to(t, env.shape) for n, t in tf.items() if n != "temporalis_muscle")
    near = (env.canal_distance() < 1.5) & env.exterior & (env.dist > 0) & (env.dist < 3.0)
    return float(np.median(total[near])) if near.any() else float(np.max(total))


def mouth_tissue(env, eac, points, skin_mm):
    """Field < 0 in the tissue around the canal's mouth: within skin depth of bone, in a tapered zone."""
    p = np.asarray(points, float)
    t, rad = eac.axial(p)
    w = _smoothstep((eac.end_radius + MOUTH_MM - rad) / MOUTH_TAPER_MM) * _smoothstep((t + MOUTH_MEDIAL_MM) / MOUTH_TAPER_MM)
    out = np.full(len(p), 1e3, np.float32)
    m = w > 0
    g = (p[m] - env.origin) / env.h
    d = ndimage.map_coordinates(env.dist, g.T, order=1, mode="nearest")  # 0 in bone
    out[m] = d + 0.05 - (skin_mm + MOUTH_OVER_MM) * w[m]
    return out


def bone_drape(env, ear_sd, points):
    """Field < 0 within ``DRAPE_MM`` of bone near the ear: skin the ear lays over bone it meets.

    The CT ear comes from another head: its back surface and crease run into this volume's bone,
    and bone pokes through it in places. Joined to the ear (smoothly), this lets the ear's skin pass
    over that bone, following the ear's contours, instead of leaving a gap or bare bone. Bone
    further from the ear is the envelope's business. ``ear_sd``: the ear's signed distance.
    """
    p = np.asarray(points, float)
    w = _smoothstep((DRAPE_REACH_MM - np.asarray(ear_sd)) / DRAPE_REACH_MM)
    out = np.full(len(p), 1e3, np.float32)
    m = w > 0
    g = (p[m] - env.origin) / env.h
    d = ndimage.map_coordinates(env.dist, g.T, order=1, mode="nearest")  # 0 in bone
    out[m] = d + 0.05 - DRAPE_MM * w[m]
    return out


def _grow(tris, allowed, seeds):
    """Vertices in ``allowed`` connected (through ``allowed`` vertices) to a seed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    t = np.asarray(tris)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = e[allowed[e].all(axis=1)]
    n = len(allowed)
    _k, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    return allowed & np.isin(lab, np.unique(lab[seeds]))


class SideMap:
    """A grid seen from the lateral side: (anterior, superior) pixels, heights along lateral."""

    def __init__(self, frame, points, pix=BOWL_PIX_MM, pad_mm=5.0):
        self.L, self.A, self.S = (np.asarray(x, float) for x in (frame.lateral, frame.anterior, frame.superior))
        p = np.asarray(points, float)
        self.c = p.mean(axis=0)
        ab = (p - self.c) @ np.stack([self.A, self.S], axis=1)
        self.lo = ab.min(axis=0) - pad_mm
        self.shape = tuple(np.ceil((ab.max(axis=0) + pad_mm - self.lo) / pix).astype(int) + 1)
        self.pix = pix
        self.top = float(((p - self.c) @ self.L).max()) + 20.0

    def coords(self, points):
        """(a, b) pixel coordinates (float) and height of points."""
        r = np.asarray(points, float) - self.c
        return (r @ np.stack([self.A, self.S], axis=1) - self.lo) / self.pix, r @ self.L

    def heights(self, verts, tris):
        """Height of the first surface met coming from lateral, per pixel (nan: none)."""
        import open3d as o3d

        i, j = np.indices(self.shape)
        start = self.c + self.top * self.L + (self.lo[0] + i[..., None] * self.pix) * self.A + (self.lo[1] + j[..., None] * self.pix) * self.S
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(verts, np.float32)), o3d.core.Tensor(np.asarray(tris, np.uint32)))
        rays = np.concatenate([start.reshape(-1, 3), np.broadcast_to(-self.L, (start.size // 3, 3))], axis=1)
        th = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy().reshape(self.shape)
        return np.where(np.isfinite(th), self.top - th, np.nan)

    def sample(self, image, points, order=1, cval=0.0):
        ab, _h = self.coords(points)
        return ndimage.map_coordinates(np.asarray(image, np.float32), ab.T, order=order, mode="constant", cval=cval)


def holes_through(side, ear, eac):
    """Holes right through the ear seen from lateral, except the canal's mouth: a 2D mask.

    A hole is the canal's if most of its rays meet the canal's lumen within the ear's own depth
    there (from 3 mm under the rim around it to the rim's top). ``eac`` None: every hole.
    """
    he = side.heights(*ear)
    foot = np.isfinite(he)
    hole = ndimage.binary_fill_holes(foot) & ~foot
    lab, n = ndimage.label(hole)
    out = np.zeros(side.shape, bool)
    for k in range(1, n + 1):
        m = lab == k
        rim = he[ndimage.binary_dilation(m, iterations=3) & foot]
        if eac is None:
            out |= m
            continue
        i, j = np.nonzero(m)
        over = np.zeros(len(i), bool)
        for z in np.arange(rim.min() - 3.0, rim.max() + 0.25, 0.25):
            q = side.c + z * side.L + (side.lo[0] + i[:, None] * side.pix) * side.A + (side.lo[1] + j[:, None] * side.pix) * side.S
            over |= eac.lumen(q) < 0
        if over.mean() < 0.5:
            out |= m
    return out


def patch_holes(sd, pts, side, holes, he):
    """The ear's signed distance with holes through it plugged: under each hole, everything up to
    a cap fitted to the surface around it (``he``: the ear's heights from lateral) is filled,
    blended into the ear over ``PATCH_MM``.
    """
    if not holes.any():
        return sd
    lab, n = ndimage.label(holes)
    ab, h = side.coords(pts)
    out = np.asarray(sd, np.float32).copy()
    for k in range(1, n + 1):
        m = lab == k
        rim = ndimage.binary_dilation(m, iterations=int(np.ceil(1.0 / side.pix))) & np.isfinite(he)
        i, j = np.nonzero(rim)
        A = np.stack([np.ones(len(i)), i, j], axis=1)
        c, *_ = np.linalg.lstsq(A, he[rim], rcond=None)
        cap = c[0] + c[1] * ab[:, 0] + c[2] * ab[:, 1]
        # distance (mm) outside the hole grown by PATCH_REACH_MM, seen from lateral
        grown = ndimage.binary_dilation(m, iterations=int(np.ceil(PATCH_REACH_MM / side.pix)))
        d2 = (ndimage.distance_transform_edt(~grown) - ndimage.distance_transform_edt(grown)) * side.pix
        across = ndimage.map_coordinates(d2.astype(np.float32), ab.T, order=1, mode="constant", cval=1e3)
        plug = np.maximum(h - cap, across)
        out = smooth_min(out, plug.reshape(out.shape), PATCH_MM)
    return out


def height_over_envelope(env, thickness, muscle, points):
    """How far points stand out of the envelope's outer skin (mm, < 0 under it, in bone ~ -total)."""
    tf = env.thickness_fields(muscle or ev.Muscle(), thickness)
    total = sum(np.broadcast_to(t, env.shape) for t in tf.values()).astype(np.float32)
    g = ((np.asarray(points, float) - env.origin) / env.h).T
    return ndimage.map_coordinates(env.dist, g, order=1, mode="nearest") - ndimage.map_coordinates(total, g, order=0, mode="nearest")


def flatten_weight(points, regions, ear=None):
    """1 inside the flatten loops (``ear_cuts.MeshCut`` in volume mm), 0 outside, a smooth slope
    ``FLATTEN_EDGE_MM`` wide across each loop's outline. With ``ear`` (verts, tris) and a loop's
    ``depth_mm`` > 0, the loop reaches only that far past the first surface of the ear met along
    the view, so tissue on the far side (the auricle behind the loop) is left alone."""
    w = np.zeros(len(points), np.float32)
    p = np.asarray(points, float)
    for c in regions or ():
        r = ec._region(p, ec.MeshCut(c.target, c.outline_obj, c.direction_obj, 0.0))
        if ear is not None and c.depth_mm > 0:
            r = np.maximum(r, _past_first_surface(p, c, ear) - c.depth_mm)
        w = np.maximum(w, _smoothstep(0.5 - r / FLATTEN_EDGE_MM))
    return w


def _past_first_surface(points, cut, ear, res=0.3):
    """How far along the loop's view direction points lie past the ear's first surface (mm)."""
    import open3d as o3d

    d = np.asarray(cut.direction_obj, float)
    d /= np.linalg.norm(d)
    a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    o = np.asarray(cut.outline_obj, float)
    uv_o = np.stack([o @ a, o @ b], axis=1)
    lo = uv_o.min(axis=0) - 2 * FLATTEN_EDGE_MM
    n = np.ceil((uv_o.max(axis=0) + 2 * FLATTEN_EDGE_MM - lo) / res).astype(int) + 1
    i, j = np.indices(n)
    v = np.asarray(ear[0], float)
    t_start = float((v @ d).min()) - 5.0
    start = (lo[0] + i[..., None] * res) * a + (lo[1] + j[..., None] * res) * b + t_start * d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(np.asarray(ear[1], np.uint32)))
    rays = np.concatenate([start.reshape(-1, 3), np.broadcast_to(d, (start.size // 3, 3))], axis=1).astype(np.float32)
    th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(n)
    first = np.where(np.isfinite(th), t_start + th, 1e3).astype(np.float32)
    first = ndimage.grey_erosion(first, size=3)  # a little margin at the silhouette
    uv_p = np.stack([points @ a, points @ b], axis=1)
    f = ndimage.map_coordinates(first, ((uv_p - lo) / res).T, order=1, mode="nearest")
    return points @ d - f


def painted_auricle(env, thickness, muscle, verts, weight, k=4):
    """f(points): the painted auricle weight (0..1) of the nearest ear vertices (inverse-distance
    over ``k``). Vertices buried under the envelope's skin are left out: their paint does not
    matter (they are trimmed), and inside a thick block the visible surface's paint should win."""
    from scipy.spatial import cKDTree

    v = np.asarray(verts, float)
    w = np.clip(np.asarray(weight, float), 0.0, 1.0)
    keep = height_over_envelope(env, thickness, muscle, v) > -0.5
    if keep.sum() >= k:
        v, w = v[keep], w[keep]
    tree = cKDTree(v)

    def f(points):
        d, i = tree.query(np.asarray(points, float), k=k)
        wt = 1.0 / np.maximum(d, 1e-3)
        return ((w[i] * wt).sum(axis=1) / wt.sum(axis=1)).astype(np.float32)

    return f


def _not_auricle(aw):
    """1 where the paint says not auricle (weight under 0.25), 0 over 0.75, smooth between."""
    return _smoothstep((0.75 - np.asarray(aw)) / 0.5)


def behind_bone(vol, points, reach_mm=BEHIND_BONE_MM, step_mm=0.5):
    """True where bone lies lateral of a point (it is behind the bone, as seen from lateral)."""
    p = np.asarray(points, float)
    lat = np.asarray(vol.frame.lateral, float)
    out = np.zeros(len(p), bool)
    shape = np.array(vol.shape)
    for s in np.arange(step_mm, reach_mm + 1e-6, step_mm):
        idx = np.round((p + s * lat) / vol.mm).astype(int)
        ok = np.all((idx >= 0) & (idx < shape), axis=1) & ~out
        out[ok] = vol.solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out


def bowl_mask(side, ear, scalp):
    """The auricle's bowls seen from lateral: where the ear's surface dips under the scalp skin
    in a hollow its raised parts (standing out of the scalp, or with no scalp behind them)
    enclose all round: the concha, cymba, fossae. Periauricular tissue under the scalp that
    reaches the ear's edge is not a bowl. Gaps under ``BOWL_CLOSE_MM`` in the raised parts
    still enclose (a narrow notch). The result reaches ``reach_mm`` further, under overhangs
    (crus of the helix) where the floor is hidden from lateral, but never into edge tissue.
    """
    he = side.heights(*ear)
    hs = side.heights(*scalp)
    ear_px = np.isfinite(he)
    under = ear_px & np.isfinite(hs) & (hs > he)
    r = int(np.ceil(BOWL_CLOSE_MM / side.pix))
    disk = np.hypot(*np.mgrid[-r : r + 1, -r : r + 1]) <= r
    raised = ndimage.binary_closing(np.pad(ear_px & ~under, r + 1), disk)[r + 1 : -r - 1, r + 1 : -r - 1]
    lab, _n = ndimage.label(~raised)  # under-scalp patches and the outside
    out_lab = np.unique(lab[~ear_px])
    edge = ndimage.binary_dilation(np.isin(lab, out_lab[out_lab > 0]) & ear_px, disk) & under
    return under & ~edge, edge  # (small patches the closing covers are still bowls)


def open_far_end(cv, ct, eac, cos_min=0.6):
    """The canal lining without its medial cap (the segmentation's end at the eardrum), so the
    middle ear shows through: faces in the canal's medial half that face lateral (into the lumen
    from its end). ``ct`` faces the lumen."""
    cv, ct = np.asarray(cv), np.asarray(ct)
    if not len(ct):
        return cv, ct
    fn = np.cross(cv[ct[:, 1]] - cv[ct[:, 0]], cv[ct[:, 2]] - cv[ct[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    tc, _r = eac.axial(cv[ct].mean(axis=1))
    t_all, _ = eac.axial(cv)
    medial = tc < 0.5 * (t_all.min() + t_all.max())
    cap = medial & (fn @ eac.lateral > cos_min)
    keep = ~cap
    used = np.unique(ct[keep])
    remap = np.full(len(cv), -1)
    remap[used] = np.arange(len(used))
    return cv[used], remap[ct[keep]]


def _is_open(tris):
    """True if the mesh has open edges (a sheet, not a closed solid)."""
    t = np.asarray(tris)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _u, c = np.unique(e, axis=0, return_counts=True)
    return bool(np.any(c == 1))


def open_meatus(vol, env, ear, scalp_skin, thickness, grid_mm=GRID_MM, eac=None, log=print, others=None, flatten=None, muscle=None,
                auricle=None, framework_max_mm=0.0, cartilage=None):
    """The placed CT ear opened onto the canal, trimmed, and joined to the scalp skin.

    ``ear``: (verts, tris), volume mm, closed. ``scalp_skin``: the envelope's skin layer.
    ``flatten``: loops (``ear_cuts.MeshCut``, volume mm) around CT tissue that is not auricle
    (periauricular skin the segmentation took in): inside them the tissue is pressed under the
    envelope's skin, sloping up to the auricle at their edge, so the scalp skin covers it.
    ``auricle``: per-vertex weight of ``ear`` (1 = auricle, painted in Blender, ``blender_edit``).
    If ``ear`` is then an open skin sheet (cut down in Blender) it is joined as it is
    (``sheet_skin.join_sheet``: skirt onto the scalp skin, canal lining, cartilage from the sheet);
    ``framework_max_mm`` is then the cartilage's thickness under a single skin (0: the default).
    ``cartilage``: (verts, tris), volume mm, the cartilage as edited in Blender (sheet only); its
    medial face is covered by skin where it goes on past the sheet's base edge.
    Unpainted tissue is pressed under the envelope's skin like a flatten loop, and the cartilage
    framework is built inside the painted auricle only: the carved solid ``au.COVER_MM`` (skin
    and perichondrium) in from its surface, down to the envelope's skin, at most
    ``framework_max_mm`` thick (0: no limit, filling the space under the concha and around the
    canal).
    Returns dict: ear (open, volume mm), scalp (opened where the ear rises out of it), canal
    (lining, normals into the lumen), skin_mm (tissue depth over bone around the canal's mouth),
    framework (with ``auricle``).
    """
    import time

    import open3d as o3d

    import surface

    t0 = time.time()
    eac = eac or EacField(vol)
    if auricle is not None and _is_open(ear[1]):  # a skin sheet from Blender, not a solid: join it as it is
        import cartilage as cartilage_mod
        import sheet_skin

        return sheet_skin.join_sheet(vol, env, (np.asarray(ear[0], float), np.asarray(ear[1]), np.asarray(auricle)), scalp_skin,
                                     thickness, muscle=muscle, eac=eac, others=others,
                                     framework_max_mm=framework_max_mm if framework_max_mm > 0 else cartilage_mod.MAX_MM,
                                     cartilage=cartilage, log=log)
    skin_mm = skin_depth_at_canal(env, thickness)
    v = np.asarray(ear[0], float)
    lo = v.min(axis=0) - 1.5
    hi = v.max(axis=0) + 1.5
    shape = np.ceil((hi - lo) / grid_mm).astype(int) + 1
    axes = [lo[k] + grid_mm * np.arange(shape[k]) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    ear_sd = ec._sdf(v, ear[1], pts)
    # the CT's own canal stub runs right through its block, off the volume's canal: closed,
    # then the volume's canal is carved (below)
    side0 = SideMap(vol.frame, v)
    he0 = side0.heights(v, ear[1])
    holes = holes_through(side0, (v, ear[1]), None)
    ear_sd = patch_holes(ear_sd.reshape(shape), pts, side0, holes, he0).ravel()
    if holes.any():
        log(f"patched {holes.sum() * side0.pix ** 2:.1f} mm2 of holes through the ear")
    paint = painted_auricle(env, thickness, muscle, v, auricle) if auricle is not None else None
    fw = flatten_weight(pts, flatten, (v, ear[1])) if flatten else np.zeros(len(pts), np.float32)
    if paint is not None:  # what is not painted auricle is pressed under too
        near = ear_sd < 2.0
        fw[near] = np.maximum(fw[near], _not_auricle(paint(pts[near])))
    if fw.any():  # periauricular tissue: pressed under the envelope's skin
        m = fw > 0
        lim = height_over_envelope(env, thickness, muscle, pts[m]) + FLATTEN_UNDER_MM - 30.0 * (1.0 - fw[m])
        ear_sd[m] = smooth_max(ear_sd[m], lim, 1.0)
    field = smooth_min(ear_sd, bone_drape(env, ear_sd, pts), DRAPE_BLEND_MM)  # skin over bone the ear meets
    field = smooth_min(field, mouth_tissue(env, eac, pts, skin_mm), UNION_MM)
    field = smooth_max(field, -eac.lumen(pts), BLEND_MM)  # carve the canal, edge rounded
    ov, ot = ev.marching_cubes_field(field.reshape(shape), np.full(3, grid_mm), lo)
    ov = surface.smooth_surface(ov, ot, iterations=3, lam=0.5, max_step=0.5 * grid_mm)
    log(f"carved: {len(ot)} tris, {time.time() - t0:.1f} s")
    # the scalp skin gives way inside the solid; the solid is buried only under the scalp skin that
    # is left (tested against the whole scalp, the meatus walls would go wherever the scalp
    # passed through the ear, and nothing would cover the bone behind them)
    carved = o3d.t.geometry.RaycastingScene()
    carved.add_triangles(o3d.core.Tensor(ov.astype(np.float32)), o3d.core.Tensor(np.asarray(ot, np.uint32)))
    sv, st = (np.asarray(x) for x in scalp_skin)
    scalp = au.clip_mesh(sv, st, -carved.compute_signed_distance(o3d.core.Tensor(sv.astype(np.float32))).numpy())
    deep = ec.deep_to_skin(scalp, vol)
    cut = -deep(ov)  # > 0: in bone or under the scalp skin
    # the auricle's bowls stay whole: where the concha (or any hollow the auricle's raised parts
    # enclose) dips under the scalp skin, the skin and the layers under it give way instead, so
    # water poured into the auricle can only leave over its rim or down the canal
    side = SideMap(vol.frame, ov)
    bowl, edge = bowl_mask(side, (ov, ot), scalp)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(ov), o3d.utility.Vector3iVector(ot))
    nl = np.asarray(m.compute_vertex_normals().vertex_normals) @ side.L
    idx = np.round(ov / vol.mm).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
    in_bone = np.zeros(len(ov), bool)
    in_bone[ok] = vol.solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    # the bowl's floor on the mesh: from what lateral sees of it, along the surface under any
    # overhang, but not into periauricular tissue at the ear's edge
    edge_near = ndimage.binary_dilation(edge, iterations=int(np.ceil(1.0 / side.pix)))
    cand = (cut > 0) & ~in_bone & (nl > -0.3) & (side.sample(edge_near, ov, order=0) < 0.5)
    keep = _grow(ot, cand, cand & (side.sample(bowl, ov, order=0) > 0.5))
    floor = np.zeros(side.shape, bool)
    if keep.any():
        ab, _h = side.coords(ov[keep])
        ij = np.round(ab).astype(int)
        floor[ij[:, 0], ij[:, 1]] = True
        floor = ndimage.binary_closing(floor, iterations=2) & ~edge_near
    # signed distance from the floors (mm, < 0 within BOWL_MARGIN_MM of them), on the side map
    give = (ndimage.distance_transform_edt(~floor) - ndimage.distance_transform_edt(floor)) * side.pix - BOWL_MARGIN_MM
    if floor.any():
        scalp = au.clip_mesh(*scalp, side.sample(-give, scalp[0], cval=-1e3))
        deep = ec.deep_to_skin(scalp, vol)
        cut = -deep(ov)
        cut[keep] = -np.abs(cut[keep])
    log(f"bowls kept: {int(keep.sum())} vertices over {floor.sum() * side.pix ** 2:.0f} mm2")
    cut[behind_bone(vol, ov)] = 1.0  # CT tissue round the back of the bone
    if flatten or paint is not None:  # flattened tissue past the envelope's edge (no scalp skin over it to cut it)
        fw = flatten_weight(ov, flatten, (v, ear[1])) if flatten else np.zeros(len(ov), np.float32)
        if paint is not None:
            fw = np.maximum(fw, _not_auricle(paint(ov)))
        gone = (fw > 0.95) & (height_over_envelope(env, thickness, muscle, ov) < 0)
        cut[gone] = np.maximum(cut[gone], 1.0)
    opened = au.clip_mesh(ov, ot, cut)
    opened = ev._largest_component(*opened, share=0.01)[:2]  # scraps left in pores inside bone
    log(f"joined: {time.time() - t0:.1f} s")
    # the lining: the EAC's wall where the carved solid does not back it, i.e. in bone or under the
    # scalp skin; not where the lumen carries on past the EAC's lateral end
    cv, ct = eac.mesh(-LINING_MM)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cv), o3d.utility.Vector3iVector(ct))
    q = cv + PROBE_MM * np.asarray(m.compute_vertex_normals().vertex_normals)
    val = np.maximum(deep(q), -eac.lumen(q))
    cv, ct = au.clip_mesh(cv, ct[:, [0, 2, 1]], val)  # facing the lumen
    cv, ct = open_far_end(cv, ct, eac)  # open at the eardrum: the middle ear shows
    log(f"canal lining: {len(ct)} tris, {time.time() - t0:.1f} s")
    layers = {}
    for name, mesh in (others or {}).items():  # deeper layers give way under the bowls too
        lv, lt = (np.asarray(x) for x in mesh)
        layers[name] = au.clip_mesh(lv, lt, side.sample(-give, lv, cval=-1e3)) if floor.any() else (lv, lt)
    framework = None
    if paint is not None:  # cartilage inside the painted auricle only
        inside = field < 0
        aw = np.zeros(len(pts), np.float32)
        aw[inside] = paint(pts[inside])
        hg = np.full(len(pts), -1.0, np.float32)
        hg[inside] = height_over_envelope(env, thickness, muscle, pts[inside])
        cart = np.maximum.reduce([field + au.COVER_MM, 0.5 - aw, -hg])
        if framework_max_mm > 0:
            cart = np.maximum(cart, -field - (au.COVER_MM + framework_max_mm))
        fv, ft = ev.marching_cubes_field(cart.reshape(shape), np.full(3, grid_mm), lo)
        if len(ft):
            fv = surface.smooth_surface(fv, ft, iterations=3, lam=0.5, max_step=0.5 * grid_mm)
            fv, ft = ev._largest_component(fv, ft, share=0.02)[:2]
            framework = (fv, ft)
        log(f"framework: {0 if framework is None else len(framework[1])} tris, {time.time() - t0:.1f} s")
    return dict(ear=opened, scalp=scalp, canal=(cv, ct), skin_mm=skin_mm, layers=layers, framework=framework,
                bowl=dict(side=side, mask=bowl, floor=floor))
