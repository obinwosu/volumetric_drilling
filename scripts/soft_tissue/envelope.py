"""Soft-tissue envelope over the lateral skull of a FIVRS volume.

The volumes hold bone (and a few labels) but no soft tissue. This builds the
layers that lie on the lateral bone surface, inside to outside:

  periosteum          0.3 mm everywhere
  temporalis muscle   only under the outline picked in the placement viewer;
                      thickest inside, tapering to nothing at its edge
  temporalis fascia   invests the muscle; ends where the muscle ends
  loose areolar/TPF   1.0 mm
  subcutaneous fat    1.375 mm, half of what would fill Wrobel 2012's 5.25 mm
                      skin-to-periosteum mean (periauricular fat is sparse), and
                      none at the EAC opening, tapering in over FAT_FREE_MM
  skin                1.5 mm

Each layer's outer surface is the level set {distance to bone = sum of the
thicknesses up to that layer}. Distance offsets round off bony crevices the
way soft tissue does. Thickness can vary over the skull (the muscle does): it
is looked up at the nearest bone point, projected onto the plane across the
canal axis (the "uv" plane), where the muscle outline is drawn.

Only the lateral bony surface gets tissue. The FIVRS lateral cortex mask
(``rtx_view/cortex_masks.py`` mask 1, the surface the RTX tuner shades) is the
guide: it seeds a region that grows (up to ``GROW_MM`` along the surface) over
connected surface of the FIVRS bone representation: every solid label but
sinus/dura (the condyle counts), closed by flat plates where the volume was
cropped, as AMBF and the RTX window draw it. The region takes surface that
sees out laterally and faces lateral (within ``FACING_DEG``), stops short of
the sinus/dura label, and has small gaps closed. A tissue point is kept only
where its nearest solid is in that region and, looking down the lateral axis,
the first solid it meets is covered and is not sinus/dura. Medial bone gets
nothing. Under the picked temporalis outline, lateral-facing bone is covered
as well (``MUSCLE_FACING_DEG``): the muscle fills the temporal fossa and the
space by the zygomatic root, and the skin drapes over it. Surface over the
non-lateral crop faces is dropped (the envelope ends there, open), and the
ear canal is punched out (canal skin is built separately, ``canal_skin.py``).

Coordinates are volume millimetres: voxel index (x column, y row from the
bottom, z slice, as ``fivrs_rtx_view._load_occupancy``) times the voxel size.
``to_world_au`` maps them to the AMBF world frame.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/envelope.py ADF/volume_256_RT143.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rtx_view"))

import canal_skin as cs  # noqa: E402
import surface  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "resources" / "soft_tissue" / "envelope"

# name, nominal thickness (mm), source. Muscle and fascia are 0 outside the outline.
# Retroauricular soft tissue, skin surface to periosteum: mean 5.25 mm, range 2.0-11.0, by
# ultrasound in 100 patients (Wrobel, Popko, Szyfter 2012, Otol Neurotol 33:421-4); it rises
# with BMI. CT in cochlear-implant patients gives a leaner 4.5 +- 1.2 mm (range 2-7).
# Over the mastoid the layers above periosteum add up to 5.25 mm.
LAYERS = (
    ("periosteum", 0.3, "estimate 0.1-0.5 mm (mastoid periosteum)"),
    ("temporalis_muscle", 0.0, "picked outline; thickness and taper from the viewer"),
    ("temporalis_fascia", 1.0, "user (2026-11-11): even over the muscle, submillimetre (~0.6 mm), drawn 1.0 mm; "
                               "Zwirner 2021 measured 1.46 mm"),
    # layers 4 and 3 of the scalp (user, 2026-11-11): the loose areolar plane (4) is a potential
    # space: nothing at rest, the musculoaponeurotic layer (3: temporoparietal fascia / SMAS, with the
    # extrinsic auricular muscles in it) lies straight on the temporalis fascia / periosteum, under the
    # fat. It opens on traction or dissection (drawn then as fibres, auricle_pull_viewer). The old
    # single 1.0 mm "loose areolar" layer is all layer 3 now, so the total over the mastoid is unchanged.
    ("loose_areolar", 0.0, "user: a potential space at rest (layer 4)"),
    ("auricular_muscles", 0.0, "layer 3: the auricular muscles' bellies, a thickness map (auricular_muscles.py)"),
    ("tpf", 1.0, "estimate (layer 3): temporoparietal fascia / SMAS; 2-3 mm higher up the scalp"),
    ("subcutaneous_fat", 1.375, "half the 2.75 mm that fills Wrobel's 5.25 mm: periauricular fat is sparse (user); none at the canal"),
    ("skin", 1.5, "tissue_params skin (estimate)"),
)
LAYER_NAMES = tuple(name for name, _t, _s in LAYERS)
LAYER_RGB = {
    "periosteum": (185, 140, 235),
    "temporalis_muscle": (205, 45, 55),
    "temporalis_fascia": (190, 215, 255),
    "loose_areolar": (120, 220, 200),
    "auricular_muscles": (190, 60, 55),
    "tpf": (235, 200, 185),
    "subcutaneous_fat": (255, 190, 40),
    "skin": (235, 150, 120),
}

DOWNSAMPLE = 2  # envelope grid = 2 voxels (~0.39 mm)
LATERAL_PAD_MM = 40.0  # room outside the cube for tissue and muscle
UV_MM = 0.5  # muscle map resolution
FASCIA_TAPER_MM = 1.5  # fascia thins to nothing over this distance at the muscle edge
MESH_MUSCLE_SMOOTH_MM = 2.0  # a placed muscle mesh's thickness map is smoothed over this (Gaussian sigma)
MESH_MUSCLE_TAPER_MM = 3.0  # and tapers in from its outline over this
CANAL_MARGIN_MM = 0.3
VIEW_CONE_DEG = 40.0  # exterior = air that sees out within this angle of lateral
FACING_DEG = 65.0  # lateral bone surface: outward normal within this angle of lateral
MUSCLE_FACING_DEG = 85.0  # under the muscle outline, bone facing within this angle of lateral is covered too
MUSCLE_EDGE_MM = 6.0  # coverage under the muscle reaches this far past its outline
GROW_MM = 6.0  # the region grows at most this far (along the surface) from the FIVRS mask
LABEL_CLEAR_MM = 0.4  # the region stops this far short of exposed sinus/dura
DURA_SHARE = 0.8  # surface is exposed dura where this share of it within DURA_RADIUS_MM is dura;
DURA_RADIUS_MM = 1.0  # a speckled bone/dura plate (thin bone over the sinus) counts as bone
SINUS_DURA_RGB = (0x6E, 0xB8, 0xD1)  # "sinus_dura" in tuner_materials.json
GAP_CLOSE_MM = 1.5  # close holes in the region up to about this size
FACE_COS = 0.3  # layers only over bone facing out to the side (normal within ~70 deg of lateral)
EDGE_COS = 0.65  # trim tissue that wraps round the cortex edge: offset direction vs bone normal
FAT_FREE_MM = 10.0  # fat tapers to nothing within this distance of the EAC's lateral opening
FACE_REACH_MM = 28.0  # the face's tissue reaches this far forward of the auricle's front edge
FACE_TOTAL_MM = 4.2  # periosteum + loose areolar + fat + skin over the face's deep tissue (nominal)
FACE_EDGE_DROP_MM = 0.5  # the face's skin sits this little below the auricle's front edge (a slight pretragal crease)
FACE_LAYER_SMOOTH = 12  # extra smoothing passes for the layers over the face's deep tissue
FACE_SMOOTH_MM = 1.2  # the face's skin height is smoothed over about this (no grid ripples)
FACE_UNDER_MM = 3.0  # the deep tissue reaches this far back under the auricle's front edge
FACE_DEPTH_MM = 30.0  # the deep tissue (parotid, masseter, the joint) fills at most this deep under it
LEVEL_TOL = 0.75
MIN_SOLID_MM = 0.4  # thinnest layer the solids view can draw on this grid (~1 cell; display only)  # keep surface only where it is the true offset (in grid cells)  # the canal hole is the EAC's cross-section, grown this much


def upgrade_thickness(d):
    """Saved thicknesses from before layers 4 and 3 were split: the old "loose areolar" value is
    layer 3 (tpf) now, and layer 4 is a potential space (0); the old default fascia (1.46 mm) gives
    way to the current one (the user's 1.0)."""
    d = dict(d or {})
    if "loose_areolar" in d and "tpf" not in d:
        d["tpf"], d["loose_areolar"] = float(d["loose_areolar"]), 0.0
    if abs(float(d.get("temporalis_fascia", 0.0)) - 1.46) < 1e-6:
        d.pop("temporalis_fascia")
    return d


@dataclasses.dataclass
class Muscle:
    """Temporalis 'pancake': an outline on the uv plane and a thickness profile."""

    outline_mm: list = dataclasses.field(default_factory=list)  # picked points, volume mm (3D)
    max_mm: float = 8.0
    taper_mm: float = 10.0  # distance from the edge to full thickness
    # or: a placed muscle mesh (PLY, volume mm; atlas_placement.py), its thickness measured across
    # it along the canal axis, cell by cell; the outline is then where it is thicker than 0.3 mm
    mesh_ply: str = ""


AURICLE_NEAR = {  # layer thickness (mm) under the auricle and around its root: thin skin, no fat
    "skin": (1.0, "estimate (user): skin around the concha and meatus is thin"),
    "subcutaneous_fat": (0.0, "user: essentially no fat around the auricle"),
    "tpf": (0.3, "estimate: thin under the auricle"),
    "periosteum": (0.3, "unchanged: periosteum lies on the bone everywhere"),
}


@dataclasses.dataclass
class AuricleRamp:
    """The envelope thins around the ear: ``near`` thicknesses within ``near_mm`` of the canal's
    lateral opening (the concha and meatus, the front of the ear), the usual thicknesses beyond
    ``far_mm`` (behind and above the auricle), a smooth ramp between. Distance is measured as
    for the fat taper: from each cell's nearest bone point, across the canal axis, to the
    opening's outline. Layers not in ``near`` keep their thickness (temporalis muscle and fascia
    come from the picked muscle outline).
    """

    near: dict = dataclasses.field(default_factory=lambda: {n: t for n, (t, _s) in AURICLE_NEAR.items()})
    near_mm: float = 8.0
    far_mm: float = 20.0
    base_outline: np.ndarray | None = None  # the auricle's base (volume mm): measure from it, not the canal
    base_near_mm: float = 2.0  # with a base: thin within this of it (and under the auricle) ...
    base_far_mm: float = 12.0  # ... the usual thicknesses beyond this

    def weight(self, dist_mm):
        """0 within ``near_mm`` of the opening (or the auricle's base), rising smoothly to 1 at ``far_mm``."""
        near, far = (self.base_near_mm, self.base_far_mm) if self.base_outline is not None else (self.near_mm, self.far_mm)
        s = np.clip((np.asarray(dist_mm) - near) / max(far - near, 1e-6), 0.0, 1.0)
        return (s * s * (3 - 2 * s)).astype(np.float32)


@dataclasses.dataclass
class Frame:
    """Directions in volume mm."""

    lateral: np.ndarray  # canal axis, medial to lateral
    superior: np.ndarray  # world up, made perpendicular to lateral
    anterior: np.ndarray  # completes a right-handed (anterior, superior, lateral) for a right ear
    canal_center: np.ndarray
    canal_lateral_end: np.ndarray  # centre of the canal's lateral opening

    def uv(self, p):
        p = np.asarray(p, dtype=np.float64)
        return np.stack([p @ self.anterior, p @ self.superior], axis=-1)


class Volume:
    def __init__(self, adf: pathlib.Path):
        self.adf = pathlib.Path(adf)
        occupied, colors, mm, pose, laterality = cs.load_volume(self.adf)
        self.shape = occupied.shape
        self.mm = mm
        self.dimensions, self.rotation, self.translation = pose
        self.laterality = laterality
        self.eac = occupied & cs._is_rgb(colors, cs.EAC_RGB)
        self.bone = occupied & cs._is_rgb(colors, cs.BONE_RGB)
        self.dura = occupied & cs._is_rgb(colors, SINUS_DURA_RGB)
        self.solid = occupied & ~self.eac  # every label but the canal plug
        ossicles = np.zeros_like(occupied)
        for rgb in cs._ossicle_rgbs():
            ossicles |= occupied & cs._is_rgb(colors, rgb)
        center, axis = cs._canal_axis(self.eac, mm, ossicles)
        self.cortex, cmf = lateral_cortex_voxels(self.adf, self.bone)
        # Anatomical axes from the FIVRS cortex frame: anterior toward the TMJ, superior from the
        # EAC's superior tangent plane. (World up in AMBF is not anatomical superior here.)
        if getattr(cmf, "anterior_ok", False) and getattr(cmf, "superior_ok", False):
            ant = np.asarray(cmf.anterior_normal, float)
            ant = ant - (ant @ axis) * axis
            ant /= np.linalg.norm(ant)
            sup = np.asarray(cmf.superior_normal, float)
            sup = sup - (sup @ axis) * axis - (sup @ ant) * ant
            sup /= np.linalg.norm(sup)
            self.axes_from = "cortex frame (TMJ anterior, EAC superior tangent)"
        else:
            up_local = self.rotation.T @ np.array([0.0, 0.0, 1.0])  # fallback: world up
            sup = up_local - (up_local @ axis) * axis
            sup /= np.linalg.norm(sup)
            ant = np.cross(sup, axis) * (-1.0 if laterality == "left" else 1.0)
            self.axes_from = "world up (cortex frame had no anterior/superior plane)"
        pts = np.argwhere(self.eac) * mm
        t = (pts - center) @ axis
        lateral_end = center + axis * np.percentile(t, 98)
        self.frame = Frame(axis, sup, ant, center, lateral_end)
        # cortex grown a couple of voxels into the solid, so a ray that lands just beside a marked voxel still counts
        self.cortex_hit = ndimage.binary_dilation(self.cortex, iterations=2) & self.solid


    def to_world_au(self, p_mm):
        index = np.asarray(p_mm, dtype=np.float64) / self.mm
        return surface.local_to_world(surface.index_to_local(index, self.dimensions, self.shape), self.rotation, self.translation)

    def mm_to_world_matrix(self):
        """4x4 taking volume mm (column vector, homogeneous) to AMBF world units."""
        scale = self.dimensions / np.asarray(self.shape, dtype=np.float64) / self.mm
        A = np.eye(4)
        A[:3, :3] = self.rotation @ np.diag(scale)
        A[:3, 3] = self.translation - self.rotation @ (self.dimensions * 0.5)
        return A

    def bone_mesh(self, step=2, smooth=True):
        """Bone (all solid labels) surface in volume mm, for display."""
        # an empty border closes the solid where the volume was cropped: the flat plates AMBF and the RTX window draw
        occ = np.pad(self.solid[::step, ::step, ::step], 1)
        v, t, *_ = surface.marching_cubes(occ)
        if smooth:
            v = surface.smooth_surface(v, t)
        return (v - 1.0) * self.mm * step, t

    def eac_mesh(self, step=1):
        v, t, *_ = surface.marching_cubes(self.eac[::step, ::step, ::step])
        return surface.smooth_surface(v, t) * self.mm * step, t


def lateral_cortex_voxels(adf, bone):
    """Bone voxels under the FIVRS lateral cortex mask (cortex_masks mask 1), and the cortex frame."""
    import cortex_masks as cm

    occ, col, dims, lat = cm.load_intact_volume(pathlib.Path(adf))
    eac_rgb, bone_rgb = cs.EAC_RGB, cs.BONE_RGB
    frame = cm.derive_lateral_frame(occ, col, eac_rgb, bone_rgb, dims, lat)
    if not frame.ok:
        raise SystemExit(f"Lateral cortex frame failed: {frame.summary}")
    cache = cm.cache_from_occupancy(occ, col, frame, bone_rgb)
    mask = cm.evaluate_masks(cache, cm.load_parameters()).masks["mask1"]
    shape = np.asarray(occ.shape, np.float64)
    index = (cache.world.astype(np.float64) + dims * 0.5) * shape / dims  # local -> continuous voxel index
    cen = index[cache.tris[mask]].mean(axis=1)
    out = np.zeros(occ.shape, bool)
    # a surface vertex sits between voxel centres: mark the bone voxels of the cell around each centroid
    base = np.floor(cen).astype(int)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                g = np.clip(base + (dx, dy, dz), 0, np.array(occ.shape) - 1)
                out[g[:, 0], g[:, 1], g[:, 2]] = True
    return out & bone, frame


# -- scalar marching cubes ----------------------------------------------------------
def marching_cubes_field(field, spacing, origin, level=0.0):
    """Surface {field = level} of a sampled scalar field; normals point toward larger values.

    ``field`` is indexed (x, y, z); sample (i, j, k) sits at origin + (i, j, k) * spacing.
    """
    f = np.asarray(field, dtype=np.float32) - np.float32(level)
    inside = f < 0
    nx, ny, nz = f.shape
    bits = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.uint16)
    for bit, (dx, dy, dz) in enumerate(surface.CORNERS):
        bits |= inside[dx : dx + nx - 1, dy : dy + ny - 1, dz : dz + nz - 1].astype(np.uint16) << bit
    ii, jj, kk = np.nonzero((bits != 0) & (bits != 255))
    if not len(ii):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    rows = surface._TRI[bits[ii, jj, kk].astype(np.int32)]
    used = rows >= 0
    edge = rows[used].astype(np.int32)
    cell = np.repeat(np.arange(len(ii)), used.sum(axis=1))
    a = surface.CORNERS[surface.EDGES[edge, 0]]
    b = surface.CORNERS[surface.EDGES[edge, 1]]
    lo = np.minimum(a, b)
    axis = np.argmax(np.abs(b - a), axis=1)
    ex, ey, ez = ii[cell] + lo[:, 0], jj[cell] + lo[:, 1], kk[cell] + lo[:, 2]
    key = (((axis.astype(np.int64) * nx + ex) * ny + ey) * nz) + ez
    uniq, inverse = np.unique(key, return_inverse=True)
    uz = uniq % nz
    rest = uniq // nz
    uy = rest % ny
    rest //= ny
    ux = rest % nx
    uaxis = rest // nx
    p0 = np.stack([ux, uy, uz], axis=1)
    p1 = p0.copy()
    p1[np.arange(len(p1)), uaxis] += 1
    f0 = f[p0[:, 0], p0[:, 1], p0[:, 2]]
    f1 = f[p1[:, 0], p1[:, 1], p1[:, 2]]
    s = np.clip(f0 / np.where(f0 == f1, 1e-12, f0 - f1), 0.0, 1.0)
    verts = (p0 + (p1 - p0) * s[:, None]) * np.asarray(spacing) + np.asarray(origin)
    tris = inverse.reshape(-1, 3).astype(np.int32)
    tris[:, [1, 2]] = tris[:, [2, 1]]  # face toward the outside (larger field)
    return verts, tris


def _clean(verts, tris, keep_tri):
    tris = tris[keep_tri]
    used = np.unique(tris)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris].astype(np.int32), used


def _largest_component(verts, tris, share=0.05):
    """Keep connected pieces holding at least ``share`` of the triangles (drops specks)."""
    if not len(tris):
        return verts, tris, np.arange(len(verts))
    from scipy import sparse
    from scipy.sparse import csgraph

    rows = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    cols = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(verts), len(verts)))
    _n, labels = csgraph.connected_components(graph, directed=False)
    tri_label = labels[tris[:, 0]]
    counts = np.bincount(tri_label)
    keep = counts[tri_label] >= share * len(tris)
    return _clean(verts, tris, keep)


def _cone(axis, degrees, ring=8):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    th = np.radians(degrees)
    dirs = [axis]
    for k in range(ring):
        phi = 2 * np.pi * k / ring
        dirs.append(np.cos(th) * axis + np.sin(th) * (np.cos(phi) * a + np.sin(phi) * b))
    return dirs


def _shadow(solid, dirs, h):
    """Cells that are solid or have solid somewhere along +dir, for every dir (blocked all ways)."""
    blocked_all = None
    for d in dirs:
        step = np.asarray(d) / np.asarray(h)  # one-cell march, in cells
        reach = solid.copy()
        length = 1.0
        while length < max(solid.shape) * 1.8:
            # reach |= reach shifted back along d by `length` cells
            off = -np.round(step / np.abs(step).max() * length).astype(int)
            reach |= _shift(reach, off)
            length *= 2
        blocked_all = reach if blocked_all is None else (blocked_all & reach)
    return blocked_all


def _shift(a, off):
    out = np.zeros_like(a)
    if any(abs(o) >= n for o, n in zip(off, a.shape)):
        return out
    src = tuple(slice(max(0, -o), a.shape[i] - max(0, o)) for i, o in enumerate(off))
    dst = tuple(slice(max(0, o), a.shape[i] - max(0, -o)) for i, o in enumerate(off))
    out[dst] = a[src]
    return out


# -- the envelope -------------------------------------------------------------------
class Envelope:
    """Distance field and lookups, built once per volume; layers rebuild in seconds."""

    def __init__(self, vol: Volume, face=None):
        """``face``: (points, outward dirs), volume mm: the auricle's front edge. The face's soft
        tissue in front of the ear (over the TMJ and the zygomatic arch) is then built too: its deep
        part (parotid, masseter, the joint) as a smooth fill under the envelope, its skin coming
        up to the auricle's front edge (``_face_fill``)."""
        t0 = time.time()
        self.vol = vol
        d = DOWNSAMPLE
        sx, sy, sz = (n // d for n in vol.shape)
        solid = vol.solid[: sx * d, : sy * d, : sz * d].reshape(sx, d, sy, d, sz, d).any(axis=(1, 3, 5))
        self.h = vol.mm * d
        # Pad the cube faces that look lateral; the others are where the skull is cut.
        normals = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)
        lat = vol.frame.lateral
        pad = [int(np.ceil(LATERAL_PAD_MM / self.h[i // 2])) if normals[i] @ lat > 0.25 else 1 for i in range(6)]
        self.pad_lo = np.array(pad[0::2])
        solid = np.pad(solid, [(pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])])
        self.shape = solid.shape
        # fine-voxel centre of coarse cell g is index d*g + (d-1)/2
        self.origin = (-self.pad_lo * d + (d - 1) / 2.0) * vol.mm
        self.fill = np.zeros(self.shape, bool)  # the face's deep tissue, counted as the base the layers lie on
        self.face_cols = np.zeros(self.shape, bool)
        if face is not None and len(face[0]):
            seed_ = np.pad(vol.cortex[: sx * d, : sy * d, : sz * d].reshape(sx, d, sy, d, sz, d).any(axis=(1, 3, 5)),
                           [(pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])])
            self.fill, self.face_cols = self._face_fill(solid, seed_, face)
            solid = solid | self.fill
        # Exterior: air that sees out laterally (within a cone around the canal axis)
        # without crossing solid. Plain flood fill would also reach the medial side:
        # the cube holds a cropped temporal bone with air all around it.
        self.exterior = ~solid & ~_shadow(solid, _cone(lat, VIEW_CONE_DEG), self.h)
        dist, idx = ndimage.distance_transform_edt(~solid, sampling=self.h, return_indices=True)
        self.dist = dist.astype(np.float32)
        self.near = idx.astype(np.int32)  # nearest solid cell, padded coarse index
        near_g = self.near - self.pad_lo[:, None, None, None]
        # Nearest bone on a face where the cube cuts the skull: no tissue there.
        core = np.array([sx, sy, sz])
        cut = np.zeros(self.shape, bool)
        for ax in range(3):
            for side in (0, 1):
                if pad[2 * ax + side] > 1:
                    continue
                edge = 0 if side == 0 else core[ax] - 1
                cut |= np.abs(near_g[ax] - edge) <= 1
        self.cut = cut
        self.near_mm = np.moveaxis(near_g, 0, -1) * (vol.mm * d) + (d - 1) / 2.0 * vol.mm
        self.near_uv = vol.frame.uv(self.near_mm).astype(np.float32)
        # outward bone normal (from smoothed occupancy)
        sm = ndimage.gaussian_filter(solid.astype(np.float32), 1.5)
        self.bone_grad = np.stack(np.gradient(sm, *self.h), axis=-1)
        # lateral bony surface: grown from the FIVRS cortex mask
        P = [(pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])]
        down = lambda m: np.pad(m[: sx * d, : sy * d, : sz * d].reshape(sx, d, sy, d, sz, d).any(axis=(1, 3, 5)), P)  # noqa: E731
        seed = down(vol.cortex)
        # the FIVRS bone representation: every solid label (condyle included), closed at crop plates
        bone = down(vol.solid & ~vol.dura)
        # exposed dura: surface that is mostly dura over ~1 mm, not the speckled bone/dura plate
        fine_surface = vol.solid & ndimage.binary_dilation(~vol.solid, structure=ndimage.generate_binary_structure(3, 1))
        def count(m):  # noqa: E306
            return np.pad(m[: sx * d, : sy * d, : sz * d].reshape(sx, d, sy, d, sz, d).sum(axis=(1, 3, 5)).astype(np.float32), P)
        sig = DURA_RADIUS_MM / float(self.h.mean())
        dura_n = ndimage.gaussian_filter(count(fine_surface & vol.dura), sig)
        surf_n = ndimage.gaussian_filter(count(fine_surface), sig)
        other = down(vol.dura) & (dura_n > DURA_SHARE * np.maximum(surf_n, 1e-6))
        self.exposed_dura = other
        clear = int(np.ceil(LABEL_CLEAR_MM / self.h.min()))
        near_other = ndimage.binary_dilation(other, iterations=clear) if clear else other
        outward = -self.bone_grad
        cosf = (outward @ lat) / np.maximum(np.linalg.norm(outward, axis=-1), 1e-9)
        surface_cells = bone & ndimage.binary_dilation(self.exterior, iterations=1)
        cand = surface_cells & (cosf > np.cos(np.radians(FACING_DEG))) & ~near_other
        steps = int(np.ceil(GROW_MM / self.h.min()))
        region = ndimage.binary_dilation(seed, structure=np.ones((3, 3, 3)), iterations=steps, mask=cand | seed) & ~near_other
        k = int(np.ceil(GAP_CLOSE_MM / self.h.min()))
        region = ndimage.binary_closing(region, structure=np.ones((3, 3, 3)), iterations=k) & surface_cells & ~near_other
        region = ndimage.binary_dilation(region, iterations=1) & solid  # a little into the bone for lookups
        if self.face_cols.any():  # the face's tissue is covered: its fill and the bone under it
            region |= solid & self.face_cols
            self.cut |= ndimage.binary_dilation(self.cut, iterations=3) & self.face_cols  # and stops clear of the crop's cut faces
        self.face_near = ndimage.binary_dilation(self.face_cols, iterations=3) if self.face_cols.any() else self.face_cols
        self.region = region
        self.on_cortex = region[self.near[0], self.near[1], self.near[2]]
        # bone (not another label) that faces lateral: what the temporalis may cover outside the region
        self.bone_lat = bone & surface_cells & ~near_other & (cosf > np.cos(np.radians(MUSCLE_FACING_DEG)))
        self.bone_lat = ndimage.binary_dilation(self.bone_lat, iterations=1) & solid
        self.near_bone_lat = self.bone_lat[self.near[0], self.near[1], self.near[2]]
        self.seed_cells, self.region_cells = int(seed.sum()), int((region & surface_cells).sum())
        self._canal_hole()
        self.build_seconds = time.time() - t0

    def _face_fill(self, solid, seed, face, uv_mm=0.5):
        """The face's deep tissue in front of the ear: (fill, columns), coarse padded grids.

        On a grid across the side of the head (anterior, superior), the face's skin height (along
        lateral) is solved as a smooth (harmonic) surface: held at the auricle's front edge (just
        below it) and at the envelope's own height over the lateral cortex around, free in front of
        the edge (out to ``FACE_REACH_MM``). Under that skin, less the layers' nominal thickness,
        everything down to the bone is the deep tissue (the layers then lie over it, bridging the
        gap between the condyle and the arch as real tissue does)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve
        from scipy.spatial import cKDTree

        fr = self.vol.frame
        L, A, S = (np.asarray(x, float) for x in (fr.lateral, fr.anterior, fr.superior))
        pts, dirs = (np.asarray(x, float) for x in face)
        surf = solid & ndimage.binary_dilation(~solid, iterations=1)
        idx = np.argwhere(surf)
        P = self.origin + idx * self.h
        puv = np.stack([P @ A, P @ S], axis=1)
        euv = np.stack([pts @ A, pts @ S], axis=1)
        lo = euv.min(axis=0) - FACE_REACH_MM - 10.0
        n = np.ceil((euv.max(axis=0) + FACE_REACH_MM + 10.0 - lo) / uv_mm).astype(int) + 1
        cell = lambda q: np.round((q - lo) / uv_mm).astype(int)  # noqa: E731
        g = cell(puv)
        ok = np.all((g >= 0) & (g < n), axis=1)
        top_all = np.full(n, -np.inf)
        np.maximum.at(top_all, (g[ok, 0], g[ok, 1]), P[ok] @ L)
        is_seed = seed[idx[:, 0], idx[:, 1], idx[:, 2]] & ok
        top_cx = np.full(n, -np.inf)
        np.maximum.at(top_cx, (g[is_seed, 0], g[is_seed, 1]), P[is_seed] @ L)
        known = np.isfinite(top_cx)
        h = np.where(known, top_cx + FACE_TOTAL_MM, np.nan)  # the envelope's skin over the cortex
        # free: in front of the auricle's front edge, within reach, over bone (the gaps bridged)
        gu, gv = np.meshgrid(np.arange(n[0]), np.arange(n[1]), indexing="ij")
        cuv = lo + np.stack([gu, gv], axis=-1) * uv_mm
        tree = cKDTree(euv)
        d, k = tree.query(cuv.reshape(-1, 2))
        d, k = d.reshape(n), k.reshape(n)
        duv = np.stack([dirs @ A, dirs @ S], axis=1)
        duv /= np.maximum(np.linalg.norm(duv, axis=1, keepdims=True), 1e-9)
        ahead = np.einsum("ijk,ijk->ij", cuv - euv[k], duv[k]) > -FACE_UNDER_MM  # a little under the edge too (no wall at the junction)
        # and above the top of the front edge (over the zygomatic root), forward of the helix root:
        # the face's surface blends up into the temple instead of stopping there
        top = euv[np.argmax(euv[:, 1])]
        ahead |= (cuv[..., 1] > top[1]) & (cuv[..., 0] > top[0] - 2.0)
        over_bone = ndimage.binary_closing(np.isfinite(top_all), iterations=int(round(4.0 / uv_mm)))
        free = ahead & (d < FACE_REACH_MM) & over_bone  # held only by the edge and the envelope further out (no dip into the fossa)
        h[free] = np.nan
        # the edge holds the skin just below it
        ge = cell(euv)
        oke = np.all((ge >= 0) & (ge < n), axis=1)
        h[ge[oke, 0], ge[oke, 1]] = pts[oke] @ L - FACE_EDGE_DROP_MM
        free[ge[oke, 0], ge[oke, 1]] = False
        fixed = np.isfinite(h)
        # harmonic fill on the free cells (4-neighbours; fixed cells are boundary values, others are open)
        fid = -np.ones(n, int)
        fid[free] = np.arange(free.sum())
        rows, cols, vals = [], [], []
        rhs = np.zeros(free.sum())
        diag = np.zeros(free.sum())
        fi, fj = np.nonzero(free)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = fi + di, fj + dj
            inside = (ni >= 0) & (ni < n[0]) & (nj >= 0) & (nj < n[1])
            a = fid[fi[inside], fj[inside]]
            ni, nj = ni[inside], nj[inside]
            to_free = free[ni, nj]
            to_fixed = fixed[ni, nj] & ~to_free
            np.add.at(diag, a[to_free | to_fixed], 1.0)
            rows.append(a[to_free]); cols.append(fid[ni[to_free], nj[to_free]]); vals.append(-np.ones(to_free.sum()))
            np.add.at(rhs, a[to_fixed], h[ni[to_fixed], nj[to_fixed]])
        lap = coo_matrix((np.concatenate([np.concatenate(vals), diag + 1e-6]),
                          (np.concatenate([np.concatenate(rows), np.arange(len(diag))]),
                           np.concatenate([np.concatenate(cols), np.arange(len(diag))]))), shape=(len(diag), len(diag))).tocsc()
        # free cells with no fixed cell in reach stay unsolved (dropped)
        from scipy.sparse.csgraph import connected_components

        _nc, lab = connected_components(lap, directed=False)
        anchored = np.zeros(lab.max() + 1, bool)
        anchored[np.unique(lab[rhs != 0])] = True
        sol = spsolve(lap, rhs)
        skin = np.full(n, np.nan)
        good = anchored[lab]
        skin[fi[good], fj[good]] = sol[good]
        skin[ge[oke, 0], ge[oke, 1]] = h[ge[oke, 0], ge[oke, 1]]  # under the edge itself too (no trench along it)
        # the holding cells leave single-cell gaps between solved ones: close them
        gaps = np.isnan(skin) & ndimage.binary_closing(np.isfinite(skin), iterations=2)
        if gaps.any():
            filled = ndimage.generic_filter(np.nan_to_num(skin, nan=0.0), np.sum, size=3)
            cnt = ndimage.generic_filter(np.isfinite(skin).astype(float), np.sum, size=3)
            skin[gaps] = (filled / np.maximum(cnt, 1))[gaps]
        # smoothed (the grid's steps would show as ripples in the skin), the edge's own height kept
        ok_s = np.isfinite(skin)
        num = ndimage.gaussian_filter(np.where(ok_s, skin, 0.0), FACE_SMOOTH_MM / uv_mm)
        den = ndimage.gaussian_filter(ok_s.astype(float), FACE_SMOOTH_MM / uv_mm)
        sm = np.where(ok_s, num / np.maximum(den, 1e-6), np.nan)
        edge_cells = np.zeros(n, bool)
        edge_cells[ge[oke, 0], ge[oke, 1]] = True
        keep_edge = ndimage.binary_dilation(edge_cells, iterations=1)
        skin = np.where(keep_edge, skin, sm)
        # the layers are thinner near the auricle (``AuricleRamp``: no fat at its base), so the deep
        # tissue stands higher there and the skin still comes up to the edge
        near_total = sum(v_[0] for v_ in AURICLE_NEAR.values())
        ramp = AuricleRamp()
        sgrid = np.clip((d - ramp.base_near_mm) / max(ramp.base_far_mm - ramp.base_near_mm, 1e-6), 0.0, 1.0)
        total_here = near_total + (FACE_TOTAL_MM - near_total) * sgrid * sgrid * (3 - 2 * sgrid)
        hull = skin - total_here
        # the fill: every coarse cell in those columns from the hull down FACE_DEPTH_MM (bone stays bone)
        allidx = np.indices(self.shape).reshape(3, -1).T
        C = self.origin + allidx * self.h
        gc = cell(np.stack([C @ A, C @ S], axis=1))
        okc = np.all((gc >= 0) & (gc < n), axis=1)
        hc = np.full(len(C), np.nan)
        hc[okc] = hull[gc[okc, 0], gc[okc, 1]]
        lc = C @ L
        col = np.isfinite(hc)
        fill = col & (lc < hc) & (lc > hc - FACE_DEPTH_MM)
        return fill.reshape(self.shape) & ~solid, col.reshape(self.shape)

    def _canal_hole(self):
        """uv-like mask across the canal axis: the EAC's cross-section, grown a little."""
        f = self.vol.frame
        e1 = f.anterior
        e2 = f.superior
        pts = np.argwhere(self.vol.eac) * self.vol.mm - f.canal_center
        t = pts @ f.lateral
        lateral_half = pts[t > np.percentile(t, 50)]  # cartilaginous half: the opening's shape
        q = np.stack([lateral_half @ e1, lateral_half @ e2], axis=1)
        self.canal_lo = q.min(axis=0) - 2.0
        n = np.ceil((q.max(axis=0) + 2.0 - self.canal_lo) / 0.25).astype(int) + 1
        grid = np.zeros(n, bool)
        g = np.round((q - self.canal_lo) / 0.25).astype(int)
        grid[g[:, 0], g[:, 1]] = True
        grid = ndimage.binary_closing(grid, iterations=2)
        grid = ndimage.binary_fill_holes(grid)
        self.canal_grid = ndimage.binary_dilation(grid, iterations=int(round(CANAL_MARGIN_MM / 0.25)))
        self.canal_medial_t = np.percentile(t, 2)

    def in_canal(self, p_mm):
        f = self.vol.frame
        r = np.asarray(p_mm) - f.canal_center
        q = np.stack([r @ f.anterior, r @ f.superior], axis=1)
        g = np.round((q - self.canal_lo) / 0.25).astype(int)
        ok = np.all((g >= 0) & (g < np.array(self.canal_grid.shape)), axis=1)
        hit = np.zeros(len(r), bool)
        hit[ok] = self.canal_grid[g[ok, 0], g[ok, 1]]
        return hit & (r @ f.lateral > self.canal_medial_t)

    # -- thickness maps -------------------------------------------------------------
    def muscle_map(self, muscle: Muscle):
        """Muscle and fascia thickness (mm) on the uv grid, and the grid's origin."""
        lo = self.near_uv.reshape(-1, 2).min(axis=0) - 1.0
        hi = self.near_uv.reshape(-1, 2).max(axis=0) + 1.0
        n = np.ceil((hi - lo) / UV_MM).astype(int) + 1
        if muscle.mesh_ply:
            return self._mesh_muscle_map(muscle.mesh_ply, lo, n)
        if len(muscle.outline_mm) < 3:
            z = np.zeros(n, np.float32)
            return z, z, lo
        from PIL import Image, ImageDraw

        poly = (self.vol.frame.uv(np.asarray(muscle.outline_mm)) - lo) / UV_MM
        img = Image.new("L", (int(n[0]), int(n[1])), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in poly], fill=1)
        inside = np.asarray(img, dtype=bool).T
        depth = ndimage.distance_transform_edt(inside) * UV_MM  # distance from the outline, inside
        s = np.clip(depth / max(muscle.taper_mm, 1e-6), 0.0, 1.0)
        muscle_t = muscle.max_mm * s * s * (3 - 2 * s)
        f = np.clip(depth / FASCIA_TAPER_MM, 0.0, 1.0)
        fascia_t = dict((n_, t) for n_, t, _ in LAYERS)["temporalis_fascia"] * f * f * (3 - 2 * f)
        return muscle_t.astype(np.float32), fascia_t.astype(np.float32), lo

    def _mesh_muscle_map(self, path, lo, n):
        """Muscle thickness from a mesh: rays along the canal axis through every uv cell, its first
        to last hit; the fascia over it tapers in from where it is thicker than 0.3 mm."""
        import open3d as o3d
        import trimesh

        m = trimesh.load(path, process=False)
        f = self.vol.frame
        A, S, L = (np.asarray(x, float) for x in (f.anterior, f.superior, f.lateral))
        ii, jj = np.mgrid[0:n[0], 0:n[1]]
        uv = lo + np.stack([ii, jj], axis=-1).reshape(-1, 2) * UV_MM
        top = float((np.asarray(m.vertices) @ L).max()) + 5.0
        span = top - float((np.asarray(m.vertices) @ L).min()) + 10.0
        o = uv[:, :1] * A + uv[:, 1:] * S + top * L

        def first_hit(v, t, origins, d):
            # Open3D's ray caster (Embree): trimesh's pure-numpy one runs out of memory on the CT bone
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.core.Tensor(np.asarray(v, np.float32)), o3d.core.Tensor(np.asarray(t, np.uint32)))
            rays = np.concatenate([origins, np.repeat(d[None], len(origins), axis=0)], axis=1).astype(np.float32)
            return sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(float)

        # the muscle's outer face (first hit from outside) and inner face (first hit from the far side)
        first = first_hit(m.vertices, m.faces, o, -L)
        back = first_hit(m.vertices, m.faces, o - span * L, L)
        last = span - back
        t = np.where(np.isfinite(first) & np.isfinite(back), np.maximum(last - first, 0.0), 0.0)
        # only where the muscle is the outermost thing over the bone: under the zygomatic arch it runs
        # deep to bone, and covering that bone (facing every way) breaks the layers into holes
        bv, bt = self.vol.bone_mesh()
        bone_first = first_hit(bv, bt, o, -L)
        t[bone_first < first - 0.5] = 0.0
        t = t.reshape(n[0], n[1])
        # smooth: the atlas mesh's long sliver triangles make the ray thickness jagged, and every layer
        # above is traced where the summed thickness crosses a level (a jagged sum leaves holes)
        t = ndimage.gaussian_filter(t, MESH_MUSCLE_SMOOTH_MM / UV_MM)
        inside = t > 0.3
        edge = ndimage.distance_transform_edt(inside) * UV_MM
        e = np.clip(edge / MESH_MUSCLE_TAPER_MM, 0.0, 1.0)
        t = t * e * e * (3 - 2 * e)  # tapers in from its outline, as the picked outline does
        d = ndimage.distance_transform_edt(inside) * UV_MM
        fl = np.clip(d / FASCIA_TAPER_MM, 0.0, 1.0)
        fascia_t = dict((n_, x) for n_, x, _ in LAYERS)["temporalis_fascia"] * fl * fl * (3 - 2 * fl)
        return t.astype(np.float32), fascia_t.astype(np.float32), lo

    def thickness_fields(self, muscle: Muscle, thickness=None):
        """Per-layer thickness (mm) at every grid cell, looked up at its nearest bone."""
        nominal = {n: t for n, t, _ in LAYERS}
        nominal.update(thickness or {})
        muscle_t, fascia_t, lo = self.muscle_map(muscle)
        # bone the muscle covers, grown a little so tilted rays at its tapered edge still land on it
        self._muscle_uv = (ndimage.binary_dilation(muscle_t > 0.05, iterations=int(MUSCLE_EDGE_MM / UV_MM)), lo)
        g = np.round((self.near_uv - lo) / UV_MM).astype(np.int32)
        g[..., 0] = np.clip(g[..., 0], 0, muscle_t.shape[0] - 1)
        g[..., 1] = np.clip(g[..., 1], 0, muscle_t.shape[1] - 1)
        out = {}
        for name in LAYER_NAMES:
            if name == "temporalis_muscle":
                out[name] = muscle_t[g[..., 0], g[..., 1]]
            elif name == "temporalis_fascia":
                out[name] = fascia_t[g[..., 0], g[..., 1]] * (nominal[name] / dict((n, t) for n, t, _ in LAYERS)[name])
            elif name == "subcutaneous_fat":
                out[name] = (nominal[name] * self.fat_taper()).astype(np.float32)
            elif name == "auricular_muscles":  # a uv map set by the build (auricular_muscles.py), else none
                am = getattr(self, "auricular_map", None)
                if am is None:
                    out[name] = np.float32(0.0)
                else:
                    m_, lo_ = am
                    ga = np.round((self.near_uv - lo_) / UV_MM).astype(np.int32)
                    inside = (ga[..., 0] >= 0) & (ga[..., 0] < m_.shape[0]) & (ga[..., 1] >= 0) & (ga[..., 1] < m_.shape[1])
                    ga[..., 0] = np.clip(ga[..., 0], 0, m_.shape[0] - 1)
                    ga[..., 1] = np.clip(ga[..., 1], 0, m_.shape[1] - 1)
                    out[name] = np.where(inside, m_[ga[..., 0], ga[..., 1]], 0.0).astype(np.float32)
            else:
                out[name] = np.float32(nominal[name])
        ramp = getattr(self, "auricle_ramp", None)
        if ramp is not None:
            d = self.base_distance(ramp.base_outline) if ramp.base_outline is not None else self.canal_distance()
            w = ramp.weight(d)
            for name, near in ramp.near.items():
                if name in out:
                    out[name] = (near + (out[name] - near) * w).astype(np.float32)
        return out

    def canal_distance(self):
        """Distance (mm) from each cell's nearest bone point, across the canal axis, to the outline
        of the canal's lateral opening (0 inside it)."""
        if getattr(self, "_canal_d", None) is not None:
            return self._canal_d
        f = self.vol.frame
        out_d = ndimage.distance_transform_edt(~self.canal_grid) * 0.25  # mm outside the opening's outline
        r = self.near_mm - f.canal_center
        q = np.stack([r @ f.anterior, r @ f.superior], axis=-1)
        g = ((q - self.canal_lo) / 0.25).reshape(-1, 2).T
        d = ndimage.map_coordinates(out_d.astype(np.float32), g, order=1, mode="nearest").reshape(self.shape)
        # beyond the map's edge the distance keeps growing
        span = np.array(self.canal_grid.shape) * 0.25
        extra = np.linalg.norm(np.maximum(np.maximum(-(q - self.canal_lo), (q - self.canal_lo) - span), 0.0), axis=-1)
        self._canal_d = (d + extra).astype(np.float32)
        return self._canal_d

    def base_distance(self, outline, res=0.25):
        """Distance (mm) from each cell's nearest bone point, seen from lateral, to the outline of
        the auricle's base (0 inside it: under the auricle)."""
        from PIL import Image, ImageDraw

        key = hash(np.asarray(outline, np.float32).tobytes())
        if getattr(self, "_base_d", None) is not None and self._base_d[0] == key:
            return self._base_d[1]
        uv_o = self.vol.frame.uv(np.asarray(outline, float))
        uv_c = self.near_uv.reshape(-1, 2)
        lo = np.minimum(uv_o.min(axis=0), uv_c.min(axis=0)) - 2.0
        hi = np.maximum(uv_o.max(axis=0), uv_c.max(axis=0)) + 2.0
        n = np.ceil((hi - lo) / res).astype(int) + 1
        img = Image.new("L", (int(n[0]), int(n[1])), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in (uv_o - lo) / res], fill=1)
        inside = np.asarray(img, bool).T
        out_d = (ndimage.distance_transform_edt(~inside) * res).astype(np.float32)
        d = ndimage.map_coordinates(out_d, ((uv_c - lo) / res).T, order=1, mode="nearest").reshape(self.shape)
        self._base_d = (key, d.astype(np.float32))
        return self._base_d[1]

    def fat_taper(self, free_mm=None):
        """0 at the EAC's lateral opening rising smoothly to 1 at ``FAT_FREE_MM`` from it.

        Distance is measured from each cell's nearest bone point, across the canal axis, to the
        outline of the canal opening: the thin skin around the meatus has no fat, which makes
        the deep postauricular crease.
        """
        free = FAT_FREE_MM if free_mm is None else free_mm
        s = np.clip(self.canal_distance() / max(free, 1e-6), 0.0, 1.0)
        return (s * s * (3 - 2 * s)).astype(np.float32)

    def _under_muscle(self, p_mm):
        tm, lo = getattr(self, "_muscle_uv", (None, None))
        if tm is None:
            return np.zeros(len(p_mm), bool)
        g = np.round((self.vol.frame.uv(p_mm) - lo) / UV_MM).astype(int)
        ok = np.all((g >= 0) & (g < np.array(tm.shape)), axis=1)
        out = np.zeros(len(p_mm), bool)
        out[ok] = tm[g[ok, 0], g[ok, 1]]
        return out

    def lies_on_cortex(self, v, tr, reach_mm, step_mm=0.15):
        """Looking straight down the lateral axis from each point, the first solid is covered bone.

        Covered bone is the lateral region, or any bone under the muscle outline.
        Never the sinus/dura label. A point with no solid under it is
        kept only under the muscle (it fills the temporal fossa and the space by
        the zygomatic root).
        """
        vol = self.vol
        lat = vol.frame.lateral
        shape = np.array(vol.shape)
        result = np.zeros(len(v), bool)
        done = np.zeros(len(v), bool)
        for d in np.arange(0.0, reach_mm + 15.0, step_mm):
            p = v - lat * d
            i = np.round(p / vol.mm).astype(int)
            ok = np.all((i >= 0) & (i < shape), axis=1) & ~done
            if not ok.any():
                continue
            hit = np.zeros(len(v), bool)
            hit[ok] = vol.solid[i[ok, 0], i[ok, 1], i[ok, 2]]
            if self.fill.any():  # the face's deep tissue counts as covered
                gq = np.clip(np.round((p[ok] - self.origin) / self.h).astype(int), 0, np.array(self.shape) - 1)
                in_fill = np.zeros(len(v), bool)
                in_fill[ok] = self.fill[gq[:, 0], gq[:, 1], gq[:, 2]]
                result[in_fill & ~hit] = True
                done |= in_fill & ~hit
            if hit.any():
                g = np.clip(np.round((p[hit] - self.origin) / self.h).astype(int), 0, np.array(self.shape) - 1)
                is_bone = ~vol.dura[i[hit, 0], i[hit, 1], i[hit, 2]]
                result[hit] = is_bone & (self.region[g[:, 0], g[:, 1], g[:, 2]] | self._under_muscle(p[hit]))
            done |= hit
        result[~done] = self._under_muscle(v[~done])
        return result

    # -- surfaces -------------------------------------------------------------------
    def layers(self, muscle: Muscle | None = None, thickness=None, smooth=3):
        """Outer surface of each layer: {name: (verts_mm, tris)}. Layers of zero thickness share a surface."""
        muscle = muscle or Muscle()
        t_fields = self.thickness_fields(muscle, thickness)
        total = np.zeros(self.shape, np.float32)
        out = {}
        for name in LAYER_NAMES:
            t = t_fields[name]
            if not np.any(np.asarray(t) > 0.05):
                continue
            total = total + t
            if not np.any(total > 0):
                continue
            field = self.dist - total
            field[~self.exterior] = -1.0  # bone and closed cavities are inside
            # crop to the band around the surface
            band = np.abs(field) < 3 * self.h.max()
            if not band.any():
                continue
            nz = np.nonzero(band)
            lo = np.maximum([a.min() - 1 for a in nz], 0)
            hi = np.minimum([a.max() + 2 for a in nz], self.shape)
            sub = tuple(slice(a, b) for a, b in zip(lo, hi))
            v, tr = marching_cubes_field(field[sub], self.h, self.origin + lo * self.h)
            if not len(tr):
                continue
            # drop: over cut bone, in the canal, where this layer has (almost) no thickness
            gi = np.clip(np.round((v - self.origin) / self.h).astype(int), 0, np.array(self.shape) - 1)
            covered = self.on_cortex[gi[:, 0], gi[:, 1], gi[:, 2]] | (
                self.near_bone_lat[gi[:, 0], gi[:, 1], gi[:, 2]] & self._under_muscle(self.near_mm[gi[:, 0], gi[:, 1], gi[:, 2]]))
            q = self.near[:, gi[:, 0], gi[:, 1], gi[:, 2]].T  # nearest solid cell
            # over the face's deep tissue: covered, and its fill's sides and steps are not bone edges
            face_ok = self.face_cols[gi[:, 0], gi[:, 1], gi[:, 2]] | self.face_cols[q[:, 0], q[:, 1], q[:, 2]]
            off0 = v - self.near_mm[gi[:, 0], gi[:, 1], gi[:, 2]]
            on_top = off0 @ np.asarray(self.vol.frame.lateral, float) > 0.5 * np.linalg.norm(off0, axis=1)
            face_ok &= on_top  # on top of it, not down its sides at the patch's edge
            covered |= face_ok
            bad_v = self.cut[gi[:, 0], gi[:, 1], gi[:, 2]] | self.in_canal(v) | ~covered
            nb = -self.bone_grad[q[:, 0], q[:, 1], q[:, 2]]
            off = v - self.near_mm[gi[:, 0], gi[:, 1], gi[:, 2]]
            cosang = np.einsum("ij,ij->i", off, nb) / np.maximum(np.linalg.norm(off, axis=1) * np.linalg.norm(nb, axis=1), 1e-9)
            bad_v |= (cosang < EDGE_COS) & ~face_ok
            # no wrapping round the back of the bone (the crop's edges, behind the sigmoid): the
            # bone under the layer must face out to the side
            lat = np.asarray(self.vol.frame.lateral, float)
            bad_v |= ((nb @ lat) < FACE_COS * np.maximum(np.linalg.norm(nb, axis=1), 1e-9)) & ~face_ok
            # the -1 fill behind the shadow line makes a false surface there; the true one sits at dist = total
            true_d = ndimage.map_coordinates(self.dist, ((v - self.origin) / self.h).T, order=1)
            want = ndimage.map_coordinates(total if np.ndim(total) else np.full(1, total), ((v - self.origin) / self.h).T, order=1) if np.ndim(total) else float(total)
            bad_v |= np.abs(true_d - want) > LEVEL_TOL * float(self.h.max())
            if np.ndim(t):
                bad_v |= t[gi[:, 0], gi[:, 1], gi[:, 2]] < 0.05
            keep = ~bad_v[tr].any(axis=1)
            if self.face_cols.any():  # the face's tissue: only where it faces out, not down the fill's sides at the crop
                fn = np.cross(v[tr[:, 1]] - v[tr[:, 0]], v[tr[:, 2]] - v[tr[:, 0]])
                facing = fn @ lat > 0.3 * np.linalg.norm(fn, axis=1)
                in_face = self.face_near[gi[:, 0], gi[:, 1], gi[:, 2]][tr].any(axis=1)
                keep &= ~in_face | facing
            v, tr, _ = _clean(v, tr, keep)
            v, tr, _ = _largest_component(v, tr)
            if not len(tr):
                continue
            if smooth and len(tr):
                v = surface.smooth_surface(v, tr, iterations=smooth, lam=0.5, max_step=0.5 * float(self.h.max()))
                if self.face_cols.any():  # over the face's deep tissue (a voxel fill): smoothed more, no ripples
                    gi2 = np.clip(np.round((v - self.origin) / self.h).astype(int), 0, np.array(self.shape) - 1)
                    on_face = self.face_near[gi2[:, 0], gi2[:, 1], gi2[:, 2]]
                    if on_face.any():
                        from scipy.sparse import coo_matrix

                        e = np.concatenate([tr[:, [0, 1]], tr[:, [1, 2]], tr[:, [2, 0]]])
                        e = np.concatenate([e, e[:, ::-1]])
                        adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v))).tocsr()
                        deg = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1)[:, None]
                        for _ in range(FACE_LAYER_SMOOTH):
                            mean = (adj @ v) / deg
                            v[on_face] += 0.5 * (mean[on_face] - v[on_face])
                v, tr, _ = _clean(v, tr, ~self.in_canal(v)[tr].any(axis=1))  # smoothing can creep into the canal
            # the layer must lie over covered bone, never over dura or sinus
            on = self.lies_on_cortex(v, tr, float(np.max(total)) + 3.0)
            v, tr, _ = _clean(v, tr, on[tr].all(axis=1))
            v, tr, _ = _largest_component(v, tr)
            if not len(tr):
                continue
            out[name] = (v, tr)
        if self.face_cols.any() and "skin" in out:  # over the face: no inner layer pokes through the skin
            import open3d as o3d

            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.core.Tensor(np.asarray(out["skin"][0], np.float32)), o3d.core.Tensor(np.asarray(out["skin"][1], np.uint32)))
            lat = np.asarray(self.vol.frame.lateral, float)
            for name, (v, tr) in list(out.items()):
                if name == "skin":
                    continue
                gi2 = np.clip(np.round((v - self.origin) / self.h).astype(int), 0, np.array(self.shape) - 1)
                on_face = self.face_near[gi2[:, 0], gi2[:, 1], gi2[:, 2]]
                if not on_face.any():
                    continue
                q = v[on_face].astype(np.float32)
                down = sc.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(-lat, q.shape).astype(np.float32)], axis=1)))["t_hit"].numpy()
                above = np.isfinite(down) & (down < 3.0)
                if above.any():
                    v = v.copy()
                    idx = np.nonzero(on_face)[0][above]
                    v[idx] -= (down[above] + 0.2)[:, None] * lat
                    out[name] = (v, tr)
        return out


    def _covered_cells(self, surfaces):
        """Air cells the envelope fills: nearest bone under the kept periosteum, not in the canal."""
        foot = np.zeros(self.shape, bool)
        if "periosteum" in surfaces:
            pv_ = surfaces["periosteum"][0]
            gi = np.clip(np.round((pv_ - self.origin) / self.h).astype(int), 0, np.array(self.shape) - 1)
            q = self.near[:, gi[:, 0], gi[:, 1], gi[:, 2]]
            foot[q[0], q[1], q[2]] = True
            foot = ndimage.binary_dilation(foot, iterations=1)
        ok = foot[self.near[0], self.near[1], self.near[2]] & self.exterior & ~self.cut
        centres = self.origin + np.indices(self.shape).reshape(3, -1).T * self.h
        return ok & ~self.in_canal(centres).reshape(self.shape)

    # -- solids, for seeing the stack -------------------------------------------------
    def extruded_solid(self, v, t, thickness, box=None):
        """A layer as a closed slab from its outer surface (``v``, ``t``): the surface, a copy pushed
        in by the layer's own thickness (``thickness``: its field on the grid) along the bone
        distance field's gradient (the direction the layers were stacked in, so the inner face lands
        on the layer beneath), and walls round its edge. Cut exactly at ``box`` = (lo, hi) first, so
        the walls there are the block's flat cut faces. Thin layers stay whole (the grid's tracing
        breaks a 1-2 cell slab into fragments)."""
        import auricle as au

        v, t = np.asarray(v, float), np.asarray(t)
        if box is not None:
            lo, hi = (np.asarray(x, float) for x in box)
            v, t = au.clip_mesh(v, t, np.max(np.concatenate([lo - v, v - hi], axis=1), axis=1))
            v, t = np.asarray(v, float), np.asarray(t)
        if not len(t):
            return v, t
        if getattr(self, "_dist_grad", None) is None:
            self._dist_grad = [g.astype(np.float32) for g in np.gradient(self.dist.astype(np.float32), *self.h)]
        g = ((v - self.origin) / self.h).T
        th = ndimage.map_coordinates(np.broadcast_to(np.asarray(thickness, np.float32), self.shape), g, order=1, mode="nearest")
        d = np.stack([ndimage.map_coordinates(c, g, order=1, mode="nearest") for c in self._dist_grad], axis=1)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        inner = v - np.maximum(th, 0.0)[:, None] * d
        n = len(v)
        # walls along the open edges (edges used by one triangle), wound to face out
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        rim = e[cnt[inv.ravel()] == 1]
        walls = np.concatenate([np.stack([rim[:, 1], rim[:, 0], rim[:, 0] + n], 1), np.stack([rim[:, 1], rim[:, 0] + n, rim[:, 1] + n], 1)])
        return np.concatenate([v, inner]), np.concatenate([t, t[:, ::-1] + n, walls]).astype(np.int64)

    def fill_solid(self, box=None, smooth=2):
        """The face's deep tissue in front of the ear (``_face_fill``: parotid, masseter, the joint),
        the base the layers lie on there, as a closed solid cut at ``box``; (verts, tris) or None."""
        if not self.fill.any():
            return None
        field = 0.5 - ndimage.gaussian_filter(self.fill.astype(np.float32), 0.7)  # (inside bone it is hidden)
        if box is not None:
            centres = self.origin + np.indices(self.shape).reshape(3, -1).T * self.h
            blo, bhi = (np.asarray(x, float) for x in box)
            field = np.maximum(field, np.max(np.concatenate([blo - centres, centres - bhi], axis=1), axis=1).reshape(self.shape))
        v, t = marching_cubes_field(field.astype(np.float32), self.h, self.origin)
        if not len(t):
            return None
        return surface.smooth_surface(v, t, iterations=smooth, max_step=0.5 * float(self.h.max())), t

    def scene_box(self, skin_v, inset_mm=1.5):
        """The scene's box (lo, hi), volume mm: the CT cube inset ``inset_mm`` on the faces where the
        skull is cut; on its lateral side(s) it reaches out to the skin (``skin_v``) so tissue there
        is not cut."""
        bone_v, _t = self.vol.bone_mesh()
        lo, hi = bone_v.min(0) + inset_mm, bone_v.max(0) - inset_mm
        L = np.asarray(self.vol.frame.lateral, float)
        for ax in range(3):
            if L[ax] > 0.25:
                hi[ax] = max(hi[ax], float(np.asarray(skin_v)[:, ax].max()) + 1.0)
            if L[ax] < -0.25:
                lo[ax] = min(lo[ax], float(np.asarray(skin_v)[:, ax].min()) - 1.0)
        return lo, hi

    def layer_solids(self, muscle: Muscle | None = None, thickness=None, cut=None, smooth=2, box=None,
                     surfaces=None):
        """Each layer as a closed solid between its inner and outer surface, plus the bone.

        ``cut`` = (point, normal): everything on the normal's side of that plane is removed and
        the cut faces are closed, so the stack shows in section. ``box`` = (lo, hi) volume mm:
        everything outside it is removed the same way (the scene's cutaway block, ``scene_box``).
        Each solid's inner face is the one below's outer face (no gaps); thin layers are drawn at
        least MIN_SOLID_MM thick, and the skin's outer face stays the true one. The footprint is where the
        periosteum surface of ``layers`` lies (same coverage rules).
        Returns {name: (verts, tris)} with "bone" first.
        """
        muscle = muscle or Muscle()
        surfaces = surfaces if surfaces is not None else self.layers(muscle, thickness)
        t_fields = self.thickness_fields(muscle, thickness)
        ok = self._covered_cells(surfaces)
        centres = self.origin + np.indices(self.shape).reshape(3, -1).T * self.h
        plane = None
        if cut is not None:
            p0, nrm = (np.asarray(x, float) for x in cut)
            nrm = nrm / np.linalg.norm(nrm)
            plane = ((centres - p0) @ nrm).reshape(self.shape).astype(np.float32)
        if box is not None:  # outside the box: > 0 (the max of the six plane distances)
            blo, bhi = (np.asarray(x, float) for x in box)
            bsd = np.max(np.concatenate([blo - centres, centres - bhi], axis=1), axis=1).reshape(self.shape).astype(np.float32)
            plane = bsd if plane is None else np.maximum(plane, bsd)
        out = {}
        # bone: smoothed occupancy, closed at the cut
        solid = self.dist <= 0
        bone = 0.5 - ndimage.gaussian_filter(solid.astype(np.float32), 0.7)
        if plane is not None:
            bone = np.maximum(bone, plane)
        v, t = marching_cubes_field(bone, self.h, self.origin)
        out["bone"] = (surface.smooth_surface(v, t, iterations=smooth, max_step=0.5 * float(self.h.max())) if len(t) else v, t)
        inner = np.zeros(self.shape, np.float32)
        true_total = np.zeros(self.shape, np.float32)
        for name in LAYER_NAMES:
            true_total = true_total + np.asarray(t_fields[name], np.float32)
        for name in LAYER_NAMES:
            t_ = t_fields[name]
            outer = inner + t_
            if not np.any(np.asarray(t_) > 0.05):
                inner = outer
                continue
            # a layer thinner than the grid can't be meshed: draw it at least MIN_SOLID_MM thick and
            # stack the next layer on what was drawn (display only; ``layers`` keeps true depths)
            t_arr = np.asarray(t_, np.float32)
            shown = inner + np.where(t_arr > 0.05, np.maximum(t_arr, MIN_SOLID_MM), 0.0)
            if name == LAYER_NAMES[-1]:  # the skin ends on the true outer surface (it takes up the excess)
                shown = np.maximum(true_total, inner + 0.5 * MIN_SOLID_MM)
            field = np.maximum(inner - self.dist, self.dist - shown)
            field[~ok] = np.maximum(field[~ok], 0.5)
            if plane is not None:
                field = np.maximum(field, plane)
            band = field < 2 * self.h.max()
            if band.any():
                nz = np.nonzero(band)
                lo = np.maximum([a.min() - 1 for a in nz], 0)
                hi = np.minimum([a.max() + 2 for a in nz], self.shape)
                sub = tuple(slice(a, b) for a, b in zip(lo, hi))
                v, t = marching_cubes_field(field[sub], self.h, self.origin + lo * self.h)
                if len(t):
                    v, t, _ = _largest_component(v, t, share=0.02)
                    if smooth:
                        v = surface.smooth_surface(v, t, iterations=smooth, lam=0.5, max_step=0.3 * float(self.h.max()))
                    out[name] = (v, t)
            inner = shown
        return out


# -- saving -------------------------------------------------------------------------
def write_ply(path, verts, tris, rgb=None):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(verts, np.float32)
    tris = np.asarray(tris, np.int32)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(verts)}",
            "property float x", "property float y", "property float z"]
    if rgb is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head += [f"element face {len(tris)}", "property list uchar int vertex_indices", "end_header"]
    vdt = [("p", "<f4", 3)] + ([("c", "u1", 3)] if rgb is not None else [])
    va = np.zeros(len(verts), dtype=vdt)
    va["p"] = verts
    if rgb is not None:
        va["c"] = np.broadcast_to(np.asarray(rgb, np.uint8), (len(verts), 3))
    fa = np.zeros(len(tris), dtype=[("n", "u1"), ("i", "<i4", 3)])
    fa["n"] = 3
    fa["i"] = tris
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode())
        fh.write(va.tobytes())
        fh.write(fa.tobytes())


def placement_path(adf, ear=None):
    """Saved placement for a volume; each ear model other than ear.obj keeps its own file."""
    stem = pathlib.Path(adf).stem
    ear_stem = pathlib.Path(ear).stem if ear else "ear"
    return OUT / (f"{stem}_placement.json" if ear_stem == "ear" else f"{stem}_{ear_stem}_placement.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adf", nargs="?", default=str(ROOT / "ADF" / "volume_256_RT143.yaml"))
    ap.add_argument("--placement", help="placement JSON (muscle outline); default: the saved one for this volume")
    args = ap.parse_args()
    vol = Volume(pathlib.Path(args.adf))
    env = Envelope(vol)
    muscle = Muscle()
    path = pathlib.Path(args.placement) if args.placement else placement_path(args.adf)
    if path.exists():
        doc = json.loads(path.read_text())
        muscle = Muscle(**doc.get("muscle", {}))
    print(f"envelope grid {env.shape}, {env.h.round(3)} mm, built in {env.build_seconds:.1f} s")
    t0 = time.time()
    layers = env.layers(muscle)
    print(f"layers in {time.time() - t0:.1f} s")
    stem = pathlib.Path(args.adf).stem
    for name, (v, t) in layers.items():
        write_ply(OUT / stem / f"{name}.ply", v, t, LAYER_RGB[name])
        print(f"  {name:18s} {len(t):7d} tris")


if __name__ == "__main__":
    main()
