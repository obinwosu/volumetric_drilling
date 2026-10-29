"""Lateral cortical masks for the FIVRS RTX Tuner.

Diagnostic overlays only. This module does not edit the volume, the drill,
or the default structure materials. The EAC segmentation is a closed solid.
The lateral direction is its long axis, signed toward the end whose surface
meets boundary-connected exterior air. The cap plane is a robust fit of that
end's face: surface voxels that touch exterior air and point along the canal.
The camera is never an input. A second plane is the outer tangent of the
EAC wall that faces the TMJ, held perpendicular to the cap plane. A third
plane turns that anterior normal 90° counter-clockwise about the cap normal
and fits the same kind of tangent on the EAC. Mask 3 drops cortex on the
TMJ side of the anterior plane and on the superior side of the new plane.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import yaml
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import rtx_materials
from surface import index_to_local, local_to_world

HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = HERE / "tuner_cortex_mask.json"
# Same millimetre scale the burr mesh uses in fivrs_rtx_view.
MM_TO_SIM = 0.01007

MASK_COLORS = {
    "mask1": (0.05, 0.82, 0.95),
    "mask2": (1.0, 0.48, 0.05),
    "mask3": (0.95, 0.12, 0.72),
}
PLANE_COLOR = (0.98, 0.92, 0.15)
ANTERIOR_COLOR = (0.40, 0.78, 1.0)
SUPERIOR_COLOR = (0.78, 0.62, 1.0)
CENTROID_COLOR = (0.95, 0.95, 0.95)
CAP_COLOR = (1.0, 0.25, 0.20)
VECTOR_COLOR = (0.30, 1.0, 0.45)
MASK_KEYS = ("mask1", "mask2", "mask3")
# Mask 1 is the lateral cortex surface. Mask 3 is the markings shader, lifted
# just enough to sit on that surface and fade out at its own perimeter.
APPEARANCE_LAYERS = (
    ("mask1", "lateral_cortex", 0.0),
    ("mask3", "cortex_markings", 0.05),
)

DEFAULT_PARAMETERS = {
    "seed_distance_mm": 6.0,
    "normal_angle_deg": 70.0,
    "plane_offset_mm": 8.0,
    "plane_depth_scale": 3.0,
    "cleanup_steps": 2,
    "crop_margin_voxels": 0,
    "opacity": 0.62,
    "selected_mask": "off",
    "show_plane": True,
    "show_anterior_plane": True,
    "show_superior_plane": True,
    "show_centroid": True,
    "show_cap": True,
    "show_vector": True,
}


def named_rgb(document, name: str, fallback_hex: str) -> tuple[int, int, int]:
    """Original segmentation RGB for a structure name such as bone or eac."""
    hex_id = ""
    if document is not None:
        hex_id = rtx_materials.hex_of(document, name) or ""
    if not rtx_materials.is_hex_id(hex_id):
        hex_id = fallback_hex
    return tuple(int(hex_id[offset : offset + 2], 16) for offset in (0, 2, 4))


def intact_slice_paths(folder: pathlib.Path, prefix: str, extension: str, count: int) -> list[pathlib.Path]:
    """One slice per index, including files padded wider than ``count``.

    Chai only tries zero-padding up to the digit width of ``count``. This
    specimen stores slice 10 as ``plane0010.png`` and slice 100 as
    ``plane00100.png``, so that walk stops at slice 9. The intact stack is
    every index that exists, preferring the shortest name when two paddings
    share an index.
    """
    matches = {}
    for path in folder.glob(f"{prefix}*.{extension}"):
        suffix = path.stem[len(prefix) :]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if index > int(count):
            continue
        current = matches.get(index)
        if current is None or len(path.stem) < len(current.stem):
            matches[index] = path
    return [matches[index] for index in sorted(matches)]


def load_intact_volume(anatomy_adf: pathlib.Path):
    """Occupied voxels, RGB labels, and the ADF dimensions. Rows match Chai."""
    from PIL import Image

    document = yaml.safe_load(anatomy_adf.read_text())
    volume = None
    for key, value in document.items():
        if isinstance(key, str) and key.startswith("VOLUME") and isinstance(value, dict):
            volume = value
            break
    if volume is None:
        raise RuntimeError(f"No VOLUME block in {anatomy_adf}")
    images = volume["images"]
    folder = (anatomy_adf.parent / images["path"]).resolve()
    paths = intact_slice_paths(folder, images["prefix"], images["format"], int(images["count"]))
    if not paths:
        raise RuntimeError(f"No slices in {folder}")
    first = np.asarray(Image.open(paths[0]).convert("RGBA"))
    height, width = first.shape[:2]
    occupied = np.zeros((width, height, len(paths)), dtype=bool)
    colors = np.zeros((width, height, len(paths), 3), dtype=np.uint8)
    for index, path in enumerate(paths):
        rgba = np.flipud(np.asarray(Image.open(path).convert("RGBA")))
        occupied[:, :, index] = rgba[:, :, 3].T > 10
        colors[:, :, index] = np.transpose(rgba[:, :, :3], (1, 0, 2))
    raw = volume.get("dimensions") or {}
    dimensions = np.array(
        [float(raw.get("x", 1.0)), float(raw.get("y", 1.0)), float(raw.get("z", 1.0))],
        dtype=np.float64,
    )
    laterality = str(volume.get("laterality") or "")
    return occupied, colors, dimensions, laterality


def _color_mask(colors: np.ndarray, occupied: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    return (
        occupied
        & (colors[:, :, :, 0] == rgb[0])
        & (colors[:, :, :, 1] == rgb[1])
        & (colors[:, :, :, 2] == rgb[2])
    )


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def exterior_air(occupied: np.ndarray) -> np.ndarray:
    """Empty voxels that flood-fill, face to face, to the volume boundary."""
    empty = ~np.asarray(occupied, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    seed = np.zeros(empty.shape, dtype=bool)
    seed[0] = empty[0]
    seed[-1] = empty[-1]
    seed[:, 0] |= empty[:, 0]
    seed[:, -1] |= empty[:, -1]
    seed[:, :, 0] |= empty[:, :, 0]
    seed[:, :, -1] |= empty[:, :, -1]
    return ndimage.binary_propagation(seed, structure=structure, mask=empty)


class LateralFrame:
    """EAC centroid, lateral cap, fitted plane, and lateral vector, in millimetres."""

    def __init__(self):
        self.ok = False
        self.summary = "EAC segmentation was not found."
        self.centroid_mm = np.zeros(3)
        self.cap_center_mm = np.zeros(3)
        self.cap_points_mm = np.zeros((0, 3))
        self.lateral_vector = np.array([1.0, 0.0, 0.0])
        self.long_axis = np.array([1.0, 0.0, 0.0])
        self.plane_point_mm = np.zeros(3)
        self.plane_normal = np.array([1.0, 0.0, 0.0])
        self.cap_radius_mm = 1.0
        self.length_mm = 0.0
        self.plane_angle_deg = 0.0
        self.scores = {"min": 0, "max": 0}
        self.lateral_end = ""
        self.closed = False
        self.components = 0
        self.mm_per_axis = np.ones(3)
        self.dimensions = np.ones(3)
        self.shape = (1, 1, 1)
        self.laterality = ""
        self.eac_hex = ""
        self.bone_hex = ""
        self.anterior_ok = False
        self.anterior_point_mm = np.zeros(3)
        self.anterior_normal = np.array([0.0, 0.0, 1.0])
        self.anterior_radius_mm = 1.0
        self.tmj_hex = ""
        self.tmj_voxels = 0
        self.tmj_gap_mm = 0.0
        self.superior_ok = False
        self.superior_point_mm = np.zeros(3)
        self.superior_normal = np.array([0.0, 1.0, 0.0])
        self.superior_radius_mm = 1.0


# Surface normals within this cosine of the canal axis are the lateral face
# of the closed EAC solid. Canal walls point across the axis. On RT143, 0.50
# and 0.65 produce the same plane, so the cut is the wider of those two.
_FACE_ALIGN_COSINE = 0.5


def _exterior_facing(surface: np.ndarray, exterior: np.ndarray):
    """Closed-solid surface voxels that meet exterior air, and those steps.

    A hit is adjacency to boundary-connected empty space. The label itself
    stays closed; this is not an opening in the segmentation.
    """
    touch = np.zeros(surface.shape, dtype=bool)
    steps = np.zeros(surface.shape + (3,), dtype=np.float32)
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for offset in offsets:
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        for axis, step in enumerate(offset):
            if step == 1:
                src[axis] = slice(0, -1)
                dst[axis] = slice(1, None)
            elif step == -1:
                src[axis] = slice(1, None)
                dst[axis] = slice(0, -1)
        air = np.zeros(surface.shape, dtype=bool)
        air[tuple(src)] = surface[tuple(src)] & exterior[tuple(dst)]
        touch |= air
        steps[tuple(src)] += air[tuple(src)][..., None] * np.asarray(offset, dtype=np.float32)
    return touch, steps


def _lateral_face_points(surface, exterior, centroid, long_axis, mm_per_axis):
    """Largest connected patch of the lateral face, in millimetres.

    The face is the closed EAC surface on the lateral side of the centroid
    whose outward steps into exterior air lie along the canal axis.
    """
    touch, steps = _exterior_facing(surface, exterior)
    coords = np.argwhere(touch)
    if len(coords) < 6:
        return None
    points = coords.astype(np.float64) * mm_per_axis
    projection = (points - centroid) @ long_axis
    raw = steps[tuple(coords.T)].astype(np.float64) * mm_per_axis
    length = np.linalg.norm(raw, axis=1)
    length[length < 1e-8] = 1.0
    align = (raw / length[:, None]) @ long_axis
    keep = (projection > 0.0) & (align >= _FACE_ALIGN_COSINE)
    if int(keep.sum()) < 6:
        return None
    chosen = np.zeros(surface.shape, dtype=bool)
    kept = coords[keep]
    chosen[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    labeled, count = ndimage.label(chosen, structure=ndimage.generate_binary_structure(3, 3))
    if count == 0:
        return None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    face = np.argwhere(labeled == int(np.argmax(sizes))).astype(np.float64)
    if len(face) < 6:
        return None
    return face * mm_per_axis


def _robust_plane(points: np.ndarray, prefer: np.ndarray):
    """SVD plane with a few median-residual rejections. Normal follows ``prefer``."""
    keep = np.ones(len(points), dtype=bool)
    center = np.median(points, axis=0)
    normal = np.asarray(prefer, dtype=np.float64)
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    for _ in range(4):
        chosen = points[keep]
        if len(chosen) < 6:
            break
        center = np.median(chosen, axis=0)
        _, _, vh = np.linalg.svd(chosen - center, full_matrices=False)
        candidate = vh[-1]
        if np.dot(candidate, prefer) < 0.0:
            candidate = -candidate
        normal = candidate
        distance = np.abs((points - center) @ normal)
        med = np.median(distance[keep])
        mad = np.median(np.abs(distance[keep] - med)) + 1e-6
        updated = distance <= max(2.5 * 1.4826 * mad, 0.8)
        if int(updated.sum()) < 6 or np.array_equal(updated, keep):
            keep = updated
            break
        keep = updated
    inliers = points[keep] if int(keep.sum()) >= 6 else points
    center = np.median(inliers, axis=0)
    if np.dot(normal, prefer) < 0.0:
        normal = -normal
    radial = np.linalg.norm(np.cross(inliers - center, normal), axis=1)
    radius = float(np.percentile(radial, 90)) if len(radial) else 1.0
    return center, normal / max(np.linalg.norm(normal), 1e-8), max(radius, 1.0)


def _tmj_label(colors, occupied, eac, centroid, mm_per_axis, skip_rgb):
    """Nearest label that sits entirely beyond the closed EAC.

    The TMJ is that segmentation: every one of its voxels is on one side of
    the canal, and it is the closest such label. Bone and the EAC are skipped.
    """
    packed = (
        colors[:, :, :, 0].astype(np.int32)
        | (colors[:, :, :, 1].astype(np.int32) << 8)
        | (colors[:, :, :, 2].astype(np.int32) << 16)
    )
    packed = np.where(occupied, packed, 0)
    skip = {_pack(rgb) for rgb in skip_rgb}
    eac_pts = np.argwhere(eac).astype(np.float64) * mm_per_axis
    best = None
    values, counts = np.unique(packed, return_counts=True)
    for value, count in zip(values, counts):
        if int(value) == 0 or int(value) in skip or int(count) < 1000:
            continue
        pts = np.argwhere(packed == value).astype(np.float64) * mm_per_axis
        center = pts.mean(axis=0)
        delta = center - centroid
        distance = float(np.linalg.norm(delta))
        if distance < 1.0:
            continue
        axis = delta / distance
        gap = float(((pts - centroid) @ axis).min() - ((eac_pts - centroid) @ axis).max())
        if gap < -0.5:
            continue
        if best is None or distance < best[0]:
            best = (distance, int(value), center, gap, int(count))
    return best


def _orthogonal_tangent(points, lateral_normal, prefer):
    """Outer tangent whose normal is perpendicular to the lateral cap plane.

    The normal is the best fit inside that constraint. The plane point is the
    95th percentile along the normal, so the plane lies on the outside of the
    wall and the canal stays on the other side.
    """
    lateral = np.asarray(lateral_normal, dtype=np.float64)
    lateral = lateral / max(np.linalg.norm(lateral), 1e-8)
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(lateral[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(lateral, helper)
    tangent = tangent / max(np.linalg.norm(tangent), 1e-8)
    bitangent = np.cross(lateral, tangent)
    center = np.median(points, axis=0)
    local = np.column_stack([(points - center) @ tangent, (points - center) @ bitangent])
    _values, vectors = np.linalg.eigh(np.cov(local.T))
    normal = vectors[0, 0] * tangent + vectors[1, 0] * bitangent
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    if np.dot(normal, prefer) < 0.0:
        normal = -normal
    height = (points - center) @ normal
    level = float(np.percentile(height, 95))
    point = center + normal * level
    radial = np.linalg.norm(np.cross(points - point, normal), axis=1)
    radius = float(np.percentile(radial, 90)) if len(radial) else 1.0
    return point, normal, max(radius, 1.0)


def _fit_anterior_plane(frame, colors, occupied, eac, bone_rgb) -> str:
    """Tangent of the closed EAC on the side that faces the TMJ."""
    frame.anterior_ok = False
    found = _tmj_label(
        colors,
        occupied,
        eac,
        frame.centroid_mm,
        frame.mm_per_axis,
        (bone_rgb, (int(frame.eac_hex[0:2], 16), int(frame.eac_hex[2:4], 16), int(frame.eac_hex[4:6], 16))),
    )
    if found is None:
        return "No segmentation sits entirely beyond the EAC, so there is no anterior plane."
    _distance, packed, center, gap, count = found
    rgb = (packed & 255, (packed >> 8) & 255, (packed >> 16) & 255)
    frame.tmj_hex = "".join(f"{channel:02x}" for channel in rgb)
    frame.tmj_voxels = count
    frame.tmj_gap_mm = gap
    raw = center - frame.centroid_mm
    lateral = frame.plane_normal
    side = raw - lateral * np.dot(raw, lateral)
    if np.linalg.norm(side) < 0.25 * max(np.linalg.norm(raw), 1e-8):
        return (
            f"#{frame.tmj_hex} is the segmentation beyond the EAC, but it lies along the cap "
            "normal, so a perpendicular tangent is not defined."
        )
    prefer = side / np.linalg.norm(side)
    # Neighbors outside the solid, on the TMJ side of the centroid, facing that way.
    wall = _lateral_face_points(eac, ~eac, frame.centroid_mm, prefer, frame.mm_per_axis)
    if wall is None or len(wall) < 6:
        return f"#{frame.tmj_hex} is beyond the EAC, but that wall is too small to fit a plane."
    point, normal, radius = _orthogonal_tangent(wall, lateral, prefer)
    frame.anterior_ok = True
    frame.anterior_point_mm = point
    frame.anterior_normal = normal
    frame.anterior_radius_mm = radius
    orthogonality = float(np.degrees(np.arccos(np.clip(abs(np.dot(normal, lateral)), 0.0, 1.0))))
    return (
        f"#{frame.tmj_hex} ({count} voxels) lies entirely beyond the EAC "
        f"({gap:.1f} mm clear) and is the TMJ side. The anterior plane is the outer "
        f"tangent of that wall ({len(wall)} voxels, in-plane radius {radius:.1f} mm), "
        f"held at {orthogonality:.0f}° to the cap plane. "
        "Mask 3 drops cortex on the TMJ side of it."
    )


def _superior_direction(plane_normal, anterior_normal) -> np.ndarray:
    """Anterior normal turned 90° counter-clockwise about the cap normal.

    Counter-clockwise is the view in which the cap normal points toward the
    camera, the same side as the lateral arrow.
    """
    lateral = np.asarray(plane_normal, dtype=np.float64)
    lateral = lateral / max(np.linalg.norm(lateral), 1e-8)
    anterior = np.asarray(anterior_normal, dtype=np.float64)
    anterior = anterior - lateral * np.dot(anterior, lateral)
    anterior = anterior / max(np.linalg.norm(anterior), 1e-8)
    superior = np.cross(lateral, anterior)
    return superior / max(np.linalg.norm(superior), 1e-8)


def _fit_superior_plane(frame, eac) -> str:
    """Outer EAC tangent on the side 90° counter-clockwise from the anterior plane."""
    frame.superior_ok = False
    if not frame.anterior_ok:
        return "No anterior plane, so there is no superior plane."
    prefer = _superior_direction(frame.plane_normal, frame.anterior_normal)
    wall = _lateral_face_points(eac, ~eac, frame.centroid_mm, prefer, frame.mm_per_axis)
    if wall is None or len(wall) < 6:
        return "The superior side of the EAC is too small to fit a plane."
    point, normal, radius = _orthogonal_tangent(wall, frame.plane_normal, prefer)
    frame.superior_ok = True
    frame.superior_point_mm = point
    frame.superior_normal = normal
    frame.superior_radius_mm = radius
    return (
        f"The superior plane turns the anterior normal 90° counter-clockwise about the cap "
        f"normal and fits the outer EAC tangent there ({len(wall)} voxels, in-plane radius "
        f"{radius:.1f} mm). Mask 3 drops cortex on the superior side of it."
    )


def derive_lateral_frame(occupied, colors, eac_rgb, bone_rgb, dimensions, laterality="") -> LateralFrame:
    """Long axis by PCA. The plane fits the lateral face of the closed EAC solid."""
    frame = LateralFrame()
    frame.dimensions = np.asarray(dimensions, dtype=np.float64)
    frame.shape = tuple(int(v) for v in occupied.shape)
    frame.laterality = laterality
    frame.eac_hex = "".join(f"{channel:02x}" for channel in eac_rgb)
    frame.bone_hex = "".join(f"{channel:02x}" for channel in bone_rgb)
    frame.mm_per_axis = frame.dimensions / np.asarray(frame.shape, dtype=np.float64) / MM_TO_SIM
    eac = _color_mask(colors, occupied, eac_rgb)
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, count = ndimage.label(eac, structure=structure)
    frame.components = int(count)
    if count == 0:
        frame.summary = f"No voxels use the EAC color #{frame.eac_hex}."
        return frame
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    # A closed canal is the usual case. Prefer the largest component that
    # does not touch the crop, so a stray boundary speck cannot set the axis.
    order = np.argsort(sizes)[::-1]
    chosen = int(order[0])
    for label in order:
        if sizes[label] == 0:
            continue
        if not _touches_boundary(labeled == label):
            chosen = int(label)
            break
    mask = labeled == chosen
    frame.closed = not _touches_boundary(mask)
    coords = np.argwhere(mask).astype(np.float64)
    points = coords * frame.mm_per_axis
    centroid = points.mean(axis=0)
    axes, values = _pca(points - centroid)
    long_axis = axes[:, 0]
    projection = (points - centroid) @ long_axis
    length = float(projection.max() - projection.min())
    frame.length_mm = length
    band = float(np.clip(0.04 * length, 2.5, 8.0))
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = mask & ~eroded
    exterior = exterior_air(occupied)
    surface_coords = np.argwhere(surface).astype(np.float64)
    surface_mm = surface_coords * frame.mm_per_axis
    surface_projection = (surface_mm - centroid) @ long_axis
    step = long_axis / frame.mm_per_axis
    step = step / max(np.linalg.norm(step), 1e-8)
    ends = {}
    for name, sign in (("min", -1.0), ("max", 1.0)):
        if sign < 0.0:
            selected = surface_projection <= (surface_projection.min() + band)
        else:
            selected = surface_projection >= (surface_projection.max() - band)
        cap_coords = surface_coords[selected]
        cap_mm = surface_mm[selected]
        outward = step * sign
        neighbor = np.rint(cap_coords + outward).astype(np.int32)
        shape = np.asarray(mask.shape)
        inside = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
        facing = np.zeros(len(cap_coords), dtype=bool)
        if np.any(inside):
            facing[inside] = exterior[tuple(neighbor[inside].T)]
        ends[name] = {
            "sign": sign,
            "score": int(facing.sum()),
            "points": cap_mm[facing] if np.any(facing) else cap_mm,
            "surface": cap_mm,
            "coords": cap_coords,
        }
    frame.scores = {name: ends[name]["score"] for name in ("min", "max")}
    lateral_name = "min" if ends["min"]["score"] >= ends["max"]["score"] else "max"
    if frame.scores["min"] == frame.scores["max"]:
        # Equal contact: the wider end is the meatus. Still not the camera.
        lateral_name = "min" if len(ends["min"]["coords"]) >= len(ends["max"]["coords"]) else "max"
    picked = ends[lateral_name]
    cap_points = picked["points"]
    if len(cap_points) < 6:
        frame.summary = "The EAC caps are too small to fit a plane."
        return frame
    cap_center = np.median(cap_points, axis=0)
    toward_cap = cap_center - centroid
    if np.dot(long_axis, toward_cap) < 0.0:
        long_axis = -long_axis
    lateral = toward_cap / max(np.linalg.norm(toward_cap), 1e-8)
    # Centroid-to-cap can lean off a curved canal. Its sign still follows the axis.
    if np.dot(lateral, long_axis) < 0.0:
        lateral = -lateral
    # The end band only chooses the side. The plane fits the face of the closed
    # solid on that side: exterior-air contact, normals along the canal.
    face_points = _lateral_face_points(surface, exterior, centroid, long_axis, frame.mm_per_axis)
    used_face = face_points is not None and len(face_points) >= 6
    if used_face:
        cap_center = np.median(face_points, axis=0)
        toward_cap = cap_center - centroid
        lateral = toward_cap / max(np.linalg.norm(toward_cap), 1e-8)
        if np.dot(lateral, long_axis) < 0.0:
            lateral = -lateral
        plane_center, plane_normal, radius = _robust_plane(face_points, lateral)
        cap_points = face_points
        cap_center = plane_center
        toward_cap = cap_center - centroid
        lateral = toward_cap / max(np.linalg.norm(toward_cap), 1e-8)
        if np.dot(lateral, long_axis) < 0.0:
            lateral = -lateral
        if np.dot(plane_normal, lateral) < 0.0:
            plane_normal = -plane_normal
    else:
        plane_center, plane_normal, radius = _robust_plane(picked["surface"], lateral)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(plane_normal, lateral), -1.0, 1.0))))
    frame.ok = True
    frame.centroid_mm = centroid
    frame.cap_center_mm = cap_center
    frame.cap_points_mm = np.ascontiguousarray(cap_points, dtype=np.float64)
    frame.lateral_vector = lateral
    frame.long_axis = long_axis
    frame.plane_point_mm = plane_center
    frame.plane_normal = plane_normal
    frame.cap_radius_mm = radius
    frame.plane_angle_deg = angle
    frame.lateral_end = lateral_name
    other = "max" if lateral_name == "min" else "min"
    closed = "closed" if frame.closed else "open to the volume boundary"
    anterior_clause = _fit_anterior_plane(frame, colors, occupied, mask, bone_rgb)
    superior_clause = _fit_superior_plane(frame, mask)
    if used_face:
        face_clause = (
            f"The cap plane fits that side's face of the closed solid "
            f"({len(cap_points)} surface voxels meeting exterior air and facing along the canal, "
            f"in-plane radius {radius:.1f} mm)."
        )
    else:
        face_clause = "That face was too small, so the plane falls back to the end band."
    frame.summary = (
        f"EAC #{frame.eac_hex} is {closed} ({frame.components} component"
        f"{'' if frame.components == 1 else 's'}, largest used). "
        f"PCA length {length:.1f} mm. The {lateral_name}-projection side meets "
        f"exterior air on {frame.scores[lateral_name]} end-band voxels; "
        f"the {other} side meets it on {frame.scores[other]}. "
        f"{face_clause} "
        f"The lateral vector runs from the EAC centroid to that face "
        f"(alignment with the long axis {np.dot(lateral, long_axis):.2f}). "
        f"The fitted plane is {angle:.1f}° from that vector. "
        f"{anterior_clause} "
        f"{superior_clause} "
        f"Volume laterality is {laterality or 'unset'} and was not used."
    )
    return frame


def _pca(centered: np.ndarray):
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(values)[::-1]
    return vectors[:, order], values[order]


def _pack(rgb) -> int:
    return int(rgb[0]) | (int(rgb[1]) << 8) | (int(rgb[2]) << 16)


def _face_labels(rgb: np.ndarray, tris: np.ndarray) -> np.ndarray:
    packed = (
        rgb[:, 0].astype(np.int32)
        | (rgb[:, 1].astype(np.int32) << 8)
        | (rgb[:, 2].astype(np.int32) << 16)
    )
    face = packed[tris]
    return np.where(
        (face[:, 0] == face[:, 1]) | (face[:, 0] == face[:, 2]),
        face[:, 0],
        np.where(face[:, 1] == face[:, 2], face[:, 1], face[:, 0]),
    )


def _welded(index_verts: np.ndarray, tris: np.ndarray):
    """Shared edge keys across block seams. Display vertices stay put."""
    quantized = np.round(np.asarray(index_verts, dtype=np.float64), 3)
    _unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
    welded = inverse[tris].astype(np.int32)
    valid = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 2] != welded[:, 0])
    )
    return welded, valid


def _edge_list(tris: np.ndarray):
    pairs = np.stack(
        [
            np.sort(tris[:, [0, 1]], axis=1),
            np.sort(tris[:, [1, 2]], axis=1),
            np.sort(tris[:, [2, 0]], axis=1),
        ],
        axis=0,
    ).reshape(-1, 2)
    owners = np.broadcast_to(np.arange(len(tris), dtype=np.int32), (3, len(tris))).reshape(-1)
    span = int(pairs[:, 1].max()) + 1 if len(pairs) else 1
    key = pairs[:, 0].astype(np.int64) * span + pairs[:, 1].astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    owners = owners[order]
    change = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(key)]))
    size = ends - starts
    pair = size == 2
    left = owners[starts[pair]]
    right = owners[starts[pair] + 1]
    extra_left = []
    extra_right = []
    for start, end in zip(starts[size > 2], ends[size > 2]):
        ids = owners[start:end]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                extra_left.append(int(ids[i]))
                extra_right.append(int(ids[j]))
    if extra_left:
        left = np.concatenate([left, np.asarray(extra_left, dtype=np.int32)])
        right = np.concatenate([right, np.asarray(extra_right, dtype=np.int32)])
    edges_a = np.concatenate([left, right]).astype(np.int32, copy=False)
    edges_b = np.concatenate([right, left]).astype(np.int32, copy=False)
    return edges_a, edges_b


def _orient_and_sample(index_verts, tris, occupied, exterior):
    a = index_verts[tris[:, 0]]
    b = index_verts[tris[:, 1]]
    c = index_verts[tris[:, 2]]
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1)
    length[length < 1e-8] = 1.0
    normal = normal / length[:, None]
    centroid = (a + b + c) / 3.0
    shape = np.asarray(occupied.shape, dtype=np.int32)

    def at(sign, step):
        point = centroid + normal * (sign * step)
        index = np.rint(point).astype(np.int32)
        outside = np.any((index < 0) | (index >= shape), axis=1)
        occupied_hit = np.zeros(len(tris), dtype=bool)
        inside = ~outside
        if np.any(inside):
            picked = index[inside]
            occupied_hit[inside] = occupied[picked[:, 0], picked[:, 1], picked[:, 2]]
        return occupied_hit

    sign = -1.0 if at(1.0, 0.8).mean() > at(-1.0, 0.8).mean() else 1.0
    normal = normal * sign
    is_exterior = np.zeros(len(tris), dtype=bool)
    boundary = np.full(len(tris), 999.0)
    found = np.zeros(len(tris), dtype=bool)
    for step in (0.45, 0.95, 1.45):
        point = centroid + normal * step
        index = np.rint(point).astype(np.int32)
        outside = np.any((index < 0) | (index >= shape), axis=1)
        pending = ~found
        crop = pending & outside
        boundary[crop] = 0.0
        found[crop] = True
        inside = pending & ~outside
        if not np.any(inside):
            continue
        picked = index[inside]
        air = ~occupied[picked[:, 0], picked[:, 1], picked[:, 2]]
        accept = np.zeros(len(tris), dtype=bool)
        accept[inside] = air
        if not np.any(accept):
            continue
        found[accept] = True
        chosen = index[accept]
        is_exterior[accept] = exterior[chosen[:, 0], chosen[:, 1], chosen[:, 2]]
        distance = np.minimum.reduce(
            [
                chosen[:, 0],
                chosen[:, 1],
                chosen[:, 2],
                shape[0] - 1 - chosen[:, 0],
                shape[1] - 1 - chosen[:, 1],
                shape[2] - 1 - chosen[:, 2],
            ]
        )
        boundary[accept] = distance
    return is_exterior, boundary, centroid, sign


class SurfaceCache:
    def __init__(self):
        self.ok = False
        self.world = np.zeros((0, 3), np.float32)
        self.normals = np.zeros((0, 3), np.float32)
        self.tris = np.zeros((0, 3), np.int32)
        self.area_mm2 = np.zeros((0,), np.float64)
        self.exterior = np.zeros((0,), bool)
        self.boundary = np.zeros((0,), np.float64)
        self.dist_cap_mm = np.zeros((0,), np.float64)
        self.signed_mm = np.zeros((0,), np.float64)
        self.angle_deg = np.zeros((0,), np.float64)
        self.anterior_ok = False
        self.anterior_mm = np.zeros((0,), np.float64)
        self.superior_ok = False
        self.superior_mm = np.zeros((0,), np.float64)
        self.edges_a = np.zeros((0,), np.int32)
        self.edges_b = np.zeros((0,), np.int32)


def build_surface_cache(index_verts, tris, vertex_rgb, world, normals, bone_rgb, occupied, frame) -> SurfaceCache:
    """Bone/air triangles of the intact surface, with the fields the sliders reuse."""
    cache = SurfaceCache()
    if not frame.ok or len(tris) == 0:
        return cache
    labels = _face_labels(vertex_rgb, tris)
    bone = labels == _pack(bone_rgb)
    if not np.any(bone):
        return cache
    faces = tris[bone]
    welded, valid = _welded(index_verts, faces)
    if not np.any(valid):
        return cache
    faces = faces[valid]
    welded = welded[valid]
    exterior = exterior_air(occupied)
    is_exterior, boundary, centroid, sign = _orient_and_sample(index_verts, faces, occupied, exterior)
    centroid_mm = centroid * frame.mm_per_axis
    millimeters = index_verts * frame.mm_per_axis
    ma = millimeters[faces[:, 0]]
    mb = millimeters[faces[:, 1]]
    mc = millimeters[faces[:, 2]]
    physical = np.cross(mb - ma, mc - ma)
    area = 0.5 * np.linalg.norm(physical, axis=1)
    length = np.linalg.norm(physical, axis=1)
    length[length < 1e-8] = 1.0
    physical = physical / length[:, None] * sign
    angle = np.degrees(np.arccos(np.clip(physical @ frame.lateral_vector, -1.0, 1.0)))
    # Side of the drawn cap plane. The normal points the same way as the arrow,
    # so positive is the lateral side and negative is the side without the arrow.
    signed = (centroid_mm - frame.plane_point_mm) @ frame.plane_normal
    _distance, _index = cKDTree(frame.cap_points_mm).query(centroid_mm, k=1, workers=-1)
    edges_a, edges_b = _edge_list(welded)
    cache.ok = True
    cache.world = np.ascontiguousarray(world, dtype=np.float32) if world is not None else np.zeros((0, 3), np.float32)
    cache.normals = (
        np.ascontiguousarray(normals, dtype=np.float32) if normals is not None else np.zeros_like(cache.world)
    )
    cache.tris = np.ascontiguousarray(faces, dtype=np.int32)
    cache.area_mm2 = area
    cache.exterior = is_exterior
    cache.boundary = boundary
    cache.dist_cap_mm = np.asarray(_distance, dtype=np.float64)
    cache.signed_mm = signed
    cache.angle_deg = angle
    cache.anterior_ok = bool(frame.anterior_ok)
    if frame.anterior_ok:
        cache.anterior_mm = (centroid_mm - frame.anterior_point_mm) @ frame.anterior_normal
    else:
        cache.anterior_mm = np.full(len(area), -1.0e6)
    cache.superior_ok = bool(frame.superior_ok)
    if frame.superior_ok:
        cache.superior_mm = (centroid_mm - frame.superior_point_mm) @ frame.superior_normal
    else:
        cache.superior_mm = np.full(len(area), -1.0e6)
    cache.edges_a = edges_a
    cache.edges_b = edges_b
    return cache


def cache_from_occupancy(occupied, colors, frame, bone_rgb) -> SurfaceCache:
    """Unsmoothed whole-volume mesh. Used to check the rule without the RTX window."""
    from surface import _border_slab, marching_cubes

    shape = occupied.shape
    slab, rgb = _border_slab(occupied.astype(np.uint8), colors, -1, shape[0], -1, shape[1], -1, shape[2])
    verts, tris, _keys, _cells, vert_rgb = marching_cubes(slab, rgb, origin=(-1, -1, -1))
    world = index_to_local(verts, frame.dimensions, frame.shape).astype(np.float32)
    return build_surface_cache(verts, tris, vert_rgb, world, None, bone_rgb, occupied, frame)


def _neighbors(frontier: np.ndarray, edges_a: np.ndarray, edges_b: np.ndarray, count: int) -> np.ndarray:
    found = np.zeros(count, dtype=bool)
    if len(edges_a) == 0 or not np.any(frontier):
        return found
    hit = frontier[edges_a]
    if np.any(hit):
        found[edges_b[hit]] = True
    return found


def _component_labels(count: int, edges_a, edges_b, active: np.ndarray) -> np.ndarray:
    if not np.any(active):
        return np.full(count, -1, dtype=np.int32)
    use = active[edges_a] & active[edges_b] if len(edges_a) else np.zeros(0, dtype=bool)
    if not np.any(use):
        labels = np.full(count, -1, dtype=np.int32)
        labels[active] = np.arange(int(active.sum()), dtype=np.int32)
        return labels
    left = edges_a[use]
    right = edges_b[use]
    graph = sparse.coo_matrix(
        (np.ones(len(left), dtype=np.float32), (left, right)),
        shape=(count, count),
    )
    _n, labels = connected_components(graph, directed=False, return_labels=True)
    labels = labels.astype(np.int32, copy=False)
    labels[~active] = -1
    return labels


def _bridged(active, seeds, edges_a, edges_b, cleanup: int) -> np.ndarray:
    """Active triangles connected to ``seeds``, plus pieces a few faces apart."""
    count = len(active)
    labels = _component_labels(count, edges_a, edges_b, active)
    seed_ok = seeds & active & (labels >= 0)
    if not np.any(seed_ok):
        return np.zeros(count, dtype=bool)
    selected = active & np.isin(labels, np.unique(labels[seed_ok]))
    steps = int(max(0, cleanup))
    if steps == 0:
        return selected
    frontier = selected.copy()
    visited_bridge = np.zeros(count, dtype=bool)
    for _ in range(steps):
        nearby = _neighbors(frontier, edges_a, edges_b, count)
        hit = nearby & active & ~selected & (labels >= 0)
        if np.any(hit):
            joined = active & np.isin(labels, np.unique(labels[hit]))
            selected |= joined
        bridge = nearby & ~active & ~visited_bridge
        visited_bridge |= bridge
        frontier = bridge | (hit & selected)
        if not np.any(frontier):
            break
    return selected


def _nearest(active: np.ndarray, distance: np.ndarray) -> np.ndarray:
    seeds = np.zeros(len(active), dtype=bool)
    if not np.any(active):
        return seeds
    masked = np.where(active, distance, np.inf)
    seeds[int(np.argmin(masked))] = True
    return seeds


class AppearanceIndex:
    """Original mask faces, in voxel index space.

    Mask 1 keeps the lateral-cortex material. Mask 3 is also drawn with the
    markings shader, lifted just off that surface, using the channels saved
    from the tuner. A face opened by the drill sits about a voxel further in,
    so it stays ordinary bone. The occupancy grid is not edited.
    """

    def __init__(self, centroids, bone_rgb, markings_centroids=None, markings_points=None, markings_channels=None, bake=None):
        self.centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
        self.bone_packed = int(bone_rgb[0]) | (int(bone_rgb[1]) << 8) | (int(bone_rgb[2]) << 16)
        self.radius = 0.65
        self.tree = cKDTree(self.centroids) if len(self.centroids) else None
        self.markings_centroids = np.asarray(
            markings_centroids if markings_centroids is not None else np.zeros((0, 3)),
            dtype=np.float64,
        ).reshape(-1, 3)
        self.markings_tree = cKDTree(self.markings_centroids) if len(self.markings_centroids) else None
        self.markings_points = np.asarray(
            markings_points if markings_points is not None else np.zeros((0, 3)),
            dtype=np.float64,
        ).reshape(-1, 3)
        self.markings_channels = np.asarray(
            markings_channels if markings_channels is not None else np.zeros((0, 4)),
            dtype=np.float32,
        ).reshape(-1, 4)
        self._channel_tree = cKDTree(self.markings_points) if len(self.markings_points) else None
        self._bake = bake

    def rebuild_channels(self, params: dict):
        import markings

        self.markings_channels = markings.rebake(self._bake, params)
        return self.markings_channels

    def classify(self, centroids) -> np.ndarray:
        """0 ordinary bone, 1 lateral cortex, 2 cortical markings."""
        centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
        kind = np.zeros(len(centroids), dtype=np.int8)
        if len(centroids) == 0:
            return kind
        if self.tree is not None:
            distance, _index = self.tree.query(centroids, distance_upper_bound=self.radius, workers=-1)
            kind[np.isfinite(distance) & (distance <= self.radius)] = 1
        if self.markings_tree is not None:
            distance, _index = self.markings_tree.query(centroids, distance_upper_bound=self.radius, workers=-1)
            kind[np.isfinite(distance) & (distance <= self.radius)] = 2
        return kind

    def sample_channels(self, index_points) -> np.ndarray:
        points = np.asarray(index_points, dtype=np.float64).reshape(-1, 3)
        out = np.zeros((len(points), 4), np.float32)
        if self._channel_tree is None or len(points) == 0 or len(self.markings_channels) == 0:
            return out
        distance, index = self._channel_tree.query(points, distance_upper_bound=self.radius, workers=-1)
        hit = np.isfinite(distance) & (distance <= self.radius)
        picked = np.where(hit, index, 0)
        out[hit] = self.markings_channels[picked[hit]]
        return out


def build_appearance_index(blocks, occupied, colors, dimensions, laterality, bone_rgb, eac_rgb):
    """Mask 1 on the mesh the live view already built. None if the frame cannot be fit."""
    parts = [mesh for mesh in blocks.meshes.values() if len(mesh.tris)]
    if not parts:
        return None
    index_verts = np.concatenate([mesh.verts for mesh in parts])
    triangles = []
    offset = 0
    for mesh in parts:
        triangles.append(np.asarray(mesh.tris, dtype=np.int64) + offset)
        offset += len(mesh.verts)
    triangles = np.concatenate(triangles)
    vertex_rgb = np.concatenate([mesh.colors for mesh in parts])
    frame = derive_lateral_frame(occupied, colors, eac_rgb, bone_rgb, dimensions, laterality)
    if not frame.ok:
        print(frame.summary, flush=True)
        return None
    cache = build_surface_cache(
        index_verts,
        triangles,
        vertex_rgb,
        np.ascontiguousarray(index_verts, dtype=np.float32),
        None,
        bone_rgb,
        occupied,
        frame,
    )
    if not cache.ok:
        print("Lateral cortex mask is empty; the live bone material is unchanged.", flush=True)
        return None
    result = evaluate_masks(cache, load_parameters())
    faces = cache.tris[result.masks["mask1"]]
    centroids = index_verts[faces].mean(axis=1) if len(faces) else np.zeros((0, 3))
    markings_faces = cache.tris[result.masks["mask3"]]
    markings_centroids = index_verts[markings_faces].mean(axis=1) if len(markings_faces) else np.zeros((0, 3))
    import markings

    markings_points, markings_channels, bake = markings.field_on_mask(
        index_verts,
        cache.tris,
        result.masks["mask3"],
        frame.mm_per_axis,
        markings.load_parameters(),
    )
    print(
        f"Live lateral cortex {result.line('mask1')} and markings shader {result.line('mask3')} "
        "from the tuner. Drill faces stay on the bone material.",
        flush=True,
    )
    return AppearanceIndex(
        centroids,
        bone_rgb,
        markings_centroids,
        markings_points,
        markings_channels,
        bake,
    )


class MaskResult:
    def __init__(self):
        self.masks = {key: np.zeros((0,), bool) for key in MASK_KEYS}
        self.stats = {key: {"triangles": 0, "area_mm2": 0.0} for key in MASK_KEYS}
        self.notes = []

    def line(self, key: str) -> str:
        stat = self.stats[key]
        return f"{stat['triangles']:,} tris, {stat['area_mm2'] / 100.0:.2f} cm²"


def evaluate_masks(cache: SurfaceCache, parameters: dict) -> MaskResult:
    """Lateral cortex stops at the cap plane.

    Mask 1 is the bone surface on the arrow side of the EAC cap plane,
    connected to the cap. Contiguous bone on the other side of that plane
    stays in the bone structure. Mask 3 is that cap with the TMJ side of the
    anterior plane and the superior side of the superior plane removed.
    Islands that do not touch the cap stay bone.
    """
    result = MaskResult()
    if not cache.ok:
        result.notes.append("No bone surface was available.")
        return result
    margin = float(parameters["crop_margin_voxels"])
    if margin <= 0.0:
        facing = cache.exterior | (cache.boundary == 0.0)
    else:
        facing = cache.exterior & (cache.boundary > margin)
    # Positive distance is the arrow side of the cap plane.
    cap_side = cache.signed_mm >= -0.25
    candidate = facing & cap_side
    seed_mm = float(parameters["seed_distance_mm"])
    seeds = candidate & (cache.dist_cap_mm <= seed_mm)
    if not np.any(seeds):
        seeds = _nearest(candidate, cache.dist_cap_mm)
        result.notes.append("No bone on the arrow side of the cap plane was inside the seed distance; the nearest face was used.")
    mask1 = _bridged(candidate, seeds, cache.edges_a, cache.edges_b, 0)
    mask2 = mask1
    keep = np.ones(len(cache.area_mm2), dtype=bool)
    if cache.anterior_ok:
        on_tmj_side = cache.anterior_mm > 0.0
        dropped = float(cache.area_mm2[mask1 & on_tmj_side].sum()) if np.any(mask1 & on_tmj_side) else 0.0
        keep &= ~on_tmj_side
        if dropped > 0.0:
            result.notes.append(
                f"Mask 3 drops {dropped / 100.0:.2f} cm² on the TMJ side of the anterior plane."
            )
        else:
            result.notes.append("The anterior plane does not cross this cap, so it does not narrow mask 3.")
    else:
        result.notes.append("No anterior plane, so mask 3 is not cut on the TMJ side.")
    if cache.superior_ok:
        on_superior_side = cache.superior_mm > 0.0
        dropped = float(cache.area_mm2[mask1 & keep & on_superior_side].sum()) if np.any(mask1 & keep & on_superior_side) else 0.0
        keep &= ~on_superior_side
        if dropped > 0.0:
            result.notes.append(
                f"Mask 3 drops {dropped / 100.0:.2f} cm² on the superior side of the superior plane."
            )
        else:
            result.notes.append("The superior plane does not cross this cap, so it does not narrow mask 3.")
    else:
        result.notes.append("No superior plane, so mask 3 is not cut superiorly.")
    mask3 = mask1 & keep
    for key, mask in (("mask1", mask1), ("mask2", mask2), ("mask3", mask3)):
        result.masks[key] = mask
        result.stats[key] = {
            "triangles": int(mask.sum()),
            "area_mm2": float(cache.area_mm2[mask].sum()) if np.any(mask) else 0.0,
        }
    if result.stats["mask1"]["triangles"] == 0:
        result.notes.append("Mask 1 is empty. Nothing on the arrow side of the cap plane touches the cap.")
    return result


def _round_list(values) -> list[float]:
    return [round(float(value), 4) for value in np.asarray(values).tolist()]


def configuration(frame: LateralFrame, result: MaskResult, parameters: dict, anatomy: str) -> dict:
    offset = float(parameters["plane_offset_mm"])
    return {
        "version": 1,
        "kind": "derived_appearance_mask",
        "target": "bone",
        "anatomy": anatomy,
        "selected_mask": parameters.get("selected_mask", "mask3"),
        "parameters": {
            "seed_distance_mm": float(parameters["seed_distance_mm"]),
            "normal_angle_deg": float(parameters["normal_angle_deg"]),
            "plane_offset_mm": offset,
            "plane_depth_mm": offset * float(parameters["plane_depth_scale"]),
            "plane_depth_scale": float(parameters["plane_depth_scale"]),
            "cleanup_steps": int(round(parameters["cleanup_steps"])),
            "crop_margin_voxels": int(round(parameters["crop_margin_voxels"])),
            "opacity": float(parameters["opacity"]),
        },
        "frame": {
            "space": "voxel_millimetres",
            "eac_hex": frame.eac_hex,
            "bone_hex": frame.bone_hex,
            "eac_closed": bool(frame.closed),
            "lateral_end": frame.lateral_end,
            "cap_scores": frame.scores,
            "centroid_mm": _round_list(frame.centroid_mm),
            "cap_center_mm": _round_list(frame.cap_center_mm),
            "lateral_vector": _round_list(frame.lateral_vector),
            "long_axis": _round_list(frame.long_axis),
            "plane_point_mm": _round_list(frame.plane_point_mm),
            "plane_normal": _round_list(frame.plane_normal),
            "plane_angle_deg": round(float(frame.plane_angle_deg), 2),
            "length_mm": round(float(frame.length_mm), 2),
            "anterior_ok": bool(frame.anterior_ok),
            "anterior_point_mm": _round_list(frame.anterior_point_mm),
            "anterior_normal": _round_list(frame.anterior_normal),
            "anterior_radius_mm": round(float(frame.anterior_radius_mm), 2),
            "tmj_hex": frame.tmj_hex,
            "tmj_voxels": int(frame.tmj_voxels),
            "tmj_gap_mm": round(float(frame.tmj_gap_mm), 2),
            "superior_ok": bool(frame.superior_ok),
            "superior_point_mm": _round_list(frame.superior_point_mm),
            "superior_normal": _round_list(frame.superior_normal),
            "superior_radius_mm": round(float(frame.superior_radius_mm), 2),
            "summary": frame.summary,
        },
        "masks": result.stats,
        "notes": list(result.notes),
    }


def save_configuration(payload: dict, path: pathlib.Path = CONFIG_PATH) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_parameters(path: pathlib.Path = CONFIG_PATH) -> dict:
    parameters = dict(DEFAULT_PARAMETERS)
    if not path.is_file():
        return parameters
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return parameters
    stored = saved.get("parameters") or {}
    for key in (
        "seed_distance_mm",
        "normal_angle_deg",
        "plane_offset_mm",
        "plane_depth_scale",
        "cleanup_steps",
        "crop_margin_voxels",
        "opacity",
    ):
        if key in stored:
            parameters[key] = stored[key]
    if stored.get("plane_depth_mm") and stored.get("plane_offset_mm"):
        parameters["plane_depth_scale"] = float(stored["plane_depth_mm"]) / max(float(stored["plane_offset_mm"]), 1e-6)
    selected = str(saved.get("selected_mask") or "")
    if selected in ("off",) + MASK_KEYS:
        parameters["selected_mask"] = selected
    return parameters


def mm_to_world(points_mm, frame: LateralFrame, rotation, translation) -> np.ndarray:
    index = np.asarray(points_mm, dtype=np.float64) / frame.mm_per_axis
    local = index_to_local(index, frame.dimensions, frame.shape)
    return local_to_world(local, rotation, translation)


def direction_to_world(direction_mm, rotation) -> np.ndarray:
    world = np.asarray(direction_mm, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T
    length = np.linalg.norm(world)
    if length < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return world / length


def _basis(normal: np.ndarray):
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(normal[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    tangent = np.cross(normal, helper)
    tangent = tangent / max(np.linalg.norm(tangent), 1e-8)
    bitangent = np.cross(normal, tangent)
    return tangent, bitangent


def _sphere(center, radius, stacks=6, slices=10):
    points = []
    for i in range(stacks + 1):
        phi = np.pi * (i / stacks - 0.5)
        ring = np.cos(phi) * radius
        z = np.sin(phi) * radius
        for j in range(slices):
            theta = 2.0 * np.pi * j / slices
            points.append(center + np.array([np.cos(theta) * ring, np.sin(theta) * ring, z]))
    verts = np.asarray(points, dtype=np.float32)
    tris = []
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = i * slices + (j + 1) % slices
            c = (i + 1) * slices + j
            d = (i + 1) * slices + (j + 1) % slices
            tris.extend((a, c, b, b, c, d))
    return verts, np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def _arrow(origin, direction, length, radius):
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), 1e-8)
    tangent, bitangent = _basis(direction)
    sides = 10
    shaft = 0.72 * length
    ring = []
    head = []
    for j in range(sides):
        theta = 2.0 * np.pi * j / sides
        offset = tangent * np.cos(theta) + bitangent * np.sin(theta)
        ring.append(origin + offset * radius)
        head.append(origin + direction * shaft + offset * radius * 1.7)
    points = [origin, origin + direction * shaft, origin + direction * length]
    points.extend(ring)
    points.extend(head)
    verts = np.asarray(points, dtype=np.float32)
    tris = []
    for j in range(sides):
        nxt = (j + 1) % sides
        a = 3 + j
        b = 3 + nxt
        c = 3 + sides + j
        d = 3 + sides + nxt
        tris.extend((0, b, a, 1, a, b, 1, d, c, 2, c, d))
    return verts, np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def _quad(center, normal, half):
    tangent, bitangent = _basis(np.asarray(normal, dtype=np.float64))
    corners = [
        center + tangent * half + bitangent * half,
        center - tangent * half + bitangent * half,
        center - tangent * half - bitangent * half,
        center + tangent * half - bitangent * half,
    ]
    verts = np.asarray(corners, dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return verts, tris


def subset_mesh(world, normals, tris, mask, offset, with_normals=False):
    faces = tris[mask]
    if len(faces) == 0:
        empty_points = np.zeros((0, 3), np.float32)
        empty_tris = np.zeros((0, 3), np.int32)
        if with_normals:
            return empty_points, empty_tris, empty_points.copy()
        return empty_points, empty_tris
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    points = np.array(world[used], dtype=np.float32, copy=True)
    norms = None
    if normals is not None and len(normals) == len(world):
        norms = np.ascontiguousarray(normals[used], dtype=np.float32)
        points = points + norms * np.float32(offset)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    points = np.ascontiguousarray(points, dtype=np.float32)
    if with_normals:
        if norms is None:
            norms = np.zeros_like(points)
        return points, faces, norms
    return points, faces


def overlay_material(color, opacity: float) -> dict:
    params = dict(rtx_materials.FALLBACK)
    params["diffuse_reflection_color"] = [float(channel) for channel in color]
    params["specular_reflection_roughness"] = 0.55
    params["enable_opacity"] = True
    params["geometry_opacity"] = float(np.clip(opacity, 0.02, 1.0))
    params["emission_weight"] = 0.45
    params["emission_color"] = [float(channel) for channel in color]
    return params


class CortexPanel:
    """Lateral Cortex Masks window. Visibility and overlays stay in this tuner."""

    def __init__(self, viewer, materials, cache, frame, rotation, translation, anatomy: str):
        self.viewer = viewer
        self.materials = materials
        self.cache = cache
        self.frame = frame
        self.rotation = rotation
        self.translation = translation
        self.anatomy = anatomy
        self.parameters = load_parameters()
        self.result = evaluate_masks(cache, self.parameters) if cache is not None and cache.ok else MaskResult()
        self.status = "Diagnostic overlay. Bone, drilling, and default materials stay as they are."
        self.eac_id = frame.eac_hex if frame is not None else ""
        self.prims = {}
        self.pivot = None
        self._authored = False
        self.on_masks_changed = None
        if frame is not None and frame.ok:
            self.pivot = mm_to_world(frame.cap_center_mm, frame, rotation, translation)
            self._author_guides()
            self._author_masks()
            self._author_appearance()
            self._authored = True
            print(frame.summary, flush=True)
            for key in MASK_KEYS:
                print(f"{key}: {self.result.line(key)}", flush=True)
            for note in self.result.notes:
                print(note, flush=True)

    def _log(self, name, segment, points, tris, color, opacity, hidden):
        import warp as wp

        if len(tris) == 0:
            points = np.zeros((3, 3), np.float32)
            tris = np.array([[0, 1, 2]], np.int32)
            hidden = True
        self.viewer.log_mesh(
            name,
            wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3),
            wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
            color=tuple(color),
            roughness=0.45,
            metallic=0.0,
            opacity=float(opacity),
            hidden=bool(hidden),
            backface_culling=False,
            dynamic=True,
        )
        prim = self.viewer.stage.GetPrimAtPath(self.viewer._get_path(self.viewer._qualify(name)))
        self.prims[name] = prim
        if segment and prim and prim.IsValid():
            rtx_materials.apply_params(self.viewer, segment, overlay_material(color, opacity))
            rtx_materials.bind_prim(self.viewer, prim, segment, self.viewer._material_document)
        return prim

    def _author_guides(self):
        frame = self.frame
        cap = mm_to_world(frame.cap_center_mm, frame, self.rotation, self.translation)
        centroid = mm_to_world(frame.centroid_mm, frame, self.rotation, self.translation)
        lateral = direction_to_world(frame.lateral_vector, self.rotation)
        normal = direction_to_world(frame.plane_normal, self.rotation)
        radius = 1.6 * MM_TO_SIM
        self._log("cortex_centroid", "lateral_centroid", *_sphere(centroid, radius), CENTROID_COLOR, 1.0, not self.parameters["show_centroid"])
        self._log("cortex_cap", "lateral_cap", *_sphere(cap, radius), CAP_COLOR, 1.0, not self.parameters["show_cap"])
        reach = cap + lateral * (18.0 * MM_TO_SIM)
        shaft_len = float(np.linalg.norm(cap - centroid))
        shaft_dir = (cap - centroid) / max(shaft_len, 1e-8)
        shaft_pts, shaft_tris = _arrow(centroid, shaft_dir, shaft_len, 0.45 * MM_TO_SIM)
        head_pts, head_tris = _arrow(cap, lateral, 18.0 * MM_TO_SIM, 0.7 * MM_TO_SIM)
        points = np.vstack([shaft_pts, head_pts])
        tris = np.vstack([shaft_tris, head_tris + len(shaft_pts)])
        self._log("cortex_vector", "lateral_vector", points, tris, VECTOR_COLOR, 1.0, not self.parameters["show_vector"])
        half = max(frame.cap_radius_mm, 8.0) * 1.25 * MM_TO_SIM
        plane_center = mm_to_world(frame.plane_point_mm, frame, self.rotation, self.translation)
        self._log(
            "cortex_plane",
            "lateral_plane",
            *_quad(plane_center, normal, half),
            PLANE_COLOR,
            self.parameters["opacity"],
            not self.parameters["show_plane"],
        )
        if frame.anterior_ok:
            anterior_point = mm_to_world(frame.anterior_point_mm, frame, self.rotation, self.translation)
            anterior_normal = direction_to_world(frame.anterior_normal, self.rotation)
            anterior_half = max(frame.anterior_radius_mm, 8.0) * 1.25 * MM_TO_SIM
            quad_points, quad_tris = _quad(anterior_point, anterior_normal, anterior_half)
            arrow_points, arrow_tris = _arrow(anterior_point, anterior_normal, 16.0 * MM_TO_SIM, 0.55 * MM_TO_SIM)
            points = np.vstack([quad_points, arrow_points])
            tris = np.vstack([quad_tris, arrow_tris + len(quad_points)])
            self._log(
                "cortex_anterior",
                "lateral_anterior",
                points,
                tris,
                ANTERIOR_COLOR,
                self.parameters["opacity"],
                not self.parameters["show_anterior_plane"],
            )
        if frame.superior_ok:
            superior_point = mm_to_world(frame.superior_point_mm, frame, self.rotation, self.translation)
            superior_normal = direction_to_world(frame.superior_normal, self.rotation)
            superior_half = max(frame.superior_radius_mm, 8.0) * 1.25 * MM_TO_SIM
            quad_points, quad_tris = _quad(superior_point, superior_normal, superior_half)
            arrow_points, arrow_tris = _arrow(superior_point, superior_normal, 16.0 * MM_TO_SIM, 0.55 * MM_TO_SIM)
            points = np.vstack([quad_points, arrow_points])
            tris = np.vstack([quad_tris, arrow_tris + len(quad_points)])
            self._log(
                "cortex_superior",
                "lateral_superior",
                points,
                tris,
                SUPERIOR_COLOR,
                self.parameters["opacity"],
                not self.parameters["show_superior_plane"],
            )
        # ``reach`` keeps the arrow end available for a later extent check.
        self._reach = reach

    def _author_masks(self):
        offset = 0.25 * MM_TO_SIM
        hidden_key = self.parameters["selected_mask"]
        for key, color in MASK_COLORS.items():
            points, tris = subset_mesh(
                self.cache.world, self.cache.normals, self.cache.tris, self.result.masks[key], offset
            )
            self._log(f"cortex_{key}", f"lateral_{key}", points, tris, color, self.parameters["opacity"], key != hidden_key)

    def _appearance_mesh(self, key: str, lift_mm: float):
        return subset_mesh(
            self.cache.world,
            self.cache.normals,
            self.cache.tris,
            self.result.masks[key],
            lift_mm * MM_TO_SIM,
        )

    def _log_appearance(self, segment: str, points, tris):
        import warp as wp

        hidden = len(tris) == 0
        if hidden:
            points = np.zeros((3, 3), np.float32)
            tris = np.array([[0, 1, 2]], np.int32)
        color = tuple(rtx_materials.resolve(self.materials.document, segment)["diffuse_reflection_color"])
        self.viewer.log_mesh(
            segment,
            wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3),
            wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
            color=color,
            roughness=0.45,
            metallic=0.0,
            hidden=hidden,
            backface_culling=False,
            dynamic=True,
        )
        prim = self.viewer.stage.GetPrimAtPath(self.viewer._get_path(self.viewer._qualify(segment)))
        self.prims[segment] = prim
        if prim and prim.IsValid():
            if segment != "cortex_markings":
                params = rtx_materials.resolve(self.materials.document, segment)
                rtx_materials.apply_params(self.viewer, segment, params)
                rtx_materials.bind_prim(self.viewer, prim, segment, self.viewer._material_document)
            shown = self.materials.visible.get(segment, True) and not hidden
            rtx_materials.set_visibility(self.viewer, prim, shown)
        return prim

    def _log_bone_surface(self):
        """Bone triangles outside mask 1. The lateral cortex occupies that surface."""
        if self.cache is None or not self.cache.ok or self.frame is None:
            return
        mask = self.result.masks.get("mask1")
        if mask is None:
            return
        points, tris, normals = subset_mesh(
            self.cache.world,
            self.cache.normals,
            self.cache.tris,
            ~np.asarray(mask, dtype=bool),
            0.0,
            with_normals=True,
        )
        name = rtx_materials.name_of(self.materials.document, self.frame.bone_hex) or "bone"
        segment = self.frame.bone_hex
        hidden = len(tris) == 0
        if hidden:
            points = np.zeros((3, 3), np.float32)
            tris = np.array([[0, 1, 2]], np.int32)
            normals = np.zeros((3, 3), np.float32)
        color = tuple(rtx_materials.identity_color(segment))
        qualified = self.viewer._qualify(name)
        import warp as wp

        try:
            self.viewer.log_mesh(
                name,
                wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3),
                wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
                normals=wp.array(np.ascontiguousarray(normals, np.float32), dtype=wp.vec3),
                color=color,
                roughness=0.55,
                metallic=0.0,
                dynamic=True,
                backface_culling=False,
            )
        except Exception as exc:
            print(f"bone surface update failed: {exc}", flush=True)
            return
        prim = self.viewer.stage.GetPrimAtPath(self.viewer._get_path(qualified))
        if not prim or not prim.IsValid():
            return
        for item in self.materials.segments:
            if item["id"] == segment:
                item["prim"] = prim
        params = rtx_materials.resolve(self.materials.document, segment)
        rtx_materials.apply_params(self.viewer, segment, params)
        rtx_materials.bind_prim(self.viewer, prim, segment, self.viewer._material_document)
        rtx_materials.set_visibility(self.viewer, prim, not hidden)

    def _author_appearance(self):
        """Lateral cortex is mask 1. The markings shader is drawn on the cap."""
        self._log_bone_surface()
        for key, segment, lift in APPEARANCE_LAYERS:
            self.materials.visible.setdefault(segment, True)
            points, tris = self._appearance_mesh(key, lift)
            prim = self._log_appearance(segment, points, tris)
            if prim is None:
                continue
            if not any(item["id"] == segment for item in self.materials.segments):
                self.materials.segments.append({"id": segment, "prim": prim})

    def _refresh_masks(self):
        offset = 0.25 * MM_TO_SIM
        for key in MASK_KEYS:
            points, tris = subset_mesh(
                self.cache.world, self.cache.normals, self.cache.tris, self.result.masks[key], offset
            )
            self._log(
                f"cortex_{key}",
                "",
                points,
                tris,
                MASK_COLORS[key],
                self.parameters["opacity"],
                key != self.parameters["selected_mask"],
            )
        for key, segment, lift in APPEARANCE_LAYERS:
            points, tris = self._appearance_mesh(key, lift)
            self._log_appearance(segment, points, tris)
        self._log_bone_surface()
        if self.on_masks_changed is not None:
            self.on_masks_changed()

    def _show(self, name: str, visible: bool):
        prim = self.prims.get(name)
        if prim is not None and prim.IsValid():
            rtx_materials.set_visibility(self.viewer, prim, bool(visible))

    def _apply_opacity(self):
        opacity = float(self.parameters["opacity"])
        for key, color in MASK_COLORS.items():
            rtx_materials.apply_params(self.viewer, f"lateral_{key}", overlay_material(color, opacity))
        rtx_materials.apply_params(self.viewer, "lateral_plane", overlay_material(PLANE_COLOR, opacity))
        if "cortex_anterior" in self.prims:
            rtx_materials.apply_params(self.viewer, "lateral_anterior", overlay_material(ANTERIOR_COLOR, opacity))
        if "cortex_superior" in self.prims:
            rtx_materials.apply_params(self.viewer, "lateral_superior", overlay_material(SUPERIOR_COLOR, opacity))

    def _recompute(self):
        self.result = evaluate_masks(self.cache, self.parameters)
        self._refresh_masks()
        self._apply_visibility()

    def _apply_visibility(self):
        selected = self.parameters["selected_mask"]
        for key in MASK_KEYS:
            count = self.result.stats[key]["triangles"]
            self._show(f"cortex_{key}", selected == key and count > 0)
        self._show("cortex_plane", self.parameters["show_plane"])
        self._show("cortex_anterior", self.parameters["show_anterior_plane"] and self.frame.anterior_ok)
        self._show("cortex_superior", self.parameters["show_superior_plane"] and self.frame.superior_ok)
        self._show("cortex_centroid", self.parameters["show_centroid"])
        self._show("cortex_cap", self.parameters["show_cap"])
        self._show("cortex_vector", self.parameters["show_vector"])

    def _save(self):
        payload = configuration(self.frame, self.result, self.parameters, self.anatomy)
        save_configuration(payload)
        self.status = f"Saved {CONFIG_PATH.name}"

    def draw(self, imgui):
        imgui.set_next_window_pos(imgui.ImVec2(848, 16), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(420, 780), imgui.Cond_.first_use_ever)
        imgui.begin("Lateral Cortex Masks")
        imgui.text("Lateral cortex")
        imgui.text_disabled("Diagnostic masks. They do not change the bone.")
        if not self._authored:
            imgui.text_wrapped(self.frame.summary if self.frame is not None else "No frame.")
            imgui.end()
            return
        imgui.text_wrapped(self.frame.summary)
        imgui.separator()
        imgui.text("View")
        imgui.text_disabled("Left-drag looks. Middle-drag orbits the cap. Scroll zooms.")
        imgui.text_disabled("Lateral cortex is mask 1. The markings shader sits on the cap and fades at its edge.")
        options = (("off", "Off"), ("mask1", "Mask 1"), ("mask2", "Mask 2"), ("mask3", "Mask 3"))
        for index, (key, label) in enumerate(options):
            clicked, _chosen = imgui.selectable(
                f"{label}##cortex_{key}",
                self.parameters["selected_mask"] == key,
                0,
                imgui.ImVec2(88, 0),
            )
            if clicked and self.parameters["selected_mask"] != key:
                self.parameters["selected_mask"] = key
                self._apply_visibility()
            if index < len(options) - 1:
                imgui.same_line()
        changed, opacity = imgui.slider_float(
            "Opacity", float(self.parameters["opacity"]), 0.05, 1.0, "%.2f"
        )
        if changed:
            self.parameters["opacity"] = float(opacity)
            self._apply_opacity()
        imgui.separator()
        if self.eac_id in self.materials.visible:
            shown = bool(self.materials.visible[self.eac_id])
            toggled, value = imgui.checkbox("EAC segmentation", shown)
            if toggled:
                self.materials.visible[self.eac_id] = bool(value)
                self.materials._show()
        else:
            imgui.text_disabled("EAC segmentation is not in this mesh.")
        for key, label in (
            ("show_plane", "Cap plane"),
            ("show_anterior_plane", "Anterior plane"),
            ("show_superior_plane", "Superior plane"),
            ("show_centroid", "EAC centroid"),
            ("show_cap", "Cap center"),
            ("show_vector", "Lateral vector"),
        ):
            toggled, value = imgui.checkbox(label, bool(self.parameters[key]))
            if toggled:
                self.parameters[key] = bool(value)
                self._apply_visibility()
        imgui.separator()
        dirty = False
        dirty |= self._slider(imgui, "seed_distance_mm", "Seed distance", 1.0, 20.0, "%.1f mm")
        dirty |= self._slider(imgui, "normal_angle_deg", "Normal angle", 30.0, 85.0, "%.0f deg")
        dirty |= self._slider(imgui, "plane_offset_mm", "Plane offset / depth", 2.0, 25.0, "%.1f mm")
        dirty |= self._slider(imgui, "cleanup_steps", "Connectivity cleanup", 0.0, 6.0, "%.0f steps")
        dirty |= self._slider(imgui, "crop_margin_voxels", "Crop-boundary exclusion", 0.0, 12.0, "%.0f vox")
        imgui.text_disabled("Mask 1 stops at the cap plane, on the arrow side. The other side stays bone.")
        if dirty:
            self.parameters["cleanup_steps"] = int(round(self.parameters["cleanup_steps"]))
            self.parameters["crop_margin_voxels"] = int(round(self.parameters["crop_margin_voxels"]))
            self._recompute()
        imgui.separator()
        imgui.text("Mask 1  arrow side of the cap plane")
        imgui.text_disabled(self.result.line("mask1"))
        imgui.text("Mask 2  lateral of the cap plane")
        imgui.text_disabled(self.result.line("mask2"))
        imgui.text("Mask 3  cap, anterior, and superior planes")
        imgui.text_disabled(self.result.line("mask3"))
        for note in self.result.notes:
            imgui.text_wrapped(note)
        imgui.separator()
        if imgui.button("Save mask"):
            self._save()
        imgui.text_disabled(self.status)
        imgui.text_disabled(str(CONFIG_PATH))
        imgui.end()

    def _slider(self, imgui, key, label, low, high, fmt) -> bool:
        changed, value = imgui.slider_float(label, float(self.parameters[key]), float(low), float(high), fmt)
        if changed:
            self.parameters[key] = float(value)
        return bool(changed)


def _self_check() -> None:
    """Closed canal in a bone block. The air-facing end must win, without a camera."""
    shape = (48, 40, 40)
    occupied = np.zeros(shape, dtype=bool)
    colors = np.zeros(shape + (3,), dtype=np.uint8)
    bone = (255, 249, 219)
    eac = (255, 225, 214)
    occupied[8:40, 8:32, 8:32] = True
    colors[occupied] = bone
    yy, zz = np.ogrid[0:40, 0:40]
    disk = (yy - 20) ** 2 + (zz - 20) ** 2 <= 9
    for x in range(14, 32):
        slab = disk.copy()
        occupied[x, slab] = True
        colors[x, slab] = eac
    # Open a lateral tunnel on +x so only that cap meets exterior air.
    occupied[32:40, 16:25, 16:25] = False
    # A block entirely on +z, clear of the canal, stands in for the TMJ.
    tmj = (100, 0, 0)
    occupied[16:30, 14:26, 28:36] = True
    colors[16:30, 14:26, 28:36] = tmj
    frame = derive_lateral_frame(occupied, colors, eac, bone, np.ones(3), laterality="right")
    if not frame.ok:
        raise SystemExit(f"self-check frame failed: {frame.summary}")
    if frame.cap_center_mm[0] <= frame.centroid_mm[0]:
        raise SystemExit("self-check picked the buried cap")
    if abs(frame.lateral_vector[0]) < 0.8 or frame.lateral_vector[0] < 0.0:
        raise SystemExit(f"self-check lateral vector {frame.lateral_vector}")
    if np.dot(frame.lateral_vector, frame.long_axis) <= 0.0:
        raise SystemExit("self-check lateral vector opposes the long axis")
    if not frame.closed:
        raise SystemExit("self-check expected a closed EAC solid")
    if frame.plane_point_mm[0] <= frame.centroid_mm[0]:
        raise SystemExit("self-check plane is not on the lateral face")
    if frame.cap_radius_mm < 3.0:
        raise SystemExit(f"self-check plane is a tip, radius {frame.cap_radius_mm:.2f}")
    if not frame.anterior_ok:
        raise SystemExit(f"self-check anterior plane missing: {frame.summary}")
    if abs(float(np.dot(frame.anterior_normal, frame.plane_normal))) > 0.08:
        raise SystemExit(f"self-check anterior plane is not perpendicular: {frame.anterior_normal}")
    if frame.anterior_normal[2] <= 0.7 or frame.anterior_point_mm[2] <= frame.centroid_mm[2]:
        raise SystemExit(
            f"self-check anterior plane does not face the TMJ "
            f"{frame.anterior_normal} at {frame.anterior_point_mm}"
        )
    if not frame.superior_ok:
        raise SystemExit(f"self-check superior plane missing: {frame.summary}")
    if abs(float(np.dot(frame.superior_normal, frame.plane_normal))) > 0.08:
        raise SystemExit(f"self-check superior plane is not perpendicular to the cap: {frame.superior_normal}")
    if abs(float(np.dot(frame.superior_normal, frame.anterior_normal))) > 0.08:
        raise SystemExit(f"self-check superior plane is not perpendicular to the anterior plane: {frame.superior_normal}")
    expected = _superior_direction(frame.plane_normal, frame.anterior_normal)
    if float(np.dot(frame.superior_normal, expected)) < 0.7:
        raise SystemExit(f"self-check superior normal {frame.superior_normal} is not the counter-clockwise turn {expected}")
    if frame.superior_normal[1] >= -0.7 or frame.superior_point_mm[1] >= frame.centroid_mm[1]:
        raise SystemExit(
            f"self-check superior plane is not the counter-clockwise side "
            f"{frame.superior_normal} at {frame.superior_point_mm}"
        )
    cache = cache_from_occupancy(occupied, colors, frame, bone)
    result = evaluate_masks(cache, dict(DEFAULT_PARAMETERS))
    if result.stats["mask1"]["triangles"] == 0 or result.stats["mask3"]["triangles"] == 0:
        raise SystemExit(f"self-check empty masks {result.stats}")
    if result.stats["mask3"]["triangles"] > result.stats["mask1"]["triangles"]:
        raise SystemExit("self-check mask 3 is wider than mask 1")
    for key in MASK_KEYS:
        chosen = result.masks[key]
        if np.any(chosen) and float(cache.signed_mm[chosen].min()) < -0.5:
            raise SystemExit(f"self-check {key} crosses medial of the cap plane")
    if result.stats["mask2"]["triangles"] > result.stats["mask1"]["triangles"]:
        raise SystemExit("self-check mask 2 is wider than mask 1")
    if not np.all(result.masks["mask3"] <= result.masks["mask2"]):
        raise SystemExit("self-check mask 3 is not inside the cap-plane region")
    if not np.all(result.masks["mask2"] <= result.masks["mask1"]):
        raise SystemExit("self-check mask 2 is not inside mask 1")
    if np.any(result.masks["mask3"]) and float(cache.superior_mm[result.masks["mask3"]].max()) > 0.5:
        raise SystemExit("self-check mask 3 crosses the superior plane")
    print(
        f"self-check ok  mask1 {result.line('mask1')}  mask3 {result.line('mask3')}",
        flush=True,
    )


if __name__ == "__main__":
    _self_check()
