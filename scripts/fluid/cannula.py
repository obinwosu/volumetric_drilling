"""Irrigation cannula mounted on the drill shaft, not in the shaft OBJ.

The drill files paint silver, gold, black, and grey by region. The cannula is
its own mesh and one stainless material, and it uses the drill body's pose
with no burr spin. Local +X runs from the burr center toward the handle.
Authored OBJ units are 0.01 per millimetre, the same numbers AMBF loads at
scale 1. The 1, 2, and 6 mm meshes are the supplied tubes. The 4 mm shaft's
second taper sits 30 mm further up the handle than the 6 mm, so that tube is
the 6 mm cannula with a straight section inserted and the clip carried to 74 mm.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

MM_TO_SIM = 0.01007
MESH_MM = 0.01
# Pulls the geometric tangent slightly inside the burr so the jet meets the
# cutting surface instead of skating past it. Same value for every size.
AIM_INSET_MM = 0.35
SIZES = (1, 2, 4, 6)
# Clip on the 6.06 mm-radius neck, just proximal to the second taper.
CLIP_MM = {1: 134.0, 2: 124.0, 4: 74.0, 6: 44.0}

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "ADF" / "meshes" / "high_res"
_SOURCE = {
    1: Path("/home/obihome/Downloads/long_irrigation_cannulas_1mm_2mm/irrigation_cannula_1mm_corrected.obj"),
    2: Path("/home/obihome/Downloads/long_irrigation_cannulas_1mm_2mm/irrigation_cannula_2mm_corrected.obj"),
    6: Path("/home/obihome/Downloads/irrigation_cannulas/irrigation_cannula_6mm_short.obj"),
}


def mesh_path(size_mm: int) -> Path:
    return MESH_DIR / f"cannula_{int(size_mm)}mm.obj"


def load_obj(path: Path):
    verts = []
    faces = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                corners = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for index in range(1, len(corners) - 1):
                    faces.append((corners[0], corners[index], corners[index + 1]))
    if not verts or not faces:
        raise ValueError(f"no mesh in {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def write_obj(path: Path, points, tris) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# stainless irrigation cannula, one material\n")
        for point in np.asarray(points, dtype=np.float64):
            handle.write(f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
        for tri in np.asarray(tris, dtype=np.int32):
            handle.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def _boundary_loops(tris: np.ndarray):
    counts = {}
    for tri in tris:
        corners = (int(tri[0]), int(tri[1]), int(tri[2]))
        for start in range(3):
            a = corners[start]
            b = corners[(start + 1) % 3]
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    adjacent = defaultdict(list)
    for (a, b), count in counts.items():
        if count != 1:
            continue
        adjacent[a].append(b)
        adjacent[b].append(a)
    seen = set()
    loops = []
    for start in list(adjacent):
        if start in seen or len(adjacent[start]) == 0:
            continue
        loop = [start]
        seen.add(start)
        previous = None
        current = start
        while True:
            nxt = [item for item in adjacent[current] if item != previous]
            if not nxt:
                break
            nxt = nxt[0]
            if nxt == start:
                break
            if nxt in seen:
                break
            loop.append(nxt)
            seen.add(nxt)
            previous, current = current, nxt
        if len(loop) >= 6:
            loops.append(loop)
    return loops


def _signed_area(points, loop) -> float:
    y = points[loop, 1]
    z = points[loop, 2]
    return float(0.5 * np.sum(y * np.roll(z, -1) - np.roll(y, -1) * z))


def _stitch_rings(points, distal, proximal, outward: bool):
    """Triangle strip between two ordered rings. New rings fill a long gap."""
    distal = list(distal)
    proximal = list(proximal)
    if len(distal) != len(proximal) or len(distal) < 3:
        raise ValueError(f"ring size {len(distal)} != {len(proximal)}")
    if _signed_area(points, distal) * _signed_area(points, proximal) < 0.0:
        proximal.reverse()
    anchor = points[distal[0]]
    nearest = min(range(len(proximal)), key=lambda index: np.linalg.norm(points[proximal[index]] - anchor))
    proximal = proximal[nearest:] + proximal[:nearest]
    gap = abs(float(points[proximal[0], 0] - points[distal[0], 0]))
    steps = max(1, int(round(gap / (2.0 * MESH_MM))))
    rings = [distal]
    extras = []
    for step in range(1, steps):
        weight = step / steps
        ring = []
        for a, b in zip(distal, proximal):
            ring.append(len(points) + len(extras))
            extras.append((1.0 - weight) * points[a] + weight * points[b])
        rings.append(ring)
    rings.append(proximal)
    if extras:
        points = np.vstack((points, np.asarray(extras, dtype=np.float64)))
    faces = []
    for left, right in zip(rings, rings[1:]):
        count = len(left)
        for index in range(count):
            nxt = (index + 1) % count
            faces.append((left[index], right[index], right[nxt]))
            faces.append((left[index], right[nxt], left[nxt]))
    center = np.mean(points[distal], axis=0)
    sample = np.mean(points[[faces[0][0], faces[0][1], faces[0][2]]], axis=0)
    edge_a = points[faces[0][1]] - points[faces[0][0]]
    edge_b = points[faces[0][2]] - points[faces[0][0]]
    normal = np.cross(edge_a, edge_b)
    away = sample - center
    away[0] = 0.0
    # Outer wall points away from the tube center. Inner wall points into the bore.
    if outward and float(np.dot(normal, away)) < 0.0:
        faces = [(a, c, b) for a, b, c in faces]
    if not outward and float(np.dot(normal, away)) > 0.0:
        faces = [(a, c, b) for a, b, c in faces]
    return points, faces


def extend_clip(points, tris, shift: float):
    """Carry the clip up the shaft and fill the gap with a straight tube."""
    xs = np.unique(np.round(points[:, 0], 5))
    window = xs[(xs > 0.26) & (xs < 0.36)]
    if len(window) >= 2:
        gaps = np.diff(window)
        slot = int(np.argmax(gaps))
        cut = float(0.5 * (window[slot] + window[slot + 1]))
    else:
        cut = 0.30
    distal_vert = points[:, 0] <= cut
    face_distal = distal_vert[tris].all(axis=1)
    face_proximal = ~distal_vert[tris].any(axis=1)
    distal_tris = tris[face_distal]
    proximal_tris = tris[face_proximal]
    distal_loops = [
        loop
        for loop in _boundary_loops(distal_tris)
        if float(points[loop, 0].mean()) > cut - 0.02
    ]
    proximal_loops = [
        loop
        for loop in _boundary_loops(proximal_tris)
        if float(points[loop, 0].mean()) < cut + 0.02
    ]
    if len(distal_loops) < 2 or len(proximal_loops) < 2:
        raise RuntimeError(
            f"expected inner and outer rings at the cut, got {len(distal_loops)} and {len(proximal_loops)}"
        )
    points = np.array(points, copy=True)
    points[~distal_vert, 0] += float(shift)
    def _tube_radius(loop) -> float:
        center = np.mean(points[loop], axis=0)
        return float(np.linalg.norm(points[loop, 1:] - center[1:], axis=1).mean())

    added = []
    order = sorted(distal_loops, key=_tube_radius)
    outer_radius = _tube_radius(order[-1])
    for distal in order:
        match = min(
            proximal_loops,
            key=lambda loop: (
                abs(len(loop) - len(distal)),
                abs(float(points[loop, 1].mean()) - float(points[distal, 1].mean())),
            ),
        )
        proximal_loops.remove(match)
        points, faces = _stitch_rings(points, distal, match, outward=_tube_radius(distal) >= outer_radius - 1e-6)
        added.extend(faces)
        if not proximal_loops:
            break
    kept = np.vstack((distal_tris, proximal_tris))
    tris = np.vstack((kept, np.asarray(added, dtype=np.int32)))
    return points, tris


def _shaft_curve(size_mm: int) -> np.ndarray:
    verts = []
    with (MESH_DIR / f"{int(size_mm)}mm.OBJ").open() as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    shaft = np.asarray(verts, dtype=np.float64)
    order = np.argsort(shaft[:, 0])
    xs = shaft[order, 0]
    curve = []
    start = 0
    for index in range(1, len(xs) + 1):
        if index == len(xs) or xs[index] - xs[start] > 0.0015:
            chosen = order[start:index]
            radius = float(np.median(np.linalg.norm(shaft[chosen, 1:], axis=1)))
            curve.append((float(shaft[chosen, 0].mean()), radius))
            start = index
    return np.asarray(curve, dtype=np.float64)


def _radius_at(curve: np.ndarray, x: np.ndarray) -> np.ndarray:
    station = curve[:, 0]
    radius = curve[:, 1]
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(len(x), dtype=np.float64)
    out[x <= station[0]] = radius[0]
    out[x >= station[-1]] = radius[-1]
    middle = (x > station[0]) & (x < station[-1])
    index = np.searchsorted(station, x[middle])
    span = station[index] - station[index - 1]
    weight = (x[middle] - station[index - 1]) / np.maximum(span, 1e-12)
    out[middle] = (1.0 - weight) * radius[index - 1] + weight * radius[index]
    return out


def lift_off_shaft(points: np.ndarray, size_mm: int, gap_mm: float = 0.4) -> np.ndarray:
    """Slide each tube ring out in +Y until it clears the shaft. The clip stays put."""
    curve = _shaft_curve(size_mm)
    points = np.array(points, copy=True)
    order = np.argsort(points[:, 0])
    xs = points[order, 0]
    groups = []
    start = 0
    for index in range(1, len(xs) + 1):
        if index == len(xs) or xs[index] - xs[start] > 0.001:
            groups.append(order[start:index])
            start = index
    gap = float(gap_mm) * MESH_MM
    raw = np.zeros(len(groups), dtype=np.float64)
    station = np.zeros(len(groups), dtype=np.float64)
    clip = np.zeros(len(groups), dtype=bool)
    for index, chosen in enumerate(groups):
        ring = points[chosen]
        station[index] = float(np.median(ring[:, 0]))
        center = np.median(ring, axis=0)
        outer = float(np.percentile(np.linalg.norm(ring - center, axis=1), 90))
        span = float(ring[:, 2].max() - ring[:, 2].min())
        clip[index] = span > 3.0 * MESH_MM
        if clip[index]:
            continue
        near = float(center[1] - outer)
        need = float(_radius_at(curve, np.array([center[0]]))[0] + gap)
        raw[index] = max(0.0, need - near)
    # Ease into the clip so the C stays fitted on the neck.
    blend = raw.copy()
    clip_at = station[clip]
    if len(clip_at):
        first_clip = float(clip_at.min())
        ramp = 12.0 * MESH_MM
        for index, place in enumerate(station):
            if clip[index]:
                blend[index] = 0.0
                continue
            if place > first_clip - ramp:
                blend[index] *= max(0.0, (first_clip - place) / ramp)
    for chosen, shift in zip(groups, blend):
        if shift > 1e-8:
            points[chosen, 1] += shift
    return points


def install_meshes() -> None:
    """Copy the supplied cannulas in and build the 4 mm tube beside them."""
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    six = None
    for size, source in _SOURCE.items():
        destination = mesh_path(size)
        if source.is_file():
            points, tris = load_obj(source)
        elif destination.is_file():
            points, tris = load_obj(destination)
        else:
            raise FileNotFoundError(source)
        points = lift_off_shaft(points, size)
        write_obj(destination, points, tris)
        if size == 6:
            six = (points, tris)
    if six is None:
        six = load_obj(mesh_path(6))
    shift = (CLIP_MM[4] - CLIP_MM[6]) * MESH_MM
    points, tris = extend_clip(six[0], six[1], shift)
    points = lift_off_shaft(points, 4)
    write_obj(mesh_path(4), points, tris)


def _opening_center(points: np.ndarray) -> np.ndarray:
    distal = float(points[:, 0].min())
    ring = points[points[:, 0] <= distal + 0.15 * MESH_MM]
    if len(ring) < 6:
        ring = points[np.argsort(points[:, 0])[:24]]
    return np.median(ring, axis=0)


def measure(size_mm: int) -> dict:
    points, _tris = load_obj(mesh_path(size_mm))
    opening = _opening_center(points)
    proximal = points[points[:, 0] > 20.0 * MESH_MM]
    radial = np.linalg.norm(proximal[:, 1:], axis=1)
    # The clip is the part that reaches in toward the shaft.
    grip = proximal[int(np.argmin(radial))]
    return {
        "size_mm": int(size_mm),
        "outlet": opening,
        "clip": grip,
        "points": points,
    }


_CACHE = {}


def spec(size_mm: int) -> dict:
    size = int(size_mm)
    if size not in SIZES:
        raise ValueError(f"cannula size must be one of {SIZES}")
    cached = _CACHE.get(size)
    if cached is None:
        opening = measure(size)["outlet"]
        cached = {"outlet": np.asarray(opening, dtype=np.float64)}
        _CACHE[size] = cached
    return cached


def burr_radius(size_mm: int) -> float:
    return 0.5 * float(size_mm) * MM_TO_SIM


def aim_point(outlet, radius: float, inset_mm: float = AIM_INSET_MM) -> np.ndarray:
    """Point on the cannula side of the burr, just inside a geometric tangent."""
    outlet = np.asarray(outlet, dtype=np.float64).reshape(3)
    inset = max(0.0, float(inset_mm)) * MM_TO_SIM
    # Keep the aim near the surface. A fixed inset would bury the jet in a 1 mm burr.
    aim_radius = max(float(radius) - inset, 0.72 * float(radius))
    origin = outlet[:2]
    distance = float(np.linalg.norm(origin))
    if distance <= aim_radius + 1e-8:
        return np.zeros(3, dtype=np.float64)
    perpendicular = np.array((-origin[1], origin[0]), dtype=np.float64)
    reach = float(np.sqrt(max(distance * distance - aim_radius * aim_radius, 0.0)))
    base = (aim_radius * aim_radius) * origin
    offset = aim_radius * reach * perpendicular
    scale = distance * distance
    first = (base + offset) / scale
    second = (base - offset) / scale
    chosen = first if first[1] >= second[1] else second
    return np.array((chosen[0], chosen[1], 0.0), dtype=np.float64)


def local_frame(size_mm: int, inset_mm: float = AIM_INSET_MM) -> dict:
    outlet = np.array(spec(size_mm)["outlet"], dtype=np.float64, copy=True)
    radius = burr_radius(size_mm)
    target = aim_point(outlet, radius, inset_mm)
    emit = target - outlet
    length = float(np.linalg.norm(emit))
    if length < 1e-8:
        emit = np.array((-1.0, 0.0, 0.0), dtype=np.float64)
    else:
        emit = emit / length
    return {"outlet": outlet, "target": target, "emit": emit, "radius": radius}


def cannula_frame(tip, quat, size_mm: int, inset_mm: float = AIM_INSET_MM) -> dict:
    """World nozzle and jet. The nozzle is the mesh opening. The jet aims at the burr."""
    from fluid.outlet import quat_axes

    local = local_frame(size_mm, inset_mm)
    axes = quat_axes(quat)
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    outlet = tip + axes @ local["outlet"]
    target = tip + axes @ local["target"]
    emit = axes @ local["emit"]
    emit = emit / max(float(np.linalg.norm(emit)), 1e-12)
    shaft = np.asarray(axes[:, 0], dtype=np.float64)
    right = np.asarray(axes[:, 2], dtype=np.float64)
    right = right - emit * float(np.dot(right, emit))
    right_length = float(np.linalg.norm(right))
    if right_length < 1e-8:
        right = np.asarray(axes[:, 1], dtype=np.float64)
        right = right - emit * float(np.dot(right, emit))
        right_length = float(np.linalg.norm(right))
    right = right / max(right_length, 1e-12)
    tangent = np.cross(emit, right)
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
    return {
        "tip": tip,
        "outlet": outlet,
        "target": target,
        "shaft": shaft,
        "right": right,
        "radial": right,
        "tangent": tangent,
        "emit": emit,
        "size_mm": int(size_mm),
        "radius": float(local["radius"]),
        "proximal_mm": float(local["outlet"][0] / MESH_MM),
        "radial_mm": float(local["outlet"][1] / MESH_MM),
        "offset_mm": float(np.linalg.norm(local["outlet"]) / MESH_MM),
        "emit_dot_shaft": float(np.dot(emit, shaft)),
        "emit_dot_tangent": float(np.dot(emit, tangent)),
        "emit_dot_right": float(np.dot(emit, right)),
    }
