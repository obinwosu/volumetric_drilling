"""Build the ear canal skin sheet from a FIVRS volume's EAC label.

The EAC label is a solid plug. It loads as air (``rtx_view/air_labels.py``),
and its wall is where the skin lies. This script meshes that wall, tags each
vertex by what it lies on, and writes the sheet for the soft-tissue work.

Regions
  bony       on bone. Anchored by peel bonds.
  annulus    skin within ``ANNULUS_MM`` of the medial edge. Stronger bonds.
  bony_gap   inside the bony canal with no bone underneath. Free.
  cartilage  lateral canal with no bone under it (cartilage is not segmented). Pinned.
Dropped: the medial drum face, the lateral opening, and wall on other labels.

The annulus is the sheet's medial edge loop: the medial rim of the EAC
segmentation, where the canal wall meets the drum face. Its size and tilt
are reported. Skin is ``thickness`` thick (1 mm by default) and tapers to
``ANNULUS_EDGE_MM`` at that edge over ``TAPER_MM``. Thickness is also capped at
``CLEARANCE_SHARE`` of the distance to the opposite wall, so the two sides of a
narrow wedge never meet.

The attachment surface is the canal wall. The lumen surface is that wall
moved inward by the skin thickness. Millimetre coordinates are in voxel index
space (x column, y row from the bottom, z slice), the indexing of
``fivrs_rtx_view._load_occupancy``. ``verts_world_au`` is the AMBF world frame.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/canal_skin.py ADF/volume_256_RT143.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import yaml
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "rtx_view"))
sys.path.insert(0, str(HERE.parent))

import surface  # noqa: E402

AU_MM = 49.664
EAC_RGB = (0xFF, 0xE1, 0xD6)
BONE_RGB = (0xFF, 0xF9, 0xDB)
OSSICLE_NAMES = ("malleus", "incus", "stapes")

GAP_MM = 0.4  # empty voxels this close to both EAC and bone count as canal
TOUCH_MM = 0.3  # a wall vertex this close to bone lies on bone
BAND_FRACTION = 0.5  # bony canal: axial stretch where at least this much wall is on bone
CAP_COS = 0.6  # wall facing along the axis more than this is the drum or the opening
ANNULUS_MM = 1.0
EDGE_MM = 0.4  # target triangle edge
DEFAULT_THICKNESS_MM = {"bony": 1.0, "annulus": 1.0, "bony_gap": 1.0, "cartilage": 1.0}
ANNULUS_EDGE_MM = 0.1  # about the drum's thickness
TAPER_MM = 4.0
CLEARANCE_SHARE = 0.45  # of the distance to the opposite wall, per side

REGIONS = ("bony", "annulus", "bony_gap", "cartilage")
REGION_RGB = {
    "bony": (205, 140, 120),
    "annulus": (230, 170, 40),
    "bony_gap": (120, 160, 220),
    "cartilage": (150, 200, 140),
}
_DROP = 255


def _is_rgb(colors, rgb):
    return (colors[..., 0] == rgb[0]) & (colors[..., 1] == rgb[1]) & (colors[..., 2] == rgb[2])


def _ossicle_rgbs():
    path = HERE.parent / "rtx_view" / "tuner_materials.json"
    names = json.loads(path.read_text()).get("names", {})
    return [
        tuple(int(hex_id[i : i + 2], 16) for i in (0, 2, 4))
        for hex_id, name in names.items()
        if name and any(key in name for key in OSSICLE_NAMES)
    ]


def _volume_pose(adf: pathlib.Path):
    doc = yaml.safe_load(adf.read_text())
    volume = next(v for k, v in doc.items() if isinstance(k, str) and k.startswith("VOLUME"))
    location = volume.get("location") or {}
    position = location.get("position") or {}
    orientation = location.get("orientation") or {}
    dims = volume.get("dimensions") or {}
    translation = np.array([float(position.get(a, 0.0)) for a in "xyz"])
    roll, pitch, yaw = (float(orientation.get(a, 0.0)) for a in ("r", "p", "y"))
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    dimensions = np.array([float(dims.get(a, 1.0)) for a in "xyz"])
    return dimensions, rotation, translation, volume.get("laterality", "")


def _canal_axis(eac, mm, ossicles):
    """Centre and unit axis of the EAC, pointing medial to lateral."""
    pts = np.argwhere(eac) * mm
    center = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts[:: max(1, len(pts) // 50000)] - center, full_matrices=False)
    axis = vt[0]
    if not ossicles.any():
        raise SystemExit("No ossicle labels in this volume, so the medial end of the canal is unknown.")
    toward_ossicles = (np.argwhere(ossicles) * mm).mean(axis=0) - center
    if toward_ossicles @ axis > 0:
        axis = -axis
    return center, axis


def _bony_band(t, on_bone, step=1.0):
    """Axial range of the bony canal: the stretch around the most bone-covered
    1 mm bin where at least BAND_FRACTION of the wall lies on bone."""
    edges = np.arange(t.min(), t.max() + step, step)
    total = np.histogram(t, edges)[0].astype(float)
    bone = np.histogram(t[on_bone], edges)[0].astype(float)
    frac = np.where(total > 0, bone / np.maximum(total, 1), 0.0)
    frac = np.convolve(frac, np.ones(3) / 3, mode="same")
    peak = int(np.argmax(frac))
    lo = hi = peak
    while lo > 0 and frac[lo - 1] >= BAND_FRACTION:
        lo -= 1
    while hi < len(frac) - 1 and frac[hi + 1] >= BAND_FRACTION:
        hi += 1
    return float(edges[lo]), float(edges[hi + 1]), frac


def _largest_component(tris, n_verts):
    if len(tris) == 0:
        return tris
    rows = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    cols = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_verts))
    _n, labels = csgraph.connected_components(graph, directed=False)
    tri_label = labels[tris[:, 0]]
    keep = np.bincount(tri_label).argmax()
    return tris[tri_label == keep]


def _compact(verts, tris):
    used = np.unique(tris)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris].astype(np.int32), used


def _mesh_quality(verts, tris):
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    edges = np.concatenate([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)])

    def angle(p, q, r):
        u, v = q - p, r - p
        cos = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    min_angle = np.minimum(np.minimum(angle(a, b, c), angle(b, c, a)), angle(c, a, b))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    return {
        "triangles": int(len(tris)),
        "vertices": int(len(verts)),
        "area_mm2": round(float(area), 1),
        "edge_mm_p5_p50_p95": [round(float(x), 3) for x in np.percentile(edges, [5, 50, 95])],
        "min_angle_deg_p5_p50": [round(float(x), 1) for x in np.percentile(min_angle, [5, 50])],
    }


def load_volume(adf: pathlib.Path):
    from fivrs_rtx_view import _load_occupancy

    occupied, colors = _load_occupancy(adf, keep_air_labels=True)
    dimensions, rotation, translation, laterality = _volume_pose(adf)
    mm = dimensions * AU_MM / np.asarray(occupied.shape, dtype=np.float64)
    return occupied, colors, mm, (dimensions, rotation, translation), laterality


def _boundary_loops(tris):
    """Closed boundary loops of a triangle mesh, as vertex index lists."""
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(edges, axis=0, return_counts=True)
    border = uniq[count == 1]
    neighbors = {}
    for a, b in border:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    used = set()
    loops = []
    for start in neighbors:
        if start in used:
            continue
        loop = [start]
        used.add(start)
        prev, cur = None, start
        while True:
            nxt = [n for n in neighbors[cur] if n != prev and (n not in used or (n == start and len(loop) > 2))]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            loop.append(cur)
            used.add(cur)
        if len(loop) > 2:
            loops.append(loop)
    return loops


def _loop_metrics(points, axis):
    """Size of the annulus loop, measured in its best-fit plane.

    The voxel rim zig-zags, so its path length overstates the ring. The
    perimeter is the convex outline in the plane; the diameter comes from the
    enclosed area.
    """
    from scipy.spatial import ConvexHull

    center = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - center, full_matrices=False)
    xy = (points - center) @ vt[:2].T
    area = 0.5 * abs(float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])))
    hull = ConvexHull(xy)
    extents = np.ptp(xy, axis=0)
    tilt = math.degrees(math.acos(min(1.0, abs(float(vt[2] @ axis)))))
    return {
        "perimeter_mm": round(float(hull.area), 2),
        "area_mm2": round(area, 1),
        "diameter_from_area_mm": round(2.0 * math.sqrt(area / math.pi), 2),
        "long_and_short_axis_mm": [round(float(x), 2) for x in extents],
        "plane_tilt_to_canal_axis_deg": round(tilt, 1),
    }


def _wall_clearance(verts, tris, normals):
    """Distance from each vertex, inward along its normal, to the sheet's opposite side."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(verts.astype(np.float32)), o3d.core.Tensor(tris.astype(np.uint32))
    )
    origins = verts - normals * 0.02
    rays = np.concatenate([origins, -normals], axis=1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
    return hit + 0.02


def build_sheet(
    occupied,
    colors,
    mm,
    thickness=None,
    edge_mm=EDGE_MM,
    smooth_iterations=10,
    taper_mm=TAPER_MM,
    annulus_edge_mm=ANNULUS_EDGE_MM,
):
    """Return the sheet arrays and a summary dict. Coordinates in mm (voxel index space)."""
    import open3d as o3d

    thickness = dict(DEFAULT_THICKNESS_MM if thickness is None else thickness)
    mm = np.asarray(mm, dtype=np.float64)
    eac = _is_rgb(colors, EAC_RGB) & occupied
    if not eac.any():
        raise SystemExit("This volume has no EAC label (#ffe1d6).")
    bone = _is_rgb(colors, BONE_RGB) & occupied
    other = occupied & ~eac & ~bone
    ossicles = np.zeros_like(occupied)
    for rgb in _ossicle_rgbs():
        ossicles |= _is_rgb(colors, rgb) & occupied
    center, axis = _canal_axis(eac, mm, ossicles)

    # Crop around the canal. Close one- and two-voxel gaps between label and bone.
    reach = int(math.ceil(GAP_MM / mm.min()))
    lo = np.maximum(np.argwhere(eac).min(axis=0) - reach - 4, 0)
    hi = np.minimum(np.argwhere(eac).max(axis=0) + reach + 5, np.asarray(occupied.shape))
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    eac_c, bone_c, other_c, occ_c = eac[box], bone[box], other[box], occupied[box]
    gap = ~occ_c & ndimage.binary_dilation(eac_c, iterations=reach) & ndimage.binary_dilation(bone_c, iterations=reach)
    canal = eac_c | gap
    dist_bone = ndimage.distance_transform_edt(~bone_c, sampling=mm)
    dist_other = ndimage.distance_transform_edt(~other_c, sampling=mm)

    verts_i, tris, *_rest = surface.marching_cubes(canal)
    normals_i = surface.vertex_normals(verts_i, tris)  # out of the canal, toward the wall
    probe = np.clip(np.rint(verts_i + 0.75 * normals_i).astype(np.int64), 0, np.asarray(canal.shape) - 1)
    on_bone = dist_bone[probe[:, 0], probe[:, 1], probe[:, 2]] <= TOUCH_MM
    on_other = ~on_bone & (dist_other[probe[:, 0], probe[:, 1], probe[:, 2]] <= TOUCH_MM)

    verts_mm = (verts_i + lo) * mm
    normals_mm = normals_i * mm
    normals_mm /= np.maximum(np.linalg.norm(normals_mm, axis=1, keepdims=True), 1e-12)
    t = (verts_mm - center) @ axis
    along = normals_mm @ axis
    t_med, t_lat, profile = _bony_band(t, on_bone)

    region = np.full(len(verts_mm), _DROP, dtype=np.uint8)
    air = ~on_bone & ~on_other
    # The drum faces medially, but so can a ledge in the label further out.
    # Only the medial half of the canal can hold the drum.
    medial_half = t < 0.5 * (t.min() + t.max())
    drum = air & ((t < t_med) | ((along < -CAP_COS) & medial_half))
    opening = air & (t > t_lat) & (along > CAP_COS)
    region[on_bone] = REGIONS.index("bony")
    region[air & (t >= t_med) & (t <= t_lat) & ~drum] = REGIONS.index("bony_gap")
    region[air & (t > t_lat) & ~drum & ~opening] = REGIONS.index("cartilage")

    kept = region != _DROP
    tris = tris[kept[tris].all(axis=1)]
    tris = _largest_component(tris, len(verts_mm))
    raw_verts, raw_tris, used = _compact(verts_mm, tris)
    raw_region = region[used]
    raw_area = _mesh_quality(raw_verts, raw_tris)["area_mm2"]

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(raw_verts), o3d.utility.Vector3iVector(raw_tris))
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    smoothed = np.asarray(mesh.vertices).copy()
    target = int(raw_area / (math.sqrt(3) / 4 * edge_mm**2))
    if target < len(raw_tris):
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices).copy()
    tris = np.asarray(mesh.triangles).astype(np.int32)
    tris = _largest_component(tris, len(verts))
    verts, tris, _used = _compact(verts, tris)
    region = raw_region[cKDTree(smoothed).query(verts)[1]]

    normals = surface.vertex_normals(verts, tris)
    radial = (verts - center) - np.outer((verts - center) @ axis, axis)
    if np.mean(np.einsum("ij,ij->i", normals, radial)) < 0:
        tris = tris[:, [0, 2, 1]].copy()
        normals = -normals

    # Annulus: the medial edge loop, the rim of the EAC segmentation at the drum.
    loops = _boundary_loops(tris)
    if not loops:
        raise SystemExit("The canal sheet has no open edge, so the annulus cannot be placed.")
    loop_t = [float(((verts[loop] - center) @ axis).mean()) for loop in loops]
    loop_len = [len(loop) for loop in loops]
    medial_t = min(loop_t)
    # The most medial loop of any real size (small holes in the sheet are skipped).
    candidates = [i for i in range(len(loops)) if loop_len[i] >= 0.25 * max(loop_len)]
    annulus_loop = loops[min(candidates, key=lambda i: loop_t[i])]
    annulus_pts = verts[annulus_loop]
    annulus_size = _loop_metrics(annulus_pts, axis)
    to_annulus = cKDTree(annulus_pts).query(verts)[0]
    near = (to_annulus <= ANNULUS_MM) & np.isin(region, [REGIONS.index("bony"), REGIONS.index("bony_gap")])
    region[near] = REGIONS.index("annulus")

    full = np.array([thickness[REGIONS[r]] for r in region])
    s = np.clip(to_annulus / max(taper_mm, 1e-6), 0.0, 1.0)
    thick = annulus_edge_mm + (full - annulus_edge_mm) * (s * s * (3.0 - 2.0 * s))
    clearance = _wall_clearance(verts, tris, normals)
    cap = CLEARANCE_SHARE * clearance
    capped = thick > cap
    thick = np.where(capped, cap, thick)
    lumen = verts - normals * thick[:, None]

    # Lumen fit. Inscribed radius of the canal per 1 mm along the axis, and
    # whether each lumen point stays inside the canal.
    dist_inside = ndimage.distance_transform_edt(canal, sampling=mm)
    canal_pts = np.argwhere(canal)
    canal_t = ((canal_pts + lo) * mm - center) @ axis
    radius_by_t = {}
    for start in np.arange(math.floor(t_med), math.ceil(t_lat)):
        sel = (canal_t >= start) & (canal_t < start + 1)
        if sel.any():
            radius_by_t[float(start)] = float(dist_inside[tuple(canal_pts[sel].T)].max())
    lumen_idx = np.rint(lumen / mm - lo).astype(np.int64)
    inside_box = np.all((lumen_idx >= 0) & (lumen_idx < np.asarray(canal.shape)), axis=1)
    crossed = np.ones(len(lumen), dtype=bool)
    ok = inside_box.copy()
    crossed[ok] = ~canal[tuple(lumen_idx[ok].T)]

    counts = {name: int((region == i).sum()) for i, name in enumerate(REGIONS)}
    min_radius = min(radius_by_t.values()) if radius_by_t else float("nan")
    summary = {
        "annulus": {
            **annulus_size,
            "loop_vertices": len(annulus_loop),
            "open_edges": len(loops),
            "medial_most_loop_mm_from_center": round(medial_t, 2),
        },
        "thickness": {
            "full_mm": thickness,
            "annulus_edge_mm": annulus_edge_mm,
            "taper_mm": taper_mm,
            "clearance_share": CLEARANCE_SHARE,
            "capped_by_clearance": int(capped.sum()),
            "mm_p5_p50_p95": [round(float(x), 3) for x in np.percentile(thick, [5, 50, 95])],
        },
        "voxel_mm": [round(float(x), 4) for x in mm],
        "axis_medial_to_lateral": [round(float(x), 4) for x in axis],
        "canal_center_mm": [round(float(x), 3) for x in center],
        "bony_canal_mm_from_center": [round(t_med, 2), round(t_lat, 2)],
        "bony_canal_length_mm": round(t_lat - t_med, 2),
        "wall_area_before_smoothing_mm2": raw_area,
        "mesh": _mesh_quality(verts, tris),
        "region_vertices": counts,
        "bony_canal_min_inscribed_radius_mm": round(min_radius, 2),
        "inscribed_radius_by_mm": {f"{k:.0f}": round(v, 2) for k, v in radius_by_t.items()},
        "lumen_points_outside_canal": int(crossed.sum()),
        "lumen_points_outside_canal_by_region": {
            name: int((crossed & (region == i)).sum()) for i, name in enumerate(REGIONS)
        },
    }
    sheet = {
        "verts_mm": verts,
        "tris": tris,
        "normals_mm": normals,
        "region": region,
        "thickness_mm": thick,
        "lumen_mm": lumen,
        "axial_mm": (verts - center) @ axis,
        "axis": axis,
        "annulus_loop": np.asarray(annulus_loop, dtype=np.int32),
        "clearance_mm": clearance,
        "center_mm": center,
    }
    return sheet, summary


def _write_ply(path, verts, tris, rgb):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64) / 255.0)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adf", type=pathlib.Path, help="Volume ADF, e.g. ADF/volume_256_RT143.yaml")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Output folder (default resources/soft_tissue/<adf stem>)")
    parser.add_argument("--thickness", type=float, default=None, help="Skin thickness in mm for every region")
    parser.add_argument("--edge", type=float, default=EDGE_MM, help="Target triangle edge in mm")
    parser.add_argument("--taper", type=float, default=TAPER_MM, help="Taper length from the annulus in mm")
    parser.add_argument("--annulus-edge", type=float, default=ANNULUS_EDGE_MM, help="Skin thickness at the annulus in mm")
    args = parser.parse_args()

    adf = args.adf.resolve()
    out = args.out or REPO / "resources" / "soft_tissue" / adf.stem
    out.mkdir(parents=True, exist_ok=True)
    thickness = None if args.thickness is None else {name: args.thickness for name in REGIONS}

    print(f"Loading {adf.name}", flush=True)
    occupied, colors, mm, (dimensions, rotation, translation), laterality = load_volume(adf)
    sheet, summary = build_sheet(
        occupied,
        colors,
        mm,
        thickness=thickness,
        edge_mm=args.edge,
        taper_mm=args.taper,
        annulus_edge_mm=args.annulus_edge,
    )
    index = sheet["verts_mm"] / mm
    world = surface.local_to_world(surface.index_to_local(index, dimensions, occupied.shape), rotation, translation)
    summary = {"volume": adf.name, "laterality": laterality, **summary}

    np.savez_compressed(
        out / "canal_skin.npz",
        verts_world_au=world,
        verts_index=index,
        region_names=np.array(REGIONS),
        **sheet,
    )
    rgb = np.array([REGION_RGB[REGIONS[r]] for r in sheet["region"]])
    _write_ply(out / "canal_skin.ply", sheet["verts_mm"], sheet["tris"], rgb)
    _write_ply(out / "canal_lumen.ply", sheet["lumen_mm"], sheet["tris"], rgb)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
