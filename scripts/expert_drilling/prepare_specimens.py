#!/usr/bin/env python3
"""Turn the cadaveric specimen segmentations (bone + inner ear) into simulator volumes.

Each Slicer .seg.nrrd becomes PNG slices, an ADF, an icon and an entry in specimens.yaml,
at the scan's own voxel size: nothing is resampled, so a recorded voxel index maps exactly
to a voxel of the source segmentation (``T_lps_voxel`` in specimens.yaml).

The files do not say which side an ear is, and the blocks lie in the scanner at arbitrary
angles. Both come from the inner ear: its shape is chiral, so it is rigidly fitted (rotation
only, no mirroring) to the labyrinth of a known right and a known left anatomy. The better
fit gives the side, and its pose places the specimen where that anatomy lies, so the saved
microscope views look at it from the usual direction.

    python3 scripts/expert_drilling/prepare_specimens.py --source ~/Downloads/ur5_specimens123
    python3 scripts/expert_drilling/prepare_specimens.py --validate   # known anatomies only
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "utils"))
import seg_nrrd_to_volume as conv  # noqa: E402

MANIFEST = HERE / "specimens.yaml"
MM_PER_AU = conv.METERS_PER_AU * 1000.0
BONE_RGB = (255, 249, 219)
INNER_EAR_RGB = (91, 123, 91)  # the tuner's inner_ear material
INNER_EAR_HEX = "5b7b5b"
REFERENCES = {"right": "volume_256_RT143.yaml", "left": "volume_256_LT143.yaml"}
VALIDATION = ("volume_256_RT138.yaml", "volume_256_RT147.yaml", "volume_256_LT151.yaml",
              "volume_left_dehiscence_ct_0p196mm.yaml")
FIT_SPACING_MM = 0.5
ROTATION_SAMPLES = 6000
REFINE_STARTS = 24


# ---------------------------------------------------------------- volumes on disk
def read_adf_volume(adf: pathlib.Path):
    doc = yaml.safe_load(adf.read_text())
    block = doc[doc["volumes"][0]]
    images = block["images"]
    folder = (adf.parent / images["path"]).resolve()
    pattern = re.compile(re.escape(images["prefix"]) + r"(\d+)\." + re.escape(images["format"]) + "$")
    names = sorted((p for p in folder.iterdir() if pattern.match(p.name)),
                   key=lambda p: int(pattern.match(p.name).group(1)))
    first = np.asarray(Image.open(names[0]).convert("RGBA"))
    rgb = np.zeros((first.shape[1], first.shape[0], len(names), 3), np.uint8)
    alpha = np.zeros(rgb.shape[:3], bool)
    for k, path in enumerate(names):
        # Voxel x = PNG column, voxel y = PNG row from the bottom (Chai flips rows).
        image = np.flipud(np.asarray(Image.open(path).convert("RGBA")))
        rgb[:, :, k] = np.transpose(image[:, :, :3], (1, 0, 2))
        alpha[:, :, k] = image[:, :, 3].T > 10
    dims = block["dimensions"]
    location = block["location"]
    return {
        "rgb": rgb, "alpha": alpha,
        "dimensions_au": np.array([dims["x"], dims["y"], dims["z"]], float),
        "position_au": np.array([location["position"][a] for a in "xyz"], float),
        "rotation": euler_to_matrix(*(location["orientation"][a] for a in "rpy")),
        "laterality": block.get("laterality"),
    }


def euler_to_matrix(roll, pitch, yaw):
    """AMBF's ADF orientation: extrinsic x, y, z = Rz(yaw) Ry(pitch) Rx(roll)."""
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def matrix_to_euler(matrix):
    yaw, pitch, roll = Rotation.from_matrix(matrix).as_euler("ZYX")
    return float(roll), float(pitch), float(yaw)


def local_mm(index_xyz, shape, dimensions_au):
    """Voxel index -> volume-local millimetres (surface.index_to_local, in mm)."""
    dims_mm = np.asarray(dimensions_au, float) * MM_PER_AU
    return np.asarray(index_xyz, float) * dims_mm / np.asarray(shape, float) - dims_mm / 2.0


# ---------------------------------------------------------------- labyrinth fit
def surface_points(mask, shape, dimensions_au, spacing=FIT_SPACING_MM, seed=0):
    shell = mask & ~ndi.binary_erosion(mask)
    points = local_mm(np.argwhere(shell) + 0.5, shape, dimensions_au)
    keys = np.unique(np.floor(points / spacing).astype(np.int64), axis=0, return_index=True)[1]
    points = points[keys]
    rng = np.random.default_rng(seed)
    if len(points) > 2500:
        points = points[rng.choice(len(points), 2500, replace=False)]
    return points


def rigid_fit(source, target):
    """Best proper rotation and translation taking source onto paired target points."""
    cs, ct = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, ct - rotation @ cs


def icp(source, tree, target, rotation, translation, iterations=40):
    for _ in range(iterations):
        moved = source @ rotation.T + translation
        distance, nearest = tree.query(moved)
        keep = distance <= max(np.percentile(distance, 90), 1e-6)
        new_rotation, new_translation = rigid_fit(source[keep], target[nearest[keep]])
        change = np.abs(new_rotation - rotation).max() + np.abs(new_translation - translation).max()
        rotation, translation = new_rotation, new_translation
        if change < 1e-5:
            break
    return rotation, translation


def fit_labyrinth(source, target, seed=0):
    """(symmetric mean surface distance in mm, rotation, translation): source -> target."""
    tree = cKDTree(target)
    source_tree_points = source
    center_s, center_t = source.mean(0), target.mean(0)
    rotations = Rotation.random(ROTATION_SAMPLES, random_state=seed).as_matrix()
    subset = source[:: max(1, len(source) // 300)] - center_s
    coarse = np.array([tree.query(subset @ r.T + center_t)[0].mean() for r in rotations])
    best = (np.inf, None, None)
    for index in np.argsort(coarse)[:REFINE_STARTS]:
        start = rotations[index]
        rotation, translation = icp(source, tree, target, start, center_t - start @ center_s)
        forward = tree.query(source @ rotation.T + translation)[0].mean()
        back = cKDTree(source @ rotation.T + translation).query(target)[0].mean()
        score = 0.5 * (forward + back)
        if score < best[0]:
            best = (float(score), rotation, translation)
    del source_tree_points
    return best


def load_references():
    out = {}
    for side, name in REFERENCES.items():
        volume = read_adf_volume(REPO / "ADF" / name)
        mask = volume["alpha"] & np.all(volume["rgb"] == np.array(INNER_EAR_RGB, np.uint8), axis=-1)
        if not mask.any():
            raise SystemExit(f"{name} has no inner ear label #{INNER_EAR_HEX}")
        volume["points"] = surface_points(mask, mask.shape, volume["dimensions_au"])
        out[side] = volume
    return out


def classify(points, references):
    """Both fits, and the winner's side."""
    fits = {side: fit_labyrinth(points, ref["points"]) for side, ref in references.items()}
    side = min(fits, key=lambda key: fits[key][0])
    return side, fits


# ---------------------------------------------------------------- one specimen
def read_specimen(path: pathlib.Path):
    header, data = conv.read_nrrd(path)
    layers, spacing = conv.spatial_layers(header, data)
    segments = conv.read_segments(header)
    bone = next(s for s in segments if s["name"].lower().startswith("bone"))
    ear = next(s for s in segments if s["name"].lower().startswith("inner"))
    bone_mask = layers[bone["layer"]] == bone["label"]
    ear_mask = layers[ear["layer"]] == ear["label"]
    directions = np.array([[float(v) for v in d.split(",")]
                           for d in re.findall(r"\(([^)]*)\)", header["space directions"])])
    origin = np.array([float(v) for v in re.findall(r"\(([^)]*)\)", header["space origin"])[0].split(",")])
    return bone_mask, ear_mask, spacing, directions, origin


def build_specimen(path, number, references, margin=2):
    bone, ear, spacing, directions, origin = read_specimen(path)
    if np.ptp(spacing) > 1e-3:
        raise SystemExit(f"{path.name}: voxels are not isotropic ({spacing})")
    return write_specimen(bone, ear, float(spacing.mean()), directions, origin, f"ur5_specimen{number}",
                          f"Specimen {number}", f"Cadaveric temporal bone {number}", references, path.name,
                          hashlib.sha256(path.read_bytes()).hexdigest(), margin)


def write_specimen(bone, ear, voxel_mm, directions, origin, name, display_name, description, references,
                   source_file, source_sha256, margin=2):
    """Masks on a scan's grid (rows of directions = its axes) -> PNG slices, ADF, icon, manifest entry."""
    # The inner ear wins where the two overlap, so it stays one complete structure.
    labels = np.zeros(bone.shape, np.uint8)
    labels[bone] = 1
    labels[ear] = 2
    occupied = np.argwhere(labels > 0)
    low = np.maximum(occupied.min(0) - margin, 0)
    high = np.minimum(occupied.max(0) + 1 + margin, labels.shape)
    labels = labels[low[0]:high[0], low[1]:high[1], low[2]:high[2]]

    segments = [
        {"id": 1, "name": "bone", "rgb": list(BONE_RGB), "hex": conv.BONE_HEX, "alias": "bone"},
        {"id": 2, "name": "inner_ear", "rgb": list(INNER_EAR_RGB), "hex": INNER_EAR_HEX, "alias": "inner_ear"},
    ]
    rows, cols, count = labels.shape
    # write_slices stores NRRD i as PNG rows and j as columns; Chai reads rows bottom-up, so
    # voxel (x, y, z) = (j, rows - 1 - i, k).
    voxel_shape = (cols, rows, count)
    dims_au = np.array([cols, rows, count], float) * voxel_mm / MM_PER_AU
    i_of = np.array([[0, -1, 0, rows - 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)  # (i,j,k,1) from voxel
    crop = np.eye(4)
    crop[:3, 3] = low
    index_to_lps = np.eye(4)
    index_to_lps[:3, :3] = directions.T
    index_to_lps[:3, 3] = origin
    t_lps_voxel = index_to_lps @ crop @ i_of

    voxel_labels = np.flip(np.transpose(labels, (1, 0, 2)), axis=1)
    ear_points = surface_points(voxel_labels == 2, voxel_shape, dims_au)
    side, fits = classify(ear_points, references)
    score, rotation, translation = fits[side]
    other = fits["left" if side == "right" else "right"][0]
    reference = references[side]
    world_rotation = reference["rotation"] @ rotation
    world_position = reference["rotation"] @ (translation / MM_PER_AU) + reference["position_au"]

    tag = f"{name}_{voxel_mm:.3f}mm".replace(".", "p")
    folder = REPO / "resources" / "volumes" / tag
    adf = REPO / "ADF" / f"volume_{tag}.yaml"
    icon = REPO / "scripts" / "study_gui" / "assets" / f"{name}.png"
    conv.write_slices(labels, segments, folder)
    volume_name = "mastoidectomy_volume_high" if max(labels.shape) > 384 else "mastoidectomy_volume_low"
    conv.write_adf(adf, folder, count, dims_au.tolist(), side, volume_name)
    roll, pitch, yaw = matrix_to_euler(world_rotation)
    text = adf.read_text()
    text = re.sub(r"position: \{[^}]*\}.*", "position: {x: %.9f, y: %.9f, z: %.9f}  # inner ear on the %s reference anatomy's"
                  % (*world_position, side), text, count=1)
    text = re.sub(r"orientation: \{[^}]*\}", "orientation: {r: %.9f, p: %.9f, y: %.9f}" % (roll, pitch, yaw), text, count=1)
    adf.write_text(text)
    conv.write_icon(labels, segments, icon)

    return {
        "name": display_name,
        "description": f"{description}: bone + inner ear, {side} ear, {voxel_mm:.2f} mm voxels.",
        "adf_path": str(adf.relative_to(REPO)),
        "icon_path": str(icon.relative_to(REPO)),
        "laterality": side,
        "laterality_source": "inner ear fit to the reference anatomies; not confirmed by hand",
        "laterality_fit_mm": {side: round(score, 3), ("left" if side == "right" else "right"): round(other, 3)},
        "voxel_mm": voxel_mm,
        "voxel_shape_xyz": [int(v) for v in voxel_shape],
        "voxels": {"bone": int((labels == 1).sum()), "inner_ear": int((labels == 2).sum())},
        "labels": {"bone": conv.BONE_HEX, "inner_ear": INNER_EAR_HEX},
        "source_file": source_file,
        "source_sha256": source_sha256,
        "T_lps_voxel": [[float(v) for v in row] for row in t_lps_voxel],
    }


def load_manifest(path=None) -> list:
    path = pathlib.Path(path or MANIFEST)
    return (yaml.safe_load(path.read_text()) or {}).get("specimens") or [] if path.is_file() else []


def add_to_manifest(entry, path=None):
    """Append one specimen; the file is replaced in one step."""
    path = pathlib.Path(path or MANIFEST)
    specimens = load_manifest(path) + [entry]
    partial = path.with_suffix(".part")
    partial.write_text("# Written by prepare_specimens.py and ur5_ct_worker.py. Listed in the launcher's UR5 tab.\n"
                       + yaml.safe_dump({"specimens": specimens}, sort_keys=False, allow_unicode=True))
    partial.replace(path)


def validate(references):
    """The method on anatomies whose side is known (not the two references)."""
    wrong = 0
    for name in VALIDATION:
        volume = read_adf_volume(REPO / "ADF" / name)
        mask = volume["alpha"] & np.all(volume["rgb"] == np.array(INNER_EAR_RGB, np.uint8), axis=-1)
        points = surface_points(mask, mask.shape, volume["dimensions_au"])
        side, fits = classify(points, references)
        truth = volume["laterality"] or ("right" if "_RT" in name else "left")
        wrong += side != truth
        print(f"{name}: truth {truth}, fit {side} "
              f"(right {fits['right'][0]:.3f} mm, left {fits['left'][0]:.3f} mm)", flush=True)
    print(f"{len(VALIDATION) - wrong}/{len(VALIDATION)} correct", flush=True)
    return wrong


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=lambda p: pathlib.Path(p).expanduser())
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    references = load_references()
    if args.validate:
        return 1 if validate(references) else 0
    if args.source is None:
        parser.error("--source is required")
    specimens = []
    for number, path in enumerate(sorted(args.source.glob("seg_TB_bone*.seg.nrrd")), start=1):
        entry = build_specimen(path, number, references)
        specimens.append(entry)
        print(f"{entry['name']}: {entry['laterality']} {entry['laterality_fit_mm']} mm, "
              f"{entry['voxel_shape_xyz']} voxels, {entry['voxels']}", flush=True)
    MANIFEST.write_text("# Written by prepare_specimens.py. Listed in the launcher's Expert drilling tab.\n"
                        + yaml.safe_dump({"specimens": specimens}, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
