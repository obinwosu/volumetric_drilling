"""Put every ear in one pose per side, so a saved camera view fits every volume of that side.

Each volume's inner ear is rigidly fitted to its side's reference (Anatomy A = RT143 for right,
Anatomy D = LT143 for left), and the ADF's position and orientation are set from that fit. The
references do not move, so the views saved on them stay right. Voxels are never changed.

    middle_ear_newton/.venv/bin/python scripts/segmentation/align_ear_pose.py            # report only
    middle_ear_newton/.venv/bin/python scripts/segmentation/align_ear_pose.py --apply
    middle_ear_newton/.venv/bin/python scripts/segmentation/align_ear_pose.py --apply ADF/volume_x.yaml

--apply first copies each ADF it changes to resources/intermediate_volumes/adf_before_ear_pose/.
A volume is left alone if it has no full inner ear label, no slices here, or a poor fit.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "expert_drilling"))
import otic_capsule as oc  # noqa: E402
import prepare_specimens as ps  # noqa: E402

BACKUP = REPO / "resources" / "intermediate_volumes" / "adf_before_ear_pose"
SETUP = REPO / "scripts" / "study_gui" / "gui_setup.yaml"
MANIFEST = REPO / "scripts" / "expert_drilling" / "specimens.yaml"
GOOD_FIT_MM = 0.8  # mean surface distance; above it, try every orientation
MAX_FIT_MM = 1.2   # above it, the volume is left where it is
MIN_POINTS = 200


def all_adfs() -> list[pathlib.Path]:
    setup = yaml.safe_load(SETUP.read_text())
    found = [(SETUP.parent / setup[key]["adf_path"]).resolve() for key in setup["volumes"]]
    if MANIFEST.is_file():
        found += [(REPO / s["adf_path"]).resolve() for s in yaml.safe_load(MANIFEST.read_text())["specimens"]]
    return list(dict.fromkeys(found))


def world_ear_points(adf: pathlib.Path):
    """(inner ear surface in world mm, volume dict), or (None, reason)."""
    doc = yaml.safe_load(adf.read_text())
    if not doc.get("volumes"):
        return None, "not a volume"
    block = doc[doc["volumes"][0]]
    if not (adf.parent / block["images"]["path"]).resolve().is_dir():
        return None, "slices not on this machine"
    volume = ps.read_adf_volume(adf)
    if volume["laterality"] not in ("left", "right"):
        return None, "no laterality in the ADF"
    sidecar = oc.load_sidecar(adf)
    if sidecar is not None:
        mask = sidecar["ear"]
    else:
        mask = volume["alpha"] & np.all(volume["rgb"] == np.array(ps.INNER_EAR_RGB, np.uint8), axis=-1)
    if int(mask.sum()) < MIN_POINTS:
        return None, "no full inner ear label"
    local = ps.surface_points(mask, mask.shape, volume["dimensions_au"])
    return local @ volume["rotation"].T + volume["position_au"] * ps.MM_PER_AU, volume


def score(source, target, tree, rotation, translation) -> float:
    moved = source @ rotation.T + translation
    return 0.5 * float(tree.query(moved)[0].mean() + cKDTree(moved).query(target)[0].mean())


def fit(source, target):
    """(mean surface distance in mm, R, t) taking the world points onto the reference."""
    tree = cKDTree(target)
    rotation, translation = ps.icp(source, tree, target, np.eye(3), target.mean(0) - source.mean(0))
    best = score(source, target, tree, rotation, translation)
    if best > GOOD_FIT_MM:
        wide, wide_rotation, wide_translation = ps.fit_labyrinth(source, target)
        if wide < best:
            return wide, wide_rotation, wide_translation
    return best, rotation, translation


def new_pose(volume, rotation, translation):
    matrix = rotation @ volume["rotation"]
    position = rotation @ (volume["position_au"] * ps.MM_PER_AU) + translation
    yaw, pitch, roll = Rotation.from_matrix(matrix).as_euler("ZYX")
    return position / ps.MM_PER_AU, (float(roll), float(pitch), float(yaw))


def write_pose(adf: pathlib.Path, position, rpy) -> None:
    """Only the two numbers-in-braces lines of the volume's location change."""
    text = adf.read_text()
    new_position = "position: {x: %.9f, y: %.9f, z: %.9f}" % tuple(position)
    new_orientation = "orientation: {r: %.9f, p: %.9f, y: %.9f}" % tuple(rpy)
    start = text.index("location:")
    head, tail = text[:start], text[start:]
    tail, moved = re.subn(r"position:\s*\{[^}]*\}", new_position, tail, count=1)
    tail, turned = re.subn(r"(?m)^(\s*)orientation:\s*\{[^}]*\}", lambda m: m.group(1) + new_orientation, tail, count=1)
    if moved != 1 or turned != 1:
        raise ValueError(f"Could not find the pose in {adf}")
    BACKUP.mkdir(parents=True, exist_ok=True)
    if not (BACKUP / adf.name).exists():
        shutil.copy2(adf, BACKUP / adf.name)
    partial = adf.with_suffix(".yaml.part")
    partial.write_text(head + tail)
    partial.replace(adf)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("adfs", nargs="*", help="Only these ADFs (default: every volume)")
    parser.add_argument("--apply", action="store_true", help="Write the new poses")
    args = parser.parse_args(argv)
    references = {}
    for side, name in ps.REFERENCES.items():
        references[side] = world_ear_points(REPO / "ADF" / name)[0]
    reference_files = {(REPO / "ADF" / name).resolve() for name in ps.REFERENCES.values()}
    adfs = [pathlib.Path(a).resolve() for a in args.adfs] or all_adfs()
    print(f"{'volume':46}{'side':7}{'fit mm':>8}{'moved mm':>10}{'turned deg':>12}  result", flush=True)
    for adf in adfs:
        if adf in reference_files:
            print(f"{adf.stem[:45]:46}{'':7}{'':>8}{'':>10}{'':>12}  reference", flush=True)
            continue
        points, volume = world_ear_points(adf)
        if points is None:
            print(f"{adf.stem[:45]:46}{'':7}{'':>8}{'':>10}{'':>12}  left alone: {volume}", flush=True)
            continue
        side = volume["laterality"]
        distance, rotation, translation = fit(points, references[side])
        moved = float(np.linalg.norm(rotation @ points.mean(0) + translation - points.mean(0)))
        turned = float(np.degrees(Rotation.from_matrix(rotation).magnitude()))
        if distance > MAX_FIT_MM:
            result = "left alone: poor fit"
        elif args.apply:
            write_pose(adf, *new_pose(volume, rotation, translation))
            result = "aligned"
        else:
            result = "would align"
        print(f"{adf.stem[:45]:46}{side:7}{distance:8.2f}{moved:10.1f}{turned:12.1f}  {result}", flush=True)


if __name__ == "__main__":
    main()
