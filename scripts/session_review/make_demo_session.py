"""A synthetic drilling session on a real specimen, for testing the review app without the rig.

A burr sweeps across the bone in strokes, removes what it touches, changes size, and the bone is
reset once. Written through the real recorder, so the files have the real layout.

    python3 scripts/session_review/make_demo_session.py <output folder> [--stereo]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import session_data  # noqa: E402,F401  (finds h5py)

sys.path[:0] = [str(REPO / "scripts" / "expert_drilling"), str(REPO / "scripts" / "study_gui")]
import cv2  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import expert_dataset as dataset  # noqa: E402
import expert_recorder  # noqa: E402

T0 = 1_800_000_000.0
BURRS = ((0.0, 6), (9.0, 4), (16.0, 2))
RESET_AT = 21.0
DURATION = 26.0


def load_volume(adf):
    doc = yaml.safe_load(adf.read_text())
    block = doc[doc["volumes"][0]]
    return expert_recorder.load_rgba_from_adf(adf), block


def build(folder, specimen, stereo=False):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    adf = REPO / specimen["adf_path"]
    rgba, block = load_volume(adf)
    shape = np.array(rgba.shape[:3], float)
    dims = np.array([block["dimensions"][a] for a in "xyz"], float)
    orientation = block["location"]["orientation"]
    rotation = Rotation.from_euler("ZYX", [orientation["y"], orientation["p"], orientation["r"]]).as_matrix()
    position = np.array([block["location"]["position"][a] for a in "xyz"], float)
    meta = {
        "format": "fivrs_expert_drilling_v1", "development": True,
        "participant": {"first_name": "Demo", "last_name": "Session"},
        "specimen": {k: specimen.get(k) for k in ("name", "laterality", "labels", "voxel_mm", "voxel_shape_xyz")},
        "session_number": 1, "started": "2026-11-21T12:00:00",
        "video": {"mode": "stereo" if stereo else "mono"}, "launch": {"starting_burr_mm": 6},
        "volume": {"p_world_volume": position.tolist(), "R_world_volume": rotation.tolist(), "dimensions_sim": dims.tolist()},
    }
    (folder / "session.json").write_text(json.dumps(meta))
    recorder = expert_recorder.SessionRecorder(folder, ("left", "right") if stereo else ("left",))
    recorder.add_epoch(rgba.copy(), 0)
    current = rgba.copy()
    centres = (np.indices(rgba.shape[:3]).reshape(3, -1).T)
    solid_world = None
    camera_position = position + rotation @ np.array([0.0, 0.0, 3.2])
    look = position - camera_position
    look /= np.linalg.norm(look)
    up = rotation[:, 1] - look * np.dot(rotation[:, 1], look)
    up /= np.linalg.norm(up)
    right = np.cross(look, up)
    camera_rotation = np.column_stack([look, -right, up])
    camera_quat = Rotation.from_matrix(camera_rotation).as_quat()
    del centres, solid_world
    occupied = np.argwhere(rgba[..., 3] > dataset.SOLID_ALPHA)
    world = (occupied * dims / shape - dims / 2) @ rotation.T + position
    depth = (world - camera_position) @ look
    top = np.percentile(depth, 4)
    centre = world.mean(0)
    reach = np.percentile(np.linalg.norm((world - centre) - np.outer((world - centre) @ look, look), axis=1), 55)
    alive = np.ones(len(occupied), bool)
    burr = BURRS[0][1]
    voxel_batch = 0
    for k in range(int(DURATION * 1000)):
        t = k / 1000.0
        size = [b for start, b in BURRS if start <= t][-1]
        cutting = (t % 3.2) < 2.3 and t < DURATION - 1.0
        phase = 2 * np.pi * t / 2.3
        tip = centre + right * reach * np.sin(phase) * (0.5 + 0.08 * (t % 5)) + up * reach * 0.55 * np.cos(phase * 0.37)
        tip = tip + look * (top - np.dot(centre - camera_position, look) + (0.012 * t if cutting else -0.06))
        drill_quat = Rotation.from_matrix(np.column_stack([-look, right, -up])).as_quat()
        pose = (T0 + t, T0 + t, *tip, *drill_quat)
        recorder.add("drill", pose)
        magnitude = (0.8 + 0.5 * np.sin(t * 7.0)) if cutting else 0.0
        recorder.add("force", (T0 + t, T0 + t, -look[0] * magnitude, -look[1] * magnitude, -look[2] * magnitude))
        if k % 10 == 0:
            recorder.add("state", expert_recorder.state_row({
                "sim_time": T0 + t, "on": cutting, "size_mm": size, "style": "cutting",
                "voxel_removal_limit": {1: 1, 2: 3, 4: 6, 6: 10}[size], "burr_diameter_sim": size * dataset.BURR_SIM_PER_NOMINAL_MM,
                "px": tip[0], "py": tip[1], "pz": tip[2], "qx": drill_quat[0], "qy": drill_quat[1], "qz": drill_quat[2],
                "qw": drill_quat[3], "cx": camera_position[0], "cy": camera_position[1], "cz": camera_position[2],
                "lx": look[0], "ly": look[1], "lz": look[2], "ux": up[0], "uy": up[1], "uz": up[2], "fov": 0.36}, T0 + t))
            recorder.add("camera", (T0 + t, T0 + t, *camera_position, *camera_quat))
        if size != burr:
            burr = size
            recorder.add("burr", (T0 + t, T0 + t, float(size)))
        if k == int(RESET_AT * 1000):
            current = rgba.copy()
            alive[:] = True
            recorder.add_epoch(rgba.copy(), recorder.batch_count())
        if cutting and k % 16 == 0:
            radius = size * dataset.BURR_SIM_PER_NOMINAL_MM * 1.6
            hit = alive & (np.sum((world - tip) ** 2, axis=1) <= radius * radius)
            indices = occupied[hit]
            if len(indices):
                alive[hit] = False
                current[indices[:, 0], indices[:, 1], indices[:, 2]] = 0
                recorder.on_voxels(T0 + t, float(voxel_batch), indices.tolist())
                voxel_batch += 1
        if k % 33 == 0:
            image = np.full((360, 640, 3), 18, np.uint8)
            cv2.putText(image, f"t = {t:6.2f} s  burr {size} mm", (24, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (210, 225, 235), 2)
            images = [image, cv2.flip(image, 1)] if stereo else [image]
            recorder.frame(T0 + t, T0 + t, (*camera_position, *camera_quat), 0.36, 3.2, images)
    recorder.add_final(current)
    recorder.close()
    meta = json.loads((folder / "session.json").read_text())
    meta["video_result"] = {"frames": recorder.video.frames, "frames_dropped": recorder.video.dropped}
    (folder / "session.json").write_text(json.dumps(meta))
    return dataset.finalize(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output")
    parser.add_argument("--stereo", action="store_true")
    args = parser.parse_args()
    specimens = yaml.safe_load((REPO / "scripts/expert_drilling/specimens.yaml").read_text())["specimens"]
    print(build(args.output, specimens[0], args.stereo))
