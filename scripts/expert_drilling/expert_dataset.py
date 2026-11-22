"""One recorded expert drilling session: layout, post-processing and replay.

A session folder holds
    session.json       who, which specimen, session number, launch settings, result of the check
    ground_truth.h5    every simulator stream (below)
    video.h5           the microscope view: JPEG per frame, "left" (mono) or "left" + "right" (stereo)

ground_truth.h5
    streams/<name>     float64 table, column names in the "columns" attribute. t_sim is the
                       simulator clock, t_wall the computer clock when the sample arrived.
        drill                    t_sim t_wall px py pz qx qy qz qw (world, simulator units), each physics step
        drill_rigid_body, camera, suction   same columns, from AMBF's State topics (about 50 Hz)
        force                    t_sim t_wall fx fy fz (force sent to the Touch)
        burr                     t_sim t_wall size_mm, one row per size change the plugin published
        reposition               t_sim t_wall, pose before (7), pose after (7)
        state                    the plugin's state each graphics frame: drill on, clutches,
                                 microscope button, zoom, camera look/up/fov, stylus position...
        voxel_batches            t_sim t_wall batch_id first count (into voxels/indices)
    voxels/indices     (N, 3) voxel x y z, in removal order
    voxels/label       per voxel: 0 unknown, 1 bone, 2 inner ear
    voxels/cause       per voxel: 0 drill, 1 detached piece the simulator cleared
    volume/epoch_<k>/rgba   the whole volume as AMBF held it when epoch k began (x, y, z, RGBA);
                            attrs first_batch, t_sim. A bone reset starts a new epoch.
    volume/inner_ear        UR5 specimens: the hollow inner ear (x, y, z), empty space in the voxels
    volume/final_rgba       AMBF's volume when the recording stopped, to check the replay against
    haptic/json, haptic/t_wall  full Touch state (joints, gimbal, buttons...) when the monitor feed runs
    derived/drill      t_sim, velocity, speed, acceleration and angular velocity, in mm and rad
    derived/burr_timeline  the burr in use: first row = the session's starting burr, then one row
                       per change of size, type or removal limit
    derived/strokes    one row per stroke (a run of removal with gaps under STROKE_GAP_S, on one burr)
    voxels/burr_mm, voxels/burr_diamond   the burr that was active when each voxel was removed

Poses, voxel indices and T_world_volume are enough to rebuild the scene at any time:
``ExpertSession.occupancy_at(t)`` gives the exact voxel state.
"""
from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np

MM_PER_SIM = 49.664            # the volumes' scale: 1 simulator unit = 49.664 mm of anatomy
BURR_SIM_PER_NOMINAL_MM = 0.01007  # the plugin's m_units_mmToSim
STROKE_GAP_S = 0.25
DETACHED_MARGIN_VOXELS = 3.0
SOLID_ALPHA = 10

DRILL_COLUMNS = ("t_sim", "t_wall", "px", "py", "pz", "qx", "qy", "qz", "qw")
STROKE_COLUMNS = ("t_start", "t_end", "duration_s", "voxels", "inner_ear_voxels", "path_mm",
                  "mean_speed_mm_s", "mean_force_n", "max_force_n", "burr_mm", "burr_diamond", "first_batch", "last_batch")
BURR_COLUMNS = ("t_sim", "t_wall", "size_mm", "diamond", "voxel_removal_limit", "diameter_sim")


def burr_diameter_mm(nominal_mm: float) -> float:
    """The burr's size against the anatomy: the plugin and the volumes use different scales."""
    return float(nominal_mm) * BURR_SIM_PER_NOMINAL_MM * MM_PER_SIM


def table(handle, name):
    """(array, {column: index}) of one stream; empty if it was never written."""
    key = f"streams/{name}"
    if key not in handle:
        return np.zeros((0, 2)), {}
    data = handle[key]
    return data[()], {c: i for i, c in enumerate(data.attrs["columns"].split(","))}


def quat_to_matrix(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3))
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def drill_kinematics(drill: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Velocity, acceleration and angular velocity from the drill poses (paused samples removed)."""
    columns = ["t_sim", "vx_mm_s", "vy_mm_s", "vz_mm_s", "speed_mm_s", "ax_mm_s2", "ay_mm_s2", "az_mm_s2",
               "wx_rad_s", "wy_rad_s", "wz_rad_s"]
    if len(drill) < 3:
        return np.zeros((0, len(columns))), columns
    keep = np.concatenate([[True], np.diff(drill[:, 0]) > 0])
    t, position, quat = drill[keep, 0], drill[keep, 2:5] * MM_PER_SIM, drill[keep, 5:9].copy()
    if len(t) < 3:
        return np.zeros((0, len(columns))), columns
    for k in range(1, len(quat)):  # q and -q are the same rotation
        if np.dot(quat[k], quat[k - 1]) < 0:
            quat[k] = -quat[k]
    velocity = np.gradient(position, t, axis=0)
    acceleration = np.gradient(velocity, t, axis=0)
    dq = np.gradient(quat, t, axis=0)
    x, y, z, w = quat.T
    dx, dy, dz, dw = dq.T
    omega = 2 * np.stack([w * dx - x * dw - y * dz + z * dy,
                          w * dy + x * dz - y * dw - z * dx,
                          w * dz - x * dy + y * dx - z * dw], axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    return np.column_stack([t, velocity, speed, acceleration, omega]), columns


def voxel_world(indices, shape, dimensions, rotation, position):
    dims = np.asarray(dimensions, float)
    local = np.asarray(indices, float) * dims / np.asarray(shape, float) - dims / 2.0
    return local @ np.asarray(rotation).T + np.asarray(position)


def label_of(rgba, hexes: dict) -> np.ndarray:
    """Label id of each RGBA row: 1 bone, 2 inner ear, 0 anything else."""
    packed = (rgba[..., 0].astype(np.uint32) << 16) | (rgba[..., 1].astype(np.uint32) << 8) | rgba[..., 2]
    out = np.zeros(packed.shape, np.uint8)
    out[packed == int(hexes["bone"], 16)] = 1
    ear = hexes["inner_ear"]  # one color (UR5) or several (the Main tab's anatomies)
    out[np.isin(packed, [int(h, 16) for h in ([ear] if isinstance(ear, str) else ear)])] = 2
    return out


def _epochs(handle):
    names = sorted((n for n in handle["volume"] if n.startswith("epoch_")), key=lambda n: int(n.split("_")[1]))
    return [(name, int(handle["volume"][name].attrs["first_batch"])) for name in names]


def replay_final(handle) -> np.ndarray:
    """Solid mask at the end of the recording, from the last epoch's volume and its removals."""
    batches, col = table(handle, "voxel_batches")
    name, first_batch = _epochs(handle)[-1]
    solid = handle["volume"][name]["rgba"][..., 3] > SOLID_ALPHA
    if len(batches) > first_batch:
        start = int(batches[first_batch, col["first"]])
        indices = handle["voxels/indices"][start:].astype(np.intp)
        solid[indices[:, 0], indices[:, 1], indices[:, 2]] = False
    return solid


def burr_timeline(state, state_col, burr, starting_mm) -> np.ndarray:
    """The burr in use over the session: its first row, then a row at each change."""
    if len(state) and "voxel_removal_limit" in state_col:
        keys = ("size_mm", "burr_diamond", "voxel_removal_limit", "burr_diameter_sim")
        rows = state[np.isfinite(state[:, state_col["t_sim"]])]
        if len(rows):
            values = rows[:, [state_col[k] for k in keys]]
            changed = np.concatenate([[True], np.any(values[1:] != values[:-1], axis=1)])
            return np.column_stack([rows[changed, state_col["t_sim"]], rows[changed, state_col["t_wall"]], values[changed]])
    if len(burr):
        return np.column_stack([burr[:, 0], burr[:, 1], burr[:, 2], np.zeros(len(burr)),
                                np.full(len(burr), np.nan), burr[:, 2] * BURR_SIM_PER_NOMINAL_MM])
    return np.array([[0.0, 0.0, float(starting_mm), 0.0, np.nan, float(starting_mm) * BURR_SIM_PER_NOMINAL_MM]])


def burr_slot(timeline, times) -> np.ndarray:
    """Timeline row in use at each time (the first row also covers anything before it)."""
    return np.clip(np.searchsorted(timeline[:, 0], times, side="right") - 1, 0, len(timeline) - 1)


def find_strokes(batches, col, labels, drill_kin, drill, force, timeline):
    rows = []
    if not len(batches):
        return np.zeros((0, len(STROKE_COLUMNS)))
    times = batches[:, col["t_sim"]]
    slots = burr_slot(timeline, times)
    burr_of = timeline[slots][:, [2, 3]]
    breaks = (np.diff(times) > STROKE_GAP_S) | np.any(burr_of[1:] != burr_of[:-1], axis=1)
    starts = np.concatenate([[0], np.flatnonzero(breaks) + 1])
    ends = np.concatenate([starts[1:], [len(times)]])
    for a, b in zip(starts, ends):
        t0, t1 = float(times[a]), float(times[b - 1])
        first = int(batches[a, col["first"]])
        last = int(batches[b - 1, col["first"]] + batches[b - 1, col["count"]])
        path = mean_speed = 0.0
        if len(drill):
            inside = (drill[:, 0] >= t0) & (drill[:, 0] <= t1)
            if inside.sum() > 1:
                path = float(np.linalg.norm(np.diff(drill[inside, 2:5] * MM_PER_SIM, axis=0), axis=1).sum())
        if len(drill_kin):
            inside = (drill_kin[:, 0] >= t0) & (drill_kin[:, 0] <= t1)
            if inside.any():
                mean_speed = float(drill_kin[inside, 4].mean())
        mean_force = max_force = 0.0
        if len(force):
            inside = (force[:, 0] >= t0) & (force[:, 0] <= t1)
            if inside.any():
                magnitude = np.linalg.norm(force[inside, 2:5], axis=1)
                mean_force, max_force = float(magnitude.mean()), float(magnitude.max())
        rows.append((t0, t1, t1 - t0, last - first, int((labels[first:last] == 2).sum()), path, mean_speed,
                     mean_force, max_force, burr_of[a, 0], burr_of[a, 1], int(batches[a, col["batch_id"]]),
                     int(batches[b - 1, col["batch_id"]])))
    return np.asarray(rows, float)


def finalize(folder) -> dict:
    """Labels, causes, kinematics, strokes and the replay check. Safe to run again."""
    folder = pathlib.Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    with h5py.File(folder / "ground_truth.h5", "r+") as f:
        batches, col = table(f, "voxel_batches")
        drill, _ = table(f, "drill")
        if not len(drill):  # a plugin built before the physics-rate pose topic
            drill, _ = table(f, "drill_rigid_body")
        force, _ = table(f, "force")
        burr, _ = table(f, "burr")
        state, state_col = table(f, "state")
        timeline = burr_timeline(state, state_col, burr, meta["launch"].get("starting_burr_mm", 4))
        total = int(f["voxels/indices"].shape[0]) if "voxels/indices" in f else 0
        labels = np.zeros(total, np.uint8)
        causes = np.zeros(total, np.uint8)
        epochs = _epochs(f) if "volume" in f else []
        volume = meta["volume"]
        rotation, position = np.asarray(volume["R_world_volume"]), np.asarray(volume["p_world_volume"])
        for number, (name, first_batch) in enumerate(epochs):
            stop_batch = epochs[number + 1][1] if number + 1 < len(epochs) else len(batches)
            if stop_batch <= first_batch or first_batch >= len(batches):
                continue
            start = int(batches[first_batch, col["first"]])
            stop = int(batches[stop_batch, col["first"]]) if stop_batch < len(batches) else total
            indices = f["voxels/indices"][start:stop].astype(np.intp)
            base = f["volume"][name]["rgba"][()]
            rgba = base[indices[:, 0], indices[:, 1], indices[:, 2]]
            labels[start:stop] = label_of(rgba, meta["specimen"]["labels"])
            if len(drill):
                world = voxel_world(indices, base.shape[:3], volume["dimensions_sim"], rotation, position)
                row_times = np.repeat(batches[first_batch:stop_batch, col["t_sim"]],
                                      batches[first_batch:stop_batch, col["count"]].astype(int))
                order = np.argsort(drill[:, 0])
                tip = np.column_stack([np.interp(row_times, drill[order, 0], drill[order, c]) for c in (2, 3, 4)])
                nominal = timeline[burr_slot(timeline, row_times), 2]
                reach = nominal * BURR_SIM_PER_NOMINAL_MM * 0.5 + DETACHED_MARGIN_VOXELS * np.max(
                    np.asarray(volume["dimensions_sim"]) / np.asarray(base.shape[:3]))
                causes[start:stop] = np.linalg.norm(world - tip, axis=1) > reach
        voxel_burr = np.zeros(total, np.uint8)
        voxel_diamond = np.zeros(total, np.uint8)
        if len(batches):
            voxel_slots = np.repeat(burr_slot(timeline, batches[:, col["t_sim"]]), batches[:, col["count"]].astype(int))
            voxel_burr = timeline[voxel_slots, 2].astype(np.uint8)
            voxel_diamond = timeline[voxel_slots, 3].astype(np.uint8)
        for key, data in (("voxels/label", labels), ("voxels/cause", causes), ("voxels/burr_mm", voxel_burr),
                          ("voxels/burr_diamond", voxel_diamond)):
            if key in f:
                del f[key]
            f.create_dataset(key, data=data, compression="gzip")
        kinematics, kin_columns = drill_kinematics(drill)
        strokes = find_strokes(batches, col, labels, kinematics, drill, force, timeline)
        for key, data, columns in (("derived/drill", kinematics, kin_columns),
                                   ("derived/strokes", strokes, STROKE_COLUMNS),
                                   ("derived/burr_timeline", timeline, BURR_COLUMNS)):
            if key in f:
                del f[key]
            f.create_dataset(key, data=data, compression="gzip").attrs["columns"] = ",".join(columns)
        check = {"status": "not_checked", "detail": "AMBF's final volume was not captured"}
        if "volume/final_rgba" in f and epochs:
            truth = f["volume/final_rgba"][..., 3] > SOLID_ALPHA
            replayed = replay_final(f)
            missed = int((replayed & ~truth).sum())
            still_in_ambf = ~replayed & truth
            # Removals that arrived after the final volume was requested may not be in it yet.
            during_capture = 0
            before = f["volume/final_rgba"].attrs.get("batches_before_capture")
            if before is not None and int(before) < len(batches):
                late = f["voxels/indices"][int(batches[int(before), col["first"]]):].astype(np.intp)
                during_capture = int(still_in_ambf[late[:, 0], late[:, 1], late[:, 2]].sum())
                still_in_ambf[late[:, 0], late[:, 1], late[:, 2]] = False
            extra = int(still_in_ambf.sum())
            check = {"status": "exact" if missed + extra == 0 else "mismatch", "voxels_ambf_removed_not_recorded": missed,
                     "voxels_recorded_but_still_in_ambf": extra, "voxels_removed_during_final_capture": during_capture,
                     "solid_voxels_at_end": int(truth.sum())}
        duration = float(drill[-1, 0] - drill[0, 0]) if len(drill) > 1 else 0.0
        summary = {
            "duration_s": round(duration, 3), "drill_samples": int(len(drill)),
            "drill_rate_hz": round(len(drill) / duration, 1) if duration > 0 else 0.0,
            "voxels_removed": total, "bone_voxels_removed": int((labels == 1).sum()),
            "inner_ear_voxels_removed": int((labels == 2).sum()),
            "detached_voxels_cleared": int((causes == 1).sum()), "strokes": int(len(strokes)),
            "volume_resets": max(len(epochs) - 1, 0), "replay_check": check,
            "starting_burr_mm": int(timeline[0, 2]),
            "burr_sequence": [{"t_sim": round(float(r[0]), 3), "size_mm": int(r[2]),
                               "type": "diamond" if r[3] else "cutting"} for r in timeline],
            "voxels_by_burr_mm": {str(int(v)): int((voxel_burr == v).sum()) for v in np.unique(voxel_burr)},
        }
    meta["summary"] = summary
    partial = folder / "session.json.part"
    partial.write_text(json.dumps(meta, indent=2))
    partial.replace(folder / "session.json")
    return summary


class ExpertSession:
    """Read a recorded session: the scene at any moment of the simulator clock."""

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.meta = json.loads((self.folder / "session.json").read_text())
        self.h5 = h5py.File(self.folder / "ground_truth.h5", "r")
        self.batches, self.batch_col = table(self.h5, "voxel_batches")
        self.epochs = _epochs(self.h5)

    def close(self):
        self.h5.close()

    def pose_at(self, stream: str, t_sim: float) -> np.ndarray:
        """Latest recorded x y z qx qy qz qw of drill / camera / suction at or before t_sim."""
        data, _ = table(self.h5, stream)
        index = max(int(np.searchsorted(data[:, 0], t_sim, side="right")) - 1, 0)
        return data[index, 2:9]

    def occupancy_at(self, t_sim: float) -> np.ndarray:
        """Solid voxels (x, y, z) at this time: the epoch's volume minus its removals so far."""
        times = self.batches[:, self.batch_col["t_sim"]] if len(self.batches) else np.zeros(0)
        done = int(np.searchsorted(times, t_sim, side="right"))
        current = self.epochs[0]
        for epoch in self.epochs:
            if float(self.h5["volume"][epoch[0]].attrs["t_sim"]) <= t_sim:
                current = epoch
        # Removals of this epoch only: none before its first batch, none after the next reset.
        following = [epoch[1] for epoch in self.epochs if epoch[1] > current[1]]
        done = max(min(done, following[0]) if following else done, current[1])
        solid = self.h5["volume"][current[0]]["rgba"][..., 3] > SOLID_ALPHA
        if done > current[1]:
            start = int(self.batches[current[1], self.batch_col["first"]])
            last = self.batches[done - 1]
            stop = int(last[self.batch_col["first"]] + last[self.batch_col["count"]])
            indices = self.h5["voxels/indices"][start:stop].astype(np.intp)
            solid[indices[:, 0], indices[:, 1], indices[:, 2]] = False
        return solid
