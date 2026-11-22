"""Read a recorded UR5 drilling session for review. No GUI here, so scripts can use it too.

    session = ReviewSession(folder)
    session.duration                    seconds; every time below counts from the session start
    session.channels["force"]           a curve: .t, .y, .label, .unit, .step
    session.value_at("speed", 12.3)
    session.drill_at(12.3)              x y z qx qy qz qw
    session.voxels.labels_at(12.3)      the bone at that moment: 0 air, 1 bone, 2 inner ear, 3 other
    session.video.frame_at(12.3)        BGR image (eye="left" or "right")
    export_video(...), export_csv(...)
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
try:
    import h5py
except ImportError:
    # The system Python has Qt and VTK; h5py comes from the feedback environment (same Python).
    # Appended, so the system's numpy stays the one in use.
    for site in glob.glob(str(REPO / "feedback_system" / ".venv" / "lib" / "python*" / "site-packages")):
        sys.path.append(site)
    import h5py

import cv2  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "expert_drilling"))
import expert_dataset as dataset  # noqa: E402

DATA_FOLDERS = (("expert_drillings_dev", "Development"), ("expert_drillings", "Expert"),
                ("reviewable_sessions", "Main"))
SOLID_ALPHA = dataset.SOLID_ALPHA
BURR_COLORS = {1: "#7fd1ae", 2: "#e7c66b", 4: "#6fa8dc", 6: "#d98cb3"}


class Channel:
    def __init__(self, key, label, unit, group, t, y, step=False):
        self.key, self.label, self.unit, self.group, self.step = key, label, unit, group, step
        self.t = np.asarray(t, np.float64)
        self.y = np.asarray(y, np.float64)

    def value_at(self, t):
        if not len(self.t) or t < self.t[0]:
            return None
        if self.step or t >= self.t[-1]:
            return float(self.y[max(int(np.searchsorted(self.t, t, side="right")) - 1, 0)])
        return float(np.interp(t, self.t, self.y))


def _sorted_times(t):
    """Sample times that never run backwards (the replay relies on it)."""
    return np.maximum.accumulate(t) if len(t) else t


class VoxelReplay:
    """The exact voxel state at any time, from each epoch's volume and its removals."""

    def __init__(self, session):
        h5 = session.h5
        self.session = session
        self.h5 = h5
        self.batches, self.col = dataset.table(h5, "voxel_batches")
        self.times = _sorted_times(self.batches[:, self.col["t_sim"]]) if len(self.batches) else np.zeros(0)
        self.total = int(h5["voxels/indices"].shape[0]) if "voxels/indices" in h5 else 0
        self.epochs = []
        names = [n for n in h5["volume"] if n.startswith("epoch_")] if "volume" in h5 else []
        for name in sorted(names, key=lambda n: int(n.split("_")[1])):
            attrs = h5["volume"][name].attrs
            self.epochs.append({"name": name, "first_batch": int(attrs["first_batch"]),
                                "t_sim": float(attrs.get("t_sim", -np.inf))})
        self.reset_times = [e["t_sim"] - session.t0 for e in self.epochs[1:]]
        self._loaded = None
        self._base = None
        self._order = None

    @property
    def shape(self):
        return None if self._base is None else self._base.shape

    def _voxel_at_batch(self, batch):
        if batch >= len(self.batches):
            return self.total
        return int(self.batches[batch, self.col["first"]])

    def _load(self, number):
        if self._loaded == number:
            return
        epoch = self.epochs[number]
        rgba = self.h5["volume"][epoch["name"]]["rgba"][()]
        labels = dataset.label_of(rgba, self.session.meta["specimen"]["labels"])
        solid = rgba[..., 3] > SOLID_ALPHA
        labels[solid & (labels == 0)] = 3
        labels[~solid] = 0
        if "volume/inner_ear" in self.h5:  # a hollow cavity: shown, never "removed"
            labels[self.h5["volume/inner_ear"][()]] = 2
        start = self._voxel_at_batch(epoch["first_batch"])
        following = self.epochs[number + 1]["first_batch"] if number + 1 < len(self.epochs) else len(self.batches)
        stop = self._voxel_at_batch(following)
        # Each voxel's removal number; the earliest wins if a voxel was reported twice.
        order = np.full(labels.shape, np.iinfo(np.int64).max, np.int64)
        if stop > start:
            indices = self.h5["voxels/indices"][start:stop].astype(np.intp)[::-1]
            order[indices[:, 0], indices[:, 1], indices[:, 2]] = np.arange(stop - 1, start - 1, -1)
        self._base, self._order, self._loaded = labels, order, number

    def position_at(self, t):
        """(epoch number, voxels removed so far counted from the session start)."""
        t_sim = t + self.session.t0
        number = 0
        for k, epoch in enumerate(self.epochs):
            if epoch["t_sim"] <= t_sim:
                number = k
        done = int(np.searchsorted(self.times, t_sim, side="right"))
        low = self.epochs[number]["first_batch"]
        high = self.epochs[number + 1]["first_batch"] if number + 1 < len(self.epochs) else len(self.batches)
        return number, self._voxel_at_batch(min(max(done, low), high))

    def labels_at(self, t):
        number, voxel = self.position_at(t)
        self._load(number)
        return np.where(self._order >= voxel, self._base, 0).astype(np.uint8)

    def removed_at(self, t):
        number, voxel = self.position_at(t)
        self._load(number)
        return (self._order < voxel) & (self._base > 0)


class Video:
    def __init__(self, path, t0):
        self.h5 = h5py.File(path, "r")
        self.eyes = tuple(e for e in str(self.h5.attrs.get("eyes", "")).split(",") if e and e in self.h5)
        rows = self.h5["frames"][()] if "frames" in self.h5 else np.zeros((0, 11))
        self.count = len(rows) if self.eyes else 0
        for eye in self.eyes:
            self.count = min(self.count, self.h5[eye].shape[0])
        rows = rows[: self.count]
        self.t = _sorted_times(rows[:, 0] - t0) if self.count else np.zeros(0)
        self.rows = rows
        self.fps = float((self.count - 1) / (self.t[-1] - self.t[0])) if self.count > 1 and self.t[-1] > self.t[0] else 0.0
        self.dropped = int(self.h5.attrs.get("frames_dropped", 0))

    def index_at(self, t):
        if not self.count:
            return -1
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, self.count - 1))

    def frame(self, index, eye="left"):
        return cv2.imdecode(self.h5[eye][index], cv2.IMREAD_COLOR)

    def frame_at(self, t, eye="left"):
        index = self.index_at(t)
        return None if index < 0 else self.frame(index, eye)

    def close(self):
        self.h5.close()


class ReviewSession:
    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.meta = json.loads((self.folder / "session.json").read_text())
        self.summary = self.meta.get("summary") or {}
        self.h5 = h5py.File(self.folder / "ground_truth.h5", "r")
        self.drill, _ = dataset.table(self.h5, "drill")
        if not len(self.drill):
            self.drill, _ = dataset.table(self.h5, "drill_rigid_body")
        if not len(self.drill):
            self.h5.close()
            raise ValueError("This session has no drill data.")
        self.drill = self.drill[np.argsort(self.drill[:, 0], kind="stable")]
        self.t0 = float(self.drill[0, 0])
        self.duration = float(self.drill[-1, 0] - self.t0)
        self.drill_t = self.drill[:, 0] - self.t0
        self.state, self.state_col = dataset.table(self.h5, "state")
        if len(self.state):
            self.state = self.state[np.isfinite(self.state[:, 0])]
        self.state_t = self.state[:, 0] - self.t0 if len(self.state) else np.zeros(0)
        self.burr = self._read_burr()
        self.strokes = self._read_table("derived/strokes")
        self.voxels = VoxelReplay(self)
        self.video = None
        if (self.folder / "video.h5").exists():
            try:
                self.video = Video(self.folder / "video.h5", self.t0)
            except OSError:
                self.video = None
        self.channels = {}
        self._build_channels()

    def close(self):
        self.h5.close()
        if self.video is not None:
            self.video.close()

    def _read_table(self, key):
        if key not in self.h5:
            return {"rows": np.zeros((0, 1)), "col": {}}
        data = self.h5[key]
        return {"rows": data[()], "col": {c: i for i, c in enumerate(data.attrs["columns"].split(","))}}

    def _read_burr(self):
        table = self._read_table("derived/burr_timeline")
        if len(table["rows"]):
            return table["rows"]
        raw, _ = dataset.table(self.h5, "burr")
        state, col = self.state, self.state_col
        start = self.meta.get("launch", {}).get("starting_burr_mm", 4)
        return dataset.burr_timeline(state, col, raw, start)

    def _add(self, key, label, unit, group, t, y, step=False):
        if len(t):
            self.channels[key] = Channel(key, label, unit, group, t, y, step)

    def _build_channels(self):
        t, drill = self.drill_t, self.drill
        kin = self._read_table("derived/drill")
        rows, col = kin["rows"], kin["col"]
        if not len(rows):
            rows, names = dataset.drill_kinematics(drill)
            col = {c: i for i, c in enumerate(names)}
        if len(rows):
            kt = rows[:, 0] - self.t0
            self._add("speed", "Drill speed", "mm/s", "Drill motion", kt, rows[:, col["speed_mm_s"]])
            acc = np.linalg.norm(rows[:, [col["ax_mm_s2"], col["ay_mm_s2"], col["az_mm_s2"]]], axis=1)
            self._add("acceleration", "Drill acceleration", "mm/s²", "Drill motion", kt, acc)
            spin = np.linalg.norm(rows[:, [col["wx_rad_s"], col["wy_rad_s"], col["wz_rad_s"]]], axis=1)
            self._add("angular_speed", "Drill angular speed", "rad/s", "Drill motion", kt, spin)
        offset = (drill[:, 2:5] - drill[0, 2:5]) * dataset.MM_PER_SIM
        for k, axis in enumerate("xyz"):
            self._add(f"pos_{axis}", f"Drill {axis} from start", "mm", "Drill position", t, offset[:, k])

        force, _ = dataset.table(self.h5, "force")
        if len(force):
            ft = force[:, 0] - self.t0
            self._add("force", "Force magnitude", "N", "Force", ft, np.linalg.norm(force[:, 2:5], axis=1))
            for k, axis in enumerate("xyz"):
                self._add(f"force_{axis}", f"Force {axis}", "N", "Force", ft, force[:, 2 + k])

        replay_batches, bcol = dataset.table(self.h5, "voxel_batches")
        if len(replay_batches):
            bt = _sorted_times(replay_batches[:, bcol["t_sim"]]) - self.t0
            counts = replay_batches[:, bcol["count"]]
            self._add("voxels", "Voxels removed", "voxels", "Bone removal", np.r_[0.0, bt], np.r_[0.0, np.cumsum(counts)], step=True)
            bins = np.arange(0.0, max(self.duration, 0.25) + 0.25, 0.25)
            rate, _ = np.histogram(bt, bins=bins, weights=counts)
            self._add("removal_rate", "Removal rate", "voxels/s", "Bone removal", bins[:-1] + 0.125, rate / 0.25)
            if "voxels/label" in self.h5:
                labels = self.h5["voxels/label"][()]
                last = (replay_batches[:, bcol["first"]] + counts - 1).astype(np.intp)
                self._add("inner_ear", "Inner ear voxels removed", "voxels", "Bone removal", np.r_[0.0, bt],
                          np.r_[0.0, np.cumsum(labels == 2)[last]], step=True)

        if len(self.burr):
            burr_t = np.append(np.maximum(self.burr[:, 0] - self.t0, 0.0), self.duration)
            burr_t[0] = 0.0  # the starting burr was in use from the first moment
            self._add("burr", "Burr size", "mm", "Drill state", burr_t, np.append(self.burr[:, 2], self.burr[-1, 2]), step=True)
        if len(self.state):
            for key, label in (("on", "Drill on"), ("device_clutch", "Device clutch"), ("cam_clutch", "Camera clutch"),
                               ("paused", "Paused")):
                if key in self.state_col:
                    self._add(key, label, "", "Drill state", self.state_t, self.state[:, self.state_col[key]], step=True)
            if "fov" in self.state_col:
                self._add("fov", "Microscope field of view", "deg", "Microscope", self.state_t,
                          np.degrees(self.state[:, self.state_col["fov"]]), step=True)
            if "cx" in self.state_col:
                cam = self.state[:, [self.state_col[c] for c in ("cx", "cy", "cz")]]
                if np.isfinite(cam).all():
                    moved = np.linalg.norm(cam - cam[0], axis=1) * dataset.MM_PER_SIM
                    self._add("camera_moved", "Microscope moved from start", "mm", "Microscope", self.state_t, moved)

    # ---- values at a moment ----
    def value_at(self, key, t):
        channel = self.channels.get(key)
        return None if channel is None else channel.value_at(t)

    def drill_at(self, t):
        index = int(np.clip(np.searchsorted(self.drill_t, t, side="right") - 1, 0, len(self.drill) - 1))
        return self.drill[index, 2:9]

    def burr_at(self, t):
        """(size in mm, "cutting" or "diamond", diameter in simulator units)."""
        slot = int(dataset.burr_slot(self.burr, np.array([t + self.t0]))[0])
        row = self.burr[slot]
        diameter = float(row[5]) if len(row) > 5 and np.isfinite(row[5]) else float(row[2]) * dataset.BURR_SIM_PER_NOMINAL_MM
        return int(row[2]), "diamond" if row[3] else "cutting", diameter

    def camera_at(self, t):
        """Recorded microscope: position, look, up, vertical field of view (radians)."""
        if not len(self.state) or "cx" not in self.state_col:
            return None
        row = self.state[int(np.clip(np.searchsorted(self.state_t, t, side="right") - 1, 0, len(self.state) - 1))]
        get = lambda names: np.array([row[self.state_col[n]] for n in names])
        position, look, up = get(("cx", "cy", "cz")), get(("lx", "ly", "lz")), get(("ux", "uy", "uz"))
        fov = float(row[self.state_col["fov"]])
        if not (np.isfinite(position).all() and np.isfinite(look).all() and np.isfinite(up).all() and fov > 0):
            return None
        return position, look, up, fov

    def path_until(self, t, seconds=None, max_points=1500):
        """Drill tip path up to t (world coordinates)."""
        stop = int(np.searchsorted(self.drill_t, t, side="right"))
        start = 0 if seconds is None else int(np.searchsorted(self.drill_t, t - seconds))
        if stop - start < 2:
            return np.zeros((0, 3))
        step = max(1, (stop - start) // max_points)
        return self.drill[start:stop:step, 2:5]

    def stroke_spans(self):
        rows, col = self.strokes["rows"], self.strokes["col"]
        if not len(rows) or "t_start" not in col:
            return []
        return [(float(r[col["t_start"]] - self.t0), float(r[col["t_end"]] - self.t0)) for r in rows]

    def burr_spans(self):
        spans = []
        for k, row in enumerate(self.burr):
            start = max(float(row[0] - self.t0), 0.0) if k else 0.0
            stop = float(self.burr[k + 1, 0] - self.t0) if k + 1 < len(self.burr) else self.duration
            spans.append((start, max(stop, start), int(row[2]), "diamond" if row[3] else "cutting"))
        return spans


def discover(base) -> list[dict]:
    """Every session under the recording folder, development and expert."""
    out = []
    for folder_name, kind in DATA_FOLDERS:
        for path in sorted(pathlib.Path(base, folder_name).glob("*/*/session_*/session.json")):
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            folder = path.parent
            who = meta.get("participant") or {}
            person = f"{who.get('last_name', '')}, {who.get('first_name', '')}".strip(", ") or folder.parent.parent.name
            out.append({"folder": folder, "kind": kind, "development": kind == "Development", "person": person,
                        "specimen": (meta.get("specimen") or {}).get("name") or folder.parent.name,
                        "number": int(meta.get("session_number") or folder.name.split("_")[-1]),
                        "started": meta.get("started", ""), "summary": meta.get("summary") or {}})
    return out


def export_video(session, path, start, stop, eye="left", progress=None):
    """Selected stretch as MP4 at the recording's frame rate. eye: left, right or both."""
    video = session.video
    if video is None or not video.count:
        raise ValueError("This session has no video.")
    fps = max(video.fps, 1.0)
    times = np.arange(start, stop, 1.0 / fps)
    if not len(times):
        raise ValueError("The selected stretch is empty.")
    writer = None
    try:
        for k, t in enumerate(times):
            index = video.index_at(float(t))
            if eye == "both" and "right" in video.eyes:
                image = cv2.hconcat([video.frame(index, "left"), video.frame(index, "right")])
            else:
                image = video.frame(index, eye if eye in video.eyes else video.eyes[0])
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not write {path}")
            writer.write(image)
            if progress is not None and not progress(k + 1, len(times)):
                break
    finally:
        if writer is not None:
            writer.release()
    return len(times)


def export_csv(session, path, start, stop, keys=None, rate_hz=100.0):
    """The chosen curves on one time base, for a spreadsheet or a plot."""
    keys = [k for k in (keys or list(session.channels)) if k in session.channels]
    times = np.arange(start, stop + 0.5 / rate_hz, 1.0 / rate_hz)
    columns = [times]
    header = ["time_s"]
    for key in keys:
        channel = session.channels[key]
        header.append(f"{key}_{channel.unit.replace('/', '_per_').replace('²', '2')}" if channel.unit else key)
        if channel.step:
            slot = np.searchsorted(channel.t, times, side="right") - 1
            values = channel.y[np.clip(slot, 0, len(channel.y) - 1)].astype(float)
            values[slot < 0] = np.nan
        else:
            values = np.interp(times, channel.t, channel.y, left=np.nan, right=np.nan)
        columns.append(values)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in zip(*columns):
            writer.writerow(["" if not np.isfinite(v) else f"{v:.6g}" if i else f"{v:.4f}" for i, v in enumerate(row)])
    return len(times)
