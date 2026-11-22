"""Ground-truth recorder for one expert drilling session (layout: expert_dataset.py).

Started by the launcher's Expert drilling tab with ROS 2 sourced; SIGINT or SIGTERM ends the
session, takes AMBF's final volume, and runs the replay check.

Nothing is thinned out: every ROS message is kept. Tables reach the disk once a second, so a
crash loses at most the last second. Video is JPEG per frame (quality 95) on worker threads;
if the encoder cannot keep up a frame is dropped and counted, never the ground truth.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import mmap
import pathlib
import queue
import signal
import sys
import threading
import time

import cv2
import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "rtx_view"))
import expert_dataset  # noqa: E402
import rtx_record  # noqa: E402

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
DUMP_REQUEST = RUNTIME / "recorder_dump.request"
DUMP_FILE = RUNTIME / "recorder_dump.bin"
DRILL_STATE = RUNTIME / "drill_state.json"
HAPTIC_STATE = RUNTIME / "haptic_state.json"
EYES_PATH = pathlib.Path("/dev/shm/fivrs_rtx_eyes")
JPEG_QUALITY = 95
MAX_PENDING_FRAMES = 48

POSE = expert_dataset.DRILL_COLUMNS
STREAM_COLUMNS = {
    "drill": POSE, "drill_rigid_body": POSE, "camera": POSE, "suction": POSE,
    "force": ("t_sim", "t_wall", "fx", "fy", "fz"),
    "burr": ("t_sim", "t_wall", "size_mm"),
    "reposition": ("t_sim", "t_wall") + tuple(f"from_{c}" for c in POSE[2:]) + tuple(f"to_{c}" for c in POSE[2:]),
    "voxel_batches": ("t_sim", "t_wall", "batch_id", "first", "count"),
}
STATE_FLAGS = ("burr_diamond", "on", "paused", "cam_clutch", "device_clutch", "orbit", "scope_button", "drill_locked")
STATE_NUMBERS = ("size_mm", "px", "py", "pz", "qx", "qy", "qz", "qw", "cx", "cy", "cz", "lx", "ly", "lz",
                 "ux", "uy", "uz", "fov", "hx", "hy", "hz", "zoom_steps", "zoom_travel", "recenter_count",
                 "removed_voxels", "volume_reset_count", "voxel_removal_limit", "burr_diameter_sim")
STREAM_COLUMNS["state"] = ("t_sim", "t_wall") + STATE_FLAGS + STATE_NUMBERS


def state_row(state: dict, wall: float) -> list:
    state = {**state, "burr_diamond": state.get("style") == "diamond"}
    row = [float(state.get("sim_time", np.nan)), wall]
    row += [1.0 if state.get(key) else 0.0 for key in STATE_FLAGS]
    row += [float(state.get(key, np.nan)) for key in STATE_NUMBERS]
    return row


def read_dump(path: pathlib.Path) -> np.ndarray:
    """AMBF's dump as (x, y, z, RGBA)."""
    raw = path.read_bytes()
    width, height, depth, bpp = (int(v) for v in np.frombuffer(raw[:16], np.int32))
    if bpp != 4:
        raise ValueError(f"expected RGBA voxels, got {bpp} bytes per voxel")
    data = np.frombuffer(raw[16:], np.uint8).reshape(depth, height, width, 4)
    return np.transpose(data, (2, 1, 0, 3)).copy()


def request_dump(timeout_s: float = 6.0, stop_waiting=None) -> np.ndarray | None:
    """Ask AMBF for its voxels. None if it does not answer (no RTX bridge, or it has closed)."""
    DUMP_FILE.unlink(missing_ok=True)
    partial = DUMP_REQUEST.with_suffix(".tmp")
    partial.write_text("dump\n")
    partial.replace(DUMP_REQUEST)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if DUMP_FILE.exists() and not DUMP_REQUEST.exists():
            data = read_dump(DUMP_FILE)
            DUMP_FILE.unlink(missing_ok=True)
            return data
        if stop_waiting is not None and stop_waiting():
            break
        time.sleep(0.02)
    DUMP_REQUEST.unlink(missing_ok=True)
    return None


class EyeReader:
    """New stereo frames from the headset share (rtx_eyes.py). Costs the renderer nothing."""

    def __init__(self, path=EYES_PATH):
        self.path = pathlib.Path(path)
        self._map = None
        self._inode = None
        self._seq = 0

    def poll(self):
        """(fov_y, focus, left BGR, right BGR), or None if there is no new complete frame."""
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        if self._map is None or info.st_ino != self._inode:
            if self._map is not None:
                self._map.close()
                self._map = None
            if info.st_size < 64:
                return None
            with open(self.path, "rb") as handle:
                self._map = mmap.mmap(handle.fileno(), info.st_size, access=mmap.ACCESS_READ)
            self._inode = info.st_ino
        m = self._map
        if m[0:4] != b"FVEY":
            return None
        before = int(np.frombuffer(m[8:16], np.uint64)[0])
        if before % 2 or before == self._seq:
            return None
        width = int(np.frombuffer(m[16:20], np.uint32)[0])
        height = int(np.frombuffer(m[20:24], np.uint32)[0])
        eye = width * height * 4
        if 64 + 2 * eye > len(m):
            return None
        fov = float(np.frombuffer(m[24:28], np.float32)[0])
        focus = float(np.frombuffer(m[28:32], np.float32)[0])
        left = np.frombuffer(m, np.uint8, count=eye, offset=64).reshape(height, width, 4)[::-1, :, 2::-1].copy()
        right = np.frombuffer(m, np.uint8, count=eye, offset=64 + eye).reshape(height, width, 4)[::-1, :, 2::-1].copy()
        if int(np.frombuffer(m[8:16], np.uint64)[0]) != before:
            return None
        self._seq = before
        return fov, focus, left, right


class VideoWriter:
    """JPEG frames in video.h5, in order, encoded off the recording threads."""

    COLUMNS = ("t_sim", "t_wall", "px", "py", "pz", "qx", "qy", "qz", "qw", "fov_y", "focus")

    def __init__(self, path, eyes):
        self.path = pathlib.Path(path)
        self.eyes = tuple(eyes)
        self.frames = 0
        self.dropped = 0
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="expert-jpeg")
        self._pending = queue.Queue()
        self._slots = threading.BoundedSemaphore(MAX_PENDING_FRAMES)
        self._thread = threading.Thread(target=self._loop, name="expert-video", daemon=True)
        self._thread.start()

    @staticmethod
    def _encode(image):
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return data.ravel()

    def add(self, row, images):
        if not self._slots.acquire(blocking=False):
            self.dropped += 1
            return
        self._pending.put((row, [self._pool.submit(self._encode, image) for image in images]))

    def close(self):
        self._pending.put(None)
        self._thread.join()
        self._pool.shutdown()

    def _loop(self):
        byte_type = h5py.vlen_dtype(np.uint8)
        with h5py.File(self.path, "w") as f:
            f.attrs["eyes"] = ",".join(self.eyes)
            table = f.create_dataset("frames", shape=(0, len(self.COLUMNS)), maxshape=(None, len(self.COLUMNS)),
                                     chunks=(256, len(self.COLUMNS)), dtype=np.float64)
            table.attrs["columns"] = ",".join(self.COLUMNS)
            images = {eye: f.create_dataset(eye, shape=(0,), maxshape=(None,), chunks=(16,), dtype=byte_type)
                      for eye in self.eyes}
            last_flush = time.monotonic()
            while True:
                item = self._pending.get()
                if item is None:
                    break
                row, futures = item
                try:
                    encoded = [future.result() for future in futures]
                    index = table.shape[0]
                    table.resize(index + 1, axis=0)
                    table[index] = row
                    for eye, data in zip(self.eyes, encoded):
                        images[eye].resize(index + 1, axis=0)
                        images[eye][index] = data
                    self.frames += 1
                except Exception as exc:
                    self.dropped += 1
                    print(f"ERROR! Video frame lost: {exc}", flush=True)
                finally:
                    self._slots.release()
                if time.monotonic() - last_flush > 2.0:
                    f.flush()
                    last_flush = time.monotonic()
            f.attrs["frames_dropped"] = self.dropped


class SessionRecorder:
    """Collects every stream and writes ground_truth.h5. No ROS, so tests can drive it."""

    def __init__(self, folder, video_eyes=()):
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._rows = collections.defaultdict(list)
        self._voxels = []
        self._voxel_total = 0
        self._batch_total = 0
        self._jobs = queue.Queue()
        self.h5 = h5py.File(self.folder / "ground_truth.h5", "w")
        self.video = VideoWriter(self.folder / "video.h5", video_eyes) if video_eyes else None
        self.epochs = 0
        self.latest = {}
        self.write_errors = 0
        self._writer = threading.Thread(target=self._write_loop, name="expert-write", daemon=True)
        self._writer.start()

    # ---- inputs (any thread) ----
    def add(self, stream: str, row) -> None:
        with self._lock:
            self._rows[stream].append(row)
            if stream in ("drill", "camera"):
                self.latest[stream] = row

    def on_voxels(self, t_sim: float, batch_id: float, indices) -> None:
        indices = np.asarray(indices, np.uint16).reshape(-1, 3)
        if not len(indices):
            return
        with self._lock:
            self._rows["voxel_batches"].append((t_sim, time.time(), batch_id, self._voxel_total, len(indices)))
            self._voxels.append(indices)
            self._voxel_total += len(indices)
            self._batch_total += 1

    def on_haptic(self, wall: float, text: str) -> None:
        with self._lock:
            self._rows["_haptic"].append((wall, text))

    def add_epoch(self, rgba: np.ndarray, first_batch: int | None = None, source: str = "ambf") -> None:
        """The whole volume at the start of an epoch; later removals apply on top of it."""
        with self._lock:
            first = self._batch_total if first_batch is None else first_batch
            number = self.epochs
            self.epochs += 1
            # The simulator clock when this volume took over; the first one covers all earlier times.
            t_sim = float("-inf") if number == 0 else float(self.latest.get("drill", (float("-inf"),))[0])
        self._jobs.put(("epoch", number, rgba, first, source, t_sim))

    def add_final(self, rgba: np.ndarray, batches_before: int | None = None) -> None:
        """AMBF's last volume. batches_before: removals received before it was requested."""
        self._jobs.put(("final", rgba, batches_before))

    def add_inner_ear(self, mask: np.ndarray) -> None:
        """The hollow cavity: not part of the voxels, but where the anatomy's inner ear is."""
        self._jobs.put(("inner_ear", mask))

    def batch_count(self) -> int:
        with self._lock:
            return self._batch_total

    def frame(self, t_wall: float, t_sim: float, pose7, fov: float, focus: float, images) -> None:
        if self.video is not None:
            self.video.add([t_sim, t_wall, *pose7, fov, focus], images)

    # ---- disk (writer thread) ----
    def _append(self, name, rows, columns):
        data = np.asarray(rows, np.float64)
        key = f"streams/{name}"
        if key not in self.h5:
            dataset = self.h5.create_dataset(key, data=data, maxshape=(None, data.shape[1]),
                                             chunks=(min(4096, max(len(data), 256)), data.shape[1]),
                                             compression="gzip", compression_opts=1)
            dataset.attrs["columns"] = ",".join(columns)
            return
        dataset = self.h5[key]
        start = dataset.shape[0]
        dataset.resize(start + len(data), axis=0)
        dataset[start:] = data

    def _flush(self):
        with self._lock:
            rows, self._rows = self._rows, collections.defaultdict(list)
            voxels, self._voxels = self._voxels, []
        for name, data in rows.items():
            if name == "_haptic":
                continue
            if data:
                self._append(name, data, STREAM_COLUMNS[name])
        if voxels:
            data = np.vstack(voxels)
            if "voxels/indices" not in self.h5:
                self.h5.create_dataset("voxels/indices", data=data, maxshape=(None, 3), chunks=(8192, 3),
                                       compression="gzip", compression_opts=1)
            else:
                dataset = self.h5["voxels/indices"]
                start = dataset.shape[0]
                dataset.resize(start + len(data), axis=0)
                dataset[start:] = data
        haptic = rows.get("_haptic")
        if haptic:
            if "haptic/json" not in self.h5:
                self.h5.create_dataset("haptic/json", shape=(0,), maxshape=(None,), chunks=(256,),
                                       dtype=h5py.string_dtype(), compression="gzip", compression_opts=1)
                self.h5.create_dataset("haptic/t_wall", shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.float64)
            start = self.h5["haptic/json"].shape[0]
            for key in ("haptic/json", "haptic/t_wall"):
                self.h5[key].resize(start + len(haptic), axis=0)
            self.h5["haptic/t_wall"][start:] = [row[0] for row in haptic]
            self.h5["haptic/json"][start:] = [row[1] for row in haptic]
        self.h5.flush()

    def _write_volume(self, job):
        if job[0] == "inner_ear":
            self.h5.create_dataset("volume/inner_ear", data=job[1], chunks=True, compression="gzip", compression_opts=4)
        elif job[0] == "epoch":
            _, number, rgba, first, source, t_sim = job
            group = self.h5.create_group(f"volume/epoch_{number}")
            group.create_dataset("rgba", data=rgba, chunks=True, compression="gzip", compression_opts=4)
            group.attrs["first_batch"] = int(first)
            group.attrs["source"] = source
            group.attrs["t_sim"] = t_sim
            group.attrs["t_wall"] = time.time()
        else:
            final = self.h5.create_dataset("volume/final_rgba", data=job[1], chunks=True,
                                           compression="gzip", compression_opts=4)
            if job[2] is not None:
                final.attrs["batches_before_capture"] = int(job[2])
        self.h5.flush()

    def _write_loop(self):
        running = True
        while running:
            try:
                job = self._jobs.get(timeout=1.0)
            except queue.Empty:
                job = "flush"
            try:
                if job is None:
                    running = False
                    self._flush()
                elif job == "flush":
                    self._flush()
                else:
                    # Removals received so far go to disk first, so a volume always follows them.
                    self._flush()
                    self._write_volume(job)
            except Exception as exc:
                self.write_errors += 1
                print(f"ERROR! Recording write failed: {exc}", flush=True)

    def close(self):
        if self.video is not None:
            self.video.close()
        self._jobs.put(None)
        self._writer.join()
        self.h5.close()


def _slices_folder(adf: pathlib.Path):
    import yaml

    doc = yaml.safe_load(adf.read_text())
    images = doc[doc["volumes"][0]]["images"]
    return (adf.parent / images["path"]).resolve(), images


def load_rgba_from_adf(adf: pathlib.Path) -> np.ndarray:
    """The volume as stored, for launches without the RTX bridge (no AMBF dump available)."""
    import re

    folder, images = _slices_folder(adf)
    pattern = re.compile(re.escape(images["prefix"]) + r"(\d+)\." + re.escape(images["format"]) + "$")
    names = sorted((p for p in folder.iterdir() if pattern.match(p.name)), key=lambda p: int(pattern.match(p.name).group(1)))
    slices = [np.flipud(cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)) for p in names]
    return np.transpose(np.stack(slices), (2, 1, 0, 3))


def load_inner_ear(adf: pathlib.Path):
    """The hollow inner ear's mask (x, y, z) of a UR5 specimen, or None."""
    path = _slices_folder(adf)[0] / "otic_capsule.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        return np.unpackbits(data["ear"])[: int(np.prod(shape))].reshape(shape).astype(bool)


def run(args):
    import rclpy
    from ambf_msgs.msg import CameraState, RigidBodyState
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from volumetric_drilling_msgs.msg import DrillReposition, DrillSize, Voxels

    folder = pathlib.Path(args.output_dir)
    video_mode = args.video
    eyes = {"none": (), "mono": ("left",), "stereo": ("left", "right")}[video_mode]
    recorder = SessionRecorder(folder, eyes)

    def stamp(header):
        value = header.stamp.sec + header.stamp.nanosec * 1e-9
        return value if value > 0 else time.time()

    def pose_row(message_stamp, p):
        return (message_stamp, time.time(), p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)

    rclpy.init()
    node = rclpy.create_node("fivrs_expert_recorder")
    latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    stream = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2048, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE)
    prefix = "/ambf/env"
    plugin = prefix + "/plugin/volumetric_drilling"
    for topic, name, kind in ((prefix + "/mastoidectomy_drill/State", "drill_rigid_body", RigidBodyState),
                              (prefix + "/frazier_suction/State", "suction", RigidBodyState),
                              (prefix + "/cameras/main_camera/State", "camera", CameraState)):
        node.create_subscription(kind, topic, (lambda n: lambda m: recorder.add(n, pose_row(stamp(m.header), m.pose)))(name), stream)
    # Every physics step (about 1 kHz); AMBF's own State topic above runs near 50 Hz.
    node.create_subscription(PoseStamped, plugin + "/drill_pose_physics", lambda m: recorder.add(
        "drill", pose_row(stamp(m.header), m.pose)), QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=4096, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE))
    node.create_subscription(Voxels, plugin + "/voxels_removed", lambda m: recorder.on_voxels(
        stamp(m.header), float(m.header.frame_id or 0), [(p.x, p.y, p.z) for p in m.indices]), stream)
    node.create_subscription(WrenchStamped, plugin + "/drill_force_feedback", lambda m: recorder.add(
        "force", (stamp(m.header), time.time(), m.wrench.force.x, m.wrench.force.y, m.wrench.force.z)), stream)
    node.create_subscription(DrillSize, plugin + "/drill_size", lambda m: recorder.add(
        "burr", (stamp(m.header), time.time(), float(m.size.data))), latched)
    node.create_subscription(DrillReposition, plugin + "/drill_reposition", lambda m: recorder.add(
        "reposition", pose_row(stamp(m.header), m.pose_from) + pose_row(0.0, m.pose_to)[2:]), stream)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    spin = threading.Thread(target=lambda: _spin(rclpy, node, stop), name="expert-ros", daemon=True)
    spin.start()
    time.sleep(0.5)  # subscriptions up before the first volume, so no removal falls between them

    baseline = request_dump(stop_waiting=stop.is_set)
    source = "ambf"
    if baseline is None:
        baseline = load_rgba_from_adf(pathlib.Path(args.volume_adf))
        source = "adf"
        print("WARNING! AMBF gave no voxel dump; the first volume comes from the ADF", flush=True)
    recorder.add_epoch(baseline, 0, source)
    inner_ear = load_inner_ear(pathlib.Path(args.volume_adf))
    if inner_ear is not None:
        recorder.add_inner_ear(inner_ear)

    reader = rtx_record.RecordReader() if video_mode == "mono" else None
    eye_reader = EyeReader() if video_mode == "stereo" else None
    if video_mode == "mono":
        rtx_record.ACTIVE_PATH.touch()
    (folder / "recording.started").write_text(str(time.time()))
    print("Recording started", flush=True)

    last_state_time = None
    last_haptic = None
    resets_seen = None
    next_eye = 0.0
    eye_period = 1.0 / max(args.video_hz, 1.0)
    last_report = time.monotonic()
    state = {}
    try:
        while not stop.is_set():
            try:
                info = DRILL_STATE.stat()
                if info.st_mtime_ns != last_state_time:
                    last_state_time = info.st_mtime_ns
                    state = json.loads(DRILL_STATE.read_text())
                    recorder.add("state", state_row(state, time.time()))
                    count = state.get("volume_reset_count")
                    if count is not None:
                        if resets_seen is not None and count != resets_seen:
                            first = recorder.batch_count()
                            time.sleep(0.1)  # the texture reset lands a frame later
                            rgba = request_dump(stop_waiting=stop.is_set)
                            if rgba is not None:
                                recorder.add_epoch(rgba, first)
                                print(f"Bone reset: epoch {recorder.epochs - 1}", flush=True)
                            else:
                                print("ERROR! Bone was reset but AMBF gave no volume", flush=True)
                        resets_seen = count
            except (OSError, ValueError):
                pass
            try:
                info = HAPTIC_STATE.stat()
                if info.st_mtime_ns != last_haptic:
                    last_haptic = info.st_mtime_ns
                    recorder.on_haptic(time.time(), HAPTIC_STATE.read_text())
            except OSError:
                pass
            if reader is not None:
                for frame_stamp, pose, fov, image in reader.poll():
                    recorder.frame(time.time(), frame_stamp, pose, fov, 0.0, [image])
            if eye_reader is not None and time.monotonic() >= next_eye:
                frame = eye_reader.poll()
                if frame is not None:
                    next_eye = time.monotonic() + eye_period
                    fov, focus, left, right = frame
                    camera = recorder.latest.get("camera")
                    if camera is not None:
                        recorder.frame(time.time(), camera[0], camera[2:9], fov, focus, [left, right])
            if time.monotonic() - last_report > 5.0:
                last_report = time.monotonic()
                video = f", {recorder.video.frames} frames ({recorder.video.dropped} dropped)" if recorder.video else ""
                print(f"Recording: {recorder.batch_count()} voxel batches{video}", flush=True)
            time.sleep(0.002)
    finally:
        rtx_record.ACTIVE_PATH.unlink(missing_ok=True)
        # Let the drill stop so the final volume and the last removals describe the same moment.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                if not json.loads(DRILL_STATE.read_text()).get("on"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        time.sleep(0.3)
        batches_before = recorder.batch_count()
        final = request_dump(timeout_s=3.0)
        time.sleep(0.3)  # removals published just before the dump
        if final is not None:
            recorder.add_final(final, batches_before)
        rclpy.try_shutdown()
        spin.join(timeout=2.0)
        recorder.close()
        video_note = {}
        if recorder.video is not None:
            video_note = {"frames": recorder.video.frames, "frames_dropped": recorder.video.dropped}
        meta_path = folder / "session.json"
        meta = json.loads(meta_path.read_text())
        meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["video_result"] = video_note
        meta["write_errors"] = recorder.write_errors
        meta_path.write_text(json.dumps(meta, indent=2))
        (folder / "recording.started").unlink(missing_ok=True)
        summary = expert_dataset.finalize(folder)
        print("RESULT " + json.dumps(summary), flush=True)


def _spin(rclpy, node, stop):
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except Exception:
        if not stop.is_set():
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--volume_adf", required=True)
    parser.add_argument("--video", choices=("none", "mono", "stereo"), default="mono")
    parser.add_argument("--video_hz", type=float, default=30.0)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
