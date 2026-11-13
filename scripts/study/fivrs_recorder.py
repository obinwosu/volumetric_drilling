"""Study recorder for the FIVRS simulator (ROS 2).

Writes the same HDF5 layout as scripts/data_record.py (format v3), which the
expert recordings used and the Mastoid feedback system reads:

    metadata_v3/      camera_intrinsic, camera_extrinsic, voxel_volume, README
                      frame_fov_y, frazier_suction_pose/{time_stamp, pose}
    vision_data/continuous_data/
                      time, l_img, pose_mastoidectomy_drill, pose_main_camera,
                      pose_mastoidectomy_volume
    physics_data/continuous_data/high_frequency_poses/<name>_pose/{time_stamp, pose}
                      (mastoidectomy_drill, main_camera)
    physics_data/intermittent_data/voxels_removed/{voxel_removed, voxel_color, voxel_time_stamp}
                                   burr_change/{burr_size, time_stamp}
                                   drill_force_feedback/{wrench, time_stamp}

Differences from data_record.py:
- ROS 2 (rclpy) instead of rospy, which this machine does not have.
- l_img is the RTX frame the resident saw, read from the RTX window through
  /dev/shm (see scripts/rtx_view/rtx_record.py), about 10 per second.
- metadata_v3/frame_fov_y records the vertical field of view of each frame:
  the stylus zoom changes it during a session, so camera_intrinsic only holds
  for the chunk's first frame.
- The suction pose is kept under metadata_v3: the feedback system's reader
  rejects stream fields it does not know, and it does not use the suction.
- No depth image; the RTX window does not produce one.

Run with ROS 2 sourced; stop with SIGINT or SIGTERM, which writes the last chunk.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import queue
import signal
import sys
import threading
import time

import h5py
import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "rtx_view"))
import rtx_record  # noqa: E402

# Pre-multiplies recorded poses to the OpenCV convention (T_cv_ambf), as in data_record.py.
EXTRINSIC = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
POSE_OBJECTS = ("mastoidectomy_drill", "frazier_suction", "main_camera")
README = (
    "All position information is in meters unless specified otherwise. \n"
    "Quaternion is a list in the order of [qx, qy, qz, qw]. \n"
    "Poses are defined to be T_world_obj. \n"
    "Depth in CV convention (corrected by extrinsic, T_cv_ambf) when a depth image is present; "
    "this recorder saves none. \n"
    "l_img is the RTX microscope view the participant saw, BGR, top row first. \n"
    "camera_intrinsic uses the chunk's first frame; frame_fov_y gives each frame's vertical field of view. \n"
    "user_controls is one row per frame: device_clutch, cam_clutch, scope_button (1 = held), "
    "read from the simulator's drill_state.json; -1 when it could not be read. \n"
)
USER_CONTROLS = ("device_clutch", "cam_clutch", "scope_button")
DRILL_STATE = pathlib.Path("/tmp/fivrs-rtx/drill_state.json")


def read_user_controls(path: pathlib.Path = DRILL_STATE):
    """Clutch and microscope-button state at this moment, so the analysis can skip
    stretches where the resident was moving the camera or repositioning the stylus."""
    try:
        state = json.loads(path.read_text())
        return [1 if state.get(key) else 0 for key in USER_CONTROLS]
    except (OSError, ValueError):
        return [-1] * len(USER_CONTROLS)
VOXEL_README = (
    "voxels_removed contains a group of voxels (Gv) removed. \n"
    "The voxel_time_stamp contains the time that the Gv was removed. The voxels_removed contains the voxels that comprise the Gv. \n"
    "The voxel_color contains the color of voxels that comprise the Gv. \n"
)


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def recorder_scale(nrrd_header) -> float:
    """Simulator units to meters, computed as data_record.py does."""
    with open(nrrd_header, "rb") as handle:
        header = pickle.load(handle)
    directions = header["space directions"]
    sizes = header["sizes"]
    largest = int(np.argmax(sizes))
    return float(np.linalg.norm(directions[largest]) * sizes[largest] / 1000.0)


def volume_pose(volume_adf, scale: float) -> np.ndarray:
    with open(volume_adf) as handle:
        params = yaml.safe_load(handle)
    name = params["volumes"][0]
    location = params[name]["location"]
    position = [location["position"][axis] * scale for axis in ("x", "y", "z")]
    rpy = location["orientation"]
    return np.array(position + list(rpy_to_quat(rpy["r"], rpy["p"], rpy["y"])), dtype=np.float64)


def intrinsic_for(height: int, width: int, fov_y: float) -> np.ndarray:
    focal = height / (2.0 * math.tan(fov_y / 2.0))
    return np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)


def _scaled(pose7, scale: float) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64).reshape(7).copy()
    pose[:3] *= scale
    return pose


class Recorder:
    """Collects messages and frames and writes one HDF5 chunk per ``chunk_frames`` frames.

    Has no ROS dependency, so tests can drive it directly.
    """

    def __init__(self, out_dir, volume_adf, nrrd_header, chunk_frames: int = 300):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scale = recorder_scale(nrrd_header)
        self.volume_pose = volume_pose(volume_adf, self.scale)
        self.chunk_frames = int(chunk_frames)
        self.voxel_volume = None
        self.chunks_written = 0
        self.frames_total = 0
        self.voxel_batches_total = 0
        self._lock = threading.Lock()
        self._latest = {}
        self._last_pose_stamp = {}
        self._last_frame_stamp = None
        self._writes = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, name="fivrs-recorder-write", daemon=True)
        self._writer.start()
        self._reset()

    def _reset(self):
        self.frames = []
        self.poses = {name: ([], []) for name in POSE_OBJECTS}
        self.voxels = ([], [], [])
        self.burr = ([], [])
        self.force = ([], [])

    # ---- inputs (any thread) ----
    def on_volume_info(self, dimensions, voxel_count):
        resolution = np.divide(np.asarray(dimensions, np.float64), np.asarray(voxel_count, np.float64)) * 1000.0
        with self._lock:
            self.voxel_volume = float(np.prod(resolution) * self.scale ** 3)

    def on_pose(self, name: str, stamp: float, pose7):
        pose = _scaled(pose7, self.scale)
        with self._lock:
            self._latest[name] = pose
            if name in self.poses:
                # While paused the simulator keeps publishing poses with a frozen sim time;
                # the feedback analysis rejects non-increasing timestamps (also across
                # chunks), so keep only samples newer than the last one recorded.
                last = self._last_pose_stamp.get(name)
                if last is not None and float(stamp) <= last:
                    return
                self._last_pose_stamp[name] = float(stamp)
                self.poses[name][0].append(float(stamp))
                self.poses[name][1].append(pose)

    def on_voxels(self, stamp: float, indices, colors01):
        indices = np.asarray(indices, np.float64).reshape(-1, 3)
        colors = np.asarray(colors01, np.float64).reshape(-1, 4) * 255.0
        with self._lock:
            self.voxels[0].append(float(stamp))
            self.voxels[1].append(indices)
            self.voxels[2].append(colors)
            self.voxel_batches_total += 1

    def on_burr(self, stamp: float, size: int):
        with self._lock:
            self.burr[0].append(float(stamp))
            self.burr[1].append(int(size))

    def on_force(self, stamp: float, wrench6):
        with self._lock:
            self.force[0].append(float(stamp))
            self.force[1].append(np.asarray(wrench6, np.float64).reshape(6))

    def on_frame(self, stamp: float, camera_pose7, fov_y: float, bgr):
        camera = _scaled(camera_pose7, self.scale)
        job = None
        with self._lock:
            drill = self._latest.get("mastoidectomy_drill")
            if drill is None:
                # The analysis needs a drill pose on every frame.
                return
            # Frames rendered while paused carry the frozen sim time; the analysis
            # rejects non-increasing frame times (also across chunks), so skip them.
            if self._last_frame_stamp is not None and float(stamp) <= self._last_frame_stamp:
                return
            self._last_frame_stamp = float(stamp)
            self.frames.append(
                {
                    "time": float(stamp),
                    "l_img": np.asarray(bgr, np.uint8),
                    "pose_mastoidectomy_drill": drill,
                    "pose_main_camera": camera,
                    "fov_y": float(fov_y),
                    "user_controls": read_user_controls(),
                }
            )
            self.frames_total += 1
            if len(self.frames) >= self.chunk_frames:
                job = self._take()
        if job is not None:
            self._writes.put(job)

    # ---- chunks ----
    def _take(self):
        job = {
            "frames": self.frames,
            "poses": self.poses,
            "voxels": self.voxels,
            "burr": self.burr,
            "force": self.force,
            "voxel_volume": self.voxel_volume,
            "index": self.chunks_written,
        }
        self.chunks_written += 1
        self._reset()
        return job

    def close(self):
        """Write what is left and wait for every chunk to reach disk."""
        with self._lock:
            job = self._take() if self.frames else None
        if job is not None:
            self._writes.put(job)
        self._writes.put(None)
        self._writer.join()

    def _write_loop(self):
        while True:
            job = self._writes.get()
            if job is None:
                return
            try:
                self._write(job)
            except Exception as exc:  # keep recording later chunks
                print(f"ERROR! Chunk {job['index']} was not written: {exc}", flush=True)

    def _write(self, job):
        frames = job["frames"]
        if job["voxel_volume"] is None:
            print("WARNING! No volume_info yet; voxel_volume is missing from this chunk", flush=True)
        name = time.strftime("%Y%m%d_%H%M%S", time.localtime(frames[0]["time"])) + f"_{job['index']:03d}.hdf5"
        path = self.out_dir / name
        partial = path.with_suffix(".part")
        height, width = frames[0]["l_img"].shape[:2]
        with h5py.File(partial, "w") as f:
            meta = f.create_group("metadata_v3")
            meta.create_dataset("camera_intrinsic", data=intrinsic_for(height, width, frames[0]["fov_y"]))
            meta.create_dataset("camera_extrinsic", data=EXTRINSIC)
            meta.create_dataset("README", data=README)
            if job["voxel_volume"] is not None:
                meta.create_dataset("voxel_volume", data=job["voxel_volume"]).attrs["units"] = "mm^3, millimeters cubed"
            vision = f.create_group("vision_data")
            vision.create_group("intermittent_data")
            continuous = vision.create_group("continuous_data")
            meta.create_dataset("frame_fov_y", data=np.asarray([row["fov_y"] for row in frames]))
            for key in ("time", "pose_mastoidectomy_drill", "pose_main_camera"):
                continuous.create_dataset(key, data=np.stack([row[key] for row in frames]), compression="gzip")
            continuous.create_dataset(
                "l_img",
                data=np.stack([row["l_img"] for row in frames]),
                compression="gzip",
                compression_opts=4,
                chunks=(1, height, width, 3),
            )
            controls = continuous.create_dataset(
                "user_controls", data=np.asarray([row["user_controls"] for row in frames], np.int8))
            controls.attrs["columns"] = ",".join(USER_CONTROLS)
            continuous.create_dataset(
                "pose_mastoidectomy_volume", data=np.stack([self.volume_pose] * len(frames)), compression="gzip"
            )
            physics = f.create_group("physics_data")
            hf = physics.create_group("continuous_data").create_group("high_frequency_poses")
            for obj, (stamps, poses) in job["poses"].items():
                if stamps:
                    parent = meta if obj == "frazier_suction" else hf
                    group = parent.create_group(obj + "_pose")
                    group.create_dataset("time_stamp", data=np.asarray(stamps), compression="gzip")
                    group.create_dataset("pose", data=np.stack(poses), compression="gzip")
            intermittent = physics.create_group("intermittent_data")
            stamps, indices, colors = job["voxels"]
            rows = [(i, idx, col) for i, (idx, col) in enumerate(zip(indices, colors)) if len(idx)]
            if rows:
                group = intermittent.create_group("voxels_removed")
                group.create_dataset("voxel_time_stamp", data=np.asarray(stamps), compression="gzip")
                group.create_dataset(
                    "voxel_removed",
                    data=np.vstack([np.hstack([np.full((len(idx), 1), i), idx]) for i, idx, _ in rows]),
                    compression="gzip",
                )
                group.create_dataset(
                    "voxel_color",
                    data=np.vstack([np.hstack([np.full((len(col), 1), i), col]) for i, _, col in rows]),
                    compression="gzip",
                )
                group.create_dataset("README", data=VOXEL_README)
            if job["burr"][0]:
                group = intermittent.create_group("burr_change")
                group.create_dataset("time_stamp", data=np.asarray(job["burr"][0]))
                group.create_dataset("burr_size", data=np.asarray(job["burr"][1]))
            if job["force"][0]:
                group = intermittent.create_group("drill_force_feedback")
                group.create_dataset("time_stamp", data=np.asarray(job["force"][0]), compression="gzip")
                group.create_dataset("wrench", data=np.stack(job["force"][1]), compression="gzip")
        partial.replace(path)
        print(f"Wrote {path.name}: {len(frames)} frames", flush=True)


def _stamp(header, fallback=None) -> float:
    value = header.stamp.sec + header.stamp.nanosec * 1e-9
    if value > 0:
        return value
    return time.time() if fallback is None else fallback


def run_ros(recorder: Recorder, frame_poll_s: float = 0.02):
    import rclpy
    from ambf_msgs.msg import CameraState, RigidBodyState
    from geometry_msgs.msg import WrenchStamped
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from volumetric_drilling_msgs.msg import DrillSize, VolumeInfo, Voxels

    rclpy.init()
    node = rclpy.create_node("fivrs_study_recorder")
    latched = QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=1,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    stream = QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=256,
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
    )
    prefix = "/ambf/env"
    plugin = prefix + "/plugin/volumetric_drilling"
    node.create_subscription(
        VolumeInfo, plugin + "/volume_info",
        lambda m: recorder.on_volume_info(list(m.dimensions), list(m.voxel_count)), latched,
    )
    node.create_subscription(DrillSize, plugin + "/drill_size", lambda m: recorder.on_burr(_stamp(m.header), m.size.data), latched)
    node.create_subscription(
        Voxels, plugin + "/voxels_removed",
        lambda m: recorder.on_voxels(
            _stamp(m.header),
            [(p.x, p.y, p.z) for p in m.indices],
            [(c.r, c.g, c.b, c.a) for c in m.colors],
        ),
        stream,
    )
    node.create_subscription(
        WrenchStamped, plugin + "/drill_force_feedback",
        lambda m: recorder.on_force(
            _stamp(m.header),
            (m.wrench.force.x, m.wrench.force.y, m.wrench.force.z,
             m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z),
        ),
        stream,
    )

    def pose_cb(name):
        def cb(m):
            p = m.pose
            recorder.on_pose(
                name, _stamp(m.header),
                (p.position.x, p.position.y, p.position.z,
                 p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
            )
        return cb

    node.create_subscription(RigidBodyState, prefix + "/mastoidectomy_drill/State", pose_cb("mastoidectomy_drill"), stream)
    node.create_subscription(RigidBodyState, prefix + "/frazier_suction/State", pose_cb("frazier_suction"), stream)
    node.create_subscription(CameraState, prefix + "/cameras/main_camera/State", pose_cb("main_camera"), stream)

    stop = threading.Event()
    reader = rtx_record.RecordReader()

    def frames_loop():
        while not stop.is_set():
            for stamp, pose, fov, image in reader.poll():
                recorder.on_frame(stamp, pose, fov, image)
            stop.wait(frame_poll_s)

    def request_stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rtx_record.ACTIVE_PATH.touch()
    frames = threading.Thread(target=frames_loop, name="fivrs-recorder-frames", daemon=True)
    frames.start()
    print("Recording started", flush=True)
    last_report = time.monotonic()
    try:
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - last_report > 5.0:
                last_report = time.monotonic()
                print(
                    f"Recording: {recorder.frames_total} frames, {recorder.voxel_batches_total} voxel batches, "
                    f"{reader.dropped} frames dropped",
                    flush=True,
                )
    finally:
        rtx_record.ACTIVE_PATH.unlink(missing_ok=True)
        stop.set()
        frames.join(timeout=2.0)
        for stamp, pose, fov, image in reader.poll():
            recorder.on_frame(stamp, pose, fov, image)
        recorder.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f"Recording stopped: {recorder.frames_total} frames in {recorder.chunks_written} files", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--volume_adf", required=True)
    parser.add_argument("--nrrd_header", default=str(REPO / "resources" / "volumes" / "nrrd_header.pkl"))
    parser.add_argument("--chunk_frames", type=int, default=300, help="Frames per HDF5 file (30 s at 10 Hz)")
    args = parser.parse_args(argv)
    recorder = Recorder(args.output_dir, args.volume_adf, args.nrrd_header, args.chunk_frames)
    run_ros(recorder)


if __name__ == "__main__":
    main()
