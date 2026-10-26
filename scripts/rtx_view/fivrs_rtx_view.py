#!/usr/bin/env python3
"""RTX window for the FIVRS bone.

AMBF stays in charge of the volume, the drill constraint, haptics, and the
camera. This process only draws. It reads the anatomy slices in the same order
the simulator does, subscribes to the camera, drill, and voxel-removal topics,
and paints each surface from the label color stored in that voxel.

RTX FPS is the rate of this viewer's frames. It is not the AMBF graphics Hz.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import queue
import struct
import sys
import threading
import time

import numpy as np
import warp as wp
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from surface import BlockVolume, blocks_touched_by, index_to_local, local_to_world, vertex_normals

# DrillManager stores a millimetre as 0.01007 simulation units, then the
# cutter and the fluted mesh both use half of that as the radius. Converting
# millimetres through the volume's metre scale instead draws a sphere twice
# as large as the sphere that removes bone.
MM_TO_SIM = 0.01007
MESH_DIR = pathlib.Path(__file__).resolve().parents[2] / "ADF" / "meshes" / "high_res"
# main_camera field view angle, radians, full vertical angle.
VERTICAL_FOV_RAD = 0.17951634837990105
RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
BONE_ROUGHNESS = 0.55
# The specimen is one surface a few units in front of the camera. A distant
# beam lights that whole surface evenly, so it stays off unless toggled.
# A sphere of radius 0.04 is invisible here even at 1e6 nits (measured: black
# on a white card that a distant light of 200 lights to mid gray). The cone
# is a disk of radius 0.2, 0.3 units in front of the lens, emitting along the
# look direction. 1.2e7 nits matches that distant beam on axis at 5 units;
# each frame rescales by distance squared so dollying does not blow the
# surface out. The scope light itself comes from tuner_light.json, written
# by the FIVRS RTX Tuner. Intensity is that saved value, times exposure and
# distance-squared, with no extra gain from the cone angle.
LIGHT_BASE = {
    "/root/_ScopeBeam": 200.0,
    "/root/_ScopeFill": 8.0,
    "/root/_ScopeLight": 4.95e6,
}
CONE_RADIUS = 0.2
CONE_STANDOFF = 0.3
CONE_REFERENCE_DISTANCE = 5.0
CONE_ANGLE_DEG = 12.0
CONE_ANGLE_MIN = 8.0
CONE_ANGLE_MAX = 36.0
CONE_SOFTNESS = 0.2
CONE_FOCUS = 1.0
TUNER_LIGHT_PATH = pathlib.Path(__file__).resolve().parent / "tuner_light.json"
EXPOSURE_MIN = 0.05
EXPOSURE_MAX = 8.0
# Geometry sent on the draw thread. A cut used to upload every touched block
# before the drill could move again.
UPLOAD_BUDGET_S = 0.008


def _quat_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _num_digits(value: int) -> int:
    """chai3d cNumDigits. Zero has one digit."""
    value = abs(int(value))
    if value == 0:
        return 1
    digits = 0
    while value >= 1:
        value //= 10
        digits += 1
    return digits


def _slice_paths(folder: pathlib.Path, prefix: str, extension: str, count: int):
    """Files cMultiImage::loadFromFiles would open, in that order.

    The names are not zero-padded to one width (``plane009.png`` and
    ``plane00100.png`` both exist), so a lexical sort stacks slice 10 beside
    slice 1. Chai walks the index and tries every padding up to the width of
    ``count``.
    """
    paths = []
    width = _num_digits(int(count))
    for index in range(int(count) + 1):
        for digits in range(min(_num_digits(index), width), width + 1):
            path = folder / f"{prefix}{index:0{digits}d}.{extension}"
            if path.is_file():
                paths.append(path)
    return paths


def _load_occupancy(anatomy_adf: pathlib.Path):
    from PIL import Image

    text = anatomy_adf.read_text()
    # The volume ADF is a small custom YAML. PyYAML reads the image block.
    doc = yaml.safe_load(text)
    volume = None
    for key, value in doc.items():
        if isinstance(key, str) and key.startswith("VOLUME") and isinstance(value, dict):
            volume = value
            break
    if volume is None:
        raise SystemExit(f"No VOLUME block in {anatomy_adf}")
    images = volume["images"]
    folder = (anatomy_adf.parent / images["path"]).resolve()
    names = _slice_paths(folder, images["prefix"], images["format"], int(images["count"]))
    if not names:
        raise SystemExit(f"No slices in {folder}")
    first = np.asarray(Image.open(names[0]).convert("RGBA"))
    height, width = first.shape[:2]
    occupied = np.zeros((width, height, len(names)), dtype=bool)
    colors = np.zeros((width, height, len(names), 3), dtype=np.uint8)
    for index, path in enumerate(names):
        # Chai flips PNG rows, so the bottom row is voxel y = 0, and x is the column.
        rgba = np.flipud(np.asarray(Image.open(path).convert("RGBA")))
        occupied[:, :, index] = rgba[:, :, 3].T > 10
        colors[:, :, index, :] = np.transpose(rgba[:, :, :3], (1, 0, 2))
    return occupied, colors


def _load_dump(path: pathlib.Path):
    raw = path.read_bytes()
    width, height, depth, bpp = np.frombuffer(raw[:16], dtype=np.int32)
    payload = np.frombuffer(raw[16:], dtype=np.uint8)
    voxels = int(width) * int(height) * int(depth)
    if bpp == 4:
        rgba = payload.reshape(int(depth), int(height), int(width), 4)
        alpha = np.transpose(rgba[:, :, :, 3], (2, 1, 0))
        colors = np.transpose(rgba[:, :, :, :3], (2, 1, 0, 3))
    elif bpp == 1:
        alpha = np.transpose(payload.reshape(int(depth), int(height), int(width)), (2, 1, 0))
        colors = np.zeros(alpha.shape + (3,), dtype=np.uint8)
        colors[alpha > 10] = (248, 248, 216)
    else:
        raise RuntimeError(f"Unexpected occupancy bpp {bpp}")
    # File layout is z, y, x. The mesh is indexed x, y, z.
    return alpha > 10, np.ascontiguousarray(colors), voxels


def _read_catchup(path: pathlib.Path):
    """Return ``(history_missing, [(batch_id, [(x, y, z), ...]), ...])``."""
    raw = path.read_bytes()
    magic, missing, batch_count = struct.unpack_from("<III", raw, 0)
    if int(magic) != 0x48544346:
        raise RuntimeError(f"bad catch-up file {int(magic)}")
    offset = 12
    batches = []
    for _ in range(int(batch_count)):
        batch_id, count = struct.unpack_from("<QI", raw, offset)
        offset += 12
        voxels = []
        if count:
            coords = np.frombuffer(raw, dtype="<i4", offset=offset, count=count * 3).reshape(-1, 3)
            voxels = [tuple(int(v) for v in row) for row in coords]
            offset += count * 12
        batches.append((batch_id, voxels))
    return bool(int(missing)), batches


def _sphere(radius: float, stacks: int = 8, slices: int = 12):
    verts = []
    for i in range(stacks + 1):
        v = i / stacks
        phi = np.pi * (v - 0.5)
        z = np.sin(phi) * radius
        r = np.cos(phi) * radius
        for j in range(slices):
            theta = 2.0 * np.pi * j / slices
            verts.append((np.cos(theta) * r, np.sin(theta) * r, z))
    points = np.asarray(verts, dtype=np.float32)
    tris = []
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = i * slices + (j + 1) % slices
            c = (i + 1) * slices + j
            d = (i + 1) * slices + (j + 1) % slices
            tris.extend((a, c, b, b, c, d))
    return points, np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def _load_obj(path):
    """Positions and authored normals. Corners stay split so flute edges stay sharp."""
    positions = []
    normals = []
    corners = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                positions.append([float(item) for item in line.split()[1:4]])
            elif line.startswith("vn "):
                normals.append([float(item) for item in line.split()[1:4]])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    parts = token.split("/")
                    vertex = int(parts[0]) - 1
                    normal = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else vertex
                    face.append((vertex, normal))
                for index in range(1, len(face) - 1):
                    corners.extend((face[0], face[index], face[index + 1]))
    lookup = {}
    points = []
    point_normals = []
    indices = []
    for vertex, normal in corners:
        slot = lookup.get((vertex, normal))
        if slot is None:
            slot = len(points)
            lookup[(vertex, normal)] = slot
            points.append(positions[vertex])
            if 0 <= normal < len(normals):
                point_normals.append(normals[normal])
            else:
                point_normals.append((0.0, 0.0, 1.0))
        indices.append(slot)
    points = np.asarray(points, dtype=np.float32)
    point_normals = np.asarray(point_normals, dtype=np.float32)
    lengths = np.linalg.norm(point_normals, axis=1, keepdims=True)
    point_normals /= np.maximum(lengths, 1e-8)
    return points, np.asarray(indices, dtype=np.int32).reshape(-1, 3), point_normals


def _mul_quat(left, right):
    """Hamilton product, both quaternions stored as x, y, z, w."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _spin_about_x(angle):
    half = 0.5 * float(angle)
    return (float(np.sin(half)), 0.0, 0.0, float(np.cos(half)))


def _tuner_lights_from_file():
    """Scope light saved by the FIVRS RTX Tuner. Missing keys keep the defaults."""
    lights = {
        "cone_angle_deg": CONE_ANGLE_DEG,
        "cone_softness": CONE_SOFTNESS,
        "intensity_nits": LIGHT_BASE["/root/_ScopeLight"],
        "fill": LIGHT_BASE["/root/_ScopeFill"],
        "disk_radius": CONE_RADIUS,
        "disk_standoff": CONE_STANDOFF,
        "disk_color": [1.0, 0.94, 0.86],
        "fill_color": [0.85, 0.78, 0.70],
    }
    try:
        saved = json.loads(TUNER_LIGHT_PATH.read_text())
        lights.update(saved.get("lights") or {})
        mtime = TUNER_LIGHT_PATH.stat().st_mtime
        found = True
    except (OSError, json.JSONDecodeError) as exc:
        print(f"RTX light file unavailable ({exc}); using built-in defaults", flush=True)
        mtime = None
        found = False
    return lights, mtime, found


def _apply_tuner_lights(viewer, lights):
    """Copy the saved scope light onto the viewer. The distant beam is separate."""
    LIGHT_BASE["/root/_ScopeFill"] = float(lights["fill"])
    LIGHT_BASE["/root/_ScopeLight"] = float(lights["intensity_nits"])
    viewer._cone_angle = float(lights["cone_angle_deg"])
    viewer._cone_softness = float(lights["cone_softness"])
    viewer._cone_radius = float(lights["disk_radius"])
    viewer._cone_standoff = float(lights["disk_standoff"])
    viewer._disk_color = tuple(float(v) for v in lights["disk_color"])
    viewer._fill_color = tuple(float(v) for v in lights["fill_color"])


def _reload_tuner_lights(viewer):
    """Pick up a tuner save without restarting, when the file changes."""
    try:
        mtime = TUNER_LIGHT_PATH.stat().st_mtime
    except OSError:
        return
    if mtime == getattr(viewer, "_tuner_light_mtime", None):
        return
    lights, mtime, found = _tuner_lights_from_file()
    if not found:
        return
    _apply_tuner_lights(viewer, lights)
    viewer._tuner_light_mtime = mtime
    _paint_light_colors(viewer)
    _announce_tuner_light(viewer)


def _announce_tuner_light(viewer):
    print(
        "RTX light from tuner_light.json: "
        f"angle {viewer._cone_angle:.1f}°  softness {viewer._cone_softness:.2f}  "
        f"intensity {LIGHT_BASE['/root/_ScopeLight']:.3g}  fill {LIGHT_BASE['/root/_ScopeFill']:.1f}  "
        f"radius {viewer._cone_radius:.3f}",
        flush=True,
    )


def _paint_light_colors(viewer):
    from pxr import Gf

    stage = getattr(viewer, "stage", None)
    if stage is None:
        return
    disk = stage.GetPrimAtPath("/root/_ScopeLight")
    dome = stage.GetPrimAtPath("/root/_ScopeFill")
    if disk:
        disk.GetAttribute("inputs:color").Set(Gf.Vec3f(*viewer._disk_color))
    if dome:
        dome.GetAttribute("inputs:color").Set(Gf.Vec3f(*viewer._fill_color))


def _author_lights(viewer):
    from pxr import Gf, UsdLux

    lights, mtime, found = _tuner_lights_from_file()
    _apply_tuner_lights(viewer, lights)
    viewer._tuner_light_mtime = mtime
    distant = UsdLux.DistantLight.Define(viewer.stage, "/root/_ScopeBeam")
    distant.CreateIntensityAttr(0.0)
    distant.CreateColorAttr(Gf.Vec3f(*viewer._disk_color))
    distant.CreateAngleAttr(2.0)
    dome = UsdLux.DomeLight.Define(viewer.stage, "/root/_ScopeFill")
    dome.CreateIntensityAttr(LIGHT_BASE["/root/_ScopeFill"])
    dome.CreateColorAttr(Gf.Vec3f(*viewer._fill_color))
    disk = UsdLux.DiskLight.Define(viewer.stage, "/root/_ScopeLight")
    disk.CreateIntensityAttr(LIGHT_BASE["/root/_ScopeLight"])
    disk.CreateRadiusAttr(viewer._cone_radius)
    disk.CreateColorAttr(Gf.Vec3f(*viewer._disk_color))
    shaping = UsdLux.ShapingAPI.Apply(disk.GetPrim())
    shaping.CreateShapingConeAngleAttr(viewer._cone_angle)
    shaping.CreateShapingConeSoftnessAttr(viewer._cone_softness)
    shaping.CreateShapingFocusAttr(CONE_FOCUS)
    viewer._exposure = 1.0
    viewer._exposure_dirty = True
    viewer._beam_on = False
    viewer._cone_distance = CONE_REFERENCE_DISTANCE
    _remember_cone_angle(viewer._cone_angle)
    if found:
        _announce_tuner_light(viewer)
    print("RTX distant beam off    L toggles it", flush=True)


def _aim_lights(viewer, position, forward):
    # Stored here and written inside the camera update, so it shares that
    # renderer's write ordinal. A separate write is rejected.
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, forward))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(right, forward)
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = right
    mat[1, :3] = up
    mat[2, :3] = -forward
    mat[3, :3] = position
    viewer._fivrs_light_matrix = mat


def _bind_lights(viewer):
    original = viewer._update_ovrtx_camera

    def update():
        original()
        if getattr(viewer, "_rtx", None) is None:
            return
        packed = getattr(viewer, "_fivrs_cam", None)
        if packed is not None:
            position, right, up, forward = packed
            mat = np.eye(4, dtype=np.float64)
            mat[0, :3] = right
            mat[1, :3] = up
            mat[2, :3] = -forward
            mat[3, :3] = position
            try:
                viewer._write_runtime_attribute(
                    [viewer._camera_prim_path],
                    "omni:xform",
                    np.ascontiguousarray(mat, dtype=np.float64)[np.newaxis, ...],
                    is_matrix=True,
                )
            except Exception as exc:
                if not getattr(viewer, "_cam_warned", False):
                    print(f"camera write failed: {exc}", flush=True)
                    viewer._cam_warned = True
        _push_exposure(viewer)
        matrix = getattr(viewer, "_fivrs_light_matrix", None)
        if matrix is None:
            return
        beam = np.ascontiguousarray(matrix, dtype=np.float64)
        # Row 2 is -forward, so this steps the disk out along the look axis,
        # past the near plane, instead of leaving it inside the eye.
        disk = beam.copy()
        disk[3, :3] = beam[3, :3] - beam[2, :3] * float(
            getattr(viewer, "_cone_standoff", CONE_STANDOFF)
        )
        try:
            viewer._write_runtime_attribute(
                ["/root/_ScopeBeam"], "omni:xform", beam[np.newaxis, ...], is_matrix=True
            )
            viewer._write_runtime_attribute(
                ["/root/_ScopeLight"],
                "omni:xform",
                np.ascontiguousarray(disk)[np.newaxis, ...],
                is_matrix=True,
            )
        except Exception as exc:
            if not getattr(viewer, "_light_warned", False):
                print(f"scope light follow failed: {exc}", flush=True)
                viewer._light_warned = True

    viewer._update_ovrtx_camera = update
    _bind_exposure_keys(viewer)
    # This window follows the simulator camera. Viewer mouse-look would fight it.
    if getattr(viewer, "gui", None) is not None:
        viewer.gui.handle_mouse_drag = lambda *args, **kwargs: None
        viewer.gui.update_camera_from_keys = lambda *args, **kwargs: None


class Bridge:
    def __init__(self):
        import rclpy
        from ambf_msgs.msg import CameraState, RigidBodyState
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from volumetric_drilling_msgs.msg import DrillSize, VolumeInfo, Voxels

        rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node("fivrs_rtx_view")
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        stream = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=256,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Depth 1: a newer pose replaces one the viewer has not read yet.
        # Reliable matches the simulator publishers, so the topics still connect.
        pose = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.volume = None
        self.camera = None
        self.drill = None
        self.burr_mm = 6
        self.pending = []
        self.applied = 0
        self.contiguous = 0
        self._seen = set()
        self.missing_from = None
        self.missing_to = None
        self.gap = False
        self.gap_after = None
        self._arrived = 0
        self._lock = threading.Lock()
        self._incoming = []
        self._running = True
        self.node.create_subscription(VolumeInfo, "/ambf/env/plugin/volumetric_drilling/volume_info", self._volume, latched)
        self.node.create_subscription(DrillSize, "/ambf/env/plugin/volumetric_drilling/drill_size", self._burr, latched)
        self.node.create_subscription(Voxels, "/ambf/env/plugin/volumetric_drilling/voxels_removed", self._voxels, stream)
        self.node.create_subscription(CameraState, "/ambf/env/cameras/main_camera/State", self._camera, pose)
        self.node.create_subscription(RigidBodyState, "/ambf/env/mastoidectomy_drill/State", self._drill, pose)
        # The sized burrs are named 1mm, 2mm, 4mm, and 6mm. ROS 2 rejects those
        # topic names, so the visible pose is read from drill_state.json.
        self._nudge_totals = {
            "seq": 0,
            "yaw": 0.0,
            "pitch": 0.0,
            "pan_right": 0.0,
            "pan_up": 0.0,
            "dolly": 0.0,
        }
        self._nudge_announced = False
        self._nudge_warned = False
        self._spin_thread = threading.Thread(target=self._spin_loop, name="fivrs-rtx-recv", daemon=True)
        self._spin_thread.start()

    def _spin_loop(self):
        # Receipt stays on this thread so a mesh rebuild or a snapshot read
        # cannot stop new deletions from being recorded.
        while self._running:
            self._rclpy.spin_once(self.node, timeout_sec=0.01)

    def publish_camera_nudge(self, yaw=0.0, pitch=0.0, pan_right=0.0, pan_up=0.0, dolly=0.0):
        """Ask AMBF to move main_camera. This window then draws the published pose.

        The simulator's drilling node is never spun, so a ROS subscriber on it
        does not receive. Running totals are written for the graphics loop.
        """
        totals = self._nudge_totals
        totals["seq"] += 1
        totals["yaw"] += float(yaw)
        totals["pitch"] += float(pitch)
        totals["pan_right"] += float(pan_right)
        totals["pan_up"] += float(pan_up)
        totals["dolly"] += float(dolly)
        try:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            temporary = RUNTIME / "camera_nudge.tmp"
            temporary.write_text(
                f"{totals['seq']} {totals['yaw']:.8f} {totals['pitch']:.8f} "
                f"{totals['pan_right']:.8f} {totals['pan_up']:.8f} {totals['dolly']:.8f}\n"
            )
            temporary.replace(RUNTIME / "camera_nudge.txt")
        except OSError as exc:
            if not self._nudge_warned:
                print(f"RTX camera nudge was not saved: {exc}", flush=True)
                self._nudge_warned = True
            return
        if not self._nudge_announced:
            print("RTX camera gesture sent to AMBF main_camera.", flush=True)
            self._nudge_announced = True

    def spin(self):
        return

    def take_batches(self):
        with self._lock:
            rows = self._incoming
            self._incoming = []
            missing = (self.missing_from, self.missing_to)
            return rows, missing

    def note_batches(self, batch_ids):
        with self._lock:
            for batch_id in batch_ids:
                self._seen.add(int(batch_id))
            self._advance()

    def assume_through(self, batch_id):
        """Skip a hole wider than retained history and cover it with a snapshot."""
        with self._lock:
            batch_id = int(batch_id)
            if batch_id > self.contiguous:
                self.contiguous = batch_id
                self.applied = self.contiguous
            self._advance()

    def _advance(self):
        stale = [batch for batch in self._seen if batch <= self.contiguous]
        for batch in stale:
            self._seen.discard(batch)
        while (self.contiguous + 1) in self._seen:
            self.contiguous += 1
            self._seen.discard(self.contiguous)
            self.applied = self.contiguous
        if not self._seen or max(self._seen) <= self.contiguous:
            self.missing_from = None
            self.missing_to = None
            return
        self.missing_from = self.contiguous + 1
        self.missing_to = max(self._seen) - 1
        if self.missing_to < self.missing_from:
            self.missing_from = None
            self.missing_to = None

    def close(self):
        self._running = False
        self.node.destroy_node()
        self._rclpy.shutdown()

    def _volume(self, msg):
        self._arrived += 1
        pose = msg.pose
        self.volume = {
            "translation": np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            "rotation": _quat_matrix(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ),
            "dimensions": np.array(list(msg.dimensions), dtype=np.float64),
            "counts": np.array(list(msg.voxel_count), dtype=np.float64),
        }

    def _burr(self, msg):
        self._arrived += 1
        size = int(msg.size.data)
        if size > 0:
            self.burr_mm = size

    def _camera(self, msg):
        self._arrived += 1
        pose = msg.pose
        self.camera = {
            "position": np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            "rotation": _quat_matrix(
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ),
        }

    def _pose(self, msg):
        pose = msg.pose
        return {
            "position": np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            "quat": (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        }

    def _drill(self, msg):
        self._arrived += 1
        self.drill = self._pose(msg)

    def _voxels(self, msg):
        self._arrived += 1
        try:
            batch_id = int(msg.header.frame_id)
        except ValueError:
            return
        voxels = [(int(p.x), int(p.y), int(p.z)) for p in msg.indices]
        with self._lock:
            self._incoming.append((time.perf_counter(), time.time(), batch_id, voxels))
            self._seen.add(batch_id)
            self._advance()


def _apply_camera(viewer, camera):
    rotation = camera["rotation"]
    # Chai looks along local -X, up is local +Z, right is local +Y.
    forward = -rotation[:, 0]
    up = rotation[:, 2]
    position = camera["position"]
    viewer.camera.pos = type(viewer.camera.pos)(float(position[0]), float(position[1]), float(position[2]))
    viewer.camera._set_orientation_from_direction(forward)
    viewer.camera.sync_pivot_to_view()
    # Roll about the view axis. The viewer's pitch/yaw camera keeps world-up,
    # so the matrix is replaced with chai's right and up.
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-8)
    rolled_up = np.cross(right, forward)
    viewer._fivrs_cam = (position, right, rolled_up, forward)
    if not getattr(viewer, "_aim_printed", False):
        print(
            f"aim pos {np.round(position, 3).tolist()} forward {np.round(forward, 3).tolist()} "
            f"pitch {viewer.camera.pitch:.1f} yaw {viewer.camera.yaw:.1f}",
            flush=True,
        )
        viewer._aim_printed = True
    viewer._camera_dirty = True
    origin = getattr(viewer, "_volume_translation", None)
    if origin is not None:
        viewer._cone_distance = max(float(np.linalg.norm(position - origin)), 0.5)
    _aim_lights(viewer, position, forward / max(np.linalg.norm(forward), 1e-8))


def _light_intensity(viewer, path):
    """Exposure, the beam toggle, and the cone's distance compensation.

    The disk and the dome use the numbers in tuner_light.json. The cone
    angle does not scale the intensity.
    """
    scale = float(getattr(viewer, "_exposure", 1.0))
    base = LIGHT_BASE[path]
    if path == "/root/_ScopeBeam":
        return base * scale if getattr(viewer, "_beam_on", False) else 0.0
    if path == "/root/_ScopeLight":
        distance = max(float(getattr(viewer, "_cone_distance", CONE_REFERENCE_DISTANCE)), 0.5)
        return base * scale * (distance / CONE_REFERENCE_DISTANCE) ** 2
    return base * scale


def _remember_cone_angle(angle):
    """Last half-angle the RTX window was left on, in degrees."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "cone_angle.txt").write_text(f"{float(angle):.1f}\n")


def _push_exposure(viewer):
    _reload_tuner_lights(viewer)
    for path in LIGHT_BASE:
        value = _light_intensity(viewer, path)
        prim = viewer.stage.GetPrimAtPath(path)
        attr = prim.GetAttribute("inputs:intensity") if prim else None
        if attr:
            attr.Set(value)
        try:
            viewer._write_runtime_attribute([path], "inputs:intensity", np.array([value], dtype=np.float32))
        except Exception as exc:
            if not getattr(viewer, "_exposure_warned", False):
                print(f"exposure write failed: {exc}", flush=True)
                viewer._exposure_warned = True
    angle = float(getattr(viewer, "_cone_angle", CONE_ANGLE_DEG))
    shaping = {
        "inputs:shaping:cone:angle": angle,
        "inputs:shaping:cone:softness": float(getattr(viewer, "_cone_softness", CONE_SOFTNESS)),
        "inputs:shaping:focus": CONE_FOCUS,
        "inputs:radius": float(getattr(viewer, "_cone_radius", CONE_RADIUS)),
    }
    light = viewer.stage.GetPrimAtPath("/root/_ScopeLight")
    for name, value in shaping.items():
        attr = light.GetAttribute(name) if light else None
        if attr:
            attr.Set(value)
        try:
            viewer._write_runtime_attribute(
                ["/root/_ScopeLight"], name, np.array([value], dtype=np.float32)
            )
        except Exception as exc:
            if not getattr(viewer, "_cone_warned", False):
                print(f"cone write failed: {exc}", flush=True)
                viewer._cone_warned = True
    viewer._exposure_dirty = False


def _set_cone_angle(viewer, angle):
    viewer._cone_angle = float(min(CONE_ANGLE_MAX, max(CONE_ANGLE_MIN, angle)))
    _remember_cone_angle(viewer._cone_angle)
    print(
        f"RTX cone {viewer._cone_angle:.0f}°   - narrower   = wider",
        flush=True,
    )


def _set_exposure(viewer, scale):
    viewer._exposure = float(min(EXPOSURE_MAX, max(EXPOSURE_MIN, scale)))
    viewer._exposure_dirty = True
    print(f"RTX exposure {viewer._exposure:.2f}   [ darker   ] brighter   \\ reset", flush=True)


def _bind_exposure_keys(viewer):
    """Attach once the presentation window exists.

    ViewerRTX creates that window on the first displayed frame, so a bind
    during setup finds nothing and the keys never arrive.
    """
    if getattr(viewer, "_exposure_keys_bound", False):
        return
    window = getattr(viewer, "_window", None)
    if window is None:
        return
    import pyglet.window.key as key

    def on_key_press(symbol, modifiers):
        if symbol == key.BRACKETLEFT:
            _set_exposure(viewer, viewer._exposure / 1.25)
        elif symbol == key.BRACKETRIGHT:
            _set_exposure(viewer, viewer._exposure * 1.25)
        elif symbol == key.BACKSLASH:
            _set_exposure(viewer, 1.0)
        elif symbol == key.L and not (modifiers & (key.MOD_CTRL | key.MOD_ALT | key.MOD_SHIFT)):
            viewer._beam_on = not getattr(viewer, "_beam_on", False)
            viewer._exposure_dirty = True
            state = "on" if viewer._beam_on else "off"
            print(f"RTX distant beam {state}", flush=True)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT, key.EQUAL, key.PLUS, key.NUM_ADD):
            step = 10.0 if modifiers & key.MOD_SHIFT else 2.0
            if symbol in (key.MINUS, key.NUM_SUBTRACT):
                step = -step
            _set_cone_angle(viewer, getattr(viewer, "_cone_angle", CONE_ANGLE_DEG) + step)
            return True
        else:
            return None
        return True

    window.push_handlers(on_key_press=on_key_press)
    viewer._exposure_keys_bound = True
    print(
        "RTX lighting keys ready: [ darker   ] brighter   \\ reset   "
        "- narrower cone   = wider cone   L distant beam",
        flush=True,
    )


def _pan_scale(viewer, bridge):
    """World units per pixel, from the current camera-to-volume distance."""
    distance = 5.0
    camera = getattr(bridge, "camera", None)
    volume = getattr(bridge, "volume", None)
    if camera is not None and volume is not None:
        distance = float(np.linalg.norm(camera["position"] - volume["translation"]))
    distance = max(distance, 0.5)
    height = max(float(getattr(viewer.camera, "height", 720.0)), 1.0)
    return 2.0 * distance * float(np.tan(VERTICAL_FOV_RAD * 0.5)) / height


def _rtx_mouse_drag(viewer, bridge, x, y, dx, dy, buttons, to_framebuffer_coords, modifiers=0):
    import pyglet

    del x, y, to_framebuffer_coords, modifiers
    if buttons & pyglet.window.mouse.LEFT:
        bridge.publish_camera_nudge(yaw=-float(dx) * 0.25, pitch=float(dy) * 0.25)
    elif buttons & (pyglet.window.mouse.RIGHT | pyglet.window.mouse.MIDDLE):
        scale = _pan_scale(viewer, bridge)
        bridge.publish_camera_nudge(pan_right=-float(dx) * scale, pan_up=-float(dy) * scale)


def _rtx_mouse_scroll(viewer, bridge, scroll_y, is_ctrl_down=False):
    del viewer, is_ctrl_down
    if abs(float(scroll_y)) < 1e-6:
        return
    bridge.publish_camera_nudge(dolly=float(scroll_y) * 0.15)


def _bind_window_input(viewer, bridge):
    """Camera gestures go to AMBF. The view keeps following main_camera.

    The viewer's own drag handler is installed before ImGui, and ImGui sits
    above it. Handlers pushed after the window exists are dispatched first.
    """
    if getattr(viewer, "_window_input_bound", False):
        return
    window = getattr(viewer, "_window", None)
    if window is None:
        return
    _bind_exposure_keys(viewer)

    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        _rtx_mouse_drag(viewer, bridge, x, y, dx, dy, buttons, None, modifiers)
        return True

    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        del x, y, scroll_x
        _rtx_mouse_scroll(viewer, bridge, scroll_y)
        return True

    window.push_handlers(on_mouse_drag, on_mouse_scroll)
    gui = getattr(viewer, "gui", None)
    if gui is not None:
        gui.handle_mouse_drag = lambda *args, **kwargs: None
        gui.handle_mouse_scroll = lambda *args, **kwargs: None
        gui.update_camera_from_keys = lambda *args, **kwargs: None
    viewer._window_input_bound = True
    print("RTX camera: left-drag orbit, right-drag pan, wheel dolly. AMBF keeps main_camera.", flush=True)


def _log_chunk(viewer, name, points, tris, normals, color):
    """Update an existing prim from host arrays.

    ViewerRTX downloads a Warp array back to the host on every dynamic
    update, so wrapping arrays that are already numpy only adds a transfer.
    A new prim still has to be created from Warp arrays.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    indices = np.ascontiguousarray(tris.reshape(-1), dtype=np.int32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    qualified = viewer._qualify(name)
    if qualified in getattr(viewer, "_mesh_prim_paths", {}):
        viewer.log_mesh(
            name,
            points,
            indices,
            normals=normals,
            color=color,
            roughness=BONE_ROUGHNESS,
            metallic=0.0,
            dynamic=True,
            backface_culling=False,
        )
        return
    viewer.log_mesh(
        name,
        wp.array(points, dtype=wp.vec3),
        wp.array(indices, dtype=wp.int32),
        normals=wp.array(normals, dtype=wp.vec3),
        color=color,
        roughness=BONE_ROUGHNESS,
        metallic=0.0,
        dynamic=True,
        backface_culling=False,
    )


def _color_tuple(packed: int):
    return ((packed & 255) / 255.0, ((packed >> 8) & 255) / 255.0, ((packed >> 16) & 255) / 255.0)


def _send_chunk(viewer, key, points, tris, normals, color):
    spatial, packed = key
    names = getattr(viewer, "_chunk_names", {})
    name = names.get(key, f"anatomy/bone_{spatial}_{packed:06x}")
    try:
        _log_chunk(viewer, name, points, tris, normals, color)
    except Exception as exc:
        name = f"anatomy/bone_{spatial}_{packed:06x}_{time.time_ns()}"
        print(f"chunk {key} resize failed ({exc}); new prim", flush=True)
        _log_chunk(viewer, name, points, tris, normals, color)
    names[key] = name
    viewer._chunk_names = names


_EMPTY_POINTS = np.zeros((3, 3), np.float32)
_EMPTY_TRIS = np.array([[0, 1, 2]], np.int32)
_EMPTY_NORMALS = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (3, 1))


def _face_labels(mesh, faces):
    """One segment color per triangle, packed RGB. The majority vertex wins."""
    rgb = mesh.colors[faces]
    packed = (
        rgb[:, :, 0].astype(np.int32)
        | (rgb[:, :, 1].astype(np.int32) << 8)
        | (rgb[:, :, 2].astype(np.int32) << 16)
    )
    first, second, third = packed[:, 0], packed[:, 1], packed[:, 2]
    agree = (first == second) | (first == third)
    return np.where(agree, first, np.where(second == third, second, first))


def _spatial_key(block):
    bx, by, bz = block
    return int(bx + by * 16 + bz * 256)


def _prepare_block_mesh(mesh, volume, spatial):
    """Color-split one block. A missing color is cleared later from this spatial id."""
    marker = (spatial, None, _EMPTY_POINTS, _EMPTY_TRIS, _EMPTY_NORMALS)
    if len(mesh.tris) == 0:
        return [marker]
    dims = volume["dimensions"]
    counts = volume["counts"]
    rotation = volume["rotation"]
    translation = volume["translation"]
    if getattr(mesh, "world", None) is not None and getattr(mesh, "normals", None) is not None:
        world = mesh.world
        normals = mesh.normals
    else:
        world = local_to_world(index_to_local(mesh.verts, dims, counts), rotation, translation).astype(np.float32)
        normals = vertex_normals(world.astype(np.float64), mesh.tris).astype(np.float32)
    labels = _face_labels(mesh, mesh.tris)
    order = np.argsort(labels, kind="mergesort")
    ordered = labels[order]
    breaks = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(order)]))
    chunks = [marker]
    for start, end in zip(starts, ends):
        group = mesh.tris[order[start:end]]
        sub_used, sub_inverse = np.unique(group, return_inverse=True)
        chunks.append(
            (
                spatial,
                int(ordered[start]),
                world[sub_used],
                sub_inverse.reshape(-1, 3).astype(np.int32),
                normals[sub_used],
            )
        )
    return chunks


def _prepare_blocks(blocks, volume, dirty):
    """Upload payloads for the rebuilt blocks. ``dirty`` None means every block."""
    chosen = blocks.meshes if dirty is None else {key: blocks.meshes[key] for key in dirty if key in blocks.meshes}
    chunks = []
    for block, mesh in chosen.items():
        if dirty is None and len(mesh.tris) == 0:
            continue
        chunks.extend(_prepare_block_mesh(mesh, volume, _spatial_key(block)))
    return chunks


def _begin_upload(chunks, full, extract_ms, receipt, serials=None, detail=None, resync=False):
    return {
        "chunks": chunks,
        "full": full,
        "resync": resync,
        "cursor": 0,
        "sent": set(),
        "dirty": set(),
        "stale": None,
        "extract_ms": extract_ms,
        "receipt": receipt,
        "serials": list(serials or []),
        "detail": detail,
        "enqueued": time.perf_counter(),
        "host_log_ms": 0.0,
    }


def _drain_upload(viewer, upload, budget_s):
    """Send part of one update. The drill pose is applied on the frames in between."""
    started = time.perf_counter()
    chunks = upload["chunks"]
    while upload["cursor"] < len(chunks) and time.perf_counter() - started < budget_s:
        spatial, color, points, tris, normals = chunks[upload["cursor"]]
        upload["cursor"] += 1
        upload["dirty"].add(spatial)
        if color is None:
            continue
        if not getattr(viewer, "_bbox_printed", False) and len(points):
            print(
                f"bone chunk world min {points.min(0).round(3).tolist()} max {points.max(0).round(3).tolist()} "
                f"fov {viewer.camera.fov:.2f}",
                flush=True,
            )
            viewer._bbox_printed = True
        key = (spatial, color)
        _send_chunk(viewer, key, points, tris, normals, _color_tuple(color))
        upload["sent"].add(key)
    if upload["cursor"] < len(chunks):
        return False
    if upload["stale"] is None:
        names = getattr(viewer, "_chunk_names", {})
        upload["stale"] = [
            key for key in names if key not in upload["sent"] and (upload["full"] or key[0] in upload["dirty"])
        ]
    while upload["stale"] and time.perf_counter() - started < budget_s:
        key = upload["stale"].pop()
        _send_chunk(viewer, key, _EMPTY_POINTS, _EMPTY_TRIS, _EMPTY_NORMALS, (0.0, 0.0, 0.0))
    return not upload["stale"]


class SurfaceWorker:
    """Rebuilds drilled blocks away from the camera and drill update."""

    def __init__(self, blocks, occupied, colors, volume):
        self._blocks = blocks
        self._occupied = occupied
        self._colors = colors
        self._volume = volume
        self._inbox = queue.Queue()
        self._outbox = queue.Queue()
        self.busy = False
        self._replay = []
        self._thread = threading.Thread(target=self._run, name="fivrs-rtx-surface", daemon=True)
        self._thread.start()

    def submit_cuts(self, batches):
        self._inbox.put(("cut", batches))

    def submit_catchup(self, batches):
        self._inbox.put(("catchup", batches))

    def submit_resync(self, dump_path, reason="gap"):
        self._inbox.put(("resync", (str(dump_path), reason)))

    def submit_export(self, path):
        self._inbox.put(("export", str(path)))

    def poll(self):
        try:
            return self._outbox.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        try:
            from warp_mesh import STAGE, WarpMesher

            if STAGE >= 1:
                self._blocks._warp = WarpMesher()
                self._blocks._warp.warmup(self._occupied, self._colors)
                print(f"Warp mesh kernels warmed, stage {STAGE}", flush=True)
        except Exception:
            import traceback
            print("Warp mesh warmup failed; CPU meshing stays in use", flush=True)
            traceback.print_exc()
            self._blocks._warp = None
        while True:
            jobs = [self._inbox.get()]
            while True:
                try:
                    jobs.append(self._inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                self._apply(jobs)
            except Exception:
                import traceback
                print("surface update failed:", flush=True)
                traceback.print_exc()

    def _apply(self, jobs):
        self.busy = True
        try:
            exports = [payload for op, payload in jobs if op == "export"]
            mesh_jobs = [job for job in jobs if job[0] != "export"]
            if mesh_jobs:
                self._apply_jobs(mesh_jobs)
            for path in exports:
                destination = pathlib.Path(path)
                np.save(destination, self._occupied)
                destination.with_suffix(".ready").write_text("ok\n")
        finally:
            self.busy = False

    def _apply_jobs(self, jobs):
        started = time.perf_counter()
        receipt = None
        receipt_wall = None
        shape = self._occupied.shape
        detail = {
            "batches": 0,
            "replace_calls": 0,
            "voxels": 0,
            "voxel_clear_ms": 0.0,
            "copy_ms": 0.0,
            "extract_ms": 0.0,
            "edge_ms": 0.0,
            "rewrite_ms": 0.0,
            "stitch_ms": 0.0,
            "smooth_ms": 0.0,
            "prepare_ms": 0.0,
            "prepare_gpu_ms": 0.0,
            "transfer_ms": 0.0,
            "download_bytes": 0,
            "upload_bytes": 0,
            "kept_tris": 0,
            "new_tris": 0,
            "blocks": 0,
            "queue_wait_ms": 0.0,
            "worker_start": started,
            "serials": [],
            "reason": None,
        }
        resync = None
        catchup = False
        cut_payloads = []
        for op, payload in jobs:
            if op == "resync":
                resync = payload
            else:
                if op == "catchup":
                    catchup = True
                cut_payloads.append(payload)
        dirty = set()
        if resync is not None:
            path, reason = resync if isinstance(resync, tuple) else (resync, "gap")
            loading = time.perf_counter()
            occupied, colors, _count = _load_dump(pathlib.Path(path))
            detail["load_ms"] = (time.perf_counter() - loading) * 1e3
            if occupied.shape != shape:
                print(f"resync shape {occupied.shape} != {shape}", flush=True)
            else:
                diffing = time.perf_counter()
                dirty.update(
                    self._blocks.changed_blocks(self._occupied, self._colors, occupied, colors)
                )
                detail["diff_ms"] = (time.perf_counter() - diffing) * 1e3
                self._occupied = occupied
                self._colors = colors
                detail["reason"] = reason
                if reason == "restore":
                    self._replay.clear()
                else:
                    replay_x = []
                    replay_y = []
                    replay_z = []
                    for _batch_id, replay_voxels in self._replay:
                        for x, y, z in replay_voxels:
                            if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                                continue
                            self._occupied[x, y, z] = False
                            self._colors[x, y, z] = 0
                            replay_x.append(x)
                            replay_y.append(y)
                            replay_z.append(z)
                    if replay_x:
                        dirty.update(blocks_touched_by(replay_x, replay_y, replay_z, shape))
        if catchup and detail["reason"] is None:
            detail["reason"] = "catchup"
        clearing = time.perf_counter()
        xs = []
        ys = []
        zs = []
        for payload in cut_payloads:
            for stamp, wall, serial, voxels in payload:
                receipt = stamp if receipt is None else min(receipt, stamp)
                receipt_wall = wall if receipt_wall is None else min(receipt_wall, wall)
                detail["serials"].append(int(serial))
                detail["batches"] += 1
                detail["voxels"] += len(voxels)
                self._replay.append((int(serial), voxels))
                if len(self._replay) > 8192:
                    del self._replay[: len(self._replay) - 8192]
                for x, y, z in voxels:
                    if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                        continue
                    self._occupied[x, y, z] = False
                    self._colors[x, y, z] = 0
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
        detail["voxel_clear_ms"] = (time.perf_counter() - clearing) * 1e3
        if xs:
            dirty.update(blocks_touched_by(xs, ys, zs, shape))
        if not dirty and resync is None:
            return
        if self._volume is not None:
            volume = self._volume
            self._blocks.pose = (
                volume["dimensions"],
                volume["counts"],
                volume["rotation"],
                volume["translation"],
            )
        phases = self._blocks.rebuild(
            self._occupied,
            self._colors,
            dirty,
            cleared=None if resync is not None else (xs, ys, zs),
            replace=resync is not None,
        )
        # One unioned rebuild for every bite already queued, not one per bite.
        detail["replace_calls"] = 1 if dirty else 0
        for key in (
            "copy_ms", "extract_ms", "edge_ms", "rewrite_ms", "stitch_ms", "smooth_ms",
            "prepare_gpu_ms", "transfer_ms", "download_bytes", "upload_bytes", "new_tris", "blocks",
        ):
            detail[key] += phases[key]
        if receipt is not None:
            detail["queue_wait_ms"] = (started - receipt) * 1e3
        detail["receipt_wall"] = receipt_wall
        preparing = time.perf_counter()
        chunks = _prepare_blocks(self._blocks, self._volume, dirty)
        detail["prepare_ms"] = (time.perf_counter() - preparing) * 1e3 + phases.get("prepare_gpu_ms", 0.0)
        detail["mesh_ms"] = (
            phases["extract_ms"]
            + phases["edge_ms"]
            + phases["smooth_ms"]
            + detail["prepare_ms"]
            + phases.get("transfer_ms", 0.0)
        )
        detail["triangles"] = self._blocks.triangle_count
        detail["worker_done"] = time.perf_counter()
        print(
            f"blocks {detail['blocks']} bites {detail['batches']} "
            f"extract {detail['extract_ms']:.0f} ms smooth {detail['smooth_ms']:.0f} ms "
            f"prepare {detail['prepare_ms']:.0f} ms",
            flush=True,
        )
        self._outbox.put(
            {
                "chunks": chunks,
                "full": False,
                "resync": resync is not None,
                "extract_ms": (time.perf_counter() - started) * 1e3,
                "receipt": receipt,
                "serials": detail["serials"],
                "detail": detail,
            }
        )


def _append_timing(event):
    with (RUNTIME / "rtx_timing.jsonl").open("a") as handle:
        handle.write(json.dumps(event) + "\n")


def _finish_presented(viewer, token):
    """Record the render product that was submitted after the attribute write.

    ``display_accept_wall`` is when that product is handed to the window.
    Monitor vblank is not visible from here. Writing the mesh is earlier.
    """
    receipt = token.get("receipt_wall")
    display = token.get("display_accept_wall")
    worst = 0.0
    counted = 0
    stalls_50 = 0
    stalls_100 = 0
    for wall, frame_ms in viewer._frame_samples:
        if receipt is not None and wall + 0.05 < receipt:
            continue
        if display is not None and wall > display + 0.05:
            continue
        counted += 1
        worst = max(worst, frame_ms)
        if frame_ms > 50.0:
            stalls_50 += 1
        if frame_ms > 100.0:
            stalls_100 += 1
    _append_timing(
        {
            "kind": "presented",
            "serials": token.get("serials") or [],
            "attribute_write_ms": token.get("attribute_write_ms"),
            "attribute_wall": token.get("attribute_wall"),
            "product_ready_wall": token.get("product_ready_wall"),
            "display_accept_wall": display,
            "reason": token.get("reason"),
            "host_log_ms": token.get("host_log_ms"),
            "upload_spread_ms": token.get("upload_spread_ms"),
            "queued_wall": token.get("queued_wall"),
            "frame_ms_max": worst,
            "frames": counted,
            "stalls_over_50ms": stalls_50,
            "stalls_over_100ms": stalls_100,
        }
    )


def _install_frame_probes(viewer):
    """Time attribute write and the later completed frame without changing the draw."""
    viewer._bone_queued = None
    viewer._bone_in_flight = None
    viewer._bone_display = None
    viewer._frame_samples = []
    update_scene = viewer._update_scene
    mesh_points = viewer._update_ovrtx_mesh_points
    accept_render = viewer._accept_render
    end_frame = viewer.end_frame

    def timed_mesh_points():
        started = time.perf_counter()
        mesh_points()
        viewer._mesh_write_ms = (time.perf_counter() - started) * 1e3
        viewer._mesh_write_wall = time.time()

    def timed_update_scene():
        if viewer._bone_in_flight is not None:
            viewer._bone_in_flight["product_ready_wall"] = time.time()
            viewer._bone_display = viewer._bone_in_flight
            viewer._bone_in_flight = None
        update_scene()
        queued = viewer._bone_queued
        if queued is not None:
            queued["attribute_write_ms"] = float(getattr(viewer, "_mesh_write_ms", 0.0))
            queued["attribute_wall"] = float(getattr(viewer, "_mesh_write_wall", time.time()))
            viewer._bone_in_flight = queued
            viewer._bone_queued = None

    def timed_accept(products):
        token = viewer._bone_display
        if token is not None:
            token["display_accept_wall"] = time.time()
            _finish_presented(viewer, token)
            viewer._bone_display = None
        accept_render(products)

    def timed_end():
        end_frame()
        token = viewer._bone_display
        if token is not None:
            token["display_accept_wall"] = None
            _finish_presented(viewer, token)
            viewer._bone_display = None

    viewer._update_ovrtx_mesh_points = timed_mesh_points
    viewer._update_scene = timed_update_scene
    viewer._accept_render = timed_accept
    viewer.end_frame = timed_end


def _log_instance(viewer, name, mesh, position, quat, scale, hidden=False):
    xform = wp.transform(
        wp.vec3(float(position[0]), float(position[1]), float(position[2])),
        wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )
    viewer.log_instances(
        name,
        mesh,
        wp.array([xform], dtype=wp.transform),
        wp.array([wp.vec3(float(scale), float(scale), float(scale))], dtype=wp.vec3),
        None,
        None,
        hidden=hidden,
    )


def _log_static_mesh(viewer, name, points, tris, normals, color, roughness, metallic):
    # The mesh is a prototype. Leaving it visible draws it a second time at
    # the origin, which is where the giant drills were coming from.
    viewer.log_mesh(
        name,
        wp.array(np.ascontiguousarray(points), dtype=wp.vec3),
        wp.array(np.ascontiguousarray(tris.reshape(-1)), dtype=wp.int32),
        normals=wp.array(np.ascontiguousarray(normals), dtype=wp.vec3),
        color=color,
        roughness=roughness,
        metallic=metallic,
        dynamic=False,
        hidden=True,
    )


def _read_drill_state():
    try:
        return json.loads((RUNTIME / "drill_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _visual_pose(state):
    """Pose of the burr AMBF is drawing. The sized body has no ROS topic."""
    try:
        return {
            "position": np.array(
                [float(state["px"]), float(state["py"]), float(state["pz"])], dtype=np.float64
            ),
            "quat": (float(state["qx"]), float(state["qy"]), float(state["qz"]), float(state["qw"])),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _log_instrument(viewer, bridge):
    """Shaft plus the same tip AMBF draws, at the cutter radius rather than around it."""
    state = _read_drill_state()
    size = int(state.get("size_mm") or bridge.burr_mm or 6)
    pose = _visual_pose(state) or bridge.drill
    if pose is None:
        return
    radius = float(size) * MM_TO_SIM * 0.5
    now = time.perf_counter()
    previous = getattr(viewer, "_burr_spin_t", now)
    angle = getattr(viewer, "_burr_spin", 0.0)
    if state.get("on"):
        # Four visible turns per second, matching DrillManager.
        angle += (now - previous) * 8.0 * np.pi
    viewer._burr_spin = angle
    viewer._burr_spin_t = now
    meshes = getattr(viewer, "_instrument_meshes", {})
    cutting = state.get("style") != "diamond"
    head = meshes.get("cutting") or "drill/sphere"
    head_quat = _mul_quat(pose["quat"], _spin_about_x(angle)) if cutting else pose["quat"]
    _log_instance(
        viewer, "drill/cutting", head, pose["position"], head_quat, radius, hidden=not cutting
    )
    _log_instance(
        viewer, "drill/diamond", "drill/sphere", pose["position"], pose["quat"], radius, hidden=cutting
    )
    for shaft_size, mesh in meshes.get("shafts", {}).items():
        _log_instance(
            viewer,
            f"drill/shaft_inst_{shaft_size}",
            mesh,
            pose["position"],
            pose["quat"],
            1.0,
            hidden=shaft_size != size,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy-adf", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    import newton

    print("Loading anatomy slices", flush=True)
    occupied, colors = _load_occupancy(pathlib.Path(args.anatomy_adf))
    labels = np.unique(colors[occupied].reshape(-1, 3), axis=0)
    print(f"Volume {occupied.shape} with {len(labels)} segment colors", flush=True)
    bridge = Bridge()
    deadline = time.perf_counter() + 45.0
    while bridge.volume is None and time.perf_counter() < deadline:
        bridge.spin()
        time.sleep(0.05)
    if bridge.volume is None:
        raise SystemExit("AMBF did not publish volume_info. Start the simulator first.")
    counts = bridge.volume["counts"]
    if tuple(int(v) for v in counts) != occupied.shape:
        print(f"WARNING volume_info count {counts.tolist()} != slices {occupied.shape}", flush=True)

    started = time.perf_counter()
    blocks = BlockVolume(occupied, colors)
    extract_s = time.perf_counter() - started
    print(
        f"Initial surface {blocks.vertex_count} verts {blocks.triangle_count} tris "
        f"in {len(blocks.meshes)} blocks in {extract_s:.2f}s",
        flush=True,
    )

    _patch = None
    import newton.viewer as viewer_mod

    # OVStage time_code mismatch, same patch the surreal viewer uses.
    try:
        import math
        import ovstage.population as population

        original = population.open_usd

        def open_usd(stage, path, ordinal, time_code=None, time=math.nan, **kwargs):
            if time_code is not None:
                time = float(time_code)
            return original(stage, path, ordinal, time=time, **kwargs)

        population.open_usd = open_usd
    except Exception as exc:
        print(f"OVStage patch skipped: {exc}", flush=True)

    viewer = newton.viewer.ViewerRTX(
        width=args.width,
        height=args.height,
        vsync=False,
        headless=False,
        up_axis="Z",
        environment="none",
        async_rendering=True,
        render_settings={
            "omni:rtx:pt:samplesPerPixel": ("uint", 1),
            "omni:rtx:pt:denoising:enabled": ("bool", True),
            "omni:rtx:reflections:denoiser:enabled": ("bool", True),
            "omni:rtx:indirectDiffuse:denoiser:enabled": ("bool", False),
            "omni:rtx:dlss:frameGeneration": ("bool", False),
            "omni:rtx:pt:maxBounces": ("int", 4),
        },
    )
    model = newton.ModelBuilder().finalize()
    viewer.set_model(model)
    # set_model replaces the camera, so the field of view has to be set after it.
    viewer.camera.fov = float(np.degrees(VERTICAL_FOV_RAD))
    viewer.camera.near = 0.1
    viewer.camera.far = 100.0
    viewer.camera.width = args.width
    viewer.camera.height = args.height
    _author_lights(viewer)
    _bind_lights(viewer)
    viewer.show_ui = False
    if hasattr(viewer, "show_ground"):
        viewer.show_ground = False
    viewer.picking_enabled = False

    sphere_local, sphere_tris = _sphere(1.0)
    sphere_normals = vertex_normals(sphere_local.astype(np.float64), sphere_tris).astype(np.float32)
    _log_static_mesh(
        viewer, "drill/sphere", sphere_local, sphere_tris, sphere_normals, (0.86, 0.87, 0.88), 0.12, 0.95
    )
    instrument = {"cutting": "drill/sphere", "shafts": {}}
    flute_path = MESH_DIR / "fluted_burr.obj"
    try:
        flute_points, flute_tris, flute_normals = _load_obj(flute_path)
        _log_static_mesh(
            viewer, "drill/flutes", flute_points, flute_tris, flute_normals, (0.82, 0.83, 0.85), 0.18, 0.92
        )
        instrument["cutting"] = "drill/flutes"
        print(f"Fluted burr {len(flute_tris)} tris", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Fluted burr unavailable ({exc}); using the diamond sphere", flush=True)
    for size in (1, 2, 4, 6):
        shaft_path = MESH_DIR / f"{size}mm.OBJ"
        try:
            shaft_points, shaft_tris, shaft_normals = _load_obj(shaft_path)
            _log_static_mesh(
                viewer,
                f"drill/shaft_{size}",
                shaft_points,
                shaft_tris,
                shaft_normals,
                (0.55, 0.56, 0.58),
                0.35,
                0.75,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Shaft {size} mm unavailable ({exc})", flush=True)
            continue
        instrument["shafts"][size] = f"drill/shaft_{size}"
        print(f"Shaft {size} mm {len(shaft_tris)} tris", flush=True)
    viewer._instrument_meshes = instrument
    viewer._volume_translation = np.asarray(bridge.volume["translation"], dtype=np.float64)
    uploads = [_begin_upload(_prepare_blocks(blocks, bridge.volume, None), True, extract_s * 1e3, None)]
    worker = SurfaceWorker(blocks, occupied, colors, bridge.volume)
    _install_frame_probes(viewer)
    print("RTX brightness: [ darker    ] brighter    \\ reset    L distant beam", flush=True)
    waiting_dump_since = None
    waiting_catchup = None
    catchup_after = 0
    frame_ms = []
    removal_delay_ms = []
    last_extract_ms = extract_s * 1e3
    last_upload_ms = 0.0
    last_report = time.perf_counter()
    previous = time.perf_counter()
    frame = 0
    mesh_generation = 0
    RUNTIME.mkdir(parents=True, exist_ok=True)

    while viewer.is_running():
        batches, missing = bridge.take_batches()
        now = time.perf_counter()
        wall_now = time.time()
        if missing[0] is not None and waiting_catchup is None and waiting_dump_since is None:
            first, last = int(missing[0]), int(missing[1])
            if last - first > 8192:
                print(
                    f"Catch-up span {first}-{last} is wider than retained history; "
                    "falling back to the occupancy file",
                    flush=True,
                )
                bridge.assume_through(last)
                (RUNTIME / "occupancy.bin").unlink(missing_ok=True)
                (RUNTIME / "resync.request").write_text("gap\n")
                waiting_dump_since = wall_now
            else:
                (RUNTIME / "catchup.bin").unlink(missing_ok=True)
                (RUNTIME / "catchup.request").write_text(f"{first} {last}\n")
                waiting_catchup = (wall_now, first, last)
                print(f"Catch-up requested for batches {first}-{last}", flush=True)
        if waiting_catchup is not None:
            catchup_path = RUNTIME / "catchup.bin"
            requested_at, first, last = waiting_catchup
            if catchup_path.is_file() and catchup_path.stat().st_mtime >= requested_at - 0.05:
                history_missing, found = _read_catchup(catchup_path)
                catchup_path.unlink(missing_ok=True)
                waiting_catchup = None
                if found:
                    print(f"Catch-up {sum(len(voxels) for _batch, voxels in found)} voxels", flush=True)
                    recovered = [
                        (time.perf_counter(), time.time(), int(batch_id), voxels)
                        for batch_id, voxels in found
                    ]
                    for stamp, wall, batch_id, voxels in recovered:
                        _append_timing(
                            {
                                "kind": "receipt",
                                "serial": int(batch_id),
                                "n": len(voxels),
                                "wall": wall,
                                "stamp": stamp,
                                "recovered": True,
                            }
                        )
                    worker.submit_catchup(recovered)
                    bridge.note_batches(batch_id for batch_id, _voxels in found)
                if history_missing:
                    print(
                        f"Catch-up history missed part of {first}-{last}; "
                        "falling back to the occupancy file",
                        flush=True,
                    )
                    bridge.note_batches(range(int(first), int(last) + 1))
                    (RUNTIME / "occupancy.bin").unlink(missing_ok=True)
                    (RUNTIME / "resync.request").write_text("gap\n")
                    waiting_dump_since = wall_now
            elif wall_now - requested_at > 2.0:
                print("Catch-up file did not arrive", flush=True)
                waiting_catchup = None
        if waiting_dump_since is not None:
            dump = RUNTIME / "occupancy.bin"
            if dump.is_file() and dump.stat().st_mtime >= waiting_dump_since - 0.05:
                worker.submit_resync(dump)
                waiting_dump_since = None
                print("Resync queued from the live volume", flush=True)
            elif wall_now - waiting_dump_since > 2.0:
                print("Resync dump did not arrive", flush=True)
                waiting_dump_since = None
        reload_request = RUNTIME / "reload.request"
        if waiting_dump_since is None and reload_request.is_file():
            dump = RUNTIME / "occupancy.bin"
            if dump.is_file():
                worker.submit_resync(dump, "restore")
                reload_request.unlink()
                print("Restore reload queued", flush=True)
        export_request = RUNTIME / "rtx_occupied.request"
        if export_request.is_file():
            export_request.unlink()
            target = RUNTIME / "rtx_occupied.npy"
            target.unlink(missing_ok=True)
            target.with_suffix(".ready").unlink(missing_ok=True)
            worker.submit_export(target)
        if batches:
            handed = batches
            inbox = worker._inbox.qsize()
            busy = bool(worker.busy)
            for stamp, wall, serial, voxels in handed:
                _append_timing(
                    {
                        "kind": "receipt",
                        "serial": int(serial),
                        "n": len(voxels),
                        "wall": wall,
                        "stamp": stamp,
                        "pending_batches": len(handed),
                        "inbox": inbox,
                        "worker_busy": busy,
                        "recovered": False,
                    }
                )
            worker.submit_cuts(handed)

        while True:
            result = worker.poll()
            if result is None:
                break
            uploads.append(
                _begin_upload(
                    result["chunks"],
                    result["full"],
                    result["extract_ms"],
                    result["receipt"],
                    result.get("serials"),
                    result.get("detail"),
                    result.get("resync", False),
                )
            )
            detail = result.get("detail")
            if detail is not None:
                _append_timing({"kind": "surface", **detail})

        viewer.begin_frame(frame * (1.0 / 60.0))
        if bridge.camera is not None:
            _apply_camera(viewer, bridge.camera)
        if uploads:
            t0 = time.perf_counter()
            finished = _drain_upload(viewer, uploads[0], UPLOAD_BUDGET_S)
            uploads[0]["host_log_ms"] += (time.perf_counter() - t0) * 1e3
            if finished:
                done = uploads.pop(0)
                last_extract_ms = done["extract_ms"]
                if done["receipt"] is not None:
                    removal_delay_ms.append((time.perf_counter() - done["receipt"]) * 1e3)
                    if len(removal_delay_ms) > 240:
                        del removal_delay_ms[:-240]
                detail = done.get("detail") or {}
                if done.get("serials") or done.get("resync") or detail.get("reason"):
                    viewer._bone_queued = {
                        "serials": done.get("serials") or [],
                        "receipt_wall": detail.get("receipt_wall"),
                        "host_log_ms": done["host_log_ms"],
                        "upload_spread_ms": (time.perf_counter() - done["enqueued"]) * 1e3,
                        "queued_wall": time.time(),
                        "reason": detail.get("reason"),
                    }
                if not uploads and (done["full"] or done.get("resync")):
                    mesh_generation += 1
                    (RUNTIME / "mesh_ready.json").write_text(
                        json.dumps(
                            {
                                "generation": mesh_generation,
                                "wall": time.time(),
                                "reason": (done.get("detail") or {}).get("reason"),
                            }
                        )
                    )
                    print(f"Bone upload finished, {len(getattr(viewer, '_chunk_names', {}))} meshes", flush=True)
            last_upload_ms = (time.perf_counter() - t0) * 1e3
        if bridge.drill is not None:
            _log_instrument(viewer, bridge)
        viewer.end_frame()
        _bind_window_input(viewer, bridge)
        frame += 1
        stamp = time.perf_counter()
        if previous > 0.0:
            elapsed_ms = (stamp - previous) * 1e3
            frame_ms.append(elapsed_ms)
            viewer._frame_samples.append((time.time(), elapsed_ms))
            if len(frame_ms) > 480:
                del frame_ms[:-480]
            if len(viewer._frame_samples) > 4000:
                del viewer._frame_samples[:-4000]
        previous = stamp
        if stamp - last_report > 1.0:
            last_report = stamp
            graphics_hz = physics_hz = None
            status_path = RUNTIME / "status.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text())
                    graphics_hz = status.get("graphics_hz")
                    physics_hz = status.get("physics_hz")
                except json.JSONDecodeError:
                    pass
            samples = np.asarray(frame_ms, dtype=np.float64) if frame_ms else np.zeros(1)
            delays = np.asarray(removal_delay_ms, dtype=np.float64) if removal_delay_ms else np.zeros(0)
            summary = {
                "graphics_hz": graphics_hz,
                "graphics_hz_note": "AMBF render loop, the on-screen Gfx counter. Not the RTX frame rate.",
                "physics_hz": physics_hz,
                "physics_hz_note": "AMBF physics loop, the on-screen Phx counter. Haptics run here.",
                "rtx_fps": 1000.0 / float(samples.mean()) if len(frame_ms) else 0.0,
                "rtx_fps_note": "Frames completed by this ViewerRTX window. A separate clock from graphics Hz.",
                "rtx_frame_ms_avg": float(samples.mean()) if len(frame_ms) else 0.0,
                "rtx_frame_ms_p95": float(np.percentile(samples, 95)) if len(frame_ms) else 0.0,
                "rtx_frame_ms_max": float(samples.max()) if len(frame_ms) else 0.0,
                "extract_ms_last": last_extract_ms,
                "upload_ms_last": last_upload_ms,
                "removal_to_surface_ms_avg": float(delays.mean()) if len(delays) else None,
                "removal_to_surface_ms_max": float(delays.max()) if len(delays) else None,
                "removal_to_surface_ms_note": "Receipt until chunk logging finishes. Not the displayed frame.",
                "triangles": blocks.triangle_count,
                "vertices": blocks.vertex_count,
            }
            (RUNTIME / "summary.json").write_text(json.dumps(summary, indent=2))
            title = (
                f"FIVRS RTX  {summary['rtx_fps']:.0f} fps"
                f"   exposure {getattr(viewer, '_exposure', 1.0):.2f}"
                f"   Gfx {graphics_hz} Hz   Phx {physics_hz} Hz"
                f"   extract {last_extract_ms:.0f} ms"
            )
            window = getattr(viewer, "_window", None)
            if window is not None:
                window.set_caption(title)
        if frame == 1:
            print(f"ViewerRTX first frame {(time.perf_counter() - started) * 1e3:.0f} ms", flush=True)

    bridge.close()
    viewer.close()


if __name__ == "__main__":
    main()
