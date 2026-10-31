#!/usr/bin/env python3
"""RTX window for the FIVRS bone.

AMBF stays in charge of the volume, the drill constraint, haptics, and the
camera. This process only draws. It reads the anatomy slices in the same order
the simulator does, subscribes to the camera, drill, and voxel-removal topics,
and paints each surface from the label color stored in that voxel.
Bone triangles that still lie on the lateral-cortex mask use that tuner
material. The cortical-markings mask is drawn again, just off that surface,
with the shader saved from the tuner. Removing a voxel still comes only from
the simulator.

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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bone_debris
import cortex_masks
import markings
import pause_menu
import rtx_materials
import rtx_config
from fluid.cannula import SIZES as CANNULA_SIZES, load_obj as load_cannula_obj, mesh_path as cannula_mesh_path
from fluid.irrigation_bench import (
    DRAW_SPHERES,
    DRAW_TUBE,
    IRRIGATION_PATH,
    IrrigationBench,
    load_irrigation,
    water_from_irrigation,
)
from fluid.occupancy import VolumeFrame
from fluid.surface_film import SurfaceFilm
from fluid.surface_sph import SurfaceSph, film_surface
from fluid.rtx_stream import PrescribedStream
from fluid.sph_dump import DEFAULT_PATH as SPH_DUMP, dump_is_fresh, read_positions
from fluid.water_mesh import water_surface
from surface import BlockVolume, blocks_touched_by, index_to_local, local_to_world, vertex_normals

# Chunk colors for bone faces that take a tuner appearance. They are not
# segmentation labels, and they never go back into the occupancy grid.
CORTEX_CHUNK = -1
MARKINGS_CHUNK = -2
MARKINGS_LIFT = 0.05 * cortex_masks.MM_TO_SIM

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


def _read_obj_groups(path):
    """Positions, normals, and face corners grouped by usemtl, in file order."""
    positions = []
    normals = []
    groups = {}
    current = ""
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                positions.append([float(item) for item in line.split()[1:4]])
            elif line.startswith("vn "):
                normals.append([float(item) for item in line.split()[1:4]])
            elif line.startswith("usemtl "):
                current = line.split(None, 1)[1].strip()
                groups.setdefault(current, [])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    parts = token.split("/")
                    vertex = int(parts[0]) - 1
                    normal = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else vertex
                    face.append((vertex, normal))
                corners = groups.setdefault(current, [])
                for index in range(1, len(face) - 1):
                    corners.extend((face[0], face[index], face[index + 1]))
    return positions, normals, groups


def _weld_corners(positions, normals, corners):
    """One mesh. Corners with the same position and normal stay one vertex."""
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
    if len(point_normals):
        lengths = np.linalg.norm(point_normals, axis=1, keepdims=True)
        point_normals /= np.maximum(lengths, 1e-8)
    tris = np.asarray(indices, dtype=np.int32).reshape(-1, 3) if indices else np.zeros((0, 3), dtype=np.int32)
    return points, tris, point_normals


def _merge_mesh(existing, mesh):
    if existing is None:
        return mesh
    points, tris, normals = existing
    extra_points, extra_tris, extra_normals = mesh
    return (
        np.concatenate((points, extra_points)),
        np.concatenate((tris, extra_tris + len(points))),
        np.concatenate((normals, extra_normals)),
    )


def _combine_shaft(parts):
    """One mesh per burr size. Face ranges keep each paint addressable."""
    points = []
    tris = []
    normals = []
    ranges = {}
    vertex = 0
    face = 0
    for part_id, (part_points, part_tris, part_normals) in parts.items():
        if len(part_tris) == 0:
            continue
        points.append(part_points)
        tris.append(part_tris + vertex)
        normals.append(part_normals)
        ranges[part_id] = (face, int(len(part_tris)))
        vertex += len(part_points)
        face += int(len(part_tris))
    if not points:
        return None
    return np.concatenate(points), np.concatenate(tris), np.concatenate(normals), ranges


def _load_obj(path):
    """Positions and authored normals. Corners stay split so flute edges stay sharp."""
    positions, normals, groups = _read_obj_groups(path)
    corners = []
    for group in groups.values():
        corners.extend(group)
    return _weld_corners(positions, normals, corners)


def _load_obj_parts(path):
    """Same triangles as _load_obj, one mesh per Blender material."""
    positions, normals, groups = _read_obj_groups(path)
    merged = {}
    for mtl, corners in groups.items():
        if not corners:
            continue
        part_id = rtx_materials.shaft_part_id(mtl)
        merged[part_id] = _merge_mesh(merged.get(part_id), _weld_corners(positions, normals, corners))
    ordered = {}
    for part_id in rtx_materials.SHAFT_PART_ORDER:
        if part_id in merged:
            ordered[part_id] = merged[part_id]
    for part_id, mesh in merged.items():
        if part_id not in ordered:
            ordered[part_id] = mesh
    return ordered


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
    rtx_materials.attach(viewer)
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
    rtx_materials.reload_if_changed(viewer)
    _sync_markings(viewer)
    _sync_irrigation(viewer)
    rtx_materials.flush_materials(viewer)
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
        if symbol == key.P and modifiers & key.MOD_CTRL and not (modifiers & key.MOD_ALT):
            pause_menu.toggle(viewer)
            return True
        if pause_menu.is_open(viewer):
            return True
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
        "- narrower cone   = wider cone   L distant beam   Ctrl+P pause",
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
        if not pause_menu.is_open(viewer):
            _rtx_mouse_drag(viewer, bridge, x, y, dx, dy, buttons, None, modifiers)
        return True

    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        del x, y, scroll_x
        if not pause_menu.is_open(viewer):
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


def _sync_markings(viewer):
    """Pick up tuner_markings.json. Looks update in place. Pattern edits rebake the channels."""
    path = markings.CONFIG_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    params = getattr(viewer, "_markings_params_live", None)
    if mtime != getattr(viewer, "_markings_mtime", None):
        viewer._markings_mtime = mtime
        fresh = markings.load_parameters()
        appearance = getattr(viewer, "_appearance", None)
        if appearance is not None and markings.pattern_differs(params, fresh):
            appearance.rebuild_channels(fresh)
            for name, index_pts in list(getattr(viewer, "_markings_surfaces", {}).items()):
                prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(name)))
                if prim and prim.IsValid():
                    markings.write_channels(viewer, prim, appearance.sample_channels(index_pts), name)
        markings.push_look(viewer, fresh)
        viewer._markings_params_live = fresh
    markings.flush_runtime(viewer)


def _sync_irrigation(viewer):
    """Pick up tuner_irrigation.json for the live jet look and SPH draw mode."""
    try:
        mtime = IRRIGATION_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if mtime == getattr(viewer, "_irrigation_mtime", None):
        return
    viewer._irrigation_mtime = mtime
    settings = load_irrigation(IRRIGATION_PATH)
    viewer._irrigation = settings
    water = water_from_irrigation(settings, rtx_materials.WATER)
    rtx_materials.apply_params(viewer, "water", water)
    if not getattr(viewer, "_irrigation_logged", False):
        print(
            f"Irrigation {IRRIGATION_PATH.name}: {settings['draw']}  "
            f"{settings['jet_mm']:g} mm  {settings['cc_per_s']:g} cc/s  "
            f"opacity {settings['opacity']:g}",
            flush=True,
        )
        viewer._irrigation_logged = True


def _send_chunk(viewer, key, points, tris, normals, color, channels=None, index_pts=None):
    spatial, packed = key
    if packed == CORTEX_CHUNK:
        segment = "lateral_cortex"
        token = "cortex"
        color = tuple(rtx_materials.EXTRA_DEFAULTS["lateral_cortex"]["diffuse_reflection_color"])
    elif packed == MARKINGS_CHUNK:
        segment = "cortex_markings"
        token = "markings"
    else:
        segment = rtx_materials.segment_id(packed)
        token = f"{packed:06x}"
    names = getattr(viewer, "_chunk_names", {})
    name = names.get(key, f"anatomy/bone_{spatial}_{token}")
    try:
        _log_chunk(viewer, name, points, tris, normals, color)
    except Exception as exc:
        name = f"anatomy/bone_{spatial}_{token}_{time.time_ns()}"
        print(f"chunk {key} resize failed ({exc}); new prim", flush=True)
        _log_chunk(viewer, name, points, tris, normals, color)
    names[key] = name
    viewer._chunk_names = names
    if packed == MARKINGS_CHUNK:
        surfaces = getattr(viewer, "_markings_surfaces", None)
        if surfaces is None:
            surfaces = {}
            viewer._markings_surfaces = surfaces
        if index_pts is None or len(np.asarray(index_pts)) == 0:
            surfaces.pop(name, None)
            return
        surfaces[name] = np.asarray(index_pts, np.float64)
        prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(name)))
        if prim and prim.IsValid():
            markings.bind_shader(viewer, prim)
            if channels is not None:
                markings.write_channels(viewer, prim, channels, name)
        return
    try:
        rtx_materials.bind_named_mesh(
            viewer, name, segment, viewer._material_document
        )
    except Exception as exc:
        if not getattr(viewer, "_material_bind_warned", False):
            print(f"material bind failed: {exc}", flush=True)
            viewer._material_bind_warned = True


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


def _prepare_block_mesh(mesh, volume, spatial, appearance=None):
    """Color-split one block. A missing color is cleared later from this spatial id.

    Bone faces still on the original lateral-cortex mask are a separate chunk.
    Their voxels stay bone voxels; only the bound material changes.
    """
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
    markings_group = None
    if appearance is not None:
        bone = np.flatnonzero(labels == appearance.bone_packed)
        if len(bone):
            centroids = mesh.verts[mesh.tris[bone]].mean(axis=1)
            kind = appearance.classify(centroids)
            cortex = kind > 0
            if np.any(cortex):
                labels = labels.copy()
                labels[bone[cortex]] = CORTEX_CHUNK
            if np.any(kind == 2):
                markings_group = mesh.tris[bone[kind == 2]]
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
    if markings_group is not None and len(markings_group):
        sub_used, sub_inverse = np.unique(markings_group, return_inverse=True)
        lifted = world[sub_used] + normals[sub_used] * np.float32(MARKINGS_LIFT)
        index_pts = np.asarray(mesh.verts[sub_used], np.float64)
        chunks.append(
            (
                spatial,
                MARKINGS_CHUNK,
                np.ascontiguousarray(lifted, np.float32),
                sub_inverse.reshape(-1, 3).astype(np.int32),
                normals[sub_used],
                appearance.sample_channels(index_pts),
                index_pts,
            )
        )
    return chunks


def _prepare_blocks(blocks, volume, dirty, appearance=None):
    """Upload payloads for the rebuilt blocks. ``dirty`` None means every block."""
    chosen = blocks.meshes if dirty is None else {key: blocks.meshes[key] for key in dirty if key in blocks.meshes}
    chunks = []
    for block, mesh in chosen.items():
        if dirty is None and len(mesh.tris) == 0:
            continue
        chunks.extend(_prepare_block_mesh(mesh, volume, _spatial_key(block), appearance))
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
        item = chunks[upload["cursor"]]
        upload["cursor"] += 1
        spatial, color, points, tris, normals = item[:5]
        channels = item[5] if len(item) > 5 else None
        index_pts = item[6] if len(item) > 6 else None
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
        _send_chunk(viewer, key, points, tris, normals, _color_tuple(color), channels, index_pts)
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

    def __init__(self, blocks, occupied, colors, volume, appearance=None):
        self._blocks = blocks
        self._occupied = occupied
        self._colors = colors
        self._volume = volume
        self._appearance = appearance
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
        chunks = _prepare_blocks(self._blocks, self._volume, dirty, self._appearance)
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


def _tuck_baked_tip(points, tris, normals, nominal_mm):
    """Pull the leftover tip sphere inside the burr. The neck past the gap stays put.

    Those vertices also anchor the long triangles that run back to the shaft,
    so deleting them would open the nose. The ring sits inside the flute
    grooves and inside the diamond grains.
    """
    if len(points) == 0:
        return points, tris, normals, 0
    radius = 0.5 * float(nominal_mm) * MM_TO_SIM
    cut = 10.0 * MM_TO_SIM
    tip = points[:, 0] < cut
    count = int(np.count_nonzero(tip))
    if count == 0:
        return points, tris, normals, 0
    points = np.array(points, copy=True, dtype=np.float32)
    yz = points[tip, 1:].astype(np.float64)
    radial = np.linalg.norm(yz, axis=1, keepdims=True)
    direction = np.divide(yz, np.maximum(radial, 1e-8))
    points[tip, 0] = np.float32(0.20 * radius)
    points[tip, 1:] = np.ascontiguousarray(direction * (0.45 * radius), dtype=np.float32)
    return points, tris, normals, count


def _material_preview(viewer, part_id):
    document = getattr(viewer, "_material_document", None) or rtx_materials.empty_document()
    params = rtx_materials.resolve(document, part_id)
    color = tuple(float(channel) for channel in params["diffuse_reflection_color"])
    return color, float(params["specular_reflection_roughness"]), float(params["metalness"])


def _bind_named(viewer, name, part_id, document, warned_attr):
    try:
        rtx_materials.bind_named_mesh(viewer, name, part_id, document)
    except Exception as exc:
        if not getattr(viewer, warned_attr, False):
            print(f"{part_id} material bind failed: {exc}", flush=True)
            setattr(viewer, warned_attr, True)


def _bind_shaft_paints(viewer, mesh_name, ranges, document):
    """Four paints on one mesh. Face subsets, not four copies of the handpiece."""
    from pxr import UsdGeom

    if not ranges or document is None:
        return
    path = viewer._get_path(viewer._qualify(mesh_name))
    prim = viewer.stage.GetPrimAtPath(path)
    if not prim:
        return
    for part_id, (start, count) in ranges.items():
        subset_path = prim.GetPath().AppendChild(part_id)
        subset_prim = viewer.stage.GetPrimAtPath(subset_path)
        subset = UsdGeom.Subset(subset_prim) if subset_prim else UsdGeom.Subset.Define(viewer.stage, subset_path)
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr().Set("materialBind")
        subset.CreateIndicesAttr().Set(list(range(int(start), int(start) + int(count))))
        try:
            rtx_materials.bind_prim(viewer, subset.GetPrim(), part_id, document)
        except Exception as exc:
            if not getattr(viewer, "_shaft_material_warned", False):
                print(f"{part_id} material bind failed: {exc}", flush=True)
                viewer._shaft_material_warned = True
    UsdGeom.Subset.SetFamilyType(UsdGeom.Imageable(prim), "materialBind", UsdGeom.Tokens.nonOverlapping)


def _bind_shown_head(viewer, instance_name, part_id):
    bound = getattr(viewer, "_bound_instance_materials", None)
    if bound is None:
        bound = set()
        viewer._bound_instance_materials = bound
    if instance_name in bound:
        return
    prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(instance_name)) + "/instance_0")
    if not prim:
        return
    document = getattr(viewer, "_material_document", None)
    if document is None:
        return
    _bind_named(viewer, instance_name, part_id, document, "_burr_material_warned")
    bound.add(instance_name)


def _bind_instrument(viewer):
    document = getattr(viewer, "_material_document", None)
    if document is None:
        return
    for name in ("drill/sphere", "drill/flutes", "drill/cutting"):
        _bind_named(viewer, name, "burr", document, "_burr_material_warned")
    meshes = getattr(viewer, "_instrument_meshes", {}) or {}
    diamond = meshes.get("diamond")
    if diamond:
        for name in (diamond, "drill/diamond"):
            _bind_named(viewer, name, "burr_diamond", document, "_burr_material_warned")
    faces = meshes.get("shaft_faces", {})
    for size, mesh in meshes.get("shafts", {}).items():
        if isinstance(mesh, str):
            _bind_shaft_paints(viewer, mesh, faces.get(size, {}), document)
    for mesh in meshes.get("cannulas", {}).values():
        if isinstance(mesh, str):
            _bind_named(viewer, mesh, "cannula", document, "_cannula_material_warned")
    for mesh in meshes.get("suctions", {}).values():
        if isinstance(mesh, str):
            _bind_named(viewer, mesh, "suction", document, "_suction_material_warned")


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


def _read_suction_state():
    try:
        return json.loads((RUNTIME / "suction_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _log_suction(viewer):
    """12 Fr on the distal tip. Smaller sizes stay unloaded until F cycles them."""
    state = _read_suction_state()
    pose = _visual_pose(state) if state else None
    if pose is None:
        return
    quat = np.array(pose["quat"], dtype=np.float64)
    previous_quat = getattr(viewer, "_suction_quat", None)
    if previous_quat is not None and float(np.dot(quat, previous_quat)) < 0.0:
        quat = -quat
    viewer._suction_quat = quat
    pose = {"position": pose["position"], "quat": tuple(float(v) for v in quat)}
    try:
        size = int(state.get("size_fr") or 12)
    except (AttributeError, TypeError, ValueError):
        size = 7
    meshes = (getattr(viewer, "_instrument_meshes", {}) or {}).get("suctions", {})
    previous = getattr(viewer, "_rtx_active_suction", None)
    if previous is not None and previous != size and previous in meshes:
        _log_instance(
            viewer,
            f"suction/inst_{previous}",
            meshes[previous],
            pose["position"],
            pose["quat"],
            1.0,
            hidden=True,
        )
    if size not in meshes:
        return
    viewer._rtx_active_suction = size
    _log_instance(
        viewer,
        f"suction/inst_{size}",
        meshes[size],
        pose["position"],
        pose["quat"],
        1.0,
        hidden=False,
    )
    _bind_shown_head(viewer, f"suction/inst_{size}", "suction")


def _read_drill_state():
    try:
        return json.loads((RUNTIME / "drill_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _sph_snapshot(viewer):
    dump = getattr(viewer, "_sph_dump", None)
    if dump is None:
        return {
            "live": False,
            "count": 0,
            "generation": None,
            "draw_ms": 0.0,
            "mesh_ms": 0.0,
            "triangles": 0,
            "radius": None,
        }
    return {
        "live": bool(getattr(viewer, "_sph_live", False)),
        "count": int(dump.get("count") or 0),
        "generation": dump.get("generation"),
        "draw_ms": float(getattr(viewer, "_sph_draw_ms", 0.0)),
        "mesh_ms": float(getattr(viewer, "_water_mesh_ms", 0.0)),
        "triangles": int(getattr(viewer, "_water_tris", 0)),
        "radius": dump.get("radius"),
    }


def _publish_water_mesh(viewer, mesh, hidden, cull=False):
    """Author the sheet, bind water, then let the RTX viewer publish the copy."""
    points = wp.array(np.ascontiguousarray(mesh["verts"]), dtype=wp.vec3)
    indices = wp.array(np.ascontiguousarray(mesh["tris"].reshape(-1)), dtype=wp.int32)
    normals = wp.array(np.ascontiguousarray(mesh["normals"]), dtype=wp.vec3)
    kwargs = dict(normals=normals, hidden=hidden, backface_culling=bool(cull), dynamic=True)
    if not getattr(viewer, "_water_mesh_bound", False):
        from newton._src.viewer.viewer_usd import ViewerUSD

        ViewerUSD.log_mesh(viewer, "fluid/water", points, indices, **kwargs)
        prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify("fluid/water")))
        if prim and prim.IsValid():
            rtx_materials.bind_water(viewer, prim)
            viewer._water_mesh_bound = True
    viewer.log_mesh("fluid/water", points, indices, **kwargs)
    viewer._water_mesh_last = mesh
    viewer._water_mesh_visible = not hidden


def _hide_sph_points(viewer):
    if getattr(viewer, "_sph_logged", False):
        viewer.log_points("fluid/sph", None, hidden=True)


def _hide_water(viewer):
    if not getattr(viewer, "_water_mesh_visible", False):
        return
    last = getattr(viewer, "_water_mesh_last", None)
    if last is None:
        return
    _publish_water_mesh(viewer, last, hidden=True)
    viewer._water_tris = 0


def _draw_sph(viewer) -> bool:
    """Mesh fresh SPH positions into one water sheet. Isolated specks stay invisible."""
    started = time.perf_counter()
    live = dump_is_fresh(SPH_DUMP)
    viewer._sph_live = live
    if not live:
        _hide_sph_points(viewer)
        _hide_water(viewer)
        viewer._sph_draw_ms = (time.perf_counter() - started) * 1e3
        viewer._water_mesh_ms = 0.0
        return False
    dump = read_positions(SPH_DUMP)
    viewer._sph_dump = dump
    xyz = None if dump is None else dump["xyz"]
    if xyz is not None and len(xyz):
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if dump is None or xyz is None or len(xyz) == 0:
        _hide_sph_points(viewer)
        _hide_water(viewer)
        viewer._sph_draw_ms = (time.perf_counter() - started) * 1e3
        viewer._water_mesh_ms = 0.0
        return True
    settings = getattr(viewer, "_irrigation", None) or load_irrigation(IRRIGATION_PATH)
    if settings.get("draw") == DRAW_SPHERES:
        _hide_water(viewer)
        radius = float(dump.get("radius") or 0.5 * float(settings["particle_mm"]) * MM_TO_SIM)
        color = tuple(water_from_irrigation(settings, rtx_materials.WATER)["diffuse_reflection_color"])
        viewer.log_points("fluid/sph", wp.array(xyz, dtype=wp.vec3), radii=radius, colors=color)
        viewer._sph_logged = True
        viewer._water_mesh_ms = 0.0
        viewer._water_tris = 0
        viewer._sph_draw_ms = (time.perf_counter() - started) * 1e3
        return True
    mesh_started = time.perf_counter()
    mesh = water_surface(xyz)
    viewer._water_mesh_ms = (time.perf_counter() - mesh_started) * 1e3
    _hide_sph_points(viewer)
    if mesh is None:
        _hide_water(viewer)
    else:
        _publish_water_mesh(viewer, mesh, hidden=False)
        viewer._water_tris = int(len(mesh["tris"]))
    viewer._sph_draw_ms = (time.perf_counter() - started) * 1e3
    return True


def _debris_zones(viewer) -> dict:
    """Where live irrigation and the suction tip remove paste. No drawn sphere."""
    from fluid.irrigation_bench import SPLASH_CLEAR_MM

    spheres = []
    segment = None
    bench = getattr(viewer, "_irrigation_bench", None)
    drilling = _read_drill_state()
    if (
        bench is not None
        and bench.last_hit is not None
        and bench.last_frame is not None
        and _emit_requested(drilling)
    ):
        hit = np.asarray(bench.last_hit, dtype=np.float64)
        spheres.append((hit, float(SPLASH_CLEAR_MM) * MM_TO_SIM))
        outlet = np.asarray(bench.last_frame["outlet"], dtype=np.float64)
        jet_r = 0.5 * float((bench.settings or {}).get("jet_mm") or 1.0) * MM_TO_SIM
        segment = (outlet, hit, max(jet_r, 0.35 * MM_TO_SIM))
    suction = _read_suction_state()
    if suction and "px" in suction:
        try:
            tip = np.array(
                [float(suction["px"]), float(suction["py"]), float(suction["pz"])],
                dtype=np.float64,
            )
            radius = bone_debris.suction_radius_mm(float(suction.get("size_fr") or 12)) * MM_TO_SIM
            spheres.append((tip, radius))
        except (TypeError, ValueError):
            pass
    sph = getattr(viewer, "_surface_sph", None)
    return {"spheres": spheres, "segment": segment, "film": sph}


def _step_bone_debris(viewer, dt) -> None:
    """Dust from this frame's removals. Irrigation and suction can clear it."""
    debris = getattr(viewer, "_bone_debris", None)
    if debris is None:
        return
    state = _read_drill_state()
    if state and state.get("paused"):
        dt = 0.0
    pose = _visual_pose(state) if state else None
    spin = float(getattr(viewer, "_burr_spin", 0.0))
    bench = getattr(viewer, "_irrigation_bench", None)
    down = getattr(bench, "down", None) if bench is not None else None
    zones = _debris_zones(viewer)
    if pose is None:
        debris.step(dt, None, None, spin, down, zones)
    else:
        debris.step(dt, pose["position"], pose["quat"], spin, down, zones)
    debris.draw(viewer)


def _emit_requested(state) -> bool:
    return bool(state and state.get("on"))


def _draw_live_irrigation(viewer, worker, dt) -> bool:
    """Same tube as the tuner: a fixed-width jet to the first bone hit, then gone."""
    settings = getattr(viewer, "_irrigation", None) or load_irrigation(IRRIGATION_PATH)
    bench = getattr(viewer, "_irrigation_bench", None)
    if bench is None:
        return False
    occupied = getattr(worker, "_occupied", None)
    if occupied is not None:
        bench.occupied = occupied
    live = dict(settings)
    live["draw"] = DRAW_TUBE
    live["film_ms"] = 0.0
    bench.apply(live)
    _hide_sph_points(viewer)
    _hide_water(viewer)
    state = _read_drill_state()
    pose = _visual_pose(state) if state else None
    if pose is None or not _emit_requested(state):
        bench.hide(viewer)
        return False
    try:
        size_mm = int(state.get("size_mm") or 6)
    except (AttributeError, TypeError, ValueError):
        size_mm = 6
    bench.step(dt, pose["position"], pose["quat"], size_mm=size_mm)
    if bench.last_hit is None:
        bench.hide(viewer)
        return False
    bench.draw(viewer, splash=False)
    viewer._sph_draw_ms = bench.draw_ms
    viewer._water_tris = 0
    return True


def _suction_tip():
    suction = _read_suction_state()
    if not suction or "px" not in suction:
        return None, 0.0
    try:
        tip = np.array(
            [float(suction["px"]), float(suction["py"]), float(suction["pz"])],
            dtype=np.float64,
        )
        radius = bone_debris.suction_radius_mm(float(suction.get("size_fr") or 12))
    except (TypeError, ValueError):
        return None, 0.0
    return tip, radius


def _draw_surface_film(viewer, sph, view) -> None:
    """One sheet. The particles are not drawn."""
    _hide_sph_points(viewer)
    mesh = film_surface(sph.positions(), view)
    started = time.perf_counter()
    if mesh is None:
        _hide_water(viewer)
        viewer._water_tris = 0
    else:
        _publish_water_mesh(viewer, mesh, hidden=False, cull=True)
        viewer._water_tris = int(len(mesh["tris"]))
    viewer._water_mesh_ms = (time.perf_counter() - started) * 1e3


def _step_surface_sph(viewer, worker, dt) -> None:
    """Particles on the cut bone. The picture is a thin film, not the particles."""
    sph = getattr(viewer, "_surface_sph", None)
    bench = getattr(viewer, "_irrigation_bench", None)
    if sph is None or bench is None:
        return
    occupied = getattr(worker, "_occupied", None)
    if occupied is not None:
        bench.occupied = occupied
    settings = getattr(viewer, "_irrigation", None) or load_irrigation(IRRIGATION_PATH)
    state = _read_drill_state()
    if state and state.get("paused"):
        dt = 0.0
    hit = None
    if settings.get("draw") != DRAW_SPHERES and _emit_requested(state) and bench.last_hit is not None:
        hit = bench.last_hit
    tip, radius = _suction_tip()
    cam = getattr(viewer, "_fivrs_cam", None)
    if cam is not None:
        view = np.asarray(cam[3], dtype=np.float64)
        gravity = -np.asarray(cam[2], dtype=np.float64)
    else:
        emit = (bench.last_frame or {}).get("emit")
        view = np.asarray(emit if emit is not None else (0.0, 0.0, -1.0), dtype=np.float64)
        gravity = np.asarray(bench.down, dtype=np.float64)
    sph.step(
        dt,
        hit,
        float(settings.get("jet_mm") or 0.58),
        view,
        gravity,
        frame=bench.frame,
        occupied=bench.occupied,
        suction_tip=tip,
        suction_radius_mm=radius,
    )
    # Keep the old square sheet hidden. This film replaces it.
    old = getattr(viewer, "_surface_film", None)
    if old is not None and (old._sheet_visible or old._wet_visible):
        old.draw(viewer)
    _draw_surface_film(viewer, sph, view)


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
    pose = _visual_pose(state) if state else None
    if pose is None:
        # A torn read used to fall through to mastoidectomy_drill, a different
        # body. Skip the frame. Irrigation reads this file itself and is not held.
        return
    quat = np.array(pose["quat"], dtype=np.float64)
    previous_quat = getattr(viewer, "_shaft_quat", None)
    if previous_quat is not None:
        # q and -q are the same rotation. Taking the sign closest to the last
        # drawn frame keeps a long shaft from whipping when the sign flips.
        if float(np.dot(quat, previous_quat)) < 0.0:
            quat = -quat
            print("shaft quaternion sign flipped", flush=True)
        jump = float(np.linalg.norm(pose["position"] - viewer._shaft_position))
        if jump > 0.05:
            print(
                f"shaft jumped {jump / MM_TO_SIM:.1f} mm to {np.round(pose['position'], 3).tolist()}",
                flush=True,
            )
    viewer._shaft_quat = quat
    viewer._shaft_position = pose["position"].copy()
    pose = {"position": pose["position"], "quat": tuple(float(v) for v in quat)}
    size = int(state.get("size_mm") or bridge.burr_mm or 6)
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
    # Same spin for both heads. The angle only advances while the drill is on.
    head_quat = _mul_quat(pose["quat"], _spin_about_x(angle))
    diamond = meshes.get("diamond") or "drill/sphere"
    # Hidden sizes and the unused head stay where they were. Updating all of
    # them every frame was a renderer write for geometry the scope cannot see.
    active_name = "drill/cutting" if cutting else "drill/diamond"
    active_mesh = head if cutting else diamond
    previous_head = getattr(viewer, "_rtx_active_head", None)
    if previous_head and previous_head != active_name:
        previous_mesh = head if previous_head == "drill/cutting" else diamond
        _log_instance(
            viewer, previous_head, previous_mesh, pose["position"], head_quat, radius, hidden=True
        )
    viewer._rtx_active_head = active_name
    _log_instance(
        viewer, active_name, active_mesh, pose["position"], head_quat, radius, hidden=False
    )
    _bind_shown_head(viewer, active_name, "burr" if cutting else "burr_diamond")
    shafts = meshes.get("shafts", {})
    previous_size = getattr(viewer, "_rtx_active_shaft", None)
    if previous_size is not None and previous_size != size and previous_size in shafts:
        _log_instance(
            viewer,
            f"drill/shaft_inst_{previous_size}",
            shafts[previous_size],
            pose["position"],
            pose["quat"],
            1.0,
            hidden=True,
        )
    if size in shafts:
        viewer._rtx_active_shaft = size
        _log_instance(
            viewer,
            f"drill/shaft_inst_{size}",
            shafts[size],
            pose["position"],
            pose["quat"],
            1.0,
            hidden=False,
        )
    # The cannula is fixed to the shaft. It does not take the burr's spin,
    # and it is not part of the shaft's region paints.
    cannulas = meshes.get("cannulas", {})
    previous_cannula = getattr(viewer, "_rtx_active_cannula", None)
    if previous_cannula is not None and previous_cannula != size and previous_cannula in cannulas:
        _log_instance(
            viewer,
            f"drill/cannula_inst_{previous_cannula}",
            cannulas[previous_cannula],
            pose["position"],
            pose["quat"],
            1.0,
            hidden=True,
        )
    if size in cannulas:
        viewer._rtx_active_cannula = size
        _log_instance(
            viewer,
            f"drill/cannula_inst_{size}",
            cannulas[size],
            pose["position"],
            pose["quat"],
            1.0,
            hidden=False,
        )
        _bind_shown_head(viewer, f"drill/cannula_inst_{size}", "cannula")
    if not getattr(viewer, "_instrument_material_bound", False):
        _bind_instrument(viewer)
        viewer._instrument_material_bound = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy-adf", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stream-cc-s", type=float, default=1.0)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    import newton

    print("Loading anatomy slices", flush=True)
    occupied, colors = _load_occupancy(pathlib.Path(args.anatomy_adf))
    labels = np.unique(colors[occupied].reshape(-1, 3), axis=0)
    print(f"Volume {occupied.shape} with {len(labels)} segment colors", flush=True)
    bridge = Bridge()
    deadline = time.perf_counter() + 180.0
    announced = 0.0
    while bridge.volume is None and time.perf_counter() < deadline:
        bridge.spin()
        now = time.perf_counter()
        if now - announced > 5.0:
            print("Waiting for AMBF volume_info before opening the RTX window", flush=True)
            announced = now
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

    from pxr import Gf

    # Solid field behind the bone. The path tracer was sampling a dark dome,
    # so the clear water sheet read as black. Lighting stays on the scope rig.
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
            # Opacity on the water sheet. Authored before the render product exists.
            "omni:rtx:rt:fractionalOpacity": ("bool", True),
            "omni:rtx:background:source:type": ("token", "color"),
            "omni:rtx:background:source:color": ("color3f", Gf.Vec3f(0.62, 0.78, 0.90)),
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
    pause_menu.install(viewer)
    if hasattr(viewer, "show_ground"):
        viewer.show_ground = False
    viewer.picking_enabled = False
    
    # Configure RTX for photorealistic water rendering with caustics
    rtx_config.setup_water_rendering(viewer, enable_caustics=True, quality="high")

    sphere_local, sphere_tris = _sphere(1.0)
    sphere_normals = vertex_normals(sphere_local.astype(np.float64), sphere_tris).astype(np.float32)
    _log_static_mesh(
        viewer, "drill/sphere", sphere_local, sphere_tris, sphere_normals, (0.86, 0.87, 0.88), 0.12, 0.95
    )
    instrument = {"cutting": "drill/sphere", "shafts": {}, "shaft_faces": {}, "cannulas": {}, "suctions": {}}
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
    diamond_path = MESH_DIR / "diamond_burr.obj"
    try:
        diamond_points, diamond_tris, diamond_normals = _load_obj(diamond_path)
        color, roughness, metallic = _material_preview(viewer, "burr_diamond")
        _log_static_mesh(
            viewer,
            "drill/diamond_grit",
            diamond_points,
            diamond_tris,
            diamond_normals,
            color,
            roughness,
            metallic,
        )
        instrument["diamond"] = "drill/diamond_grit"
        print(f"Diamond burr {len(diamond_tris)} tris", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Diamond burr unavailable ({exc}); using the smooth sphere", flush=True)
    for size in (1, 2, 4, 6):
        shaft_path = MESH_DIR / f"{size}mm.OBJ"
        try:
            parts = _load_obj_parts(shaft_path)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Shaft {size} mm unavailable ({exc})", flush=True)
            continue
        tucked_parts = {}
        counts = []
        tucked_tip = 0
        for part_id, (shaft_points, shaft_tris, shaft_normals) in parts.items():
            shaft_points, shaft_tris, shaft_normals, tucked = _tuck_baked_tip(
                shaft_points, shaft_tris, shaft_normals, size
            )
            tucked_tip += tucked
            if len(shaft_tris) == 0:
                continue
            tucked_parts[part_id] = (shaft_points, shaft_tris, shaft_normals)
            label = rtx_materials.display_name(viewer._material_document, part_id)
            counts.append(f"{label} {len(shaft_tris)}")
        combined = _combine_shaft(tucked_parts)
        if combined is None:
            continue
        shaft_points, shaft_tris, shaft_normals, ranges = combined
        mesh_name = f"drill/shaft_{size}"
        color, roughness, metallic = _material_preview(viewer, "shaft_silver")
        _log_static_mesh(
            viewer,
            mesh_name,
            shaft_points,
            shaft_tris,
            shaft_normals,
            color,
            roughness,
            metallic,
        )
        instrument["shafts"][size] = mesh_name
        instrument["shaft_faces"][size] = ranges
        tip = f", tucked {tucked_tip} tip verts inside the burr" if tucked_tip else ""
        print(f"Shaft {size} mm {', '.join(counts)} tris, one mesh{tip}", flush=True)
    for size in CANNULA_SIZES:
        try:
            cannula_points, cannula_tris = load_cannula_obj(cannula_mesh_path(size))
        except (OSError, ValueError) as exc:
            print(f"Cannula {size} mm unavailable ({exc})", flush=True)
            continue
        cannula_normals = vertex_normals(cannula_points, cannula_tris).astype(np.float32)
        color, roughness, metallic = _material_preview(viewer, "cannula")
        mesh_name = f"drill/cannula_{size}"
        _log_static_mesh(
            viewer,
            mesh_name,
            np.ascontiguousarray(cannula_points, dtype=np.float32),
            cannula_tris,
            cannula_normals,
            color,
            roughness,
            metallic,
        )
        instrument["cannulas"][size] = mesh_name
        print(f"Cannula {size} mm {len(cannula_tris)} tris, stainless", flush=True)
    for french in (12, 7, 5, 3):
        suction_path = MESH_DIR / f"suction_{french}fr.obj"
        try:
            suction_points, suction_tris = load_cannula_obj(suction_path)
            suction_normals = vertex_normals(suction_points, suction_tris).astype(np.float32)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Suction {french} Fr unavailable ({exc})", flush=True)
            continue
        color, roughness, metallic = _material_preview(viewer, "suction")
        # USD prim names cannot start with a digit, so this is fr_7 rather than 7fr.
        mesh_name = f"suction/fr_{french}"
        try:
            _log_static_mesh(
                viewer,
                mesh_name,
                np.ascontiguousarray(suction_points, dtype=np.float32),
                suction_tris,
                suction_normals,
                color,
                roughness,
                metallic,
            )
        except Exception as exc:
            print(f"Suction {french} Fr was not added to RTX ({exc})", flush=True)
            continue
        instrument["suctions"][french] = mesh_name
        print(f"Suction {french} Fr {len(suction_tris)} tris, satin stainless", flush=True)
    viewer._instrument_meshes = instrument
    _bind_instrument(viewer)
    stream = None
    if not args.no_stream:
        irrigation = load_irrigation(IRRIGATION_PATH)
        stream = PrescribedStream(
            cc_per_s=float(irrigation["cc_per_s"]),
            particle_mm=float(irrigation["particle_mm"]),
            jet_mm=float(irrigation["jet_mm"]),
            offset_mm=float(irrigation["offset_mm"]),
            proximal_mm=float(irrigation["proximal_mm"]),
            config_path=IRRIGATION_PATH,
        )
        print(
            f"Irrigation stream {stream.cc_per_s:g} cc/s, "
            f"{stream.jet_mm:g} mm jet, "
            f"{stream.offset_mm:g} mm drill-right. {IRRIGATION_PATH.name}",
            flush=True,
        )
    viewer._volume_translation = np.asarray(bridge.volume["translation"], dtype=np.float64)
    document = rtx_materials.load_document()
    print("Fitting the lateral cortex onto the live surface", flush=True)
    appearance = cortex_masks.build_appearance_index(
        blocks,
        occupied,
        colors,
        bridge.volume["dimensions"],
        "",
        cortex_masks.named_rgb(document, "bone", "fff9db"),
        cortex_masks.named_rgb(document, "eac", "ffe1d6"),
    )
    viewer._appearance = appearance
    viewer._markings_surfaces = {}
    viewer._markings_params_live = markings.load_parameters()
    try:
        viewer._markings_mtime = markings.CONFIG_PATH.stat().st_mtime
    except OSError:
        viewer._markings_mtime = None
    markings.push_look(viewer, viewer._markings_params_live)
    _sync_irrigation(viewer)
    viewer._irrigation_bench = IrrigationBench(
        occupied,
        VolumeFrame(
            np.asarray(bridge.volume["translation"], dtype=np.float64),
            np.asarray(bridge.volume["rotation"], dtype=np.float64),
            np.asarray(bridge.volume["dimensions"], dtype=np.float64),
            np.asarray(occupied.shape, dtype=np.float64),
        ),
    )
    viewer._irrigation_bench.apply(getattr(viewer, "_irrigation", None) or load_irrigation(IRRIGATION_PATH))
    viewer._irrigation_bench.prime(viewer)
    viewer._surface_film = SurfaceFilm()
    viewer._surface_film.prime(viewer)
    viewer._surface_sph = SurfaceSph()
    viewer._bone_debris = bone_debris.BoneDebris(viewer._irrigation_bench.frame)
    uploads = [
        _begin_upload(
            _prepare_blocks(blocks, bridge.volume, None, appearance),
            True,
            extract_s * 1e3,
            None,
        )
    ]
    worker = SurfaceWorker(blocks, occupied, colors, bridge.volume, appearance)
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
            debris = getattr(viewer, "_bone_debris", None)
            if debris is not None:
                debris.note_removals(handed)

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
        pause_menu.sync(viewer, _read_drill_state())
        if bridge.drill is not None:
            _log_instrument(viewer, bridge)
        _log_suction(viewer)
        dt = min(frame_ms[-1] * 1e-3, 0.05) if frame_ms else 0.0
        settings = getattr(viewer, "_irrigation", None) or load_irrigation(IRRIGATION_PATH)
        if settings.get("draw") == DRAW_SPHERES:
            if getattr(viewer, "_irrigation_bench", None) is not None:
                viewer._irrigation_bench.hide(viewer)
            _draw_sph(viewer)
        else:
            _draw_live_irrigation(viewer, worker, dt)
            _step_surface_sph(viewer, worker, dt)
        if stream is not None:
            stream.hide(viewer)
        _step_bone_debris(viewer, dt)
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
                "stream": None if stream is None else stream.snapshot(),
                "sph": _sph_snapshot(viewer),
                "debris": viewer._bone_debris.snapshot(),
            }
            (RUNTIME / "summary.json").write_text(json.dumps(summary, indent=2))
            sph = summary.get("sph") or {}
            sph_note = (
                f"   SPH {sph.get('count') or 0}"
                f"   water {sph.get('mesh_ms') or 0:.1f} ms"
                f"   {sph.get('triangles') or 0} tris"
                if sph.get("live")
                else ""
            )
            title = (
                f"FIVRS RTX  {summary['rtx_fps']:.0f} fps"
                f"   exposure {getattr(viewer, '_exposure', 1.0):.2f}"
                f"   Gfx {graphics_hz} Hz   Phx {physics_hz} Hz"
                f"   extract {last_extract_ms:.0f} ms"
                f"{sph_note}"
            )
            window = getattr(viewer, "_window", None)
            if window is not None:
                window.set_caption(title)
        if frame == 1:
            print(f"ViewerRTX first frame {(time.perf_counter() - started) * 1e3:.0f} ms", flush=True)

    bridge.close()
    viewer.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        text = traceback.format_exc()
        print(text, flush=True)
        try:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            (RUNTIME / "rtx_crash.log").write_text(text)
        except OSError:
            pass
        raise
