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
import sys
import threading
import time

import numpy as np
import warp as wp
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from surface import BoneMesh, cell_corners, index_to_local, local_to_world, vertex_normals

# The volume ADF states 1 AMBF unit = 0.049664 m. Burr diameters arrive in mm.
METERS_PER_AU = 0.049664
# main_camera field view angle, radians, full vertical angle.
VERTICAL_FOV_RAD = 0.17951634837990105
RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
BONE_ROUGHNESS = 0.55
# Scope lights at exposure 1. The bone albedo is nearly white, so these sit
# well below the first tuning, which clipped the whole view.
LIGHT_BASE = {
    "/root/_ScopeBeam": 200.0,
    "/root/_ScopeFill": 8.0,
    "/root/_ScopeLight": 8000.0,
}
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
    return points, np.asarray(tris, dtype=np.int32)


def _author_lights(viewer):
    from pxr import Gf, UsdLux

    color = Gf.Vec3f(1.0, 0.985, 0.96)
    distant = UsdLux.DistantLight.Define(viewer.stage, "/root/_ScopeBeam")
    distant.CreateIntensityAttr(LIGHT_BASE["/root/_ScopeBeam"])
    distant.CreateColorAttr(color)
    distant.CreateAngleAttr(2.0)
    dome = UsdLux.DomeLight.Define(viewer.stage, "/root/_ScopeFill")
    dome.CreateIntensityAttr(LIGHT_BASE["/root/_ScopeFill"])
    dome.CreateColorAttr(Gf.Vec3f(0.93, 0.93, 0.92))
    sphere = UsdLux.SphereLight.Define(viewer.stage, "/root/_ScopeLight")
    sphere.CreateIntensityAttr(LIGHT_BASE["/root/_ScopeLight"])
    sphere.CreateRadiusAttr(0.04)
    sphere.CreateColorAttr(color)
    shaping = UsdLux.ShapingAPI.Apply(sphere.GetPrim())
    shaping.CreateShapingConeAngleAttr(16.0)
    shaping.CreateShapingConeSoftnessAttr(0.55)
    viewer._exposure = 1.0
    viewer._exposure_dirty = True


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
        if getattr(viewer, "_exposure_dirty", False):
            _push_exposure(viewer)
        matrix = getattr(viewer, "_fivrs_light_matrix", None)
        if matrix is None:
            return
        posed = np.ascontiguousarray(matrix, dtype=np.float64)[np.newaxis, ...]
        try:
            for path in ("/root/_ScopeBeam", "/root/_ScopeLight"):
                viewer._write_runtime_attribute([path], "omni:xform", posed, is_matrix=True)
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
        self.gap = False
        self._arrived = 0
        self.node.create_subscription(VolumeInfo, "/ambf/env/plugin/volumetric_drilling/volume_info", self._volume, latched)
        self.node.create_subscription(DrillSize, "/ambf/env/plugin/volumetric_drilling/drill_size", self._burr, latched)
        self.node.create_subscription(Voxels, "/ambf/env/plugin/volumetric_drilling/voxels_removed", self._voxels, stream)
        self.node.create_subscription(CameraState, "/ambf/env/cameras/main_camera/State", self._camera, pose)
        self.node.create_subscription(RigidBodyState, "/ambf/env/mastoidectomy_drill/State", self._drill, pose)

    def spin(self):
        # One spin_once takes a single callback. Camera, drill, and voxel
        # messages all arrive together, so drain until the queue is empty and
        # the pose callbacks have overwritten anything older.
        for _ in range(1024):
            seen = self._arrived
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            if self._arrived == seen:
                break

    def close(self):
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

    def _drill(self, msg):
        self._arrived += 1
        pose = msg.pose
        self.drill = {
            "position": np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            "quat": (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        }

    def _voxels(self, msg):
        self._arrived += 1
        serial = 0
        try:
            serial = int(msg.header.frame_id)
        except ValueError:
            serial = self.applied + len(msg.indices)
        if serial > self.applied + len(msg.indices):
            self.gap = True
        self.pending.append((time.perf_counter(), serial, [(int(p.x), int(p.y), int(p.z)) for p in msg.indices]))
        self.applied = max(self.applied, serial)


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
    _aim_lights(viewer, position, forward / max(np.linalg.norm(forward), 1e-8))


def _chunk_key(tri_cell):
    ci, cj, ck = cell_corners(tri_cell)
    # The crop-face cells sit one step outside the scan. They belong to the
    # first or last block, not to their own block.
    ci = np.clip(ci, 0, None)
    cj = np.clip(cj, 0, None)
    ck = np.clip(ck, 0, None)
    return (ci // 48).astype(np.int32), (cj // 48).astype(np.int32), (ck // 48).astype(np.int32)


def _push_exposure(viewer):
    scale = float(getattr(viewer, "_exposure", 1.0))
    for path, base in LIGHT_BASE.items():
        value = base * scale
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
    viewer._exposure_dirty = False


def _set_exposure(viewer, scale):
    viewer._exposure = float(min(EXPOSURE_MAX, max(EXPOSURE_MIN, scale)))
    viewer._exposure_dirty = True
    print(f"RTX exposure {viewer._exposure:.2f}   [ darker   ] brighter   \\ reset", flush=True)


def _bind_exposure_keys(viewer):
    import pyglet.window.key as key

    def on_key_press(symbol, modifiers):
        if symbol == key.BRACKETLEFT:
            _set_exposure(viewer, viewer._exposure / 1.25)
        elif symbol == key.BRACKETRIGHT:
            _set_exposure(viewer, viewer._exposure * 1.25)
        elif symbol == key.BACKSLASH:
            _set_exposure(viewer, 1.0)

    window = getattr(viewer, "_window", None)
    if window is not None:
        window.push_handlers(on_key_press=on_key_press)


def _log_chunk(viewer, name, points, tris, normals, color):
    viewer.log_mesh(
        name,
        wp.array(np.ascontiguousarray(points), dtype=wp.vec3),
        wp.array(np.ascontiguousarray(tris.reshape(-1)), dtype=wp.int32),
        normals=wp.array(np.ascontiguousarray(normals), dtype=wp.vec3),
        color=color,
        roughness=BONE_ROUGHNESS,
        metallic=0.0,
        dynamic=True,
        backface_culling=False,
    )


def _color_tuple(packed: int):
    return ((packed & 255) / 255.0, ((packed >> 8) & 255) / 255.0, ((packed >> 16) & 255) / 255.0)


def _chunks_covering(box):
    """Chunk ids whose 48-voxel block meets the half-open cell box."""
    i0, i1, j0, j1, k0, k1 = box
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return []
    keys = []
    for x in range(i0 // 48, (i1 - 1) // 48 + 1):
        for y in range(j0 // 48, (j1 - 1) // 48 + 1):
            for z in range(k0 // 48, (k1 - 1) // 48 + 1):
                keys.append(x + y * 16 + z * 256)
    return keys


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


def _prepare_chunks(mesh, volume, dirty_box):
    """CPU copy of the blocks a cut touched. The draw thread only sends the result.

    Each block is split by label color. ViewerRTX paints a whole mesh one color,
    and a later update does not change that color, so a label that leaves a
    block has to be sent as its own empty mesh.
    """
    if len(mesh.tris) == 0:
        keys = _chunks_covering(dirty_box) if dirty_box is not None else []
        return [(key, None, _EMPTY_POINTS, _EMPTY_TRIS, _EMPTY_NORMALS) for key in keys]
    cx, cy, cz = _chunk_key(mesh.tri_cell)
    if dirty_box is None:
        selected = np.ones(len(mesh.tris), dtype=bool)
        expected = None
    else:
        i0, i1, j0, j1, k0, k1 = dirty_box
        selected = (
            (cx >= i0 // 48) & (cx <= (max(i1 - 1, i0) // 48))
            & (cy >= j0 // 48) & (cy <= (max(j1 - 1, j0) // 48))
            & (cz >= k0 // 48) & (cz <= (max(k1 - 1, k0) // 48))
        )
        expected = _chunks_covering(dirty_box)
    chunks = []
    present = set()
    if np.any(selected):
        sel_index = np.flatnonzero(selected)
        packed = (cx[selected] + cy[selected] * 16 + cz[selected] * 256).astype(np.int32)
        dims = volume["dimensions"]
        counts = volume["counts"]
        rotation = volume["rotation"]
        translation = volume["translation"]
        for key in np.unique(packed):
            faces = mesh.tris[sel_index[packed == key]]
            used, inverse = np.unique(faces, return_inverse=True)
            local = index_to_local(mesh.verts[used], dims, counts)
            world = local_to_world(local, rotation, translation).astype(np.float32)
            local_tris = inverse.reshape(-1, 3).astype(np.int32)
            normals = vertex_normals(world.astype(np.float64), local_tris).astype(np.float32)
            labels = _face_labels(mesh, faces)
            order = np.argsort(labels, kind="mergesort")
            ordered = labels[order]
            breaks = np.flatnonzero(np.diff(ordered)) + 1
            starts = np.concatenate(([0], breaks))
            ends = np.concatenate((breaks, [len(order)]))
            for start, end in zip(starts, ends):
                group = local_tris[order[start:end]]
                sub_used, sub_inverse = np.unique(group, return_inverse=True)
                chunks.append(
                    (
                        int(key),
                        int(ordered[start]),
                        world[sub_used],
                        sub_inverse.reshape(-1, 3).astype(np.int32),
                        normals[sub_used],
                    )
                )
            present.add(int(key))
    if expected is not None:
        for key in expected:
            if key not in present:
                chunks.append((key, None, _EMPTY_POINTS, _EMPTY_TRIS, _EMPTY_NORMALS))
    return chunks


def _begin_upload(chunks, full, extract_ms, receipt):
    return {
        "chunks": chunks,
        "full": full,
        "cursor": 0,
        "sent": set(),
        "dirty": set(),
        "stale": None,
        "extract_ms": extract_ms,
        "receipt": receipt,
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

    def __init__(self, mesh, occupied, colors, volume):
        self._mesh = mesh
        self._occupied = occupied
        self._colors = colors
        self._volume = volume
        self._inbox = queue.Queue()
        self._outbox = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="fivrs-rtx-surface", daemon=True)
        self._thread.start()

    def submit_cuts(self, batches):
        self._inbox.put(("cut", batches))

    def submit_resync(self, dump_path):
        self._inbox.put(("resync", dump_path))

    def poll(self):
        try:
            return self._outbox.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        while True:
            jobs = [self._inbox.get()]
            while True:
                try:
                    jobs.append(self._inbox.get_nowait())
                except queue.Empty:
                    break
            resync_at = max((i for i, (op, _) in enumerate(jobs) if op == "resync"), default=None)
            if resync_at is not None:
                jobs = jobs[resync_at:]
            try:
                self._apply(jobs)
            except Exception as exc:
                print(f"surface update failed: {exc}", flush=True)

    def _apply(self, jobs):
        started = time.perf_counter()
        receipt = None
        full = False
        dirty = None
        shape = self._occupied.shape
        for op, payload in jobs:
            if op == "resync":
                occupied, colors, _count = _load_dump(pathlib.Path(payload))
                if occupied.shape != shape:
                    print(f"resync shape {occupied.shape} != {shape}", flush=True)
                    continue
                self._occupied = occupied
                self._colors = colors
                self._mesh = BoneMesh(self._occupied, self._colors)
                full = True
                dirty = None
                continue
            i0 = j0 = k0 = 10**9
            i1 = j1 = k1 = -1
            for stamp, _serial, voxels in payload:
                receipt = stamp if receipt is None else min(receipt, stamp)
                for x, y, z in voxels:
                    if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                        continue
                    self._occupied[x, y, z] = False
                    self._colors[x, y, z] = 0
                    i0, j0, k0 = min(i0, x), min(j0, y), min(k0, z)
                    i1, j1, k1 = max(i1, x), max(j1, y), max(k1, z)
            if i1 < i0:
                continue
            box = (i0 - 2, i1 + 3, j0 - 2, j1 + 3, k0 - 2, k1 + 3)
            self._mesh.replace_cells(self._occupied, *box, self._colors)
            if full:
                continue
            if dirty is None:
                dirty = box
            else:
                dirty = tuple(
                    min(dirty[i], box[i]) if i % 2 == 0 else max(dirty[i], box[i]) for i in range(6)
                )
        if not full and dirty is None:
            return
        chunks = _prepare_chunks(self._mesh, self._volume, None if full else dirty)
        self._outbox.put(
            {
                "chunks": chunks,
                "full": full,
                "extract_ms": (time.perf_counter() - started) * 1e3,
                "receipt": receipt,
            }
        )


def _log_burr(viewer, position, quat, radius):
    xform = wp.transform(
        wp.vec3(float(position[0]), float(position[1]), float(position[2])),
        wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )
    viewer.log_instances(
        "drill/burr",
        "drill/sphere",
        wp.array([xform], dtype=wp.transform),
        wp.array([wp.vec3(radius, radius, radius)], dtype=wp.vec3),
        None,
        None,
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
    mesh = BoneMesh(occupied, colors)
    extract_s = time.perf_counter() - started
    print(f"Initial surface {len(mesh.verts)} verts {len(mesh.tris)} tris in {extract_s:.2f}s", flush=True)

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
    viewer.log_mesh(
        "drill/sphere",
        wp.array(sphere_local, dtype=wp.vec3),
        wp.array(sphere_tris.reshape(-1), dtype=wp.int32),
        color=(0.75, 0.75, 0.78),
        roughness=0.25,
        metallic=0.85,
        dynamic=False,
    )
    uploads = [_begin_upload(_prepare_chunks(mesh, bridge.volume, None), True, extract_s * 1e3, None)]
    worker = SurfaceWorker(mesh, occupied, colors, bridge.volume)
    print("RTX brightness: [ darker    ] brighter    \\ reset", flush=True)
    waiting_dump_since = None
    frame_ms = []
    removal_delay_ms = []
    last_extract_ms = extract_s * 1e3
    last_upload_ms = 0.0
    last_report = time.perf_counter()
    previous = time.perf_counter()
    frame = 0
    RUNTIME.mkdir(parents=True, exist_ok=True)

    while viewer.is_running():
        bridge.spin()
        now = time.perf_counter()
        if bridge.gap:
            bridge.gap = False
            (RUNTIME / "resync.request").write_text("gap\n")
            waiting_dump_since = now
        if waiting_dump_since is not None:
            dump = RUNTIME / "occupancy.bin"
            if dump.is_file() and dump.stat().st_mtime >= waiting_dump_since:
                worker.submit_resync(dump)
                waiting_dump_since = None
                print("Resync queued from the live volume", flush=True)
            elif now - waiting_dump_since > 2.0:
                print("Resync dump did not arrive", flush=True)
                waiting_dump_since = None
        if bridge.pending:
            # Hand the removals off and keep drawing. The worker owns the mesh.
            worker.submit_cuts(bridge.pending)
            bridge.pending.clear()

        while True:
            result = worker.poll()
            if result is None:
                break
            uploads.append(_begin_upload(result["chunks"], result["full"], result["extract_ms"], result["receipt"]))

        viewer.begin_frame(frame * (1.0 / 60.0))
        if bridge.camera is not None:
            _apply_camera(viewer, bridge.camera)
        if uploads:
            t0 = time.perf_counter()
            if _drain_upload(viewer, uploads[0], UPLOAD_BUDGET_S):
                done = uploads.pop(0)
                last_extract_ms = done["extract_ms"]
                if done["receipt"] is not None:
                    removal_delay_ms.append((time.perf_counter() - done["receipt"]) * 1e3)
                    if len(removal_delay_ms) > 240:
                        del removal_delay_ms[:-240]
                if not uploads and done["full"]:
                    print(f"Bone upload finished, {len(getattr(viewer, '_chunk_names', {}))} meshes", flush=True)
            last_upload_ms = (time.perf_counter() - t0) * 1e3
        if bridge.drill is not None:
            radius = (bridge.burr_mm * 0.001 / METERS_PER_AU) * 0.5
            _log_burr(viewer, bridge.drill["position"], bridge.drill["quat"], radius)
        viewer.end_frame()
        frame += 1
        stamp = time.perf_counter()
        if previous > 0.0:
            frame_ms.append((stamp - previous) * 1e3)
            if len(frame_ms) > 480:
                del frame_ms[:-480]
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
                "triangles": int(len(mesh.tris)),
                "vertices": int(len(mesh.verts)),
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
