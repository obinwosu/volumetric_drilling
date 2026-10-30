#!/usr/bin/env python3
"""FIVRS RTX Tuner.

The view opens on the right transtemporal pose and can orbit. The camera
panel reads and writes ADF/world/canonical_views.yaml, the same slots the
simulator pause menu uses. The light panel writes tuner_light.json. The
structure panel writes tuner_materials.json, keyed by segmentation hex and
by a regular name such as bone. The lateral cortex panel writes
tuner_cortex_mask.json. The shaders panel writes tuner_markings.json. The
irrigation panel writes tuner_irrigation.json. Visibility stays in this window.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import camera_views
import cortex_masks
import markings
import rtx_materials
from fluid.irrigation_bench import (
    DRAW_SPHERES,
    DRAW_TUBE,
    IRRIGATION_PATH,
    STANDOFF_MM,
    IrrigationBench,
    load_irrigation,
    quat_aiming_at_bone,
    save_irrigation,
    screen_down,
    water_from_irrigation,
)
from fluid.occupancy import VolumeFrame
from fluid.outlet import MM_TO_SIM

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
ADF = REPO / "ADF" / "volume_256_RT143.yaml"
SAVE_PATH = HERE / "tuner_light.json"

# Starting point from the bone comparison: a pool wider than 8° and
# dimmer than the 16° frame, so the center is not already clipped.
DEFAULT_LIGHTS = {
    "cone_angle_deg": 12.0,
    "cone_softness": 0.4,
    "intensity_nits": 2.5e6,
    "fill": 0.0,
    "disk_radius": 0.2,
    "disk_standoff": 0.3,
    "disk_color": [1.0, 0.94, 0.86],
    "fill_color": [0.85, 0.78, 0.70],
}


def load_document() -> dict:
    document = {"version": 1, "lights": dict(DEFAULT_LIGHTS)}
    if not SAVE_PATH.is_file():
        return document
    try:
        saved = json.loads(SAVE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {SAVE_PATH}: {exc}", flush=True)
        return document
    lights = dict(DEFAULT_LIGHTS)
    lights.update(saved.get("lights") or {})
    document["lights"] = lights
    return document


def save_document(document: dict) -> None:
    payload = {
        "version": 1,
        "lights": document["lights"],
    }
    temporary = SAVE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(SAVE_PATH)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """AMBF afMatrix3d::setRPY, column-vector convention."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _right_ear_camera():
    """Home camera times the saved right transtemporal view.

    Home looks down -X with up +Z, so its rotation is identity and the
    view translation is added to (1.5, 0, 0).
    """
    from fivrs_rtx_view import _quat_matrix

    rotation = _quat_matrix(-0.380217, 0.0698274, 0.131694, 0.912807)
    position = np.array([1.5, 0.0, 0.0]) + np.array([-2.77177, 0.934857, -1.13597])
    forward = -rotation[:, 0]
    forward = forward / np.linalg.norm(forward)
    up = rotation[:, 2]
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return position, right, up, forward


VERTICAL_FOV_RAD = 0.17951634837990105


def _orthonormal(right, up, forward):
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = np.asarray(right, dtype=np.float64)
    right = right - forward * float(np.dot(right, forward))
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    return right, up, forward


def _rotate_about(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + sine * k + (1.0 - cosine) * (k @ k)


def _scope_matrix(position, right, up, forward):
    """Chai camera: look along local -X, up +Z, right +Y."""
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = right
    mat[1, :3] = up
    mat[2, :3] = -forward
    mat[3, :3] = position
    return mat


def _scope_orbit(scope, d_yaw, d_pitch):
    pivot = scope["pivot"]
    position = scope["position"]
    right, up, forward = _orthonormal(scope["right"], scope["up"], scope["forward"])
    rel = position - pivot
    if abs(float(d_yaw)) > 1e-9:
        rot = _rotate_about(up, d_yaw)
        rel, forward, right, up = rot @ rel, rot @ forward, rot @ right, rot @ up
    if abs(float(d_pitch)) > 1e-9:
        rot = _rotate_about(right, d_pitch)
        rel, forward, right, up = rot @ rel, rot @ forward, rot @ right, rot @ up
    right, up, forward = _orthonormal(right, up, forward)
    scope["position"] = pivot + rel
    scope["right"] = right
    scope["up"] = up
    scope["forward"] = forward


def _scope_pan(scope, delta_right, delta_up):
    right, up, forward = _orthonormal(scope["right"], scope["up"], scope["forward"])
    delta = right * float(delta_right) + up * float(delta_up)
    scope["position"] = scope["position"] + delta
    scope["pivot"] = scope["pivot"] + delta
    scope["right"], scope["up"], scope["forward"] = right, up, forward


def _scope_dolly(scope, amount):
    right, up, forward = _orthonormal(scope["right"], scope["up"], scope["forward"])
    rel = scope["position"] - scope["pivot"]
    distance = max(float(np.linalg.norm(rel)), 0.05)
    new_distance = max(distance * float(math.exp(-amount)), 0.05)
    along = rel / distance
    scope["position"] = scope["pivot"] + along * new_distance
    scope["right"], scope["up"], scope["forward"] = right, up, forward


def _install_scope_mouse(viewer):
    """Orbit the packed microscope camera. Newton yaw/pitch would drop the roll."""
    gui = viewer.gui
    if gui is None:
        return False

    def handle_mouse_drag(x, y, dx, dy, buttons, to_framebuffer_coords, modifiers=0):
        import pyglet

        if gui.should_ignore_mouse_input():
            return
        scope = viewer._scope_cam
        middle = bool(buttons & pyglet.window.mouse.MIDDLE)
        left = bool(buttons & pyglet.window.mouse.LEFT)
        if not (middle or left):
            return
        if modifiers & pyglet.window.key.MOD_SHIFT:
            height = max(float(viewer.camera.height), 1.0)
            scale = 2.0 * max(float(np.linalg.norm(scope["position"] - scope["pivot"])), 0.05)
            scale *= math.tan(0.5 * VERTICAL_FOV_RAD) / height
            _scope_pan(scope, -dx * scale, -dy * scale)
        elif modifiers & pyglet.window.key.MOD_CTRL:
            _scope_dolly(scope, dy * 0.01)
        else:
            _scope_orbit(scope, -dx * 0.1, dy * 0.1)
        viewer._camera_dirty = True

    def handle_mouse_scroll(scroll_y, is_ctrl_down=False):
        if gui.should_ignore_mouse_input():
            return
        if is_ctrl_down:
            return
        _scope_dolly(viewer._scope_cam, scroll_y * 0.15)
        viewer._camera_dirty = True

    gui.handle_mouse_drag = handle_mouse_drag
    gui.handle_mouse_scroll = handle_mouse_scroll
    gui.update_camera_from_keys = lambda *args, **kwargs: None
    return True


def _bone_groups(occupied, colors, bone_rgb):
    from surface import BLOCK
    from warp_mesh import WarpMesher

    print(f"Volume {occupied.shape}, {int(occupied.sum())} occupied voxels", flush=True)
    rotation = _rpy(1.57079, 1.4, 0.0)
    translation = np.array([-6.040592783505154, 0.0, 0.0], dtype=np.float64)
    mesher = WarpMesher()
    mesher.upload_volume(occupied, colors)
    shape = occupied.shape
    blocks = [
        (ix, iy, iz)
        for ix in range(math.ceil(shape[0] / BLOCK))
        for iy in range(math.ceil(shape[1] / BLOCK))
        for iz in range(math.ceil(shape[2] / BLOCK))
    ]
    outputs, _phases = mesher.mesh_blocks(
        blocks,
        smooth=True,
        prepare_pose=(np.ones(3), np.array(shape, dtype=np.float64), rotation, translation),
    )
    groups: dict[int, list] = {}
    bone_key = int(bone_rgb[0]) | (int(bone_rgb[1]) << 8) | (int(bone_rgb[2]) << 16)
    bone_parts = {"index": [], "world": [], "normals": [], "rgb": [], "tris": []}
    bone_base = 0
    for _block, mesh in outputs:
        _verts, tris, rgb, world, normals = mesh
        if world is None or len(tris) == 0:
            continue
        face_rgb = rgb[tris]
        packed = (
            face_rgb[:, :, 0].astype(np.int32)
            | (face_rgb[:, :, 1].astype(np.int32) << 8)
            | (face_rgb[:, :, 2].astype(np.int32) << 16)
        )
        label = np.where(
            (packed[:, 0] == packed[:, 1]) | (packed[:, 0] == packed[:, 2]),
            packed[:, 0],
            np.where(packed[:, 1] == packed[:, 2], packed[:, 1], packed[:, 0]),
        )
        order = np.argsort(label, kind="mergesort")
        ordered = label[order]
        breaks = np.flatnonzero(np.diff(ordered)) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(order)]))
        for start, end in zip(starts, ends):
            key = int(ordered[start])
            group = tris[order[start:end]]
            used, inverse = np.unique(group, return_inverse=True)
            bucket = groups.setdefault(key, [[], [], []])
            base = sum(len(part) for part in bucket[0])
            bucket[0].append(world[used])
            bucket[1].append(normals[used])
            bucket[2].append(inverse.reshape(-1, 3).astype(np.int32) + base)
        bone_faces = label == bone_key
        if np.any(bone_faces):
            group = tris[bone_faces]
            used, inverse = np.unique(group, return_inverse=True)
            bone_parts["index"].append(_verts[used])
            bone_parts["world"].append(world[used])
            bone_parts["normals"].append(normals[used])
            bone_parts["rgb"].append(rgb[used])
            bone_parts["tris"].append(inverse.reshape(-1, 3).astype(np.int32) + bone_base)
            bone_base += int(len(used))
    print(f"Bone mesh in {len(groups)} colors", flush=True)

    def _cat(parts, dtype):
        if not parts:
            return np.zeros((0, 3), dtype)
        return np.ascontiguousarray(np.concatenate(parts), dtype)

    bone_mesh = {
        "index": _cat(bone_parts["index"], np.float64),
        "world": _cat(bone_parts["world"], np.float32),
        "normals": _cat(bone_parts["normals"], np.float32),
        "rgb": _cat(bone_parts["rgb"], np.uint8),
        "tris": _cat(bone_parts["tris"], np.int32),
    }
    return groups, bone_mesh, rotation, translation


def _patch_open_usd():
    import ovstage.population as population

    original = population.open_usd

    def open_usd(stage, path, ordinal, time_code=None, time=math.nan, **kwargs):
        if time_code is not None:
            time = float(time_code)
        return original(stage, path, ordinal, time=time, **kwargs)

    population.open_usd = open_usd


def _quiet_stock_panels():
    """This window is the tuner. The viewer's model panel stays out of it."""
    from newton._src.viewer.viewer_gui import ViewerGui

    ViewerGui._render_left_panel = lambda self: None
    ViewerGui._render_stats_overlay = lambda self: None


def _burr_pose(groups, camera):
    """Park the burr in the lower-right of the scope, 25 mm above the bone."""
    position, right, up, forward = camera
    parts = [np.concatenate(points) for points, _normals, _tris in groups.values() if points]
    if not parts:
        center = position + forward * 1.6
        return center, right, up, forward
    cloud = np.concatenate(parts)
    rel = cloud - position
    along = rel @ forward
    xs = rel @ right
    ys = rel @ up
    visible = along > 0.15
    if not np.any(visible):
        visible = along > 0.0
    pts = cloud[visible]
    along_v = along[visible]
    xs_v = xs[visible]
    ys_v = ys[visible]
    half = math.tan(0.5 * VERTICAL_FOV_RAD)
    target_ax = 0.32 * half
    target_ay = -0.32 * half
    ang = (xs_v / np.maximum(along_v, 1e-6) - target_ax) ** 2 + (
        ys_v / np.maximum(along_v, 1e-6) - target_ay
    ) ** 2
    near = float(np.min(along_v))
    front = along_v < near + 0.12
    if not np.any(front):
        front = np.ones(len(pts), dtype=bool)
    bone_pt = pts[front][int(np.argmin(ang[front]))]
    center = bone_pt - forward * (STANDOFF_MM * MM_TO_SIM)
    return center, right, up, forward


def _author_burr(viewer, groups, camera):
    from fivrs_rtx_view import MESH_DIR, MM_TO_SIM, _load_obj, _sphere
    from surface import vertex_normals

    scale = 6.0 * MM_TO_SIM * 0.5
    center, right, up, forward = _burr_pose(groups, camera)
    flute_path = MESH_DIR / "fluted_burr.obj"
    try:
        points, tris, normals = _load_obj(flute_path)
        print(f"Tuner burr {len(tris)} tris", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        points, tris = _sphere(1.0)
        normals = vertex_normals(points.astype(np.float64), tris).astype(np.float32)
        print(f"Fluted burr unavailable ({exc}); using the diamond sphere", flush=True)
    # Mesh -X is the cutting end. Face that toward the camera so the flutes show.
    axes = np.column_stack((forward, right, up))
    world = np.ascontiguousarray(points @ axes.T * scale + center, np.float32)
    world_normals = np.ascontiguousarray(normals @ axes.T, np.float32)
    color = tuple(rtx_materials.resolve(viewer._material_document, "burr")["diffuse_reflection_color"])
    viewer.log_mesh(
        "burr",
        wp.array(world, dtype=wp.vec3),
        wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
        normals=wp.array(world_normals, dtype=wp.vec3),
        color=color,
        roughness=0.18,
        metallic=0.92,
    )
    prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify("burr")))
    if not prim.IsValid():
        print("No mesh prim for burr", flush=True)
        return None
    rtx_materials.bind_prim(viewer, prim, "burr", viewer._material_document)
    print(
        f"Tuner burr at {center.round(3).tolist()}  {STANDOFF_MM:g} mm above the bone",
        flush=True,
    )
    return {
        "id": "burr",
        "prim": prim,
        "center": center,
        "right": right,
        "up": up,
        "forward": forward,
    }


def _author_shaft(viewer, burr):
    """6 mm handpiece on the burr axis, one mesh per OBJ material."""
    from fivrs_rtx_view import MESH_DIR, _load_obj_parts

    center = np.asarray(burr["center"], dtype=np.float64)
    axes = np.column_stack(
        (
            np.asarray(burr["forward"], dtype=np.float64),
            np.asarray(burr["right"], dtype=np.float64),
            np.asarray(burr["up"], dtype=np.float64),
        )
    )
    try:
        parts = _load_obj_parts(MESH_DIR / "6mm.OBJ")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Tuner shaft unavailable ({exc})", flush=True)
        return []
    segments = []
    for part_id, (points, tris, normals) in parts.items():
        if len(tris) == 0:
            continue
        world = np.ascontiguousarray(points @ axes.T + center, np.float32)
        world_normals = np.ascontiguousarray(normals @ axes.T, np.float32)
        params = rtx_materials.resolve(viewer._material_document, part_id)
        name = f"shaft/{part_id}"
        viewer.log_mesh(
            name,
            wp.array(world, dtype=wp.vec3),
            wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
            normals=wp.array(world_normals, dtype=wp.vec3),
            color=tuple(float(channel) for channel in params["diffuse_reflection_color"]),
            roughness=float(params["specular_reflection_roughness"]),
            metallic=float(params["metalness"]),
        )
        prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(name)))
        if not prim or not prim.IsValid():
            print(f"No mesh prim for {part_id}", flush=True)
            continue
        rtx_materials.bind_prim(viewer, prim, part_id, viewer._material_document)
        segments.append({"id": part_id, "prim": prim})
        label = rtx_materials.display_name(viewer._material_document, part_id)
        print(f"Tuner {label} {len(tris)} tris", flush=True)
    return segments


def _author_diamond(viewer, burr):
    """Grit burr beside the fluted head, at the same 6 mm diameter."""
    from fivrs_rtx_view import MESH_DIR, MM_TO_SIM, _load_obj

    try:
        points, tris, normals = _load_obj(MESH_DIR / "diamond_burr.obj")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Tuner diamond burr unavailable ({exc})", flush=True)
        return None
    scale = 6.0 * MM_TO_SIM * 0.5
    center = np.asarray(burr["center"], dtype=np.float64)
    right = np.asarray(burr["right"], dtype=np.float64)
    center = center + right * (8.0 * MM_TO_SIM)
    axes = np.column_stack(
        (
            np.asarray(burr["forward"], dtype=np.float64),
            right,
            np.asarray(burr["up"], dtype=np.float64),
        )
    )
    world = np.ascontiguousarray(points @ axes.T * scale + center, np.float32)
    world_normals = np.ascontiguousarray(normals @ axes.T, np.float32)
    params = rtx_materials.resolve(viewer._material_document, "burr_diamond")
    viewer.log_mesh(
        "burr_diamond",
        wp.array(world, dtype=wp.vec3),
        wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
        normals=wp.array(world_normals, dtype=wp.vec3),
        color=tuple(float(channel) for channel in params["diffuse_reflection_color"]),
        roughness=float(params["specular_reflection_roughness"]),
        metallic=float(params["metalness"]),
    )
    prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify("burr_diamond")))
    if not prim or not prim.IsValid():
        print("No mesh prim for diamond burr", flush=True)
        return None
    rtx_materials.bind_prim(viewer, prim, "burr_diamond", viewer._material_document)
    print(f"Tuner diamond burr {len(tris)} tris", flush=True)
    return {"id": "burr_diamond", "prim": prim}


def _author_scene(viewer, lights, camera):
    from pxr import Gf, UsdLux

    print("Loading intact volume", flush=True)
    occupied, colors, dimensions, laterality = cortex_masks.load_intact_volume(ADF)
    document = viewer._material_document
    bone_rgb = cortex_masks.named_rgb(document, "bone", "fff9db")
    eac_rgb = cortex_masks.named_rgb(document, "eac", "ffe1d6")
    print("Fitting the EAC lateral frame", flush=True)
    frame = cortex_masks.derive_lateral_frame(
        occupied, colors, eac_rgb, bone_rgb, dimensions, laterality
    )
    frame.dimensions = dimensions
    groups, bone_mesh, rotation, translation = _bone_groups(occupied, colors, bone_rgb)
    print("Classifying lateral cortex masks", flush=True)
    cache = cortex_masks.build_surface_cache(
        bone_mesh["index"],
        bone_mesh["tris"],
        bone_mesh["rgb"],
        bone_mesh["world"],
        bone_mesh["normals"],
        bone_rgb,
        occupied,
        frame,
    )
    segments = []
    bone_surface = None
    if cache.ok:
        preview = cortex_masks.evaluate_masks(cache, cortex_masks.load_parameters())
        bone_surface = cortex_masks.subset_mesh(
            cache.world,
            cache.normals,
            cache.tris,
            ~preview.masks["mask1"],
            0.0,
            with_normals=True,
        )
    for key, (points, normals, tris) in groups.items():
        if sum(len(part) for part in tris) < 40:
            continue
        segment = rtx_materials.segment_id(key)
        name = rtx_materials.name_of(viewer._material_document, segment) or f"bone_{key:06x}"
        color = tuple(rtx_materials.identity_color(segment))
        if bone_surface is not None and segment == frame.bone_hex and len(bone_surface[1]):
            point_src, tri_src, normal_src = bone_surface
        else:
            point_src = np.concatenate(points)
            tri_src = np.concatenate(tris)
            normal_src = np.concatenate(normals)
        viewer.log_mesh(
            name,
            wp.array(np.ascontiguousarray(point_src, np.float32), dtype=wp.vec3),
            wp.array(np.ascontiguousarray(tri_src.reshape(-1), np.int32), dtype=wp.int32),
            normals=wp.array(np.ascontiguousarray(normal_src, np.float32), dtype=wp.vec3),
            color=color,
            roughness=0.55,
            metallic=0.0,
            dynamic=segment == frame.bone_hex,
        )
        prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(name)))
        if not prim.IsValid():
            print(f"No mesh prim for {name}", flush=True)
            continue
        rtx_materials.bind_prim(viewer, prim, segment, viewer._material_document)
        segments.append({"id": segment, "prim": prim})
    segments.sort(key=lambda item: item["id"])
    burr = _author_burr(viewer, groups, camera)
    if burr:
        segments.append(burr)
        diamond = _author_diamond(viewer, burr)
        if diamond:
            segments.append(diamond)
        segments.extend(_author_shaft(viewer, burr))
    disk_color = Gf.Vec3f(*lights["disk_color"])
    disk = UsdLux.DiskLight.Define(viewer.stage, "/root/_ScopeLight")
    disk.CreateIntensityAttr(float(lights["intensity_nits"]))
    disk.CreateRadiusAttr(float(lights["disk_radius"]))
    disk.CreateColorAttr(disk_color)
    shaping = UsdLux.ShapingAPI.Apply(disk.GetPrim())
    shaping.CreateShapingConeAngleAttr(float(lights["cone_angle_deg"]))
    shaping.CreateShapingConeSoftnessAttr(float(lights["cone_softness"]))
    shaping.CreateShapingFocusAttr(1.0)
    dome = UsdLux.DomeLight.Define(viewer.stage, "/root/_ScopeFill")
    dome.CreateIntensityAttr(float(lights["fill"]))
    dome.CreateColorAttr(Gf.Vec3f(*lights["fill_color"]))
    return segments, cache, frame, rotation, translation, occupied, burr


def _bind_light(viewer, lights):
    """Write the packed microscope camera, then keep the disk on that pose.

    Newton's yaw/pitch camera drops roll, so the opening view would not be
    the saved right-transtemporal pose if we used it as the USD xform.
    """
    original = viewer._update_ovrtx_camera

    def update():
        original()
        if viewer._rtx is None:
            return
        scope = getattr(viewer, "_scope_cam", None)
        if scope is None:
            return
        position = np.asarray(scope["position"], dtype=np.float64)
        right, up, forward = _orthonormal(scope["right"], scope["up"], scope["forward"])
        mat = _scope_matrix(position, right, up, forward)
        light = mat.copy()
        light[3, :3] = position + forward * float(lights["disk_standoff"])
        try:
            viewer._write_runtime_attribute(
                [viewer._camera_prim_path],
                "omni:xform",
                np.ascontiguousarray(mat)[np.newaxis, ...],
                is_matrix=True,
            )
            viewer._write_runtime_attribute(
                ["/root/_ScopeLight"],
                "omni:xform",
                np.ascontiguousarray(light)[np.newaxis, ...],
                is_matrix=True,
            )
            values = {
                "/root/_ScopeLight": {
                    "inputs:intensity": float(lights["intensity_nits"]),
                    "inputs:radius": float(lights["disk_radius"]),
                    "inputs:shaping:cone:angle": float(lights["cone_angle_deg"]),
                    "inputs:shaping:cone:softness": float(lights["cone_softness"]),
                    "inputs:shaping:focus": 1.0,
                },
                "/root/_ScopeFill": {"inputs:intensity": float(lights["fill"])},
            }
            for path, attrs in values.items():
                for name, value in attrs.items():
                    viewer._write_runtime_attribute([path], name, np.array([value], dtype=np.float32))
        except Exception as exc:
            if not getattr(viewer, "_tune_warned", False):
                print(f"light write failed: {exc}", flush=True)
                viewer._tune_warned = True
        rtx_materials.flush_materials(viewer)
        markings.flush_runtime(viewer)

    viewer._update_ovrtx_camera = update


def _apply_scope(scope, position, right, up, forward):
    scope["position"] = np.asarray(position, dtype=np.float64).copy()
    scope["right"] = np.asarray(right, dtype=np.float64).copy()
    scope["up"] = np.asarray(up, dtype=np.float64).copy()
    scope["forward"] = np.asarray(forward, dtype=np.float64).copy()


def _apply_ear(state, chosen, write=True):
    side = state.get("side_draft") or ""
    if side not in ("left", "right"):
        state["status"] = "Choose left or right ear."
        return False
    if chosen is None or chosen.get("laterality") == side:
        return True
    chosen["laterality"] = side
    if write:
        camera_views.save_views(state["views"])
        label = chosen.get("label") or chosen["id"]
        state["status"] = f"{label} is a {side} ear view"
    return True


def _rename_view(state, chosen, write=True):
    name = state.get("name_draft", "").strip()
    if chosen is None:
        return False
    if not name:
        state["name_draft"] = chosen.get("label") or chosen["id"]
        state["status"] = "A view needs a name."
        return False
    previous_side = chosen.get("laterality")
    if not _apply_ear(state, chosen, write=False):
        return False
    renamed = name != (chosen.get("label") or "")
    if renamed:
        chosen["label"] = name
    if write and (renamed or chosen.get("laterality") != previous_side):
        camera_views.save_views(state["views"])
        if renamed:
            state["status"] = f"Renamed to {name}"
        else:
            label = chosen.get("label") or chosen["id"]
            state["status"] = f"{label} is a {chosen['laterality']} ear view"
    return True


def _add_named_view(state, translation, quat):
    name = state.get("name_draft", "").strip()
    if not name:
        state["status"] = "Type a name, then add the view."
        return
    if not _apply_ear(state, None, write=False):
        return
    view = {
        "id": camera_views.unique_id(name, [item["id"] for item in state["views"]]),
        "label": name,
        "laterality": state["side_draft"],
        "valid": True,
        "t": [float(channel) for channel in translation],
        "q": [float(channel) for channel in quat],
    }
    state["views"].append(view)
    state["selected"] = view["id"]
    state["name_draft_id"] = view["id"]
    camera_views.save_views(state["views"])
    state["status"] = f"Added {name}"


def _draw_camera_panel(imgui, viewer, state):
    """Current microscope pose, and the named views the simulator recalls."""
    scope = getattr(viewer, "_scope_cam", None)
    if scope is None:
        return
    imgui.set_next_window_pos(imgui.ImVec2(390, 16), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(380, 520), imgui.Cond_.first_use_ever)
    imgui.begin("Camera")
    imgui.text("Main camera")
    imgui.text_disabled("Name a view, mark left or right ear, then save.")
    imgui.separator()
    position = [float(channel) for channel in scope["position"]]
    changed, edited = imgui.drag_float3("Position", position, 0.01, 0.0, 0.0, "%.3f")
    if changed:
        scope["position"] = np.asarray(edited, dtype=np.float64)
    translation, quat = camera_views.view_offset(
        scope["position"], scope["right"], scope["up"], scope["forward"]
    )
    imgui.text_disabled(
        "Home offset  "
        + "  ".join(f"{channel:.3f}" for channel in translation)
    )
    imgui.text_disabled(
        "Quaternion  "
        + "  ".join(f"{channel:.4f}" for channel in quat)
    )
    imgui.separator()
    views = state["views"]
    selected = state.get("selected") or ""
    chosen = None
    for view in views:
        mark = "saved" if view["valid"] else "empty"
        side = view["laterality"] or "either"
        label = f"{view['label'] or view['id']}   {side}   {mark}##{view['id']}"
        clicked, _ = imgui.selectable(label, view["id"] == selected)
        if clicked:
            state["selected"] = view["id"]
        if view["id"] == state.get("selected"):
            chosen = view
    if state.get("name_draft_id") != (chosen["id"] if chosen else None):
        state["name_draft_id"] = chosen["id"] if chosen else None
        state["name_draft"] = (chosen.get("label") or chosen["id"]) if chosen else ""
        stored_side = (chosen or {}).get("laterality") or ""
        state["side_draft"] = stored_side if stored_side in ("left", "right") else ""
    submitted, drafted = imgui.input_text(
        "Name", state.get("name_draft") or "", imgui.InputTextFlags_.enter_returns_true
    )
    state["name_draft"] = drafted
    if submitted:
        _rename_view(state, chosen)
    side = state.get("side_draft") or ""
    if imgui.radio_button("Left ear", side == "left"):
        state["side_draft"] = "left"
        _apply_ear(state, chosen)
    imgui.same_line()
    if imgui.radio_button("Right ear", side == "right"):
        state["side_draft"] = "right"
        _apply_ear(state, chosen)
    imgui.separator()
    if imgui.button("Rename"):
        _rename_view(state, chosen)
    imgui.same_line()
    if imgui.button("New view"):
        _add_named_view(state, translation, quat)
    imgui.separator()
    if imgui.button("Home"):
        _apply_scope(scope, *camera_views.home_pose())
        state["status"] = "Home. This is the ADF camera, not a saved slot."
    imgui.same_line()
    if imgui.button("Go to view") and chosen is not None:
        if chosen["valid"]:
            _apply_scope(scope, *camera_views.pose_from_view(chosen))
            state["status"] = f"Showing {chosen['label'] or chosen['id']}"
        else:
            state["status"] = f"{chosen['label'] or chosen['id']} is empty. Frame it, then save."
    imgui.same_line()
    if imgui.button("Save this view") and chosen is not None and _rename_view(state, chosen, write=False):
        chosen["t"] = [float(channel) for channel in translation]
        chosen["q"] = [float(channel) for channel in quat]
        chosen["valid"] = True
        camera_views.save_views(views)
        state["status"] = f"Saved {chosen['label'] or chosen['id']}"
    if imgui.button("Reload file"):
        state["views"] = camera_views.load_views()
        state["name_draft_id"] = None
        state["status"] = f"Loaded {camera_views.VIEWS_PATH.name}"
    imgui.text_disabled(state["status"])
    imgui.text_disabled(str(camera_views.VIEWS_PATH))
    imgui.end()


def _slider(imgui, label, value, low, high, fmt, flags=0):
    changed, new_value = imgui.slider_float(label, float(value), float(low), float(high), fmt, flags)
    return bool(changed), float(new_value)


def _draw_panel(imgui, document, status):
    lights = document["lights"]
    log_flag = imgui.SliderFlags_.logarithmic.value
    imgui.set_next_window_pos(imgui.ImVec2(16, 16), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(360, 520), imgui.Cond_.first_use_ever)
    imgui.begin("FIVRS RTX Tuner")
    imgui.text("Scope light")
    imgui.text_disabled("Angle and softness, then intensity, then fill.")
    imgui.separator()
    changed = False
    touched, lights["cone_angle_deg"] = _slider(
        imgui, "Cone angle", lights["cone_angle_deg"], 6.0, 24.0, "%.1f deg"
    )
    changed |= touched
    touched, lights["cone_softness"] = _slider(
        imgui, "Softness", lights["cone_softness"], 0.05, 1.0, "%.2f"
    )
    changed |= touched
    imgui.separator()
    touched, lights["intensity_nits"] = _slider(
        imgui, "Intensity", lights["intensity_nits"], 1.0e5, 2.0e7, "%.3g nits", log_flag
    )
    changed |= touched
    imgui.separator()
    touched, lights["fill"] = _slider(imgui, "Fill", lights["fill"], 0.0, 40.0, "%.1f")
    changed |= touched
    imgui.separator()
    imgui.text("Source size")
    imgui.text_disabled("Highlight width. Leave this while shaping the pool.")
    touched, lights["disk_radius"] = _slider(
        imgui, "Disk radius", lights["disk_radius"], 0.05, 0.50, "%.3f"
    )
    changed |= touched
    imgui.separator()
    if imgui.button("Save light") or changed:
        save_document(document)
        status["text"] = f"Saved {SAVE_PATH.name}"
    imgui.text_disabled(status["text"])
    imgui.text_disabled(str(SAVE_PATH))
    imgui.end()


WATER_GROUPS = ("Base", "Specular", "Transmission", "Opacity")


def _ensure_water(irrigation: dict) -> dict:
    water = water_from_irrigation(irrigation, rtx_materials.WATER)
    irrigation["water"] = water
    return water


def _draw_irrigation_panel(imgui, viewer, irrigation, status):
    imgui.set_next_window_pos(imgui.ImVec2(840, 16), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(380, 640), imgui.Cond_.first_use_ever)
    imgui.begin("Irrigation")
    imgui.text("Irrigation")
    imgui.text_disabled("Drag a control. Save writes the live SPH jet and RTX water look.")
    imgui.separator()
    changed = False
    if imgui.button("1 mm tube"):
        irrigation["draw"] = DRAW_TUBE
        irrigation["jet_mm"] = 1.0
        changed = True
    imgui.same_line()
    if imgui.button("0.2 mm spheres"):
        irrigation["draw"] = DRAW_SPHERES
        irrigation["particle_mm"] = 0.2
        irrigation["jet_mm"] = 1.0
        changed = True
    imgui.text_disabled(f"Draw: {irrigation['draw']}")
    imgui.separator()
    touched, irrigation["jet_mm"] = _slider(imgui, "Jet", irrigation["jet_mm"], 0.4, 2.0, "%.2f mm")
    changed |= touched
    touched, irrigation["particle_mm"] = _slider(
        imgui, "Spheres", irrigation["particle_mm"], 0.10, 0.60, "%.2f mm"
    )
    changed |= touched
    touched, irrigation["cc_per_s"] = _slider(imgui, "Flow", irrigation["cc_per_s"], 5.0, 120.0, "%.0f cc/s")
    changed |= touched
    touched, irrigation["film_ms"] = _slider(imgui, "Film", irrigation["film_ms"], 20.0, 80.0, "%.0f ms")
    changed |= touched
    water = _ensure_water(irrigation)
    imgui.separator()
    imgui.text("Water material")
    current = None
    opened = True
    for group, key, label, kind, low, high in rtx_materials.FIELDS:
        if group not in WATER_GROUPS:
            continue
        if group != current:
            current = group
            imgui.set_next_item_open(True, imgui.Cond_.once)
            opened = imgui.collapsing_header(group)
        if not opened:
            continue
        imgui.push_id(f"water_{key}")
        if kind == "bool":
            touched, value = imgui.checkbox(label, bool(water.get(key, False)))
            if touched:
                water[key] = bool(value)
                changed = True
        elif kind == "color":
            color = [float(channel) for channel in water.get(key, [1.0, 1.0, 1.0])]
            touched, value = imgui.color_edit3(label, color)
            if touched:
                water[key] = [float(channel) for channel in value]
                changed = True
        else:
            touched, value = imgui.slider_float(
                label, float(water.get(key, 0.0)), float(low), float(high), "%.3f"
            )
            if touched:
                water[key] = float(value)
                changed = True
        imgui.pop_id()
    irrigation["opacity"] = float(water.get("geometry_opacity", irrigation["opacity"]))
    irrigation["water"] = water
    rtx_materials.apply_params(viewer, "water", dict(water))
    imgui.separator()
    if imgui.button("Save irrigation") or changed:
        save_irrigation(IRRIGATION_PATH, irrigation)
        status["text"] = f"Saved {IRRIGATION_PATH.name} for the simulator"
    imgui.text_disabled(status["text"])
    imgui.text_disabled(str(IRRIGATION_PATH))
    imgui.end()


def main():
    _patch_open_usd()
    _quiet_stock_panels()
    import newton

    document = load_document()
    lights = document["lights"]
    irrigation = load_irrigation(IRRIGATION_PATH)
    camera = _right_ear_camera()
    position, right, up, forward = camera
    viewer = newton.viewer.ViewerRTX(
        width=1280,
        height=720,
        vsync=False,
        headless=False,
        up_axis="Z",
        environment="none",
        async_rendering=True,
        render_settings={
            "omni:rtx:pt:samplesPerPixel": ("uint", 1),
            "omni:rtx:pt:denoising:enabled": ("bool", True),
            "omni:rtx:pt:maxBounces": ("int", 4),
            "omni:rtx:rt:fractionalOpacity": ("bool", True),
        },
    )
    viewer.set_model(newton.ModelBuilder().finalize())
    viewer.camera.fov = float(np.degrees(VERTICAL_FOV_RAD))
    viewer.camera.near = 0.1
    viewer.camera.far = 100.0
    viewer.camera.pos = type(viewer.camera.pos)(float(position[0]), float(position[1]), float(position[2]))
    viewer.camera._set_orientation_from_direction(forward)
    viewer._scope_cam = {
        "position": np.asarray(position, dtype=np.float64).copy(),
        "right": np.asarray(right, dtype=np.float64).copy(),
        "up": np.asarray(up, dtype=np.float64).copy(),
        "forward": np.asarray(forward, dtype=np.float64).copy(),
        "pivot": np.asarray(position + forward * 2.0, dtype=np.float64),
    }
    print(
        f"aim pos {np.round(position, 3).tolist()} forward {np.round(forward, 3).tolist()}",
        flush=True,
    )
    viewer.show_ui = True
    viewer.show_ground = False
    viewer.picking_enabled = False
    rtx_materials.attach(viewer)
    segments, cache, frame, rotation, translation, occupied, pose = _author_scene(
        viewer, lights, camera
    )
    _bind_light(viewer, lights)
    volume_frame = VolumeFrame(
        np.asarray(translation, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64),
        np.asarray(frame.dimensions, dtype=np.float64),
        np.asarray(occupied.shape, dtype=np.float64),
    )
    bench = IrrigationBench(occupied, volume_frame, down=screen_down())
    bench.apply(irrigation)
    rtx_materials.apply_params(viewer, "water", _ensure_water(irrigation))
    bench.prime(viewer)
    status = {"text": "Moves save as you drag."}
    camera_state = {
        "views": camera_views.load_views(),
        "selected": "right",
        "status": "Frame the shot, then save it into a slot.",
    }
    materials = rtx_materials.MaterialUi(viewer, segments)
    cortex = cortex_masks.CortexPanel(
        viewer,
        materials,
        cache,
        frame,
        rotation,
        translation,
        str(ADF.relative_to(REPO)),
    )
    if cortex.pivot is not None:
        viewer._scope_cam["pivot"] = np.asarray(cortex.pivot, dtype=np.float64).copy()
        viewer.camera.set_pivot(tuple(float(value) for value in cortex.pivot))
    shaders = markings.ShaderPanel(viewer, cortex)
    cortex.on_masks_changed = shaders.apply_pattern
    viewer.register_ui_callback(lambda imgui: _draw_panel(imgui, document, status), position="free")
    viewer.register_ui_callback(
        lambda imgui: _draw_camera_panel(imgui, viewer, camera_state), position="free"
    )
    viewer.register_ui_callback(materials.draw, position="free")
    viewer.register_ui_callback(cortex.draw, position="free")
    viewer.register_ui_callback(shaders.draw, position="free")
    viewer.register_ui_callback(
        lambda imgui: _draw_irrigation_panel(imgui, viewer, irrigation, status), position="free"
    )
    print(
        f"Tuner ready. Views {camera_views.VIEWS_PATH.name}  light {SAVE_PATH.name}  "
        f"materials {rtx_materials.MATERIALS_PATH.name}  "
        f"cortex {cortex_masks.CONFIG_PATH.name}  shaders {markings.CONFIG_PATH.name}  "
        f"irrigation {IRRIGATION_PATH.name}",
        flush=True,
    )
    clock = time.perf_counter()
    mouse = False
    irr_mtime = IRRIGATION_PATH.stat().st_mtime if IRRIGATION_PATH.is_file() else None
    while viewer.is_running():
        now = time.perf_counter()
        dt = min(now - clock, 0.05)
        clock = now
        if not mouse:
            mouse = _install_scope_mouse(viewer)
        try:
            file_mtime = IRRIGATION_PATH.stat().st_mtime
        except OSError:
            file_mtime = None
        if file_mtime is not None and irr_mtime is not None and file_mtime > irr_mtime + 1e-4:
            irrigation.update(load_irrigation(IRRIGATION_PATH))
            status["text"] = f"Loaded {IRRIGATION_PATH.name}"
        if file_mtime is not None:
            irr_mtime = file_mtime
        viewer._camera_dirty = True
        viewer.begin_frame(0.0)
        if pose is not None:
            bench.apply(irrigation)
            rtx_materials.apply_params(viewer, "water", _ensure_water(irrigation))
            bench.step(dt, pose["center"], quat_aiming_at_bone(pose["forward"], pose["right"]))
            bench.draw(viewer)
        viewer.end_frame()
        if viewer._window is not None:
            fps = (1.0 / dt) if dt > 1e-4 else 0.0
            viewer._window.set_caption(
                f"FIVRS RTX Tuner  {fps:.0f} fps  {irrigation['draw']}  "
                f"hit {bench.hit_mm:.1f} mm  film {bench.live_film}"
            )
    viewer.close()


if __name__ == "__main__":
    main()
