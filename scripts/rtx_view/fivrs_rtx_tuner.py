#!/usr/bin/env python3
"""FIVRS RTX Tuner.

Without --anatomy-adf the app opens a volume-and-view menu first, then
loads that anatomy. The camera panel reads and writes
ADF/world/canonical_views.yaml, the same slots the simulator pause menu
uses. The light panel writes tuner_light.json. The structure panel writes
tuner_materials.json, keyed by segmentation hex and by a regular name
such as bone. The lateral cortex panel writes tuner_cortex_mask.json. The
shaders panel writes tuner_markings.json. The irrigation panel writes
tuner_irrigation.json. The paste panel writes paste_look.json and the
grain size in debris.json. Visibility stays in this window.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import yaml

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import camera_views
import cortex_masks
import markings
import nerves
import or_room
import rtx_materials
import tuner_launch_dialog
from bone_debris import CONFIG_PATH as DEBRIS_PATH, load_config as load_debris
from fluid.irrigation_bench import (
    DRAW_SPHERES,
    DRAW_TUBE,
    IRRIGATION_PATH,
    STANDOFF_MM,
    IrrigationBench,
    film_look_from_irrigation,
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
DEFAULT_ADF = REPO / "ADF" / "volume_256_RT143.yaml"
ADF = DEFAULT_ADF
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
    **or_room.DEFAULTS,
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


def _adf_laterality(path: pathlib.Path) -> str:
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    for value in data.values():
        if isinstance(value, dict) and "laterality" in value:
            lat = str(value["laterality"]).strip().lower()
            if lat in ("left", "right"):
                return lat
    return ""


def _startup_camera(view_id: str, laterality: str):
    """Named view from canonical_views.yaml, or the laterality default."""
    if view_id == "home":
        return camera_views.home_pose()
    views = camera_views.load_views()
    if view_id and view_id != "default":
        for view in views:
            if view["id"] == view_id and view.get("valid"):
                return camera_views.pose_from_view(view)
    side = (laterality or "").strip().lower()
    if side in ("left", "right"):
        for view in views:
            if view["id"] == side and view.get("valid"):
                return camera_views.pose_from_view(view)
    for view in views:
        if view["id"] == "right" and view.get("valid"):
            return camera_views.pose_from_view(view)
    return camera_views.home_pose()


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


def _face_packed(rgb_v, tris):
    packed = (
        rgb_v[:, 0].astype(np.int32)
        | (rgb_v[:, 1].astype(np.int32) << 8)
        | (rgb_v[:, 2].astype(np.int32) << 16)
    )
    first, second, third = packed[tris[:, 0]], packed[tris[:, 1]], packed[tris[:, 2]]
    agree = (first == second) | (first == third)
    return np.where(agree, first, np.where(second == third, second, first))


def _bone_groups(occupied, colors, bone_rgb, document=None, close_all=False):
    from surface import (
        BLOCK,
        blocks_for_mask,
        closed_label_mask,
        occupancy_for_rgb,
        packed_rgb,
        rgb_hex,
        segment_colors,
    )
    from warp_mesh import WarpMesher

    print(f"Volume {occupied.shape}, {int(occupied.sum())} occupied voxels", flush=True)
    rotation = _rpy(1.57079, 1.4, 0.0)
    translation = np.array([-6.040592783505154, 0.0, 0.0], dtype=np.float64)
    pose = (np.ones(3), np.array(occupied.shape, dtype=np.float64), rotation, translation)
    mesher = WarpMesher()
    mesher.upload_volume(occupied, colors)
    groups: dict[int, list] = {}
    bone_key = packed_rgb(bone_rgb)
    wanted = rtx_materials.closed_structure_hexes(document)
    bone_parts = {"index": [], "world": [], "normals": [], "rgb": [], "tris": []}
    bone_base = [0]
    labels = segment_colors(occupied, colors)
    if close_all:
        closed_rgbs = list(labels)
        wanted = {
            rgb_hex(rgb): rtx_materials.name_of(document or {}, rgb_hex(rgb)) or rgb_hex(rgb)
            for rgb in labels
        }
    else:
        closed_rgbs = [rgb for rgb in labels if rgb_hex(rgb) in wanted]
    remesh = np.asarray(occupied, dtype=bool).copy()
    if closed_rgbs and not close_all:
        remesh[closed_label_mask(occupied, colors, closed_rgbs)] = False

    def _append(key, verts, tris, rgb_v, world, normals):
        if len(tris) == 0:
            return 0
        used, inverse = np.unique(tris, return_inverse=True)
        bucket = groups.setdefault(int(key), [[], [], []])
        base = sum(len(part) for part in bucket[0])
        bucket[0].append(world[used])
        bucket[1].append(normals[used])
        bucket[2].append(inverse.reshape(-1, 3).astype(np.int32) + base)
        if int(key) == bone_key:
            bone_parts["index"].append(verts[used])
            bone_parts["world"].append(world[used])
            bone_parts["normals"].append(normals[used])
            bone_parts["rgb"].append(rgb_v[used])
            bone_parts["tris"].append(inverse.reshape(-1, 3).astype(np.int32) + bone_base[0])
            bone_base[0] += int(len(used))
        return int(len(tris))

    if close_all:
        print(
            f"Meshing every label as its own solid ({len(closed_rgbs)} structures)",
            flush=True,
        )
    else:
        print(
            f"Meshing union occupancy for bone, EAC, sinus/dura, and other open labels "
            f"({len(closed_rgbs)} closed structures separate)",
            flush=True,
        )
    union_blocks = [] if close_all else blocks_for_mask(remesh, BLOCK)
    mesher.set_occupancy(remesh)
    union_tris: dict[int, int] = {}
    if union_blocks:
        outputs, _phases = mesher.mesh_blocks(union_blocks, smooth=True, prepare_pose=pose)
        for _block, mesh in outputs:
            verts, tris, rgb_v, world, normals = mesh
            if world is None or len(tris) == 0:
                continue
            face = _face_packed(rgb_v, tris)
            for key in np.unique(face):
                union_tris[int(key)] = union_tris.get(int(key), 0) + _append(
                    key, verts, tris[face == key], rgb_v, world, normals
                )
    for key, tris_n in sorted(union_tris.items(), key=lambda item: -item[1]):
        hex_id = rtx_materials.segment_id(key)
        alias = rtx_materials.name_of(document or {}, hex_id) or hex_id
        print(f"  union {alias} #{hex_id}  {tris_n} tris", flush=True)

    for rgb in closed_rgbs:
        mask = occupancy_for_rgb(occupied, colors, rgb)
        blocks = blocks_for_mask(mask, BLOCK)
        if not blocks:
            continue
        key = packed_rgb(rgb)
        hex_id = rgb_hex(rgb)
        alias = wanted.get(hex_id) or hex_id
        mesher.set_occupancy(mask)
        outputs, _phases = mesher.mesh_blocks(blocks, smooth=True, prepare_pose=pose)
        tris_n = 0
        for _block, mesh in outputs:
            verts, tris, rgb_v, world, normals = mesh
            if world is None or len(tris) == 0:
                continue
            tris_n += _append(key, verts, tris, rgb_v, world, normals)
        print(
            f"  closed {alias} #{hex_id}  {int(mask.sum())} voxels  {len(blocks)} blocks  {tris_n} tris",
            flush=True,
        )
    mesher.set_occupancy(occupied)
    print(f"Surfaces for {len(groups)} structures", flush=True)

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


# Close enough that the jet is obvious, far enough that it is not buried in the bone.
PREVIEW_STANDOFF_MM = 15.0
# Same burr the tuner shaft is built for. The live window emits from this cannula.
PREVIEW_SIZE_MM = 6


def _burr_pose(groups, camera):
    """Park the burr in the middle of the scope, just off the bone toward the camera."""
    position, right, up, forward = camera
    parts = [np.concatenate(points) for points, _normals, _tris in groups.values() if points]
    if not parts:
        center = position + forward * 1.6
        return center, right, up, forward, center + forward * (PREVIEW_STANDOFF_MM * MM_TO_SIM)
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
    target_ax = 0.0
    target_ay = 0.05 * half
    ang = (xs_v / np.maximum(along_v, 1e-6) - target_ax) ** 2 + (
        ys_v / np.maximum(along_v, 1e-6) - target_ay
    ) ** 2
    near = float(np.min(along_v))
    front = along_v < near + 0.12
    if not np.any(front):
        front = np.ones(len(pts), dtype=bool)
    bone_pt = pts[front][int(np.argmin(ang[front]))]
    center = bone_pt - forward * (PREVIEW_STANDOFF_MM * MM_TO_SIM)
    return center, right, up, forward, bone_pt


def _author_burr(viewer, groups, camera):
    from fivrs_rtx_view import MESH_DIR, MM_TO_SIM, _load_obj, _sphere
    from surface import vertex_normals

    scale = 6.0 * MM_TO_SIM * 0.5
    center, right, up, forward, bone_pt = _burr_pose(groups, camera)
    flute_path = MESH_DIR / "fluted_burr.obj"
    try:
        points, tris, normals = _load_obj(flute_path)
        print(f"Tuner burr {len(tris)} tris", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        points, tris = _sphere(1.0)
        normals = vertex_normals(points.astype(np.float64), tris).astype(np.float32)
        print(f"Fluted burr unavailable ({exc}); using the diamond sphere", flush=True)
    # Mesh -X is the cutting end. Point it at the bone, so the handpiece is toward the camera.
    axes = np.column_stack((-forward, right, up))
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
        f"Tuner burr at {center.round(3).tolist()}  {PREVIEW_STANDOFF_MM:g} mm off the bone",
        flush=True,
    )
    return {
        "id": "burr",
        "prim": prim,
        "center": center,
        "right": right,
        "up": up,
        "forward": forward,
        "bone": bone_pt,
    }


def _author_shaft(viewer, burr):
    """6 mm handpiece on the burr axis, one mesh per OBJ material."""
    from fivrs_rtx_view import MESH_DIR, _load_obj_parts

    center = np.asarray(burr["center"], dtype=np.float64)
    axes = np.column_stack(
        (
            -np.asarray(burr["forward"], dtype=np.float64),
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
            -np.asarray(burr["forward"], dtype=np.float64),
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


def _author_cannula(viewer, burr):
    """6 mm irrigation tube on the same pose as the handpiece. The jet leaves its opening."""
    from fluid.cannula import load_obj as load_cannula_obj, mesh_path
    from surface import vertex_normals

    try:
        points, tris = load_cannula_obj(mesh_path(PREVIEW_SIZE_MM))
    except (OSError, ValueError) as exc:
        print(f"Tuner cannula unavailable ({exc})", flush=True)
        return None
    normals = vertex_normals(points, tris).astype(np.float32)
    center = np.asarray(burr["center"], dtype=np.float64)
    axes = np.column_stack(
        (
            -np.asarray(burr["forward"], dtype=np.float64),
            np.asarray(burr["right"], dtype=np.float64),
            np.asarray(burr["up"], dtype=np.float64),
        )
    )
    world = np.ascontiguousarray(points @ axes.T + center, np.float32)
    world_normals = np.ascontiguousarray(normals @ axes.T, np.float32)
    params = rtx_materials.resolve(viewer._material_document, "cannula")
    viewer.log_mesh(
        "cannula",
        wp.array(world, dtype=wp.vec3),
        wp.array(np.ascontiguousarray(tris.reshape(-1), np.int32), dtype=wp.int32),
        normals=wp.array(world_normals, dtype=wp.vec3),
        color=tuple(float(channel) for channel in params["diffuse_reflection_color"]),
        roughness=float(params["specular_reflection_roughness"]),
        metallic=float(params["metalness"]),
    )
    prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify("cannula")))
    if not prim or not prim.IsValid():
        print("No mesh prim for cannula", flush=True)
        return None
    rtx_materials.bind_prim(viewer, prim, "cannula", viewer._material_document)
    print(f"Tuner cannula {PREVIEW_SIZE_MM:g} mm {len(tris)} tris", flush=True)
    return {"id": "cannula", "prim": prim}


def _author_scene(viewer, lights, camera, close_all=False):
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
    groups, bone_mesh, rotation, translation = _bone_groups(
        occupied, colors, bone_rgb, document, close_all=close_all
    )
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
    closed_hexes = set(rtx_materials.closed_structure_hexes(viewer._material_document))
    if close_all:
        closed_hexes = {rtx_materials.segment_id(key) for key in groups}
        closed_hexes.discard(frame.bone_hex)
    for key, (points, normals, tris) in groups.items():
        if sum(len(part) for part in tris) < (4 if close_all else 40):
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
            if segment in closed_hexes and len(point_src):
                # Sit just off the bone canal wall so the closed tube does not z-fight.
                point_src = point_src + normal_src * np.float32(0.0004)
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
        if nerves.is_nerve(name):
            # The tuner meshes the volume in a unit cube; the ADF dimensions give sim mm.
            mm_per_local = np.asarray(dimensions, np.float64) / nerves.MM_TO_SIM
            viewer._nerve_scene_per_mm = nerves.MM_TO_SIM / float(np.mean(dimensions))
            try:
                nerves.attach(
                    viewer,
                    prim,
                    name,
                    nerves.world_to_mm(point_src, rotation, translation, mm_per_local),
                    tri_src,
                )
            except Exception as exc:
                print(f"Nerve shader skipped on {name}: {exc}", flush=True)
        segments.append({"id": segment, "prim": prim})
    segments.sort(key=lambda item: item["id"])
    burr = _author_burr(viewer, groups, camera)
    if burr:
        segments.append(burr)
        diamond = _author_diamond(viewer, burr)
        if diamond:
            segments.append(diamond)
        segments.extend(_author_shaft(viewer, burr))
        cannula = _author_cannula(viewer, burr)
        if cannula:
            segments.append(cannula)
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
    or_room.author_room(viewer.stage, "/root/_ScopeFill", lights)
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
                "/root/_ScopeFill": or_room.dome_inputs(lights),
            }
            for path in or_room.OR_LIGHT_PATHS:
                values[path] = or_room.or_light_inputs(lights)
            for path, attrs in values.items():
                for name, value in attrs.items():
                    viewer._write_runtime_attribute([path], name, np.array([value], dtype=np.float32))
        except Exception as exc:
            if not getattr(viewer, "_tune_warned", False):
                print(f"light write failed: {exc}", flush=True)
                viewer._tune_warned = True
        rtx_materials.flush_materials(viewer)
        markings.flush_runtime(viewer)
        nerves.flush_runtime(viewer)

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


PANELS_PATH = HERE / "tuner_panels.json"


def _load_panels(names) -> dict:
    """Cards start closed. Only the top toggle bar is visible."""
    return {name: False for name in names}


def _draw_panel_bar(imgui, shown: dict) -> None:
    """One row of toggles. Each tuner window only draws while its box is ticked."""
    display = imgui.get_io().display_size
    imgui.set_next_window_pos(imgui.ImVec2(display[0] * 0.5, 4.0), imgui.Cond_.always, imgui.ImVec2(0.5, 0.0))
    flags = (
        imgui.WindowFlags_.no_title_bar
        | imgui.WindowFlags_.always_auto_resize
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
    )
    imgui.begin("Panels", None, flags)
    changed = False
    for index, name in enumerate(shown):
        if index:
            imgui.same_line()
        touched, shown[name] = imgui.checkbox(name, shown[name])
        changed |= touched
    imgui.end()
    if changed:
        PANELS_PATH.write_text(json.dumps(shown, indent=2) + "\n")


def _register_panels(viewer, panels) -> None:
    shown = _load_panels([name for name, _draw in panels])
    viewer.register_ui_callback(lambda imgui: _draw_panel_bar(imgui, shown), position="free")
    for name, draw in panels:
        viewer.register_ui_callback(
            lambda imgui, name=name, draw=draw: draw(imgui) if shown[name] else None, position="free"
        )


def _section(imgui, label: str, open_first: bool) -> bool:
    """Collapsible block. open_first is its state when the tuner starts."""
    imgui.set_next_item_open(open_first, imgui.Cond_.once)
    return bool(imgui.collapsing_header(label))


def _draw_panel(imgui, document, status):
    lights = document["lights"]
    log_flag = imgui.SliderFlags_.logarithmic.value
    imgui.set_next_window_pos(imgui.ImVec2(16, 16), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(360, 520), imgui.Cond_.first_use_ever)
    imgui.begin("Light")
    changed = False
    if _section(imgui, "Scope light", True):
        imgui.text_disabled("Angle and softness, then intensity, then fill.")
        touched, lights["cone_angle_deg"] = _slider(
            imgui, "Cone angle", lights["cone_angle_deg"], 6.0, 24.0, "%.1f deg"
        )
        changed |= touched
        touched, lights["cone_softness"] = _slider(
            imgui, "Softness", lights["cone_softness"], 0.05, 1.0, "%.2f"
        )
        changed |= touched
        touched, lights["intensity_nits"] = _slider(
            imgui, "Intensity", lights["intensity_nits"], 1.0e5, 2.0e7, "%.3g nits", log_flag
        )
        changed |= touched
        touched, lights["fill"] = _slider(imgui, "Fill", lights["fill"], 0.0, 200.0, "%.1f")
        changed |= touched
        imgui.text_disabled("Fill is the light the room puts on the bone.")
        touched, lights["disk_radius"] = _slider(
            imgui, "Disk radius", lights["disk_radius"], 0.05, 0.50, "%.3f"
        )
        changed |= touched
        imgui.text_disabled("Disk radius is the highlight width.")
    if _section(imgui, "Operating room", True):
        imgui.text_disabled("Seen in the wet bone only. It does not light the bone.")
        touched, lights["room_nits"] = _slider(
            imgui, "Room in reflections", lights["room_nits"], 0.0, 1.0e5, "%.3g nits", log_flag
        )
        changed |= touched
        touched, lights["or_lights"] = _slider(
            imgui, "Overhead lights", lights["or_lights"], 0.0, 1.0e6, "%.3g nits", log_flag
        )
        changed |= touched
        touched, lights["room"] = _slider(imgui, "Room", lights["room"], 0.0, 1.0, "%.2f")
        changed |= touched
        touched, color = imgui.color_edit3("Drapes", [float(c) for c in lights["drape_color"]])
        if touched:
            lights["drape_color"] = [float(c) for c in color]
            changed = True
        imgui.text_disabled("Room 0 is the old flat fill in Fill color. Room and drapes apply on restart.")
    if _section(imgui, "Camera", False):
        touched, lights["bloom"] = _slider(imgui, "Bloom", lights["bloom"], 0.0, 2.0, "%.2f")
        changed |= touched
        touched, lights["bloom_cutoff"] = _slider(
            imgui, "Bloom cutoff", lights["bloom_cutoff"], 0.5, 10.0, "%.2f", log_flag
        )
        changed |= touched
        imgui.text_disabled("Bloom applies when both windows restart.")
    imgui.separator()
    if imgui.button("Save light") or changed:
        save_document(document)
        status["text"] = f"Saved {SAVE_PATH.name}"
    imgui.text_disabled(status["text"])
    imgui.text_disabled(str(SAVE_PATH))
    imgui.end()


WATER_GROUPS = ("Base", "Specular", "Transmission", "Opacity")
# Ray bounces the renderer was created with. It cannot change while running.
_START_BOUNCES = 4

# The scope light is a small disk. The path tracer cannot light bone through a
# refracting or transmissive surface from it, so water drawn that way is black
# from every angle. Fractional opacity lets the light and the bone through;
# what is opaque is a water-like reflection with no body color.
# Specular roughness is left alone so the highlights you tuned stay.
# Glint boost is the strength an opaque film would have; opacity does not dim it.
WATER_PRESETS = {
    "Clear water": {
        "glint_boost": 4.0,
        "diffuse_reflection_weight": 0.0,
        "diffuse_reflection_color": [1.0, 1.0, 1.0],
        "metalness": 0.0,
        "specular_reflection_weight": 1.0,
        "specular_reflection_color": [1.0, 1.0, 1.0],
        "specular_reflection_ior": 1.33,
        "enable_specular_transmission": False,
        "specular_transmission_weight": 0.0,
        "specular_transmission_scattering_depth": 0.0,
        "thin_walled": True,
        "enable_opacity": True,
        "geometry_opacity": 0.2,
    },
    "Wet bone": {
        "glint_boost": 1.0,
        "diffuse_reflection_weight": 0.06,
        "diffuse_reflection_color": [0.7, 0.88, 1.0],
        "metalness": 0.0,
        "specular_reflection_weight": 1.0,
        "specular_reflection_color": [1.0, 1.0, 1.0],
        "specular_reflection_ior": 1.33,
        "enable_specular_transmission": False,
        "specular_transmission_weight": 0.0,
        "specular_transmission_scattering_depth": 0.0,
        "thin_walled": True,
        "enable_opacity": True,
        # Light passes the film twice, in and back out, so bone under it reads
        # at about (1 - opacity) squared.
        "geometry_opacity": 0.25,
    },
    "Pooled water": {
        "glint_boost": 1.0,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": [0.75, 0.9, 1.0],
        "metalness": 0.0,
        "specular_reflection_weight": 1.0,
        "specular_reflection_color": [1.0, 1.0, 1.0],
        "specular_reflection_ior": 1.33,
        "enable_specular_transmission": False,
        "specular_transmission_weight": 0.0,
        "specular_transmission_scattering_depth": 0.0,
        "thin_walled": True,
        "enable_opacity": True,
        "geometry_opacity": 0.35,
    },
}


def _ensure_water(irrigation: dict) -> dict:
    water = water_from_irrigation(irrigation, rtx_materials.WATER)
    irrigation["water"] = water
    return water


def _ensure_film(irrigation: dict) -> dict:
    film = film_look_from_irrigation(irrigation, rtx_materials.WATER)
    irrigation["film_look"] = film
    return film


def _draw_look_fields(imgui, look, prefix) -> bool:
    changed = False
    current = None
    opened = True
    for group, key, label, kind, low, high in rtx_materials.FIELDS:
        if group not in WATER_GROUPS:
            continue
        if group != current:
            current = group
            opened = _section(imgui, f"{prefix} {group}", False)
        if not opened:
            continue
        imgui.push_id(f"{prefix}_{key}")
        if kind == "bool":
            touched, value = imgui.checkbox(label, bool(look.get(key, False)))
            if touched:
                look[key] = bool(value)
                changed = True
        elif kind == "color":
            color = [float(channel) for channel in look.get(key, [1.0, 1.0, 1.0])]
            touched, value = imgui.color_edit3(label, color)
            if touched:
                look[key] = [float(channel) for channel in value]
                changed = True
        else:
            # Scene units are about 99 mm, so useful water depths sit near 0.01-0.1.
            flags = imgui.SliderFlags_.logarithmic.value if key == "specular_transmission_scattering_depth" else 0
            touched, value = imgui.slider_float(
                label, float(look.get(key, 0.0)), float(low), float(high), "%.3f", flags
            )
            if touched:
                look[key] = float(value)
                changed = True
        imgui.pop_id()
    return changed


def _draw_irrigation_panel(imgui, viewer, irrigation, status):
    imgui.set_next_window_pos(imgui.ImVec2(840, 16), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(400, 860), imgui.Cond_.first_use_ever)
    imgui.begin("Irrigation")
    changed = False
    running = bool(irrigation.get("enabled", True))
    if imgui.button("Turn irrigation off" if running else "Turn irrigation on"):
        irrigation["enabled"] = not running
        changed = True
    imgui.same_line()
    imgui.text("on" if running else "off")
    if _section(imgui, "Jet", False):
        imgui.text_disabled("The column leaving the cannula.")
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
        touched, irrigation["jet_mm"] = _slider(imgui, "Jet", irrigation["jet_mm"], 0.4, 2.0, "%.2f mm")
        changed |= touched
        touched, irrigation["particle_mm"] = _slider(
            imgui, "Spheres", irrigation["particle_mm"], 0.10, 0.60, "%.2f mm"
        )
        changed |= touched
        touched, irrigation["cc_per_s"] = _slider(imgui, "Flow", irrigation["cc_per_s"], 5.0, 120.0, "%.0f cc/s")
        changed |= touched
    if _section(imgui, "Film flow", True):
        touched, irrigation["film_spread_mm"] = _slider(
            imgui, "Film width", float(irrigation.get("film_spread_mm", 20.0)), 2.0, 30.0, "%.1f mm"
        )
        changed |= touched
        touched, irrigation["film_depth_mm"] = _slider(
            imgui, "Film depth", float(irrigation.get("film_depth_mm", 3.0)), 0.2, 6.0, "%.2f mm"
        )
        changed |= touched
        touched, irrigation["film_life_s"] = _slider(
            imgui, "Life", float(irrigation.get("film_life_s", 8.0)), 0.2, 20.0, "%.1f s"
        )
        changed |= touched
        imgui.text_disabled("How long water stays. Longer fills the cavity and runs down the bone.")
        touched, irrigation["film_smooth"] = _slider(
            imgui, "Smooth", float(irrigation.get("film_smooth", 0.65)), 0.0, 1.0, "%.2f"
        )
        changed |= touched
        imgui.text_disabled("Blends the 0.5 mm cells. The facets are that grid, not particles.")
        touched, irrigation["film_ripple"] = _slider(
            imgui, "Ripple", float(irrigation.get("film_ripple", 0.5)), 0.0, 1.0, "%.2f"
        )
        changed |= touched
        imgui.text_disabled("Small waves in the highlights that run downhill.")
        touched, irrigation["film_rivulets"] = _slider(
            imgui, "Rivulets", float(irrigation.get("film_rivulets", 0.5)), 0.0, 1.0, "%.2f"
        )
        changed |= touched
        imgui.text_disabled("0 spreads as an even sheet. 1 breaks the front into streams.")
    gravity = [float(channel) for channel in (irrigation.get("gravity") or [0.0, 0.0, -1.0])]
    while len(gravity) < 3:
        gravity.append(0.0)
    gravity = gravity[:3]
    if _section(imgui, "Gravity", False):
        imgui.text_disabled("Downhill direction in the scene. World down is (0, 0, −1).")
        for index, label in enumerate(("Down X", "Down Y", "Down Z")):
            touched, gravity[index] = _slider(imgui, label, gravity[index], -1.0, 1.0, "%.2f")
            changed |= touched
        if imgui.button("World down"):
            gravity = [0.0, 0.0, -1.0]
            changed = True
    irrigation["gravity"] = gravity
    water = _ensure_water(irrigation)
    film = _ensure_film(irrigation)
    if _section(imgui, "Film shader", True):
        imgui.text_disabled("Opacity sets how much bone shows through. Transmission or Thin walled off go black.")
        for index, (name, preset) in enumerate(WATER_PRESETS.items()):
            if index:
                imgui.same_line()
            if imgui.button(name):
                film.update({key: (list(value) if isinstance(value, list) else value) for key, value in preset.items()})
                changed = True
        touched, film[rtx_materials.GLINT_KEY] = _slider(
            imgui, "Glint boost", float(film.get(rtx_materials.GLINT_KEY, 1.0)), 1.0, 10.0, "%.1fx"
        )
        changed |= touched
        imgui.text_disabled("Glints keep this strength at any opacity. Opacity only sets how much bone shows.")
        changed |= _draw_look_fields(imgui, film, "Film")
    if _section(imgui, "Stream shader", False):
        changed |= _draw_look_fields(imgui, water, "Stream")
    irrigation["opacity"] = float(water.get("geometry_opacity", irrigation["opacity"]))
    irrigation["water"] = water
    if _section(imgui, "Render", False):
        bounces = int(irrigation.get("max_bounces", 4))
        touched, bounces = imgui.slider_int("Max bounces", bounces, 2, 12)
        if touched:
            irrigation["max_bounces"] = int(bounces)
            changed = True
        imgui.text_disabled(
            f"Refracting water needs 6-8; black specks mean too few. Restart both windows to apply (now {_START_BOUNCES})."
        )
    irrigation["film_look"] = film
    rtx_materials.apply_params(viewer, "water", dict(water))
    rtx_materials.apply_params(viewer, "film", dict(film))
    imgui.separator()
    if imgui.button("Save irrigation") or changed:
        save_irrigation(IRRIGATION_PATH, irrigation)
        status["text"] = f"Saved {IRRIGATION_PATH.name} for the simulator"
    imgui.text_disabled(status["text"])
    imgui.text_disabled(str(IRRIGATION_PATH))
    imgui.end()


def _paste_smear(center, right, up, radius_mm, clump_mm):
    rng = np.random.default_rng(4)
    count = 48
    angle = rng.random(count) * (2.0 * math.pi)
    span = np.sqrt(rng.random(count)) * (2.2 * MM_TO_SIM)
    points = (
        np.asarray(center, dtype=np.float64)
        + np.asarray(right, dtype=np.float64) * (np.cos(angle) * span)[:, None]
        + np.asarray(up, dtype=np.float64) * (np.sin(angle) * span * 0.6)[:, None]
    )
    sizes = float(radius_mm) + (float(clump_mm) - float(radius_mm)) * np.sqrt(rng.random(count))
    return points.astype(np.float32), (sizes * MM_TO_SIM).astype(np.float32)


def _write_paste(state) -> None:
    look = rtx_materials.slurry_params()
    look["diffuse_reflection_color"] = [float(channel) for channel in state["color"]]
    look["specular_reflection_roughness"] = float(state["roughness"])
    look["coat_weight"] = float(state["coat"])
    temporary = rtx_materials.PASTE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(look, indent=2) + "\n")
    temporary.replace(rtx_materials.PASTE_PATH)
    try:
        saved = json.loads(DEBRIS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    saved["radius_mm"] = float(state["radius_mm"])
    saved["clump_mm"] = float(state["clump_mm"])
    saved["color"] = look["diffuse_reflection_color"]
    temporary = DEBRIS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(saved, indent=2) + "\n")
    temporary.replace(DEBRIS_PATH)
    state["look"] = look


def _save_tuner_picture(viewer) -> str:
    path = pathlib.Path("/tmp/fivrs-rtx/paste_preview.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = viewer.get_frame()
    pixels = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    from PIL import Image

    Image.fromarray(pixels).save(path)
    return str(path)


def _draw_paste_panel(imgui, viewer, state):
    imgui.set_next_window_pos(imgui.ImVec2(16, 560), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(340, 300), imgui.Cond_.first_use_ever)
    imgui.begin("Paste")
    imgui.text("Bone-dust paste")
    imgui.text_disabled("Shown on the bone under the burr. A running microscope follows the file.")
    changed, state["show"] = imgui.checkbox("Show smear", bool(state["show"]))
    touched, state["radius_mm"] = _slider(imgui, "Small grain", state["radius_mm"], 0.05, 1.2, "%.2f mm")
    changed |= touched
    touched, state["clump_mm"] = _slider(
        imgui, "Large grain", state["clump_mm"], state["radius_mm"], 1.4, "%.2f mm"
    )
    changed |= touched
    touched, state["roughness"] = _slider(imgui, "Roughness", state["roughness"], 0.02, 0.6, "%.2f")
    changed |= touched
    touched, state["coat"] = _slider(imgui, "Wet coat", state["coat"], 0.0, 1.0, "%.2f")
    changed |= touched
    touched, color = imgui.color_edit3("Color", state["color"])
    if touched:
        state["color"] = [float(channel) for channel in color]
        changed = True
    if changed:
        state["clump_mm"] = max(float(state["clump_mm"]), float(state["radius_mm"]))
        _write_paste(state)
        rtx_materials.apply_params(viewer, "slurry", state["look"])
        state["status"] = "Wrote paste_look.json and debris.json"
    if imgui.button("Save picture"):
        try:
            state["status"] = _save_tuner_picture(viewer)
        except Exception as exc:
            state["status"] = f"Picture failed: {exc}"
    imgui.text_disabled(state["status"])
    imgui.end()


def _draw_paste(viewer, pose, state) -> None:
    if pose is None or not state["show"]:
        if state.get("visible"):
            viewer.log_points("swatch/paste", None, hidden=True)
            state["visible"] = False
        return
    bone = np.asarray(pose["center"], dtype=np.float64) + np.asarray(pose["forward"], dtype=np.float64) * (
        STANDOFF_MM * MM_TO_SIM
    )
    lift = -np.asarray(pose["forward"], dtype=np.float64) * (0.4 * MM_TO_SIM)
    points, radii = _paste_smear(bone + lift, pose["right"], pose["up"], state["radius_mm"], state["clump_mm"])
    viewer.log_points(
        "swatch/paste",
        wp.array(points, dtype=wp.vec3),
        radii=wp.array(radii, dtype=wp.float32),
        colors=tuple(float(channel) for channel in state["color"]),
    )
    state["visible"] = True
    if not state.get("bound"):
        instancer = viewer._ensure_point_batch_primitive(viewer._qualify("swatch/paste"))
        sphere = instancer.GetPrim().GetChild("sphere") if instancer is not None else None
        if sphere:
            rtx_materials.bind_slurry(viewer, sphere)
            state["bound"] = True


def _parse_args():
    parser = argparse.ArgumentParser(description="FIVRS RTX Tuner")
    parser.add_argument(
        "--anatomy-adf",
        type=lambda p: pathlib.Path(p).expanduser(),
        default=None,
        help="Volume ADF to inspect. Omit to pick a volume and view first.",
    )
    parser.add_argument(
        "--view",
        default="default",
        help="canonical_views.yaml id, or default/home.",
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        help="Mesh every label as its own solid so unnamed colors can be identified.",
    )
    return parser.parse_args()


def _anatomy_label(path: pathlib.Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def main():
    global ADF
    args = _parse_args()
    view_id = args.view
    volume_name = ""
    close_all = bool(args.close_all)
    if args.anatomy_adf is None:
        choice = tuner_launch_dialog.pick_launch()
        if choice is None:
            return
        ADF = pathlib.Path(choice["adf"]).expanduser().resolve()
        view_id = choice.get("view") or "default"
        volume_name = choice.get("name") or ""
        close_all = close_all or bool(choice.get("close_all"))
    else:
        ADF = pathlib.Path(args.anatomy_adf).expanduser().resolve()
    if not ADF.is_file():
        raise SystemExit(f"Anatomy ADF not found: {ADF}")
    volume_name = volume_name or tuner_launch_dialog.volume_title(ADF)
    laterality = _adf_laterality(ADF)
    print(
        f"Tuner anatomy {volume_name}  {_anatomy_label(ADF)}  "
        f"view {view_id}  laterality {laterality or 'unknown'}"
        f"{'  close every structure' if close_all else ''}",
        flush=True,
    )
    print("Loading Newton", flush=True)
    _patch_open_usd()
    _quiet_stock_panels()
    import newton

    document = load_document()
    lights = document["lights"]
    irrigation = load_irrigation(IRRIGATION_PATH)
    irrigation["enabled"] = False
    global _START_BOUNCES
    _START_BOUNCES = int(irrigation.get("max_bounces", 4))
    camera = _startup_camera(view_id, laterality)
    position, right, up, forward = camera
    print("Opening RTX window", flush=True)
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
            # Same upscaling as the live window, so the two look alike.
            "omni:rtx:post:dlss:execMode": ("token", "performance"),
            **rtx_materials.bounce_settings(_START_BOUNCES),
            "omni:rtx:rt:fractionalOpacity": ("bool", True),
            **or_room.render_settings(lights),
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
        viewer, lights, camera, close_all=close_all
    )
    _bind_light(viewer, lights)
    volume_frame = VolumeFrame(
        np.asarray(translation, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64),
        np.asarray(frame.dimensions, dtype=np.float64),
        np.asarray(occupied.shape, dtype=np.float64),
    )
    or_room.author_or_lights(
        viewer.stage, volume_frame.index_to_world(np.asarray(occupied.shape, dtype=np.float64) * 0.5), lights
    )
    bench = IrrigationBench(occupied, volume_frame, down=screen_down(), gpu_film=True)
    bench.apply(irrigation)
    rtx_materials.apply_params(viewer, "water", _ensure_water(irrigation))
    rtx_materials.apply_params(viewer, "film", _ensure_film(irrigation))
    bench.prime(viewer)
    bench.hide(viewer)
    # The renderer copies the sheet on the first frame and then only moves
    # those vertices. Fill it first, while that mesh can still be replaced.
    if pose is not None and irrigation.get("enabled", False):
        quat = quat_aiming_at_bone(pose["forward"], pose["right"])
        for _ in range(45):
            bench.step(1.0 / 60.0, pose["center"], quat, size_mm=PREVIEW_SIZE_MM)
        bench.draw(viewer, splash=False)
        print(
            f"Tuner sheet {bench.film.n_tris} tris  {bench.film.volume_mm3:.2f} mm3  "
            f"hit {bench.hit_mm:.1f} mm",
            flush=True,
        )
    status = {"text": "Moves save as you drag."}
    camera_state = {
        "views": camera_views.load_views(),
        "selected": view_id if view_id not in ("default", "home") else (laterality or "right"),
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
        _anatomy_label(ADF),
    )
    if pose is not None and pose.get("bone") is not None:
        bone = np.asarray(pose["bone"], dtype=np.float64)
        viewer._scope_cam["pivot"] = bone.copy()
        viewer.camera.set_pivot(tuple(float(value) for value in bone))
    elif cortex.pivot is not None:
        viewer._scope_cam["pivot"] = np.asarray(cortex.pivot, dtype=np.float64).copy()
        viewer.camera.set_pivot(tuple(float(value) for value in cortex.pivot))
    shaders = markings.ShaderPanel(viewer, cortex)
    cortex.on_masks_changed = shaders.apply_pattern
    nerve_panel = nerves.NervePanel(viewer)
    shaders.others = [("Nerves", nerve_panel.draw_controls)]
    debris = load_debris()
    slurry = rtx_materials.slurry_params()
    paste = {
        "show": True,
        "visible": False,
        "bound": False,
        "radius_mm": float(debris["radius_mm"]),
        "clump_mm": float(debris["clump_mm"]),
        "roughness": float(slurry["specular_reflection_roughness"]),
        "coat": float(slurry["coat_weight"]),
        "color": [float(channel) for channel in slurry["diffuse_reflection_color"]],
        "look": slurry,
        "status": "Drag a slider. Save picture writes a png you can drop in chat.",
    }
    _register_panels(
        viewer,
        (
            ("Light", lambda imgui: _draw_panel(imgui, document, status)),
            ("Irrigation", lambda imgui: _draw_irrigation_panel(imgui, viewer, irrigation, status)),
            ("Paste", lambda imgui: _draw_paste_panel(imgui, viewer, paste)),
            ("Camera", lambda imgui: _draw_camera_panel(imgui, viewer, camera_state)),
            ("Structures", materials.draw),
            ("Cortex masks", cortex.draw),
            ("Shaders", shaders.draw),
        ),
    )
    print(
        f"Tuner ready. Views {camera_views.VIEWS_PATH.name}  light {SAVE_PATH.name}  "
        f"materials {rtx_materials.MATERIALS_PATH.name}  "
        f"cortex {cortex_masks.CONFIG_PATH.name}  shaders {markings.CONFIG_PATH.name} {nerves.CONFIG_PATH.name}  "
        f"irrigation {IRRIGATION_PATH.name}  paste {rtx_materials.PASTE_PATH.name}",
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
            rtx_materials.apply_params(viewer, "film", _ensure_film(irrigation))
            quat = quat_aiming_at_bone(pose["forward"], pose["right"])
            if irrigation.get("enabled", True):
                bench.step(dt, pose["center"], quat, size_mm=PREVIEW_SIZE_MM)
                bench.draw(viewer, splash=False)
            else:
                bench.hide(viewer)
        _draw_paste(viewer, pose, paste)
        viewer.end_frame()
        if viewer._window is not None:
            fps = (1.0 / dt) if dt > 1e-4 else 0.0
            state = "on" if irrigation.get("enabled", True) else "off"
            viewer._window.set_caption(
                f"FIVRS RTX Tuner  {fps:.0f} fps  irrigation {state}  "
                f"jet {float(irrigation['jet_mm']):.2f} mm  "
                f"film {float(irrigation.get('film_spread_mm', 20)):.1f} x "
                f"{float(irrigation.get('film_depth_mm', 3)):.2f} mm  "
                f"hit {bench.hit_mm:.1f} mm  sheet {bench.film.n_tris} tris"
            )
    viewer.close()


if __name__ == "__main__":
    main()
