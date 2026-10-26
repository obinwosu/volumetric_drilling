#!/usr/bin/env python3
"""FIVRS RTX Tuner.

A fixed view of the undrilled bone for setting the microscope light.
Material sliders can be added later in the same window; their values
already have a place in the saved file.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

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
    document = {"version": 1, "lights": dict(DEFAULT_LIGHTS), "materials": {}}
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
    document["materials"] = dict(saved.get("materials") or {})
    return document


def save_document(document: dict) -> None:
    payload = {
        "version": 1,
        "lights": document["lights"],
        "materials": document.get("materials") or {},
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


def _bone_groups():
    from fivrs_rtx_view import _load_occupancy
    from surface import BLOCK
    from warp_mesh import WarpMesher

    print("Loading bone slices", flush=True)
    occupied, colors = _load_occupancy(ADF)
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
    print(f"Bone mesh in {len(groups)} colors", flush=True)
    return groups


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


def _author_scene(viewer, lights):
    from pxr import Gf, UsdLux

    groups = _bone_groups()
    for key, (points, normals, tris) in groups.items():
        if sum(len(part) for part in tris) < 40:
            continue
        color = ((key & 255) / 255.0, ((key >> 8) & 255) / 255.0, ((key >> 16) & 255) / 255.0)
        viewer.log_mesh(
            f"bone_{key:06x}",
            wp.array(np.ascontiguousarray(np.concatenate(points), np.float32), dtype=wp.vec3),
            wp.array(np.ascontiguousarray(np.concatenate(tris).reshape(-1), np.int32), dtype=wp.int32),
            normals=wp.array(np.ascontiguousarray(np.concatenate(normals), np.float32), dtype=wp.vec3),
            color=color,
            roughness=0.55,
            metallic=0.0,
        )
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


def _bind_light(viewer, lights, camera):
    position, right, up, forward = camera
    original = viewer._update_ovrtx_camera

    def update():
        original()
        if viewer._rtx is None:
            return
        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -forward
        mat[3, :3] = position
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

    viewer._update_ovrtx_camera = update


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
    imgui.text_disabled("Materials")
    imgui.text_disabled("Omniverse material sliders will go here.")
    imgui.separator()
    if imgui.button("Save light") or changed:
        save_document(document)
        status["text"] = f"Saved {SAVE_PATH.name}"
    imgui.text_disabled(status["text"])
    imgui.text_disabled(str(SAVE_PATH))
    imgui.end()


def main():
    _patch_open_usd()
    _quiet_stock_panels()
    import newton

    document = load_document()
    lights = document["lights"]
    camera = _right_ear_camera()
    position, _right, _up, forward = camera
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
        },
    )
    viewer.set_model(newton.ModelBuilder().finalize())
    viewer.camera.fov = float(np.degrees(0.17951634837990105))
    viewer.camera.near = 0.1
    viewer.camera.far = 100.0
    viewer.camera.pos = type(viewer.camera.pos)(float(position[0]), float(position[1]), float(position[2]))
    viewer.camera._set_orientation_from_direction(forward)
    viewer.show_ui = True
    viewer.show_ground = False
    viewer.picking_enabled = False
    _author_scene(viewer, lights)
    _bind_light(viewer, lights, camera)
    status = {"text": "Moves save as you drag."}
    viewer.register_ui_callback(lambda imgui: _draw_panel(imgui, document, status), position="free")
    print(f"Tuner ready. Light file {SAVE_PATH}", flush=True)
    locked = False
    while viewer.is_running():
        viewer._camera_dirty = True
        viewer.begin_frame(0.0)
        viewer.end_frame()
        if not locked and viewer._window is not None:
            viewer._window.set_caption("FIVRS RTX Tuner")
            if viewer.gui is not None:
                viewer.gui.handle_mouse_drag = lambda *args, **kwargs: None
                viewer.gui.handle_mouse_scroll = lambda *args, **kwargs: None
                viewer.gui.update_camera_from_keys = lambda *args, **kwargs: None
            locked = True
    viewer.close()


if __name__ == "__main__":
    main()
