"""Stereo microscope eyes from the RTX window, for the headset.

The RTX window renders two cameras: the window's own camera becomes the left
eye, and a second camera is the right eye. Like a surgical microscope, the
eyes sit a stereo base apart (23 mm) and both aim at the field. Each frame
both images go to /dev/shm/fivrs_rtx_eyes, where the AMBF OpenXR eyepiece
plugin picks them up and shows them in the headset. The desktop window keeps
showing the left eye.

Shared memory layout (little endian), read by ambf_openxr_eyepiece.cpp:
    0   4  magic "FVEY"
    4   4  version (1)
    8   8  sequence: odd while writing, even when a frame is complete
    16  4  width
    20  4  height
    24  4  vertical field of view of each eye, radians
    28 36  reserved
    64     left image, then right image: RGBA8, bottom row first (OpenGL)
"""

from __future__ import annotations

import mmap
import os
import pathlib

import numpy as np

SHARE_PATH = pathlib.Path("/dev/shm/fivrs_rtx_eyes")
MAGIC = b"FVEY"
VERSION = 1
HEADER = 64
# A surgical microscope's two optical paths are about 22-25 mm apart.
BASE_MM = 23.0
MM_TO_SIM = 0.01007


def eye_matrices(position, right, up, forward, base_units: float, focus_units: float):
    """Left and right camera matrices (rows: right, up, -forward, position).

    Both eyes sit half the base to either side of the scope and aim at the
    point focus_units ahead, so the field centre has no disparity.
    """
    position = np.asarray(position, np.float64)
    right = np.asarray(right, np.float64)
    up = np.asarray(up, np.float64)
    forward = np.asarray(forward, np.float64)
    focus = position + forward * max(float(focus_units), 1e-3)
    out = []
    for side in (-0.5, 0.5):
        eye = position + right * (side * float(base_units))
        look = focus - eye
        look /= max(np.linalg.norm(look), 1e-12)
        eye_right = np.cross(look, up)
        eye_right /= max(np.linalg.norm(eye_right), 1e-12)
        eye_up = np.cross(eye_right, look)
        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = eye_right
        mat[1, :3] = eye_up
        mat[2, :3] = -look
        mat[3, :3] = eye
        out.append(mat)
    return out[0], out[1]


class EyeShare:
    """Writes the two eye images for the headset plugin."""

    def __init__(self, path=SHARE_PATH):
        self.path = pathlib.Path(path)
        self._map = None
        self._shape = None
        self._seq = 0
        self.frames = 0

    def _open(self, width: int, height: int):
        size = HEADER + 2 * width * height * 4
        if self._map is not None:
            self._map.close()
        # Write the new file aside and rename it, so a reader never maps half a resize.
        partial = self.path.with_suffix(".part")
        with open(partial, "wb") as handle:
            handle.truncate(size)
        os.replace(partial, self.path)
        with open(self.path, "r+b") as handle:
            self._map = mmap.mmap(handle.fileno(), size)
        header = np.zeros(HEADER, np.uint8)
        header[0:4] = np.frombuffer(MAGIC, np.uint8)
        header[4:8] = np.frombuffer(np.uint32(VERSION).tobytes(), np.uint8)
        header[16:20] = np.frombuffer(np.uint32(width).tobytes(), np.uint8)
        header[20:24] = np.frombuffer(np.uint32(height).tobytes(), np.uint8)
        self._map[0:HEADER] = header.tobytes()
        self._shape = (height, width)
        self._seq = 0

    def write(self, left: np.ndarray, right: np.ndarray, fov_y: float = 0.0) -> None:
        """left, right: (H, W, 4) uint8, top row first as the renderer returns them."""
        height, width = left.shape[:2]
        if right.shape[:2] != (height, width):
            return
        if self._shape != (height, width):
            self._open(width, height)
        m = self._map
        eye = width * height * 4
        self._seq += 1  # odd: writing
        m[8:16] = np.uint64(self._seq).tobytes()
        m[24:28] = np.float32(fov_y).tobytes()
        for k, image in enumerate((left, right)):
            flipped = np.ascontiguousarray(image[::-1, :, :4], dtype=np.uint8)
            start = HEADER + k * eye
            m[start : start + eye] = flipped.tobytes()
        self._seq += 1  # even: complete
        m[8:16] = np.uint64(self._seq).tobytes()
        self.frames += 1
        if self.frames % 30 == 1:
            # Writes through the map may not touch the file time; the plugin uses it
            # to tell a live file from one left over by an earlier session.
            try:
                os.utime(self.path)
            except OSError:
                pass

    def close(self):
        if self._map is not None:
            self._map.close()
            self._map = None


def read(path=SHARE_PATH):
    """(seq, left, right) as the plugin sees them, bottom row first. For tests."""
    data = pathlib.Path(path).read_bytes()
    if data[0:4] != MAGIC:
        return None
    seq = int(np.frombuffer(data[8:16], np.uint64)[0])
    width = int(np.frombuffer(data[16:20], np.uint32)[0])
    height = int(np.frombuffer(data[20:24], np.uint32)[0])
    eye = width * height * 4
    left = np.frombuffer(data[HEADER : HEADER + eye], np.uint8).reshape(height, width, 4)
    right = np.frombuffer(data[HEADER + eye : HEADER + 2 * eye], np.uint8).reshape(height, width, 4)
    return seq, left, right


# ---------------------------------------------------------------- ViewerRTX wiring

class _RendererWithEye:
    """The OVRTX renderer, with the right eye added to every step."""

    def __init__(self, renderer, eye_product: str):
        object.__setattr__(self, "_inner", renderer)
        object.__setattr__(self, "_eye_product", eye_product)

    def _with_eye(self, kwargs):
        products = kwargs.get("render_products")
        if products is not None:
            kwargs = dict(kwargs)
            kwargs["render_products"] = set(products) | {self._eye_product}
        return kwargs

    def step(self, *args, **kwargs):
        return self._inner.step(*args, **self._with_eye(kwargs))

    def step_async(self, *args, **kwargs):
        return self._inner.step_async(*args, **self._with_eye(kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


class Eyes:
    """Install on a ViewerRTX before set_model; call wrap_renderer after it."""

    def __init__(self, viewer, base_mm: float = BASE_MM, focus_mm: float | None = None):
        self.viewer = viewer
        self.base_units = float(base_mm) * MM_TO_SIM
        self.focus_units = None if focus_mm is None else float(focus_mm) * MM_TO_SIM
        self.share = EyeShare()
        # An eye file left by an earlier session would show its last frame.
        SHARE_PATH.unlink(missing_ok=True)
        self.eye_camera = None
        self.eye_product = None
        self.eye_var = None
        self.failed = False
        original = viewer._add_camera_lights_and_render_product

        def author():
            original()
            self._author_eye()

        viewer._add_camera_lights_and_render_product = author
        accept = viewer._accept_render

        def accept_render(products):
            if products is None:
                return accept(products)
            main = viewer._render_product_path
            shown = {name: product for name, product in products.items() if name == main} or products
            accept(shown)
            self._share(products)

        viewer._accept_render = accept_render

    def _author_eye(self):
        from pxr import Sdf, UsdGeom

        viewer = self.viewer
        stage = viewer.stage
        layer = stage.GetRootLayer()
        camera = viewer._camera_prim_path
        product = viewer._render_product_path
        var = viewer._render_var_path
        self.eye_camera = camera + "Right"
        self.eye_product = product + "Right"
        self.eye_var = str(Sdf.Path(var).GetParentPath()) + "Right/LdrColor"
        for path in (self.eye_camera, self.eye_product, self.eye_var):
            parent = Sdf.Path(path).GetParentPath()
            if not stage.GetPrimAtPath(parent):
                stage.DefinePrim(parent, "Scope")
        Sdf.CopySpec(layer, camera, layer, self.eye_camera)
        Sdf.CopySpec(layer, var, layer, self.eye_var)
        Sdf.CopySpec(layer, product, layer, self.eye_product)
        eye = stage.GetPrimAtPath(self.eye_product)
        eye.GetRelationship("camera").SetTargets([Sdf.Path(self.eye_camera)])
        eye.GetRelationship("orderedVars").SetTargets([Sdf.Path(self.eye_var)])
        settings = stage.GetPrimAtPath("/Render/OmniverseGlobalRenderSettings")
        if settings:
            rel = settings.GetRelationship("products")
            targets = list(rel.GetTargets())
            rel.SetTargets(targets + [Sdf.Path(self.eye_product)])
        UsdGeom.Camera(stage.GetPrimAtPath(self.eye_camera))
        print(f"Headset eyes: right eye {self.eye_product}, base {self.base_units / MM_TO_SIM:.0f} mm", flush=True)

    def wrap_renderer(self):
        rtx = getattr(self.viewer, "_rtx", None)
        if rtx is not None and not isinstance(rtx, _RendererWithEye) and self.eye_product:
            self.viewer._rtx = _RendererWithEye(rtx, self.eye_product)

    def write_cameras(self, packed, distance_units: float):
        """Left eye on the window camera, right eye on its own camera."""
        position, right, up, forward = packed
        focus = self.focus_units if self.focus_units is not None else distance_units
        left_mat, right_mat = eye_matrices(position, right, up, forward, self.base_units, focus)
        viewer = self.viewer
        for path, mat in ((viewer._camera_prim_path, left_mat), (self.eye_camera, right_mat)):
            viewer._write_runtime_attribute(
                [path], "omni:xform", np.ascontiguousarray(mat, np.float64)[np.newaxis, ...], is_matrix=True
            )

    def write_focal_length(self, focal: float):
        if self.eye_camera:
            self.viewer._write_runtime_attribute([self.eye_camera], "focalLength", np.array([focal], np.float32))

    def _share(self, products):
        from ovrtx import Device

        images = {}
        for name, product in products.items():
            for frame in product.frames:
                for var in frame.render_vars.values():
                    with var.map(device=Device.CPU) as mapping:
                        images[name] = np.array(np.from_dlpack(mapping), copy=True)
                    break
                break
        left = images.get(self.viewer._render_product_path)
        right = images.get(self.eye_product)
        if left is None or right is None:
            if not self.failed:
                print(f"Headset eyes: missing a render ({sorted(images)})", flush=True)
                self.failed = True
            return
        fov = float(np.radians(getattr(self.viewer.camera, "fov", 10.29)))
        self.share.write(left, right, fov)
