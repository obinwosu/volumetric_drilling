"""Thin bone over a structure takes on that structure's colour.

Each bone surface vertex carries (nerve_mm, dura_mm, otic_mm, valid): how
much bone is left between it and the nearest facial nerve / chorda voxel,
sinus_dura voxel, and inner-ear voxel (FAR_VALUE past ``FAR_MM``). The distances
come from KD-trees of each structure's voxels that touch bone, built once
from the intact volume. The structures never move, so drilling only needs
the remeshed blocks' new vertices looked up, on the surface worker thread.

``mdl/ThinBone.mdl`` passes every bone input from the Structures panel
straight through to OmniSurfaceBase, then tints toward the structure colour
as the bone thins, and toward yellow near the inner ear (otic capsule).
This is RTX only: AMBF voxels and haptics are unchanged.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from scipy.spatial import cKDTree

import rtx_materials

HERE = pathlib.Path(__file__).resolve().parent
MDL_PATH = HERE / "mdl" / "ThinBone.mdl"
CONFIG_PATH = HERE / "tuner_thin_bone.json"
MM_TO_SIM = 0.01007
PRIMVAR = "thin_bone"
# Search radius. Anything farther reads as FAR_VALUE, which no depth or
# preview reaches, so the tint never ends on a hard edge: keep the panel's
# preview + depth maxima below FAR_MM.
FAR_MM = 6.0
FAR_VALUE = 99.0
GROUPS = (
    ("nerve", ("facial_nerve", "chorda_tympani")),
    ("dura", ("sinus_dura",)),
    ("otic", ("inner_ear", "scala_tympani", "scala_vestibuli", "labyrinth")),
)

DEFAULTS = {
    "thin_enabled": True,
    "nerve_color": [0.93, 0.52, 0.56],
    "nerve_depth_mm": 1.0,
    "nerve_strength": 0.55,
    "dura_color": [0.55, 0.68, 0.92],
    "dura_depth_mm": 1.0,
    "dura_strength": 0.55,
    "otic_color": [1.0, 0.92, 0.70],
    "otic_band_mm": 2.0,
    "otic_strength": 0.6,
    "plate_gloss": 0.3,
}
BOOL_KEYS = ("thin_enabled",)
# Tuner-only: pretends the bone is this much thinner, so intact bone can be tuned.
PREVIEW_KEY = "preview_thinning_mm"


def load_parameters() -> dict:
    params = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            stored = json.loads(CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            stored = {}
        for key, value in stored.get("parameters", stored).items():
            if key in BOOL_KEYS:
                params[key] = bool(value)
            elif key in params:
                params[key] = value
    return params


def save_parameters(params: dict):
    payload = {"version": 1, "parameters": {key: params[key] for key in DEFAULTS}}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n")


# ------------------------------------------------------------------ distances


def _hex_rgb(hex_id: str) -> tuple[int, int, int]:
    return tuple(int(hex_id[offset : offset + 2], 16) for offset in (0, 2, 4))


def _label_mask(occupied, colors, rgb, box=None):
    if box is not None:
        occupied = occupied[box]
        colors = colors[box]
    return (
        np.asarray(occupied, dtype=bool)
        & (colors[..., 0] == rgb[0])
        & (colors[..., 1] == rgb[1])
        & (colors[..., 2] == rgb[2])
    )


def _bounds(mask):
    """Index box of a mask, as (lo, hi) per axis, hi exclusive."""
    box = []
    for axis in range(3):
        other = tuple(index for index in range(3) if index != axis)
        hits = np.flatnonzero(np.any(mask, axis=other))
        if len(hits) == 0:
            return None
        box.append((int(hits[0]), int(hits[-1]) + 1))
    return box


def _touching(mask, bone):
    """Mask voxels with a bone voxel as a face neighbour."""
    out = np.zeros_like(mask)
    out[1:] |= bone[:-1]
    out[:-1] |= bone[1:]
    out[:, 1:] |= bone[:, :-1]
    out[:, :-1] |= bone[:, 1:]
    out[:, :, 1:] |= bone[:, :, :-1]
    out[:, :, :-1] |= bone[:, :, 1:]
    return mask & out


class ThinBoneIndex:
    """Per-group KD-trees in millimetres. ``sample`` takes voxel-index points."""

    def __init__(self, trees: dict, mm_per_voxel, bone_packed: int, counts: dict):
        self.trees = trees
        self.mm_per_voxel = np.asarray(mm_per_voxel, np.float64)
        self.half_voxel_mm = 0.5 * float(np.mean(self.mm_per_voxel))
        self.bone_packed = int(bone_packed)
        self.counts = counts

    def sample(self, index_pts, preview_mm: float = 0.0) -> np.ndarray:
        """Float4 per point: remaining bone (mm) to nerve, dura, otic; valid."""
        points = np.asarray(index_pts, np.float64).reshape(-1, 3) * self.mm_per_voxel
        out = np.full((len(points), 4), FAR_VALUE, np.float32)
        out[:, 3] = 1.0
        if len(points) == 0:
            return out
        for column, (group, _names) in enumerate(GROUPS):
            tree = self.trees.get(group)
            if tree is None:
                continue
            distance, _index = tree.query(points, k=1, distance_upper_bound=FAR_MM, workers=-1)
            # Vertices sit half a voxel off the structure's voxel centres.
            left = np.clip(distance - self.half_voxel_mm - float(preview_mm), 0.0, None)
            out[:, column] = np.where(np.isfinite(distance), left, FAR_VALUE).astype(np.float32)
        return out


def build_index(occupied, colors, document, mm_per_voxel, bone_rgb) -> ThinBoneIndex:
    """KD-trees of each group's voxels that touch bone, from the intact volume.

    Each label is cropped to its own box before the neighbour test, so the
    big volume is only scanned once per label.
    """
    started = time.perf_counter()
    names = rtx_materials.name_map(document)
    bone_rgb = tuple(int(channel) for channel in bone_rgb)
    mm = np.asarray(mm_per_voxel, np.float64)
    shape = np.asarray(occupied).shape
    trees, counts = {}, {}
    for group, wanted in GROUPS:
        found = []
        for hex_id, alias in names.items():
            if alias not in wanted:
                continue
            mask = _label_mask(occupied, colors, _hex_rgb(hex_id))
            box = _bounds(mask)
            if box is not None:
                found.append((mask, box))
        pieces = []
        for mask, box in found:
            grown = tuple(slice(max(lo - 1, 0), min(hi + 1, shape[axis])) for axis, (lo, hi) in enumerate(box))
            bone = _label_mask(occupied, colors, bone_rgb, grown)
            touch = _touching(mask[grown], bone)
            offset = np.array([part.start for part in grown], np.float64)
            pieces.append(np.argwhere(touch).astype(np.float64) + offset)
            del mask
        points = np.concatenate(pieces) if pieces else np.zeros((0, 3))
        counts[group] = int(len(points))
        if len(points):
            trees[group] = cKDTree(points * mm)
    bone_packed = bone_rgb[0] | (bone_rgb[1] << 8) | (bone_rgb[2] << 16)
    index = ThinBoneIndex(trees, mm, bone_packed, counts)
    summary = ", ".join(f"{group} {counts[group]}" for group, _names in GROUPS)
    print(f"Thin bone index ({summary} voxels touching bone) in {time.perf_counter() - started:.1f} s", flush=True)
    return index


# ------------------------------------------------- numpy mirror of ThinBone.mdl


def _smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def weights(values, params: dict, preview_mm: float = 0.0):
    """Nerve, dura, otic weights per point, as ThinBone.mdl computes them."""
    v = np.asarray(values, np.float64).reshape(-1, 4)
    on = (1.0 if params["thin_enabled"] else 0.0) * np.clip(v[:, 3], 0.0, 1.0)

    def fade(column, depth, strength):
        left = np.maximum(v[:, column] - preview_mm, 0.0)
        return float(strength) * (1.0 - _smoothstep(0.0, max(float(depth), 1e-3), left)) * on

    nerve = fade(0, params["nerve_depth_mm"], params["nerve_strength"])
    dura = fade(1, params["dura_depth_mm"], params["dura_strength"])
    otic = fade(2, params["otic_band_mm"], params["otic_strength"])
    return nerve, dura, otic


def tinted_color(values, params: dict, bone_color, preview_mm: float = 0.0) -> np.ndarray:
    """Diffuse colour per point, as ThinBone.mdl mixes it."""
    nerve, dura, otic = weights(values, params, preview_mm)
    bone = np.asarray(bone_color, np.float64)
    otic_tint = np.asarray(params["otic_color"], np.float64)
    hued = bone[None, :] * (1.0 - otic[:, None] + otic_tint[None, :] * otic[:, None])
    show = np.clip(nerve + dura, 0.0, 1.0)
    total = np.maximum(nerve + dura, 1e-4)
    under = (
        np.asarray(params["nerve_color"], np.float64)[None, :] * nerve[:, None]
        + np.asarray(params["dura_color"], np.float64)[None, :] * dura[:, None]
    ) / total[:, None]
    return hued * (1.0 - show[:, None]) + under * show[:, None]


# ------------------------------------------------------------ USD and runtime


def bone_hex(document) -> str:
    return rtx_materials.hex_of(document or {}, "bone") or "fff9db"


def _bone_inputs(viewer) -> dict:
    """The bone segment's OmniSurface inputs, as the Structures panel set them."""
    document = getattr(viewer, "_material_document", None) or {}
    segment = bone_hex(document)
    params = getattr(viewer, "_material_params", {}).get(segment)
    if params is None:
        # The live window never binds the plain bone material, so read the
        # saved document. It is replaced (not edited) when the file reloads.
        cached = getattr(viewer, "_thin_bone_cache", None)
        if cached is not None and cached[0] is document:
            return cached[1]
        resolved = rtx_materials.resolve(document, segment)
        params = {}
        for _group, key, _label, _kind, _low, _high in rtx_materials.FIELDS:
            if key in resolved:
                params[key] = resolved[key]
        if rtx_materials.GLINT_KEY in resolved:
            params[rtx_materials.GLINT_KEY] = float(resolved[rtx_materials.GLINT_KEY])
        inputs = rtx_materials._shader_inputs(params)
        viewer._thin_bone_cache = (document, inputs)
        return inputs
    return rtx_materials._shader_inputs(params)


def shader_inputs(params: dict, bone_inputs: dict, preview_mm: float = 0.0) -> dict:
    inputs = dict(bone_inputs)
    for key in DEFAULTS:
        inputs[key] = params[key]
    inputs[PREVIEW_KEY] = float(preview_mm)
    return inputs


def ensure_shader(viewer):
    from pxr import Sdf, UsdShade

    path = f"{viewer._root_path}/Looks/thin_bone_shader"
    stage = viewer.stage
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    prim = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(MDL_PATH)))
    prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("ThinBone")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for name in ("surface", "displacement", "volume"):
        material.CreateOutput(f"mdl:{name}", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.ConnectableAPI(), "out"
        )
    return material, shader


def push_look(viewer, params: dict, preview_mm: float = 0.0):
    """Author the inputs. The camera update sends them to the renderer."""
    _material, shader = ensure_shader(viewer)
    bone = _bone_inputs(viewer)
    stored = shader_inputs(params, bone, preview_mm)
    for key, value in stored.items():
        rtx_materials._author_input(shader, key, value)
    viewer._thin_inputs = stored
    viewer._thin_bone_seen = bone
    viewer._thin_look_dirty = True


def bind_shader(viewer, prim):
    from pxr import UsdShade

    material, _shader = ensure_shader(viewer)
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def write_values(viewer, prim, values: np.ndarray, name: str):
    from pxr import Gf, Sdf, UsdGeom, Vt

    data = np.ascontiguousarray(values, np.float32).reshape(-1, 4)
    api = UsdGeom.PrimvarsAPI(UsdGeom.Mesh(prim))
    primvar = api.GetPrimvar(PRIMVAR)
    if not primvar or primvar.GetTypeName() != Sdf.ValueTypeNames.Float4Array:
        primvar = api.CreatePrimvar(PRIMVAR, Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex)
    try:
        primvar.Set(Vt.Vec4fArray.FromNumpy(data))
    except Exception:
        primvar.Set([Gf.Vec4f(*(float(value) for value in row)) for row in data])
    pending = getattr(viewer, "_thin_pending", None)
    if pending is None:
        pending = {}
        viewer._thin_pending = pending
    pending[name] = data


def attach(viewer, prim, name: str, values: np.ndarray):
    """Bind the thin-bone shader to one bone mesh and give it its values."""
    write_values(viewer, prim, values, name)
    bind_shader(viewer, prim)


def attach_subset(viewer, prim, name: str, tris, face_mask, hidden: bool = False):
    """Tuner: the bone prim is a face subset of the intact surface cache.

    ``viewer._thin_cache_index`` holds that cache's vertices in voxel index
    space; the subset keeps them in ``np.unique`` order, as subset_mesh does.
    """
    index = getattr(viewer, "_thin_index", None)
    verts = getattr(viewer, "_thin_cache_index", None)
    if index is None or verts is None:
        return
    if hidden:
        values = np.full((3, 4), FAR_VALUE, np.float32)
        values[:, 3] = 0.0
    else:
        used = np.unique(np.asarray(tris)[np.asarray(face_mask, dtype=bool)].reshape(-1))
        values = index.sample(np.asarray(verts)[used])
    attach(viewer, prim, name, values)


def flush_runtime(viewer):
    """Run from the camera update, where the other tuners write."""
    if getattr(viewer, "_rtx", None) is None:
        return
    # A Structures-panel edit to bone reaches the thin-bone shader too.
    if getattr(viewer, "_thin_inputs", None) is not None:
        bone = _bone_inputs(viewer)
        if bone != getattr(viewer, "_thin_bone_seen", None):
            stored = dict(viewer._thin_inputs)
            stored.update(bone)
            _material, shader = ensure_shader(viewer)
            for key, value in bone.items():
                rtx_materials._author_input(shader, key, value)
            viewer._thin_inputs = stored
            viewer._thin_bone_seen = bone
            viewer._thin_look_dirty = True
    if getattr(viewer, "_thin_look_dirty", False):
        _material, shader = ensure_shader(viewer)
        failed = False
        for key, value in getattr(viewer, "_thin_inputs", {}).items():
            rtx_materials._author_input(shader, key, value)
            if not rtx_materials._runtime_input(viewer, shader, key, value):
                failed = True
        if not failed:
            viewer._thin_look_dirty = False
    pending = getattr(viewer, "_thin_pending", None) or {}
    for name, data in list(pending.items()):
        path = viewer._mesh_prim_paths.get(viewer._qualify(name))
        if path is None:
            continue
        try:
            viewer._write_runtime_array_attribute(path, f"primvars:{PRIMVAR}", data)
        except Exception as exc:
            if not getattr(viewer, "_thin_primvar_warned", False):
                print(f"thin bone value write failed: {exc}", flush=True)
                viewer._thin_primvar_warned = True
            return
        del pending[name]


def sync_live(viewer):
    """Live window: pick up tuner_thin_bone.json when the tuner saves it."""
    if getattr(viewer, "_thin_index", None) is None:
        return
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if mtime != getattr(viewer, "_thin_mtime", "unset"):
        viewer._thin_mtime = mtime
        push_look(viewer, load_parameters())
    flush_runtime(viewer)


# ---------------------------------------------------------------- tuner panel


class ThinBonePanel:
    """Thin-bone controls. Drawn inside the Shaders window."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.params = load_parameters()
        self.preview_mm = 0.0
        self.status = "Looks save as you drag."
        push_look(viewer, self.params, self.preview_mm)

    def _save(self):
        save_parameters(self.params)
        self.status = f"Saved {CONFIG_PATH.name}"

    def _toggle(self, imgui, key, label) -> bool:
        changed, value = imgui.checkbox(label, bool(self.params[key]))
        if changed:
            self.params[key] = bool(value)
        return bool(changed)

    def _slider(self, imgui, key, label, low, high, fmt) -> bool:
        changed, value = imgui.slider_float(label, float(self.params[key]), float(low), float(high), fmt)
        if changed:
            self.params[key] = float(value)
        return bool(changed)

    def _color(self, imgui, key, label) -> bool:
        changed, value = imgui.color_edit3(label, [float(channel) for channel in self.params[key]])
        if changed:
            self.params[key] = [float(channel) for channel in value]
        return bool(changed)

    def draw_controls(self, imgui):
        imgui.text("Thin bone")
        imgui.text_disabled("Bone takes the colour of what lies under it as it thins.")
        if getattr(self.viewer, "_thin_index", None) is None:
            imgui.text_wrapped("No bone surface to shade in this view.")
        changed = False
        changed |= self._toggle(imgui, "thin_enabled", "Enabled")
        preview_changed, preview = imgui.slider_float(
            "Preview thinning (tuner only)", float(self.preview_mm), 0.0, 2.5, "%.2f mm"
        )
        if preview_changed:
            self.preview_mm = float(preview)
        imgui.text_disabled("Pretends the bone is this much thinner. Not saved.")
        imgui.separator()
        imgui.text("Over the facial nerve and chorda")
        changed |= self._color(imgui, "nerve_color", "Nerve show-through")
        changed |= self._slider(imgui, "nerve_depth_mm", "Nerve depth", 0.1, 2.5, "%.2f mm")
        changed |= self._slider(imgui, "nerve_strength", "Nerve strength", 0.0, 1.0, "%.2f")
        imgui.text("Over sinus and dura")
        changed |= self._color(imgui, "dura_color", "Dura show-through")
        changed |= self._slider(imgui, "dura_depth_mm", "Dura depth", 0.1, 2.5, "%.2f mm")
        changed |= self._slider(imgui, "dura_strength", "Dura strength", 0.0, 1.0, "%.2f")
        imgui.text("Otic capsule")
        changed |= self._color(imgui, "otic_color", "Otic hue")
        changed |= self._slider(imgui, "otic_band_mm", "Otic band", 0.2, 3.5, "%.2f mm")
        changed |= self._slider(imgui, "otic_strength", "Otic strength", 0.0, 1.0, "%.2f")
        imgui.separator()
        changed |= self._slider(imgui, "plate_gloss", "Thin plate gloss", 0.0, 1.0, "%.2f")
        if changed:
            push_look(self.viewer, self.params, self.preview_mm)
            self._save()
        elif preview_changed:
            push_look(self.viewer, self.params, self.preview_mm)
        imgui.separator()
        imgui.text_disabled(self.status)
        imgui.text_disabled(str(CONFIG_PATH))
