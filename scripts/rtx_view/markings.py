"""One markings shader for the cortical cap.

Char, periosteum, and blood are three channels on the mask-3 surface. The
shader mixes the wet-bone, char, periosteum, and blood looks. Pattern sliders
rewrite the channels. Look sliders rewrite the shader inputs.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.spatial import cKDTree

import rtx_materials

HERE = pathlib.Path(__file__).resolve().parent
MDL_PATH = HERE / "mdl" / "Markings.mdl"
CONFIG_PATH = HERE / "tuner_markings.json"
# Same millimetre scale as cortex_masks.MM_TO_SIM.
MM_TO_SIM = 0.01007
SEGMENT = "cortex_markings"
PRIMVAR = "markings"

_IVORY = [1.0, 249 / 255, 219 / 255]

DEFAULTS = {
    "seed": 7.0,
    "char_coverage": 0.32,
    "char_scale_mm": 4.0,
    "char_margin_mm": 8.0,
    "periosteum_width_mm": 2.2,
    "blood_film": 0.18,
    "blood_pooling": 1.0,
    "edge_fade_mm": 15.0,
    "char_strength": 1.0,
    "periosteum_strength": 1.0,
    "blood_strength": 1.0,
    "char_enabled": False,
    "periosteum_enabled": False,
    "blood_enabled": False,
    "bone_color": list(_IVORY),
    "char_color": [0.10, 0.055, 0.03],
    "periosteum_color": [0.55, 0.16, 0.13],
    "blood_color": [0.40, 0.015, 0.012],
    "bone_roughness": 0.12,
    "char_roughness": 0.92,
    "periosteum_roughness": 0.62,
    "blood_roughness": 0.04,
    "bone_coat": 0.22,
    "blood_coat": 0.85,
    "bone_coat_roughness": 0.05,
    "blood_coat_roughness": 0.03,
    "coat_ior": 1.45,
}

LOOK_KEYS = (
    "bone_color",
    "char_color",
    "periosteum_color",
    "blood_color",
    "bone_roughness",
    "char_roughness",
    "periosteum_roughness",
    "blood_roughness",
    "bone_coat",
    "blood_coat",
    "bone_coat_roughness",
    "blood_coat_roughness",
    "coat_ior",
    "char_strength",
    "periosteum_strength",
    "blood_strength",
    "char_enabled",
    "periosteum_enabled",
    "blood_enabled",
)
BOOL_KEYS = ("char_enabled", "periosteum_enabled", "blood_enabled")
PATTERN_KEYS = (
    "seed",
    "char_coverage",
    "char_scale_mm",
    "char_margin_mm",
    "periosteum_width_mm",
    "blood_film",
    "blood_pooling",
    "edge_fade_mm",
)


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


def _cell_hash(ix, iy, iz, seed: int) -> np.ndarray:
    n = (
        np.asarray(ix, np.int64) * np.int64(374761393)
        + np.asarray(iy, np.int64) * np.int64(668265263)
        + np.asarray(iz, np.int64) * np.int64(2147483647)
        + np.int64(seed) * np.int64(1274126177)
    )
    n = (n ^ (n >> 13)) * np.int64(1274126177)
    n = n ^ (n >> 16)
    return (n & np.int64(0x7FFFFFFF)) / np.float64(0x7FFFFFFF)


def value_noise(points_mm: np.ndarray, scale_mm: float, seed: int) -> np.ndarray:
    scale = max(float(scale_mm), 0.25)
    p = np.asarray(points_mm, np.float64) / scale
    base = np.floor(p).astype(np.int64)
    frac = p - base
    fade = frac * frac * (3.0 - 2.0 * frac)
    def corner(dx, dy, dz):
        return _cell_hash(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, seed)
    x00 = corner(0, 0, 0) * (1.0 - fade[:, 0]) + corner(1, 0, 0) * fade[:, 0]
    x10 = corner(0, 1, 0) * (1.0 - fade[:, 0]) + corner(1, 1, 0) * fade[:, 0]
    x01 = corner(0, 0, 1) * (1.0 - fade[:, 0]) + corner(1, 0, 1) * fade[:, 0]
    x11 = corner(0, 1, 1) * (1.0 - fade[:, 0]) + corner(1, 1, 1) * fade[:, 0]
    y0 = x00 * (1.0 - fade[:, 1]) + x10 * fade[:, 1]
    y1 = x01 * (1.0 - fade[:, 1]) + x11 * fade[:, 1]
    return y0 * (1.0 - fade[:, 2]) + y1 * fade[:, 2]


def fbm(points_mm: np.ndarray, scale_mm: float, seed: int) -> np.ndarray:
    coarse = value_noise(points_mm, scale_mm, seed)
    fine = value_noise(points_mm, scale_mm * 0.45, seed + 19)
    return (coarse + 0.5 * fine) / 1.5


def concavity(points_mm: np.ndarray, normals: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Positive where neighbors rise along the outward normal: a pit that holds blood."""
    count = len(points_mm)
    acc = np.zeros(count, np.float64)
    hits = np.zeros(count, np.float64)
    if len(tris) == 0:
        return acc
    pairs = ((0, 1), (1, 2), (2, 0))
    for a, b in pairs:
        ia = tris[:, a]
        ib = tris[:, b]
        for src, dst in ((ia, ib), (ib, ia)):
            delta = points_mm[dst] - points_mm[src]
            length = np.linalg.norm(delta, axis=1)
            length[length < 1e-6] = 1.0
            slope = np.einsum("ij,ij->i", normals[src], delta) / length
            np.add.at(acc, src, slope)
            np.add.at(hits, src, 1.0)
    hits[hits < 1.0] = 1.0
    return np.clip(acc / hits, 0.0, None)


def rim_distance_mm(world, tris, mask, used) -> np.ndarray:
    """Euclidean distance, in mm, from each subset vertex to the mask edge."""
    used = np.asarray(used)
    if not np.any(mask):
        return np.zeros(len(used), np.float64)
    outside = tris[~mask]
    if len(outside) == 0:
        return np.full(len(used), 1.0e6, np.float64)
    rim = np.intersect1d(used, np.unique(outside.reshape(-1)))
    points_mm = np.asarray(world[used], np.float64) / MM_TO_SIM
    if len(rim) == 0:
        return np.full(len(used), 1.0e6, np.float64)
    rim_mm = np.asarray(world[rim], np.float64) / MM_TO_SIM
    distance, _index = cKDTree(rim_mm).query(points_mm, k=1, workers=-1)
    return np.asarray(distance, np.float64)


def compose_channels(points_mm, normals, tris, rim_mm, params: dict) -> np.ndarray:
    """Float3 channels, char / periosteum / blood, one per vertex."""
    count = len(points_mm)
    channels = np.zeros((count, 3), np.float32)
    if count == 0:
        return channels
    seed = int(round(float(params["seed"])))
    scale = float(params["char_scale_mm"])
    coverage = float(np.clip(params["char_coverage"], 0.0, 1.0))
    margin = float(params.get("char_margin_mm", 8.0))
    field = fbm(points_mm, scale, seed)
    rim = np.asarray(rim_mm, np.float64)
    if margin <= 0.05:
        band = np.zeros(count, np.float64)
    else:
        toward_edge = np.clip(1.0 - rim / margin, 0.0, 1.0)
        band = toward_edge * toward_edge * (3.0 - 2.0 * toward_edge)
    in_margin = band > 0.02
    if coverage <= 0.0 or not np.any(in_margin):
        char = np.zeros(count, np.float64)
    else:
        threshold = float(np.quantile(field[in_margin], 1.0 - coverage))
        char = np.clip((field - threshold) / 0.08, 0.0, 1.0) * band
    width = float(params["periosteum_width_mm"])
    if width <= 0.05:
        edge = np.zeros(count, np.float64)
    else:
        edge = np.exp(-np.asarray(rim_mm, np.float64) / width)
    patch = np.clip((fbm(points_mm, max(width * 3.0, 2.0), seed + 17) - 0.62) / 0.18, 0.0, 1.0)
    peri = np.clip(edge + 0.35 * patch * (1.0 - 0.5 * edge), 0.0, 1.0)
    film_noise = 0.25 + 0.75 * fbm(points_mm, max(scale * 0.7, 1.5), seed + 101)
    film = float(np.clip(params["blood_film"], 0.0, 1.0)) * film_noise
    pits = np.clip(concavity(points_mm, normals, tris) / 0.08, 0.0, 1.0)
    pits *= 0.4 + 0.6 * fbm(points_mm, max(scale * 1.3, 2.0), seed + 53)
    pool = float(max(params["blood_pooling"], 0.0)) * pits
    blood = np.clip(film + pool * (1.0 - 0.5 * film), 0.0, 1.0)
    channels[:, 0] = char.astype(np.float32)
    channels[:, 1] = peri.astype(np.float32)
    channels[:, 2] = blood.astype(np.float32)
    return channels


def edge_weight(rim_mm, fade_mm: float) -> np.ndarray:
    """0 on the markings perimeter, 1 once `fade_mm` inside it."""
    width = float(fade_mm)
    rim = np.asarray(rim_mm, np.float64)
    if width <= 0.05:
        return np.ones(len(rim), np.float32)
    t = np.clip(rim / width, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def with_fade(channels: np.ndarray, rim_mm, fade_mm: float) -> np.ndarray:
    """Char, periosteum, blood, and the opacity of the markings sheet."""
    weight = edge_weight(rim_mm, fade_mm)
    faded = np.zeros((len(channels), 4), np.float32)
    if len(channels) == 0:
        return faded
    faded[:, :3] = np.asarray(channels[:, :3], np.float32) * weight[:, None]
    faded[:, 3] = weight
    return faded


def subset_channels(world, normals, tris, mask, params: dict):
    """Subset mesh plus channels. Vertex order matches the logged mesh."""
    faces = np.asarray(tris)[np.asarray(mask, dtype=bool)]
    if len(faces) == 0:
        empty = np.zeros((0, 3), np.float32)
        return empty, np.zeros((0, 3), np.int32), np.zeros((0, 4), np.float32)
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    points = np.ascontiguousarray(world[used], np.float32)
    norms = np.ascontiguousarray(normals[used], np.float32)
    local = inverse.reshape(-1, 3).astype(np.int32)
    points_mm = np.asarray(points, np.float64) / MM_TO_SIM
    rim = rim_distance_mm(world, tris, mask, used)
    channels = compose_channels(points_mm, norms, local, rim, params)
    return points, local, with_fade(channels, rim, params.get("edge_fade_mm", 15.0))


def field_on_mask(index_verts, tris, mask, mm_per_axis, params: dict):
    """Markings channels for one mask. Positions stay in voxel index space.

    Returns the vertex positions, the float4 channels, and the arrays needed
    to rebuild those channels when the tuner file changes.
    """
    from surface import vertex_normals

    mask = np.asarray(mask, dtype=bool)
    faces = np.asarray(tris)[mask]
    empty_pts = np.zeros((0, 3), np.float64)
    empty_ch = np.zeros((0, 4), np.float32)
    if len(faces) == 0:
        return empty_pts, empty_ch, None
    index_verts = np.asarray(index_verts, np.float64)
    world = np.ascontiguousarray(index_verts * np.asarray(mm_per_axis, np.float64) * MM_TO_SIM, np.float32)
    normals = vertex_normals(np.asarray(world, np.float64), np.asarray(tris)).astype(np.float32)
    _points, local, _channels = subset_channels(world, normals, tris, mask, params)
    used = np.unique(faces.reshape(-1))
    if len(used) != len(_channels):
        raise RuntimeError("markings channel count does not match the mask vertices")
    points_mm = np.asarray(world[used], np.float64) / MM_TO_SIM
    rim = rim_distance_mm(world, tris, mask, used)
    bake = (points_mm, np.asarray(normals[used], np.float32), local, rim)
    return index_verts[used], rebake(bake, params), bake


def rebake(bake, params: dict) -> np.ndarray:
    if bake is None:
        return np.zeros((0, 4), np.float32)
    points_mm, normals, local, rim = bake
    raw = compose_channels(points_mm, normals, local, rim, params)
    return with_fade(raw, rim, params.get("edge_fade_mm", 15.0))


def pattern_differs(previous, params: dict) -> bool:
    if previous is None:
        return True
    return any(previous.get(key) != params.get(key) for key in PATTERN_KEYS)


def channel_stats(channels: np.ndarray) -> str:
    if len(channels) == 0:
        return "No markings surface."
    present = channels > 0.35
    char = 100.0 * float(present[:, 0].mean())
    peri = 100.0 * float(present[:, 1].mean())
    blood = 100.0 * float(present[:, 2].mean())
    return f"On the cap: char {char:.0f}%, periosteum {peri:.0f}%, blood {blood:.0f}%"


def ensure_shader(viewer):
    from pxr import Sdf, UsdShade

    path = f"{viewer._root_path}/Looks/markings_shader"
    stage = viewer.stage
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    prim = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(MDL_PATH)))
    prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("Markings")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for name in ("surface", "displacement", "volume"):
        material.CreateOutput(f"mdl:{name}", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.ConnectableAPI(), "out"
        )
    return material, shader


def push_look(viewer, params: dict):
    """Author the looks. The camera update sends them to the renderer."""
    _material, shader = ensure_shader(viewer)
    stored = {}
    for key in LOOK_KEYS:
        rtx_materials._author_input(shader, key, params[key])
        stored[key] = params[key]
    viewer._markings_params = stored
    viewer._markings_look_dirty = True


def write_channels(viewer, prim, channels: np.ndarray, name: str | None = None):
    from pxr import Gf, Sdf, UsdGeom, Vt

    data = np.ascontiguousarray(channels, np.float32).reshape(-1, 4)
    mesh = UsdGeom.Mesh(prim)
    api = UsdGeom.PrimvarsAPI(mesh)
    primvar = api.GetPrimvar(PRIMVAR)
    if not primvar or primvar.GetTypeName() != Sdf.ValueTypeNames.Float4Array:
        primvar = api.CreatePrimvar(PRIMVAR, Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex)
    try:
        primvar.Set(Vt.Vec4fArray.FromNumpy(data))
    except Exception:
        primvar.Set([Gf.Vec4f(float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in data])
    pending = getattr(viewer, "_markings_pending_channels", None)
    if pending is None:
        pending = {}
        viewer._markings_pending_channels = pending
    pending[name or SEGMENT] = data
    viewer._markings_channels_dirty = True


def flush_runtime(viewer):
    """Run from the camera update, which is when the other tuners write."""
    if getattr(viewer, "_rtx", None) is None:
        return
    if getattr(viewer, "_markings_look_dirty", False):
        _material, shader = ensure_shader(viewer)
        for key, value in getattr(viewer, "_markings_params", {}).items():
            rtx_materials._author_input(shader, key, value)
            rtx_materials._runtime_input(viewer, shader, key, value)
        viewer._markings_look_dirty = False
    if not getattr(viewer, "_markings_channels_dirty", False):
        return
    pending = getattr(viewer, "_markings_pending_channels", None) or {}
    for name, data in list(pending.items()):
        path = viewer._mesh_prim_paths.get(viewer._qualify(name))
        if path is None:
            continue
        try:
            viewer._write_runtime_array_attribute(path, f"primvars:{PRIMVAR}", data)
        except Exception as exc:
            if not getattr(viewer, "_markings_primvar_warned", False):
                print(f"markings channel write failed: {exc}", flush=True)
                viewer._markings_primvar_warned = True
            return
        del pending[name]
    if not pending:
        viewer._markings_channels_dirty = False


def bind_shader(viewer, prim):
    from pxr import UsdShade

    material, _shader = ensure_shader(viewer)
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


class ShaderPanel:
    """Shaders window. The markings channels and their looks live here."""

    def __init__(self, viewer, cortex):
        self.viewer = viewer
        self.cortex = cortex
        self.params = load_parameters()
        self.status = "Pattern and looks save as you drag."
        self.coverage = ""
        self._channels = None
        if getattr(cortex, "_authored", False):
            self.apply_pattern()
            push_look(viewer, self.params)

    def apply_pattern(self):
        cortex = self.cortex
        cache = getattr(cortex, "cache", None)
        result = getattr(cortex, "result", None)
        if cache is None or result is None or not getattr(cache, "ok", False):
            return
        mask = result.masks.get("mask3")
        if mask is None or len(mask) != len(cache.tris):
            return
        _points, _tris, channels = subset_channels(
            cache.world, cache.normals, cache.tris, mask, self.params
        )
        self._channels = channels
        self.coverage = channel_stats(channels)
        prim = cortex.prims.get(SEGMENT)
        if prim is None or not prim.IsValid() or len(channels) == 0:
            return
        write_channels(self.viewer, prim, channels)
        bind_shader(self.viewer, prim)
        push_look(self.viewer, self.params)

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

    def draw(self, imgui):
        imgui.set_next_window_pos(imgui.ImVec2(1264, 16), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(400, 720), imgui.Cond_.first_use_ever)
        imgui.begin("Shaders")
        imgui.text("Cortical markings")
        imgui.text_disabled("One shader. It fades out at the perimeter of the cap.")
        if not getattr(self.cortex, "_authored", False):
            imgui.text_wrapped("The markings surface is not ready.")
            imgui.end()
            return
        if self.coverage:
            imgui.text_wrapped(self.coverage)
        imgui.separator()
        imgui.text("Pattern")
        pattern = False
        pattern |= self._slider(imgui, "seed", "Seed", 0.0, 40.0, "%.0f")
        pattern |= self._slider(imgui, "char_coverage", "Char coverage", 0.0, 1.0, "%.2f")
        pattern |= self._slider(imgui, "char_margin_mm", "Char margin", 4.0, 12.0, "%.1f mm")
        pattern |= self._slider(imgui, "char_scale_mm", "Blotch size", 1.0, 12.0, "%.1f mm")
        pattern |= self._slider(imgui, "periosteum_width_mm", "Periosteum width", 0.0, 8.0, "%.1f mm")
        pattern |= self._slider(imgui, "blood_film", "Blood film", 0.0, 1.0, "%.2f")
        pattern |= self._slider(imgui, "blood_pooling", "Blood pooling", 0.0, 2.0, "%.2f")
        pattern |= self._slider(imgui, "edge_fade_mm", "Edge fade", 0.0, 30.0, "%.1f mm")
        imgui.separator()
        imgui.text("Channels")
        look = False
        look |= self._toggle(imgui, "char_enabled", "Char")
        look |= self._slider(imgui, "char_strength", "Char strength", 0.0, 1.0, "%.2f")
        look |= self._toggle(imgui, "periosteum_enabled", "Periosteum")
        look |= self._slider(imgui, "periosteum_strength", "Periosteum strength", 0.0, 1.0, "%.2f")
        look |= self._toggle(imgui, "blood_enabled", "Blood")
        look |= self._slider(imgui, "blood_strength", "Blood strength", 0.0, 1.0, "%.2f")
        imgui.separator()
        imgui.text("Looks")
        look |= self._color(imgui, "bone_color", "Bone")
        look |= self._color(imgui, "char_color", "Char")
        look |= self._color(imgui, "periosteum_color", "Periosteum")
        look |= self._color(imgui, "blood_color", "Blood")
        look |= self._slider(imgui, "bone_roughness", "Bone roughness", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "char_roughness", "Char roughness", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "periosteum_roughness", "Periosteum roughness", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "blood_roughness", "Blood roughness", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "bone_coat", "Bone coat", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "blood_coat", "Blood coat", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "bone_coat_roughness", "Bone coat roughness", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "blood_coat_roughness", "Blood coat roughness", 0.0, 1.0, "%.2f")
        look |= self._slider(imgui, "coat_ior", "Coat IOR", 1.1, 1.8, "%.2f")
        if pattern:
            self.params["seed"] = float(int(round(self.params["seed"])))
            self.apply_pattern()
            self._save()
        elif look:
            push_look(self.viewer, self.params)
            self._save()
        imgui.separator()
        imgui.text_disabled(self.status)
        imgui.text_disabled(str(CONFIG_PATH))
        imgui.end()


def _self_check():
    size = 36
    axis = np.linspace(0.0, 30.0, size)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt((xx - 15.0) ** 2 + (yy - 15.0) ** 2)
    zz = -2.2 * np.exp(-(radius ** 2) / 10.0)
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    tris = []
    for i in range(size - 1):
        for j in range(size - 1):
            a = i * size + j
            tris.append((a, a + 1, a + size))
            tris.append((a + 1, a + size + 1, a + size))
    tris = np.asarray(tris, np.int32)
    normals = np.zeros_like(points)
    normals[:, 2] = 1.0
    rim = np.minimum(np.minimum(points[:, 0], 30.0 - points[:, 0]), np.minimum(points[:, 1], 30.0 - points[:, 1]))
    low = dict(DEFAULTS)
    low["char_coverage"] = 0.05
    high = dict(DEFAULTS)
    high["char_coverage"] = 0.6
    sparse = compose_channels(points, normals, tris, rim, low)
    dense = compose_channels(points, normals, tris, rim, high)
    if float((dense[:, 0] > 0.35).mean()) <= float((sparse[:, 0] > 0.35).mean()):
        raise SystemExit("self-check char coverage did not grow")
    edge = rim < 1.5
    middle = rim > 8.0
    if float(dense[edge, 1].mean()) <= float(dense[middle, 1].mean()):
        raise SystemExit("self-check periosteum is not stronger at the rim")
    center = radius.reshape(-1) < 3.0
    flat = radius.reshape(-1) > 12.0
    if float(dense[center, 2].mean()) <= float(dense[flat, 2].mean()):
        raise SystemExit("self-check blood did not pool in the pit")
    held = dict(DEFAULTS)
    held["char_coverage"] = 1.0
    held["char_margin_mm"] = 8.0
    held = compose_channels(points, normals, tris, rim, held)
    if float(held[rim > 8.5, 0].max()) > 1e-6:
        raise SystemExit("self-check char extends past the margin")
    if float(held[rim < 2.0, 0].mean()) < 0.2:
        raise SystemExit("self-check char is missing from the perimeter band")
    weight = edge_weight(np.array([0.0, 7.5, 30.0]), 15.0)
    if not (weight[0] == 0.0 and weight[2] == 1.0 and 0.2 < weight[1] < 0.8):
        raise SystemExit(f"self-check edge fade is wrong ({weight})")
    print(
        "markings self-check ok  "
        f"char {100 * float((dense[:, 0] > 0.35).mean()):.0f}%  "
        f"rim peri {float(dense[edge, 1].mean()):.2f}  "
        f"pit blood {float(dense[center, 2].mean()):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    _self_check()
