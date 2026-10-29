"""Shared OmniSurface materials for the tuner and the live RTX window.

A volume structure is keyed by its segmentation hex and, when it has one,
by a regular name such as bone. The burr is name-only. Display color is a
separate OmniSurface input, so a saved preset can recolor a structure
without renaming it. Lighting stays in tuner_light.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MATERIALS_PATH = HERE / "tuner_materials.json"
LABELS_PATH = HERE.parents[1] / "ADF" / "color" / "volume_labels.yaml"

# Segmentation color the plugin treats as bone. Official names are keyed by
# the original volume hex in volume_labels.yaml, never by display color.
KNOWN_NAMES = {
    "fff9db": "bone",
}

# One water sheet. Thin-walled, IOR 1: transmission and opacity, no refraction.
# A single surface can sit clearer than stacked spheres, which added up to black.
WATER = {
    "diffuse_reflection_color": [0.62, 0.82, 0.94],
    "diffuse_reflection_weight": 0.15,
    "diffuse_reflection_roughness": 0.0,
    "metalness": 0.0,
    "specular_reflection_weight": 0.65,
    "specular_reflection_color": [1.0, 1.0, 1.0],
    "specular_reflection_roughness": 0.03,
    "specular_reflection_ior": 1.0,
    "coat_weight": 0.0,
    "enable_specular_transmission": True,
    "specular_transmission_weight": 0.85,
    "specular_transmission_color": [0.78, 0.9, 1.0],
    "thin_walled": True,
    "enable_opacity": True,
    "geometry_opacity": 0.22,
}

# Name-only parts. They are not a volume hex. The cortex entries are appearance
# layers on the bone masks: same ivory, different coat. The bone hex stays bone.
_BONE_IVORY = [255 / 255, 249 / 255, 219 / 255]
EXTRA_DEFAULTS = {
    "burr": {
        "diffuse_reflection_color": [0.82, 0.83, 0.85],
        "metalness": 0.92,
        "specular_reflection_roughness": 0.18,
    },
    "lateral_cortex": {
        "diffuse_reflection_color": list(_BONE_IVORY),
        "specular_reflection_roughness": 0.12,
        "coat_weight": 0.22,
        "coat_roughness": 0.05,
        "coat_color": [1.0, 1.0, 1.0],
    },
    "cortex_markings": {
        "diffuse_reflection_color": list(_BONE_IVORY),
        "specular_reflection_roughness": 0.42,
        "coat_weight": 0.0,
        "coat_roughness": 0.2,
        "coat_color": [0.45, 0.08, 0.06],
    },
}
APPEARANCE_LABELS = {
    "lateral_cortex": "lateral cortex",
    "cortex_markings": "cortex markings",
}

# Matches the bone the live view already draws: segment color, roughness 0.55,
# not metal, no coat, no subsurface, no transmission, opaque.
FALLBACK = {
    "diffuse_reflection_weight": 1.0,
    "diffuse_reflection_roughness": 0.0,
    "metalness": 0.0,
    "specular_reflection_weight": 1.0,
    "specular_reflection_color": [1.0, 1.0, 1.0],
    "specular_reflection_roughness": 0.55,
    "specular_reflection_ior": 1.5,
    "specular_reflection_anisotropy": 0.0,
    "specular_reflection_anisotropy_rotation": 0.0,
    "coat_weight": 0.0,
    "coat_color": [1.0, 1.0, 1.0],
    "coat_roughness": 0.1,
    "coat_ior": 1.5,
    "coat_anisotropy": 0.0,
    "coat_anisotropy_rotation": 0.0,
    "coat_affect_color": 0.0,
    "coat_affect_roughness": 0.0,
    "enable_diffuse_transmission": False,
    "subsurface_weight": 0.0,
    "subsurface_transmission_color": [1.0, 1.0, 1.0],
    "subsurface_scattering_color": [1.0, 1.0, 1.0],
    "subsurface_scale": 1.0,
    "subsurface_anisotropy": 0.0,
    "enable_specular_transmission": False,
    "specular_transmission_weight": 0.0,
    "specular_transmission_color": [1.0, 1.0, 1.0],
    "specular_transmission_scattering_depth": 0.0,
    "specular_transmission_scattering_color": [1.0, 1.0, 1.0],
    "specular_transmission_scatter_anisotropy": 0.0,
    "specular_transmission_dispersion_abbe": 0.0,
    "thin_walled": False,
    "specular_retro_reflection_weight": 0.0,
    "specular_retro_reflection_color": [1.0, 1.0, 1.0],
    "specular_retro_reflection_roughness": 0.3,
    "enable_opacity": False,
    "geometry_opacity": 1.0,
    "geometry_opacity_threshold": 0.0,
    "emission_weight": 0.0,
    "emission_intensity": 1.0,
    "emission_color": [1.0, 1.0, 1.0],
    "emission_use_temperature": False,
    "emission_temperature": 6500.0,
    "enable_thin_film": False,
    "thin_film_thickness": 400.0,
    "thin_film_ior": 1.52,
}

# OmniSurface inputs this renderer can author. Texture maps and anything that
# would rebuild the mesh are listed separately and are not given sliders.
FIELDS = [
    ("Base", "diffuse_reflection_color", "Display color", "color", None, None),
    ("Base", "diffuse_reflection_weight", "Base weight", "float", 0.0, 1.0),
    ("Base", "diffuse_reflection_roughness", "Diffuse roughness", "float", 0.0, 1.0),
    ("Base", "metalness", "Metallic", "float", 0.0, 1.0),
    ("Specular", "specular_reflection_weight", "Specular weight", "float", 0.0, 1.0),
    ("Specular", "specular_reflection_color", "Specular color", "color", None, None),
    ("Specular", "specular_reflection_roughness", "Roughness", "float", 0.0, 1.0),
    ("Specular", "specular_reflection_ior", "IOR", "float", 1.0, 3.0),
    ("Specular", "specular_reflection_anisotropy", "Anisotropy", "float", 0.0, 1.0),
    ("Specular", "specular_reflection_anisotropy_rotation", "Anisotropy rotation", "float", 0.0, 1.0),
    ("Coat", "coat_weight", "Coat strength", "float", 0.0, 1.0),
    ("Coat", "coat_color", "Coat color", "color", None, None),
    ("Coat", "coat_roughness", "Coat roughness", "float", 0.0, 1.0),
    ("Coat", "coat_ior", "Coat IOR", "float", 1.0, 3.0),
    ("Coat", "coat_anisotropy", "Coat anisotropy", "float", 0.0, 1.0),
    ("Coat", "coat_anisotropy_rotation", "Coat anisotropy rotation", "float", 0.0, 1.0),
    ("Coat", "coat_affect_color", "Coat affects color", "float", 0.0, 1.0),
    ("Coat", "coat_affect_roughness", "Coat affects roughness", "float", 0.0, 1.0),
    ("Subsurface", "enable_diffuse_transmission", "Enable subsurface", "bool", None, None),
    ("Subsurface", "subsurface_weight", "Subsurface weight", "float", 0.0, 1.0),
    ("Subsurface", "subsurface_transmission_color", "Transmission color", "color", None, None),
    ("Subsurface", "subsurface_scattering_color", "Scattering color", "color", None, None),
    ("Subsurface", "subsurface_scale", "Scattering scale", "float", 0.0, 10.0),
    ("Subsurface", "subsurface_anisotropy", "Scattering anisotropy", "float", -0.99, 0.99),
    ("Transmission", "enable_specular_transmission", "Enable transmission", "bool", None, None),
    ("Transmission", "specular_transmission_weight", "Transmission weight", "float", 0.0, 1.0),
    ("Transmission", "specular_transmission_color", "Transmission color", "color", None, None),
    ("Transmission", "specular_transmission_scattering_depth", "Scattering depth", "float", 0.0, 5.0),
    ("Transmission", "specular_transmission_scattering_color", "Depth color", "color", None, None),
    ("Transmission", "specular_transmission_scatter_anisotropy", "Depth anisotropy", "float", -0.99, 0.99),
    ("Transmission", "specular_transmission_dispersion_abbe", "Dispersion Abbe", "float", 0.0, 100.0),
    ("Transmission", "thin_walled", "Thin walled", "bool", None, None),
    ("Retro", "specular_retro_reflection_weight", "Retro weight", "float", 0.0, 1.0),
    ("Retro", "specular_retro_reflection_color", "Retro color", "color", None, None),
    ("Retro", "specular_retro_reflection_roughness", "Retro roughness", "float", 0.0, 1.0),
    ("Opacity", "enable_opacity", "Enable opacity", "bool", None, None),
    ("Opacity", "geometry_opacity", "Opacity", "float", 0.0, 1.0),
    ("Opacity", "geometry_opacity_threshold", "Opacity cutoff", "float", 0.0, 1.0),
    ("Emission", "emission_weight", "Emission weight", "float", 0.0, 10.0),
    ("Emission", "emission_intensity", "Emission intensity", "float", 0.0, 10.0),
    ("Emission", "emission_color", "Emission color", "color", None, None),
    ("Emission", "emission_use_temperature", "Use temperature", "bool", None, None),
    ("Emission", "emission_temperature", "Temperature", "float", 1000.0, 10000.0),
    ("Thin film", "enable_thin_film", "Enable thin film", "bool", None, None),
    ("Thin film", "thin_film_thickness", "Thickness (nm)", "float", 0.0, 1500.0),
    ("Thin film", "thin_film_ior", "Thin film IOR", "float", 1.0, 3.0),
]

UNSUPPORTED = (
    "Texture maps for color, roughness, metalness, coat, and opacity",
    "Normal maps and displacement",
    "Round-edge radius and roundness",
    "Subsurface scattering color presets",
)


def segment_id(packed: int) -> str:
    red, green, blue = packed & 255, (packed >> 8) & 255, (packed >> 16) & 255
    return f"{red:02x}{green:02x}{blue:02x}"


def is_hex_id(segment: str) -> bool:
    text = (segment or "").strip().lower()
    return len(text) == 6 and all(char in "0123456789abcdef" for char in text)


def identity_color(segment: str) -> list[float]:
    return [int(segment[0:2], 16) / 255.0, int(segment[2:4], 16) / 255.0, int(segment[4:6], 16) / 255.0]


def clean_name(text: str) -> str:
    cleaned = (text or "").strip().lower().replace(" ", "_")
    return "".join(char for char in cleaned if char.isalnum() or char == "_")


def _parse_label_yaml(text: str) -> dict[str, str]:
    names = {}
    in_labels = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "labels:":
            in_labels = True
            continue
        if not in_labels or ":" not in line:
            continue
        key, value = line.split(":", 1)
        hex_id = key.strip().lower().lstrip("#")
        alias = clean_name(value)
        if is_hex_id(hex_id) and alias and not is_hex_id(alias):
            names[hex_id] = alias
    return names


def upstream_names() -> dict[str, str]:
    """Names for the original volume hex. Ignores edited display color."""
    names = dict(KNOWN_NAMES)
    if LABELS_PATH.is_file():
        try:
            names.update(_parse_label_yaml(LABELS_PATH.read_text()))
        except OSError as exc:
            print(f"Could not read {LABELS_PATH}: {exc}", flush=True)
    return names


def name_map(document: dict | None = None) -> dict[str, str]:
    names = upstream_names()
    if document:
        for key, value in (document.get("names") or {}).items():
            hex_id = str(key).strip().lower()
            alias = clean_name(str(value))
            if is_hex_id(hex_id) and alias and not is_hex_id(alias) and hex_id not in names:
                names[hex_id] = alias
    return names


def name_of(document: dict, segment: str) -> str:
    segment = (segment or "").strip().lower()
    names = name_map(document)
    if segment in names:
        return names[segment]
    if segment in names.values() or segment in EXTRA_DEFAULTS:
        return segment
    return ""


def hex_of(document: dict, segment: str) -> str:
    segment = (segment or "").strip().lower()
    if is_hex_id(segment):
        return segment
    for hex_id, alias in name_map(document).items():
        if alias == segment:
            return hex_id
    return ""


def aliases(document: dict, segment: str) -> list[str]:
    """Hex and regular name for the same structure, so either key updates it."""
    segment = (segment or "").strip().lower()
    hex_id = hex_of(document, segment)
    alias = name_of(document, hex_id or segment)
    keys = [hex_id, alias, segment]
    seen = []
    for key in keys:
        if key and key not in seen:
            seen.append(key)
    return seen


def display_name(document: dict, segment: str) -> str:
    """List label. A named structure shows bone, not #fff9db."""
    if segment in APPEARANCE_LABELS:
        return APPEARANCE_LABELS[segment]
    return name_of(document, segment) or extra_or_hex(document, segment)


def extra_or_hex(document: dict, segment: str) -> str:
    hex_id = hex_of(document, segment)
    if hex_id:
        return f"#{hex_id}"
    return segment


def display_label(document: dict, segment: str) -> str:
    alias = name_of(document, segment)
    hex_id = hex_of(document, segment)
    if alias and hex_id:
        return f"{alias}  #{hex_id}"
    if alias:
        return alias
    if hex_id:
        return f"#{hex_id}"
    return segment


def empty_document() -> dict:
    return {"version": 1, "fallback": dict(FALLBACK), "names": dict(upstream_names()), "segments": {}}


def load_document() -> dict:
    document = empty_document()
    if not MATERIALS_PATH.is_file():
        return document
    try:
        saved = json.loads(MATERIALS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {MATERIALS_PATH}: {exc}", flush=True)
        return document
    fallback = dict(FALLBACK)
    fallback.update(saved.get("fallback") or {})
    document["fallback"] = fallback
    document["names"] = name_map(saved)
    document["segments"] = {str(key): dict(value) for key, value in (saved.get("segments") or {}).items()}
    slots = document["segments"]
    for hex_id, alias in document["names"].items():
        if hex_id in slots and alias not in slots:
            slots[alias] = dict(slots[hex_id])
        elif alias in slots and hex_id not in slots:
            slots[hex_id] = dict(slots[alias])
    return document


def save_document(document: dict) -> None:
    payload = {
        "version": 1,
        "names": name_map(document),
        "fallback": document.get("fallback") or dict(FALLBACK),
        "segments": document.get("segments") or {},
    }
    temporary = MATERIALS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(MATERIALS_PATH)


def resolve(document: dict, segment: str) -> dict:
    """Full OmniSurface values. Hex and regular name both apply."""
    params = dict(FALLBACK)
    params.update(document.get("fallback") or {})
    hex_id = hex_of(document, segment)
    alias = name_of(document, segment)
    if hex_id:
        params["diffuse_reflection_color"] = identity_color(hex_id)
    else:
        extra = dict(EXTRA_DEFAULTS.get(alias or segment) or {})
        params.update(extra)
        params.setdefault("diffuse_reflection_color", [0.8, 0.8, 0.8])
    stored = document.get("segments") or {}
    for key in (hex_id, alias, segment):
        if key and key in stored:
            params.update(stored[key])
    return params


def _mdl_asset() -> str:
    import ovrtx

    path = pathlib.Path(ovrtx.__file__).resolve().parent / "bin" / "library" / "mdl" / "Base" / "OmniSurface.mdl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def material_path(viewer, segment: str) -> str:
    return f"{viewer._root_path}/Looks/seg_{segment}"


def ensure_material(viewer, segment: str):
    """One OmniSurface per segmentation id. Meshes only reference it."""
    from pxr import Sdf, UsdShade

    path = material_path(viewer, segment)
    stage = viewer.stage
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    prim = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(_mdl_asset()))
    prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniSurface")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for name in ("surface", "displacement", "volume"):
        material.CreateOutput(f"mdl:{name}", Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "out")
    return material, shader


def attach(viewer):
    viewer._material_document = load_document()
    viewer._material_mtime = MATERIALS_PATH.stat().st_mtime if MATERIALS_PATH.is_file() else None
    viewer._material_params = {}
    viewer._segment_prims = {}
    viewer._materials_dirty = False


def _payload(value):
    if isinstance(value, bool):
        return np.array([bool(value)])
    if isinstance(value, (list, tuple)):
        return np.array([tuple(float(channel) for channel in value)], dtype=np.float32)
    return np.array([float(value)], dtype=np.float32)


def _author_input(shader, name: str, value):
    from pxr import Gf, Sdf

    prim = shader.GetPrim()
    attr_name = f"inputs:{name}"
    if isinstance(value, bool):
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
        attr.Set(bool(value))
    elif isinstance(value, (list, tuple)):
        color = tuple(float(channel) for channel in value)
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Color3f)
        attr.Set(Gf.Vec3f(*color))
    else:
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.Set(float(value))
    return attr_name


def _runtime_input(viewer, shader, name: str, value):
    if getattr(viewer, "_rtx", None) is None:
        return
    attr_name = f"inputs:{name}"
    try:
        viewer._write_runtime_attribute([str(shader.GetPrim().GetPath())], attr_name, _payload(value))
    except Exception as exc:
        warned = getattr(viewer, "_material_warned", None)
        if warned is None:
            warned = set()
            viewer._material_warned = warned
        if name not in warned:
            print(f"material input {name} is stored but did not update live ({exc})", flush=True)
            warned.add(name)


def apply_params(viewer, segment: str, params: dict):
    """Author the shared shader. The renderer write happens on the next frame."""
    _material, shader = ensure_material(viewer, segment)
    stored = {}
    for _group, key, _label, _kind, _low, _high in FIELDS:
        if key not in params:
            continue
        stored[key] = params[key]
        _author_input(shader, key, params[key])
    params_by_segment = getattr(viewer, "_material_params", None)
    if params_by_segment is None:
        params_by_segment = {}
        viewer._material_params = params_by_segment
    params_by_segment[segment] = stored
    viewer._materials_dirty = True


def flush_materials(viewer):
    """Push dirty shader inputs. Call this from the camera update."""
    if getattr(viewer, "_materials_dirty", False) and getattr(viewer, "_rtx", None) is not None:
        for segment, params in list(getattr(viewer, "_material_params", {}).items()):
            _material, shader = ensure_material(viewer, segment)
            for name, value in params.items():
                _author_input(shader, name, value)
                _runtime_input(viewer, shader, name, value)
        viewer._materials_dirty = False
    flush_visibility(viewer)


def set_visibility(viewer, prim, visible: bool):
    """Queue a preview-only hide. The live window never calls this."""
    from pxr import UsdGeom

    token = "inherited" if visible else "invisible"
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(token)
    pending = getattr(viewer, "_visibility_pending", None)
    if pending is None:
        pending = {}
        viewer._visibility_pending = pending
    pending[str(prim.GetPath())] = token


def flush_visibility(viewer):
    pending = getattr(viewer, "_visibility_pending", None)
    if not pending or getattr(viewer, "_rtx", None) is None:
        return
    for path, token in list(pending.items()):
        try:
            viewer._write_runtime_attribute([path], "visibility", [token])
        except Exception as exc:
            if not getattr(viewer, "_visibility_warned", False):
                print(f"visibility write failed: {exc}", flush=True)
                viewer._visibility_warned = True
            return
    pending.clear()


def bind_prim(viewer, prim, segment: str, document: dict):
    from pxr import UsdShade

    if segment not in getattr(viewer, "_material_params", {}):
        apply_params(viewer, segment, resolve(document, segment))
    else:
        ensure_material(viewer, segment)
    material = UsdShade.Material(viewer.stage.GetPrimAtPath(material_path(viewer, segment)))
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    paths = getattr(viewer, "_segment_prims", None)
    if paths is None:
        paths = {}
        viewer._segment_prims = paths
    paths.setdefault(segment, set()).add(str(prim.GetPath()))


def bind_water(viewer, prim):
    """One shared OmniSurface for the irrigation water sheet."""
    if "water" not in getattr(viewer, "_material_params", {}):
        params = dict(WATER)
        try:
            from fluid.irrigation_bench import load_irrigation, water_from_irrigation, IRRIGATION_PATH

            params = water_from_irrigation(load_irrigation(IRRIGATION_PATH), WATER)
        except Exception:
            pass
        apply_params(viewer, "water", params)
    bind_prim(viewer, prim, "water", getattr(viewer, "_material_document", None) or {"segments": {}, "names": {}})


def bind_named_mesh(viewer, mesh_name: str, segment: str, document: dict):
    path = viewer._get_path(viewer._qualify(mesh_name))
    prim = viewer.stage.GetPrimAtPath(path)
    if prim:
        bind_prim(viewer, prim, segment, document)


def reload_if_changed(viewer):
    """Refresh shader inputs when the tuner saves. Mesh topology stays put."""
    try:
        mtime = MATERIALS_PATH.stat().st_mtime
    except OSError:
        return
    if mtime == getattr(viewer, "_material_mtime", None):
        return
    document = load_document()
    viewer._material_document = document
    viewer._material_mtime = mtime
    for segment in list(getattr(viewer, "_segment_prims", {}) or {}):
        apply_params(viewer, segment, resolve(document, segment))
    print(f"RTX materials from {MATERIALS_PATH.name}", flush=True)


class MaterialUi:
    """Structure list and OmniSurface editor. Visibility stays in this window."""

    def __init__(self, viewer, segments: list[dict]):
        self.viewer = viewer
        self.document = viewer._material_document
        self.segments = segments
        self.selected = segments[0]["id"] if segments else ""
        self.visible = {item["id"]: True for item in segments}
        self.status = "Material changes save as you drag."
        self._name_edit_id = ""
        self._name_edit = ""

    def _prims(self, segment: str):
        return [item["prim"] for item in self.segments if item["id"] == segment]

    def _show(self):
        for item in self.segments:
            set_visibility(self.viewer, item["prim"], self.visible[item["id"]])

    def _write_params(self, params: dict) -> dict:
        stored = {}
        for _group, key, _label, kind, _low, _high in FIELDS:
            if key not in params:
                continue
            value = params[key]
            if kind == "bool":
                stored[key] = bool(value)
            elif kind == "color":
                stored[key] = [float(channel) for channel in value]
            else:
                stored[key] = float(value)
        return stored

    def _store(self, segment: str, params: dict):
        stored = self._write_params(params)
        slots = self.document.setdefault("segments", {})
        for key in aliases(self.document, segment):
            slots[key] = stored
        save_document(self.document)
        apply_params(self.viewer, segment, stored)
        self.status = f"Saved {display_label(self.document, segment)} to {MATERIALS_PATH.name}"

    def _rename(self, segment: str, text: str):
        if not is_hex_id(segment):
            return
        alias = clean_name(text)
        if is_hex_id(alias):
            alias = ""
        names = self.document.setdefault("names", {})
        previous = names.get(segment, KNOWN_NAMES.get(segment, ""))
        if alias == previous:
            return
        if alias:
            taken = next((hex_id for hex_id, name in name_map(self.document).items() if name == alias and hex_id != segment), "")
            if taken:
                self.status = f"{alias} is already #{taken}"
                return
        slots = self.document.setdefault("segments", {})
        if previous and previous != segment:
            slots.pop(previous, None)
        if alias:
            names[segment] = alias
            if segment in slots:
                slots[alias] = dict(slots[segment])
        else:
            names.pop(segment, None)
        save_document(self.document)
        self.status = f"{display_label(self.document, segment)} updates by name and hex"

    def draw(self, imgui):
        imgui.set_next_window_pos(imgui.ImVec2(400, 16), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(420, 680), imgui.Cond_.first_use_ever)
        imgui.begin("Structures")
        imgui.text("Structures")
        imgui.text_disabled("Names follow the original volume hex, not the display color.")
        if imgui.button("Isolate selected"):
            for item in self.segments:
                self.visible[item["id"]] = item["id"] == self.selected
            self._show()
        imgui.same_line()
        if imgui.button("Show all"):
            for item in self.segments:
                self.visible[item["id"]] = True
            self._show()
        child_flags = imgui.ChildFlags_.borders.value
        if imgui.begin_child("structure_list", imgui.ImVec2(0, 220), child_flags):
            rows = sorted(
                self.segments,
                key=lambda item: (name_of(self.document, item["id"]) == "", display_name(self.document, item["id"])),
            )
            for item in rows:
                segment = item["id"]
                changed, shown = imgui.checkbox(f"##vis_{segment}", self.visible[segment])
                if changed:
                    self.visible[segment] = bool(shown)
                    self._show()
                imgui.same_line()
                clicked, _selected = imgui.selectable(
                    f"{display_name(self.document, segment)}##{segment}",
                    segment == self.selected,
                )
                if clicked:
                    self.selected = segment
                hex_id = hex_of(self.document, segment)
                if hex_id and name_of(self.document, segment):
                    imgui.same_line()
                    imgui.text_disabled(f"#{hex_id}")
        imgui.end_child()
        imgui.separator()
        segment = self.selected
        if not segment:
            imgui.text_disabled("No structures in this volume.")
            imgui.end()
            return
        imgui.text(display_name(self.document, segment))
        if is_hex_id(segment):
            if segment != self._name_edit_id:
                self._name_edit_id = segment
                self._name_edit = name_of(self.document, segment)
            hint = getattr(imgui, "input_text_with_hint", None)
            if hint:
                _changed, value = hint("Name", "e.g. bone", self._name_edit)
            else:
                _changed, value = imgui.input_text("Name", self._name_edit)
            self._name_edit = value
            if imgui.is_item_deactivated_after_edit():
                self._rename(segment, value)
                self._name_edit = name_of(self.document, segment)
            imgui.text_disabled("Identity is the original hex. Display color can change.")
        elif segment in APPEARANCE_LABELS:
            imgui.text_disabled("Bone appearance inside this mask. The bone segmentation is unchanged.")
        else:
            imgui.text_disabled("Instrument part. Materials save under this name.")
        if imgui.button("Reset structure"):
            slots = self.document.get("segments", {})
            for key in aliases(self.document, segment):
                slots.pop(key, None)
            save_document(self.document)
            apply_params(self.viewer, segment, resolve(self.document, segment))
            self.status = f"{display_label(self.document, segment)} reset to the current-appearance fallback"
        params = resolve(self.document, segment)
        current = None
        for group, key, label, kind, low, high in FIELDS:
            if group != current:
                current = group
                if group in ("Base", "Specular"):
                    imgui.set_next_item_open(True, imgui.Cond_.once)
                opened = imgui.collapsing_header(group)
            else:
                opened = opened
            if not opened:
                continue
            imgui.push_id(key)
            if kind == "bool":
                changed, value = imgui.checkbox(label, bool(params[key]))
                if changed:
                    params[key] = bool(value)
                    self._store(segment, params)
            elif kind == "color":
                changed, value = imgui.color_edit3(label, [float(channel) for channel in params[key]])
                if changed:
                    params[key] = [float(channel) for channel in value]
                    self._store(segment, params)
            else:
                changed, value = imgui.slider_float(label, float(params[key]), float(low), float(high), "%.3f")
                if changed:
                    params[key] = float(value)
                    self._store(segment, params)
            imgui.pop_id()
        imgui.separator()
        imgui.text_disabled("Not editable here")
        for line in UNSUPPORTED:
            imgui.bullet_text(line)
        imgui.separator()
        if imgui.button("Save materials"):
            save_document(self.document)
            self.status = f"Saved {MATERIALS_PATH.name}"
        imgui.text_disabled(self.status)
        imgui.text_disabled(str(MATERIALS_PATH))
        imgui.end()
