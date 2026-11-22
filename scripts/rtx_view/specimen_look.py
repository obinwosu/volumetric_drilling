"""The UR5 specimens' look in the live RTX view: whiter bone and a warm cream otic capsule.

Active only for a volume with an ``otic_capsule.npz`` beside its slices (a UR5 specimen with a
hollow inner ear). It is laid over the tuner's values as they load and is never saved into them,
so every other anatomy, and the tuner itself, keep the look they had.

Values: ``ur5_specimen_look.json``.
"""
from __future__ import annotations

import json
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
LOOK_PATH = HERE / "ur5_specimen_look.json"
BONE_SEGMENTS = ("bone", "fff9db", "lateral_cortex", "cortex_markings")
_active = None


def has_capsule(anatomy_adf) -> bool:
    adf = pathlib.Path(anatomy_adf)
    doc = yaml.safe_load(adf.read_text())
    return ((adf.parent / doc[doc["volumes"][0]]["images"]["path"]).resolve() / "otic_capsule.npz").is_file()


def activate(anatomy_adf) -> bool:
    """Call once at start-up, before any material loads. True for a UR5 specimen."""
    global _active
    _active = json.loads(LOOK_PATH.read_text()) if has_capsule(anatomy_adf) else None
    if _active is not None:
        print(f"UR5 specimen look: {_active['capsule_rule']}, {_active['capsule_band_mm']:g} mm", flush=True)
    return _active is not None


def deactivate():
    global _active
    _active = None


def rule():
    return None if _active is None else _active["capsule_rule"]


def apply_materials(document: dict) -> dict:
    if _active is not None:
        for key in BONE_SEGMENTS:
            document["segments"].setdefault(key, {})["diffuse_reflection_color"] = list(_active["bone_color"])
    return document


def apply_markings(params: dict) -> dict:
    if _active is not None:
        params["bone_color"] = list(_active["bone_color"])
    return params


def apply_thin_bone(params: dict) -> dict:
    """The shader multiplies the bone color by otic_color, so this ratio lands on the capsule color."""
    if _active is not None:
        params["otic_color"] = [c / b for c, b in zip(_active["capsule_color"], _active["bone_color"])]
        params["otic_band_mm"] = float(_active["capsule_band_mm"])
        params["otic_strength"] = float(_active["capsule_strength"])
    return params
