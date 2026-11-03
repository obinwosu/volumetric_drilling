"""Thin bone: distances on the real block mesher, and the ThinBone.mdl mirror."""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rtx_materials
import thin_bone
from surface import blocks_for_mask, extract_block

BONE = (0xFF, 0xF9, 0xDB)
NERVE = (0xAA, 0x00, 0x01)
DURA = (0xAA, 0x00, 0x02)
OTIC = (0xAA, 0x00, 0x03)
DOCUMENT = {"names": {"aa0001": "facial_nerve", "aa0002": "sinus_dura", "aa0003": "inner_ear"}}
MM = 0.25
TOP = 4          # bone starts at this z; z < TOP is air
NERVE_Z = 22     # top voxel of the nerve
DURA_Z = 30      # top voxel of the dura sheet


def _volume():
    """Bone block. A nerve tube runs through it; a dura sheet lies under one half."""
    shape = (48, 48, 48)
    occupied = np.zeros(shape, bool)
    colors = np.zeros(shape + (3,), np.uint8)
    occupied[4:44, 4:44, TOP:DURA_Z] = True
    colors[occupied] = BONE
    colors[4:44, 20:24, NERVE_Z : NERVE_Z + 4] = NERVE
    occupied[4:44, 30:44, DURA_Z : DURA_Z + 4] = True
    colors[4:44, 30:44, DURA_Z : DURA_Z + 4] = DURA
    colors[34:40, 6:12, 14:18] = OTIC
    return occupied, colors


def _index(occupied, colors):
    return thin_bone.build_index(occupied, colors, DOCUMENT, np.full(3, MM), BONE)


def _bone_vertices(occupied, colors, nerve_rgb=NERVE):
    """Index-space bone vertices from the union remesh, nerve carved, as live."""
    remesh = occupied.copy()
    remesh[np.all(colors == nerve_rgb, axis=-1)] = False
    verts = []
    for block in blocks_for_mask(remesh, 32):
        mesh, _phases = extract_block(remesh, colors, block, 32)
        if len(mesh.tris) == 0:
            continue
        bone = np.all(mesh.colors == BONE, axis=-1)
        verts.append(mesh.verts[bone])
    return np.concatenate(verts)


def _floor(verts, x, y, radius=1.5):
    near = (np.abs(verts[:, 0] - x) < radius) & (np.abs(verts[:, 1] - y) < radius)
    pick = verts[near]
    return pick[pick[:, 2] == pick[:, 2].min()] if len(pick) else pick


def test_nerve_distance_is_the_bone_left():
    occupied, colors = _volume()
    index = _index(occupied, colors)
    previous = None
    for floor in (8, 14, 18, 20):
        drilled = occupied.copy()
        drilled[10:38, 14:30, :floor] = False
        verts = _bone_vertices(drilled, colors)
        top = _floor(verts, 24.0, 22.0)
        left = float(np.median(index.sample(top)[:, 0]))
        expected = (NERVE_Z - floor) * MM
        if abs(left - expected) > 0.6 * MM:
            raise SystemExit(f"floor z={floor}: {left:.2f} mm of bone over the nerve, expected {expected:.2f}")
        if previous is not None and left >= previous:
            raise SystemExit("drilling deeper should leave less bone over the nerve")
        previous = left


def test_dura_tints_only_bone_near_it():
    occupied, colors = _volume()
    index = _index(occupied, colors)
    drilled = occupied.copy()
    drilled[6:42, 32:42, : DURA_Z - 2] = False
    values = index.sample(_floor(_bone_vertices(drilled, colors), 24.0, 37.0))
    if abs(float(np.median(values[:, 1])) - 2 * MM) > 0.6 * MM:
        raise SystemExit(f"bone over the dura reads {float(np.median(values[:, 1])):.2f} mm")
    # The intact top surface far from the dura is out of range.
    far = index.sample(_floor(_bone_vertices(occupied, colors), 24.0, 8.0))
    if float(far[:, 1].min()) < thin_bone.FAR_VALUE:
        raise SystemExit("bone 6+ mm from the dura should read far")


def test_otic_band_and_missing_groups():
    occupied, colors = _volume()
    index = _index(occupied, colors)
    near = index.sample(np.array([[37.0, 9.0, 13.5]]))
    if not near[0, 2] < 0.5:
        raise SystemExit("bone touching the inner ear should read near it")
    bare = occupied.copy()
    bare_colors = colors.copy()
    bare_colors[np.all(colors == OTIC, axis=-1)] = BONE
    empty = _index(bare, bare_colors).sample(np.array([[37.0, 9.0, 13.5]]))
    if empty[0, 2] != thin_bone.FAR_VALUE or empty[0, 3] != 1.0:
        raise SystemExit("a volume without that label should read far, still valid")


def test_tint_mirror():
    bone = [0.9, 0.88, 0.8]
    params = dict(thin_bone.DEFAULTS)
    thin = np.array([[0.0, 99.0, 99.0, 1.0]])
    thick = np.array([[3.0, 99.0, 99.0, 1.0]])
    invalid = np.array([[0.0, 0.0, 0.0, 0.0]])
    if not np.allclose(thin_bone.tinted_color(thick, params, bone), bone):
        raise SystemExit("bone past the depth should keep its colour")
    if not np.allclose(thin_bone.tinted_color(invalid, params, bone), bone):
        raise SystemExit("a vertex without values should keep the bone colour")
    off = dict(params, nerve_strength=0.0, dura_strength=0.0, otic_strength=0.0)
    if not np.allclose(thin_bone.tinted_color(thin, off, bone), bone):
        raise SystemExit("zero strengths should give plain bone")
    pink = thin_bone.tinted_color(thin, params, bone)[0]
    expected = np.asarray(bone) * (1 - params["nerve_strength"]) + np.asarray(params["nerve_color"]) * params["nerve_strength"]
    if not np.allclose(pink, expected):
        raise SystemExit("bone right over the nerve should blend by the nerve strength")
    previewed = thin_bone.tinted_color(np.array([[1.2, 99.0, 99.0, 1.0]]), params, bone, preview_mm=1.2)[0]
    if not np.allclose(previewed, expected):
        raise SystemExit("preview thinning should pull the bone toward the nerve colour")
    if thin_bone.FAR_VALUE - 2.5 <= 2.5 + 3.5:
        raise SystemExit("far points must stay far under the largest preview and depth")


def test_shader_inputs_match_the_mdl():
    mdl = (HERE / "mdl" / "ThinBone.mdl").read_text()
    declared = set(re.findall(r"(?:uniform )?(?:float|bool|color) (\w+) =", mdl.split(") = let")[0]))
    bone = rtx_materials._shader_inputs(
        {key: value for key, value in rtx_materials.FALLBACK.items()}
        | {"diffuse_reflection_color": [0.9, 0.88, 0.8]}
    )
    inputs = thin_bone.shader_inputs(thin_bone.DEFAULTS, bone)
    if set(inputs) - declared:
        raise SystemExit(f"inputs the shader does not declare: {sorted(set(inputs) - declared)}")
    passed = {key for _group, key, *_rest in rtx_materials.FIELDS}
    if not passed <= declared:
        raise SystemExit(f"bone inputs missing from ThinBone.mdl: {sorted(passed - declared)}")
    for key in ("nerve_strength", "dura_strength", "otic_strength", "otic_band_mm", "plate_gloss"):
        value = re.search(rf"uniform float {key} = ([0-9.]+)", mdl).group(1)
        if abs(float(value) - float(thin_bone.DEFAULTS[key])) > 1e-6:
            raise SystemExit(f"{key} default differs between ThinBone.mdl and thin_bone.py")


if __name__ == "__main__":
    test_nerve_distance_is_the_bone_left()
    test_dura_tints_only_bone_near_it()
    test_otic_band_and_missing_groups()
    test_tint_mirror()
    test_shader_inputs_match_the_mdl()
    print("ok")
