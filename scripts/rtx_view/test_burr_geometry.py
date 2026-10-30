"""Cutting diameter is the flute width. The baked tip sphere is not part of it."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rtx_materials
from diamond_burr import MESH_PATH, mesh
from fivrs_rtx_view import MESH_DIR, MM_TO_SIM, _load_obj, _load_obj_parts, _tuck_baked_tip


def test_flute_width_is_the_nominal_diameter():
    points, _tris, _normals = _load_obj(MESH_DIR / "fluted_burr.obj")
    radius = np.linalg.norm(points[:, 1:], axis=1).max()
    # Authored radius is 1. Scaling by half the nominal size makes the
    # outermost flute edges that diameter.
    assert abs(radius - 1.0) < 0.002
    for size in (1, 2, 4, 6):
        diameter = 2.0 * radius * (size * MM_TO_SIM * 0.5) / MM_TO_SIM
        assert abs(diameter - size) < 0.02


def test_baked_tip_sphere_tucks_inside_the_burr():
    for size in (1, 2, 4, 6):
        radius = 0.5 * size * MM_TO_SIM
        parts = _load_obj_parts(MESH_DIR / f"{size}mm.OBJ")
        moved = 0
        for part in parts.values():
            original = part[0]
            points, tris, _normals, count = _tuck_baked_tip(*part, size)
            moved += count
            assert len(tris) == len(part[1])
            neck = original[:, 0] >= 10.0 * MM_TO_SIM
            assert np.allclose(points[neck], original[neck])
            tip = ~neck
            if np.any(tip):
                tucked_r = np.linalg.norm(points[tip, 1:], axis=1)
                assert np.allclose(points[tip, 0], 0.20 * radius)
                assert np.all(tucked_r <= 0.45 * radius + 1e-5)
        assert moved > 0


def test_diamond_grains_do_not_grow_the_diameter():
    points, faces, normals = mesh()
    radius = np.linalg.norm(points, axis=1)
    assert abs(float(radius.max()) - 1.0) < 1e-5
    assert float(radius.min()) < 0.95
    assert 4000 <= len(faces) <= 6000
    outward = np.sum(normals * points[faces[:, 0]], axis=1)
    assert np.all(outward > 0.0)
    loaded_points, loaded_tris, loaded_normals = _load_obj(MESH_PATH)
    loaded_radius = np.linalg.norm(loaded_points, axis=1)
    assert abs(float(loaded_radius.max()) - 1.0) < 1e-4
    assert len(loaded_tris) == len(faces)
    assert len(np.unique(np.round(loaded_normals, 4), axis=0)) > 2000
    document = rtx_materials.empty_document()
    assert rtx_materials.display_name(document, "burr_diamond") == "diamond burr"
    grit = rtx_materials.resolve(document, "burr_diamond")
    cutting = rtx_materials.resolve(document, "burr")
    assert grit["specular_reflection_roughness"] > cutting["specular_reflection_roughness"]


if __name__ == "__main__":
    test_flute_width_is_the_nominal_diameter()
    test_baked_tip_sphere_tucks_inside_the_burr()
    test_diamond_grains_do_not_grow_the_diameter()
    print("burr geometry tests passed")
