"""Shaft OBJ paints stay separate meshes with the same triangles."""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rtx_materials
from fivrs_rtx_view import MESH_DIR, _combine_shaft, _load_obj, _load_obj_parts


def _corners(points, tris, normals):
    found = set()
    for tri in tris:
        corners = []
        for index in tri:
            position = tuple(np.round(points[int(index)], 5))
            normal = tuple(np.round(normals[int(index)], 4))
            corners.append((position, normal))
        start = min(range(3), key=lambda slot: corners[slot])
        found.add(tuple(corners[start:] + corners[:start]))
    return found


def test_mtl_names_map_to_shaft_parts():
    assert rtx_materials.shaft_part_id("silver.019") == "shaft_silver"
    assert rtx_materials.shaft_part_id("golden.021") == "shaft_gold"
    assert rtx_materials.shaft_part_id("black.018") == "shaft_black"
    assert rtx_materials.shaft_part_id("grey.017") == "shaft_grey"
    assert rtx_materials.shaft_part_id("gray.001") == "shaft_grey"
    assert rtx_materials.shaft_part_id("paint.004") == "shaft_paint"


def test_defaults_keep_the_obj_colors():
    document = rtx_materials.empty_document()
    silver = rtx_materials.resolve(document, "shaft_silver")
    gold = rtx_materials.resolve(document, "shaft_gold")
    black = rtx_materials.resolve(document, "shaft_black")
    grey = rtx_materials.resolve(document, "shaft_grey")
    assert silver["diffuse_reflection_color"] == [0.531616, 0.501214, 0.514546]
    assert gold["diffuse_reflection_color"][0] > 0.7
    assert gold["diffuse_reflection_color"][2] < 0.05
    assert max(black["diffuse_reflection_color"]) < 0.01
    assert grey["diffuse_reflection_color"] == [0.617706, 0.617706, 0.617706]
    assert len({tuple(item["diffuse_reflection_color"]) for item in (silver, gold, black, grey)}) == 4
    assert rtx_materials.display_name(document, "shaft_gold") == "shaft gold"
    assert rtx_materials.display_label(document, "shaft_grey") == "shaft grey"


def test_each_sized_shaft_keeps_its_four_paints():
    for size in (1, 2, 4, 6):
        path = MESH_DIR / f"{size}mm.OBJ"
        whole_points, whole_tris, whole_normals = _load_obj(path)
        parts = _load_obj_parts(path)
        assert tuple(parts) == rtx_materials.SHAFT_PART_ORDER
        whole = _corners(whole_points, whole_tris, whole_normals)
        split = set()
        total = 0
        for mesh in parts.values():
            split |= _corners(*mesh)
            total += len(mesh[1])
        assert total == len(whole_tris)
        assert split == whole


def test_one_mesh_keeps_every_paint():
    for size in (1, 2, 4, 6):
        parts = _load_obj_parts(MESH_DIR / f"{size}mm.OBJ")
        points, tris, normals, ranges = _combine_shaft(parts)
        assert len(points) == sum(len(mesh[0]) for mesh in parts.values())
        assert len(normals) == len(points)
        assert len(tris) == sum(len(mesh[1]) for mesh in parts.values())
        cursor = 0
        for part_id, mesh in parts.items():
            start, count = ranges[part_id]
            assert (start, count) == (cursor, len(mesh[1]))
            offset = sum(len(parts[name][0]) for name in list(parts)[: list(parts).index(part_id)])
            assert np.array_equal(points[offset:offset + len(mesh[0])], mesh[0])
            assert np.array_equal(tris[start:start + count] - offset, mesh[1])
            cursor += count
        assert cursor == len(tris)


def test_unknown_paint_is_not_dropped():
    text = """
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vn 0 0 1
usemtl silver.001
f 1//1 2//1 3//1
usemtl paint.009
f 1//1 3//1 4//1
"""
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "hand.obj"
        path.write_text(text)
        parts = _load_obj_parts(path)
        whole = _load_obj(path)
    assert set(parts) == {"shaft_silver", "shaft_paint"}
    assert len(parts["shaft_silver"][1]) == 1
    assert len(parts["shaft_paint"][1]) == 1
    assert len(whole[1]) == 2


if __name__ == "__main__":
    test_mtl_names_map_to_shaft_parts()
    test_defaults_keep_the_obj_colors()
    test_each_sized_shaft_keeps_its_four_paints()
    test_one_mesh_keeps_every_paint()
    test_unknown_paint_is_not_dropped()
    print("drill material tests passed")
