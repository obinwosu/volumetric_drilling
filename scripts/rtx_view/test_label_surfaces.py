"""Buried labels keep a closed surface when meshed as their own occupancy."""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import rtx_materials
from surface import (
    closed_label_mask,
    closed_label_meshes,
    extract_block,
    labels_matching,
    occupancy_for_rgb,
    packed_rgb,
    rgb_hex,
    segment_colors,
)


def _face_count(mesh, rgb):
    if len(mesh.tris) == 0:
        return 0
    key = packed_rgb(rgb)
    packed = (
        mesh.colors[:, 0].astype(np.int32)
        | (mesh.colors[:, 1].astype(np.int32) << 8)
        | (mesh.colors[:, 2].astype(np.int32) << 16)
    )
    return int(np.count_nonzero(packed[mesh.tris[:, 0]] == key))


def test_buried_nerve_is_invisible_on_the_union_and_closed_on_its_own():
    occupied = np.zeros((40, 40, 40), dtype=bool)
    colors = np.zeros((40, 40, 40, 3), dtype=np.uint8)
    occupied[4:36, 4:36, 4:36] = True
    colors[occupied] = (255, 249, 219)
    # One-voxel facial nerve through the middle, fully inside bone.
    occupied[20, 20, 8:32] = True
    colors[20, 20, 8:32] = (255, 0, 4)
    bone = (255, 249, 219)
    nerve = (255, 0, 4)
    union, _ = extract_block(occupied, colors, (0, 0, 0), smooth=False)
    alone, _ = extract_block(occupancy_for_rgb(occupied, colors, nerve), colors, (0, 0, 0), smooth=False)
    if _face_count(union, nerve) > 0:
        raise SystemExit("union marching cubes should not paint a fully buried nerve")
    if len(alone.tris) < 80:
        raise SystemExit(f"nerve occupancy should be a closed tube, got {len(alone.tris)} tris")
    if _face_count(alone, nerve) < 80:
        raise SystemExit("the closed tube should be nerve-colored")
    labels = segment_colors(occupied, colors)
    if len(labels) != 2:
        raise SystemExit(f"expected bone and nerve, got {labels}")
    bone_only, _ = extract_block(occupancy_for_rgb(occupied, colors, bone), colors, (0, 0, 0), smooth=False)
    if len(bone_only.tris) < len(union.tris):
        raise SystemExit("bone-only occupancy should include the canal wall")
    print(
        f"buried nerve: union nerve faces {_face_count(union, nerve)}  "
        f"own occupancy {len(alone.tris)} tris  bone canal {len(bone_only.tris)} tris",
        flush=True,
    )


def test_closed_labels_skip_bone_and_eac():
    occupied = np.zeros((40, 40, 40), dtype=bool)
    colors = np.zeros((40, 40, 40, 3), dtype=np.uint8)
    bone = (255, 249, 219)
    eac = (255, 225, 214)
    nerve = (255, 0, 4)
    occupied[4:36, 4:36, 4:36] = True
    colors[occupied] = bone
    occupied[20, 20, 8:32] = True
    colors[20, 20, 8:32] = nerve
    occupied[2:6, 10:30, 10:30] = True
    colors[2:6, 10:30, 10:30] = eac
    groups = closed_label_meshes(occupied, colors, skip_rgb=(bone, eac), min_tris=40)
    keys = {packed_rgb(rgb) for rgb, _mesh in groups}
    if packed_rgb(nerve) not in keys:
        raise SystemExit("facial nerve should keep a closed surface")
    if packed_rgb(bone) in keys:
        raise SystemExit("bone should stay on the live remesh path")
    if packed_rgb(eac) in keys:
        raise SystemExit("EAC should stay an open canal")
    _verts, tris, _colors, _normals = next(mesh for rgb, mesh in groups if packed_rgb(rgb) == packed_rgb(nerve))
    if len(tris) < 80:
        raise SystemExit(f"nerve closed surface too small: {len(tris)} tris")


def test_keep_rgb_meshes_only_the_selected_structures():
    occupied = np.zeros((40, 40, 40), dtype=bool)
    colors = np.zeros((40, 40, 40, 3), dtype=np.uint8)
    bone = (255, 249, 219)
    nerve = (255, 0, 4)
    chorda = (255, 1, 4)
    occupied[4:36, 4:36, 4:36] = True
    colors[occupied] = bone
    occupied[20, 20, 8:32] = True
    colors[20, 20, 8:32] = nerve
    occupied[22, 22, 10:28] = True
    colors[22, 22, 10:28] = chorda
    keep = labels_matching(occupied, colors, {rgb_hex(nerve), rgb_hex(chorda)})
    if {item[0] for item in keep} != {rgb_hex(nerve), rgb_hex(chorda)}:
        raise SystemExit(f"expected nerve and chorda hexes, got {keep}")
    groups = closed_label_meshes(
        occupied, colors, keep_rgb=[rgb for _hex, rgb in keep], min_tris=40
    )
    keys = {packed_rgb(rgb) for rgb, _mesh in groups}
    if packed_rgb(bone) in keys:
        raise SystemExit("bone should not be in the selected closed set")
    if packed_rgb(nerve) not in keys or packed_rgb(chorda) not in keys:
        raise SystemExit("nerve and chorda should keep closed surfaces")
    remesh = occupied.copy()
    remesh[closed_label_mask(occupied, colors, [rgb for _hex, rgb in keep])] = False
    union, _ = extract_block(remesh, colors, (0, 0, 0), smooth=False)
    if _face_count(union, nerve) > 0 or _face_count(union, chorda) > 0:
        raise SystemExit("carved remesh should not show nerve or chorda islands")
    if _face_count(union, bone) < 80:
        raise SystemExit("carved remesh should still have a bone surface")


def test_named_closed_structures_include_nerve_labyrinth_ossicles_chorda():
    names = set(rtx_materials.closed_structure_hexes().values())
    for wanted in (
        "facial_nerve",
        "nerve",
        "chorda_tympani",
        "malleus",
        "incus",
        "stapes",
        "scala_tympani",
        "scala_vestibuli",
    ):
        if wanted not in names:
            raise SystemExit(f"{wanted} should be a closed occupancy structure")
    if "bone" in names or "eac" in names or "sinus_dura" in names:
        raise SystemExit("bone, EAC, and sinus/dura should stay on the union remesh")


if __name__ == "__main__":
    test_buried_nerve_is_invisible_on_the_union_and_closed_on_its_own()
    test_closed_labels_skip_bone_and_eac()
    test_keep_rgb_meshes_only_the_selected_structures()
    test_named_closed_structures_include_nerve_labyrinth_ossicles_chorda()
    print("ok")
