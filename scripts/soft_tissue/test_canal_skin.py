"""Canal skin sheet on a synthetic straight canal.

Bone block, 4 mm radius canal along z. Medial end opens into a middle-ear
cavity with an ossicle. The lateral 4 mm of canal has no bone around it
(cartilaginous canal).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import canal_skin

MM = 0.2
RADIUS_MM = 4.0


def _synthetic():
    shape = (90, 90, 130)
    x, y, z = np.meshgrid(*(np.arange(n) * MM for n in shape), indexing="ij")
    r = np.hypot(x - 9.0, y - 9.0)
    occupied = np.ones(shape, dtype=bool)
    colors = np.zeros(shape + (3,), dtype=np.uint8)
    colors[...] = canal_skin.BONE_RGB
    canal = (r <= RADIUS_MM) & (z >= 4.0) & (z < 24.0)
    colors[canal] = canal_skin.EAC_RGB
    # Middle ear: air medial to the canal, with an ossicle in it.
    cavity = (r <= 6.0) & (z >= 1.0) & (z < 4.0)
    occupied[cavity] = False
    ossicle = (np.hypot(x - 9.0, y - 11.0) <= 0.8) & (z >= 1.6) & (z < 3.0)
    occupied[ossicle] = True
    colors[ossicle] = (0xE9, 0x00, 0xFF)  # malleus
    # Cartilaginous canal: no bone around the lateral 4 mm, open to the outside.
    occupied[(r > RADIUS_MM) & (z >= 20.0)] = False
    occupied[z >= 24.0] = False
    return occupied, colors


def test_straight_canal_regions_normals_and_lumen():
    occupied, colors = _synthetic()
    sheet, summary = canal_skin.build_sheet(occupied, colors, np.full(3, MM))
    verts, region, normals, lumen = sheet["verts_mm"], sheet["region"], sheet["normals_mm"], sheet["lumen_mm"]
    names = canal_skin.REGIONS
    z = verts[:, 2]
    counts = summary["region_vertices"]
    assert counts["bony"] > counts["cartilage"] > 0, counts
    assert counts["annulus"] > 0, counts

    # Axis runs medial (low z, toward the ossicle) to lateral.
    assert sheet["axis"][2] > 0.95
    # Bony skin lies along the bony canal, cartilage skin lateral to it.
    bony = region == names.index("bony")
    cartilage = region == names.index("cartilage")
    annulus = region == names.index("annulus")
    assert np.median(z[bony]) < 20.0
    assert np.mean(z[cartilage] > 19.0) > 0.9
    # The annulus is the medial rim, next to the drum.
    assert np.mean(z[annulus] < 4.0 + canal_skin.ANNULUS_MM + 0.6) > 0.9
    # The drum face and the lateral opening are not skin.
    assert z.min() > 4.0 - 0.6 and z.max() < 24.0 + 0.6

    radial = verts[:, :2] - 9.0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    mid = (z > 8.0) & (z < 18.0)
    # Normals point out of the canal, into the wall.
    assert np.mean(np.einsum("ij,ij->i", normals[mid, :2], radial[mid])) > 0.9
    # The wall sits on the canal radius; the lumen is that radius less the thickness.
    wall_r = np.hypot(verts[mid, 0] - 9.0, verts[mid, 1] - 9.0)
    lumen_r = np.hypot(lumen[mid, 0] - 9.0, lumen[mid, 1] - 9.0)
    thickness = canal_skin.DEFAULT_THICKNESS_MM["bony"]
    assert abs(np.median(wall_r) - RADIUS_MM) < 0.3
    assert abs(np.median(lumen_r) - (RADIUS_MM - thickness)) < 0.3
    assert summary["mesh"]["edge_mm_p5_p50_p95"][1] < 1.5 * canal_skin.EDGE_MM


def test_thickness_is_a_setting():
    occupied, colors = _synthetic()
    thin = {name: 0.5 for name in canal_skin.REGIONS}
    sheet, _summary = canal_skin.build_sheet(occupied, colors, np.full(3, MM), thickness=thin)
    gap = np.linalg.norm(sheet["verts_mm"] - sheet["lumen_mm"], axis=1)
    beyond_taper = sheet["verts_mm"][:, 2] > 4.0 + canal_skin.TAPER_MM + 0.5
    assert np.allclose(gap[beyond_taper], 0.5)


def test_annulus_is_the_medial_rim_and_skin_tapers_to_it():
    occupied, colors = _synthetic()
    sheet, summary = canal_skin.build_sheet(occupied, colors, np.full(3, MM))
    annulus = summary["annulus"]
    # The medial rim is the 4 mm radius circle, square to the canal.
    assert abs(annulus["perimeter_mm"] - 2 * np.pi * RADIUS_MM) < 2.0, annulus
    assert abs(annulus["diameter_from_area_mm"] - 2 * RADIUS_MM) < 0.6, annulus
    assert annulus["plane_tilt_to_canal_axis_deg"] < 15.0, annulus
    loop_z = sheet["verts_mm"][sheet["annulus_loop"], 2]
    assert np.all(loop_z < 4.0 + 0.8)
    thick = sheet["thickness_mm"]
    assert np.allclose(thick[sheet["annulus_loop"]], canal_skin.ANNULUS_EDGE_MM, atol=0.02)
    z = sheet["verts_mm"][:, 2]
    halfway = (z > 4.0 + 0.4 * canal_skin.TAPER_MM) & (z < 4.0 + 0.6 * canal_skin.TAPER_MM)
    full = canal_skin.DEFAULT_THICKNESS_MM["bony"]
    assert np.all((thick[halfway] > canal_skin.ANNULUS_EDGE_MM + 0.1) & (thick[halfway] < full - 0.1))


def test_skin_never_closes_the_lumen():
    occupied, colors = _synthetic()
    thick = {name: 6.0 for name in canal_skin.REGIONS}  # wider than the canal
    sheet, summary = canal_skin.build_sheet(occupied, colors, np.full(3, MM), thickness=thick)
    z = sheet["verts_mm"][:, 2]
    mid = (z > 10.0) & (z < 18.0)
    lumen_r = np.hypot(sheet["lumen_mm"][mid, 0] - 9.0, sheet["lumen_mm"][mid, 1] - 9.0)
    assert summary["thickness"]["capped_by_clearance"] > 0
    assert np.median(lumen_r) > 0.05 * RADIUS_MM


if __name__ == "__main__":
    test_straight_canal_regions_normals_and_lumen()
    test_thickness_is_a_setting()
    test_annulus_is_the_medial_rim_and_skin_tapers_to_it()
    test_skin_never_closes_the_lumen()
    print("canal skin tests passed")
