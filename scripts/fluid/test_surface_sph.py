"""The surface sheet stays thin, spills at the downhill lip, and draws as one film."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.occupancy import MM_TO_SIM, identity_frame
from fluid.surface_sph import SurfaceSph, film_surface


def _island(depth_voxels=0, pit=None):
    counts = np.array((36, 36, 24), dtype=np.float64)
    voxel_mm = 0.5
    frame = identity_frame(counts, counts * voxel_mm * MM_TO_SIM)
    occupied = np.zeros((36, 36, 24), dtype=bool)
    occupied[8:28, 8:28, :12] = True
    if depth_voxels > 0:
        x0, x1, y0, y1 = pit or (16, 20, 16, 20)
        occupied[x0:x1, y0:y1, 12 - depth_voxels : 12] = False
    return occupied, frame


def _pour(sph, occupied, frame, hit, seconds, jet_mm=0.58):
    view = np.array((0.0, 0.0, -1.0))
    gravity = np.array((0.0, 1.0, 0.0))
    steps = max(1, int(round(seconds / 0.05)))
    for _ in range(steps):
        sph.step(0.05, hit, jet_mm, view, gravity, frame, occupied)
    return view, gravity


def _settle(sph, occupied, frame, seconds):
    view = np.array((0.0, 0.0, -1.0))
    gravity = np.array((0.0, 1.0, 0.0))
    steps = max(1, int(round(seconds / 0.05)))
    for _ in range(steps):
        sph.step(0.05, None, 0.58, view, gravity, frame, occupied)
    return view


def test_open_bone_stays_thin_and_runs_off():
    occupied, frame = _island()
    sph = SurfaceSph(max_particles=256)
    hit = frame.index_to_world(np.array((18.0, 10.0, 12.0)))
    view, _gravity = _pour(sph, occupied, frame, hit, 0.2)
    assert sph.count > 4
    gaps = sph.gaps_mm(view, occupied, frame)
    finite = gaps[np.isfinite(gaps)]
    assert len(finite) > 0
    assert float(np.max(finite)) < 0.8
    spawned = sph.spawned
    _settle(sph, occupied, frame, 1.2)
    assert sph.spill > 0
    assert sph.count < spawned


def test_cavity_holds_until_the_downhill_lip():
    occupied, frame = _island(depth_voxels=4, pit=(14, 22, 14, 22))
    sph = SurfaceSph(max_particles=128)
    hit = frame.index_to_world(np.array((18.0, 18.0, 8.0)))
    _pour(sph, occupied, frame, hit, 0.25)
    _settle(sph, occupied, frame, 0.4)
    assert sph.count > 4
    assert sph.spill == 0
    index = frame.world_to_index(sph.positions())
    assert np.all((index[:, 0] > 13.0) & (index[:, 0] < 23.0))
    assert np.all((index[:, 1] > 13.0) & (index[:, 1] < 23.0))


def test_extra_water_spills_and_the_list_does_not_grow():
    occupied, frame = _island(depth_voxels=2, pit=(16, 20, 16, 20))
    sph = SurfaceSph(max_particles=48)
    hit = frame.index_to_world(np.array((18.0, 18.0, 10.0)))
    _pour(sph, occupied, frame, hit, 2.0)
    _settle(sph, occupied, frame, 0.8)
    assert sph.spawned > sph.max_particles
    assert sph.count <= sph.max_particles
    assert sph.spill > 0
    assert sph.count < sph.spawned


def test_a_new_cut_pulls_water_into_the_floor():
    occupied, frame = _island()
    sph = SurfaceSph(max_particles=64)
    # Uphill of center, so the sheet is still on the island when the floor drops.
    hit = frame.index_to_world(np.array((18.0, 11.0, 12.0)))
    view = _pour(sph, occupied, frame, hit, 0.15)[0]
    assert sph.count > 4
    before = float(frame.world_to_index(sph.positions())[:, 2].mean())
    occupied[12:24, 12:26, 8:12] = False
    _settle(sph, occupied, frame, 0.35)
    assert sph.count > 0
    after = float(frame.world_to_index(sph.positions())[:, 2].mean())
    assert after < before - 0.5
    gaps = sph.gaps_mm(view, occupied, frame)
    finite = gaps[np.isfinite(gaps)]
    assert len(finite) > 0
    assert float(np.max(finite)) < 1.2


def test_suction_deletes_particles_without_reusing_a_live_slot():
    occupied, frame = _island()
    sph = SurfaceSph(max_particles=32)
    hit = frame.index_to_world(np.array((18.0, 18.0, 12.0)))
    _pour(sph, occupied, frame, hit, 0.4)
    assert sph.count > 0
    before = sph.count
    sph.suction(hit, 4.0)
    assert sph.suction_count > 0
    assert sph.count < before
    # A full list still refuses to overwrite whoever is left.
    sph.step(0.2, hit, 0.58, np.array((0.0, 0.0, -1.0)), np.array((0.0, 1.0, 0.0)), frame, occupied)
    assert sph.count <= sph.max_particles


def test_film_is_one_smooth_sheet_and_a_speck_is_invisible():
    assert film_surface(np.zeros((3, 3)), np.array((0.0, 0.0, -1.0))) is None
    grid = np.linspace(-3.0, 3.0, 14)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    sheet = np.stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)), axis=1) * MM_TO_SIM
    view = np.array((0.0, 0.0, -1.0))
    mesh = film_surface(sheet, view)
    assert mesh is not None
    assert len(mesh["tris"]) > 20
    depth = mesh["verts"] @ view
    assert float(depth.max() - depth.min()) < 0.4 * MM_TO_SIM
    facing = mesh["normals"] @ view
    assert np.all(facing < -0.2)
    edge_u = mesh["verts"][mesh["tris"][:, 1]] - mesh["verts"][mesh["tris"][:, 0]]
    edge_v = mesh["verts"][mesh["tris"][:, 2]] - mesh["verts"][mesh["tris"][:, 0]]
    geo = np.cross(edge_u, edge_v)
    geo /= np.maximum(np.linalg.norm(geo, axis=1, keepdims=True), 1e-8)
    assert np.all(geo @ view < -0.2)


if __name__ == "__main__":
    test_open_bone_stays_thin_and_runs_off()
    test_cavity_holds_until_the_downhill_lip()
    test_extra_water_spills_and_the_list_does_not_grow()
    test_a_new_cut_pulls_water_into_the_floor()
    test_suction_deletes_particles_without_reusing_a_live_slot()
    test_film_is_one_smooth_sheet_and_a_speck_is_invisible()
    print("surface sph ok", flush=True)
