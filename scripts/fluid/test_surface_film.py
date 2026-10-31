"""The irrigation film fills a cavity to the lip, then runs off."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.occupancy import MM_TO_SIM
from fluid.surface_film import GRAVITY_SLOPE, MAX_DEPTH_MM, VISUAL_SUBDIV, SurfaceFilm


def _tilted_cup():
    film = SurfaceFilm(cell_mm=1.0, span_mm=21.0)
    film.place(np.zeros(3), np.array((0.0, 0.0, 1.0)))
    n = film.n
    ii, jj = np.mgrid[0:n, 0:n]
    x = ii - (n - 1) / 2.0
    y = jj - (n - 1) / 2.0
    radius = np.hypot(x, y)
    bed = np.full((n, n), np.nan)
    floor = radius <= 5.0
    rim = (radius > 5.0) & (radius <= 7.0)
    bed[floor] = 0.25 * x[floor]
    bed[rim] = 2.2
    film.bed = bed
    film.sampled[:] = True
    return film, x, floor


def _inlet(film, x_mm, y_mm):
    i = int(np.rint(x_mm + (film.n - 1) / 2.0))
    j = int(np.rint(y_mm + (film.n - 1) / 2.0))
    return film._world(i, j, 0.0)


def _pour(film, hit, volume_mm3, step_mm3=1.0):
    left = float(volume_mm3)
    while left > 1e-6:
        film.deposit(hit, 0.58, min(step_mm3, left))
        film.settle(6)
        left -= step_mm3


def test_pool_spreads_past_the_jet_and_settles_downhill():
    film, x, floor = _tilted_cup()
    _pour(film, _inlet(film, 3.0, 0.0), 40.0)
    wet = film.depth > 0.05
    span = float(np.max(np.abs(x[wet])) * 2.0) if np.any(wet) else 0.0
    assert span > 0.58
    low = floor & (x < -1.0)
    high = floor & (x > 1.0)
    assert film.depth[low].sum() > film.depth[high].sum()
    assert film.spill_mm3 < 1.0


def test_extra_water_spills_at_the_lip_instead_of_growing():
    film, _x, floor = _tilted_cup()
    _pour(film, _inlet(film, 0.0, 0.0), 400.0, step_mm3=4.0)
    assert film.spill_mm3 > 50.0
    assert float(film.depth.max()) < 5.0
    assert float(film.depth[floor].max()) > 2.0


def test_closed_pit_stops_at_the_thickness_cap():
    film = SurfaceFilm(cell_mm=1.0, span_mm=15.0)
    film.place(np.zeros(3), np.array((0.0, 0.0, 1.0)))
    n = film.n
    ii, jj = np.mgrid[0:n, 0:n]
    x = ii - (n - 1) / 2.0
    y = jj - (n - 1) / 2.0
    radius = np.hypot(x, y)
    bed = np.full((n, n), np.nan)
    bed[radius <= 3.0] = 0.0
    bed[(radius > 3.0) & (radius <= 5.0)] = 30.0
    film.bed = bed
    film.sampled[:] = True
    _pour(film, film._world(n // 2, n // 2, 0.0), 400.0, step_mm3=8.0)
    assert float(film.depth.max()) <= MAX_DEPTH_MM + 1e-6
    assert film.spill_mm3 > 0.0
    assert film.volume_mm3 <= film.n * film.n * MAX_DEPTH_MM + 1e-3


def test_suction_takes_the_pool_and_leaves_the_bone_wet():
    film, _x, _floor = _tilted_cup()
    hit = _inlet(film, -3.0, 0.0)
    film.deposit(hit, 0.58, 30.0)
    film.settle()
    assert film.any_wet
    assert film.volume_mm3 > 5.0
    film.suction(hit, 3.0)
    assert film.suction_mm3 > 0.0
    assert film.covers(np.array([hit]))[0]
    assert film.any_wet


def _face_normals(mesh):
    verts = mesh["verts"]
    tris = mesh["tris"]
    edge_u = verts[tris[:, 1]] - verts[tris[:, 0]]
    edge_v = verts[tris[:, 2]] - verts[tris[:, 0]]
    geo = np.cross(edge_u, edge_v)
    length = np.linalg.norm(geo, axis=1, keepdims=True)
    return geo / np.maximum(length, 1e-8)


def test_coating_follows_the_slope_instead_of_facing_the_camera():
    film, _x, _floor = _tilted_cup()
    _pour(film, _inlet(film, -2.0, 0.0), 25.0)
    mesh = film.sheet_mesh()
    assert mesh is not None
    assert len(mesh["tris"]) >= 2 * VISUAL_SUBDIV * VISUAL_SUBDIV
    normals = mesh["normals"]
    facing = normals @ film.down
    assert np.all(facing < -0.2)
    # A slope tips the coating. Camera-facing tiles would all share one normal.
    assert float(normals.std(axis=0).sum()) > 0.01
    # The triangle itself points at the camera. The old winding pointed into the bone.
    assert np.all(_face_normals(mesh) @ film.down < -0.2)


def test_sheet_smooths_voxel_stairs():
    film = SurfaceFilm(cell_mm=1.0, span_mm=16.0)
    film.place(np.zeros(3), np.array((0.0, 0.0, 1.0)))
    film.bed[:] = 0.0
    film.bed[::2, :] = 1.0
    film.sampled[:] = True
    film.depth[4:12, 4:12] = 0.4
    mesh = film.sheet_mesh()
    assert mesh is not None
    elev = -((mesh["verts"] - film.origin) @ film.down) / MM_TO_SIM
    tris = mesh["tris"]
    jumps = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        jumps.append(np.abs(elev[tris[:, a]] - elev[tris[:, b]]))
    # A raw 1 mm stair would leave an edge that tall. Smoothing knocks it down.
    assert float(np.max(np.concatenate(jumps))) < 0.55


def _flat(span_mm=21.0):
    film = SurfaceFilm(cell_mm=1.0, span_mm=span_mm)
    film.place(np.zeros(3), np.array((0.0, 0.0, 1.0)))
    film.bed[:] = 0.0
    film.sampled[:] = True
    return film


def test_flat_bone_runs_toward_the_bottom_of_the_image():
    film = _flat()
    film._rebuild_level(film.across)
    mid = film.n // 2
    _pour(film, film._world(mid, mid, 0.0), 40.0, step_mm3=2.0)
    wet = np.argwhere(film.depth > 0.05)
    assert len(wet) > 0
    assert float(wet[:, 1].mean()) > mid + 1.0
    assert film.spill_mm3 > 1.0


def test_deep_hole_holds_water_and_a_shallow_dent_drains():
    deep = _flat()
    n = deep.n
    ii, jj = np.mgrid[0:n, 0:n]
    y = (jj - (n - 1) / 2.0) * deep.cell_mm
    x = (ii - (n - 1) / 2.0) * deep.cell_mm
    pit = np.hypot(x, y) <= 3.0
    deep.bed[pit] = -4.0
    deep._rebuild_level(deep.across)
    _pour(deep, deep._world(n // 2, n // 2, 0.0), 30.0, step_mm3=2.0)
    assert float(deep.depth[pit].sum()) > 10.0
    assert deep.spill_mm3 < 5.0
    # The downhill lip of a 0.15 mm dent is lower than its floor, so it does not hold.
    shallow = _flat()
    dent = np.hypot(x, y) <= 5.0
    shallow.bed[dent] = -0.15
    shallow._rebuild_level(shallow.across)
    lip = GRAVITY_SLOPE * 5.0
    assert lip > 0.15
    _pour(shallow, shallow._world(n // 2, n // 2, 0.0), 20.0, step_mm3=2.0)
    downhill = dent & (y > 1.0)
    uphill = dent & (y < -1.0)
    assert shallow.depth[downhill].sum() + shallow.spill_mm3 > shallow.depth[uphill].sum()


def test_wet_bone_stays_marked_after_the_water_leaves():
    film, _x, _floor = _tilted_cup()
    hit = _inlet(film, 0.0, 0.0)
    film.deposit(hit, 1.0, 8.0)
    marked = film.wet.copy()
    film.depth[:] = 0.0
    film.settle()
    assert np.array_equal(film.wet, marked)
    assert film.covers(np.array([hit]))[0]


if __name__ == "__main__":
    test_pool_spreads_past_the_jet_and_settles_downhill()
    test_extra_water_spills_at_the_lip_instead_of_growing()
    test_closed_pit_stops_at_the_thickness_cap()
    test_suction_takes_the_pool_and_leaves_the_bone_wet()
    test_coating_follows_the_slope_instead_of_facing_the_camera()
    test_sheet_smooths_voxel_stairs()
    test_flat_bone_runs_toward_the_bottom_of_the_image()
    test_deep_hole_holds_water_and_a_shallow_dent_drains()
    test_wet_bone_stays_marked_after_the_water_leaves()
    print("surface film ok", flush=True)
