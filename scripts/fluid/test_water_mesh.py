"""The irrigation sheet fuses neighbors and ignores a lone speck."""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.outlet import MM_TO_SIM
from fluid.water_mesh import water_surface


def _mm(points):
    return np.asarray(points, dtype=np.float64) * MM_TO_SIM


def test_one_particle_has_no_surface():
    assert water_surface(_mm([[0.0, 0.0, 0.0]])) is None


def test_separated_specks_have_no_surface():
    specks = [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    assert water_surface(_mm(specks)) is None


def test_packed_column_is_a_long_body():
    length = np.arange(0.0, 25.0, 0.4)
    column = np.stack((length, np.zeros_like(length), np.zeros_like(length)), axis=1)
    mesh = water_surface(_mm(column))
    assert mesh is not None
    span = mesh["verts"].max(axis=0) - mesh["verts"].min(axis=0)
    assert span[0] > 15.0 * MM_TO_SIM
    assert span[1] < span[0] * 0.5
    # A line of centers should stay a narrow stream, not a burr-width tube.
    assert span[1] < 2.5 * MM_TO_SIM
    assert len(mesh["tris"]) > 8


def test_sheet_and_ball_face_outward():
    grid = np.linspace(-6.0, 6.0, 24)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    sheet = np.stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)), axis=1)
    sheet_mesh = water_surface(_mm(sheet))
    assert sheet_mesh is not None
    assert len(sheet_mesh["tris"]) > 20

    ball = []
    for x in np.linspace(-2.0, 2.0, 9):
        for y in np.linspace(-2.0, 2.0, 9):
            for z in np.linspace(-2.0, 2.0, 9):
                if x * x + y * y + z * z <= 4.0:
                    ball.append((x, y, z))
    mesh = water_surface(_mm(ball))
    assert mesh is not None
    center = mesh["verts"].mean(axis=0)
    outward = np.einsum("ij,ij->i", mesh["normals"], mesh["verts"] - center)
    assert float(np.mean(outward)) > 0.0


def _time(label, points, repeats=3):
    water_surface(points)
    started = time.perf_counter()
    last = None
    for _ in range(repeats):
        last = water_surface(points)
    elapsed = (time.perf_counter() - started) * 1e3 / repeats
    tris = 0 if last is None else len(last["tris"])
    cells = None if last is None else last["cells"]
    print(f"{label}: {elapsed:.1f} ms, {len(points)} particles, {tris} tris, cells {cells}")
    return elapsed


if __name__ == "__main__":
    test_one_particle_has_no_surface()
    test_separated_specks_have_no_surface()
    test_packed_column_is_a_long_body()
    test_sheet_and_ball_face_outward()
    length = np.arange(0.0, 25.0, 0.4)
    radius = np.array([[0.0, 0.0], [0.35, 0.0], [-0.35, 0.0], [0.0, 0.35], [0.0, -0.35]])
    column = np.concatenate(
        [np.stack((length, np.full_like(length, y), np.full_like(length, z)), axis=1) for y, z in radius],
        axis=0,
    )
    _time("column", _mm(column))
    grid = np.linspace(-15.0, 15.0, 50)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    sheet = np.stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)), axis=1)
    _time("sheet", _mm(sheet))
    rng = np.random.default_rng(0)
    disk = rng.normal(scale=4.0, size=(6000, 3))
    disk[:, 2] *= 0.2
    _time("pool", _mm(disk))
    scattered = rng.uniform(-40.0, 40.0, size=(8000, 3))
    _time("scattered", _mm(scattered))
    print("water mesh ok")
