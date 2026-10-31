"""Cannula nozzle, clip station, and the jet that leaves that nozzle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fluid.cannula import (
    CLIP_MM,
    MESH_MM,
    SIZES,
    cannula_frame,
    load_obj,
    mesh_path,
)
from fluid.cannula import _radius_at, _shaft_curve


def _clip_mm(points) -> float:
    x = points[:, 0] / MESH_MM
    best = None
    for station in range(0, int(x.max()) + 1, 2):
        chosen = points[(x >= station) & (x < station + 2)]
        if len(chosen) < 20:
            continue
        span = float(chosen[:, 2].max() - chosen[:, 2].min()) / MESH_MM
        if span > 8.0:
            best = station + 1.0
    if best is None:
        raise AssertionError("no clip")
    return best


def test_each_size_has_its_own_metal_mesh():
    for size in SIZES:
        path = mesh_path(size)
        text = path.read_text(errors="replace")
        assert "usemtl" not in text
        points, tris = load_obj(path)
        assert tris.max() < len(points)
        assert _clip_mm(points) == CLIP_MM[size] or abs(_clip_mm(points) - CLIP_MM[size]) <= 2.0
        opening = cannula_frame(np.zeros(3), (0.0, 0.0, 0.0, 1.0), size)["outlet"]
        assert abs(opening[0] / MESH_MM - 7.5) < 0.3


def test_tube_stays_outside_the_shaft():
    for size in SIZES:
        points, _tris = load_obj(mesh_path(size))
        curve = _shaft_curve(size)
        radial = np.linalg.norm(points[:, 1:], axis=1)
        shaft = _radius_at(curve, points[:, 0])
        deepest_mm = float(np.max((shaft - radial) / MESH_MM))
        assert deepest_mm < 0.2, size


def test_jet_clears_the_burr_and_lands_just_past_it():
    tip = np.array((0.2, -0.4, 0.1))
    quat = (0.0, 0.0, 0.0, 1.0)
    for size in SIZES:
        frame = cannula_frame(tip, quat, size)
        assert np.allclose(frame["outlet"] - tip, cannula_frame(np.zeros(3), quat, size)["outlet"])
        travel = float(np.dot(frame["target"] - frame["outlet"], frame["emit"]))
        assert travel > 0.0
        closest = frame["outlet"] + frame["emit"] * max(0.0, -float(np.dot(frame["outlet"] - tip, frame["emit"])))
        gap = float(np.linalg.norm(closest - tip))
        assert gap > frame["radius"]
        assert gap < frame["radius"] + 2.0 * 0.01007


def test_pose_carries_the_nozzle():
    identity = cannula_frame(np.zeros(3), (0.0, 0.0, 0.0, 1.0), 6)
    half = 0.5 * np.pi
    turned = cannula_frame(np.zeros(3), (0.0, 0.0, np.sin(half / 2.0), np.cos(half / 2.0)), 6)
    # 90 degrees about Z sends local +X to world +Y.
    assert abs(turned["outlet"][1] - identity["outlet"][0]) < 1e-6
    assert abs(turned["outlet"][0] + identity["outlet"][1]) < 1e-6
    shifted = cannula_frame(np.array((1.0, 2.0, 3.0)), (0.0, 0.0, 0.0, 1.0), 2)
    base = cannula_frame(np.zeros(3), (0.0, 0.0, 0.0, 1.0), 2)
    assert np.allclose(shifted["outlet"], base["outlet"] + np.array((1.0, 2.0, 3.0)))


if __name__ == "__main__":
    test_each_size_has_its_own_metal_mesh()
    test_tube_stays_outside_the_shaft()
    test_jet_clears_the_burr_and_lands_just_past_it()
    test_pose_carries_the_nozzle()
    print("cannula tests passed")
