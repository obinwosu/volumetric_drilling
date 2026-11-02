"""Nerve coordinates and the numpy mirror of Nerve.mdl."""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nerves


def _split_into_blocks(points, tris):
    """Two halves with repeated seam vertices, as the block mesher returns them."""
    half = len(tris) // 2
    doubled = np.concatenate((points, points))
    return doubled, np.concatenate((tris[:half], tris[half:] + len(points)))


def test_bent_tube_coordinates():
    points, tris, centre = nerves._tube()
    doubled, split = _split_into_blocks(points, tris)
    coords = nerves.nerve_coordinates(doubled, split)
    length = float(np.sum(np.linalg.norm(np.diff(centre, axis=0), axis=1)))
    span = float(np.ptp(coords[:, 0]))
    if abs(span - length) > 0.1 * length:
        raise SystemExit(f"along spans {span:.2f} mm on a {length:.2f} mm tube")
    radius = np.hypot(coords[:, 1], coords[:, 2])
    if abs(float(np.median(radius)) - 0.6) > 0.05:
        raise SystemExit(f"radius {float(np.median(radius)):.2f} mm on a 0.60 mm tube")
    if not np.all(coords[:, 3] == 1.0):
        raise SystemExit("every vertex on the tube should be valid")


def test_along_is_smooth_across_edges():
    points, tris, _centre = nerves._tube()
    coords = nerves.nerve_coordinates(points, tris)
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    change = np.abs(coords[edges[:, 0], 0] - coords[edges[:, 1], 0])
    if float(np.max(change / np.maximum(length, 1e-9))) > 1.6:
        raise SystemExit("along jumps across an edge")


def test_separate_pieces_get_their_own_line():
    points, tris, _centre = nerves._tube(length_mm=8.0)
    far = points + np.array([40.0, 0.0, 0.0])
    coords = nerves.nerve_coordinates(np.concatenate((points, far)), np.concatenate((tris, tris + len(points))))
    radius = np.hypot(coords[:, 1], coords[:, 2])
    if float(radius.max()) > 1.0:
        raise SystemExit("a second piece was measured from the first piece's line")


def test_tiny_pieces_stay_plain():
    tris = np.array([[0, 1, 2]])
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    coords = nerves.nerve_coordinates(points, tris)
    if np.any(coords[:, 3] != 0.0):
        raise SystemExit("a three-vertex speck should not get vessels")


def _surface_grid(radius=0.6, length=10.0, rings=500, around=120):
    s = np.linspace(0.0, length, rings)
    theta = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    ss, tt = np.meshgrid(s, theta, indexing="ij")
    grid = np.stack((ss, radius * np.cos(tt), radius * np.sin(tt), np.ones_like(ss)), axis=-1)
    return grid.reshape(-1, 4), rings, around


def test_vessels_run_lengthwise_and_switch_off():
    grid, rings, around = _surface_grid()
    params = dict(nerves.DEFAULTS)
    main, branch = nerves.vessel_amounts(grid, params)
    if not 0.002 < float((main > 0.5).mean()) < 0.2:
        raise SystemExit(f"main vessel coverage {float((main > 0.5).mean()):.3f}")
    if not 0.002 < float((branch > 0.3).mean()) < 0.25:
        raise SystemExit(f"branch coverage {float((branch > 0.3).mean()):.3f}")
    field = (main + branch).reshape(rings, around)
    per_mm_along = np.mean(np.abs(np.diff(field, axis=0))) / (10.0 / (rings - 1))
    per_mm_around = np.mean(np.abs(np.diff(field, axis=1))) / (2.0 * np.pi * 0.6 / around)
    if per_mm_along >= per_mm_around:
        raise SystemExit("vessels should change faster around the nerve than along it")
    off_main, off_branch = nerves.vessel_amounts(grid, dict(params, vessels_enabled=False))
    if float(off_main.max()) != 0.0 or float(off_branch.max()) != 0.0:
        raise SystemExit("vessels still show when switched off")


def test_wider_setting_gives_wider_vessels():
    grid, _rings, _around = _surface_grid()
    params = dict(nerves.DEFAULTS, main_continuity=1.0)
    thin, _ = nerves.vessel_amounts(grid, dict(params, main_width_mm=0.04))
    wide, _ = nerves.vessel_amounts(grid, dict(params, main_width_mm=0.12))
    if float((wide > 0.5).mean()) < 2.0 * float((thin > 0.5).mean()):
        raise SystemExit("tripling the width should at least double main coverage")


def test_shader_inputs_match_the_mdl():
    mdl = (HERE / "mdl" / "Nerve.mdl").read_text()
    declared = set(re.findall(r"uniform (?:float|bool|color) (\w+) =", mdl))
    inputs = nerves.shader_inputs(nerves.DEFAULTS, nerves.MM_TO_SIM)
    missing = set(inputs) - declared
    if missing:
        raise SystemExit(f"inputs the shader does not declare: {sorted(missing)}")
    unset = declared - set(inputs)
    if unset:
        raise SystemExit(f"shader inputs the panel never sets: {sorted(unset)}")
    for key in ("main_count", "main_length_mm", "branch_count", "branch_length_mm", "main_continuity"):
        value = re.search(rf"uniform float {key} = ([0-9.]+)", mdl).group(1)
        if abs(float(value) - float(nerves.DEFAULTS[key])) > 1e-6:
            raise SystemExit(f"{key} default differs between Nerve.mdl and nerves.py")
    if abs(inputs["subsurface_scale"] - nerves.DEFAULTS["subsurface_mm"] * nerves.MM_TO_SIM) > 1e-9:
        raise SystemExit("subsurface depth should convert from mm to scene units")


def test_nerve_names():
    if not nerves.is_nerve("anatomy/facial_nerve") or not nerves.is_nerve("chorda_tympani"):
        raise SystemExit("facial nerve and chorda should take the nerve shader")
    if nerves.is_nerve("anatomy/iac") or nerves.is_nerve("cochleovestibular_nerve"):
        raise SystemExit("IAC and the cochleovestibular nerve are not part of this shader")


if __name__ == "__main__":
    test_bent_tube_coordinates()
    test_along_is_smooth_across_edges()
    test_separate_pieces_get_their_own_line()
    test_tiny_pieces_stay_plain()
    test_vessels_run_lengthwise_and_switch_off()
    test_wider_setting_gives_wider_vessels()
    test_shader_inputs_match_the_mdl()
    test_nerve_names()
    print("ok")
