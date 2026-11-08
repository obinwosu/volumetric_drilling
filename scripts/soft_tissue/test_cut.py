"""Cut test: incision gape, energy given up, grip force drop, and blade force."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cut_test as ct
import peel_model as pm

PARAMS = pm.SkinParams(E_mpa=1.0, nu=0.45, thickness_mm=1.0)


def test_split_keeps_tips_joined_and_sides_apart():
    plate = ct.Plate(PARAMS, 8.0, 1.0)
    plate.make_cut(-2.0, 2.0)
    # Interior cut vertices at x = -1, 0, 1 are duplicated; tips at +-2 are not.
    assert np.allclose(np.sort(plate.base_rest[plate.split, 0]), [-1.0, 0.0, 1.0])
    copies = len(plate.base_rest) + np.arange(len(plate.split))
    used_upper = np.isin(plate.tris[plate.upper], copies).any()
    used_lower = np.isin(plate.tris[~plate.upper], copies).any()
    assert used_upper and not used_lower
    assert np.all(plate.rest[plate.tris].sum(axis=1)[:, 2] == 0.0)


def test_incision_gapes_and_gives_up_energy_like_griffith():
    # Small stretch, so linear fracture theory applies.
    r = ct.run_incision(PARAMS, half_len=2.0, strain=0.002, size_mm=32.0, edge_mm=0.5)
    assert r["min_opening_mm"] > 0.0  # opens all along, no overlap
    assert 0.85 < r["max_opening_mm"] / r["expected_max_opening_mm"] < 1.0, r
    assert 0.75 < r["released_n_mm"] / r["expected_released_n_mm"] < 1.0, r
    # Held grips: the grip force drops by the same fraction as the stored energy.
    assert abs(r["force_drop_fraction"] / r["energy_drop_fraction"] - 1.0) < 0.02, r


def test_finer_mesh_releases_more_energy_toward_theory():
    ratios = [
        ct.run_incision(PARAMS, strain=0.002, size_mm=16.0, edge_mm=e)["released_n_mm"]
        / ct.run_incision(PARAMS, strain=0.002, size_mm=16.0, edge_mm=e)["expected_released_n_mm"]
        for e in (0.5, 0.25)
    ]
    assert ratios[1] > ratios[0] + 0.03, ratios


def test_energy_given_up_scales_with_tension_squared():
    lo = ct.run_incision(PARAMS, strain=0.001, size_mm=16.0, edge_mm=0.5)["released_n_mm"]
    hi = ct.run_incision(PARAMS, strain=0.002, size_mm=16.0, edge_mm=0.5)["released_n_mm"]
    assert abs(hi / lo - 4.0) < 0.05


def test_blade_force_is_toughness_less_what_tension_releases():
    Jc = 0.05
    slack = ct.run_slice(PARAMS, Jc, 0.0, start_half=1.0, end_half=3.0, size_mm=16.0, edge_mm=0.5)
    assert all(abs(r["blade_force_n"] - Jc * PARAMS.thickness_mm) < 1e-9 for r in slack)
    # Taut skin; toughness low enough that the cut eventually runs ahead.
    Jc = 0.0025
    taut = ct.run_slice(PARAMS, Jc, 0.02, start_half=1.0, end_half=3.0, size_mm=16.0, edge_mm=0.5)
    a = np.array([r["half_length_mm"] for r in taut])
    G = np.array([r["G_n_per_mm"] for r in taut])
    theory = np.array([r["G_griffith_n_per_mm"] for r in taut])
    force = np.array([r["blade_force_n"] for r in taut])
    # The mesh's alternating diagonals make G zig-zag step to step; compare trends.
    assert np.all(G > 0) and np.polyfit(a, G, 1)[0] > 0  # longer cut, more release at the tip
    assert 0.6 < G.sum() / theory.sum() < 1.0, G / theory
    assert np.polyfit(a, force, 1)[0] < 0 and force.max() < Jc * PARAMS.thickness_mm
    runs = np.array([r["runs_ahead"] for r in taut])
    assert np.array_equal(runs, G > Jc) and runs[-1] and not runs[0]


if __name__ == "__main__":
    test_split_keeps_tips_joined_and_sides_apart()
    test_incision_gapes_and_gives_up_energy_like_griffith()
    test_finer_mesh_releases_more_energy_toward_theory()
    test_energy_given_up_scales_with_tension_squared()
    test_blade_force_is_toughness_less_what_tension_releases()
    print("cut tests passed")
