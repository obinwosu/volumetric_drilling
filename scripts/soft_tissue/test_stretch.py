"""Stretch test: exact J-curve, initial stiffness, rupture at the weak row, work to rupture."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm
import stretch_test as st

SMALL = dict(length_mm=4.0, width_mm=1.0, edge_mm=0.25)


def test_uniform_pull_follows_the_exact_j_curve():
    p = st.skin_params()
    strip = st.Strip(p, **SMALL, fail_strain=10.0)
    x = strip.rest.copy()
    for s in (1.01, 1.1, 1.2, 1.3):
        x, f, e = strip.solve(s, x)
        assert abs(f / st.exact_force(p, s, 1.0) - 1.0) < 1e-6, s
        assert abs(e / (st.exact_energy_density(p, s) * 4.0) - 1.0) < 1e-6, s


def test_initial_stiffness_is_E_plus_k1():
    for p in (pm.SkinParams(E_mpa=1.0, nu=0.45), st.skin_params()):
        strip = st.Strip(p, **SMALL, fail_strain=10.0)
        _x, f, _e = strip.solve(1.0001, strip.rest.copy())
        assert abs(f / 1e-4 / ((p.E_mpa + p.k1_mpa) * 1.0 * p.thickness_mm) - 1.0) < 1e-3


def test_strip_snaps_at_its_weak_row_and_then_carries_nothing():
    p = st.skin_params()
    rows, strip, broken_at = st.run_stretch(p, max_stretch=1.4, step=0.01, **SMALL)
    summary = st.summarize(rows, p, strip, broken_at)
    assert summary["separated"] and abs(summary["force_after_n"]) < 1e-9
    assert summary["max_force_error_before_break"] < 1e-6
    # It breaks within one step after the failure stretch, at the weak row.
    assert summary["fail_stretch"] <= broken_at < summary["fail_stretch"] + 0.01 + 1e-9
    gone = strip.all_tris[~strip.alive]
    cent = strip.rest[gone].mean(axis=1)[:, 0]
    # Mostly the weak row; the tear may take a neighbour as it runs across.
    assert np.mean(abs(cent - strip.weak_x) < 0.125) > 0.7
    assert np.all(abs(cent - strip.weak_x) < 0.375)
    # Peak force is the exact curve at the last stretch before the break.
    assert abs(summary["peak_force_n"] / summary["exact_force_at_last_intact_n"] - 1.0) < 1e-6
    assert summary["peak_force_n"] <= summary["exact_peak_force_n"]
    # Work to rupture matches the energy stored at failure, within the step size.
    assert abs(summary["work_to_rupture_n_mm"] / summary["exact_work_to_rupture_n_mm"] - 1.0) < 0.06, summary


if __name__ == "__main__":
    test_uniform_pull_follows_the_exact_j_curve()
    test_initial_stiffness_is_E_plus_k1()
    test_strip_snaps_at_its_weak_row_and_then_carries_nothing()
    print("stretch tests passed")
