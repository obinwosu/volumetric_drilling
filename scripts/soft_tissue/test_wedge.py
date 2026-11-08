"""Elevator wedge: energy, push force, sharp edge, and beam-theory lift-off."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import wedge_test as wt


def _run(case, length=9.0, travel=6.0, tab=1.0):
    params, iface, blade = case
    rows, _snaps, rig = wt.run_wedge(params, iface, blade, length_mm=length, tab_mm=tab, travel_mm=travel, verbose=False)
    return wt.summarize(rows, params, iface, blade, 0.4, tab, length), rows, rig


def test_push_force_is_the_energy_slope_before_anything_breaks():
    # Blade lifts only the loose tab; bonds stretch but none fail.
    summary, rows, rig = _run(wt.default_case(), length=6.0, travel=1.5, tab=3.0)
    assert not rig.broken.any()
    assert rows[-1]["contact_work_n_mm"] > 0.0
    assert abs(summary["energy_balance_ratio"] - 1.0) < 0.01, summary["energy_balance_ratio"]


def test_thin_blunt_blade_pushes_with_adhesion_energy_times_width():
    summary, _rows, rig = _run(wt.thin_case())
    assert summary["steady_steps"] >= 50
    assert abs(summary["push_over_G_I"] - 1.0) < 0.05, summary
    assert abs(summary["energy_balance_ratio"] - 1.0) < 0.03, summary
    # Lifting frees the tissue: every bond failed by opening.
    assert np.all(rig.hist_n[rig.broken] >= rig.iface.opening.delta_c_mm)
    assert abs(summary["front_speed"] - 1.0) < 0.05


def test_sharp_edge_frees_tissue_at_the_tip_for_its_cutting_cost():
    summary, _rows, _rig = _run(wt.thin_case(sharp=True, G_edge=0.015))
    blade = wt.thin_case(sharp=True)[2]
    assert 0.0 <= summary["steady_lead_mm"] <= blade.edge_reach_mm + 0.2 + 1e-9, summary
    assert abs(summary["push_over_G_edge"] - 1.0) < 0.12, summary
    assert summary["push_over_G_I"] < 0.4  # far easier than the blunt blade


def test_thick_blade_lifts_tissue_off_where_beam_theory_says():
    summary, _rows, _rig = _run(wt.default_case(), length=8.0, travel=5.0)
    assert summary["steady_steps"] >= 50
    pred = summary["boeing_liftoff_lead_mm"]
    assert abs(summary["steady_liftoff_lead_mm"] - pred) < 0.15 * (pred + summary["ramp_mm"]), summary
    assert abs(summary["front_speed"] - 1.0) < 0.05
    # Energy is never created; a little is lost as each bond row lets go.
    assert 0.99 <= summary["energy_balance_ratio"] < 1.06, summary
    assert abs(summary["push_over_G_I"] - 1.0) < 0.08, summary


if __name__ == "__main__":
    test_push_force_is_the_energy_slope_before_anything_breaks()
    test_thin_blunt_blade_pushes_with_adhesion_energy_times_width()
    test_sharp_edge_frees_tissue_at_the_tip_for_its_cutting_cost()
    test_thick_blade_lifts_tissue_off_where_beam_theory_says()
    print("wedge tests passed")
