"""Peel model: gradients, Hessian, bending scale, bond law, and a short 90-degree peel."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm
import peel_strip as ps


def _messy_sheet(seed=0):
    rest, tris, _idx = pm.strip_mesh(3.0, 1.0, 0.5)
    sheet = pm.Sheet(rest, tris, np.arange(len(rest)), pm.SkinParams())
    rng = np.random.default_rng(seed)
    x = pm.roll_about_y(rest, 2.0) + rng.normal(0, 0.08, rest.shape)
    x[:4, 2] -= 0.2  # some vertices through the plate
    sheet.history = rng.uniform(0, 0.3, len(rest))  # bonds rising, softening and broken
    return sheet, x


def test_every_gradient_matches_finite_differences():
    sheet, x = _messy_sheet()
    for name in ("_membrane", "_bending", "_bonds", "_contact"):
        term = getattr(sheet, name)
        _e, grad = term(x)
        num = np.zeros_like(x)
        for i in range(len(x)):
            for k in range(3):
                plus, minus = x.copy(), x.copy()
                plus[i, k] += 1e-6
                minus[i, k] -= 1e-6
                num[i, k] = (term(plus)[0] - term(minus)[0]) / 2e-6
        err = np.abs(num - grad).max() / max(np.abs(num).max(), 1e-12)
        assert err < 1e-6, (name, err)


def test_coloured_hessian_matches_dense():
    sheet, x = _messy_sheet(1)
    sparse_h = pm.hessian(sheet, x).toarray()
    n = x.size
    dense = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1e-6
        dense[:, j] = (sheet.energy((x.ravel() + e).reshape(-1, 3))[1].ravel()
                       - sheet.energy((x.ravel() - e).reshape(-1, 3))[1].ravel()) / 2e-6
    dense = 0.5 * (dense + dense.T)
    assert np.abs(sparse_h - dense).max() <= 1e-9 * np.abs(dense).max()


def test_cylinder_stores_plate_bending_energy():
    rest, tris, _idx = pm.strip_mesh(6.0, 2.0, 0.25)
    sheet = pm.Sheet(rest, tris, np.arange(len(rest)), pm.SkinParams())
    for radius in (2.0, 5.0):
        energy = sheet._bending(pm.roll_about_y(rest, radius))[0]
        target = 0.5 * sheet.params.bending_D / radius**2 * sheet.area.sum()
        assert abs(energy / target - 1.0) < 0.03, (radius, energy / target)


def test_bond_law_area_is_adhesion_energy_and_damage_does_not_heal():
    p = pm.SkinParams()
    d = np.linspace(0, p.delta_c_mm * 1.2, 20001)
    area = np.trapezoid(pm.traction(d, p), d)
    assert abs(area / p.G_n_per_mm - 1.0) < 1e-3
    assert abs(pm.envelope_energy(p.delta_c_mm * 2, p) - p.G_n_per_mm) < 1e-12
    # After opening to the middle of the softening branch, a smaller opening
    # pulls back along the secant, weaker than the fresh bond.
    rest, tris, _idx = pm.strip_mesh(1.0, 1.0, 1.0)
    sheet = pm.Sheet(rest, tris, np.array([0]), p)
    mid = 0.5 * (p.delta0_mm + p.delta_c_mm)
    sheet.history[:] = mid
    x = rest.copy()
    x[0, 2] = 0.5 * mid
    pull = sheet._bonds(x)[1][0, 2] / sheet.vert_area[0]
    assert abs(pull - pm.traction(mid, p) * 0.5) < 1e-9
    assert pull < pm.traction(0.5 * mid, p)


def test_short_peel_reaches_kendall_force():
    params = pm.SkinParams()
    rows, _snaps, _sheet = ps.run_peel(params, length_mm=16.0, width_mm=1.0, edge_mm=0.25, tab_mm=4.0,
                                       lift_mm=11.0, step_mm=0.25, verbose=False)
    summary = ps.summarize(rows, params, 1.0, 4.0, 16.0)
    assert summary["steady_steps"] >= 5, summary
    assert abs(summary["force_ratio"] - 1.0) < 0.08, summary
    assert abs(summary["front_advance_per_mm_lift"] - 1.0) < 0.15, summary
    # A compliant bond rounds the front a little, so it curls less than the rigid limit.
    assert 0.6 < summary["curvature_ratio"] < 1.1, summary


if __name__ == "__main__":
    test_every_gradient_matches_finite_differences()
    test_coloured_hessian_matches_dense()
    test_cylinder_stores_plate_bending_energy()
    test_bond_law_area_is_adhesion_energy_and_damage_does_not_heal()
    test_short_peel_reaches_kendall_force()
    print("peel model tests passed")
