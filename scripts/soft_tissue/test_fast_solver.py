"""Fast C++ solvers: same energies as the reference, exact static answers, Kendall peel."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fast_solver as fs
import peel_model as pm
import peel_strip as ps
import solver_race as race
import stretch_test as st
from test_peel_model import _messy_sheet


def test_energy_and_gradient_match_the_reference():
    sheet, x = _messy_sheet()
    m = fs.FastModel.from_sheet(sheet)
    # Give the C++ bonds the same damage history as the reference sheet.
    opened = sheet.rest.copy()
    opened[sheet.bonded, 2] += sheet.history
    m.commit(opened)
    for ref, fast, state in (
        (sheet, m, x),
        (pm.Sheet(*pm.strip_mesh(3, 1, 0.5)[:2], np.zeros(0, dtype=np.int64), st.skin_params(), floor_z=-1e9), None,
         None),
    ):
        if fast is None:
            fast = fs.FastModel.from_sheet(ref)
            state = ref.rest * [1.25, 0.9, 1.0] + np.random.default_rng(1).normal(0, 0.05, ref.rest.shape)
        e_ref, g_ref = ref.energy(state)
        e_fast, g_fast = fast.energy(state)
        assert abs(e_fast - e_ref) <= 1e-12 * abs(e_ref)
        assert np.abs(g_fast - g_ref).max() <= 1e-12 * np.abs(g_ref).max()


def test_tet_energy_is_exact_for_uniform_stretch_and_gradient_matches():
    rest, tets = fs.box_tets(2.0, 1.0, 1.0, 0.5)
    E, nu = 1.0, 0.45
    m = fs.FastModel.from_tets(rest, tets, E, nu)
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    s = 1.2
    ex = 0.5 * (s * s - 1.0)
    ey = -nu * ex
    lateral = np.sqrt(1.0 + 2.0 * ey)
    Eg = np.diag([ex, ey, ey])
    density = mu * (Eg * Eg).sum() + 0.5 * lam * np.trace(Eg) ** 2
    assert abs(m.energy(rest * [s, lateral, lateral])[0] / (density * 2.0) - 1.0) < 1e-12
    x = rest + np.random.default_rng(2).normal(0, 0.05, rest.shape)
    _e, g = m.energy(x)
    num = np.zeros_like(x)
    for i in range(len(x)):
        for k in range(3):
            a, b = x.copy(), x.copy()
            a[i, k] += 1e-6
            b[i, k] -= 1e-6
            num[i, k] = (m.energy(a)[0] - m.energy(b)[0]) / 2e-6
    assert np.abs(num - g).max() <= 1e-8 * np.abs(num).max()


def _compare_hessians(model, x, label):
    for psd in (False, True):
        exact = model.element_hessians(x, psd=psd, analytic=True)
        numeric = model.element_hessians(x, psd=psd, analytic=False)
        for (kind, verts, Ha), (_k, _v, Hn) in zip(exact, numeric):
            scale = max(np.abs(Hn).max(), 1e-12)
            err = np.abs(Ha - Hn).max() / scale
            assert err < 1e-5, (label, psd, kind, verts, err)


def test_analytic_hessians_match_finite_differences():
    # J-curve sheet, stretched and crumpled, bonds rising, softening and broken
    p = st.skin_params()
    rest, tris, _idx = pm.strip_mesh(3.0, 1.0, 0.5)
    sheet = pm.Sheet(rest, tris, np.arange(len(rest)), p)
    rng = np.random.default_rng(3)
    x = rest * [1.2, 0.95, 1.0] + rng.normal(0, 0.06, rest.shape)
    x[:4, 2] -= 0.2  # some vertices through the floor
    m = fs.FastModel.from_sheet(sheet)
    opened = rest.copy()
    opened[:, 2] += rng.uniform(0, 0.3, len(rest))
    m.commit(opened)
    _compare_hessians(m, x, "sheet")
    # tets, stretched and partly compressed
    rest, tets = fs.box_tets(2.0, 1.0, 1.0, 0.5)
    m = fs.FastModel.from_tets(rest, tets, 1.0, 0.45)
    _compare_hessians(m, rest * [1.15, 0.9, 1.0] + rng.normal(0, 0.05, rest.shape), "tets")
    # mixed opening / sliding bonds
    verts = np.arange(len(rest))
    m.set_bonds(verts, rest, np.full(len(rest), 0.2), [0.05, 0.05, 0.4, 0.2, 0.02, 0.1], mixed=True)
    xb = rest + rng.normal(0, 0.15, rest.shape)
    m.commit(rest + rng.normal(0, 0.05, rest.shape))
    _compare_hessians(m, xb, "mixed bonds")


def test_newton_matches_exact_force_at_real_time_frames():
    _rows, sheet = race.sheet_case(1.0, fs.NEWTON, 3)
    assert sheet["force_error_max"] < 1e-3, sheet
    _rows, volume = race.volume_case(1.0, fs.NEWTON, 3)
    assert volume["force_error_max"] < 1e-3, volume


def test_newton_peels_at_the_kendall_force():
    params = pm.SkinParams()
    rows, summary = race.peel_case(5.0, fs.NEWTON, 3, length=16.0, width=1.0, edge=0.25, tab=4.0, lift=11.0)
    assert abs(summary["force_ratio"] - 1.0) < 0.08, summary
    assert abs(summary["front_advance_per_mm_lift"] - 1.0) < 0.15, summary
    assert abs(summary["expected_force_n"] - ps.kendall_force_per_width(params) * 1.0) < 1e-12


def test_cholmod_and_eigen_take_the_same_newton_step():
    if not fs.have_cholmod():
        return  # SuiteSparse not installed; Eigen is the only factorization
    rest, tets = fs.box_tets(3.0, 1.0, 1.0, 0.5)
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[rest[:, 0] < 1e-9] = True
    end = rest[:, 0] > 3.0 - 1e-9
    fixed[end] = True
    out = {}
    for kind in (fs.EIGEN_LDLT, fs.CHOLMOD_SUPERNODAL, fs.CHOLMOD_SIMPLICIAL):
        m = fs.FastModel.from_tets(rest, tets, 1.0, 0.45)
        m.set_factor(kind)
        m.set_fixed(fixed)
        x = rest.copy()
        x[end] += [0.3, 0.1, -0.2]
        v = np.zeros_like(x)
        for _ in range(3):
            m.step(x, v, 5.0, fs.NEWTON, 3)
        out[kind] = x
    for kind in (fs.CHOLMOD_SUPERNODAL, fs.CHOLMOD_SIMPLICIAL):
        assert np.abs(out[kind] - out[fs.EIGEN_LDLT]).max() < 1e-9


def test_xpbd_needs_mass_scaling_to_converge():
    rest, tets = fs.box_tets(6.0, 2.0, 2.0, 0.5)
    left = np.flatnonzero(rest[:, 0] < 1e-9)
    right = np.flatnonzero(rest[:, 0] > 6.0 - 1e-9)
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[left, 0] = fixed[right, 0] = True
    corner = left[np.argmin(np.linalg.norm(rest[left, 1:], axis=1))]
    fixed[corner, 1:] = True
    up = left[np.argmin(np.linalg.norm(rest[left, 1:] - [0.0, 2.0], axis=1))]
    fixed[up, 1] = True
    s = 1.1
    exact = 4.0 * s * 1.0 * 0.5 * (s * s - 1.0)
    ratios = {}
    for scale in (1.0, 1e4):
        m = fs.FastModel.from_tets(rest, tets, 1.0, 0.45, density=fs.DENSITY_G_PER_MM3 * scale)
        m.set_fixed(fixed)
        x = rest * [s, 1.0, 1.0]
        v = np.zeros_like(x)
        for _ in range(200):
            m.step(x, v, 1.0, fs.XPBD, 50, damping=0.1)
        ratios[scale] = m.energy(x)[1][right, 0].sum() / exact
    assert abs(ratios[1e4] - 1.0) < 0.01, ratios
    assert abs(ratios[1.0] - 1.0) > 0.5, ratios  # true tissue mass: far off


if __name__ == "__main__":
    test_energy_and_gradient_match_the_reference()
    test_tet_energy_is_exact_for_uniform_stretch_and_gradient_matches()
    test_analytic_hessians_match_finite_differences()
    test_newton_matches_exact_force_at_real_time_frames()
    test_newton_peels_at_the_kendall_force()
    test_cholmod_and_eigen_take_the_same_newton_step()
    test_xpbd_needs_mass_scaling_to_converge()
    print("fast solver tests passed")
