"""Layered stack: the fast C++ model matches the slow reference, element by element."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fast_solver as fs  # noqa: E402
import peel_model as pm  # noqa: E402
import stack as sk  # noqa: E402
import wedge_test as wt  # noqa: E402


def _mastoid(blade=True, edge=0.25):
    rest, tris, idx = pm.strip_mesh(3.0, 1.0, edge)
    st = sk.from_table("mastoid", rest, tris, blade=wt.Blade(height_mm=0.3, slope=1.0, heel_mm=0.2) if blade else None)
    st.tip = 1.6
    return st, idx


def _messy(st, seed=0):
    rng = np.random.default_rng(seed)
    x = st.rest + rng.normal(0, 0.06, st.rest.shape)
    x[st.layer == 0, 2] += rng.uniform(0, 0.8, (st.layer == 0).sum())  # skin lifting off
    x[: st.n // 3, 2] -= 0.05  # some through the floor and the blade
    opened = st.rest + rng.normal(0, 0.03, st.rest.shape)
    opened[:, 2] += rng.uniform(0, 6.0, len(opened))  # damage history: rising, softening, broken
    return x, opened


def test_energy_and_gradient_match_the_reference():
    st, _ = _mastoid()
    x, opened = _messy(st)
    m = fs.FastModel.from_stack(st)
    st.commit(opened)
    m.commit(opened)
    assert st.pair_broken.any() and not st.pair_broken.all()
    e_ref, g_ref = st.energy(x)
    e_fast, g_fast = m.energy(x)
    assert abs(e_fast - e_ref) <= 1e-11 * abs(e_ref), (e_fast, e_ref)
    assert np.abs(g_fast - g_ref).max() <= 1e-11 * np.abs(g_ref).max()
    assert abs(m.blade_push(x) - st.push_force(x)) <= 1e-12 + 1e-11 * abs(st.push_force(x))


def test_pair_and_blade_hessians_match_finite_differences():
    st, _ = _mastoid()
    x, opened = _messy(st, seed=1)
    m = fs.FastModel.from_stack(st)
    m.commit(opened)
    seen = {}
    for psd in (False, True):
        for (kind, verts, Ha), (_k, _v, Hn), (_k2, _v2, Hraw) in zip(
                m.element_hessians(x, psd, True), m.element_hessians(x, psd, False), m.element_hessians(x, False, True)):
            scale = max(np.abs(Hn).max(), 1e-12)
            assert np.abs(Ha - Hn).max() / scale < 1e-5, (psd, kind, verts)
            if kind in (5, 6) and not psd:
                s = seen.setdefault(kind, [0, 0])
                s[0] += 1
                s[1] += int(np.linalg.eigvalsh(Hraw).min() < -1e-9 * scale)
    # both new elements present, and some of each in their non-convex range
    assert seen[5][1] > 0 and seen[6][1] > 0, seen


def test_quasi_static_steps_match_the_reference():
    st, idx = _mastoid(blade=False)
    grip = idx[0]  # the skin's free edge (layer 0)
    fixed = np.zeros(st.rest.shape, dtype=bool)
    fixed[grip, 2] = True
    m = fs.FastModel.from_stack(st)
    m.set_fixed(fixed)
    x_ref, x_fast = st.rest.copy(), st.rest.copy()
    for lift in np.arange(0.05, 0.41, 0.05):  # small steps, as the fixtures do
        x_ref[grip, 2] = x_fast[grip, 2] = lift
        x_ref, res = pm.solve(st, x_ref, fixed, gtol=1e-9)
        st.commit(x_ref)
        m.step(x_fast, np.zeros_like(x_fast), 0.0, fs.NEWTON, 60, gtol=1e-9)
        m.commit(x_fast)
        assert res.max_grad < 1e-8, res
        assert np.abs(x_fast - x_ref).max() < 1e-6, (lift, np.abs(x_fast - x_ref).max())
    assert np.array_equal(m.pair_state()[1], st.pair_broken)


if __name__ == "__main__":
    test_energy_and_gradient_match_the_reference()
    test_pair_and_blade_hessians_match_finite_differences()
    test_quasi_static_steps_match_the_reference()
    print("stack tests passed")
