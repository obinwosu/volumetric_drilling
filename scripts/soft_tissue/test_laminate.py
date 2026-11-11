"""Bonded layers at their real thickness (stack.Stack offset=True), slow reference and fast solver,
against exact answers: partial-interaction beams, the transformed section, and a 90 degree peel.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_laminate.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import laminate_test as lt  # noqa: E402
import peel_model as pm  # noqa: E402
import stack as sk  # noqa: E402
import wedge_test as wt  # noqa: E402

BOTH = (False, True)  # slow reference, fast C++
TOL = 0.04  # the shell's own overshoot is ~2% on these meshes


def _bent_stack(laws, bending="shell"):
    rest, tris, _idx = pm.strip_mesh(4.0, 2.0, 0.5)
    layers = [lt.layer(15.0, 0.4), lt.layer(2.0, 0.15), lt.layer(5.0, 1.0)]
    st = sk.Stack(rest, tris, layers, laws, None, offset=True, bending=bending, floor_z=-100.0)
    rng = np.random.default_rng(3)
    x = st.rest.copy()
    x[:, 2] += 0.08 * x[:, 0] ** 2 - 0.05 * x[:, 0] * x[:, 1]
    x += 0.03 * rng.standard_normal(x.shape)
    return st, x


LAWS = [[lt.plane_law(1.0), lt.plane_law(0.5, 4.0)],
        [pm.BondLaw(0.2, 0.05, 0.4), wt.Interface(pm.BondLaw(0.2, 0.05, 0.4), pm.BondLaw(0.1, 0.02, 0.3))]]


def test_face_bond_gradient_matches_finite_differences():
    for laws in LAWS:
        st, x = _bent_stack(laws)
        st.face_hist_n[:] = 0.03
        st.face_hist_s[:] = 0.01
        e, g = st._face_energy(x)
        fd = np.zeros_like(x)
        h = 1e-6
        for i in range(len(x)):
            for c in range(3):
                xp, xm = x.copy(), x.copy()
                xp[i, c] += h
                xm[i, c] -= h
                fd[i, c] = (st._face_energy(xp)[0] - st._face_energy(xm)[0]) / (2 * h)
        assert np.abs(fd - g).max() < 1e-7 * np.abs(g).max(), np.abs(fd - g).max()


def test_fast_faces_match_the_reference():
    import fast_solver as fs

    for laws in LAWS:
        for bending in ("shell", ["hinge", "shell", "shell"]):
            st, x = _bent_stack(laws, bending)
            e1, g1 = st.energy(x)
            e2, g2 = fs.FastModel.from_stack(st).energy(np.ascontiguousarray(x))
            assert abs(e1 - e2) < 1e-12 * abs(e1), (e1, e2)
            assert np.abs(g1 - g2).max() < 1e-12 * np.abs(g1).max()


def test_bonded_stack_rests_and_moves_rigidly_without_force():
    for fast in BOTH:
        assert max(lt.rest_check(fast)) < 1e-9, fast


def test_two_layers_follow_partial_interaction():
    for fast in BOTH:
        for aL in (0.0, 3.0, 30.0, 100.0):
            k, want = lt.partial_interaction(aL, fast=fast)
            assert abs(k / want - 1) < TOL, (fast, aL, k / want)


def test_one_law_plane_follows_partial_interaction():
    for fast in BOTH:
        k, want = lt.partial_interaction(30.0, mixed=False, fast=fast)
        assert abs(k / want - 1) < TOL, (fast, k / want)


def test_stiffly_bonded_three_layers_bend_as_one_section():
    for fast in BOTH:
        k, full, zero = lt.laminate([lt.layer(15.0, 0.4), lt.layer(2.0, 0.15), lt.layer(5.0, 1.0)], fast=fast)
        assert abs(k / full - 1) < TOL, (fast, k / full)
        assert zero / full < 0.3


def test_peeling_a_layer_follows_kendall():
    for fast in BOTH:
        r, *_ = lt.peel(fast=fast, step=0.1, lift_total=4.0)
        assert abs(r - 1) < 0.05, (fast, r)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "ok", flush=True)
