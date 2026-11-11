"""Curved shell (auricular cartilage) bending, slow reference and fast solver, against exact answers.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_plate.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plate_test as pt  # noqa: E402

BOTH = (False, True)  # slow reference, fast C++


def test_curved_shell_rests_in_its_own_shape():
    for fast in BOTH:
        for e, g in pt.rest_check(fast):
            assert abs(e) < 1e-9 and g < 1e-9, (fast, e, g)


def test_rebent_cylinder_stores_the_plate_energy():
    for fast in BOTH:
        e, want = pt.rebend(fast=fast)
        assert abs(e / want - 1) < 0.04, (fast, e / want)


def test_cantilever_matches_beam_theory():
    for fast in BOTH:
        P, arm, k = pt.cantilever(0.2, width=1.0, edge=0.25, fast=fast)
        assert abs(P / (3 * k * 0.2 / arm**3) - 1) < 0.04, (fast, P / (3 * k * 0.2 / arm**3))


def test_strip_curls_with_the_poisson_ratio():
    for fast in BOTH:
        r, nu = pt.poisson_curl(fast=fast)
        assert abs(r + nu) < 0.04, (fast, r)


def test_large_deflection_matches_the_elastica():
    for fast in BOTH:
        d = 0.5 * 19.0
        P, arm, k = pt.cantilever(d, width=1.0, edge=0.25, fast=fast)
        assert abs(P / pt.elastica_force(d, arm, k) - 1) < 0.04, (fast, P / pt.elastica_force(d, arm, k))


def test_pinched_ring_matches_the_ring_elastica():
    for fast in BOTH:
        P, want = pt.pinched_ring(0.4, fast=fast)
        assert abs(P / want - 1) < 0.04, (fast, P / want)


def test_fast_shell_matches_the_reference():
    import fast_solver as fs

    rest, tris, flat = pt.cylinder_patch(8.0, length=8, width=4, edge=0.5)
    sh = pt.sheet(rest, tris, pt.params(1.0), curved=True, calib_rest=flat)
    m = fs.FastModel.from_sheet(sh)
    x = rest + 0.1 * np.random.default_rng(1).standard_normal(rest.shape)
    e1, g1 = sh.energy(x)
    e2, g2 = m.energy(np.ascontiguousarray(x))
    assert abs(e1 - e2) < 1e-10 * abs(e1) and np.abs(g1 - g2).max() < 1e-10 * np.abs(g1).max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
