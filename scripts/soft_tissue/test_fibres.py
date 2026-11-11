"""Tension-only fibres (the auricular muscle fans), slow reference and fast solver, against exact answers.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_fibres.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fibre_test as ft  # noqa: E402
import fibres as fb  # noqa: E402
import laminate_test as lt  # noqa: E402
import peel_model as pm  # noqa: E402
import stack as sk  # noqa: E402

BOTH = (False, True)


def _stack():
    rest, tris, idx = pm.strip_mesh(6.0, 2.0, 0.5)
    F = fb.Fibres.concat([fb.fan(idx[-1], rest, [3, -5, 6], [3, 7, 6], 0.5, per_vertex=3, **ft.LAW),
                          fb.Fibres([idx[-2][0]], [idx[0][0]], [[0, 0, 0]], [3.0], [0.2], **ft.LAW)])
    st = sk.Stack(rest, tris, [lt.layer(5.0, 1.0)], [], None, bending="shell", floor_z=-100.0, fibres=F)
    x = st.rest.copy()
    x[:, 2] -= 0.4 * x[:, 0]
    x[:, 0] *= 1.05
    return st, F, x


def test_fibre_gradient_matches_finite_differences():
    _st, F, x = _stack()
    e, g = F.energy(x)
    fd = np.zeros_like(x)
    h = 1e-6
    for i in range(len(x)):
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, c] += h
            xm[i, c] -= h
            fd[i, c] = (F.energy(xp)[0] - F.energy(xm)[0]) / (2 * h)
    assert np.abs(fd - g).max() < 1e-7 * np.abs(g).max()


def test_fast_fibres_match_the_reference():
    import fast_solver as fs

    st, _F, x = _stack()
    e1, g1 = st.energy(x)
    e2, g2 = fs.FastModel.from_stack(st).energy(np.ascontiguousarray(x))
    assert abs(e1 - e2) < 1e-12 * abs(e1)
    assert np.abs(g1 - g2).max() < 1e-12 * np.abs(g1).max()


def test_fibres_at_rest_carry_no_force():
    for fast in BOTH:
        assert ft.rest_check(fast) < 1e-12


def test_fibres_in_series_share_the_exact_tension():
    for fast in BOTH:
        for s in (1.05, 1.2, 1.4):
            T, want = ft.series_chain(s, fast)
            assert abs(T / want - 1) < 1e-6, (fast, s, T / want)


def test_fan_adds_its_stiffness_when_pulled_and_nothing_when_slack():
    for fast in BOTH:
        d = 0.02
        P0, _ = ft.fan_on_plate(d, fast, with_fan=False)
        P1, k = ft.fan_on_plate(d, fast)
        assert abs((P1 - P0) / (k * d) - 1) < 0.01, (fast, (P1 - P0) / (k * d))
        P0u, _ = ft.fan_on_plate(-d, fast, with_fan=False)
        P1u, _ = ft.fan_on_plate(-d, fast)
        assert abs(P1u - P0u) < 1e-9 * abs(P0u), (fast, P1u, P0u)


def test_set_force_stretches_two_fibres_in_series_to_that_tension():
    """A set (dead) force T on the end of two fibres in series (fast solver's applied load): each
    fibre carries exactly T, so each stretches to its own lambda with A lambda W'(e) = T."""
    import fast_solver as fs
    from scipy.optimize import brentq

    L1, L2, A1, A2 = 8.0, 12.0, 0.4, 1.0
    rest, tris, _idx = pm.strip_mesh(0.2, 0.2, 0.2)
    k = len(rest)
    pts = np.concatenate([rest + [L1, 0.0, 0.0], rest + [L1 + L2, 0.0, 0.0]])
    joint, end = np.arange(k), np.arange(k, 2 * k)
    F = fb.Fibres(np.concatenate([joint, end]), np.concatenate([-np.ones(k, int), joint]),
                  np.concatenate([rest, np.zeros((k, 3))]), np.concatenate([np.full(k, L1), np.full(k, L2)]),
                  np.concatenate([np.full(k, A1 / k), np.full(k, A2 / k)]), **ft.LAW)
    st = sk.Stack(pts, np.concatenate([tris, tris + k]), [lt.layer(1e-6, 0.01)], [], None, floor_z=-1e9, fibres=F)
    fixed = np.zeros(pts.shape, bool)
    fixed[:, 1:] = True  # along x only
    for T in (0.05, 0.5, 2.0):
        f = np.zeros_like(pts)
        f[end, 0] = T / k
        m = fs.FastModel.from_stack(st)
        m.set_fixed(fixed)
        m.set_external(f)
        x = pts.copy()
        m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 200, gtol=1e-11)
        lam = lambda A: brentq(lambda s: A * s * F.law(0.5 * (s * s - 1))[1] - T, 1.0, 3.0)
        got = x[end, 0].mean() - rest[:, 0].mean()
        want = L1 * lam(A1) + L2 * lam(A2)
        assert abs(got / want - 1) < 1e-6, (T, got, want)
        assert abs(F.tension(x)[:k].sum() / T - 1) < 1e-6


def test_grip_spring_and_fibre_in_series_share_one_tension():
    """A grip spring (finger pad, k per vertex) pulling the end of a fibre toward a target: at rest
    the spring's pull k (t - x) equals the fibre's tension, and the energy's gradient matches finite
    differences. Grabbing this way keeps the solver's sparsity pattern (no set_fixed)."""
    import fast_solver as fs

    L1, A1 = 10.0, 0.5
    rest, tris, _idx = pm.strip_mesh(0.2, 0.2, 0.2)
    k = len(rest)
    pts = rest + [L1, 0.0, 0.0]
    F = fb.Fibres(np.arange(k), -np.ones(k, int), rest.copy(), np.full(k, L1), np.full(k, A1 / k), **ft.LAW)
    st = sk.Stack(pts, tris, [lt.layer(1e-6, 0.01)], [], None, floor_z=-1e9, fibres=F)
    fixed = np.zeros(pts.shape, bool)
    fixed[:, 1:] = True
    K = 2.0  # N/mm over the whole grip
    m = fs.FastModel.from_stack(st)
    m.set_fixed(fixed)
    for D in (0.5, 2.0, 4.0):
        t = pts + [D, 0.0, 0.0]
        m.set_grip(np.full(k, K / k), t)
        x = pts.copy()
        m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 200, gtol=1e-11)
        pull = (K / k) * (t[:, 0] - x[:, 0]).sum()
        T = F.tension(x).sum()
        assert pull > 0 and abs(pull / T - 1) < 1e-6, (D, pull, T)
    xr = pts + np.random.default_rng(1).normal(0, 0.05, pts.shape)
    e, g = m.energy(xr)
    for i in (0, k // 2):
        h = np.zeros_like(xr)
        h[i, 0] = 1e-6
        fd = (m.energy(xr + h)[0] - m.energy(xr - h)[0]) / 2e-6
        assert abs(fd - g[i, 0]) < 1e-5 * max(1.0, abs(fd)), (fd, g[i, 0])
    m.set_grip(None)
    assert abs(m.energy(pts)[0]) < 1e-12


def test_obstacle_contact_holds_a_pushed_sheet_at_force_over_stiffness():
    """A very soft sheet pushed by a set force into a tilted plane obstacle (a signed distance
    grid): each vertex comes to rest F / k inside its clearance along the plane's normal, the
    energy's gradient matches finite differences, and without contact the sheet goes through."""
    import fast_solver as fs

    rest, tris, _idx = pm.strip_mesh(4.0, 2.0, 0.5)
    st = sk.Stack(rest, tris, [lt.layer(1e-6, 0.01)], [], None, floor_z=-1e9)
    n = np.array([0.3, -0.2, 1.0])
    n /= np.linalg.norm(n)
    h = 0.25
    ax = [np.arange(-6.0, 10.0 + h, h)] * 3
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    phi = (np.stack([X, Y, Z], -1) @ n).astype(np.float32)  # distance above the plane through 0
    k, F, clear = 40.0, 2.0, 0.6
    pts = rest + clear * n  # start resting on it (as the ear starts in place)
    m = fs.FastModel.from_stack(st)
    m.set_obstacle(phi, [ax[0][0]] * 3, h)
    m.set_contact(np.full(len(pts), k), clear)
    f = np.tile(-F * n, (len(pts), 1))
    m.set_external(f)
    x = pts.copy()
    m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 300, gtol=1e-10)
    depth = x @ n - clear
    assert np.allclose(depth, -F / k, atol=1e-4), (depth.min(), depth.max(), -F / k)
    xr = x + np.random.default_rng(2).normal(0, 0.02, x.shape)
    e, g = m.energy(xr)
    for i in (0, len(x) // 2):
        for c in range(3):
            d = np.zeros_like(xr)
            d[i, c] = 1e-5
            fd = (m.energy(xr + d)[0] - m.energy(xr - d)[0]) / 2e-5
            assert abs(fd - g[i, c]) < 1e-4 * max(1.0, abs(fd)), (fd, g[i, c])
    m.set_contact(None)
    x = pts.copy()
    m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 50, gtol=1e-10)
    assert (x @ n).max() < clear - 1.0  # nothing held it


def test_self_contact_holds_two_sheets_apart_by_the_clearance_less_force_over_stiffness():
    """Two parallel soft sheets (one model, no shared vertices), the lower held, the upper pushed
    down onto it by a set force per vertex: each upper vertex rests on the one below at the
    clearance less F / k (vertex grids aligned, so one pair per vertex); with it off the sheets pass
    through."""
    import fast_solver as fs

    rest, tris, _idx = pm.strip_mesh(4.0, 2.0, 1.0)
    k_, F, d0 = 30.0, 1.5, 1.0
    n = len(rest)
    pts = np.concatenate([rest, rest + [0.0, 0.0, d0]])  # upper sheet resting on the lower one
    tt = np.concatenate([tris, tris + n])
    st = sk.Stack(pts, tt, [lt.layer(1e-6, 0.01)], [], None, floor_z=-1e9)
    m = fs.FastModel.from_stack(st)
    fixed = np.zeros(pts.shape, bool)
    fixed[:n] = True
    fixed[n:, :2] = True  # the upper sheet moves only up and down
    m.set_fixed(fixed)
    f = np.zeros_like(pts)
    f[n:, 2] = -F
    m.set_external(f)
    m.set_self_contact(k_, d0 + 0.5, np.ones(2 * n, bool), tt, rings=2)  # clearance 1.5: rests 0.5 into it
    x = pts.copy()
    x[n:, 2] -= 0.32  # already touching (as contact starts in use; a free jump could tunnel past)
    for _ in range(40):
        m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 20, gtol=1e-11)
    gap = x[n:, 2] - x[:n, 2]
    # rest distance is d0 = 1.0 < clearance: the pair keeps rest distance - give = 0.7, less F / k
    assert np.allclose(gap, d0 - 0.3 - F / k_, atol=2e-3), (gap.min(), gap.max(), d0 - 0.3 - F / k_)
    m.set_self_contact(0, 0, None, tt)
    x = pts.copy()
    x[n:, 2] -= 0.32
    m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 30, gtol=1e-11)
    assert (x[n:, 2] - x[:n, 2]).max() < -0.5  # off: it goes through


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "ok", flush=True)
