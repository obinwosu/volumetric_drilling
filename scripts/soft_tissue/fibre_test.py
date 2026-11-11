"""Fibre test: tension-only fans (fibres.py), slow and fast, against exact answers.

Known answers (fibre law: Green strain e along the fibre, W = E e^2 / 2 + collagen J-curve,
tension A lambda W'(e), nothing when slack)
  rest            fibres at their rest length carry no force
  series chain    two unequal fibres (different length and area) end to end between a fixed origin
                  and an end pulled to a total length: the free joint finds the one tension T with
                  L1 lambda1(T) + L2 lambda2(T) = total (solved exactly), far up the J-curve
  fan on a plate  a fan spread across the tip of a clamped cartilage strip from origins above it
                  (fibres in the plane across the strip): pushing the tip down by d adds the fan's
                  stiffness in parallel, sum A (E + k1) / L0 cos^2(theta) d, to the plate's own;
                  pushing it up leaves the fan slack and adds nothing

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/fibre_test.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fibres as fb  # noqa: E402
import laminate_test as lt  # noqa: E402
import peel_model as pm  # noqa: E402
import stack as sk  # noqa: E402

# a J-curve like fascia / passive muscle (placeholder until the literature values are in)
LAW = dict(E_mpa=0.5, k1_mpa=1.0, k2=20.0)


def solve(st, x0, fixed, fast=False):
    if fast:
        import fast_solver as fs

        return fs.static_solve(st, x0, fixed, gtol=1e-10, max_iter=200)[0]
    return pm.solve(st, x0, fixed, gtol=1e-10, max_iter=200)[0]


def _host(n_extra=0):
    """A tiny stiff host sheet whose vertices the fibres can use (it carries no load here)."""
    rest, tris, idx = pm.strip_mesh(1.0, 1.0, 0.5)
    return rest, tris, idx


def series_chain(total_stretch, fast=False):
    """(tension, exact) for two fibres in series pulled to total_stretch x their summed rest length."""
    from scipy.optimize import brentq

    L1, L2, A1, A2 = 8.0, 12.0, 0.4, 1.0
    # vertices: 0 joint, 1 end (own tiny sheets keep the stack shape; they are far apart and unbonded)
    rest, tris, _idx = pm.strip_mesh(0.2, 0.2, 0.2)
    k = len(rest)
    pts = np.concatenate([rest + [L1, 0.0, 0.0], rest + [L1 + L2, 0.0, 0.0]])
    T = np.concatenate([tris, tris + k])
    joint, end = np.arange(k), np.arange(k, 2 * k)
    F = fb.Fibres(np.concatenate([joint, end]), np.concatenate([-np.ones(k, int), joint]),
                  np.concatenate([np.zeros((k, 3)) + rest * [1, 1, 1], np.zeros((k, 3))]),
                  np.concatenate([np.full(k, L1), np.full(k, L2)]), np.concatenate([np.full(k, A1 / k), np.full(k, A2 / k)]),
                  **LAW)
    F.origin[:k] = rest  # each joint vertex's fibre comes from straight behind it
    st = sk.Stack(pts, T, [lt.layer(1e-6, 0.01)], [], None, floor_z=-1e9, fibres=F)
    fixed = np.zeros(pts.shape, bool)
    fixed[end] = True
    fixed[:, 1:] = True  # along x only
    x0 = pts.copy()
    total = total_stretch * (L1 + L2)
    x0[end, 0] = rest[:, 0] + total  # each end straight ahead of its own origin
    x0[joint, 0] = rest[:, 0] + total * L1 / (L1 + L2)
    x = solve(st, x0, fixed, fast)
    tension = float(F.tension(x)[k:].sum())

    def lam(Tn, A):  # stretch carrying tension Tn in a fibre of area A
        return brentq(lambda s: A * s * F.law(0.5 * (s * s - 1))[1] - Tn, 1.0, 3.0)

    exact = brentq(lambda Tn: L1 * lam(Tn, A1) + L2 * lam(Tn, A2) - total, 1e-9, 1e3)
    return tension, exact


def fan_on_plate(d, fast=False, with_fan=True, length=20.0, width=4.0, edge=0.25, height=15.0, spread=10.0,
                 full=False, x_start=None, fan_thickness=0.3):
    """Tip force pushing the tip of a clamped 1 mm cartilage strip down by d (up when d < 0), and the
    fan's linear stiffness; with ``full`` also the solved positions and the stack."""
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    cart = lt.layer(5.0, 1.0)
    tip = idx[-1]
    F = None
    if with_fan:
        ys = rest[tip, 1]
        o_a = np.array([length, ys.mean() - spread / 2, height])
        o_b = np.array([length, ys.mean() + spread / 2, height])
        F = fb.fan(tip, rest, o_a, o_b, thickness_mm=fan_thickness, per_vertex=2, **LAW)
    st = sk.Stack(rest, tris, [cart], [], None, bending="shell", floor_z=-1e9, fibres=F)
    fixed = np.zeros(rest.shape, bool)
    fixed[idx[:3].ravel()] = True
    fixed[tip, 2] = True
    arm = length - 2 * edge
    s = (rest[:, 0] - 2 * edge).clip(0) / arm
    x0 = rest.copy() if x_start is None else x_start.copy()
    if x_start is None:
        x0[:, 2] -= d * s * s * (3 - s) / 2
    x0[tip, 2] = -d
    x = solve(st, x0, fixed, fast)
    g = st.energy(x)[1]
    P = -float(g[tip, 2].sum())
    k_fan = 0.0
    if F is not None:
        e = (F.origin - rest[F.a]) / F.rest_len[:, None]  # unit vectors insertion -> origin
        k_fan = float((F.area * (F.E_mpa + F.k1_mpa) / F.rest_len * e[:, 2] ** 2).sum())
    if full:
        return P, k_fan, x, st
    return P, k_fan


def rest_check(fast=False):
    rest, tris, idx = pm.strip_mesh(6.0, 2.0, 0.5)
    F = fb.fan(idx[-1], rest, [6, -4, 8], [6, 6, 8], 0.3, **LAW)
    st = sk.Stack(rest, tris, [lt.layer(5.0, 1.0)], [], None, bending="shell", floor_z=-1e9, fibres=F)
    if fast:
        import fast_solver as fs

        g = fs.FastModel.from_stack(st).energy(np.ascontiguousarray(st.rest))[1]
    else:
        g = st.energy(st.rest)[1]
    return float(np.abs(g).max())


def main():
    for fast in (False, True):
        tag = "fast" if fast else "slow"
        print(f"[{tag}] rest: max |force| {rest_check(fast):.1e} N")
        for s in (1.05, 1.2, 1.4):
            T, want = series_chain(s, fast)
            print(f"[{tag}] two fibres in series stretched x{s}: tension / exact {T / want:.5f}  ({T:.3f} N)")
        d = 0.02  # small: the stiffness is the linear one
        P0, _ = fan_on_plate(d, fast, with_fan=False)
        P1, k = fan_on_plate(d, fast)
        print(f"[{tag}] fan on a plate, tip down {d} mm: added force / fan stiffness x d {(P1 - P0) / (k * d):.4f}")
        P0u, _ = fan_on_plate(-d, fast, with_fan=False)
        P1u, _ = fan_on_plate(-d, fast)
        print(f"[{tag}] fan on a plate, tip up {d} mm: added force {abs(P1u - P0u):.1e} N (slack; plate alone {abs(P0u):.1e} N)")


if __name__ == "__main__":
    main()
