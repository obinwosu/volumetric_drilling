"""Laminate test: bonded layers with their real thickness, bending together, slow and fast.

Layers are stacked at their real heights and bonded face to face (``stack.Stack(offset=True)``):
each triangle of a layer to the same triangle of the next, at the faces where they touch. How
stiff a bonded stack is in bending then depends on how well the bond stops the faces slipping:

  no bond             each layer bends about its own centre: EI = sum E_i I_i
  rigid bond          one section (transformed section): EI = sum E_i (I_i + A_i (z_i - z_bar)^2)
  bond of slip modulus K (force per length per mm of slip): partial interaction (Newmark 1951),
                      exact for two layers below

Known answers (narrow strips, linear materials, small deflection; slip modulus K = k w with k the
bond's stiffness per area and w the strip width)
  rest              a bonded stack at rest, or moved rigidly, has no force
  partial interaction  cantilever of two bonded layers (arm L, clamped root, tip load P):
                      axial force N in each layer (opposite signs), lever r = (h1 + h2) / 2,
                      EI0 = EI1 + EI2, 1/EA* = 1/EA1 + 1/EA2, alpha^2 = K (1/EA* + r^2 / EI0):
                        N'' - alpha^2 N = -(K r / EI0) M(x),  M = P (L - x),
                        N(L) = 0 (free end), N'(0) = 0 (no slip at the clamp),
                        curvature (M - r N) / EI0, tip deflection = int curvature (L - x) dx
                      from sum EI_i (K = 0) to the transformed section (K -> infinity)
  laminate          a stiffly bonded stack of any number of layers: transformed section
  peel              the outer layer lifted at 90 degrees off a held inner layer: the face bonds
                    break in turn and the steady force per width is Kendall's for the plane's G

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/laminate_test.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402
import plate_test as pt  # noqa: E402
import stack as sk  # noqa: E402
import wedge_test as wt  # noqa: E402

# two layers for the partial-interaction sweep: a stiff thin membrane on a cartilage-like plate
TOP = dict(E_mpa=15.0, thickness_mm=0.4)
BOTTOM = dict(E_mpa=5.0, thickness_mm=1.0)  # auricular cartilage (tissue_params)
NU = 0.3


def layer(E_mpa, thickness_mm, nu=NU):
    """A linear layer: no collagen recruitment, no floor."""
    return pm.SkinParams(E_mpa=E_mpa, nu=nu, thickness_mm=thickness_mm, contact_n_per_mm3=0.0)


def elastic_law(k):
    """A bond of stiffness k (N/mm^3 per area) that never softens in these tests."""
    d0, dc = 10.0, 20.0
    return pm.BondLaw(0.5 * k * d0 * dc, d0, dc)


def plane_law(k_slide, k_open=None):
    """One law (isotropic, k_open None) or a mixed interface: sliding k_slide, opening k_open."""
    if k_open is None:
        return elastic_law(k_slide)
    return wt.Interface(elastic_law(k_open), elastic_law(k_slide))


# -- theory ---------------------------------------------------------------------------
def section(layers, width):
    """(EI0, EI_full, EA list, centres) of a stack listed outside to inside (z up)."""
    h = np.array([p.thickness_mm for p in layers])
    E = np.array([p.E_mpa for p in layers])
    z = np.array([h[k + 1:].sum() + 0.5 * h[k] for k in range(len(h))])
    EA = E * h * width
    I0 = E * width * h**3 / 12.0
    zbar = float((EA * z).sum() / EA.sum())
    return float(I0.sum()), float((I0 + EA * (z - zbar) ** 2).sum()), EA, z


def partial_interaction_stiffness(top, bottom, width, arm, K, n=20001):
    """Tip load per tip deflection of a two-layer cantilever with slip modulus K (Newmark)."""
    EI0, _full, EA, z = section([top, bottom], width)
    r = float(z[0] - z[1])
    inv_EA = 1.0 / EA[0] + 1.0 / EA[1]
    x = np.linspace(0.0, arm, n)
    M = arm - x  # unit tip load
    if K <= 0.0:
        N = np.zeros_like(x)
    else:
        a = math.sqrt(K * (inv_EA + r * r / EI0))
        beta = K * r / (EI0 * a * a)
        B = beta / a
        aL = a * arm
        # A cosh(ax) + B sinh(ax) with N(L) = 0, written to stay finite for large aL
        N = beta * M + B * (np.sinh(a * x) - np.tanh(aL) * np.cosh(a * x)) if aL < 300 else \
            beta * M + B * (np.exp(a * (x - arm)) * 0.0 - np.exp(-a * x))
    kappa = (M - r * N) / EI0
    tip = float(np.trapezoid(kappa * (arm - x), x))
    return 1.0 / tip


def slip_modulus_for(top, bottom, width, alpha_L, arm):
    """K giving this alpha L."""
    EI0, _full, EA, z = section([top, bottom], width)
    r = float(z[0] - z[1])
    return (alpha_L / arm) ** 2 / (1.0 / EA[0] + 1.0 / EA[1] + r * r / EI0)


# -- fixtures -----------------------------------------------------------------------------
def strip_stack(layers, laws, length=20.0, width=1.0, edge=0.25):
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    st = sk.Stack(rest, tris, layers, laws, None, offset=True, bending="shell", floor_z=-1e9)
    return st, idx


def solve(st, x0, fixed, fast=False):
    if fast:
        import fast_solver as fs

        x, info = fs.static_solve(st, x0, fixed, gtol=1e-8, max_iter=200)
    else:
        x, info = pm.solve(st, x0, fixed, gtol=1e-8, max_iter=200)
    return x, info


def cantilever(layers, laws, deflection=0.2, length=20.0, width=1.0, edge=0.25, fast=False, x_start=None):
    """(tip force, arm, solved x, stack, n) of a bonded stack clamped at its root (every layer), the
    innermost layer's tip pushed down; the bonds carry the load into the other layers. (Pushing every
    layer's tip to the same height over-constrains a rotated, bonded end.)"""
    st, idx = strip_stack(layers, laws, length, width, edge)
    n = st.n
    clamp = np.concatenate([st.layer_vertices(k, idx[:3].ravel()) for k in range(st.L)])
    tip = st.layer_vertices(st.L - 1, idx[-1])
    fixed = np.zeros((len(st.rest), 3), bool)
    fixed[clamp] = True
    fixed[tip, 2] = True
    arm = length - 2 * edge
    rx = np.tile(st.sheets[0].rest[:, 0], st.L)
    s = (rx - 2 * edge).clip(0) / arm
    x0 = st.rest.copy() if x_start is None else x_start.copy()
    if x_start is None:
        x0[:, 2] -= deflection * s * s * (3 - s) / 2
    x0[tip, 2] = st.rest[tip, 2] - deflection
    x, _info = solve(st, x0, fixed, fast)
    g = st.energy(x)[1]
    return -float(g[tip, 2].sum()), arm, x, st, n


def rest_check(fast=False):
    """Largest |force| (N) of a bonded three-layer stack at rest and moved rigidly (iso and mixed planes)."""
    layers = [layer(15.0, 0.4), layer(2.0, 0.15), layer(5.0, 1.0)]
    st, _idx = strip_stack(layers, [plane_law(1.0), plane_law(0.5, 4.0)], length=6.0, width=3.0, edge=0.5)
    c, s_ = 0.6, 0.8
    rot = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]]) @ np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0, 0, 1.0]])
    out = []
    for x in (st.rest, st.rest @ rot.T + [3.0, -2.0, 5.0]):
        if fast:
            import fast_solver as fs

            g = fs.FastModel.from_stack(st).energy(np.ascontiguousarray(x))[1]
        else:
            g = st.energy(x)[1]
        out.append(float(np.abs(g).max()))
    return out


K_OPEN = 1e3  # opening stiffness of the mixed plane (N/mm^3): the layers do not part (Newmark: no uplift)


def partial_interaction(alpha_L, mixed=True, fast=False, width=1.0, edge=0.25, length=20.0):
    """(model tip stiffness, Newmark) for two bonded layers at slip parameter alpha L.

    ``mixed``: sliding at the slip modulus, opening stiff (the theory's no-uplift). Otherwise one law
    on the full separation, which must then also carry the tip load across: only sensible stiff."""
    top, bottom = layer(**TOP), layer(**BOTTOM)
    arm = length - 2 * edge
    K = slip_modulus_for(top, bottom, width, alpha_L, arm) if alpha_L > 0 else 0.0
    k = K / width
    law = plane_law(max(k, 1e-9), K_OPEN if mixed else None)
    P, arm, _x, _st, _n = cantilever([top, bottom], [law], 0.2, length, width, edge, fast)
    return P / 0.2, partial_interaction_stiffness(top, bottom, width, arm, K)


def laminate(layers, k=2e3, fast=False, width=1.0, edge=0.25, length=20.0):
    """(model tip stiffness, transformed section 3 EI / L^3, sum of layers 3 EI0 / L^3), stiff bonds."""
    P, arm, _x, _st, _n = cantilever(layers, [plane_law(k)] * (len(layers) - 1), 0.2, length, width, edge, fast)
    EI0, full, _EA, _z = section(layers, width)
    return P / 0.2, 3 * full / arm**3, 3 * EI0 / arm**3


PEEL_LAW = pm.BondLaw(0.2, 0.02, 0.1)  # G 0.2 N/mm (skin on perichondrium, tissue_params); a short zone


def peel(fast=False, lift_total=None, step=None, record=None):
    """(steady force / Kendall, forces, lifts): the outer layer peeled at 90 degrees off a held inner one."""
    import tissue_params as tp

    top, bottom = layer(**TOP), layer(**BOTTOM)
    law = PEEL_LAW
    z = tp.zone_layers([top], law)
    edge = min(0.25, z / 3.0)
    tab = 1.0
    length = tab + 10.0
    width = 4 * edge
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    rx = rest[:, 0]
    st = sk.Stack(rest, tris, [top, bottom], [law], None, offset=True, bending="shell", floor_z=-1e9,
                  pair_masks=[rx >= tab - 1e-9])
    grip = st.layer_vertices(0, idx[0])
    fixed = np.zeros(st.rest.shape, bool)
    fixed[st.layer_slice(1)] = True
    fixed[grip, 2] = True  # held in height only: the grip follows the front (a 90 degree peel)
    x = st.rest.copy()
    lift_total = lift_total or 7.0
    step = step or 0.05
    expected = tp.kendall_layers([top], law.G_n_per_mm) * width
    if fast:
        import fast_solver as fs

        m = fs.FastModel.from_stack(st)
        m.set_fixed(fixed)
    forces, lifts = [], []
    k = 0
    while k * step < lift_total - 1e-9:
        k += 1
        x[grip, 2] = st.rest[grip, 2] + k * step
        if fast:
            m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 200, gtol=1e-8)
            m.commit(x)
            st.face_hist_n, st.face_hist_s, st.face_broken = m.face_state()
            g = m.energy(x)[1]
        else:
            x, _info = pm.solve(st, x, fixed, gtol=1e-8)
            st.commit(x)
            g = st.energy(x)[1]
        forces.append(float(g[grip, 2].sum()))
        lifts.append(k * step)
        if record is not None:
            record.append((x.copy(), st.face_broken.copy()))
    forces, lifts = np.array(forces), np.array(lifts)
    steady = float(np.median(forces[lifts > 0.5 * lift_total]))
    return steady / expected, forces, lifts, expected


def main():
    for fast in (False, True):
        tag = "fast" if fast else "slow"
        print(f"[{tag}] rest / rigid motion: max |force| {max(rest_check(fast)):.1e} N")
        for aL in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0):
            k, want = partial_interaction(aL, fast=fast)
            print(f"[{tag}] two layers, alpha L {aL:5.1f}: stiffness / Newmark {k / want:.4f}")
        k, want = partial_interaction(30.0, mixed=False, fast=fast)
        print(f"[{tag}] two layers, one law on the full separation, alpha L 30: stiffness / Newmark {k / want:.4f}")
        k, full, zero = laminate([layer(15.0, 0.4), layer(2.0, 0.15), layer(5.0, 1.0)], fast=fast)
        print(f"[{tag}] three layers, stiff bonds: stiffness / transformed section {k / full:.4f} "
              f"(sum of layers would be {zero / full:.3f})")
        r, *_ = peel(fast=fast)
        print(f"[{tag}] peel the outer layer off the inner: steady force / Kendall {r:.4f}")


if __name__ == "__main__":
    main()
