"""Plate test: a thin curved shell (auricular cartilage) bending, slow and fast.

Cartilage is a curved plate. The same sheet model as skin (membrane + hinges,
``peel_model.Sheet``) is used with a curved rest shape (``curved=True``): it
stretches from its rest triangles in their own planes and bends from its rest
angle at every hinge, so it holds its own shape and springs back to it.

Every fixture is loaded by displacement (vertices held where they are put) and
the force is read off as the reaction at those vertices, from both the slow
reference (``peel_model.solve``) and the fast C++ Newton (``fast_solver``).

Bending is the shell with the plate's Poisson ratio (``bending="shell"``): a
strip bent along its length curls the other way across its width (anticlastic),
so a narrow strip has the beam stiffness E w h^3 / 12 = (1 - nu^2) D w, and a
wide one approaches the plate's D w.

Known answers (D = E h^3 / (12 (1 - nu^2)), strip width w, beam EI = (1 - nu^2) D w)
  rest              a curved shell at rest, or moved rigidly, has no energy or force
  re-bend           a cylinder of rest radius R0 bent to R1 stores
                    D/2 (1/R1 - 1/R0)^2 per area (isometric: no stretch)
  cantilever        tip load P for a small tip deflection d over arm L (narrow strip):
                    P = 3 EI d / L^3
  Poisson curl      across the width the strip curls -nu times its curvature along it
  elastica          large deflection of the same cantilever under a vertical tip
                    load: exact elastica, EI theta'' = -P cos(theta),
                    theta(0) = 0, theta'(L) = 0, deflection = int sin(theta)
  pinched ring      a narrow ring of radius R squeezed across a diameter by P (curved rest
                    shape): exact ring elastica, which for small P is
                    shortening = (pi/4 - 2/pi) P R^3 / EI

Material: auricular cartilage from ``tissue_params.SHELL_MATERIALS`` (sources there).

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/plate_test.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402

import tissue_params as tp  # noqa: E402

_cart = tp.SHELL_MATERIALS["auricular_cartilage"]
CARTILAGE = dict(E_mpa=_cart.E_mpa.nominal, nu=_cart.nu)  # auricular cartilage (tissue_params, sourced)
RING_COEF = math.pi / 4.0 - 2.0 / math.pi  # 0.1488


def params(thickness_mm, **kw):
    """Plate parameters: no bonds, no floor."""
    p = dict(CARTILAGE, thickness_mm=thickness_mm, contact_n_per_mm3=0.0)
    p.update(kw)
    return pm.SkinParams(**p)


BENDING = "shell"  # cartilage: the plate with its Poisson ratio


def sheet(rest, tris, p, curved=False, calib_rest=None, bending=None):
    return pm.Sheet(np.asarray(rest, float), np.asarray(tris), np.zeros(0, np.int64), p, floor_z=-1e9,
                    curved=curved, calib_rest=calib_rest, bending=bending or BENDING)


def solve(sh, x0, fixed, fast=False):
    if fast:
        import fast_solver as fs

        x, info = fs.static_solve(sh, x0, fixed, gtol=1e-8, max_iter=200)
    else:
        x, info = pm.solve(sh, x0, fixed, gtol=1e-8, max_iter=200)
    return x, info


def reaction(sh, x, verts, axis):
    """Force the held vertices take along ``axis`` (sum of the energy gradient there)."""
    g = sh.energy(x)[1]
    return float(g[verts, axis].sum())


# -- meshes ---------------------------------------------------------------------
def cylinder_patch(radius, length=12.0, width=6.0, edge=0.5):
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    return pm.roll_about_y(rest, radius), tris, rest


def sphere_cap(radius, half=6.0, edge=0.5):
    rest, tris, idx = pm.strip_mesh(2 * half, 2 * half, edge)
    xy = rest[:, :2] - half
    r2 = (xy**2).sum(axis=1)
    out = np.column_stack([xy, radius - np.sqrt(radius**2 - r2)])
    return out, tris


def ring(radius, width, edge):
    """A closed ring: a strip rolled once round (seam merged), axis along y."""
    circ = 2 * math.pi * radius
    nx = max(8, int(round(circ / edge)))
    rest, tris, idx = pm.strip_mesh(circ, width, circ / nx)
    rolled = pm.roll_about_y(rest, radius)
    ny = idx.shape[1]
    remap = np.arange(len(rest))
    remap[idx[nx]] = idx[0]  # the last column is the first
    keep = np.ones(len(rest), bool)
    keep[idx[nx]] = False
    new = np.cumsum(keep) - 1
    t = new[remap[tris]]
    return rolled[keep], t, idx[:nx] if False else (new[idx[0]], new[idx[nx // 2]])


# -- fixtures ---------------------------------------------------------------------
def rest_check(fast=False):
    """Largest |force| (N) at rest and after a rigid motion, on a cylinder patch and a sphere cap."""
    out = []
    for rest, tris, calib in (cylinder_patch(8.0), (*sphere_cap(10.0), None)):
        sh = sheet(rest, tris, params(1.0), curved=True, calib_rest=calib)
        rot = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
        for x in (rest, rest @ rot.T + [3.0, -2.0, 5.0]):
            if fast:
                import fast_solver as fs

                m = fs.FastModel.from_sheet(sh)
                e, g = m.energy(np.ascontiguousarray(x))
            else:
                e, g = sh.energy(x)
            out.append((float(e), float(np.abs(g).max())))
    return out


def rebend(R0=8.0, R1=5.0, fast=False):
    """(bending energy, expected) bending a cylinder of rest radius R0 to R1 (isometric)."""
    rest, tris, flat = cylinder_patch(R0, length=10.0, width=6.0, edge=0.25)
    p = params(1.0)
    sh = sheet(rest, tris, p, curved=True, calib_rest=flat)
    x = pm.roll_about_y(flat, R1)
    if fast:
        import fast_solver as fs

        e = fs.FastModel.from_sheet(sh).energy(np.ascontiguousarray(x))[0]
    else:
        e = sh.energy(x, parts=True)[2]["bending"]
    area = float(sh.area.sum())
    return float(e), 0.5 * p.bending_D * (1.0 / R1 - 1.0 / R0) ** 2 * area


def cantilever(deflection, length=20.0, width=4.0, thickness=1.2, edge=0.5, fast=False, bending=None):
    """(tip force, arm) for a flat strip clamped at x <= 2 edges, tip pushed down ``deflection``."""
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    p = params(thickness)
    sh = sheet(rest, tris, p, bending=bending)
    clamp = idx[:3].ravel()  # three columns: no rotation at the root
    tip = idx[-1]
    fixed = np.zeros((len(rest), 3), bool)
    fixed[clamp] = True
    fixed[tip, 2] = True
    x0 = rest.copy()
    arm = length - 2 * edge
    s = (rest[:, 0] - 2 * edge).clip(0) / arm
    x0[:, 2] = -deflection * s * s * (3 - s) / 2  # the linear beam shape, a good start
    x0[tip, 2] = -deflection
    x, _info = solve(sh, x0, fixed, fast)
    return -reaction(sh, x, tip, 2), arm, beam_EI(p, width)


def beam_EI(p, width):
    """Narrow strip: E w h^3 / 12 = (1 - nu^2) D w."""
    return (1.0 - p.nu**2) * p.bending_D * width


def poisson_curl(width=4.0, length=20.0, thickness=1.2, edge=0.25, fast=False):
    """Curvature across the width / along the length at mid-span of the bent cantilever (plate: -nu)."""
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    p = params(thickness)
    sh = sheet(rest, tris, p)
    fixed = np.zeros((len(rest), 3), bool)
    fixed[idx[:3].ravel()] = True
    tip = idx[-1]
    fixed[tip, 2] = True
    arm = length - 2 * edge
    s = (rest[:, 0] - 2 * edge).clip(0) / arm
    x0 = rest.copy()
    x0[:, 2] = -0.2 * s * s * (3 - s) / 2
    x0[tip, 2] = -0.2
    x, _info = solve(sh, x0, fixed, fast)
    i = idx.shape[0] // 2
    across = 2 * np.polyfit(x[idx[i], 1], x[idx[i], 2], 2)[0]
    mid = idx[:, idx.shape[1] // 2]
    along = 2 * np.polyfit(x[mid[i - 3 : i + 4], 0], x[mid[i - 3 : i + 4], 2], 2)[0]
    return across / along, p.nu


def elastica_force(deflection, arm, stiffness):
    """Vertical tip load giving this tip deflection to an inextensible cantilever (exact elastica)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def tip_deflection(P):
        # shoot from the tip, where theta' = 0: theta'' = -(P/EI) cos(theta), integrating s from L to 0
        def rhs(s, y):
            return [y[1], -(P / stiffness) * math.cos(y[0])]

        def miss(theta_tip):
            sol = solve_ivp(rhs, [0.0, arm], [theta_tip, 0.0], rtol=1e-10, atol=1e-12)
            return sol.y[0, -1]  # theta at the root (must be 0)

        th_tip = brentq(miss, 1e-9, math.pi / 2 - 1e-9)
        sol = solve_ivp(rhs, [0.0, arm], [th_tip, 0.0], rtol=1e-10, atol=1e-12, dense_output=True)
        s = np.linspace(0.0, arm, 2001)
        return float(np.trapezoid(np.sin(sol.sol(s)[0]), s))

    return brentq(lambda P: tip_deflection(P) - deflection, 1e-6 * stiffness / arm**2, 20.0 * stiffness / arm**2)


def ring_exact_squeeze(P, R, EI):
    """Exact squeeze of an inextensible ring of radius R pinched by P across a diameter (elastica).

    A quarter ring from the load point (tangent level, shear P/2, unknown moment M0) to the side
    (tangent vertical): EI (phi' - 1/R) = M0 + P x / 2, with M0 shot so phi = pi/2 at the side.
    For small P it is the curved-beam formula (pi/4 - 2/pi) P R^3 / EI; the ring then softens
    (already about 4% by a squeeze of R / 20)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    L = math.pi * R / 2

    def run(M0):
        def f(s, y):
            return [1 / R + (M0 + 0.5 * P * y[1]) / EI, math.cos(y[0]), -math.sin(y[0])]

        return solve_ivp(f, [0, L], [0.0, 0.0, 0.0], rtol=1e-11, atol=1e-13)

    M0 = brentq(lambda m: run(m).y[0, -1] - math.pi / 2, -P * R, P * R)
    return 2 * (R + run(M0).y[2, -1])


def ring_exact_force(squeeze, R, EI):
    from scipy.optimize import brentq

    k = EI / (RING_COEF * R**3)
    return brentq(lambda P: ring_exact_squeeze(P, R, EI) - squeeze, 0.2 * k * squeeze, 3 * k * squeeze)


def pinched_ring(squeeze, radius=8.0, width=1.0, thickness=0.4, edge=0.25, fast=False):
    """(force, exact) squeezing a ring across a diameter by ``squeeze`` mm.

    Narrow ring (width^2 / (R h) = 0.3) so it bends as a beam: a wider ring cannot curl freely
    across its width on its curved rest shape and stiffens toward the plate (a 3 mm ring: +5%)."""
    rest, tris, (bottom, top) = ring(radius, width, edge)
    p = params(thickness)
    sh = sheet(rest, tris, p, curved=True)
    fixed = np.zeros((len(rest), 3), bool)
    fixed[bottom] = True
    fixed[top, 0] = fixed[top, 2] = True
    x0 = rest.copy()
    x0[:, 2] -= squeeze * (rest[:, 2] / (2 * radius))  # spread the squeeze as a start
    x, _info = solve(sh, x0, fixed, fast)
    P = -reaction(sh, x, top, 2)
    return P, ring_exact_force(squeeze, radius, beam_EI(p, width))


def main():
    for fast in (False, True):
        tag = "fast" if fast else "slow"
        print(f"[{tag}] rest: max |force| {max(g for _e, g in rest_check(fast)):.2e} N")
        e, want = rebend(fast=fast)
        print(f"[{tag}] re-bend R 8 -> 5 mm: energy / plate {e / want:.4f}")
        P, arm, k = cantilever(0.2, width=1.0, edge=0.25, fast=fast)
        print(f"[{tag}] cantilever 0.2 mm (1 mm strip): P / beam {P / (3 * k * 0.2 / arm**3):.4f}")
        r, nu = poisson_curl(fast=fast)
        print(f"[{tag}] Poisson curl: across / along {r:.3f} (plate {-nu})")
        P, arm, k = cantilever(0.5 * 19.0, width=1.0, edge=0.25, fast=fast)
        print(f"[{tag}] elastica 9.5 mm: P / exact {P / elastica_force(0.5 * 19.0, arm, k):.4f}")
        P, want = pinched_ring(0.4, fast=fast)
        print(f"[{tag}] pinched ring 0.4 mm (1 mm wide): P / exact elastica {P / want:.4f}")


if __name__ == "__main__":
    main()
