"""Race the fast solvers against the reference answers.

Each scenario is driven the way a user would: a grip moves a little every
frame, at a real-time frame rate, and the solver gets a fixed amount of work
per frame (Newton: iterations, XPBD: sweeps), warm-started from the last frame.
We record the force the solver reports and the wall time per frame, and
compare against the exact or reference answer.

Scenarios
  sheet    J-curve skin strip 10 x 2 mm pulled to 30%      exact force formula
  volume   tetrahedral block 6 x 2 x 2 mm pulled to 20%    exact force formula
  peel     5 x 25 mm skin strip peeled at 90 degrees      Kendall force, 244 mN
           (0.25 mm mesh, as the reference; peel_coarse: 0.5 mm)

Contenders: Newton with 1 or 3 iterations per frame; XPBD with small steps
(10 or 50 substeps per frame, one sweep each) and mass scaling, at 1 ms
(haptic rate) and 5 ms (200 Hz) frames. Skin is so stiff for its mass at this
mesh size that XPBD only converges with artificially heavier tissue; mass
scaling is harmless while the hand moves slowly compared with tissue waves,
and shows up as lag when it does not.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/solver_race.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fast_solver as fs  # noqa: E402
import peel_model as pm  # noqa: E402
import peel_strip as ps  # noqa: E402
import stretch_test as st  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
# (label, solver, iterations or substeps, mass scale)
CONTENDERS = [
    ("Newton 1 it", fs.NEWTON, 1, 1.0),
    ("Newton 3 it", fs.NEWTON, 3, 1.0),
    ("XPBD 10 sub, mass x100", fs.XPBD, 10, 1e2),
    ("XPBD 50 sub, mass x100", fs.XPBD, 50, 1e2),
    ("XPBD 10 sub, mass x1e4", fs.XPBD, 10, 1e4),
]
DAMPING = 0.1  # fraction of velocity lost per frame; tissue is heavily damped


THREADS = 8  # Newton assembly threads
FACTOR = fs.best_factor()  # Newton sparse factorization (--factor)
FACTORS = {"eigen": fs.EIGEN_LDLT, "cholmod": fs.CHOLMOD_SUPERNODAL, "cholmod_simplicial": fs.CHOLMOD_SIMPLICIAL}


def _frames(model, x, h, solver, iters, n_frames, set_targets, measure):
    model.set_options(project_psd=True, threads=THREADS)
    model.set_factor(FACTOR)
    v = np.zeros_like(x)
    rows = []
    for k in range(n_frames):
        set_targets(x, k + 1)
        stats = model.step(x, v, h, solver, iters, damping=DAMPING)
        model.commit(x)
        rows.append({**measure(x, k + 1), "ms": stats["ms"], "factor_ms": stats["factor_ms"],
                     "assemble_ms": stats["assemble_ms"]})
    return rows


# -- sheet ------------------------------------------------------------------
def sheet_case(h, solver, iters, mass_scale=1.0, length=10.0, width=2.0, edge=0.25, max_stretch=1.3, duration_ms=300.0):
    p = st.skin_params()
    rest, tris, _idx = pm.strip_mesh(length, width, edge)
    sheet = pm.Sheet(rest, tris, np.zeros(0, dtype=np.int64), p, floor_z=-1e9)
    m = fs.FastModel.from_sheet(sheet, density=fs.DENSITY_G_PER_MM3 * mass_scale)
    left = np.flatnonzero(rest[:, 0] < 1e-9)
    right = np.flatnonzero(rest[:, 0] > length - 1e-9)
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[:, 2] = True
    fixed[left, 0] = fixed[right, 0] = True
    fixed[left[np.argmin(np.abs(rest[left, 1] - 0.5 * width))], 1] = True
    m.set_fixed(fixed)
    n = int(round(duration_ms / h))

    def stretch(k):
        return 1.0 + (max_stretch - 1.0) * k / n

    def targets(x, k):
        x[right, 0] = length * stretch(k)

    def measure(x, k):
        _e, g = m.energy(x)
        s = stretch(k)
        return {"stretch": s, "force_n": float(g[right, 0].sum()),
                "exact_n": float(st.exact_force(p, s, width))}

    rows = _frames(m, rest.copy(), h, solver, iters, n, targets, measure)
    tail = rows[n // 3:]
    err = np.array([abs(r["force_n"] / r["exact_n"] - 1.0) for r in tail])
    return rows, {"force_error_median": float(np.median(err)), "force_error_max": float(err.max()),
                  "final_force_ratio": rows[-1]["force_n"] / rows[-1]["exact_n"]}


# -- volume -----------------------------------------------------------------
def volume_case(h, solver, iters, mass_scale=1.0, size=(6.0, 2.0, 2.0), edge=0.5, E=1.0, nu=0.45, max_stretch=1.2,
                duration_ms=200.0):
    rest, tets = fs.box_tets(*size, edge)
    m = fs.FastModel.from_tets(rest, tets, E, nu, density=fs.DENSITY_G_PER_MM3 * mass_scale)
    L = size[0]
    left = np.flatnonzero(rest[:, 0] < 1e-9)
    right = np.flatnonzero(rest[:, 0] > L - 1e-9)
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[left, 0] = fixed[right, 0] = True
    corner = left[np.argmin(np.linalg.norm(rest[left, 1:], axis=1))]
    fixed[corner, 1] = fixed[corner, 2] = True
    up = left[np.argmin(np.linalg.norm(rest[left, 1:] - [0.0, size[2]], axis=1))]
    fixed[up, 1] = True  # stops rotation about the pull axis
    m.set_fixed(fixed)
    n = int(round(duration_ms / h))

    def stretch(k):
        return 1.0 + (max_stretch - 1.0) * k / n

    def targets(x, k):
        x[right, 0] = L * stretch(k)

    def measure(x, k):
        _e, g = m.energy(x)
        s = stretch(k)
        exact = size[1] * size[2] * s * E * 0.5 * (s * s - 1.0)
        return {"stretch": s, "force_n": float(g[right, 0].sum()), "exact_n": float(exact)}

    rows = _frames(m, rest.copy(), h, solver, iters, n, targets, measure)
    tail = rows[n // 3:]
    err = np.array([abs(r["force_n"] / r["exact_n"] - 1.0) for r in tail])
    return rows, {"force_error_median": float(np.median(err)), "force_error_max": float(err.max()),
                  "final_force_ratio": rows[-1]["force_n"] / rows[-1]["exact_n"]}


# -- peel -------------------------------------------------------------------
def peel_case(h, solver, iters, mass_scale=1.0, length=25.0, width=5.0, edge=0.25, tab=4.0, lift=16.0, speed_mm_per_s=10.0):
    p = pm.SkinParams()
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    bonded = np.flatnonzero(rest[:, 0] >= tab - 1e-9)
    sheet = pm.Sheet(rest, tris, bonded, p)
    m = fs.FastModel.from_sheet(sheet, density=fs.DENSITY_G_PER_MM3 * mass_scale)
    grip = idx[0]
    mid = idx[:, idx.shape[1] // 2]
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[grip, 2] = True
    m.set_fixed(fixed)
    rate = speed_mm_per_s / 1000.0  # mm per ms
    n = int(round(lift / (rate * h)))

    def targets(x, k):
        x[grip, 2] = rate * h * k

    def measure(x, k):
        _e, g = m.energy(x)
        _hist, broken = m.bond_state()
        intact = ~broken
        front = float(rest[bonded[intact], 0].min()) if intact.any() else length
        line = x[mid][:, [0, 2]]
        curv = ps._centerline_curvature(line)
        near = np.abs(rest[mid][1:-1, 0] - front) <= 2.0
        return {"lift_mm": rate * h * k, "force_n": float(g[grip, 2].sum()), "front_mm": front,
                "front_curvature_per_mm": float(curv[near].max()) if near.any() else float("nan")}

    rows = _frames(m, rest.copy(), h, solver, iters, n, targets, measure)
    s = ps.summarize(rows, p, width, tab, length)
    out = {"expected_force_n": s["expected_force_n"]}
    if "steady_force_n" in s:
        out.update({k: s[k] for k in ("steady_force_n", "force_ratio", "curvature_ratio",
                                      "front_advance_per_mm_lift", "force_spread_n")})
    else:
        out["note"] = "front never reached steady peel"
    return rows, out


def peel_coarse_case(h, solver, iters, mass_scale=1.0):
    return peel_case(h, solver, iters, mass_scale, edge=0.5)


SCENARIOS = {"sheet": sheet_case, "volume": volume_case, "peel": peel_case, "peel_coarse": peel_coarse_case}


def race(scenarios, steps_ms, contenders=CONTENDERS, verbose=True):
    results = []
    for name in scenarios:
        for h in steps_ms:
            for label, solver, iters, mass_scale in contenders:
                t0 = time.perf_counter()
                rows, summary = SCENARIOS[name](h, solver, iters, mass_scale)
                ms = np.array([r["ms"] for r in rows])
                entry = {"scenario": name, "frame_ms": h, "solver": label, "frames": len(rows),
                         "solve_ms_median": float(np.median(ms)),
                         "factor_ms_median": float(np.median([r["factor_ms"] for r in rows])),
                         "assemble_ms_median": float(np.median([r["assemble_ms"] for r in rows])), "solve_ms_p95": float(np.percentile(ms, 95)),
                         "real_time": bool(np.percentile(ms, 95) <= h), "wall_s": time.perf_counter() - t0,
                         **summary}
                results.append(entry)
                if verbose:
                    acc = (f"force err median {100 * summary['force_error_median']:.2f}%"
                           if "force_error_median" in summary else
                           f"steady/Kendall {summary.get('force_ratio', float('nan')):.3f}  "
                           f"curl {summary.get('curvature_ratio', float('nan')):.2f}")
                    print(f"{name:11s} {h:g} ms  {label:24s}  solve {entry['solve_ms_median']:7.3f} ms "
                          f"(p95 {entry['solve_ms_p95']:.3f}, factor {entry['factor_ms_median']:.2f}, "
                          f"assemble {entry['assemble_ms_median']:.2f})  {acc}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--steps", nargs="+", type=float, default=[1.0, 5.0])
    parser.add_argument("--factor", choices=list(FACTORS),
                        default="cholmod" if fs.have_cholmod() else "eigen")
    parser.add_argument("--solvers", nargs="+", help="contender labels to run (default: all)")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "resources" / "soft_tissue" / "solver_race")
    args = parser.parse_args()
    global FACTOR
    FACTOR = FACTORS[args.factor]
    contenders = [c for c in CONTENDERS if not args.solvers or c[0] in args.solvers]
    results = race(args.scenarios, args.steps, contenders)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"results_{args.factor}.json" if args.factor != "eigen" else args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
