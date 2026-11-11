"""Run every bench fixture on every material, with the slow and the fast solver.

Fixtures (each sized from the material's own regime, so its breaking zone is
resolved by the mesh):
  peel     90-degree peel off its attachment; steady force vs the general Kendall answer
  wedge    sharp elevator driven under it along bone; energy balance and push
  cut      incision under small tension; released energy between the Griffith
           answers for its soft (E) and stiffened (E + k1) moduli
  stretch  uniaxial pull to rupture; exact J-curve, failure at its strain

Slow: the Python reference (quasi-static, converged every step).
Fast: the C++ Newton solver. Peel and wedge are driven like a hand, in 5 ms
frames, two ways:
  converged  Newton iterates each frame until converged (up to 20): the
             correctness check against theory and the slow solver
  budget     3 iterations per frame: what the real-time budget buys; thin,
             stiff layers on fine meshes need more (recorded, not asserted)
Cut and stretch are static.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/material_bench.py [--materials ...] [--tier quick|full]
Results: resources/soft_tissue/materials/<material>.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cut_test as ct  # noqa: E402
import fast_solver as fs  # noqa: E402
import peel_model as pm  # noqa: E402
import peel_strip as ps  # noqa: E402
import stretch_test as st  # noqa: E402
import tissue_params as tp  # noqa: E402
import wedge_test as wt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "resources" / "soft_tissue" / "materials"
FRAME_MS = 5.0
BUDGET_ITERS, CONVERGED_ITERS, CONVERGED_GTOL = 3, 20, 1e-7


# -- peel -------------------------------------------------------------------------------
def peel_geometry(p, tier="full"):
    """Sized from both lengths of the peel front: the bending zone (4D/k)^(1/4), resolved
    by the mesh, and the failure separation dc, the length of still-stretching bonds
    hanging under the lifted arm. Steady peel needs both inside the strip."""
    z = tp.zone_length(p)
    dc = p.delta_c_mm
    edge = min(0.5, z / 3.0)
    tab = 3.0 * z
    length = tab + 2.0 * dc + 12.0 * z
    f = tp.kendall_general(p)
    s = _stretch_at_force(p, f)
    return {"zone": z, "delta_c": dc, "edge": edge, "width": 4 * edge, "tab": tab, "length": length,
            "arm_stretch": s, "lift": s * length + 1.5 * dc, "kendall_per_width": f}


def _stretch_at_force(p, f):
    lo, hi = 1.0, 4.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if st.exact_force(p, mid, 1.0) < f else (lo, mid)
    return 0.5 * (lo + hi)


def _steady(front, g):
    """Once bonds have started breaking (the hanging bonds are formed) and before the far end is near."""
    return (front >= g["tab"] + 2.0 * g["zone"]) & (front <= g["length"] - 4.0 * g["zone"])


def peel_slow(p, g):
    rows, _snaps, _sheet = ps.run_peel(p, g["length"], g["width"], g["edge"], g["tab"], g["lift"], g["edge"] / 2.0,
                                       verbose=False, stop_margin_mm=3.0 * g["zone"])
    front = np.array([r["front_mm"] for r in rows])
    force = np.array([r["force_n"] for r in rows])
    return front, force


def peel_fast(p, g, speed_mm_per_s=10.0, iters=CONVERGED_ITERS, gtol=CONVERGED_GTOL):
    rest, tris, idx = pm.strip_mesh(g["length"], g["width"], g["edge"])
    bonded = np.flatnonzero(rest[:, 0] >= g["tab"] - 1e-9)
    sheet = pm.Sheet(rest, tris, bonded, p)
    m = fs.FastModel.from_sheet(sheet)
    grip = idx[0]
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[grip, 2] = True
    m.set_fixed(fixed)
    m.set_options(project_psd=True, threads=4)
    x, v = rest.copy(), np.zeros_like(rest)
    rate = speed_mm_per_s / 1000.0 * FRAME_MS
    fronts, forces, ms = [], [], []
    for k in range(1, int(g["lift"] / rate) + 1):
        x[grip, 2] = rate * k
        stats = m.step(x, v, FRAME_MS, fs.NEWTON, iters, gtol=gtol, damping=0.1)
        m.commit(x)
        ms.append(stats["ms"])
        _e, gr = m.energy(x)
        broken = m.bond_state()[1]
        front = float(rest[bonded[~broken], 0].min()) if (~broken).any() else g["length"]
        fronts.append(front)
        forces.append(float(gr[grip, 2].sum()))
        if front >= g["length"] - 3.0 * g["zone"]:
            break
    return np.array(fronts), np.array(forces), float(np.median(ms))


def run_peel(name, tier="full"):
    p = tp.skin_params(name)
    g = peel_geometry(p, tier)
    expected = g["kendall_per_width"] * g["width"]
    out = {"geometry": g, "expected_force_n": expected}
    if tier == "full":
        t0 = time.perf_counter()
        front, force = peel_slow(p, g)
        sel = _steady(front, g)
        out["slow"] = {"steady_rows": int(sel.sum()), "seconds": time.perf_counter() - t0,
                       "steady_force_n": float(np.median(force[sel])) if sel.sum() >= 5 else float("nan")}
        out["slow"]["force_ratio"] = out["slow"]["steady_force_n"] / expected
    for label, iters, gtol in (("fast", CONVERGED_ITERS, CONVERGED_GTOL), ("fast_budget", BUDGET_ITERS, 1e-9)):
        t0 = time.perf_counter()
        front, force, ms = peel_fast(p, g, iters=iters, gtol=gtol)
        sel = _steady(front, g)
        out[label] = {"steady_rows": int(sel.sum()), "seconds": time.perf_counter() - t0, "ms_per_frame": ms,
                      "steady_force_n": float(np.median(force[sel])) if sel.sum() >= 5 else float("nan")}
        out[label]["force_ratio"] = out[label]["steady_force_n"] / expected
    if "slow" in out:
        out["fast_over_slow"] = out["fast"]["steady_force_n"] / out["slow"]["steady_force_n"]
    return out


# -- wedge (sharp elevator) -----------------------------------------------------------------
def wedge_geometry(name, p, tier="full"):
    z = tp.zone_length(p)
    edge = min(0.2, z / 3.0)
    length = min(9.0, max(3.0, 40.0 * z)) if tier == "full" else min(6.0, max(2.5, 25.0 * z))
    tab = 0.5 * length / 9.0 + edge
    iface = tp.wedge_interface(tp.SOLO_ATTACHMENT[name])
    blade = wt.Blade(height_mm=0.1, slope=0.1, heel_mm=0.5, sharp=True, edge_reach_mm=max(edge / 2, 0.02),
                     G_edge_n_per_mm=iface.opening.G_n_per_mm)
    return {"edge": edge, "width": 4 * edge, "length": length, "tab": tab, "travel": length - tab - 2 * edge,
            "step": edge / 5.0}, iface, blade


def _wedge_steady(tip, front, total, g):
    span = g["length"] - g["tab"]
    sel = (front >= g["tab"] + 0.25 * span) & (front <= g["length"] - 0.2 * span)
    if sel.sum() < 5:
        return float("nan"), int(sel.sum())
    return float(np.polyfit(tip[sel], total[sel], 1)[0]), int(sel.sum())


def wedge_slow(p, g, iface, blade):
    rows, _snaps, rig = wt.run_wedge(p, iface, blade, length_mm=g["length"], width_mm=g["width"], edge_mm=g["edge"],
                                     tab_mm=g["tab"], travel_mm=g["travel"], step_mm=g["step"], verbose=False)
    tip = np.array([r["tip_mm"] for r in rows])
    front = np.array([r["front_mm"] for r in rows])
    total = np.array([r["contact_work_n_mm"] + r["cut_work_n_mm"] for r in rows])
    push, n = _wedge_steady(tip, front, total, g)
    last = rows[-1]
    balance = (last["contact_work_n_mm"] + last["cut_work_n_mm"]) / max(last["stored_n_mm"] + last["cut_work_n_mm"], 1e-15)
    return {"steady_push_n": push, "steady_rows": n, "energy_balance_ratio": balance,
            "cut_work_share": last["cut_work_n_mm"] / max(last["contact_work_n_mm"] + last["cut_work_n_mm"], 1e-15)}


def wedge_fast(p, g, iface, blade, speed_mm_per_s=5.0, iters=CONVERGED_ITERS, gtol=CONVERGED_GTOL):
    rest, tris, _idx = pm.strip_mesh(g["length"], g["width"], g["edge"])
    rest[:, 2] = 0.5 * p.thickness_mm
    sheet = pm.Sheet(rest, tris, np.zeros(0, dtype=np.int64), p, floor_z=0.5 * p.thickness_mm)
    bonded = np.flatnonzero(rest[:, 0] >= g["tab"] - 1e-9)
    m = fs.FastModel.from_sheet(sheet)
    law = [iface.opening.G_n_per_mm, iface.opening.delta0_mm, iface.opening.delta_c_mm,
           iface.sliding.G_n_per_mm, iface.sliding.delta0_mm, iface.sliding.delta_c_mm]
    m.set_bonds(bonded, rest[bonded], sheet.vert_area[bonded], law, mixed=True)
    m.set_blade(blade.contact_n_per_mm3 * sheet.vert_area, blade, sheet.floor_z)
    m.set_options(project_psd=True, threads=4)
    x, v = rest.copy(), np.zeros_like(rest)
    rate = speed_mm_per_s / 1000.0 * FRAME_MS
    weight = sheet.vert_area[bonded]
    work = cut = 0.0
    prev = 0.0
    tips, fronts, totals, ms = [], [], [], []
    for k in range(1, int(g["travel"] / rate) + 1):
        tip = rate * k
        m.set_blade_tip(tip)
        broken = m.bond_state()[1]
        reach = ~broken & (rest[bonded, 0] <= tip + blade.edge_reach_mm)
        if reach.any():  # the sharp edge severs what it reaches
            m.set_bond_broken(reach)
            cut += blade.G_edge_n_per_mm * float(weight[reach].sum())
        stats = m.step(x, v, FRAME_MS, fs.NEWTON, iters, gtol=gtol, damping=0.1)
        m.commit(x)
        ms.append(stats["ms"])
        push = m.blade_push(x)
        work += 0.5 * (push + prev) * rate
        prev = push
        broken = m.bond_state()[1]
        tips.append(tip)
        fronts.append(float(rest[bonded[~broken], 0].min()) if (~broken).any() else np.inf)
        totals.append(work + cut)
    steady, n = _wedge_steady(np.array(tips), np.array(fronts), np.array(totals), g)
    return {"steady_push_n": steady, "steady_rows": n, "ms_per_frame": float(np.median(ms))}


def run_wedge(name, tier="full"):
    p = tp.skin_params(name)
    g, iface, blade = wedge_geometry(name, p, tier)
    out = {"geometry": g, "edge_cost_push_n": blade.G_edge_n_per_mm * g["width"],
           "blunt_can_free": bool(iface.opening.delta_c_mm < 0.6 * 1.0),
           "opening_failure_separation_mm": iface.opening.delta_c_mm}
    if tier == "full":
        t0 = time.perf_counter()
        out["slow"] = wedge_slow(p, g, iface, blade)
        out["slow"]["seconds"] = time.perf_counter() - t0
    for label, iters, gtol in (("fast", CONVERGED_ITERS, CONVERGED_GTOL), ("fast_budget", BUDGET_ITERS, 1e-9)):
        t0 = time.perf_counter()
        out[label] = wedge_fast(p, g, iface, blade, iters=iters, gtol=gtol)
        out[label]["seconds"] = time.perf_counter() - t0
    if "slow" in out:
        out["fast_over_slow"] = out["fast"]["steady_push_n"] / out["slow"]["steady_push_n"]
    return out


# -- cut ---------------------------------------------------------------------------------------
def run_cut(name, tier="full"):
    p = tp.skin_params(name)
    kw = dict(half_len=2.0, strain=0.002, size_mm=16.0, edge_mm=0.5)
    out = {}
    for label, solve in (("slow", None), ("fast", fs.static_solve)):
        t0 = time.perf_counter()
        r = ct.run_incision(p, solve=solve, **kw)
        out[label] = {"released_n_mm": r["released_n_mm"], "expected_soft_n_mm": r["expected_released_n_mm"],
                      "ratio_to_soft": r["released_n_mm"] / r["expected_released_n_mm"],
                      "min_opening_mm": r["min_opening_mm"], "seconds": time.perf_counter() - t0}
    out["stiffened_over_soft"] = p.E_mpa / (p.E_mpa + p.k1_mpa)
    # The mesh's own undershoot at a crack tip: the same plate without stiffening.
    iso = pm.SkinParams(**{**p.__dict__, "k1_mpa": 0.0, "k2": 0.0})
    r = ct.run_incision(iso, solve=fs.static_solve, **kw)
    out["mesh_factor"] = r["released_n_mm"] / r["expected_released_n_mm"]
    out["fast_over_slow"] = out["fast"]["released_n_mm"] / out["slow"]["released_n_mm"]
    return out


# -- stretch --------------------------------------------------------------------------------------
def run_stretch(name, tier="full"):
    m = tp.MATERIALS[name]
    p = tp.skin_params(name)
    e_fail = m.fail_green.nominal
    s_fail = math.sqrt(1.0 + 2.0 * e_fail)
    step = (s_fail - 1.0) / (30.0 if tier == "quick" else 60.0)
    kw = dict(length_mm=4.0, width_mm=1.0, edge_mm=0.25, fail_strain=e_fail)
    out = {"fail_stretch": s_fail, "step": step}
    for label, solve in (("slow", None), ("fast", fs.static_solve)):
        t0 = time.perf_counter()
        rows, strip, broken_at = st.run_stretch(p, max_stretch=s_fail + 4 * step, step=step, solve=solve, **kw)
        sm = st.summarize(rows, p, strip, broken_at)
        out[label] = {k: sm[k] for k in ("max_force_error_before_break", "peak_force_n", "exact_peak_force_n",
                                          "broke_at_stretch", "separated", "force_after_n", "work_to_rupture_n_mm",
                                          "exact_work_to_rupture_n_mm", "small_strain_stiffness_n",
                                          "exact_small_strain_stiffness_n")}
        out[label]["seconds"] = time.perf_counter() - t0
    return out


FIXTURES = {"peel": run_peel, "wedge": run_wedge, "cut": run_cut, "stretch": run_stretch}


def run_material(name, fixtures=FIXTURES, tier="full", verbose=True):
    out = {"material": name, "attachment": tp.SOLO_ATTACHMENT[name], "tier": tier,
           "regime": tp.regime(name).__dict__}
    for f in fixtures:
        t0 = time.perf_counter()
        out[f] = FIXTURES[f](name, tier)
        if verbose:
            print(f"{name:18s} {f:8s} {time.perf_counter() - t0:7.1f} s  " + _short(f, out[f]), flush=True)
    return out


def _short(f, r):
    if f == "peel":
        slow = f"slow/Kendall {r['slow']['force_ratio']:.3f}  " if "slow" in r else ""
        return (slow + f"fast/Kendall {r['fast']['force_ratio']:.3f} ({r['fast']['ms_per_frame']:.1f} ms/frame)  "
                f"budget/Kendall {r['fast_budget']['force_ratio']:.3f} ({r['fast_budget']['ms_per_frame']:.1f} ms)  "
                f"(strip {r['geometry']['length']:.1f} mm, mesh {r['geometry']['edge']:.3f} mm, dc {r['geometry']['delta_c']:.2f} mm)")
    if f == "wedge":
        slow = (f"balance {r['slow']['energy_balance_ratio']:.3f}  slow push/edge cost "
                f"{r['slow']['steady_push_n'] / r['edge_cost_push_n']:.3f}  fast/slow {r['fast_over_slow']:.3f}  "
                if "slow" in r else "")
        return (slow + f"fast push/edge cost {r['fast']['steady_push_n'] / r['edge_cost_push_n']:.3f}  "
                f"budget {r['fast_budget']['steady_push_n'] / r['edge_cost_push_n']:.3f} ({r['fast_budget']['ms_per_frame']:.1f} ms)  "
                f"blunt can free: {r['blunt_can_free']}")
    if f == "cut":
        return (f"release/Griffith(E) {r['slow']['ratio_to_soft']:.3f} (bounds {r['mesh_factor'] * r['stiffened_over_soft']:.3f}-"
                f"{r['mesh_factor']:.3f})  "
                f"fast/slow {r['fast_over_slow']:.6f}")
    return (f"J-curve err {r['slow']['max_force_error_before_break']:.1e}  broke {r['slow']['broke_at_stretch']:.4f} "
            f"(fail {r['fail_stretch']:.4f})  fast broke {r['fast']['broke_at_stretch']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--materials", nargs="+", default=list(tp.MATERIALS))
    ap.add_argument("--fixtures", nargs="+", default=list(FIXTURES))
    ap.add_argument("--tier", choices=("quick", "full"), default="full")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.materials:
        res = run_material(name, args.fixtures, args.tier)
        (OUT / f"{name}_{args.tier}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
