"""Composite (layered) tissue tests: which plane lets go, and which layers move.

Stacks come from tissue_params.STACKS (outside to inside, on a rigid substrate).
Every case runs on the fast C++ Newton solver (converged 5 ms frames, as a hand
would drive it); the full tier also runs the slow Python reference.

Mastoid (skin / TPF / periosteum on bone)
  plane_selection   grip the skin: skin and TPF peel off periosteum together (the
                    weak areolar plane), at the Kendall force for a two-layer arm
  full_thickness    grip all three: they peel off bone together, at the Kendall
                    force for the three-layer arm with the periosteum-bone energy
  elevator          sharp elevator under periosteum: the flap comes up full
                    thickness, push = edge cost x width, layer planes intact
  incision          stretch the stack with skin and TPF cut, periosteum intact:
                    the grip force drops, but less than with the planes unbonded
                    or a full-thickness cut (the periosteum bridges the cut)
  grasp_periosteum  skin flap already elevated; grasp periosteum only: it peels off
                    bone, the skin and TPF above do not move
Temporal (skin / TPF / temporalis fascia on a rigid muscle stand-in)
  plane_selection   skin and TPF peel off temporalis fascia, which stays down
Bony canal (canal skin fused to periosteum, on bone)
  elevator          the two lift as one: the fused plane stays intact
Auricle (skin / perichondrium on cartilage)
  lap_shear         slide the layers along the substrate: perichondrium on cartilage
                    lets go at the Kendall-at-0-degrees force for its sliding
                    energy, below periosteum on bone
  plane_selection   grip the skin: the weaker of skin-perichondrium and
                    perichondrium-cartilage (opening) lets go

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/composite_bench.py [--tier quick|full] [--cases ...]
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
import stack as sk  # noqa: E402
import tissue_params as tp  # noqa: E402
import wedge_test as wt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "resources" / "soft_tissue" / "composites"
FRAME_MS, ITERS, GTOL = 5.0, 20, 1e-7


# -- drivers ---------------------------------------------------------------------------------
class Lift:
    """Grip some vertices and raise them; record force, damage and how far each layer moved."""

    def __init__(self, st: sk.Stack, grip, solver, speed_mm_per_s=10.0, step_mm=None):
        self.st, self.grip, self.solver = st, np.asarray(grip), solver
        self.fixed = np.zeros(st.rest.shape, dtype=bool)
        self.fixed[self.grip, 2] = True
        self.x = st.rest.copy()
        self.rate = speed_mm_per_s / 1000.0 * FRAME_MS if solver == "fast" else step_mm
        if solver == "fast":
            self.m = fs.FastModel.from_stack(st)
            self.m.set_fixed(self.fixed)
            self.m.set_options(project_psd=True, threads=4)
            self.v = np.zeros_like(self.x)
        self.k = 0

    def advance(self):
        self.k += 1
        self.x[self.grip, 2] = self.rate * self.k
        if self.solver == "fast":
            self.m.step(self.x, self.v, FRAME_MS, fs.NEWTON, ITERS, gtol=GTOL, damping=0.1)
            self.m.commit(self.x)
            hist, self.st.pair_broken = self.m.pair_state()
            self.st.broken = self.m.bond_state()[1]
            _e, g = self.m.energy(self.x)
        else:
            self.x, _res = pm.solve(self.st, self.x, self.fixed, gtol=1e-8)
            self.st.commit(self.x)
            _e, g = self.st.energy(self.x)
        return float(g[self.grip, 2].sum())

    def lift(self):
        return self.rate * self.k

    def layer_rise(self, k, mask=None):
        z = self.x[self.st.layer_slice(k), 2] - self.st.floor_z
        return float(z[mask].max() if mask is not None else z.max())


def _front(st, plane, rest_x, tab):
    """First intact bond of a plane (plane = L-1 means the substrate attachment)."""
    if plane == st.L - 1:
        ok = ~st.broken
        xs = rest_x[st.ids - (st.L - 1) * st.n]
    else:
        sel = st.pair_law == plane
        ok = ~st.pair_broken[sel]
        xs = rest_x[st.pair_a[sel] - plane * st.n]
    keep = ok & (xs >= tab - 1e-9)
    return float(xs[keep].min()) if keep.any() else float(rest_x.max())


def peel_case(stack_name, grab_layers, plane, arm_layers, G, law, solver, tier, extra=None):
    """Grip the x = 0 edge of ``grab_layers``; peel along ``plane`` (index; L-1 = substrate)."""
    spec = tp.STACKS[stack_name]
    params = [tp.skin_params(m, spec["attachment"]) for m in spec["layers"]]
    arm = [params[i] for i in arm_layers]
    z = tp.zone_layers(arm, law)
    dc = law.delta_c_mm
    edge = min(0.5, z / 3.0)
    tab = 3.0 * z
    length = tab + 2.0 * dc + (8.0 if tier == "quick" else 12.0) * z
    width = 4 * edge
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    rx = rest[:, 0]
    L = len(params)
    # Everything is bonded beyond the tab; inside it, only the planes the grip does not cross.
    pair_masks = [np.ones(len(rest), bool) if (k < plane and k + 1 in grab_layers and k in grab_layers)
                  else rx >= tab - 1e-9 for k in range(L - 1)]
    attach_mask = rx >= tab - 1e-9
    if extra:
        pair_masks, attach_mask = extra(rx, pair_masks, attach_mask, tab)
    st = sk.from_table(stack_name, rest, tris, pair_masks=pair_masks, attach_mask=attach_mask)
    expected = tp.kendall_layers(arm, G) * width
    grip = np.concatenate([st.layer_vertices(k, idx[0]) for k in grab_layers])
    s_arm = 1.0 + 0.5 * expected / width / max(sum(p.thickness_mm * p.E_mpa for p in arm), 1e-9)
    s_arm = min(s_arm, 2.0)
    total = length * s_arm + 1.5 * dc
    drv = Lift(st, grip, solver, step_mm=edge / 2.0)
    rows = []
    t0 = time.perf_counter()
    while drv.lift() < total:
        f = drv.advance()
        front = _front(st, plane, rx, tab)
        rows.append((drv.lift(), f, front))
        if front >= length - 3.0 * z:
            break
    rows = np.array(rows)
    steady = (rows[:, 2] >= tab + 2.0 * z) & (rows[:, 2] <= length - 4.0 * z)
    out = {"expected_force_n": expected, "zone_mm": z, "delta_c_mm": dc, "length_mm": length, "edge_mm": edge,
           "steady_rows": int(steady.sum()), "seconds": time.perf_counter() - t0,
           "steady_force_n": float(np.median(rows[steady, 1])) if steady.sum() >= 5 else float("nan")}
    out["force_ratio"] = out["steady_force_n"] / expected
    peeled = rx <= rows[-1, 2] - 1e-9
    out["planes_broken_in_peeled_region"] = [st.plane_broken_fraction(k, peeled & (rx >= tab)) for k in range(L - 1)]
    out["attachment_broken_in_peeled_region"] = st.attach_broken_fraction(peeled & (rx >= tab))
    out["layer_rise_mm"] = [drv.layer_rise(k) for k in range(L)]
    out["lift_mm"] = drv.lift()
    return out


def mastoid_plane_selection(solver, tier):
    i = tp.INTERFACES
    law = i["areolar"].opening.bond_law()
    return peel_case("mastoid", [0], 1, [0, 1], law.G_n_per_mm, law, solver, tier)


def mastoid_full_thickness(solver, tier):
    """Bone bonds have separate opening and sliding channels; at 90 degrees the bonds still
    hanging under the arm are also slid sideways, so the energy spent per area lies between
    G_I (pure opening) and G_I + G_II (both channels used up). Reported against both."""
    iface = tp.wedge_interface("periosteum_bone")
    law = iface.opening
    out = peel_case("mastoid", [0, 1, 2], 2, [0, 1, 2], law.G_n_per_mm, law, "fast" if solver == "fast" else solver, tier)
    spec = tp.STACKS["mastoid"]
    arm = [tp.skin_params(m, spec["attachment"]) for m in spec["layers"]]
    width = out["expected_force_n"] / tp.kendall_layers(arm, law.G_n_per_mm)
    out["upper_bound_force_n"] = tp.kendall_layers(arm, law.G_n_per_mm + iface.sliding.G_n_per_mm) * width
    out["upper_ratio"] = out["steady_force_n"] / out["upper_bound_force_n"]
    return out


def temporal_plane_selection(solver, tier):
    law = tp.INTERFACES["areolar"].opening.bond_law()
    return peel_case("temporal", [0], 1, [0, 1], law.G_n_per_mm, law, solver, tier)


def auricle_plane_selection(solver, tier):
    """Grip the skin: the weaker opening energy of the two planes lets go."""
    a = tp.INTERFACES["skin_perichondrium"].opening.bond_law()
    b = tp.INTERFACES["perichondrium_cartilage"].opening.bond_law()
    if a.G_n_per_mm <= b.G_n_per_mm:
        return {"weaker": "skin_perichondrium", **peel_case("auricle", [0], 0, [0], a.G_n_per_mm, a, solver, tier)}
    return {"weaker": "perichondrium_cartilage", **peel_case("auricle", [0], 1, [0, 1], b.G_n_per_mm, b, solver, tier)}


def grasp_periosteum(solver, tier):
    """Skin flap already elevated over the first part; grasp only the periosteum edge and lift."""
    spec = tp.STACKS["mastoid"]
    params = [tp.skin_params(m, spec["attachment"]) for m in spec["layers"]]
    law = tp.INTERFACES["periosteum_bone"].opening.bond_law()
    z = tp.zone_layers([params[2]], law)
    edge = min(0.2, z / 2.0)
    length, width, tab = 6.0, 4 * edge, 0.5
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    rx = rest[:, 0]
    elevated = 4.0  # the skin flap (skin + TPF) is off the periosteum up to here
    st = sk.from_table("mastoid", rest, tris, pair_masks=[np.ones(len(rest), bool), rx >= elevated],
                       attach_mask=rx >= tab)
    grip = st.layer_vertices(2, idx[0])
    drv = Lift(st, grip, solver, speed_mm_per_s=5.0, step_mm=edge / 2.0)
    forces = []
    t0 = time.perf_counter()
    while drv.lift() < 1.5:
        forces.append(drv.advance())
    return {"lift_mm": drv.lift(), "grasp_force_n": forces[-1], "force_history_n": forces[:: max(1, len(forces) // 20)],
            "periosteum_rise_mm": drv.layer_rise(2), "skin_rise_mm": drv.layer_rise(0), "tpf_rise_mm": drv.layer_rise(1),
            "attachment_broken": st.attach_broken_fraction(rx >= tab), "seconds": time.perf_counter() - t0}


# -- elevator under a stack -----------------------------------------------------------------------
def elevator_case(stack_name, solver, tier):
    spec = tp.STACKS[stack_name]
    params = [tp.skin_params(m, spec["attachment"]) for m in spec["layers"]]
    iface = tp.wedge_interface(spec["attachment"])
    z = tp.zone_layers(params, iface.opening)
    edge = min(0.2, z / 3.0)
    length = 5.0 if tier == "quick" else 8.0
    width, tab = 4 * edge, 0.6
    rest, tris, _idx = pm.strip_mesh(length, width, edge)
    rx = rest[:, 0]
    blade = wt.Blade(height_mm=0.1, slope=0.1, heel_mm=0.5, sharp=True, edge_reach_mm=max(edge / 2, 0.02),
                     G_edge_n_per_mm=iface.opening.G_n_per_mm)
    st = sk.from_table(stack_name, rest, tris, attach_mask=rx >= tab, blade=blade)
    travel = length - tab - 1.0
    rate = 5.0 / 1000.0 * FRAME_MS if solver == "fast" else edge / 5.0
    x = st.rest.copy()
    weight = st.weight
    fixed = np.zeros(x.shape, dtype=bool)
    if solver == "fast":
        m = fs.FastModel.from_stack(st)
        m.set_options(project_psd=True, threads=4)
        v = np.zeros_like(x)
    work = cut = prev = 0.0
    tips, totals, fronts = [], [], []
    t0 = time.perf_counter()
    for k in range(1, int(travel / rate) + 1):
        tip = rate * k
        st.tip = tip
        reach = ~st.broken & (st.rest[st.ids, 0] <= tip + blade.edge_reach_mm)
        if reach.any():
            st.broken |= reach
            cut += blade.G_edge_n_per_mm * float(weight[reach].sum())
        if solver == "fast":
            m.set_blade_tip(tip)
            m.set_bond_broken(st.broken)
            m.step(x, v, FRAME_MS, fs.NEWTON, ITERS, gtol=GTOL, damping=0.1)
            m.commit(x)
            st.broken = m.bond_state()[1]
            st.pair_broken = m.pair_state()[1]
            push = m.blade_push(x)
        else:
            x, _res = pm.solve(st, x, fixed, gtol=1e-8)
            st.commit(x)
            push = st.push_force(x)
        work += 0.5 * (push + prev) * rate
        prev = push
        tips.append(tip)
        totals.append(work + cut)
        fronts.append(float(rx[st.ids[~st.broken] - (st.L - 1) * st.n].min()) if (~st.broken).any() else length)
    tips, totals, fronts = map(np.array, (tips, totals, fronts))
    sel = (fronts >= tab + 0.5) & (fronts <= length - 1.5)
    push_ss = float(np.polyfit(tips[sel], totals[sel], 1)[0]) if sel.sum() >= 5 else float("nan")
    passed = rx <= tips[-1] - 0.2
    return {"steady_push_n": push_ss, "edge_cost_push_n": blade.G_edge_n_per_mm * width,
            "push_over_edge_cost": push_ss / (blade.G_edge_n_per_mm * width),
            "attachment_broken_passed": st.attach_broken_fraction(passed & (rx >= tab)),
            "planes_broken_passed": [st.plane_broken_fraction(k, passed) for k in range(st.L - 1)],
            "layer_rise_mm": [float((x[st.layer_slice(k), 2] - st.floor_z)[passed].min()) for k in range(st.L)],
            "seconds": time.perf_counter() - t0}


def mastoid_elevator(solver, tier):
    return elevator_case("mastoid", solver, tier)


def bony_canal_elevator(solver, tier):
    return elevator_case("bony_canal", solver, tier)


# -- incision under tension ------------------------------------------------------------------------
WELD = pm.BondLaw(1e3, 1e-4, 1e3)  # stiff and unbreakable: joins a split the incision did not cut


def _incision_stack(cut_layers, unbond_planes=(), size=12.0, edge=0.5, half=2.0):
    spec = tp.STACKS["mastoid"]
    base, btris, _idx = ct.plate_mesh(size, edge)
    cent = base[btris].mean(axis=1)
    upper = cent[:, 1] > 0.0
    line = np.flatnonzero(np.abs(base[:, 1]) < 1e-9)
    inside = line[(base[line, 0] > -half + 1e-9) & (base[line, 0] < half - 1e-9)] if cut_layers else np.zeros(0, int)
    rest, tris, parents = ct.split_vertices(base, btris, inside, upper)
    nb = len(base)
    copies = nb + np.arange(len(inside))
    welds = [(k, inside, copies, WELD) for k in range(len(spec["layers"])) if k not in cut_layers]
    masks = [np.zeros(len(rest), bool) if k in unbond_planes else np.ones(len(rest), bool)
             for k in range(len(spec["layers"]) - 1)]
    st = sk.from_table("mastoid", rest, tris, pair_masks=masks, attachment=None, extra_pairs=welds)
    return st, rest, size


def _stretch_stack(st, rest, size, strain, solver):
    half = 0.5 * size
    top = np.flatnonzero(rest[:, 1] > half - 1e-9)
    bottom = np.flatnonzero(rest[:, 1] < -half + 1e-9)
    x = st.rest.copy()
    fixed = np.zeros(x.shape, dtype=bool)
    fixed[:, 2] = True
    for k in range(st.L):
        t, b = st.layer_vertices(k, top), st.layer_vertices(k, bottom)
        x[t, 1] = half * (1 + strain)
        x[b, 1] = -half * (1 + strain)
        fixed[t, 1] = fixed[b, 1] = True
        fixed[st.layer_vertices(k, bottom[np.argmin(np.abs(rest[bottom, 0]))]), 0] = True
    if solver == "fast":
        m = fs.FastModel.from_stack(st)
        m.set_fixed(fixed)
        m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 200, gtol=1e-10)
        _e, g = m.energy(x)
    else:
        x, _r = pm.solve(st, x, fixed, gtol=1e-10)
        _e, g = st.energy(x)
    return float(sum(g[st.layer_vertices(k, top), 1].sum() for k in range(st.L)))


def mastoid_incision(solver, tier, strain=0.002):
    f0 = _stretch_stack(*_incision_stack([]), strain, solver)
    out = {"uncut_force_n": f0}
    for label, layers, unbond in (("skin_tpf_cut", [0, 1], ()), ("skin_tpf_cut_unbonded", [0, 1], (0, 1)),
                                  ("full_thickness_cut", [0, 1, 2], ())):
        f = _stretch_stack(*_incision_stack(layers, unbond), strain, solver)
        if unbond:
            f0u = _stretch_stack(*_incision_stack([], unbond), strain, solver)
            out[label + "_drop"] = 1.0 - f / f0u
        else:
            out[label + "_drop"] = 1.0 - f / f0
    return out


# -- lap shear (slipperiness) ------------------------------------------------------------------------
def lap_shear(stack_name, solver, tier):
    """Pull the innermost layer's edge along the substrate (0 degrees). Steady force =
    Kendall at 0 degrees with the sliding energy: sqrt(2 Eh G_II) for a linear arm."""
    spec = tp.STACKS[stack_name]
    inner = tp.skin_params(spec["layers"][-1], spec["attachment"])
    iface = tp.wedge_interface(spec["attachment"])
    law = iface.sliding
    expected_pw = tp.kendall_layers([inner], law.G_n_per_mm, angle_deg=0.0)
    s_arm = 1.0
    while tp.uniaxial_stress(inner, s_arm) * inner.thickness_mm < expected_pw and s_arm < 3:
        s_arm += 1e-4
    edge = 0.2
    # Slipping bonds spread over about dc / (arm strain): the strip must hold that zone twice over.
    slip_zone = law.delta_c_mm / max(s_arm - 1.0, 0.05)
    length = min(40.0, 0.5 + 2.5 * slip_zone + 3.0)
    width = 4 * edge
    rest, tris, idx = pm.strip_mesh(length, width, edge)
    rx = rest[:, 0]
    tab = 0.5
    st = sk.Stack(rest, tris, [inner], [], iface, attach_mask=rx >= tab, names=[spec["layers"][-1]])
    grip = idx[0]
    fixed = np.zeros(st.rest.shape, dtype=bool)
    fixed[grip, 0] = True
    fixed[:, 2] = True  # stays on the substrate: pure sliding
    x = st.rest.copy()
    rate = 2.0 / 1000.0 * FRAME_MS if solver == "fast" else 0.02
    if solver == "fast":
        m = fs.FastModel.from_stack(st)
        m.set_fixed(fixed)
        m.set_options(project_psd=True, threads=4)
        v = np.zeros_like(x)
    rows = []
    t0 = time.perf_counter()
    k = 0
    while True:
        k += 1
        x[grip, 0] = -rate * k
        if solver == "fast":
            m.step(x, v, FRAME_MS, fs.NEWTON, ITERS, gtol=GTOL, damping=0.1)
            m.commit(x)
            st.broken = m.bond_state()[1]
            _e, g = m.energy(x)
        else:
            x, _r = pm.solve(st, x, fixed, gtol=1e-9)
            st.commit(x)
            _e, g = st.energy(x)
        front = float(rx[st.ids[~st.broken]].min()) if (~st.broken).any() else length
        rows.append((rate * k, -float(g[grip, 0].sum()), front))
        if front >= length - 1.0 or rate * k > 3 * length:
            break
    rows = np.array(rows)
    steady = (rows[:, 2] >= tab + 0.5) & (rows[:, 2] <= length - 0.4 * slip_zone - 1.0)
    f_ss = float(np.median(rows[steady, 1])) if steady.sum() >= 5 else float("nan")
    return {"expected_force_n": expected_pw * width, "steady_force_n": f_ss, "force_ratio": f_ss / (expected_pw * width),
            "steady_rows": int(steady.sum()), "G_II": law.G_n_per_mm, "length_mm": length, "arm_stretch": s_arm,
            "slip_zone_mm": slip_zone,
            "seconds": time.perf_counter() - t0}


def auricle_lap_shear(solver, tier):
    return lap_shear("auricle", solver, tier)


def mastoid_lap_shear(solver, tier):
    """Periosteum on bone, for comparison with perichondrium on cartilage."""
    return lap_shear("mastoid", solver, tier)


CASES = {
    "mastoid_plane_selection": mastoid_plane_selection,
    "mastoid_full_thickness": mastoid_full_thickness,
    "mastoid_elevator": mastoid_elevator,
    "mastoid_incision": mastoid_incision,
    "grasp_periosteum": grasp_periosteum,
    "temporal_plane_selection": temporal_plane_selection,
    "bony_canal_elevator": bony_canal_elevator,
    "auricle_lap_shear": auricle_lap_shear,
    "mastoid_lap_shear": mastoid_lap_shear,
    "auricle_plane_selection": auricle_plane_selection,
}


def run(cases=CASES, tier="quick", verbose=True):
    out = {}
    for name in cases:
        out[name] = {}
        for solver in (("fast", "slow") if tier == "full" else ("fast",)):
            t0 = time.perf_counter()
            out[name][solver] = CASES[name](solver, tier)
            if verbose:
                r = out[name][solver]
                keys = [k for k in ("force_ratio", "push_over_edge_cost", "planes_broken_in_peeled_region",
                                    "attachment_broken_in_peeled_region", "planes_broken_passed",
                                    "attachment_broken_passed", "skin_tpf_cut_drop", "skin_tpf_cut_unbonded_drop",
                                    "full_thickness_cut_drop", "skin_rise_mm", "periosteum_rise_mm",
                                    "grasp_force_n", "layer_rise_mm") if k in r]
                print(f"{name:26s} {solver:4s} {time.perf_counter() - t0:7.1f} s  "
                      + "  ".join(f"{k}={_fmt(r[k])}" for k in keys), flush=True)
    return out


def _fmt(v):
    if isinstance(v, list):
        return "[" + ",".join(f"{a:.3g}" for a in v) + "]"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", choices=("quick", "full"), default="quick")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--name", default="", help="suffix for the results file (for runs split across processes)")
    args = ap.parse_args()
    res = run(args.cases, args.tier)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    (OUT / f"composites_{args.tier}{suffix}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
