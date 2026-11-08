"""Elevator wedge test: a rigid blade driven along bone under a bonded sheet.

Models flap elevation with a round knife, Rosen or Freer. The blade tip rides
on the bone surface (z = 0) and moves along +x. Behind the tip its top face
rises at a fixed slope to the blade thickness, then stays flat (the shaft).
The sheet cannot pass through the blade or the bone, and is free otherwise.

The sheet is held to bone by bonds with two channels:
  opening  separation normal to the bone (peel, mode I), its own G_I
  sliding  separation along the bone (shear, mode II), its own G_II
A bond whose opening or sliding passes its failure distance is broken in both.

A sharp edge (optional) severs the attachments it reaches, just ahead of the
tip, at a cost of G_edge per area. Otherwise the blade frees tissue only by
lifting it until the bonds fail.

Known answers it is checked against
  energy balance   work pushing the blade = elastic energy + bond work
  push force       steady force / width = energy to free 1 mm^2 of bond
                   (G_I when lifting frees it, G_edge when the edge does)
  front lead       thick blade, rigid bond: the sheet comes free a fixed
                   distance a ahead of the blade's top corner, where
                   G = 9 D h_blade^2 / (2 a^4)  (beam on rigid base,
                   the one-sided form of the Boeing wedge test)

All material numbers are placeholders until bench data replaces them.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/wedge_test.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class Interface:
    """Attachment of a sheet to bone, with separate peel and shear laws."""

    opening: pm.BondLaw
    sliding: pm.BondLaw


@dataclass
class Blade:
    height_mm: float = 0.5  # blade thickness: how far it lifts the tissue
    slope: float = 1.0  # rise per mm of the top face behind the tip
    contact_n_per_mm3: float = 2000.0
    heel_mm: float = 0.5  # rounding where the ramp meets the flat shaft
    sharp: bool = False
    edge_reach_mm: float = 0.1  # how far ahead of the tip a sharp edge cuts
    G_edge_n_per_mm: float = 0.0  # cost of cutting attachments with the edge

    @property
    def ramp_mm(self) -> float:
        return self.height_mm / self.slope

    def rise(self, behind):
        """Height of the blade's top face a distance ``behind`` the tip, and its slope.

        Straight ramp, then a rounded heel (quadratic blend) into the flat
        shaft, so contact rolls smoothly instead of jumping between mesh rows.
        """
        s, L = self.slope, self.ramp_mm
        f = min(self.heel_mm, L)
        b1, b2 = L - 0.5 * f, L + 0.5 * f
        b = np.maximum(behind, 0.0)
        h = np.where(b < b1, s * b, np.where(b < b2, s * b - s * (b - b1) ** 2 / (2.0 * max(f, 1e-12)), self.height_mm))
        dh = np.where(b < b1, s, np.where(b < b2, s - s * (b - b1) / max(f, 1e-12), 0.0))
        on = behind > 0.0
        return np.where(on, h, 0.0), np.where(on, dh, 0.0)


def _channel(s, hist, law: pm.BondLaw):
    """Energy and slope of one bilinear channel with damage history."""
    m = np.maximum(hist, law.delta0_mm)
    ks = pm.traction(m, law) / m
    inside = s <= m
    psi = np.where(inside, 0.5 * ks * s**2, 0.5 * ks * m**2 + pm.envelope_energy(s, law) - pm.envelope_energy(m, law))
    dpsi = np.where(inside, ks * s, pm.traction(s, law))
    return psi, dpsi


class ElevatorRig:
    """One sheet bonded to bone, plus a rigid blade. Duck-types ``pm.Sheet``."""

    def __init__(self, sheet: pm.Sheet, bonded: np.ndarray, iface: Interface, blade: Blade):
        self.sheet = sheet
        self.ids = np.asarray(bonded, dtype=np.int64)
        self.iface = iface
        self.blade = blade
        self.weight = sheet.vert_area[self.ids]
        self.hist_n = np.zeros(len(self.ids))
        self.hist_s = np.zeros(len(self.ids))
        self.broken = np.zeros(len(self.ids), dtype=bool)
        self.lost = np.zeros(len(self.ids))  # bond work lost when it broke
        self.cut_cost = 0.0
        self.tip = 0.0
        self.color_groups = sheet.color_groups

    # -- energy ---------------------------------------------------------------
    def energy(self, x, parts=False):
        e_s, g_s, sheet_parts = self.sheet.energy(x, parts=True)
        e_b, g_b = self._bond_energy(x)
        e_w, g_w, _push = self._blade(x)
        total = e_s + e_b + e_w
        grad = g_s + g_b + g_w
        if parts:
            return total, grad, {**sheet_parts, "bone_bonds": e_b, "blade": e_w}
        return total, grad

    def _separation(self, x):
        r = x[self.ids] - self.sheet.rest[self.ids]
        return r, r[:, 2], np.linalg.norm(r[:, :2], axis=1)

    def _bond_energy(self, x):
        r, dn, ds = self._separation(x)
        alive = ~self.broken
        psi_n, dpn = _channel(np.maximum(dn, 0.0), self.hist_n, self.iface.opening)
        psi_s, dps = _channel(ds, self.hist_s, self.iface.sliding)
        w = self.weight * alive
        energy = float(np.sum(w * (psi_n + psi_s)))
        grad = np.zeros_like(x)
        g = np.zeros_like(r)
        g[:, 2] = w * dpn * (dn > 0.0)
        g[:, :2] = (w * dps / np.maximum(ds, 1e-12))[:, None] * r[:, :2]
        grad[self.ids] = g
        return energy, grad

    def _blade(self, x):
        b = self.blade
        k = b.contact_n_per_mm3 * self.sheet.vert_area
        rise, drise = b.rise(self.tip - x[:, 0])
        gap = x[:, 2] - self.sheet.floor_z - rise
        kg = np.where(gap < 0.0, k * gap, 0.0)
        grad = np.zeros_like(x)
        grad[:, 2] = kg
        grad[:, 0] = kg * drise
        push = float(-np.sum(kg * drise))
        return float(np.sum(0.5 * kg * gap)), grad, push

    def push_force(self, x):
        return self._blade(x)[2]

    # -- damage ---------------------------------------------------------------
    def _work(self, x):
        """Work done on each bond so far: stored plus already lost."""
        _r, dn, ds = self._separation(x)
        out = np.zeros(len(self.ids))
        for s, hist, law in ((np.maximum(dn, 0.0), self.hist_n, self.iface.opening), (ds, self.hist_s, self.iface.sliding)):
            psi, _d = _channel(s, hist, law)
            m = np.maximum(hist, law.delta0_mm)
            ks = pm.traction(m, law) / m
            lost = np.where(hist > law.delta0_mm, pm.envelope_energy(m, law) - 0.5 * ks * m**2, 0.0)
            out += psi + lost
        return np.where(self.broken, self.lost, out)

    def commit(self, x):
        """Record damage after a converged step. Breaking one channel breaks both."""
        _r, dn, ds = self._separation(x)
        before = self._work(x)
        self.hist_n = np.maximum(self.hist_n, np.maximum(dn, 0.0))
        self.hist_s = np.maximum(self.hist_s, ds)
        newly = ~self.broken & (
            (self.hist_n >= self.iface.opening.delta_c_mm) | (self.hist_s >= self.iface.sliding.delta_c_mm)
        )
        self.lost[newly] = before[newly]
        self.broken |= newly

    def sever(self, x):
        """A sharp edge cuts the attachments within its reach of the tip."""
        b = self.blade
        if not b.sharp:
            return 0.0
        reach = ~self.broken & (self.sheet.rest[self.ids, 0] <= self.tip + b.edge_reach_mm)
        if not reach.any():
            return 0.0
        work = self._work(x)
        self.lost[reach] = work[reach]
        self.broken |= reach
        cost = float(b.G_edge_n_per_mm * self.weight[reach].sum())
        self.cut_cost += cost
        return cost

    def bond_work(self, x):
        return float(np.sum(self.weight * self._work(x)))

    def front(self):
        intact = ~self.broken
        return float(self.sheet.rest[self.ids[intact], 0].min()) if intact.any() else float("inf")


def boeing_crack_length(D, blade_height, G):
    """Debond length ahead of the lift point for a beam wedged off a rigid base."""
    return (9.0 * D * blade_height**2 / (2.0 * G)) ** 0.25


def run_wedge(
    params: pm.SkinParams,
    iface: Interface,
    blade: Blade,
    length_mm=12.0,
    width_mm=0.4,
    edge_mm=0.2,
    tab_mm=1.0,
    travel_mm=8.0,
    step_mm=0.02,
    verbose=True,
):
    rest, tris, idx = pm.strip_mesh(length_mm, width_mm, edge_mm)
    rest[:, 2] = 0.5 * params.thickness_mm  # mid-surface sits half a thickness off the bone
    sheet = pm.Sheet(rest, tris, np.zeros(0, dtype=np.int64), params, floor_z=0.5 * params.thickness_mm)
    bonded = np.flatnonzero(rest[:, 0] >= tab_mm - 1e-9)
    rig = ElevatorRig(sheet, bonded, iface, blade)
    mid = idx[:, idx.shape[1] // 2]
    x = rest.copy()
    fixed = np.zeros(x.shape, dtype=bool)
    e0 = rig.energy(x)[0]
    rows, snaps = [], []
    work = 0.0
    prev_push = 0.0
    started = time.perf_counter()
    every = max(1, int(round(1.0 / step_mm)))  # a snapshot per mm of travel
    for step, tip in enumerate(np.arange(step_mm, travel_mm + 1e-9, step_mm)):
        rig.tip = float(tip)
        cost = rig.sever(x)
        x, result = pm.solve(rig, x, fixed, gtol=1e-7)  # forces are ~10 mN; 1e-9 is below round-off here
        push = rig.push_force(x)
        work += 0.5 * (push + prev_push) * step_mm
        prev_push = push
        rig.commit(x)
        e_now = rig.energy(x)[0] - rig._bond_energy(x)[0]  # elastic + contact, bonds counted by work
        _r, dn, _ds = rig._separation(x)
        # Where the sheet is back on the bone: first intact bond still in its stiff range.
        seated = ~rig.broken & (dn < iface.opening.delta0_mm)
        liftoff = float(rest[rig.ids[seated], 0].min()) if seated.any() else float("inf")
        rows.append(
            {
                "tip_mm": float(tip),
                "push_n": push,
                "cut_cost_n_mm": cost,
                "front_mm": rig.front(),
                "lead_mm": rig.front() - float(tip),
                "liftoff_mm": liftoff,
                "contact_work_n_mm": work,
                "stored_n_mm": float(e_now - e0 + rig.bond_work(x)),
                "cut_work_n_mm": rig.cut_cost,
                "lift_mm": float(x[mid, 2].max() - 0.5 * params.thickness_mm),
                "iterations": int(result.nit),
                "residual_n": result.max_grad,
            }
        )
        if step % every == 0:
            snaps.append((float(tip), x[mid][:, [0, 2]].copy()))
            if verbose:
                r = rows[-1]
                print(
                    f"tip {tip:5.2f} mm  push {push * 1000:7.3f} mN  front {r['front_mm']:6.2f}  "
                    f"lead {r['lead_mm']:5.2f}  iters {result.nit:3d}  {time.perf_counter() - started:5.1f} s",
                    flush=True,
                )
        if rig.front() >= length_mm - 1.5:
            break
    rig.x = x
    return rows, snaps, rig


def summarize(rows, params: pm.SkinParams, iface: Interface, blade: Blade, width_mm, tab_mm, length_mm):
    tip = np.array([r["tip_mm"] for r in rows])
    front = np.array([r["front_mm"] for r in rows])
    total = np.array([r["contact_work_n_mm"] + r["cut_work_n_mm"] for r in rows])
    # Steady: front has moved 2 mm into the bond and is 2 mm from the end.
    steady = (front >= tab_mm + 2.0) & (front <= length_mm - 2.0)
    out = {
        "params": asdict(params),
        "interface": {"opening": asdict(iface.opening), "sliding": asdict(iface.sliding)},
        "blade": asdict(blade),
        "bending_D_n_mm": params.bending_D,
        "steady_steps": int(steady.sum()),
        "boeing_a_mm": boeing_crack_length(params.bending_D, blade.height_mm, iface.opening.G_n_per_mm),
        "ramp_mm": blade.ramp_mm,
    }
    # The bond is a stiff foundation, not a clamp: the sheet bends a little past
    # where it sits back down, by about 1/lambda (beam on an elastic foundation).
    lam = (iface.opening.k_bond / (4.0 * params.bending_D)) ** 0.25
    out["foundation_mm"] = 1.0 / lam
    out["boeing_liftoff_lead_mm"] = out["boeing_a_mm"] - out["foundation_mm"] - blade.ramp_mm
    last = rows[-1]
    # Work put in by the blade over energy accounted for (stored, spent in the
    # bonds, or spent by the edge). Above 1: energy lost in sudden releases.
    out["energy_balance_ratio"] = (last["contact_work_n_mm"] + last["cut_work_n_mm"]) / max(
        last["stored_n_mm"] + last["cut_work_n_mm"], 1e-15
    )
    if steady.sum() >= 5:
        # Steady force from the slope of total push work vs travel, so the
        # row-by-row release of bonds averages out.
        slope = float(np.polyfit(tip[steady], total[steady], 1)[0])
        out.update(
            {
                "steady_push_n": slope,
                "steady_push_per_width_n_per_mm": slope / width_mm,
                "push_over_G_I": slope / width_mm / iface.opening.G_n_per_mm,
                "push_over_G_edge": (slope / width_mm / blade.G_edge_n_per_mm) if blade.G_edge_n_per_mm > 0 else None,
                "steady_lead_mm": float(np.median(front[steady] - tip[steady])),
                "steady_liftoff_lead_mm": float(np.median(np.array([r["liftoff_mm"] for r in rows])[steady] - tip[steady])),
                "front_speed": float(np.polyfit(tip[steady], front[steady], 1)[0]),
            }
        )
    return out


def _plot(path, rows, snaps, summary, width_mm):
    from PIL import Image, ImageDraw

    W, H = 1400, 620
    img = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    tip = np.array([r["tip_mm"] for r in rows])
    total = np.array([r["contact_work_n_mm"] + r["cut_work_n_mm"] for r in rows]) * 1000
    x0, y0, w, h = 70, 40, 560, 500
    ymax = max(total.max(), 1e-6) * 1.1
    sx = lambda v: x0 + v / tip.max() * w
    sy = lambda v: y0 + h - v / ymax * h
    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(160, 160, 160))
    dr.line([(sx(a), sy(b)) for a, b in zip(tip, total)], fill=(40, 90, 170), width=3)
    if "steady_push_n" in summary:
        g = summary["interface"]["opening"]["G_n_per_mm"] * width_mm * 1000
        dr.text((x0 + 10, y0 + 10), f"steady push {summary['steady_push_n'] * 1000:.2f} mN  (G_I x width = {g:.2f} mN)",
                fill=(30, 30, 30))
    for v in np.linspace(0, ymax, 6):
        dr.text((x0 - 55, sy(v) - 6), f"{v:.2f}", fill=(90, 90, 90))
    dr.text((x0, 12), "Push work (N*mm x1000) vs blade travel (mm); slope = push force", fill=(30, 30, 30))
    px0, py0, pw, ph = 700, 40, 660, 500
    dr.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(160, 160, 160))
    pts = np.concatenate([s[1] for s in snaps])
    span_x = np.ptp(pts[:, 0]) * 1.05
    zmax = max(pts[:, 1].max(), 1.0) * 1.3
    tx = lambda v: px0 + 10 + (v - pts[:, 0].min()) / span_x * (pw - 20)
    tz = lambda v: py0 + ph - 20 - v / zmax * (ph - 40)
    dr.line([px0, tz(0), px0 + pw, tz(0)], fill=(120, 120, 120), width=2)
    for k, (t, line) in enumerate(snaps):
        shade = int(200 - 160 * k / max(1, len(snaps) - 1))
        dr.line([(tx(a), tz(b)) for a, b in line], fill=(shade, shade, 255 - shade // 3), width=2)
    dr.text((px0, 12), "Sheet centreline as the blade advances (side view, bone at bottom; height exaggerated)",
            fill=(30, 30, 30))
    img.save(path)


def _interface():
    return Interface(opening=pm.BondLaw(0.05, 0.01, 0.05), sliding=pm.BondLaw(0.1, 0.02, 0.1))


def default_case():
    """Thick, steep blade under a stiff 1 mm sheet: lifts the tissue off ahead of it."""
    return pm.SkinParams(E_mpa=5.0, nu=0.45, thickness_mm=1.0), _interface(), Blade(height_mm=0.5, slope=1.0)


def thin_case(sharp=False, G_edge=0.015):
    """Thin, low-angle blade (round-knife-like) under a 0.5 mm sheet: frees tissue at its tip."""
    blade = Blade(height_mm=0.1, slope=0.1, sharp=sharp, G_edge_n_per_mm=G_edge if sharp else 0.0)
    return pm.SkinParams(E_mpa=5.0, nu=0.45, thickness_mm=0.5), _interface(), blade


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=("thick", "thin"), default="thick")
    parser.add_argument("--sharp", action="store_true", help="Edge cuts attachments ahead of the tip")
    parser.add_argument("--G-edge", type=float, default=0.015, help="Edge cutting cost, N/mm")
    parser.add_argument("--blade-height", type=float, default=None)
    parser.add_argument("--slope", type=float, default=None)
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "resources" / "soft_tissue" / "wedge_test")
    args = parser.parse_args()
    params, iface, blade = default_case() if args.case == "thick" else thin_case()
    if args.blade_height is not None:
        blade.height_mm = args.blade_height
    if args.slope is not None:
        blade.slope = args.slope
    if args.sharp:
        blade.sharp, blade.G_edge_n_per_mm = True, args.G_edge
    rows, snaps, _rig = run_wedge(params, iface, blade)
    summary = summarize(rows, params, iface, blade, 0.4, 1.0, 12.0)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(args.out / "wedge.png", rows, snaps, summary, 0.4)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
