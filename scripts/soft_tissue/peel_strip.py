"""90-degree peel of a flat skin strip off a rigid plate.

The strip lies on z = 0 with its first ``tab`` mm unbonded. The grip edge
(x = 0) is lifted straight up and is free to slide along the plate, like a
sliding-table peel fixture, so the arm stays vertical. Each step raises the
grip, minimises energy, then records bond damage.

Checks against two closed-form results for an elastic strip:
  peel force   Kendall: (F/w)(1 - cos 90) + (F/w)^2 / (2 E h) = G
  front curl   curvature where the strip leaves the plate, sqrt(2 G / D)

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/peel_strip.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def kendall_force_per_width(p: pm.SkinParams, angle_deg=90.0):
    """Steady peel force per width for an extensible elastic strip."""
    eh = p.E_mpa * p.thickness_mm
    c = 1.0 - math.cos(math.radians(angle_deg))
    # f^2 / (2 E h) + c f - G = 0
    return eh * (-c + math.sqrt(c * c + 2.0 * p.G_n_per_mm / eh))


def front_curvature(p: pm.SkinParams):
    return math.sqrt(2.0 * p.G_n_per_mm / p.bending_D)


def _centerline_curvature(points):
    """Turning angle per unit length at each interior point of a polyline."""
    a = points[1:-1] - points[:-2]
    b = points[2:] - points[1:-1]
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(la * lb, 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0)) / np.maximum(0.5 * (la + lb), 1e-12)


def run_peel(
    params: pm.SkinParams,
    length_mm=25.0,
    width_mm=5.0,
    edge_mm=0.25,
    tab_mm=4.0,
    lift_mm=16.0,
    step_mm=0.1,
    snapshot_every=20,
    verbose=True,
    stop_margin_mm=2.0,
    solve=None,
):
    rest, tris, idx = pm.strip_mesh(length_mm, width_mm, edge_mm)
    bonded = np.flatnonzero(rest[:, 0] >= tab_mm - 1e-9)
    sheet = pm.Sheet(rest, tris, bonded, params)
    grip = idx[0]
    mid = idx[:, idx.shape[1] // 2]
    x = rest.copy()
    n = len(rest)
    fixed = np.zeros((n, 3), dtype=bool)
    fixed[grip, 2] = True
    rows, snapshots = [], []
    lifts = np.arange(step_mm, lift_mm + 1e-9, step_mm)
    started = time.perf_counter()
    for step, lift in enumerate(lifts):
        x[grip, 2] = lift
        x, result = (solve or pm.solve)(sheet, x, fixed)
        _e, grad, parts = sheet.energy(x, parts=True)
        sheet.commit(x)
        force = float(grad[grip, 2].sum())
        intact = ~sheet.broken()
        front = float(rest[bonded[intact], 0].min()) if intact.any() else length_mm
        line = x[mid][:, [0, 2]]
        curv = _centerline_curvature(line)
        near = np.abs(rest[mid][1:-1, 0] - front) <= 2.0
        kappa = float(curv[near].max()) if near.any() else float("nan")
        free = np.abs(grad).copy()
        free[grip, 2] = 0.0
        rows.append(
            {
                "lift_mm": float(lift),
                "force_n": force,
                "front_mm": front,
                "grip_x_mm": float(x[grip, 0].mean()),
                "front_curvature_per_mm": kappa,
                "residual_n": float(free.max()),
                "iterations": int(result.nit),
                **{f"energy_{k}": float(v) for k, v in parts.items()},
            }
        )
        if step % snapshot_every == 0 or step == len(lifts) - 1:
            snapshots.append((float(lift), line.copy()))
        if verbose and step % 10 == 0:
            print(
                f"lift {lift:5.2f} mm  force {force * 1000:7.2f} mN  front {front:6.2f} mm  "
                f"curv {kappa:5.3f}/mm  iters {result.nit:5d}  residual {free.max():.1e}  "
                f"{time.perf_counter() - started:5.1f} s",
                flush=True,
            )
        if front >= length_mm - stop_margin_mm:
            break
    return rows, snapshots, sheet


def summarize(rows, params: pm.SkinParams, width_mm, tab_mm, length_mm):
    """Steady peel: the front has moved at least 3 mm and is 3 mm short of the end."""
    lift = np.array([r["lift_mm"] for r in rows])
    force = np.array([r["force_n"] for r in rows])
    front = np.array([r["front_mm"] for r in rows])
    kappa = np.array([r["front_curvature_per_mm"] for r in rows])
    steady = (front >= tab_mm + 3.0) & (front <= length_mm - 3.0)
    expected_f = kendall_force_per_width(params) * width_mm
    expected_k = front_curvature(params)
    out = {
        "params": asdict(params),
        "bending_D_n_mm": params.bending_D,
        "T_max_mpa": params.T_max,
        "steady_steps": int(steady.sum()),
        "expected_force_n": expected_f,
        "expected_front_curvature_per_mm": expected_k,
        "expected_front_radius_mm": 1.0 / expected_k,
    }
    if steady.sum() >= 5:
        f_ss = float(np.median(force[steady]))
        k_ss = float(np.median(kappa[steady]))
        rate = float(np.polyfit(lift[steady], front[steady], 1)[0])
        out.update(
            {
                "steady_force_n": f_ss,
                "force_ratio": f_ss / expected_f,
                "force_spread_n": float(np.percentile(force[steady], 95) - np.percentile(force[steady], 5)),
                "steady_front_curvature_per_mm": k_ss,
                "curvature_ratio": k_ss / expected_k,
                "front_advance_per_mm_lift": rate,
                "peak_force_n": float(force.max()),
            }
        )
    return out


def _plot(path, rows, snapshots, summary, width_mm):
    from PIL import Image, ImageDraw

    W, H = 1400, 620
    img = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    # force vs lift
    lift = np.array([r["lift_mm"] for r in rows])
    force = np.array([r["force_n"] for r in rows]) * 1000
    fx0, fy0, fw, fh = 70, 40, 560, 500
    fmax = max(force.max(), summary["expected_force_n"] * 1000) * 1.15
    sx = lambda v: fx0 + v / lift.max() * fw
    sy = lambda v: fy0 + fh - v / fmax * fh
    dr.rectangle([fx0, fy0, fx0 + fw, fy0 + fh], outline=(160, 160, 160))
    ef = summary["expected_force_n"] * 1000
    dr.line([sx(0), sy(ef), sx(lift.max()), sy(ef)], fill=(200, 80, 60), width=2)
    dr.text((sx(0) + 6, sy(ef) - 16), f"Kendall {ef:.0f} mN", fill=(200, 80, 60))
    pts = [(sx(a), sy(b)) for a, b in zip(lift, force)]
    dr.line(pts, fill=(40, 90, 170), width=3)
    for v in np.linspace(0, fmax, 6):
        dr.text((fx0 - 50, sy(v) - 6), f"{v:.0f}", fill=(90, 90, 90))
    for v in np.arange(0, lift.max() + 1e-9, 4):
        dr.text((sx(v) - 6, fy0 + fh + 6), f"{v:.0f}", fill=(90, 90, 90))
    dr.text((fx0, 12), f"Peel force (mN) vs grip lift (mm), strip {width_mm:g} mm wide", fill=(30, 30, 30))
    # profiles
    px0, py0, pw, ph = 700, 40, 660, 500
    dr.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(160, 160, 160))
    all_pts = np.concatenate([s[1] for s in snapshots])
    span = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 1.05
    ox, oz = all_pts[:, 0].min(), 0.0
    tx = lambda v: px0 + 10 + (v - ox) / span * (pw - 20)
    tz = lambda v: py0 + ph - 10 - (v - oz) / span * (ph - 20)
    dr.line([px0, tz(0), px0 + pw, tz(0)], fill=(120, 120, 120), width=2)
    for k, (lv, line) in enumerate(snapshots):
        shade = int(200 - 160 * k / max(1, len(snapshots) - 1))
        dr.line([(tx(a), tz(b)) for a, b in line], fill=(shade, shade, 255 - shade // 3), width=2)
    dr.text((px0, 12), "Strip centreline at successive lifts (side view, plate at bottom)", fill=(30, 30, 30))
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edge", type=float, default=0.25)
    parser.add_argument("--width", type=float, default=5.0)
    parser.add_argument("--length", type=float, default=25.0)
    parser.add_argument("--lift", type=float, default=16.0)
    parser.add_argument("--step", type=float, default=0.2, help="Grip lift per step, mm")
    parser.add_argument("--thickness", type=float, default=None)
    parser.add_argument("--G", type=float, default=None, help="Adhesion energy, N/mm")
    parser.add_argument("--E", type=float, default=None, help="Skin modulus, MPa")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "resources" / "soft_tissue" / "peel_test")
    args = parser.parse_args()

    params = pm.SkinParams()
    if args.thickness is not None:
        params.thickness_mm = args.thickness
    if args.G is not None:
        params.G_n_per_mm = args.G
    if args.E is not None:
        params.E_mpa = args.E
    rows, snapshots, _sheet = run_peel(
        params, args.length, args.width, args.edge, lift_mm=args.lift, step_mm=args.step, snapshot_every=10
    )
    summary = summarize(rows, params, args.width, 4.0, args.length)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "peel.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(args.out / "peel.png", rows, snapshots, summary, args.width)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
