"""Stretch test: pull a strip of tissue until it breaks.

A flat strip is held at both ends and pulled along its length. The grips hold
the ends along the pull but let them slide sideways, so the strip narrows
freely and the stretch is the same everywhere (a uniform uniaxial test).

Material: St Venant-Kirchhoff plus collagen recruitment along each rest axis
(``SkinParams.k1_mpa``, ``k2``), which gives skin its J-shaped curve: soft at
first, stiffening steeply as fibres straighten.

Failure: a triangle breaks when its Green strain along a fibre direction passes
``fail_strain`` and is removed. One row is made 1% weaker so the break
starts there, as a real band snaps at its weakest point.

Known answers (exact for a uniform strip, Green strain e = (s^2 - 1)/2 at
stretch s, sides free)
  force         F = w h s (E e + k1 e exp(k2 e^2))
  small stretch F = (E + k1) w h (strain); the collagen term adds stiffness
                from the start, then rises steeply (the J)
  rupture       peak force = F at the failure strain; after it, zero
  work          work to rupture = integral of F over the pull = stored energy
                at failure, w h L (E e^2/2 + k1/(2 k2)(exp(k2 e^2) - 1))

Material numbers are placeholders until bench data replaces them.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/stretch_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def skin_params():
    """Placeholder skin: 1 MPa at small strain, J-curve stiffening, ~30% stretch to break."""
    return pm.SkinParams(E_mpa=1.0, nu=0.45, thickness_mm=1.0, k1_mpa=2.0, k2=10.0)


def green(stretch):
    return 0.5 * (np.asarray(stretch) ** 2 - 1.0)


def exact_force(p: pm.SkinParams, stretch, width_mm):
    e = green(stretch)
    s = p.E_mpa * e
    if p.k1_mpa > 0 and p.k2 > 0:
        ep = np.maximum(e, 0.0)
        s = s + p.k1_mpa * ep * np.exp(p.k2 * ep**2)
    return width_mm * p.thickness_mm * np.asarray(stretch) * s


def exact_energy_density(p: pm.SkinParams, stretch):
    e = green(stretch)
    w = 0.5 * p.E_mpa * e**2
    if p.k1_mpa > 0 and p.k2 > 0:
        ep = np.maximum(e, 0.0)
        w = w + p.k1_mpa / (2.0 * p.k2) * (np.exp(p.k2 * ep**2) - 1.0)
    return w


def stretch_at_green(e):
    return math.sqrt(1.0 + 2.0 * e)


def fibre_strain(sheet: pm.Sheet, x):
    """Largest Green strain along the fibre directions (the rest x and y axes).

    Tissue fails when its collagen fibres rupture, so failure is judged along
    the fibres, not in whatever direction shear makes largest.
    """
    t = sheet.tris
    ds = np.stack([x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]], axis=2)
    F = ds @ sheet.dm_inv
    C = np.einsum("mki,mkj->mij", F, F)
    Eg = 0.5 * (C - np.eye(2))
    return np.maximum(Eg[:, 0, 0], Eg[:, 1, 1])


class Strip:
    def __init__(self, params, length_mm=10.0, width_mm=2.0, edge_mm=0.25, fail_strain=0.35, weak=0.99):
        self.params = params
        self.length, self.width = length_mm, width_mm
        self.rest, self.all_tris, _idx = pm.strip_mesh(length_mm, width_mm, edge_mm)
        cent = self.rest[self.all_tris].mean(axis=1)
        # The weak row: one full column of cells just past mid-length.
        self.weak_x = 0.5 * length_mm + 0.5 * edge_mm
        mid = abs(cent[:, 0] - self.weak_x) < 0.5 * edge_mm
        self.fail = np.where(mid, weak * fail_strain, fail_strain)
        self.alive = np.ones(len(self.all_tris), dtype=bool)
        self.left = np.flatnonzero(self.rest[:, 0] < 1e-9)
        self.right = np.flatnonzero(self.rest[:, 0] > length_mm - 1e-9)
        self._build()

    def _build(self):
        self.sheet = pm.Sheet(self.rest, self.all_tris[self.alive], np.zeros(0, dtype=np.int64), self.params,
                              floor_z=-1e9)
        self.fail_alive = self.fail[self.alive]

    def solve(self, stretch, x0):
        x = x0.copy()
        # Start from the old shape stretched evenly to the new grip position.
        x[:, 0] *= self.length * stretch / max(float(x[self.right, 0].mean()), 1e-12)
        x[self.left, 0] = 0.0
        x[self.right, 0] = self.length * stretch
        fixed = np.zeros(x.shape, dtype=bool)
        fixed[:, 2] = True
        fixed[self.left, 0] = fixed[self.right, 0] = True
        anchor = self.left[np.argmin(np.abs(self.rest[self.left, 1] - 0.5 * self.width))]
        fixed[anchor, 1] = True  # no sideways drift
        x, _res = pm.solve(self.sheet, x, fixed, gtol=1e-9)
        e, g = self.sheet.energy(x)
        return x, float(g[self.right, 0].sum()), float(e)

    def overstrain(self, x):
        return float((fibre_strain(self.sheet, x) / self.fail_alive).max())

    def break_worst(self, x):
        """Remove the most overstrained triangle, if any. Returns True if one broke."""
        over = fibre_strain(self.sheet, x) / self.fail_alive
        k = int(np.argmax(over))
        if over[k] < 1.0:
            return False
        live = np.flatnonzero(self.alive)
        self.alive[live[k]] = False
        self._build()
        return True

    def separated(self):
        """True once no path of triangles joins the two grips."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        t = self.all_tris[self.alive]
        n = len(self.rest)
        rows = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
        cols = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _n, label = connected_components(g, directed=False)
        return not np.intersect1d(label[self.left], label[self.right]).size


def run_stretch(params, max_stretch=1.5, step=0.005, **strip_kw):
    strip = Strip(params, **strip_kw)
    x = strip.rest.copy()
    rows = []
    work, prev_f, prev_s = 0.0, 0.0, 1.0
    broken_at = None
    for s in np.arange(1.0 + step, max_stretch + 1e-9, step):
        x_prev = x
        x, f, e = strip.solve(s, x)
        if broken_at is None and strip.overstrain(x) >= 1.0:
            # Find the stretch where the first triangle just reaches its limit,
            # so the break starts at the weakest point and not everywhere at once.
            lo, hi = prev_s, s
            for _ in range(30):
                m = 0.5 * (lo + hi)
                xm, *_ = strip.solve(m, x_prev)
                lo, hi = (m, hi) if strip.overstrain(xm) < 1.0 else (lo, m)
            x_lo, f_lo, e_lo = strip.solve(lo, x_prev)
            work += 0.5 * (f_lo + prev_f) * (lo - prev_s) * strip.length
            rows.append({"stretch": float(lo), "force_n": f_lo, "energy_n_mm": e_lo, "work_n_mm": work, "broke": 0,
                         "exact_force_n": float(exact_force(params, lo, strip.width))})
            prev_f, prev_s = f_lo, lo
            s, x = hi, x_lo
            x, f, e = strip.solve(s, x)
        broke = 0
        while strip.break_worst(x):  # failures cascade at a held stretch
            broke += 1
            x, f, e = strip.solve(s, x)
        work += 0.5 * (f + prev_f) * (s - prev_s) * strip.length
        rows.append({"stretch": float(s), "force_n": f, "energy_n_mm": e, "work_n_mm": work, "broke": broke,
                     "exact_force_n": float(exact_force(params, s, strip.width))})
        prev_f, prev_s = f, s
        if broke and broken_at is None:
            broken_at = float(s)
        if broken_at is not None and strip.separated():
            break
    return rows, strip, broken_at


def summarize(rows, params, strip: Strip, broken_at):
    s = np.array([r["stretch"] for r in rows])
    f = np.array([r["force_n"] for r in rows])
    exact = np.array([r["exact_force_n"] for r in rows])
    intact = np.array([r["broke"] == 0 for r in rows])
    if broken_at is not None:
        intact &= s < broken_at
    e_fail = float(strip.fail.min())
    s_fail = stretch_at_green(e_fail)
    out = {
        "params": {k: getattr(params, k) for k in ("E_mpa", "nu", "thickness_mm", "k1_mpa", "k2")},
        "width_mm": strip.width,
        "length_mm": strip.length,
        "fail_green_strain": e_fail,
        "fail_stretch": s_fail,
        "max_force_error_before_break": float(np.max(np.abs(f[intact] - exact[intact]) / np.maximum(exact[intact], 1e-12))),
        "peak_force_n": float(f.max()),
        "exact_peak_force_n": float(exact_force(params, s_fail, strip.width)),
        "broke_at_stretch": broken_at,
        "force_after_n": float(f[-1]),
        "separated": strip.separated(),
        "work_to_rupture_n_mm": float(rows[-1]["work_n_mm"]),
        "exact_work_to_rupture_n_mm": float(exact_energy_density(params, s_fail) * strip.width * params.thickness_mm
                                            * strip.length),
        "small_strain_stiffness_n": float(f[0] / (s[0] - 1.0)),
        "exact_small_strain_stiffness_n": (params.E_mpa + params.k1_mpa) * strip.width * params.thickness_mm,
        "force_at_last_intact_n": float(f[intact][-1]) if intact.any() else 0.0,
        "exact_force_at_last_intact_n": float(exact[intact][-1]) if intact.any() else 0.0,
    }
    return out


def _plot(path, rows, summary):
    from PIL import Image, ImageDraw

    W, H = 900, 620
    img = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    s = np.array([r["stretch"] for r in rows])
    f = np.array([r["force_n"] for r in rows])
    fine = np.linspace(1.0, summary["fail_stretch"], 200)
    exact = exact_force(rows_params(summary), fine, summary["width_mm"])
    x0, y0, w, h = 80, 50, 760, 500
    fmax = max(f.max(), exact.max()) * 1.1
    sx = lambda v: x0 + (v - 1.0) / (s.max() - 1.0) * w
    sy = lambda v: y0 + h - v / fmax * h
    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(160, 160, 160))
    dr.line([(sx(a), sy(b)) for a, b in zip(fine, exact)], fill=(200, 80, 60), width=6)
    dr.line([(sx(a), sy(b)) for a, b in zip(s, f)], fill=(40, 90, 170), width=2)
    for v in np.linspace(0, fmax, 6):
        dr.text((x0 - 60, sy(v) - 6), f"{v:.2f}", fill=(90, 90, 90))
    for v in np.arange(0, s.max() - 1.0 + 1e-9, 0.1):
        dr.text((sx(1.0 + v) - 10, y0 + h + 8), f"{100 * v:.0f}%", fill=(90, 90, 90))
    dr.text((x0, 16), f"Strip {summary['width_mm']:g} mm wide pulled to rupture: force (N) vs stretch. "
            "Red = exact J-curve, blue = model", fill=(30, 30, 30))
    dr.text((x0 + 10, y0 + 10), f"peak {summary['peak_force_n']:.2f} N (exact {summary['exact_peak_force_n']:.2f});"
            f" work to rupture {summary['work_to_rupture_n_mm']:.2f} N*mm (exact {summary['exact_work_to_rupture_n_mm']:.2f})",
            fill=(30, 30, 30))
    img.save(path)


def rows_params(summary):
    return pm.SkinParams(**summary["params"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "resources" / "soft_tissue" / "stretch_test")
    args = parser.parse_args()
    params = skin_params()
    rows, strip, broken_at = run_stretch(params, step=0.01)
    summary = summarize(rows, params, strip, broken_at)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(args.out / "stretch.png", rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
