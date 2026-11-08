"""Cut test: incisions in a sheet of skin held under tension.

A square sheet is gripped along its top and bottom edges and stretched, like
skin under retraction. Grips are free to slide sideways, so the stretch is
uniaxial. Only in-plane stretch matters here, so the sheet is kept flat.

An incision is made by splitting the mesh along a line: vertices on the cut
are duplicated so the two sides no longer share them. The ends of the cut stay
joined (the cut tips).

Known answers it is checked against (linear elastic fracture, plane stress,
centre cut of length 2a in a wide sheet under stress s, finite-width factor
sec(pi a / W) applied)
  gape           the cut opens to a slit, max opening 4 s a / E
  give           energy released by cutting under held grips  pi s^2 a^2 h / E;
                 the grip force drops by the same fraction as the stored energy
  blade force    drawing a blade one tip further along the cut costs
                 (Jc - G) h per mm, G = pi s^2 a / E the energy the tension
                 releases at the tip. Slack skin: Jc h. Taut skin: less.
                 If G > Jc the cut runs ahead of the blade on its own.

Jc (cutting toughness, energy per area of new cut face) is a placeholder.

Accuracy: plain triangles are too stiff right at a cut tip, so a coarse mesh
releases less energy than theory (about 80% at 0.5 mm, 88% at 0.25 mm on a
32 mm sheet) and converges toward it as the mesh is refined. An independent
linear finite-element solve on the same mesh gives the same numbers.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/cut_test.py
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


def plate_mesh(size_mm, edge_mm):
    """Square sheet centred on the origin, with a row of vertices on y = 0."""
    rest, tris, idx = pm.strip_mesh(size_mm, size_mm, edge_mm)
    rest[:, :2] -= 0.5 * size_mm
    return rest, tris, idx


def split_vertices(rest, tris, verts, upper_side):
    """Duplicate ``verts`` for the triangles on the upper side of the cut.

    ``upper_side`` is a bool per triangle. Returns new rest, tris and, for each
    new vertex, its parent.
    """
    rest = rest.copy()
    tris = tris.copy()
    parents = []
    n = len(rest)
    for k, v in enumerate(verts):
        new = n + k
        hit = upper_side[:, None] & (tris == v)
        tris[hit] = new
        parents.append(v)
    parents = np.asarray(parents, dtype=np.int64)
    rest = np.concatenate([rest, rest[parents]])
    return rest, tris, parents


class Plate:
    """A gripped sheet with an incision along y = 0 between x0 and x1."""

    def __init__(self, params: pm.SkinParams, size_mm=16.0, edge_mm=0.25):
        self.params = params
        self.size = size_mm
        self.edge = edge_mm
        self.base_rest, self.base_tris, _idx = plate_mesh(size_mm, edge_mm)
        cent = self.base_rest[self.base_tris].mean(axis=1)
        self.upper = cent[:, 1] > 0.0
        on_line = np.abs(self.base_rest[:, 1]) < 1e-9
        self.line = np.flatnonzero(on_line)
        self.line = self.line[np.argsort(self.base_rest[self.line, 0])]
        self.cut = (0.0, 0.0)
        self._build(np.zeros(0, dtype=np.int64))

    def _build(self, split):
        rest, tris, parents = split_vertices(self.base_rest, self.base_tris, split, self.upper)
        self.rest, self.tris, self.parents, self.split = rest, tris, parents, split
        self.sheet = pm.Sheet(rest, tris, np.zeros(0, dtype=np.int64), self.params, floor_z=-1e9)
        half = 0.5 * self.size
        self.top = np.flatnonzero(rest[:, 1] > half - 1e-9)
        self.bottom = np.flatnonzero(rest[:, 1] < -half + 1e-9)

    def make_cut(self, x0, x1, x=None):
        """Cut from x0 to x1 (tips stay joined). Carries a deformed state over."""
        xs = self.base_rest[self.line, 0]
        inside = self.line[(xs > x0 + 1e-9) & (xs < x1 - 1e-9)]
        old_x = x
        self.cut = (x0, x1)
        self._build(inside)
        if old_x is None:
            return None
        n_base = len(self.base_rest)
        new_x = np.empty_like(self.rest)
        new_x[:n_base] = old_x[:n_base]
        new_x[n_base:] = old_x[self.parents]
        return new_x

    def stretch(self, strain, x0=None):
        """Hold the grips at +-strain*size/2 and solve. Returns x, grip force, energy."""
        half = 0.5 * self.size
        x = self.rest.copy() if x0 is None else x0.copy()
        x[self.top, 1] = half * (1.0 + strain)
        x[self.bottom, 1] = -half * (1.0 + strain)
        fixed = np.zeros(x.shape, dtype=bool)
        fixed[:, 2] = True  # flat sheet
        fixed[self.top, 1] = fixed[self.bottom, 1] = True
        anchor = self.bottom[np.argmin(np.abs(self.rest[self.bottom, 0]))]
        fixed[anchor, 0] = True  # no sideways drift
        x, result = pm.solve(self.sheet, x, fixed, gtol=1e-10)
        e, g = self.sheet.energy(x)
        force = float(g[self.top, 1].sum())
        return x, force, float(e), result

    def opening(self, x):
        """Gap across the cut at each split vertex (upper copy minus lower)."""
        n_base = len(self.base_rest)
        copies = n_base + np.arange(len(self.split))
        return self.base_rest[self.split, 0], x[copies, 1] - x[self.split, 1]


def width_factor(a, W):
    return 1.0 / math.cos(math.pi * a / W)


def griffith_G(stress, a, E, W):
    return math.pi * stress**2 * a / E * width_factor(a, W)


def run_incision(params, half_len=2.0, strain=0.02, size_mm=16.0, edge_mm=0.25):
    plate = Plate(params, size_mm, edge_mm)
    x_u, f_u, e_u, _ = plate.stretch(strain)
    stress = f_u / (size_mm * params.thickness_mm)
    x_c = plate.make_cut(-half_len, half_len, x_u)
    x_c, f_c, e_c, _ = plate.stretch(strain, x_c)
    xs, gap = plate.opening(x_c)
    E, h, a = params.E_mpa, params.thickness_mm, half_len
    return {
        "stress_mpa": stress,
        "grip_force_uncut_n": f_u,
        "grip_force_cut_n": f_c,
        "energy_uncut_n_mm": e_u,
        "energy_cut_n_mm": e_c,
        "released_n_mm": e_u - e_c,
        "expected_released_n_mm": math.pi * stress**2 * a**2 * h / E * width_factor(a, size_mm),
        "max_opening_mm": float(gap.max()),
        "expected_max_opening_mm": 4.0 * stress * a / E * math.sqrt(width_factor(a, size_mm)),
        "min_opening_mm": float(gap.min()),
        "force_drop_fraction": 1.0 - f_c / f_u,
        "energy_drop_fraction": 1.0 - e_c / e_u,
        "profile": (xs.tolist(), gap.tolist()),
        "x": x_c,
        "plate": plate,
    }


def run_slice(params, Jc_n_per_mm, strain, start_half=1.0, end_half=4.0, size_mm=16.0, edge_mm=0.25):
    """A blade lengthens the cut at its right tip, one mesh edge at a time.

    Each step: work the blade must supply = Jc * new cut area minus the
    elastic energy the tension releases. Grips are held.
    """
    plate = Plate(params, size_mm, edge_mm)
    x, *_ = plate.stretch(strain)
    x0 = -start_half
    x = plate.make_cut(x0, start_half, x)
    x, f, e, _ = plate.stretch(strain, x)
    stress_ref = plate_stress(params, strain, size_mm)
    rows = []
    tip = start_half
    h = params.thickness_mm
    while tip + edge_mm <= x0 + 2 * end_half + 1e-9:
        new_tip = tip + edge_mm
        x = plate.make_cut(x0, new_tip, x)
        x, f_new, e_new, _ = plate.stretch(strain, x)
        area = edge_mm * h
        G = (e - e_new) / area
        a_mid = 0.5 * (0.5 * (tip - x0) + 0.5 * (new_tip - x0))
        rows.append(
            {
                "cut_length_mm": new_tip - x0,
                "half_length_mm": a_mid,
                "G_n_per_mm": G,
                "G_griffith_n_per_mm": griffith_G(stress_ref, a_mid, params.E_mpa, size_mm),
                "blade_force_n": max(Jc_n_per_mm - G, 0.0) * h,
                "runs_ahead": bool(G > Jc_n_per_mm),
                "grip_force_n": f_new,
            }
        )
        tip, e = new_tip, e_new
    return rows


def plate_stress(params, strain, size_mm=16.0, edge_mm=1.0):
    """Stress in the uncut gripped sheet at this strain (coarse mesh is exact here)."""
    plate = Plate(params, size_mm, edge_mm)
    _x, f, _e, _r = plate.stretch(strain)
    return f / (size_mm * params.thickness_mm)


def _plot(path, inc, slices, params, Jc):
    from PIL import Image, ImageDraw

    W, H = 1400, 620
    img = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    # Left: deformed sheet around the cut, opening exaggerated.
    plate, x = inc["plate"], inc["x"]
    gain = 0.5 / max(inc["max_opening_mm"], 1e-9)  # 0.5 mm of gap per max opening, for view
    disp = (x - plate.rest)[:, :2]
    shown = plate.rest[:, :2] + disp * gain * 4.0
    win = 4.5
    cx0, cy0, cw = 40, 50, 540
    tx = lambda v: cx0 + (v + win) / (2 * win) * cw
    ty = lambda v: cy0 + cw - (v + win) / (2 * win) * cw
    for t in plate.tris:
        p = shown[t]
        if np.all(np.abs(plate.rest[t, :2]) < win):
            dr.polygon([(tx(a), ty(b)) for a, b in p], outline=(150, 160, 190))
    dr.text((cx0, 18), f"Incision 2a = {2 * (plate.cut[1]):.0f} mm in skin stretched {100 * inc['stress_mpa'] / params.E_mpa:.0f}%"
            " (displacements exaggerated)", fill=(30, 30, 30))
    dr.text((cx0, cy0 + cw + 10),
            f"max gape {inc['max_opening_mm'] * 1000:.0f} um (theory {inc['expected_max_opening_mm'] * 1000:.0f});"
            f" grip force drops {100 * inc['force_drop_fraction']:.1f}%", fill=(30, 30, 30))
    # Right: blade force vs cut length, for each tension.
    px0, py0, pw, ph = 720, 50, 640, 500
    dr.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(160, 160, 160))
    Lmax = max(r["cut_length_mm"] for rows in slices.values() for r in rows)
    fmax = Jc * params.thickness_mm * 1.15 * 1000
    sx = lambda v: px0 + v / Lmax * pw
    sy = lambda v: py0 + ph - v / fmax * ph
    colors = [(40, 90, 170), (60, 150, 90), (200, 120, 40), (190, 60, 60)]
    for k, (strain, rows) in enumerate(slices.items()):
        pts = [(sx(r["cut_length_mm"]), sy(r["blade_force_n"] * 1000)) for r in rows]
        dr.line(pts, fill=colors[k % 4], width=3)
        dr.text((pts[-1][0] - 120, pts[-1][1] - 16), f"stretch {100 * strain:.0f}%", fill=colors[k % 4])
    for v in np.linspace(0, fmax, 6):
        dr.text((px0 - 45, sy(v) - 6), f"{v:.0f}", fill=(90, 90, 90))
    dr.text((px0, 18), "Blade force (mN) drawing the cut longer, vs cut length (mm)", fill=(30, 30, 30))
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--Jc", type=float, default=0.5, help="Cutting toughness, N/mm (placeholder)")
    parser.add_argument("--edge", type=float, default=0.25)
    parser.add_argument("--slice-edge", type=float, default=0.5, help="Mesh edge for the blade runs (slow when fine)")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "resources" / "soft_tissue" / "cut_test")
    args = parser.parse_args()
    params = pm.SkinParams(E_mpa=1.0, nu=0.45, thickness_mm=1.0)
    inc = run_incision(params, strain=0.05, edge_mm=args.edge)
    slices = {s: run_slice(params, args.Jc, s, edge_mm=args.slice_edge) for s in (0.0, 0.05, 0.1, 0.2)}
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in inc.items() if k not in ("x", "plate", "profile")}
    summary["slice"] = {str(s): rows for s, rows in slices.items()}
    summary["Jc_n_per_mm"] = args.Jc
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(args.out / "cut.png", inc, slices, params, args.Jc)
    print(json.dumps({k: v for k, v in summary.items() if k != "slice"}, indent=2))
    for s, rows in slices.items():
        print(f"stretch {s:.2f}: blade force mN " + " ".join(f"{r['blade_force_n'] * 1000:.0f}" for r in rows[::2]))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
