"""GIFs of the cartilage plate tests (plate_test.py), slow reference vs fast C++ Newton.

Every frame is a solved state: the load is stepped up, both solvers solve it (each warm-started
from its own previous answer), and the frame shows both shapes and the force read off the held
vertices, against the exact answer.

  cantilever_elastica  a 1 mm cartilage strip, tip pushed down 0 -> 9.5 mm: small-load beam
                       line, then the exact elastica
  poisson_curl         a 4 mm strip bent along its length curls the other way across its width,
                       -nu times as much
  pinched_ring         a ring (curved rest shape) squeezed across a diameter, vs the exact ring
                       elastica (and its small-squeeze formula)
  rebend               a curved patch (rest radius 8 mm) rolled tighter and looser: bending energy
                       vs the plate's D/2 (1/R - 1/R0)^2 per area; also moved rigidly (no force)

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/plate_demo.py simulate
    /usr/bin/python3 scripts/soft_tissue/plate_demo.py render
Output: resources/soft_tissue/plate_demo/<case>.gif
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "resources" / "soft_tissue" / "plate_demo"
CASES = ("cantilever_elastica", "poisson_curl", "pinched_ring", "rebend")
SOLVERS = ("slow", "fast")


def _solve(sh, x0, fixed, key):
    import plate_test as pt

    t0 = time.perf_counter()
    x, _info = pt.solve(sh, x0, fixed, fast=(key == "fast"))
    return x, 1000 * (time.perf_counter() - t0)


def _strip(length, width, thickness, edge):
    import peel_model as pm
    import plate_test as pt

    rest, tris, idx = pm.strip_mesh(length, width, edge)
    p = pt.params(thickness)
    sh = pt.sheet(rest, tris, p)
    fixed = np.zeros((len(rest), 3), bool)
    fixed[idx[:3].ravel()] = True  # clamped root: three columns
    tip = idx[-1]
    fixed[tip, 2] = True
    return rest, tris, idx, p, sh, fixed, tip


def _bend_sweep(name, steps, length, width, thickness, edge, extra):
    import plate_test as pt

    rest, tris, idx, p, sh, fixed, tip = _strip(length, width, thickness, edge)
    arm = length - 2 * edge
    EI = pt.beam_EI(p, width)
    out = {k: dict(x=[], f=[], ms=[]) for k in SOLVERS}
    for key in SOLVERS:
        x = rest.copy()
        for d in steps:
            x0 = x.copy()
            x0[tip, 2] = -d
            x, ms = _solve(sh, x0, fixed, key)
            out[key]["x"].append(x.copy())
            out[key]["f"].append(-pt.reaction(sh, x, tip, 2))
            out[key]["ms"].append(ms)
            print(f"{name} [{key}] tip {d:5.2f} mm  force {out[key]['f'][-1] * 1000:8.3f} mN  {ms:7.0f} ms", flush=True)
    data = dict(steps=np.array(steps), tris=tris, idx=idx, arm=arm, EI=EI, nu=p.nu, length=length, width=width,
                thickness=thickness)
    for key in SOLVERS:
        for k, v in out[key].items():
            data[f"{key}_{k}"] = np.array(v)
    data.update(extra(data, p))
    np.savez_compressed(OUT / f"{name}.npz", **data)


def cantilever_elastica():
    import plate_test as pt

    steps = list(np.linspace(0.0, 9.5, 39)[1:])

    def extra(d, p):
        # exact elastica force at a fine set of deflections, for the curve
        dd = np.linspace(0.05, 9.5, 60)
        return dict(theory_d=dd, theory_f=np.array([pt.elastica_force(x, d["arm"], d["EI"]) for x in dd]))

    _bend_sweep("cantilever_elastica", steps, 20.0, 1.0, 1.2, 0.25, extra)


def poisson_curl():
    steps = list(np.linspace(0.0, 2.0, 21)[1:])

    def extra(d, p):
        idx = d["idx"]
        i = idx.shape[0] // 2
        mid = idx[:, idx.shape[1] // 2]
        ratios = {}
        for key in SOLVERS:
            r = []
            for x in d[f"{key}_x"]:
                across = 2 * np.polyfit(x[idx[i], 1], x[idx[i], 2], 2)[0]
                along = 2 * np.polyfit(x[mid[i - 3 : i + 4], 0], x[mid[i - 3 : i + 4], 2], 2)[0]
                r.append(across / along)
            ratios[f"{key}_ratio"] = np.array(r)
        return dict(section=idx[i], spine=mid, **ratios)

    _bend_sweep("poisson_curl", steps, 20.0, 4.0, 1.2, 0.25, extra)


def pinched_ring():
    import plate_test as pt

    radius, width, thickness, edge = 8.0, 1.0, 0.4, 0.25
    rest, tris, (bottom, top) = pt.ring(radius, width, edge)
    p = pt.params(thickness)
    sh = pt.sheet(rest, tris, p, curved=True)
    fixed = np.zeros((len(rest), 3), bool)
    fixed[bottom] = True
    fixed[top, 0] = fixed[top, 2] = True
    steps = list(np.linspace(0.0, 3.0, 31)[1:])
    out = {k: dict(x=[], f=[], ms=[]) for k in SOLVERS}
    for key in SOLVERS:
        x = rest.copy()
        prev = 0.0
        for s in steps:
            x0 = x.copy()
            x0[:, 2] -= (s - prev) * (rest[:, 2] / (2 * radius))
            x0[top, 2] = rest[top, 2] - s
            prev = s
            x, ms = _solve(sh, x0, fixed, key)
            out[key]["x"].append(x.copy())
            out[key]["f"].append(-pt.reaction(sh, x, top, 2))
            out[key]["ms"].append(ms)
            print(f"ring [{key}] squeeze {s:4.2f} mm  force {out[key]['f'][-1] * 1000:8.3f} mN  {ms:7.0f} ms", flush=True)
    # the middle row of the ring (y at mid width), in order round it
    y = rest[:, 1]
    row = np.flatnonzero(np.abs(y - np.median(np.unique(y))) < 1e-6)
    row = row[np.argsort(np.arctan2(rest[row, 0], radius - rest[row, 2]))]
    data = dict(steps=np.array(steps), rest=rest, row=row, radius=radius, width=width, thickness=thickness,
                slope=pt.beam_EI(p, width) / (pt.RING_COEF * radius**3),
                exact=np.array([pt.ring_exact_force(s, radius, pt.beam_EI(p, width)) for s in steps]))
    for key in SOLVERS:
        for k, v in out[key].items():
            data[f"{key}_{k}"] = np.array(v)
    np.savez_compressed(OUT / "pinched_ring.npz", **data)


def rebend():
    import fast_solver as fs
    import peel_model as pm
    import plate_test as pt

    R0 = 8.0
    rest, tris, flat = pt.cylinder_patch(R0, length=10.0, width=6.0, edge=0.25)
    p = pt.params(1.0)
    sh = pt.sheet(rest, tris, p, curved=True, calib_rest=flat)
    m = fs.FastModel.from_sheet(sh)
    area = float(sh.area.sum())
    radii = np.concatenate([np.linspace(R0, 4.0, 20), np.linspace(4.0, 16.0, 25)[1:], np.linspace(16.0, R0, 12)[1:]])
    rot = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    shapes, e_slow, e_fast, expect = [], [], [], []
    for R in radii:
        x = pm.roll_about_y(flat, R)
        shapes.append(x)
        e_slow.append(sh.energy(x, parts=True)[2]["bending"])
        e_fast.append(m.energy(np.ascontiguousarray(x))[0])
        expect.append(0.5 * p.bending_D * (1.0 / R - 1.0 / R0) ** 2 * area)
    # rigid motion: no force at any pose
    rigid, g_slow, g_fast = [], [], []
    for k in range(24):
        a = 2 * math.pi * k / 24
        x = rest @ rot(a).T + np.array([4 * math.sin(a), 3 * (1 - math.cos(a)), 2 * math.sin(2 * a)])
        rigid.append(x)
        g_slow.append(np.abs(sh.energy(x)[1]).max())
        g_fast.append(np.abs(m.energy(np.ascontiguousarray(x))[1]).max())
    np.savez_compressed(OUT / "rebend.npz", radii=radii, shapes=np.array(shapes), tris=tris, R0=R0,
                        e_slow=np.array(e_slow), e_fast=np.array(e_fast), expect=np.array(expect),
                        rigid=np.array(rigid), g_slow=np.array(g_slow), g_fast=np.array(g_fast))
    print(f"rebend: {len(radii)} radii, rigid max force {max(g_slow + g_fast):.1e} N")


def simulate(cases):
    sys.path.insert(0, str(HERE))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cases:
        globals()[name]()


# -- render (plain matplotlib) ----------------------------------------------------
COL = {"slow": "#3b6fb6", "fast": "#d9622b", "theory": "#888888"}
LABEL = {"slow": "slow reference", "fast": "fast C++ Newton"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _timing(d):
    return f"per load step: slow {np.median(d['slow_ms']):.0f} ms, fast {np.median(d['fast_ms']):.0f} ms (median)"


def render_cantilever_elastica(d, plt):
    steps, idx = d["steps"], d["idx"]
    spine = idx[:, idx.shape[1] // 2]
    fig, (ax, axf) = plt.subplots(2, 1, figsize=(8, 7.4), dpi=85, gridspec_kw={"height_ratios": [1.2, 1]})
    lin = 3 * d["EI"] / d["arm"] ** 3

    def update(i):
        ax.cla()
        ax.axvspan(-1, 0.5, color="#e9e4d8")
        ax.text(-0.9, 1.0, "clamp", fontsize=8, color="#7b6f55")
        ax.plot([d["length"]] * 2, [0, -steps[i]], color="#444", linewidth=0.8, linestyle=":")
        for key in SOLVERS:
            x = d[f"{key}_x"][i][spine]
            ax.plot(x[:, 0], x[:, 2], color=COL[key], linewidth=4 if key == "slow" else 1.6, label=LABEL[key],
                    alpha=0.6 if key == "slow" else 1)
        ax.set_xlim(-1.5, 21.5)
        ax.set_ylim(-11, 2)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)", fontsize=9)
        ax.set_ylabel("z (mm)", fontsize=9)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        ax.set_title(f"Cartilage strip 20 x 1 x 1.2 mm (E 5 MPa, nu 0.3), tip pushed down {steps[i]:.2f} mm", fontsize=10)
        _style(ax)
        axf.cla()
        axf.plot(d["theory_d"], d["theory_f"] * 1000, color=COL["theory"], linestyle="--", linewidth=1.2,
                 label="exact elastica")
        dd = np.linspace(0, 4, 20)
        axf.plot(dd, lin * dd * 1000, color=COL["theory"], linestyle=":", linewidth=1, label="small-load beam 3EI d / L^3")
        for key in SOLVERS:
            axf.plot(steps[: i + 1], d[f"{key}_f"][: i + 1] * 1000, "o" if key == "slow" else "-", color=COL[key],
                     markersize=5, linewidth=1.6, mfc="none", label=LABEL[key])
        r = d["fast_f"][i] / np.interp(steps[i], d["theory_d"], d["theory_f"])
        axf.text(0.02, 0.9, f"fast / exact = {r:.3f}   fast / slow = {d['fast_f'][i] / d['slow_f'][i]:.6f}",
                 transform=axf.transAxes, fontsize=8)
        axf.text(0.02, 0.8, _timing(d), transform=axf.transAxes, fontsize=8, color="#555")
        axf.set_xlim(0, 10)
        axf.set_ylim(0, d["theory_f"].max() * 1000 * 1.15)
        axf.set_xlabel("tip deflection (mm)", fontsize=9)
        axf.set_ylabel("tip force (mN)", fontsize=9)
        axf.legend(loc="lower right", fontsize=8, frameon=False)
        _style(axf)

    return fig, update, len(steps)


def render_poisson_curl(d, plt):
    steps, sec, spine = d["steps"], d["section"], d["spine"]
    nu = float(d["nu"])
    fig = plt.figure(figsize=(9, 8), dpi=85)
    ax3 = fig.add_subplot(2, 2, (1, 2), projection="3d")
    axs = fig.add_subplot(2, 2, 3)
    axr = fig.add_subplot(2, 2, 4)
    tris = d["tris"]
    i_mid = d["idx"].shape[0] // 2
    zmax = max(np.abs(d["slow_x"][-1][sec, 2] - d["slow_x"][-1][sec, 2].mean()).max(), 1e-6)

    def update(i):
        ax3.cla()
        x = d["fast_x"][i]
        ax3.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], triangles=tris, color="#9cc0f0", edgecolor="#5577aa",
                         linewidth=0.05, shade=True)
        xs = x[sec]
        ax3.plot(xs[:, 0], xs[:, 1], xs[:, 2] + 0.02, color="#d9622b", linewidth=2)
        ax3.set_box_aspect((20, 4, 5), zoom=1.5)
        ax3.set_xlim(0, 20)
        ax3.set_ylim(0, 4)
        ax3.set_zlim(-2.4, 0.6)
        ax3.view_init(elev=24, azim=-58)
        ax3.set_yticks([0, 4])
        fig.suptitle(f"4 mm cartilage strip, tip pushed down {steps[i]:.1f} mm (fast solver; orange = mid-span section)",
                      fontsize=9)
        ax3.tick_params(labelsize=7)
        axs.cla()
        along = None
        for key in SOLVERS:
            xs = d[f"{key}_x"][i][sec]
            y = xs[:, 1] - xs[:, 1].mean()
            z = xs[:, 2] - xs[:, 2].mean()
            axs.plot(y, z * 1000, "o" if key == "slow" else "-", color=COL[key], markersize=4, mfc="none",
                     linewidth=1.6, label=LABEL[key])
            if key == "slow":
                xp = d[f"{key}_x"][i][spine[i_mid - 3 : i_mid + 4]]
                along = 2 * np.polyfit(xp[:, 0], xp[:, 2], 2)[0]
        yy = np.linspace(y.min(), y.max(), 40)
        zz = 0.5 * (-nu * along) * yy**2
        axs.plot(yy, (zz - zz.mean()) * 1000, color=COL["theory"], linestyle="--", linewidth=1.2,
                 label="plate: -nu x curvature along")
        axs.set_ylim(-zmax * 1300, zmax * 1300)
        axs.set_xlabel("across the width (mm)", fontsize=9)
        axs.set_ylabel("height (um, mean removed)", fontsize=9)
        axs.set_title("mid-span cross-section", fontsize=9)
        axs.legend(loc="upper center", fontsize=7, frameon=False)
        _style(axs)
        axr.cla()
        axr.axhline(-nu, color=COL["theory"], linestyle="--", linewidth=1.2, label=f"plate: -nu = {-nu}")
        for key in SOLVERS:
            axr.plot(steps[: i + 1], d[f"{key}_ratio"][: i + 1], "o" if key == "slow" else "-", color=COL[key],
                     markersize=5, mfc="none", linewidth=1.6, label=LABEL[key])
        axr.set_xlim(0, steps[-1] * 1.05)
        axr.set_ylim(-0.4, 0)
        axr.set_xlabel("tip deflection (mm)", fontsize=9)
        axr.set_ylabel("curvature across / along", fontsize=9)
        axr.set_title("anticlastic curl", fontsize=9)
        axr.legend(loc="lower right", fontsize=7, frameon=False)
        axr.text(0.02, 0.06, _timing(d), transform=axr.transAxes, fontsize=6.5, color="#555")
        _style(axr)

    return fig, update, len(steps)


def render_pinched_ring(d, plt):
    steps, row, R = d["steps"], d["row"], float(d["radius"])
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(10, 5.2), dpi=85, gridspec_kw={"width_ratios": [1, 1.25]})

    def update(i):
        ax.cla()
        rest = d["rest"][row]
        ax.plot(np.r_[rest[:, 0], rest[:1, 0]], np.r_[rest[:, 2], rest[:1, 2]], color="#bbb", linewidth=1,
                linestyle=":", label="rest")
        for key in SOLVERS:
            x = d[f"{key}_x"][i][row]
            ax.plot(np.r_[x[:, 0], x[:1, 0]], np.r_[x[:, 2], x[:1, 2]], color=COL[key],
                    linewidth=4 if key == "slow" else 1.6, alpha=0.6 if key == "slow" else 1, label=LABEL[key])
        top = 2 * R - steps[i]
        ax.annotate("", xy=(0, top), xytext=(0, top + 2.2), arrowprops=dict(arrowstyle="->", color="#333"))
        ax.annotate("", xy=(0, 0), xytext=(0, -2.2), arrowprops=dict(arrowstyle="->", color="#333"))
        ax.set_xlim(-R - 2, R + 2)
        ax.set_ylim(-3, 2 * R + 3)
        ax.set_aspect("equal")
        ax.set_title(f"cartilage ring R 8 mm, 1 mm wide, 0.4 mm thick\nsqueezed {steps[i]:.1f} mm", fontsize=9)
        ax.legend(loc="center", fontsize=7, frameon=False)
        ax.set_xlabel("mm", fontsize=8)
        _style(ax)
        axf.cla()
        ss = np.linspace(0, steps[-1], 20)
        axf.plot(ss, d["slope"] * ss * 1000, color=COL["theory"], linestyle="--", linewidth=1.2,
                 label="small-squeeze formula (pi/4 - 2/pi) P R^3 / EI")
        axf.plot(steps, d["exact"] * 1000, color="#444", linestyle="-.", linewidth=1.2, label="exact ring elastica")
        for key in SOLVERS:
            axf.plot(steps[: i + 1], d[f"{key}_f"][: i + 1] * 1000, "o" if key == "slow" else "-", color=COL[key],
                     markersize=5, mfc="none", linewidth=1.6, label=LABEL[key])
        axf.text(0.02, 0.92, f"fast / exact = {d['fast_f'][i] / d['exact'][i]:.3f}   "
                 f"fast / slow = {d['fast_f'][i] / d['slow_f'][i]:.6f}", transform=axf.transAxes, fontsize=8)
        axf.text(0.02, 0.84, _timing(d), transform=axf.transAxes, fontsize=7.5, color="#555")
        axf.set_xlim(0, steps[-1] * 1.05)
        axf.set_ylim(0, max(d["slow_f"].max(), d["slope"] * steps[-1]) * 1000 * 1.25)
        axf.set_xlabel("squeeze (mm)", fontsize=9)
        axf.set_ylabel("pinch force (mN)", fontsize=9)
        axf.legend(loc="lower right", fontsize=7.5, frameon=False)
        _style(axf)

    return fig, update, len(steps)


def render_rebend(d, plt):
    radii, shapes, tris = d["radii"], d["shapes"], d["tris"]
    n1, n2 = len(radii), len(d["rigid"])
    fig = plt.figure(figsize=(10, 5.4), dpi=85)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    axe = fig.add_subplot(2, 2, 2)
    axg = fig.add_subplot(2, 2, 4)
    kappa = 1.0 / radii - 1.0 / float(d["R0"])

    def update(i):
        ax3.cla()
        rigid = i >= n1
        x = d["rigid"][i - n1] if rigid else shapes[i]
        ax3.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], triangles=tris, color="#9cc0f0", edgecolor="#5577aa",
                         linewidth=0.05, shade=True)
        ax3.set_xlim(-6, 10)
        ax3.set_ylim(-4, 10)
        ax3.set_zlim(-4, 10)
        ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=18, azim=-70)
        ax3.tick_params(labelsize=7)
        ax3.set_title("moved rigidly: should feel no force" if rigid else
                      f"curved cartilage patch, rest radius 8 mm, rolled to R = {radii[i]:.1f} mm", fontsize=9)
        axe.cla()
        order = np.argsort(kappa)
        axe.plot(kappa[order], d["expect"][order] * 1000, color=COL["theory"], linestyle="--", linewidth=1.2,
                 label="plate D/2 (1/R - 1/R0)^2 x area")
        k = min(i, n1 - 1)
        for key in SOLVERS:
            axe.plot(kappa[: k + 1], d[f"e_{key}"][: k + 1] * 1000, "o" if key == "slow" else "-", color=COL[key],
                     markersize=4, mfc="none", linewidth=1.6, label=LABEL[key])
        axe.plot(kappa[k], d["e_fast"][k] * 1000, "o", color=COL["fast"])
        r = d["e_fast"][k] / d["expect"][k] if d["expect"][k] > 0 else float("nan")
        axe.text(0.42, 0.6, f"fast / plate = {r:.3f}" if np.isfinite(r) else "at rest", transform=axe.transAxes, fontsize=8)
        axe.set_xlabel("change in curvature 1/R - 1/R0 (1/mm)", fontsize=8)
        axe.set_ylabel("bending energy (uJ)", fontsize=8)
        axe.legend(loc="upper left", fontsize=7, frameon=False)
        _style(axe)
        axg.cla()
        j = i - n1
        if j >= 0:
            for key in SOLVERS:
                axg.semilogy(np.arange(j + 1), np.maximum(d[f"g_{key}"][: j + 1], 1e-18), "o" if key == "slow" else "-",
                             color=COL[key], markersize=4, mfc="none", linewidth=1.6, label=LABEL[key])
        axg.set_yscale("log")
        axg.axhline(1e-6, color=COL["theory"], linestyle=":", linewidth=1)
        axg.text(0.5, 2e-6, "1 uN", fontsize=7, color="#777")
        axg.set_xlim(0, n2)
        axg.set_ylim(1e-16, 1e-4)
        axg.set_xlabel("rigid pose", fontsize=8)
        axg.set_ylabel("largest nodal force (N)", fontsize=8)
        if j >= 0:
            axg.legend(loc="upper right", fontsize=7, frameon=False)
        _style(axg)

    return fig, update, n1 + n2


def render(cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    for name in cases:
        d = dict(np.load(OUT / f"{name}.npz"))
        fig, update, n = globals()[f"render_{name}"](d, plt)
        fig.tight_layout()
        anim = FuncAnimation(fig, update, frames=n, blit=False)
        anim.save(OUT / f"{name}.gif", writer=PillowWriter(fps=6))
        plt.close(fig)
        print(f"Wrote {OUT / (name + '.gif')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["simulate", "render"])
    ap.add_argument("cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    simulate(a.cases) if a.mode == "simulate" else render(a.cases)
