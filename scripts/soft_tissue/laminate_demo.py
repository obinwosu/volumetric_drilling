"""GIFs of the laminate tests (laminate_test.py), slow reference vs fast C++ Newton.

Every frame is a solved state, drawn with the exact answer.

  partial_interaction  two bonded layers (a 0.4 mm membrane, E 15 MPa, on 1 mm of cartilage) as a
                       cantilever, the bond swept from slipping freely to rigid: the slip between
                       the faces along the strip, and the tip stiffness, against Newmark's theory
  layer_peel           the outer layer peeled at 90 degrees off the held inner one: face bonds
                       break one by one; force against Kendall

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/laminate_demo.py simulate
    /usr/bin/python3 scripts/soft_tissue/laminate_demo.py render
Output: resources/soft_tissue/laminate_demo/<case>.gif
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "resources" / "soft_tissue" / "laminate_demo"
CASES = ("partial_interaction", "layer_peel")
SOLVERS = ("slow", "fast")
DEFLECTION = 0.2


def newmark_slip(top, bottom, width, arm, K, P, n=2001):
    """Slip between the faces along a two-layer cantilever under tip load P (Newmark), and x."""
    import laminate_test as lt

    EI0, _full, EA, z = lt.section([top, bottom], width)
    r = float(z[0] - z[1])
    inv_EA = 1.0 / EA[0] + 1.0 / EA[1]
    x = np.linspace(0.0, arm, n)
    M = P * (arm - x)
    if K > 0:
        a = math.sqrt(K * (inv_EA + r * r / EI0))
        beta = K * r / (EI0 * a * a)
        B = beta * P / a
        N = beta * M + B * (np.sinh(a * x) - np.tanh(a * arm) * np.cosh(a * x)) if a * arm < 300 else \
            beta * M - B * np.exp(-a * x)
    else:
        N = np.zeros_like(x)
    kappa = (M - r * N) / EI0
    ds = N * inv_EA - r * kappa
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(x))])
    return x, s


def partial_interaction():
    import laminate_test as lt

    top, bottom = lt.layer(**lt.TOP), lt.layer(**lt.BOTTOM)
    length, width, edge = 20.0, 1.0, 0.25
    arm = length - 2 * edge
    sweep = np.concatenate([[0.0], np.logspace(-0.5, 3.0, 22)])
    data = {"aL": sweep, "arm": arm, "edge": edge}
    for key in SOLVERS:
        P_, X_, slip_, ms_ = [], [], [], []
        for aL in sweep:
            K = lt.slip_modulus_for(top, bottom, width, aL, arm) if aL > 0 else 0.0
            law = lt.plane_law(max(K / width, 1e-9), lt.K_OPEN)
            t0 = time.perf_counter()
            P, arm, x, st, n = lt.cantilever([top, bottom], [law], DEFLECTION, length, width, edge, key == "fast")
            ms_.append(1000 * (time.perf_counter() - t0))
            pa, pb, r, nb = st.face_points(x)
            e = st.face_e
            ym = 0.5 * (st.rest[e[:, 0], 1] + st.rest[e[:, 1], 1]) - st.sheets[0].rest[:, 1].mean()
            xm = 0.5 * (st.rest[e[:, 0], 0] + st.rest[e[:, 1], 0])
            mid = np.abs(ym) < 1e-6
            order = np.argsort(xm[mid])
            P_.append(P)
            X_.append(x)
            slip_.append(np.stack([xm[mid][order] - 2 * edge, r[mid][order, 0]], axis=1))
            print(f"partial [{key}] alpha L {aL:8.2f}  P {P * 1000:7.3f} mN  {ms_[-1]:6.0f} ms", flush=True)
        data[f"{key}_P"] = np.array(P_)
        data[f"{key}_x"] = np.array(X_)
        data[f"{key}_slip"] = np.array(slip_, dtype=object)
        data[f"{key}_ms"] = np.array(ms_)
    data["n"] = st.n
    data["idx_mid"] = np.flatnonzero(np.abs(st.sheets[0].rest[:, 1] - st.sheets[0].rest[:, 1].mean()) < 1e-6)
    data["heights"] = np.array([p.thickness_mm for p in (top, bottom)])
    theory_P, theory_slip = [], []
    for aL in sweep:
        K = lt.slip_modulus_for(top, bottom, width, aL, arm) if aL > 0 else 0.0
        k = lt.partial_interaction_stiffness(top, bottom, width, arm, K)
        theory_P.append(k * DEFLECTION)
        theory_slip.append(np.stack(newmark_slip(top, bottom, width, arm, K, k * DEFLECTION), axis=1))
    fine = np.logspace(-1.0, 3.3, 200)
    data["fine_aL"] = fine
    data["fine_P"] = np.array([lt.partial_interaction_stiffness(top, bottom, width, arm, lt.slip_modulus_for(
        top, bottom, width, a, arm)) * DEFLECTION for a in fine])
    EI0, full, _EA, _z = lt.section([top, bottom], width)
    data["P_free"], data["P_full"] = 3 * EI0 / arm**3 * DEFLECTION, 3 * full / arm**3 * DEFLECTION
    data["theory_P"] = np.array(theory_P)
    data["theory_slip"] = np.array(theory_slip)
    np.savez_compressed(OUT / "partial_interaction.npz", **data, allow_pickle=True)


def layer_peel():
    import laminate_test as lt

    data = {}
    for key in SOLVERS:
        rec = []
        t0 = time.perf_counter()
        r, forces, lifts, expected = lt.peel(fast=(key == "fast"), step=0.1, record=rec)
        print(f"peel [{key}] steady / Kendall {r:.4f}  {time.perf_counter() - t0:.0f} s", flush=True)
        data[f"{key}_f"], data[f"{key}_x"] = forces, np.array([x for x, _b in rec])
        data[f"{key}_broken"] = np.array([b for _x, b in rec])
        data[f"{key}_ratio"] = r
    data["lifts"], data["expected"] = lifts, expected
    # geometry for drawing: the mid row of each layer, and where each face bond sits
    import peel_model as pm
    import tissue_params as tp

    top = lt.layer(**lt.TOP)
    z = tp.zone_layers([top], lt.PEEL_LAW)
    edge = min(0.25, z / 3.0)
    rest, tris, idx = pm.strip_mesh(11.0, 4 * edge, edge)
    data["mid"] = idx[:, idx.shape[1] // 2]
    data["n"] = len(rest)
    np.savez_compressed(OUT / "layer_peel.npz", **data)


def simulate(cases):
    sys.path.insert(0, str(HERE))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cases:
        globals()[name]()


# -- render -------------------------------------------------------------------------------
COL = {"slow": "#3b6fb6", "fast": "#d9622b", "theory": "#888888"}
LABEL = {"slow": "slow reference", "fast": "fast C++ Newton"}
LAYER = ["#e8b48f", "#7aa6d9"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def render_partial_interaction(d, plt):
    aL = d["aL"]
    n, mid, h = int(d["n"]), d["idx_mid"], d["heights"]
    fig = plt.figure(figsize=(10, 7.6), dpi=85)
    ax = fig.add_subplot(2, 2, (1, 2))
    axs = fig.add_subplot(2, 2, 3)
    axk = fig.add_subplot(2, 2, 4)
    smax = max(np.abs(np.asarray(t)[:, 1]).max() for t in d["theory_slip"]) * 1000

    def update(i):
        ax.cla()
        x = d["fast_x"][i]
        for k in range(2):
            line = x[k * n + mid]
            o = np.argsort(line[:, 0])
            h0 = float(np.median(line[o[:3], 2]))  # the layer's rest height (at the clamp)
            ax.plot(line[o, 0], h0 + (line[o, 2] - h0) * 10, color=LAYER[k], linewidth=2 + 9 * h[k], solid_capstyle="butt",
                    label=["outer layer: 0.4 mm, E 15 MPa", "inner layer (cartilage): 1 mm, E 5 MPa"][k])
        ax.axvspan(-0.5, 0.5, color="#e9e4d8")
        ax.set_xlim(-0.7, 20.5)
        ax.set_ylim(-2.8, 1.6)
        ax.set_yticks([])
        ax.set_xlabel("x (mm)    (deflection drawn x10)", fontsize=8)
        ax.legend(loc="lower left", fontsize=7.5, frameon=False)
        tag = "slip free (no bond)" if aL[i] == 0 else f"bond slip parameter  alpha L = {aL[i]:.3g}"
        ax.set_title(f"Two bonded layers as a cantilever, tip pushed 0.2 mm: {tag}", fontsize=10)
        _style(ax)
        axs.cla()
        t = np.asarray(d["theory_slip"][i])
        axs.plot(t[:, 0], np.abs(t[:, 1]) * 1000, color=COL["theory"], linestyle="--", linewidth=1.3, label="Newmark")
        for key in SOLVERS:
            sl = np.asarray(d[f"{key}_slip"][i])
            axs.plot(sl[:, 0], np.abs(sl[:, 1]) * 1000, "o" if key == "slow" else "-", color=COL[key], markersize=3,
                     mfc="none", linewidth=1.5, label=LABEL[key], markevery=3)
        axs.set_xlim(0, float(d["arm"]))
        axs.set_ylim(0, max(np.abs(t[:, 1]).max() * 1000, 1e-6) * 1.3)  # rescaled every frame
        axs.set_xlabel("distance from the clamp (mm)", fontsize=8)
        axs.set_ylabel("slip between the faces (um)", fontsize=8)
        axs.set_title("slip along the strip (axis rescales each frame)\nnext to the clamp the model converges with mesh",
                      fontsize=8.5)
        axs.legend(loc="upper left", fontsize=7, frameon=False)
        _style(axs)
        axk.cla()
        fa = d["fine_aL"]
        axk.semilogx(fa, d["fine_P"] * 1000, color=COL["theory"], linestyle="--", linewidth=1.3, label="Newmark")
        axk.axhline(d["P_free"] * 1000, color="#bbb", linewidth=0.8, linestyle=":")
        axk.axhline(d["P_full"] * 1000, color="#bbb", linewidth=0.8, linestyle=":")
        axk.text(60, d["P_free"] * 1000 * 1.04, "layers slip freely: sum EI", fontsize=7, color="#777")
        axk.text(0.12, d["P_full"] * 1000 * 1.02, "rigid bond: one section", fontsize=7, color="#777")
        xs = np.where(aL > 0, aL, 0.1)
        for key in SOLVERS:
            axk.semilogx(xs[: i + 1], d[f"{key}_P"][: i + 1] * 1000, "o" if key == "slow" else "s", color=COL[key],
                         markersize=6 if key == "slow" else 3, mfc="none" if key == "slow" else COL[key],
                         linestyle="none", label=LABEL[key])
        r = d["fast_P"][i] / d["theory_P"][i]
        axk.text(0.02, 0.6, f"fast / Newmark {r:.3f}\nfast / slow {d['fast_P'][i] / d['slow_P'][i]:.5f}\n"
                 f"per solve: slow {np.median(d['slow_ms']):.0f} ms, fast {np.median(d['fast_ms']):.0f} ms",
                 transform=axk.transAxes, fontsize=7.5)
        axk.set_xlim(0.08, 2.2e3)
        axk.set_ylim(0, d["P_full"] * 1000 * 1.15)
        axk.set_xlabel("alpha L (0 plotted at 0.1)", fontsize=8)
        axk.set_ylabel("tip force (mN)", fontsize=8)
        axk.set_title("bending stiffness", fontsize=9)
        axk.legend(loc="lower right", fontsize=7, frameon=False)
        _style(axk)

    return fig, update, len(aL)


def render_layer_peel(d, plt):
    lifts, n, mid = d["lifts"], int(d["n"]), d["mid"]
    every = 2
    frames = list(range(0, len(lifts), every))
    fig, (ax, axf) = plt.subplots(2, 1, figsize=(8.5, 7.6), dpi=85, gridspec_kw={"height_ratios": [1.3, 1]})

    def update(j):
        i = frames[j]
        ax.cla()
        x = d["fast_x"][i]
        for k, (hk, lab) in enumerate(((0.4, "outer layer (peeled)"), (1.0, "inner layer (held)"))):
            line = x[k * n + mid]
            ax.plot(line[:, 0], line[:, 2], color=LAYER[k], linewidth=2 + 7 * hk, solid_capstyle="butt", label=lab)
        if d["fast_broken"][i].any():
            pass
        ax.set_xlim(-1, 11.5)
        ax.set_ylim(-1.0, 8.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("z (mm)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)
        nb = int(d["fast_broken"][i].sum())
        ax.set_title(f"Peel the outer layer off at 90 degrees: lift {lifts[i]:.1f} mm, {nb} face bonds broken", fontsize=10)
        _style(ax)
        axf.cla()
        axf.axhline(d["expected"] * 1000, color=COL["theory"], linestyle="--", linewidth=1.3, label="Kendall (G = 0.2 N/mm)")
        for key in SOLVERS:
            axf.plot(lifts[: i + 1], d[f"{key}_f"][: i + 1] * 1000, "o" if key == "slow" else "-", color=COL[key],
                     markersize=4, mfc="none", linewidth=1.6, label=LABEL[key], markevery=2)
        axf.text(0.02, 0.9, f"steady force / Kendall: slow {float(d['slow_ratio']):.4f}, fast {float(d['fast_ratio']):.4f}",
                 transform=axf.transAxes, fontsize=8)
        axf.set_xlim(0, lifts[-1])
        axf.set_ylim(0, d["expected"] * 1000 * 1.4)
        axf.set_xlabel("grip lift (mm)", fontsize=8)
        axf.set_ylabel("grip force (mN)", fontsize=8)
        axf.legend(loc="lower right", fontsize=7.5, frameon=False)
        _style(axf)

    return fig, update, len(frames)


def render(cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    for name in cases:
        d = dict(np.load(OUT / f"{name}.npz", allow_pickle=True))
        fig, update, n = globals()[f"render_{name}"](d, plt)
        fig.tight_layout()
        FuncAnimation(fig, update, frames=n, blit=False).save(OUT / f"{name}.gif", writer=PillowWriter(fps=4 if name == "partial_interaction" else 10))
        plt.close(fig)
        print(f"Wrote {OUT / (name + '.gif')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["simulate", "render"])
    ap.add_argument("cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    simulate(a.cases) if a.mode == "simulate" else render(a.cases)
