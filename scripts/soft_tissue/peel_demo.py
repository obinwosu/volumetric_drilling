"""Side-by-side GIF of one peel test: slow reference solver vs fast C++ Newton.

Same skin strip, same bonds, peeled at 90 degrees off a rigid plate.
  slow   peel_model.solve, quasi-static, converged at every 0.1 mm of lift
  fast   fast_solver Newton, 3 iterations per 5 ms frame, grip pulled at 10 mm/s
The force on the grip is plotted against the Kendall answer.

Two steps, because the project venv has no matplotlib:
    middle_ear_newton/.venv/bin/python scripts/soft_tissue/peel_demo.py simulate
    /usr/bin/python3 scripts/soft_tissue/peel_demo.py render
Output: resources/soft_tissue/peel_demo/peel_slow_vs_fast.gif
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "resources" / "soft_tissue" / "peel_demo"
LENGTH, WIDTH, EDGE, TAB = 20.0, 3.0, 0.25, 4.0
LIFT, STEP = 13.0, 0.1  # mm of grip lift, and the slow solver's lift step
SPEED = 10.0 / 1000.0  # mm per ms
FRAME_MS, NEWTON_ITERS = 5.0, 3


def simulate():
    sys.path.insert(0, str(HERE))
    import fast_solver as fs
    import peel_model as pm
    import peel_strip as ps

    p = pm.SkinParams()
    rest, tris, idx = pm.strip_mesh(LENGTH, WIDTH, EDGE)
    bonded = np.flatnonzero(rest[:, 0] >= TAB - 1e-9)
    grip = idx[0]
    fixed = np.zeros(rest.shape, dtype=bool)
    fixed[grip, 2] = True
    lifts = np.arange(STEP, LIFT + 1e-9, STEP)

    def front(broken):
        intact = ~broken
        return float(rest[bonded[intact], 0].min()) if intact.any() else LENGTH

    # slow reference
    sheet = pm.Sheet(rest, tris, bonded, p)
    x = rest.copy()
    slow_x, slow_f, slow_front, slow_ms = [], [], [], []
    t0 = time.perf_counter()
    for k, lift in enumerate(lifts):
        x[grip, 2] = lift
        tick = time.perf_counter()
        x, _res = pm.solve(sheet, x, fixed)
        slow_ms.append(1000 * (time.perf_counter() - tick))
        _e, g = sheet.energy(x)
        sheet.commit(x)
        slow_x.append(x.copy())
        slow_f.append(float(g[grip, 2].sum()))
        slow_front.append(front(sheet.broken()))
        if k % 10 == 0:
            print(f"slow lift {lift:5.2f} mm  force {slow_f[-1] * 1000:6.1f} mN  {time.perf_counter() - t0:6.1f} s",
                  flush=True)

    # fast Newton, driven like a hand at 10 mm/s
    sheet = pm.Sheet(rest, tris, bonded, p)
    m = fs.FastModel.from_sheet(sheet)
    m.set_fixed(fixed)
    m.set_options(project_psd=True, threads=4)
    x = rest.copy()
    v = np.zeros_like(x)
    per_step = int(round(STEP / (SPEED * FRAME_MS)))  # frames per 0.1 mm
    fast_x, fast_f, fast_front, fast_ms = [], [], [], []
    frame = 0
    for lift in lifts:
        for _ in range(per_step):
            frame += 1
            x[grip, 2] = SPEED * FRAME_MS * frame
            stats = m.step(x, v, FRAME_MS, fs.NEWTON, NEWTON_ITERS, damping=0.1)
            m.commit(x)
            fast_ms.append(stats["ms"])
        _e, g = m.energy(x)
        fast_x.append(x.copy())
        fast_f.append(float(g[grip, 2].sum()))
        fast_front.append(front(m.bond_state()[1]))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "frames.npz", rest=rest, tris=tris, bonded=bonded, lifts=lifts,
        slow_x=np.array(slow_x), slow_f=np.array(slow_f), slow_front=np.array(slow_front), slow_ms=np.array(slow_ms),
        fast_x=np.array(fast_x), fast_f=np.array(fast_f), fast_front=np.array(fast_front), fast_ms=np.array(fast_ms),
        kendall=ps.kendall_force_per_width(p) * WIDTH, frame_ms=FRAME_MS, iters=NEWTON_ITERS,
    )
    print(f"slow: median {np.median(slow_ms):.0f} ms per step; fast: median {np.median(fast_ms):.2f} ms per 5 ms frame")
    print(f"Wrote {OUT / 'frames.npz'}")


def render(every=2, fps=15):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = np.load(OUT / "frames.npz")
    rest, tris, lifts = d["rest"], d["tris"], d["lifts"]
    kendall = float(d["kendall"])
    frames = list(range(0, len(lifts), every))
    if frames[-1] != len(lifts) - 1:
        frames.append(len(lifts) - 1)

    fig = plt.figure(figsize=(11, 7.6), dpi=90)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1])
    axes3d = [fig.add_subplot(gs[0, c], projection="3d") for c in range(2)]
    axf = fig.add_subplot(gs[1, :])
    titles = ["Slow reference (Python, converged each 0.1 mm)",
              f"Fast C++ Newton ({int(d['iters'])} iterations per {d['frame_ms']:g} ms frame, 10 mm/s)"]
    keys = ["slow", "fast"]
    colors = {"slow": "#3b6fb6", "fast": "#d9622b"}
    tri_bonded = np.isin(tris, d["bonded"]).all(axis=1)

    def draw_strip(ax, x, front, title):
        ax.cla()
        plate = [[(-1, -1, 0), (LENGTH + 1, -1, 0), (LENGTH + 1, WIDTH + 1, 0), (-1, WIDTH + 1, 0)]]
        ax.add_collection3d(Poly3DCollection(plate, facecolor="#cfcfcf", alpha=0.5, edgecolor="none"))
        # peeled skin vs skin still stuck to the plate
        stuck = tri_bonded & (rest[tris][:, :, 0].min(axis=1) >= front - 1e-9)
        faces = x[tris]
        ax.add_collection3d(Poly3DCollection(faces[~stuck], facecolor="#e8b49a", edgecolor="#9a5b3f", linewidth=0.15))
        ax.add_collection3d(Poly3DCollection(faces[stuck], facecolor="#c98b6b", edgecolor="#7a4430", linewidth=0.15))
        ax.set_xlim(-1, LENGTH + 1)
        ax.set_ylim(-1, WIDTH + 1)
        ax.set_zlim(0, LIFT + 1)
        ax.set_box_aspect((LENGTH + 2, WIDTH + 2, LIFT + 1), zoom=1.45)
        ax.view_init(elev=16, azim=-58)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.0)
        ax.set_xlabel("x (mm)", labelpad=-2)
        ax.set_zlabel("z (mm)", labelpad=-2)
        ax.set_yticks([])
        ax.tick_params(labelsize=7, pad=0)
        ax.set_title(title, fontsize=9.5)

    def update(i):
        k = frames[i]
        for ax, key, title in zip(axes3d, keys, titles):
            draw_strip(ax, d[f"{key}_x"][k], float(d[f"{key}_front"][k]), title)
        axf.cla()
        axf.axhline(kendall * 1000, color="#444", linestyle="--", linewidth=1)
        axf.text(0.2, kendall * 1000 * 1.03, f"Kendall steady peel force {kendall * 1000:.0f} mN", fontsize=8.5,
                 color="#444")
        for key, label in (("slow", "slow reference"), ("fast", "fast Newton")):
            axf.plot(lifts[:k + 1], d[f"{key}_f"][:k + 1] * 1000, color=colors[key], linewidth=1.8, label=label)
            axf.plot(lifts[k], d[f"{key}_f"][k] * 1000, "o", color=colors[key], markersize=4)
        axf.set_xlim(0, lifts[-1])
        axf.set_ylim(0, max(d["slow_f"].max(), d["fast_f"].max(), kendall) * 1000 * 1.25)
        axf.set_xlabel("grip lift (mm)")
        axf.set_ylabel("pull force (mN)")
        axf.legend(loc="lower right", fontsize=8.5, frameon=False)
        axf.text(0.01, 0.93, f"slow: {np.median(d['slow_ms']):.0f} ms per step   "
                             f"fast: {np.median(d['fast_ms']):.1f} ms per frame (real-time budget {d['frame_ms']:g} ms)",
                 transform=axf.transAxes, fontsize=8.5, va="top")
        axf.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"90° peel of a 20 × 3 mm skin strip (E 1 MPa, 1 mm thick, G 50 J/m²)    lift {lifts[k]:.1f} mm",
                     fontsize=11)
        return []

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, hspace=0.08, wspace=0.0)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    path = OUT / "peel_slow_vs_fast.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    print(f"Wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["simulate", "render"])
    args = ap.parse_args()
    simulate() if args.mode == "simulate" else render()
