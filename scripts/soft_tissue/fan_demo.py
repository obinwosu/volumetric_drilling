"""GIF of the fan test (fibre_test.py), slow reference vs fast C++ Newton.

A clamped 1 mm cartilage strip with a fan (tension-only fibres, placeholder J-curve, drawn thin so
the plate's bending and the fan's pull are of a size) spread across its tip from origins 15 mm above. The tip is lifted 1.5 mm (the fan goes slack: the plate alone) and then pushed
3 mm down (the fan pulls taut along its J-curve). Each frame is a solved state.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/fan_demo.py simulate
    /usr/bin/python3 scripts/soft_tissue/fan_demo.py render
Output: resources/soft_tissue/fan_demo/fan_on_plate.gif
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "resources" / "soft_tissue" / "fan_demo"
FAN_MM = 0.001  # a thin fan, so its pull and the plate's bending are both visible


def simulate():
    sys.path.insert(0, str(HERE))
    import fibre_test as ft

    OUT.mkdir(parents=True, exist_ok=True)
    ds = np.concatenate([np.linspace(0.0, -1.5, 9)[1:], np.linspace(-1.5, 3.0, 28)[1:]])
    data = {"d": ds}
    for key in ("slow", "fast"):
        for fan in (True, False):
            P_, X_, ms_ = [], [], []
            x = None
            for d in ds:
                t0 = time.perf_counter()
                P, k, x, st = ft.fan_on_plate(d, key == "fast", with_fan=fan, full=True, x_start=x, fan_thickness=FAN_MM)
                ms_.append(1000 * (time.perf_counter() - t0))
                P_.append(P)
                X_.append(x.copy())
            tag = f"{key}_{'fan' if fan else 'plate'}"
            data[tag + "_P"], data[tag + "_ms"] = np.array(P_), np.array(ms_)
            if fan:
                data[tag + "_x"] = np.array(X_)
                F = st.fibres
                data["fib_a"], data["fib_origin"], data["fib_L0"] = F.a, F.origin, F.rest_len
                data["tension"] = np.array([F.tension(xx) for xx in X_]) if key == "fast" else data.get("tension")
                data["k_fan"] = k
            print(f"{tag}: done ({np.median(ms_):.0f} ms per solve)", flush=True)
    data["tris"] = st.sheets[0].tris
    np.savez_compressed(OUT / "fan_on_plate.npz", **{k: v for k, v in data.items() if v is not None})


def render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = dict(np.load(OUT / "fan_on_plate.npz"))
    ds = d["d"]
    k_fan = float(d["k_fan"])
    k_plate = d["fast_plate_P"][np.argmin(np.abs(ds - 0.2))] / ds[np.argmin(np.abs(ds - 0.2))]
    fig = plt.figure(figsize=(10.5, 5.4), dpi=85)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axf = fig.add_subplot(1, 2, 2)
    tmax = float(d["tension"].max())
    cmap = plt.get_cmap("autumn_r")

    def update(i):
        ax.cla()
        x = d["fast_fan_x"][i]
        ax.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], triangles=d["tris"], color="#7aa6d9", edgecolor="#5577aa",
                        linewidth=0.05)
        T = d["tension"][i]
        for j in range(len(T)):
            p, o = x[d["fib_a"][j]], d["fib_origin"][j]
            ax.plot([p[0], o[0]], [p[1], o[1]], [p[2], o[2]], color="#bbbbbb" if T[j] <= 0 else cmap(0.25 + 0.75 * T[j] / tmax),
                    linewidth=0.9 if T[j] <= 0 else 1.6, linestyle=":" if T[j] <= 0 else "-")
        ax.set_xlim(0, 21)
        ax.set_ylim(-6, 10)
        ax.set_zlim(-4, 16)
        ax.set_box_aspect((21, 16, 20))
        ax.view_init(elev=12, azim=-35)
        ax.tick_params(labelsize=6)
        state = "slack (grey): the plate alone" if ds[i] < 0 else "taut (colour = tension)"
        ax.set_title(f"cartilage strip with a fan on its tip\ntip {'lifted' if ds[i] < 0 else 'pushed down'} "
                     f"{abs(ds[i]):.2f} mm: fan {state}", fontsize=9)
        axf.cla()
        dd = np.linspace(ds.min(), ds.max(), 50)
        axf.plot(dd, (k_plate * dd + np.where(dd > 0, k_fan * dd, 0.0)) * 1000, color="#888", linestyle="--",
                 linewidth=1.2, label="plate + fan, small-load stiffness\n(fan only in tension)")
        order = np.argsort(ds[: i + 1])
        for key, mk, col in (("slow", "o", "#3b6fb6"), ("fast", "-", "#d9622b")):
            axf.plot(ds[: i + 1][order], d[f"{key}_fan_P"][: i + 1][order] * 1000, mk, color=col, mfc="none",
                     markersize=5, linewidth=1.6, label=f"{key}: plate with fan")
            axf.plot(ds[: i + 1][order], d[f"{key}_plate_P"][: i + 1][order] * 1000, mk, color=col, alpha=0.35,
                     mfc="none", markersize=4, linewidth=1.0, label=f"{key}: plate alone")
        axf.axvline(0, color="#ddd", linewidth=0.8)
        axf.axhline(0, color="#ddd", linewidth=0.8)
        axf.set_xlim(ds.min() - 0.1, ds.max() + 0.1)
        axf.set_ylim(min(d["fast_plate_P"].min(), 0) * 1000 * 1.2, d["fast_fan_P"].max() * 1000 * 1.1)
        axf.set_xlabel("tip pushed down (mm)", fontsize=8)
        axf.set_ylabel("tip force (mN)", fontsize=8)
        axf.legend(loc="upper left", fontsize=7, frameon=False)
        axf.text(0.98, 0.04, f"per solve: slow {np.median(d['slow_fan_ms']):.0f} ms, fast {np.median(d['fast_fan_ms']):.0f} ms",
                 transform=axf.transAxes, ha="right", fontsize=7, color="#555")
        axf.spines[["top", "right"]].set_visible(False)
        axf.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FuncAnimation(fig, update, frames=len(ds), blit=False).save(OUT / "fan_on_plate.gif", writer=PillowWriter(fps=5))
    print(f"Wrote {OUT / 'fan_on_plate.gif'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["simulate", "render"])
    simulate() if ap.parse_args().mode == "simulate" else render()
