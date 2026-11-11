"""Draw the mesh sweep (mesh_sweep.py): look vs speed, and where each model's drawn skin departs.

    /usr/bin/python3 scripts/soft_tissue/mesh_sweep_plot.py   (needs matplotlib)

Writes resources/soft_tissue/mesh_sweep/{look_vs_speed.png, error_maps.png}.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "resources" / "soft_tissue" / "mesh_sweep"
BLUE, ORANGE, GREY = "#2a6fdb", "#e07b1f", "#888888"


def look_vs_speed(R):
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=130)
    rows = R["results"]
    ax.axvspan(0.3, 1000 / 60, color="#e8f3e8", zorder=0)
    ax.text(1000 / 60 * 0.93, 0.085, "fits a 60 fps frame", ha="right", va="bottom", color="#3a7a3a", fontsize=9)
    for layers, col, label in ((3, BLUE, "3 layers (skin / cartilage / skin)"), (1, ORANGE, "1 equivalent shell")):
        rs = [r for r in rows if r["layers"] == layers and r["tag"] != "e1_L3"]
        x = [r["live"][0]["phys_ms_moving"] for r in rs]  # 1 Newton iteration per frame
        y = [r["err_mean_mm"] for r in rs]
        y95 = [r["err_p95_mm"] for r in rs]
        ax.plot(x, y, "o-", color=col, label=label + ": mean", lw=1.6)
        ax.plot(x, y95, "^:", color=col, alpha=0.6, label=label + ": 95th percentile", lw=1)
        for r, xi, yi in zip(rs, x, y):
            ax.annotate(f"{r['edge_mm']:g} mm\n{r['vertices']} v", (xi, yi), textcoords="offset points", xytext=(6, -14),
                        fontsize=7.5, color=col)
    ref = rows[0]
    ax.axvline(ref["live"][0]["phys_ms_moving"], color=GREY, ls="--", lw=1)
    ax.text(ref["live"][0]["phys_ms_moving"] * 1.05, 0.5, f"reference:\n1 mm, 3 layers\n({ref['vertices']} v)",
            ha="left", fontsize=8, color=GREY)
    today = next(r for r in rows if r["tag"] == "e1.5_L3")
    t4 = today["live"][2]["phys_ms_moving"]
    ax.plot([t4], [today["err_mean_mm"]], "s", mfc="none", mec="k", ms=9)
    ax.annotate("today's viewer:\n1.5 mm, 3 layers,\n4 Newton steps a frame", (t4, today["err_mean_mm"]), xytext=(-10, 30),
                textcoords="offset points", fontsize=8, ha="right", arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1200)
    ax.set_xlabel("physics per frame while pulling, ms (1 Newton step a frame; log)")
    ax.set_ylabel("drawn skin vs reference, mm (log)")
    ax.set_title(f"How far the drawn ear is from the finest model, against its cost\n"
                 f"{R['force_n']:g} N pinch on the helix, up and back; the reference skin moves "
                 f"{ref['ref_motion_mean_mm']:.1f} mm on average, {ref['ref_motion_max_mm']:.1f} mm at most", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(0.08, 10)
    ax.legend(fontsize=7.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "look_vs_speed.png")


FRAME = None


def _view(v, centre=None):
    """Seen from the side (lateral, face to the right... anterior to the left of the page for a right
    ear): 2D page coordinates (posterior, superior) and depth toward the viewer."""
    L, A, S = (np.asarray(FRAME[k]) for k in ("lateral", "anterior", "superior"))
    c = v.mean(axis=0) if centre is None else centre
    return (v - c) @ -A, (v - c) @ S, (v - c) @ L


def error_maps(R, Z, tags):
    rest, tris = Z["skin_rest"], Z["skin_tris"]
    ref = Z[R["results"][0]["tag"]]
    keep = tris
    a, b, depth = _view(ref)
    order = np.argsort(depth[keep].mean(axis=1))  # back to front
    keep = keep[order]
    fig, axes = plt.subplots(2, (len(tags) + 2) // 2, figsize=(4.0 * ((len(tags) + 2) // 2), 8.2), dpi=110)
    axes = axes.ravel()
    move = np.linalg.norm(ref - rest, axis=1)
    panels = [("reference: how far the skin moves", move, None)] + [(t, np.linalg.norm(Z[t] - ref, axis=1), t) for t in tags]
    by = {r["tag"]: r for r in R["results"]}
    for ax, (title, val, tag) in zip(axes, panels):
        shape = ref if tag is None else Z[tag]
        a, b, _ = _view(shape, ref.mean(axis=0))
        vmax = 8.0 if tag is None else 1.0
        cmap = "viridis" if tag is None else "magma"
        pc = PolyCollection(np.stack([a, b], axis=1)[keep], array=val[keep].mean(axis=1), cmap=cmap, edgecolors="none")
        pc.set_clim(0, vmax)
        ax.add_collection(pc)
        ax.set_xlim(a.min() - 2, a.max() + 2)
        ax.set_ylim(b.min() - 2, b.max() + 2)
        ax.set_aspect("equal")
        ax.axis("off")
        if tag is None:
            ax.set_title(title, fontsize=9)
            fig.colorbar(pc, ax=ax, shrink=0.7, label="mm")
        else:
            r = by[tag]
            fps = 1000 / (r["live"][0]["phys_ms_moving"] + r["live"][0]["place_ms_median"])
            ax.set_title(f"{r['edge_mm']:g} mm, {r['layers']} layer{'s' if r['layers'] > 1 else ''} ({r['vertices']} v)\n"
                         f"off by {r['err_mean_mm']:.2f} mm mean; {fps:.0f} fps", fontsize=9)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1.0), cmap="magma"), ax=axes[1:].tolist(), shrink=0.5,
                 label="distance from the reference skin, mm")
    fig.suptitle("Drawn skin under the same pinch, coloured by how far it sits from the 1 mm, 3-layer answer", fontsize=11)
    fig.savefig(OUT / "error_maps.png")


def main():
    global FRAME
    R = json.loads((OUT / "results.json").read_text())
    FRAME = R["frame"]
    Z = np.load(OUT / "shapes.npz")
    look_vs_speed(R)
    error_maps(R, Z, ["e1.5_L3", "e2_L3", "e1.5_L1", "e2_L1", "e3_L1", "e4_L3", "e6_L3"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
