"""Side-view GIFs of the layered (composite) tests, for review.

  mastoid_plane_selection  grip the skin: skin and TPF peel off periosteum together
  mastoid_full_thickness   grip all three: they come off bone together
  grasp_periosteum         skin flap already elevated; grasp periosteum only

Fast C++ Newton, converged 5 ms frames. Layers share one surface in the model;
for drawing, each is shown offset by the thickness of the layers under it.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/composite_demo.py simulate
    /usr/bin/python3 scripts/soft_tissue/composite_demo.py render
Output: resources/soft_tissue/composite_demo/<case>.gif
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parents[1] / "resources" / "soft_tissue" / "composite_demo"
CASES = ("mastoid_plane_selection", "mastoid_full_thickness", "grasp_periosteum")


def _record(name):
    sys.path.insert(0, str(HERE))
    import composite_bench as cb
    import peel_model as pm
    import stack as sk
    import tissue_params as tp

    spec = tp.STACKS["mastoid"]
    params = [tp.skin_params(m, spec["attachment"]) for m in spec["layers"]]
    if name == "grasp_periosteum":
        length, width, edge, tab = 12.0, 0.8, 0.2, 0.5
        rest, tris, idx = pm.strip_mesh(length, width, edge)
        rx = rest[:, 0]
        st = sk.from_table("mastoid", rest, tris, pair_masks=[np.ones(len(rest), bool), rx >= 8.0], attach_mask=rx >= tab)
        grip = st.layer_vertices(2, idx[0])
        lift_total, speed = 6.0, 10.0
        expected = None
    else:
        grab = [0] if name == "mastoid_plane_selection" else [0, 1, 2]
        arm = [params[i] for i in grab] if name != "mastoid_plane_selection" else params[:2]
        law = tp.INTERFACES["areolar" if name == "mastoid_plane_selection" else "periosteum_bone"].opening.bond_law()
        length, width, edge, tab = 20.0, 1.0, 0.25, 2.0
        rest, tris, idx = pm.strip_mesh(length, width, edge)
        rx = rest[:, 0]
        both = [np.ones(len(rest), bool) if (k + 1 in grab and k in grab) else rx >= tab for k in range(2)]
        st = sk.from_table("mastoid", rest, tris, pair_masks=both, attach_mask=rx >= tab)
        grip = np.concatenate([st.layer_vertices(k, idx[0]) for k in grab])
        lift_total, speed = 22.0, 10.0
        expected = tp.kendall_layers(arm, law.G_n_per_mm) * width
    drv = cb.Lift(st, grip, "fast", speed_mm_per_s=speed)
    mid = idx[:, idx.shape[1] // 2]
    frames, forces, lifts = [], [], []
    k = 0
    while drv.lift() < lift_total:
        f = drv.advance()
        k += 1
        forces.append(f)
        lifts.append(drv.lift())
        if k % 8 == 0:
            frames.append(np.stack([drv.x[st.layer_vertices(L, mid)][:, [0, 2]] for L in range(st.L)]))
    np.savez_compressed(OUT / f"{name}.npz", frames=np.array(frames), forces=np.array(forces), lifts=np.array(lifts),
                        every=8, thickness=np.array([p.thickness_mm for p in params]), names=np.array(spec["layers"]),
                        expected=np.nan if expected is None else expected, length=length)
    print(f"{name}: {len(frames)} frames")


def simulate():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        _record(name)


TITLES = {
    "mastoid_plane_selection": "Grip the skin: skin + TPF peel off periosteum (loose areolar plane); periosteum stays on bone",
    "mastoid_full_thickness": "Grip all three layers: they come off bone together",
    "grasp_periosteum": "Skin flap already elevated; grasp periosteum only: it peels off bone, skin and TPF stay\n"
                        "(known gap: released layers do not collide yet, so the periosteum passes through the flap)",
}


def render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    colors = ["#d9906b", "#e8d27a", "#7aa6d9"]
    for name in CASES:
        d = np.load(OUT / f"{name}.npz")
        frames, forces, lifts = d["frames"], d["forces"], d["lifts"]
        every = int(d["every"])
        h = d["thickness"]
        offsets = [h[k + 1:].sum() + 0.5 * h[k] for k in range(len(h))]  # draw each layer above the ones under it
        fig, (ax, axf) = plt.subplots(2, 1, figsize=(9, 7), dpi=85, gridspec_kw={"height_ratios": [2.2, 1]})

        def update(i):
            ax.cla()
            ax.axhspan(-1.0, 0.0, color="#e9e4d8")
            ax.text(0.2, -0.6, "bone", fontsize=9, color="#7b6f55")
            for k in range(frames.shape[1]):
                line = frames[i, k]
                ax.plot(line[:, 0], line[:, 1] + offsets[k], color=colors[k], linewidth=1.2 + 4 * h[k], label=str(d["names"][k]),
                        solid_capstyle="butt")
            ax.set_xlim(-2, float(d["length"]) + 1)
            ax.set_ylim(-1, max(8.0, float(frames[:, :, :, 1].max()) + 1.5))
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("height (mm)")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.set_title(TITLES[name], fontsize=10)
            n = (i + 1) * every
            axf.cla()
            axf.plot(lifts[:n], forces[:n] * 1000, color="#333", linewidth=1.5)
            if np.isfinite(float(d["expected"])):
                axf.axhline(float(d["expected"]) * 1000, color="#999", linestyle="--", linewidth=1)
                axf.text(0.2, float(d["expected"]) * 1000 * 1.05, "Kendall (pure peel)", fontsize=8, color="#666")
            axf.set_xlim(0, lifts[-1])
            axf.set_ylim(0, forces.max() * 1000 * 1.2)
            axf.set_xlabel("grip lift (mm)")
            axf.set_ylabel("grip force (mN)")
            axf.spines[["top", "right"]].set_visible(False)
            return []

        fig.tight_layout()
        anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
        anim.save(OUT / f"{name}.gif", writer=PillowWriter(fps=12))
        plt.close(fig)
        print(f"Wrote {OUT / (name + '.gif')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["simulate", "render"])
    simulate() if ap.parse_args().mode == "simulate" else render()
