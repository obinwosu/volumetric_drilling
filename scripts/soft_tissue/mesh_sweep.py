"""How much physics does the ear need to look right? Mesh size and layer count vs look and speed.

The same set force (a pinch on the helix, pulled up and back) is applied to the auricle built at
several mid-surface edge sizes, as the full 3-layer laminate (skin / cartilage / skin bonded face to
face) and as one equivalent shell. The drawn skin (auricle_skin.ply, the same fine mesh for every
model) is carried on each model's mid-surface, as in the pull viewer, and compared with the finest
3-layer answer. Perception is judged on what is drawn, not on the physics mesh.

Measured per model:
  look      converged drawn-skin shape under the force vs the reference: mean / 95th percentile /
            worst vertex distance (mm), and how far the pinch moved
  live      the force ramped in over RAMP frames and held for HOLD frames, with k Newton iterations
            per frame (k = 1, 2, 4): physics ms per frame while the force changes, drawn-skin placement ms, and frames until
            the drawn skin is within SETTLE_MM of its own converged shape
  size      vertices, degrees of freedom, elements

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/mesh_sweep.py [--quick]

Writes resources/soft_tissue/mesh_sweep/{results.json, shapes.npz}; mesh_sweep_plot.py draws it.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import auricle_sim as asim  # noqa: E402
import envelope as ev  # noqa: E402

OUT = ev.ROOT / "resources" / "soft_tissue" / "mesh_sweep"
EDGES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FORCE_N = 3.0  # about 3 mm of pull on today's model: clearly visible, still well inside the J-curve toe
GRIP_MM = 3.0
RAMP, HOLD = 6, 24  # a quick tug: the force is all on in 6 frames
ITERS = (1, 2, 4)
SETTLE_MM = 0.2  # drawn-skin error a viewer will not notice at arm's length (an assumption to revisit)
THREADS = 4  # as in the pull viewer


def direction(vol):
    S, A = np.asarray(vol.frame.superior, float), np.asarray(vol.frame.anterior, float)
    d = S - 0.5 * A  # up and back
    return d / np.linalg.norm(d)


def load(model, grip_point, d, force):
    """Force per vertex: ``force`` along d shared over the pinch (all layers) by vertex area."""
    st = model.stack
    local = np.flatnonzero((np.linalg.norm(model.mid_v - grip_point, axis=1) <= GRIP_MM) & ~model.base)
    if not len(local):
        local = np.array([int(np.argmin(np.linalg.norm(model.mid_v - grip_point, axis=1)))])
    ids = np.concatenate([st.layer_vertices(k, local) for k in range(model.L)])
    w = st.vert_area[ids]
    f = np.zeros_like(st.rest)
    f[ids] = (force * w / w.sum())[:, None] * d
    return f, ids


def build(vol, edge, layers, log):
    import fast_solver as fs

    t0 = time.time()
    model = asim.AuricleModel(vol, edge_mm=edge, layers=layers, log=log)
    m = fs.FastModel.from_stack(model.stack)
    m.set_options(project_psd=True, threads=THREADS)
    m.set_factor(fs.best_factor())
    m.set_fixed(model.held)
    return model, m, time.time() - t0


def converged(m, x0, f):
    import fast_solver as fs

    m.set_external(f)
    x = x0.copy()
    t0 = time.perf_counter()
    r = m.step(x, np.zeros_like(x), 0.0, fs.NEWTON, 300, gtol=1e-7)
    return x, r, time.perf_counter() - t0


def targets(m, rest, f, emb, cart):
    """Converged drawn skin at each force level of the ramp (what a perfect solver would show)."""
    x, out = rest.copy(), []
    for k in range(RAMP):
        x, _r, _t = converged(m, x, f * (k + 1) / RAMP)
        out.append(emb.place(x[cart]))
    return out


def live(m, rest, f, iters, emb, cart, want):
    """Ramp the force in, hold it; per frame physics ms, placement ms, drawn error vs converged."""
    import fast_solver as fs

    x, v = rest.copy(), np.zeros_like(rest)
    phys, place, err, nfac = [], [], [], []
    for k in range(RAMP + HOLD):
        m.set_external(f * min(1.0, (k + 1) / RAMP))
        r = m.step(x, v, 0.0, fs.NEWTON, iters, gtol=1e-7)
        v[:] = 0.0
        phys.append(r["ms"])
        nfac.append(r["factorizations"])
        t0 = time.perf_counter()
        drawn = emb.place(x[cart])
        place.append(1000 * (time.perf_counter() - t0))
        err.append(float(np.linalg.norm(drawn - want[min(k, RAMP - 1)], axis=1).mean()))
    err = np.array(err)
    hold = err[RAMP - 1:]
    ok = np.flatnonzero(hold < SETTLE_MM)
    settle = int(ok[0]) if len(ok) and (hold[ok[0]:] < SETTLE_MM).all() else None
    # while the force changes (the hand moving); frame 0 also sets up the sparse pattern
    moving = float(np.mean(phys[1:RAMP]))
    return {"iters": iters, "phys_ms_moving": moving, "phys_ms_first": float(phys[0]),
            "place_ms_median": float(np.median(place)), "frames_to_settle_after_ramp": settle,
            "final_mean_err_mm": float(hold[-1]), "worst_lag_mm": float(err.max()), "err_per_frame_mm": err.tolist(),
            "phys_ms": phys, "factorizations": nfac}


def main():
    import auricle_pull_viewer as apv

    quick = "--quick" in sys.argv
    edges = (1.5, 3.0) if quick else EDGES
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True)
    vol = ev.Volume(asim.ADF)
    d = direction(vol)
    frame = {k: np.asarray(getattr(vol.frame, k), float).tolist() for k in ("lateral", "anterior", "superior")}
    skin_v, skin_t = apv.load_drawn(["auricle_skin"])["auricle_skin"]
    # the pinch point: the default model's helix top, the same place on every mesh
    grip_point = asim.AuricleModel(vol, log=lambda *a: None).helix_top()
    configs = [(e, L) for e in sorted(edges) for L in (3, 1)]
    results, shapes = [], {"skin_rest": skin_v, "skin_tris": skin_t}
    ref = None
    for edge, layers in configs:
        tag = f"e{edge:g}_L{layers}"
        log(f"\n== {tag}: edge {edge} mm, {layers} layer(s)")
        model, m, t_build = build(vol, edge, layers, log=lambda *a: None)
        st = model.stack
        cart = st.layer_slice(model.cart_layer)
        emb = apv.Embedding(model.mid_v, model.tris, skin_v)
        f, ids = load(model, grip_point, d, FORCE_N)
        x, r, t_conv = converged(m, st.rest.copy(), f)
        drawn = emb.place(x[cart])
        grip_move = float(((x[ids] - st.rest[ids]).mean(axis=0)) @ d)
        if ref is None:
            ref = drawn  # the finest 3-layer model comes first
        dist = np.linalg.norm(drawn - ref, axis=1)
        moved = np.linalg.norm(ref - skin_v, axis=1)
        row = {"tag": tag, "edge_mm": edge, "layers": layers, "vertices": int(len(st.rest)),
               "dofs": int(3 * len(st.rest)), "mid_vertices": int(len(model.mid_v)), "triangles": int(len(model.tris)),
               "face_bonds": int(len(st.face_e)) if layers > 1 else 0, "grip_vertices": int(len(ids)),
               "build_s": t_build, "converge_iterations": r["iterations"], "converge_s": t_conv,
               "converge_max_grad": r["max_grad"], "grip_move_mm": grip_move,
               "err_mean_mm": float(dist.mean()), "err_p95_mm": float(np.percentile(dist, 95)),
               "err_max_mm": float(dist.max()), "ref_motion_mean_mm": float(moved.mean()),
               "ref_motion_max_mm": float(moved.max())}
        if layers == 1:
            row["shell"] = {"E_mpa": model.shell.E_mpa, "thickness_mm": model.shell.thickness_mm}
        log(f"  {row['vertices']} vertices, converged in {r['iterations']} iterations ({t_conv:.1f} s); "
            f"pinch moved {grip_move:.2f} mm; drawn skin vs reference: mean {row['err_mean_mm']:.3f}, "
            f"95% {row['err_p95_mm']:.3f}, worst {row['err_max_mm']:.3f} mm")
        row["live"] = []
        want = targets(m, st.rest.copy(), f, emb, cart)
        for k in ITERS:
            lv = live(m, st.rest.copy(), f, k, emb, cart, want)
            row["live"].append(lv)
            fps = 1000 / (lv["phys_ms_moving"] + lv["place_ms_median"])
            log(f"  live, {k} Newton/frame, while pulling: physics {lv['phys_ms_moving']:.1f} ms + placing the skin "
                f"{lv['place_ms_median']:.1f} ms = {fps:.1f} fps; worst lag {lv['worst_lag_mm']:.2f} mm, settles "
                f"{lv['frames_to_settle_after_ramp']} frames after the ramp; {sum(lv['factorizations'])} factorizations")
        results.append(row)
        shapes[tag] = drawn
        (OUT / "results.json").write_text(json.dumps({"force_n": FORCE_N, "direction": d.tolist(),
                                                     "grip_point": np.asarray(grip_point).tolist(),
                                                     "ramp": RAMP, "hold": HOLD, "settle_mm": SETTLE_MM,
                                                     "threads": THREADS, "frame": frame, "results": results}, indent=1))
        np.savez_compressed(OUT / "shapes.npz", **shapes)
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
