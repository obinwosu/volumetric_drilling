"""Place atlas meshes (Z-Anatomy, right side) on the patient: fit the atlas temporal bone to the CT bone.

The patient scene (volume_256_RT143) is a ~50 mm cube of CT around the ear; the atlas is a whole
head in metres. Steps:

1. Start: the atlas's anatomical axes onto the patient's (atlas: lateral -X, anterior -Y, superior
   +Z; patient: ``vol.frame``), the atlas's bony ear canal mouth onto the patient's.
2. Trimmed similarity ICP (rotation, shift, one scale): atlas bone points that land inside the CT
   cube (2 mm in from its faces, where the CT bone mesh is cut) pull toward their nearest CT bone
   point; pairs further than a shrinking cutoff are ignored (the atlas is another person).
3. The same transform carries the muscles and fascia across; they are written as
   ``atlas_<name>.ply`` next to the composite (CC BY-SA 4.0, see resources/soft_tissue/atlas).

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/atlas_register.py

Prints the fit (scale, residuals) and writes atlas_registration.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import auricle_sim as asim  # noqa: E402
import envelope as ev  # noqa: E402

ATLAS = ev.ROOT / "resources" / "soft_tissue" / "atlas" / "z_anatomy"
OUT = asim.COMPOSITE
CARRY = ("temporalis_muscle_r", "posterior_auricular_muscle_r", "superficial_layer_of_temporal_fascia_r",
         "temporoparietalis_muscle_r", "temporal_bone_r", "eminentia_conchae_r", "mastoid_region_r")
INSET_MM = 2.0
CUTOFFS_MM = (12.0, 8.0, 6.0, 4.0, 3.0, 3.0, 2.5, 2.5)


def load_atlas(name):
    import trimesh

    m = trimesh.load(ATLAS / f"{name}.obj", process=False, force="mesh")
    return np.asarray(m.vertices, float) * 1000.0, np.asarray(m.faces)


def atlas_meatus(bone_v, bone_t, concha_v):
    """The atlas's bony ear canal: rays from the side (along +X, lateral to medial for the right
    ear) on a 1 mm grid around the concha label; the canal is where a ray goes deepest before
    meeting bone while its surroundings meet bone early. Returns the mouth (where the canal's rim
    is) in mm."""
    import trimesh

    m = trimesh.Trimesh(bone_v, bone_t, process=False)
    c = concha_v.mean(axis=0)
    ys, zs = np.mgrid[c[1] - 20:c[1] + 20:1.0, c[2] - 20:c[2] + 20:1.0]
    x0 = bone_v[:, 0].min() - 5.0
    o = np.stack([np.full(ys.size, x0), ys.ravel(), zs.ravel()], axis=1)
    loc, ri, _ = trimesh.ray.ray_triangle.RayMeshIntersector(m).intersects_location(
        o, np.repeat([[1.0, 0.0, 0.0]], len(o), axis=0), multiple_hits=False)
    depth = np.full(len(o), np.nan)
    depth[ri] = loc[:, 0] - x0
    D = depth.reshape(ys.shape)
    # depth beyond the ring 5 mm around (a tunnel, not the bone's outline)
    best, arg = -np.inf, None
    for i in range(5, D.shape[0] - 5):
        for j in range(5, D.shape[1] - 5):
            ring = [D[i + di, j + dj] for di, dj in ((5, 0), (-5, 0), (0, 5), (0, -5), (4, 4), (-4, 4), (4, -4), (-4, -4))]
            if np.isnan(D[i, j]) or np.isnan(ring).any():
                continue
            s = D[i, j] - np.max(ring)
            if s > best:
                best, arg = s, (i, j)
    i, j = arg
    ring = [D[i + di, j + dj] for di, dj in ((5, 0), (-5, 0), (0, 5), (0, -5))]
    mouth = np.array([x0 + np.mean(ring), ys[i, j], zs[i, j]])
    return mouth, float(best)


def patient_meatus(vol, bone_v):
    """The patient's bony canal mouth: along the canal axis from its lateral end inward, the first
    point whose 5 mm ring is mostly bone-near (CT bone surface within 1.5 mm)."""
    from scipy.spatial import cKDTree

    f = vol.frame
    L = np.asarray(f.lateral, float)
    u = np.cross(L, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    w = np.cross(L, u)
    tree = cKDTree(bone_v)
    p0 = np.asarray(f.canal_lateral_end, float)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for t in np.arange(0.0, 25.0, 0.25):
        c = p0 - t * L
        ring = c + 5.0 * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w)
        if (tree.query(ring)[0] < 1.5).mean() > 0.5:
            return c, t
    return p0 - 8.0 * L, None


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|^2."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a**2).sum(1).mean()
    return s, R, md - s * R @ ms


def main(log=print):
    import trimesh
    from scipy.spatial import cKDTree

    vol = ev.Volume(asim.ADF)
    bone = trimesh.load(OUT / "bone.ply", process=False)
    pv = np.asarray(bone.vertices, float)
    lo, hi = pv.min(0) + INSET_MM, pv.max(0) - INSET_MM
    pin = pv[((pv > lo) & (pv < hi)).all(1)]
    pin = pin[np.random.default_rng(0).choice(len(pin), min(len(pin), 60000), replace=False)]
    tree = cKDTree(pin)

    av, at = load_atlas("temporal_bone_r")
    concha, _ = load_atlas("concha_of_auricle_r")
    am, depth = atlas_meatus(av, at, concha)
    pm_, t_in = patient_meatus(vol, pv)
    log(f"atlas meatus mouth {am.round(1)} (canal {depth:.1f} mm deeper than its rim); "
        f"patient mouth {pm_.round(1)} ({t_in} mm in from the canal's lateral end)")

    f = vol.frame
    P = np.stack([np.asarray(f.lateral, float), np.asarray(f.anterior, float), np.asarray(f.superior, float)], axis=1)
    Aa = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]).T  # atlas lateral, anterior, superior as columns
    U, _s, Vt = np.linalg.svd(P @ Aa.T)
    R = U @ Vt
    s = 1.0
    t = pm_ - R @ am
    dense = av
    # densify the atlas bone a little (its edges are ~2 mm): triangle centres as extra points
    dense = np.concatenate([av, av[at].mean(axis=1)])
    for k, cut in enumerate(CUTOFFS_MM):
        x = s * dense @ R.T + t
        inside = ((x > lo) & (x < hi)).all(1)
        d, j = tree.query(x[inside])
        ok = d < cut
        src, dst = dense[inside][ok], pin[j[ok]]
        s, R, t = umeyama(src, dst)
        log(f"  ICP {k}: cutoff {cut:4.1f} mm, {ok.sum()} pairs, residual median {np.median(d[ok]):.2f} mm, scale {s:.3f}")
    x = s * dense @ R.T + t
    inside = ((x > lo) & (x < hi)).all(1)
    d, _j = tree.query(x[inside])
    # the other way: CT bone near the atlas bone (does the atlas cover the patient?)
    d2, _ = cKDTree(x[inside]).query(pin)
    fit = {"scale": float(s), "R": R.tolist(), "t_mm": t.tolist(), "atlas_units": "m (x1000 to mm first)",
           "atlas_inside_points": int(inside.sum()), "atlas_to_ct_median_mm": float(np.median(d)),
           "atlas_to_ct_p90_mm": float(np.percentile(d, 90)), "ct_to_atlas_median_mm": float(np.median(d2)),
           "ct_to_atlas_p90_mm": float(np.percentile(d2, 90)),
           "rotation_from_start_deg": None}
    log(f"fit: scale {s:.3f}; atlas bone to CT bone median {fit['atlas_to_ct_median_mm']:.2f} mm (90% "
        f"{fit['atlas_to_ct_p90_mm']:.2f}); CT bone to atlas median {fit['ct_to_atlas_median_mm']:.2f} mm "
        f"(90% {fit['ct_to_atlas_p90_mm']:.2f})")
    for name in CARRY:
        v, tri = load_atlas(name)
        y = s * v @ R.T + t
        ev.write_ply(OUT / f"atlas_{name.removesuffix('_r')}.ply", y, tri, (200, 90, 80))
        inside = ((y > pv.min(0)) & (y < pv.max(0))).all(1).mean()
        log(f"  wrote atlas_{name.removesuffix('_r')}.ply: {len(v)} vertices, {inside:.0%} inside the CT cube")
    (OUT / "atlas_registration.json").write_text(json.dumps(fit, indent=1))
    return fit


PLACEMENT = OUT / "atlas_placement.json"


def fit_matrix():
    """4x4, atlas mm -> volume mm, from the saved ICP fit."""
    d = json.loads((OUT / "atlas_registration.json").read_text())
    T = np.eye(4)
    T[:3, :3] = d["scale"] * np.asarray(d["R"], float)
    T[:3, 3] = d["t_mm"]
    return T


def user_matrix(frame, pivot, move_mm, turn_deg, scale):
    """4x4 adjustment in volume mm (atlas_placement.py): scale and turn about ``pivot`` (turns about
    anterior, superior, lateral), then move along them."""
    import placement_viewer as pv

    A, S, L = (np.asarray(x, float) for x in (frame.anterior, frame.superior, frame.lateral))
    R = pv.rot(L, turn_deg[2]) @ pv.rot(S, turn_deg[1]) @ pv.rot(A, turn_deg[0])
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = pivot + move_mm[0] * A + move_mm[1] * S + move_mm[2] * L - scale * R @ pivot
    return T


def placed_matrix(frame):
    """Fit plus the user's adjustment, if saved."""
    T = fit_matrix()
    if PLACEMENT.exists():
        d = json.loads(PLACEMENT.read_text())
        T = user_matrix(frame, np.asarray(d["pivot_mm"], float), d["move_mm"], d["turn_deg"], d["scale"]) @ T
    return T


def write_placed(T, log=print):
    for name in CARRY:
        v, tri = load_atlas(name)
        y = v @ T[:3, :3].T + T[:3, 3]
        ev.write_ply(OUT / f"atlas_{name.removesuffix('_r')}.ply", y, tri, (200, 90, 80))
    log(f"wrote {len(CARRY)} placed atlas meshes to {OUT}")


if __name__ == "__main__":
    main()
