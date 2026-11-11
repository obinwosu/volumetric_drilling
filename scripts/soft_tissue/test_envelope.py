"""Soft-tissue envelope, auricle framework and placement transforms on RT143.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_envelope.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import auricle as au  # noqa: E402
import envelope as ev  # noqa: E402
import placement_viewer as pv  # noqa: E402

ADF = ev.ROOT / "ADF" / "volume_256_RT143.yaml"
_VOL = _ENV = None


def _setup():
    global _VOL, _ENV
    if _VOL is None:
        _VOL = ev.Volume(ADF)
        _ENV = ev.Envelope(_VOL)
    return _VOL, _ENV


def _solid_tree(vol):
    pts = np.argwhere(vol.solid) * vol.mm
    return cKDTree(pts)


def test_layers_sit_at_their_depth():
    vol, env = _setup()
    layers = env.layers()
    tree = _solid_tree(vol)
    depth = 0.0
    for name, t, _src in ev.LAYERS:
        if name in ("temporalis_muscle", "temporalis_fascia", "auricular_muscles") or t == 0.0:
            assert name not in layers  # no outline / no muscle map / a potential space: no surface
            continue
        depth += t
        v, _tris = layers[name]
        v = v[:: max(1, len(v) // 20000)]
        gi = np.clip(np.round((v - env.origin) / env.h).astype(int), 0, np.array(env.shape) - 1)
        v = v[env.fat_taper()[gi[:, 0], gi[:, 1], gi[:, 2]] > 0.99]  # away from the canal, where fat tapers
        d, _ = tree.query(v)
        # distance to the nearest bone voxel centre (half a voxel under the bone surface)
        err = d - (depth + 0.5 * vol.mm.mean())
        assert abs(np.median(err)) < 0.25, (name, np.median(err))
        assert np.mean(np.abs(err) < 0.6) > 0.9, (name, np.percentile(err, [5, 50, 95]))


def test_no_fat_at_the_canal_opening():
    vol, env = _setup()
    f = vol.frame
    taper = env.fat_taper()
    t = env.thickness_fields(ev.Muscle())["subcutaneous_fat"]
    # cells whose nearest bone is right at the opening's rim have no fat; far away, the full share
    r = env.near_mm - f.canal_lateral_end
    near = (np.linalg.norm(r - (r @ f.lateral)[..., None] * f.lateral, axis=-1) < 7.5) & env.exterior
    assert t[near & (taper < 1e-3)].size > 0 and t[near].min() == 0.0
    assert abs(t.max() - dict((n, x) for n, x, _ in ev.LAYERS)["subcutaneous_fat"]) < 1e-6


def test_canal_is_open_and_middle_ear_bare():
    vol, env = _setup()
    layers = env.layers()
    for name, (v, _t) in layers.items():
        assert not env.in_canal(v).any(), name
    # no tissue drawn inside the middle ear: nothing near the ossicles
    oss = np.zeros(vol.shape, bool)
    import canal_skin as cs

    occupied, colors, *_ = cs.load_volume(ADF)
    for rgb in cs._ossicle_rgbs():
        oss |= occupied & cs._is_rgb(colors, rgb)
    tree = cKDTree(np.argwhere(oss) * vol.mm)
    for name, (v, _t) in layers.items():
        d, _ = tree.query(v)
        assert d.min() > 3.0, (name, d.min())


def test_tissue_lies_only_on_lateral_cortex():
    """No envelope over dura, sigmoid sinus or other labels: what lies under the tissue is the lateral bone region."""
    vol, env = _setup()
    import surface

    for name, (v, t) in env.layers().items():
        n = surface.vertex_normals(v, t)
        first = np.zeros(len(v), np.int8)  # 0 none, 1 cortex, 2 other solid
        for d in np.arange(0.0, 12.0, 0.15):
            i = np.round((v - n * d) / vol.mm).astype(int)
            ok = np.all((i >= 0) & (i < np.array(vol.shape)), axis=1) & (first == 0)
            hit = np.zeros(len(v), bool)
            hit[ok] = vol.solid[i[ok, 0], i[ok, 1], i[ok, 2]]
            g = np.clip(np.round(((v - n * d)[hit] - env.origin) / env.h).astype(int), 0, np.array(env.shape) - 1)
            first[hit] = np.where(env.region[g[:, 0], g[:, 1], g[:, 2]], 1, 2)
        # and nothing over sinus/dura: first solid along the lateral direction is not that label
        lab_other = vol.dura
        over = np.zeros(len(v), np.int8)
        for d in np.arange(0.0, 15.0, 0.15):
            i = np.round((v - vol.frame.lateral * d) / vol.mm).astype(int)
            ok = np.all((i >= 0) & (i < np.array(vol.shape)), axis=1) & (over == 0)
            hit = np.zeros(len(v), bool)
            hit[ok] = vol.solid[i[ok, 0], i[ok, 1], i[ok, 2]]
            over[hit] = np.where(lab_other[i[hit, 0], i[hit, 1], i[hit, 2]], 2, 1)
        assert np.mean(over == 2) < 0.02, (name, np.bincount(over, minlength=3))
        assert np.mean(first == 1) > 0.97, (name, np.bincount(first, minlength=3))


def test_muscle_fills_its_outline():
    vol, env = _setup()
    f = vol.frame
    # a 24 x 16 mm patch on the squama above the canal, centred where there is skin
    skin = env.layers()["skin"][0]
    r = skin - f.canal_lateral_end
    up = skin[(r @ f.superior > 6) & (r @ f.superior < 20) & (np.abs(r @ f.anterior) < 12)]
    assert len(up) > 100
    c = np.median(up, axis=0)
    outline = [c + a * f.anterior + s * f.superior for a, s in ((-12, -8), (12, -8), (12, 8), (-12, 8))]
    m = ev.Muscle(outline_mm=[p.tolist() for p in outline], max_mm=6.0, taper_mm=4.0)
    tm, tf, lo = env.muscle_map(m)
    uv_c = (f.uv(c) - lo) / ev.UV_MM
    assert abs(tm[int(round(uv_c[0])), int(round(uv_c[1]))] - 6.0) < 1e-3
    assert tm.max() <= 6.0 + 1e-6 and tf.max() <= 1.46 + 1e-6
    layers = env.layers(m)
    assert "temporalis_muscle" in layers and "temporalis_fascia" in layers
    plain = env.layers()
    # skin over the patch centre rises by about muscle + fascia
    tree0 = cKDTree(plain["skin"][0])
    near = layers["skin"][0][np.linalg.norm((layers["skin"][0] - c) - ((layers["skin"][0] - c) @ f.lateral)[:, None] * f.lateral, axis=1) < 2.0]
    assert len(near)
    rise = tree0.query(near)[0]
    assert abs(np.median(rise) - (6.0 + 1.46)) < 1.0, np.median(rise)


def test_framework_is_an_even_cover_inside_the_ear():
    import open3d as o3d

    v, t = pv.load_ear(pv.OBJ_EAR)
    fw = au.framework(v, t)
    thick = np.isfinite(fw.thickness_mm) & (fw.thickness_mm > 4 * au.COVER_MM)
    # even cover wherever the ear is thick (smoothing thins it only next to thin spots)
    assert np.mean(np.abs(fw.offset_mm[thick] - au.COVER_MM) < 1e-3) > 0.98
    assert fw.offset_mm.max() <= au.COVER_MM + 1e-9
    left = fw.thickness_mm - 2 * fw.offset_mm
    ok = np.isfinite(left)
    assert np.all(left[ok] >= au.KEEP_SHARE * fw.thickness_mm[ok] - 1e-9)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(t.astype(np.uint32)))
    sd = scene.compute_signed_distance(o3d.core.Tensor(fw.verts[::50].astype(np.float32))).numpy()
    assert np.mean(sd < 0) > 0.99, np.mean(sd < 0)  # framework inside the skin surface


def test_cut_trims_the_bulge_into_a_closed_auricle():
    import trimesh

    v, t = pv.load_ear(pv.OBJ_EAR)
    axes = au.ear_axes(v, t)
    vols = []
    for depth in (1.5, 2.5, 3.5):
        c = au.Cut(depth_mm=depth, tilt1_deg=5.0, bend2_mm=1.0)
        r = au.cut(v, t, c, axes=axes)
        m = trimesh.Trimesh(r.verts, r.tris, process=False)
        assert m.is_watertight and m.volume > 0, depth
        assert r.loops == 1, depth  # one outline: the cut does not break through the auricle
        frame = au._cut_frame(*axes, c)
        assert au.cut_value(r.verts, frame, c).max() < 1e-6  # nothing left past the cut
        capv = np.unique(r.tris[r.cap_tris])
        assert np.abs(au.cut_value(r.verts[capv], frame, c)).max() < 1e-6  # cap lies on the cut surface
        vols.append(m.volume)
    assert vols[0] < vols[1] < vols[2]
    full = trimesh.Trimesh(v, t, process=False).volume
    assert vols[1] < 0.75 * full


def test_ear_landmarks_and_auto_place():
    import open3d as o3d

    vol, env = _setup()
    v, t = pv.load_ear(pv.OBJ_EAR)
    mk = au.ear_landmarks(v, t)
    c = mk["center"]
    # the superior end is the wider (helix) end
    x = (v - c) @ mk["superior"]
    e2 = np.cross(mk["face"], mk["superior"])
    top, bot = (v - c)[x > np.percentile(x, 80)] @ e2, (v - c)[x < np.percentile(x, 20)] @ e2
    assert np.ptp(top) > np.ptp(bot)
    # the concha floor lies on the surface, on the face side
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(t.astype(np.uint32)))
    sd = scene.compute_signed_distance(o3d.core.Tensor(mk["concha"][None].astype(np.float32))).numpy()[0]
    assert abs(sd) < 0.5, sd
    p = pv.Placement(vol.frame, 0.5 * (v.min(axis=0) + v.max(axis=0)))
    p.auto(mk, tilt_deg=15, flare_deg=30, meatus_mm=14)
    T = p.matrix()
    q = pv.apply(T, mk["concha"][None])[0]
    assert np.allclose(q, vol.frame.canal_lateral_end + 14 * vol.frame.lateral, atol=1e-6)
    L = T[:3, :3] / p.scale
    f = vol.frame
    assert abs((L @ mk["face"]) @ (np.cos(np.radians(30)) * f.lateral + np.sin(np.radians(30)) * f.anterior) - 1) < 1e-6
    assert (L @ mk["superior"]) @ f.superior > 0.9
    # the default start clears the bone and stands the helix off the skin
    checks = pv.Checks(vol)
    checks.set_skin(*env.layers()["skin"])
    text = checks.report(pv.apply(T, v[::20]))
    assert "ear inside bone: 0 points" in text, text
    off = float(text.split("posterior helix off the skin: ")[1].split(" mm")[0])
    assert 15.0 < off < 30.0, text


def test_layer_solids_are_closed_and_cut():
    import trimesh

    vol, env = _setup()
    f = vol.frame
    p0 = f.canal_center - 8 * f.superior
    solids = env.layer_solids(cut=(p0, f.anterior))
    assert list(solids)[:2] == ["bone", "periosteum"]
    for name in ("periosteum", "tpf", "subcutaneous_fat", "skin"):
        v, t = solids[name]
        assert trimesh.Trimesh(v, t, process=False).is_watertight, name
        assert ((v - p0) @ f.anterior).max() < 0.5, name  # nothing past the cutaway plane


def test_funnel_and_incomplete_canal_sleeve():
    import trimesh

    import ear_cuts as ec

    vol, env = _setup()
    f = vol.frame
    v, t = pv.load_ear(pv.OBJ_EAR)
    mk = au.ear_landmarks(v, t)
    p = pv.Placement(f, 0.5 * (v.min(axis=0) + v.max(axis=0)))
    p.auto(mk)
    T = p.matrix()
    # a round meatus cut through the concha floor, along the ear's face
    a = mk["superior"]
    b = np.cross(mk["face"], a)
    ring = [mk["concha"] + 6.0 * (np.cos(x) * a + np.sin(x) * b) for x in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    cut = ec.MeshCut("skin", [r.tolist() for r in ring], (-mk["face"]).tolist(), 0.0)
    fw = au.framework(v, t)
    out = ec.assemble(vol, env, (v, t), (fw.verts, fw.tris), [cut], T, env.layers()["skin"])
    assert out["funnel"] is not None and len(out["funnel"][1]) > 100
    # the funnel starts on the canal's rim
    rim = ec.eac_rim(vol, env)
    from scipy.spatial import cKDTree

    fv = out["funnel"][0]
    first = fv[: ec.FUNNEL_POINTS] if len(fv) >= ec.FUNNEL_POINTS else fv
    assert np.median(cKDTree(rim).query(first)[0]) < 1.0
    # the sleeve is closed and leaves the posterosuperior wall bare
    sv, st = out["sleeve"]
    assert trimesh.Trimesh(sv, st, process=False).is_watertight
    # the bare arc faces posterosuperior about the canal's axis: most of the sleeve faces away
    L = f.lateral
    tilt = np.radians(ec.SLEEVE_GAP_TILT_DEG)
    gap = np.cos(tilt) * f.superior - np.sin(tilt) * f.anterior
    gap -= (gap @ L) * L
    gap /= np.linalg.norm(gap)
    m = len(sv) // (2 * ec.FUNNEL_RINGS)
    # the gap is measured about the canal axis but the points are spaced about the funnel's: allow slack
    assert ec.FUNNEL_POINTS * 0.5 < m <= ec.FUNNEL_POINTS * 0.75
    centre = fv.mean(axis=0)
    r = sv - centre
    r -= np.outer(r @ L, L)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    assert np.mean(r @ gap > np.cos(np.radians(ec.SLEEVE_GAP_DEG / 2))) < 0.1
    # canal skin: thick over cartilage, thin at the bony rim
    sk = out["funnel_skin_mm"]
    assert abs(sk[: ec.FUNNEL_POINTS].max() - ec.CANAL_SKIN_BONE_MM) < 1e-6 and abs(sk.max() - ec.CANAL_SKIN_CART_MM) < 1e-6


def test_automatic_meatus_opens_onto_the_canal():
    import ear_cuts as ec

    vol, env = _setup()
    f = vol.frame
    v, t = pv.load_ear(pv.OBJ_EAR)
    mk = au.ear_landmarks(v, t)
    p = pv.Placement(f, 0.5 * (v.min(axis=0) + v.max(axis=0)))
    p.auto(mk)
    T = p.matrix()
    fw = au.framework(v, t)
    out = ec.assemble(vol, env, (v, t), (fw.verts, fw.tris), [], T, env.layers()["skin"], meatus=ec.MeatusShape())
    depth = out["meatus_depth_mm"]
    assert 5.0 < depth < 30.0
    # looking out along the canal axis from inside the canal, the way is clear of skin and cartilage
    import open3d as o3d

    for name in ("skin", "cartilage"):
        vv, tt = out[name]
        w = pv.apply(T, vv)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(w.astype(np.float32)), o3d.core.Tensor(np.asarray(tt, np.uint32)))
        ray = np.concatenate([f.canal_lateral_end - f.lateral * 1.0, f.lateral])[None].astype(np.float32)
        hit = float(scene.cast_rays(o3d.core.Tensor(ray))["t_hit"].numpy()[0])
        assert not np.isfinite(hit) or hit > depth + 1.0, (name, hit, depth)
    assert out["funnel"] is not None and out["sleeve"] is not None


def test_placement_transforms():
    vol, _env = _setup()
    p = pv.Placement(vol.frame, [125.0, 106.0, 15.0])
    p.move_mm[:] = (3.0, 9.0, 5.0)
    p.turn_deg[:] = (10.0, -20.0, 15.0)
    p.scale = 0.92
    T = p.matrix()
    assert np.allclose(pv.apply(T, p.pivot), vol.frame.canal_lateral_end + 3 * vol.frame.anterior + 9 * vol.frame.superior + 5 * vol.frame.lateral)
    assert abs(np.cbrt(np.linalg.det(T[:3, :3])) - 0.92) < 1e-9
    q = pv.Placement(vol.frame, [0, 0, 0])
    q.load(p.to_dict())
    assert np.allclose(q.matrix(), T)
    pts = np.random.default_rng(0).uniform(0, 50, (20, 3))
    W = vol.mm_to_world_matrix()
    assert np.allclose(pv.apply(W, pts), vol.to_world_au(pts))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
