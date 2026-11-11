"""CT ear meatus: canal carved open, buried tissue trimmed, canal lined.

Runs on RT143 with the saved CT-ear placement (skipped when it is missing).

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_ct_meatus.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ct_meatus as cm  # noqa: E402
import ear_cuts as ec  # noqa: E402
import envelope as ev  # noqa: E402
import placement_viewer as pv  # noqa: E402

ADF = ev.ROOT / "ADF" / "volume_256_RT143.yaml"


def test_smooth_max_rounds_only_the_crease():
    a = np.array([0.0, 5.0, -5.0, 1.0])
    b = np.array([0.0, -5.0, 5.0, 1.2])
    s = cm.smooth_max(a, b, 1.0)
    assert np.allclose(s[1:3], [5.0, 5.0])  # far apart: the plain maximum
    assert s[0] > 0 and s[3] > 1.2  # close: a little above both
    assert np.allclose(cm.smooth_min(a, b, 1.0), -cm.smooth_max(-a, -b, 1.0))


def _scene():
    path = ev.placement_path(ADF, pv.CT_EAR)
    if not (path.exists() and pv.CT_EAR.exists()):
        return None
    doc = json.loads(path.read_text())
    vol = ev.Volume(ADF)
    env = ev.Envelope(vol)
    env.auricle_ramp = ev.AuricleRamp()  # the viewer's default: thin around the ear
    th = {n: t for n, t, _ in ev.LAYERS}
    th.update(ev.upgrade_thickness(doc.get("thickness_mm", {})))
    layers = env.layers(ev.Muscle(**doc.get("muscle", {})), th)
    v, t = pv.load_ear(pv.CT_EAR)
    T = np.asarray(doc["ear"]["ear_to_volume_mm"], float)
    return vol, env, layers, (pv.apply(T, v), t), th


def test_auricle_ramp_thins_near_the_canal_only():
    vol = ev.Volume(ADF)
    env = ev.Envelope(vol)
    th = {n: t for n, t, _ in ev.LAYERS}
    ramp = ev.AuricleRamp()
    env.auricle_ramp = ramp
    tf = env.thickness_fields(ev.Muscle(), th)
    d = env.canal_distance()
    near = d < ramp.near_mm
    far = d > ramp.far_mm
    for name, t in ramp.near.items():
        f = np.broadcast_to(tf[name], d.shape)
        assert np.allclose(f[near], t, atol=1e-4), name
        full = th[name] * (env.fat_taper() if name == "subcutaneous_fat" else 1.0)
        assert np.allclose(f[far], np.broadcast_to(full, d.shape)[far], atol=1e-4), name
    mid = (d > ramp.near_mm + 1) & (d < ramp.far_mm - 1)
    skin = np.broadcast_to(tf["skin"], d.shape)[mid]
    assert skin.min() > ramp.near["skin"] and skin.max() < th["skin"]
    env.auricle_ramp = None  # off: today's uniform envelope
    assert np.allclose(env.thickness_fields(ev.Muscle(), th)["skin"], th["skin"])


def test_flatten_loop_reaches_its_depth_past_the_first_surface():
    import open3d as o3d

    box = o3d.geometry.TriangleMesh.create_box(20.0, 20.0, 10.0)  # top face at z = 10
    ear = (np.asarray(box.vertices, float), np.asarray(box.triangles))
    loop = [[5.0, 5.0, 20.0], [15.0, 5.0, 20.0], [15.0, 15.0, 20.0], [5.0, 15.0, 20.0]]
    cut = ec.MeshCut("flatten", loop, [0.0, 0.0, -1.0], 3.0)  # looking down, 3 mm deep
    p = np.array([[10.0, 10.0, 9.5], [10.0, 10.0, 8.0], [10.0, 10.0, 5.0], [2.0, 10.0, 9.5], [10.0, 10.0, 7.0]])
    w = cm.flatten_weight(p, [cut], ear)
    assert w[0] > 0.99 and w[1] > 0.99  # inside the loop, near the top
    assert w[2] < 0.01  # deeper than the loop reaches
    assert w[3] < 0.01  # outside the loop
    assert 0.0 < w[4] < 1.0  # the slope at the loop's depth
    assert np.allclose(cm.flatten_weight(p, [], ear), 0.0)


def test_envelope_does_not_wrap_behind_the_bone():
    vol = ev.Volume(ADF)
    env = ev.Envelope(vol)
    env.auricle_ramp = ev.AuricleRamp()
    sv = env.layers(ev.Muscle(), None)["skin"][0]
    gi = np.clip(np.round((sv - env.origin) / env.h).astype(int), 0, np.array(env.shape) - 1)
    off = sv - env.near_mm[gi[:, 0], gi[:, 1], gi[:, 2]]
    facing = (off @ vol.frame.lateral) / np.linalg.norm(off, axis=1)
    assert np.mean(facing < 0) < 0.001, np.mean(facing < 0)  # none over bone facing away from lateral


def test_blender_bundle_round_trip():
    import tempfile

    import blender_edit as be

    rng = np.random.default_rng(0)
    v = rng.random((50, 3)) * 10
    t = rng.integers(0, 50, (80, 3))
    w = rng.random(50)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    T[:3, 3] = [5, 6, 7]
    folder = pathlib.Path(tempfile.mkdtemp())
    placed = v @ T[:3, :3].T + T[:3, 3]
    be.export_bundle(folder, (placed, t), (v, t), (v, t), (v, t), weight=w, info=dict(ear_to_volume_mm=T.tolist()))
    assert (folder / "fivrs_scene.py").exists() and (folder / "bundle.json").exists()
    compile(be.BLENDER_SCRIPT, "fivrs_scene.py", "exec")
    assert be.read_edit(folder) is None  # nothing saved from Blender yet
    be.write_ply(folder / be.EDITED, placed, t, w)  # what Save for FIVRS writes
    pv_, pt, pw = be.read_edit(folder)
    back = (pv_ - T[:3, 3]) @ T[:3, :3]
    assert np.allclose(back, v, atol=1e-4) and (pt == t).all() and np.allclose(pw, w, atol=1e-6)


def test_painted_auricle_gets_cartilage_and_the_rest_goes_under():
    import open3d as o3d

    s = _scene()
    if s is None:
        print("  (no saved CT-ear placement: skipped)")
        return
    vol, env, layers, ear, th = s
    v = np.asarray(ear[0])
    eac = cm.EacField(vol)
    # paint: the lower half of the ear (along superior) is auricle, the upper half is not
    sup = (v - eac.end_centre) @ vol.frame.superior
    aw = (sup < 0).astype(np.float32)
    out = cm.open_meatus(vol, env, ear, layers["skin"], th, eac=eac, log=lambda *_: None, auricle=aw)
    fv, ft = (np.asarray(x) for x in out["framework"])
    assert len(ft) > 1000
    fsup = (fv - eac.end_centre) @ vol.frame.superior
    assert np.mean(fsup < 1.5) > 0.98, np.mean(fsup < 1.5)  # cartilage only where painted
    # the cartilage lies under the skin: at least most of the cover in from the opened ear
    ov, ot = (np.asarray(x) for x in out["ear"])
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(ov.astype(np.float32)), o3d.core.Tensor(ot.astype(np.uint32)))
    d = sc.compute_distance(o3d.core.Tensor(fv.astype(np.float32))).numpy()
    assert np.median(d) > 0.8 * cm.au.COVER_MM, np.median(d)  # (it meets the skin only along the trim line)
    # the unpainted upper half stands no higher than the envelope's skin (pressed under it)
    osup = (ov - eac.end_centre) @ vol.frame.superior
    hg = cm.height_over_envelope(env, th, None, ov[osup > 8.0])
    assert np.mean(hg > 1.0) < 0.02, np.mean(hg > 1.0)


def test_sheet_ear_holds_water_and_keeps_its_cartilage_inside():
    """The Blender skin sheet joined in: water poured into the auricle leaves only down the canal,
    the canal is open at the eardrum, the cartilage stays under the skin, and no envelope skin is
    left under the auricle."""
    import blender_edit as be
    import cartilage as ca
    import sheet_skin as ss

    folder = be.bundle_dir("volume_256_RT143", "ct_right_ear")
    s = _scene()
    if s is None or be.read_edit(folder) is None:
        print("  (no Blender skin sheet: skipped)")
        return
    vol, env, layers, _ear, th = s
    eac = cm.EacField(vol)
    v, t, w = be.read_edit(folder)
    ramp = env.auricle_ramp
    env = ev.Envelope(vol, face=ss.front_edge(v, t, vol.frame, eac))  # with the face's tissue, as the viewer builds it
    env.auricle_ramp = ramp
    env.auricle_ramp.base_outline = ss.base_outline(v, t, eac)
    layers = env.layers(ev.Muscle(), th)
    out = ss.join_sheet(vol, env, (v, t, w), layers["skin"], th, eac=eac, log=lambda *_: None,
                        others={k: m for k, m in layers.items() if k != "skin"})
    assert ss.meatus_leaks(out, eac) == []
    cv, ct = out["canal"]
    loops = ca.boundary_loops(ct)
    tl, _ = eac.axial(np.asarray(cv))
    assert len(loops) == 2  # open at the meatus and at the eardrum, nowhere else
    far = min(loops, key=lambda lp: np.median(tl[lp]))
    assert np.ptp(np.asarray(cv)[far], axis=0).max() > 6.0  # the eardrum's opening: about the tympanic ring
    if out["framework"] is not None:
        sv, st = out["ear"]  # the skin you see
        import open3d as o3d

        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(sv, np.float32)), o3d.core.Tensor(np.asarray(st, np.uint32)))
        fv = np.asarray(out["framework"][0], np.float32)
        r = sc.compute_closest_points(o3d.core.Tensor(fv))
        off = fv - r["points"].numpy()
        d = np.linalg.norm(off, axis=1)
        side = np.einsum("ij,ij->i", off, r["primitive_normals"].numpy())
        outside = (side > 0.05) & (np.abs(side) > 0.3 * d) & (d < 8)
        # past the auricle's front edge the face's skin covers it (the face's tissue comes up to the edge)
        L = np.asarray(vol.frame.lateral, np.float32)
        es = o3d.t.geometry.RaycastingScene()
        es.add_triangles(o3d.core.Tensor(np.asarray(out["scalp"][0], np.float32)), o3d.core.Tensor(np.asarray(out["scalp"][1], np.uint32)))
        q = fv[outside]
        up = es.cast_rays(o3d.core.Tensor(np.concatenate([q, np.broadcast_to(L, q.shape)], axis=1)))["t_hit"].numpy()
        outside[np.nonzero(outside)[0][up < 8.0]] = False
        assert outside.mean() < 0.005, outside.mean()  # the cartilage does not poke out of the skin
    gone = out["skin_keep"](np.asarray(layers["skin"][0])) > 0
    kept = np.asarray(out["scalp"][0])
    assert gone.any() and len(kept) < len(layers["skin"][0])


def test_edited_cartilage_gets_back_skin_and_a_cleft():
    """The cartilage as edited in Blender goes on past the sheet's base edge at the back and top
    of the auricle: skin closes the auricle there, stretched smooth from the sheet's base edge
    down to the envelope (the crease), the envelope's skin stays under it (the cleft), and almost
    no cartilage is left in the open."""
    import open3d as o3d

    import blender_edit as be
    import sheet_skin as ss

    folder = be.bundle_dir("volume_256_RT143", "ct_right_ear")
    s = _scene()
    if s is None or be.read_edit(folder) is None or be.read_cartilage_edit(folder) is None:
        print("  (no Blender skin sheet or cartilage: skipped)")
        return
    vol, env, layers, _ear, th = s
    eac = cm.EacField(vol)
    v, t, w = be.read_edit(folder)
    env.auricle_ramp.base_outline = ss.base_outline(v, t, eac)
    layers = env.layers(ev.Muscle(), th)
    out = ss.join_sheet(vol, env, (v, t, w), layers["skin"], th, eac=eac, log=lambda *_: None,
                        others={k: m for k, m in layers.items() if k != "skin"}, cartilage=be.read_cartilage_edit(folder))
    assert ss.meatus_leaks(out, eac) == []
    mv, mt = out["medial"]
    assert len(mt) > 1000
    # the cleft: envelope skin with the back skin facing down over it is kept
    L = np.asarray(vol.frame.lateral, float)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(mv, np.float32)), o3d.core.Tensor(np.asarray(mt, np.uint32)))
    skin = np.asarray(layers["skin"][0], np.float32)
    up = sc.cast_rays(o3d.core.Tensor(np.concatenate([skin, np.broadcast_to(L, skin.shape)], 1).astype(np.float32)))["t_hit"].numpy()
    under_back = np.isfinite(up) & (up > 1.0)
    kept = out["skin_keep"](skin) < 0
    # (a strip just behind the front edge is cleared on purpose: under the auricle where it meets the face)
    assert under_back.sum() > 50 and kept[under_back].mean() > 0.85, kept[under_back].mean()
    # cartilage in the open: its outward ray meets no tissue at all
    every = o3d.t.geometry.RaycastingScene()
    for m in [out["ear"], out["scalp"], *out["layers"].values(), vol.bone_mesh()]:
        every.add_triangles(o3d.core.Tensor(np.asarray(m[0], np.float32)), o3d.core.Tensor(np.asarray(m[1], np.uint32)))
    fv, ft = (np.asarray(x) for x in out["framework"])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(fv), o3d.utility.Vector3iVector(ft))
    m.compute_vertex_normals()
    n = np.asarray(m.vertex_normals)
    hit = every.cast_rays(o3d.core.Tensor(np.concatenate([fv + 0.02 * n, n], 1).astype(np.float32)))["t_hit"].numpy()
    # a coarse measure (any ray slipping out through a gap counts); what can be seen is at the sheet's open front edge
    assert np.mean(~np.isfinite(hit)) < 0.04, np.mean(~np.isfinite(hit))
    # the back skin is stretched smooth across the opening: no lumps over the cartilage
    import cartilage as ca

    mean, _deg = ca._neighbour_mean(np.asarray(mv, float), np.asarray(mt))
    edge = np.zeros(len(mv), bool)
    edge[np.concatenate(ca.boundary_loops(np.asarray(mt)))] = True
    assert np.linalg.norm(mean - mv, axis=1)[~edge].max() < 1e-6  # each point at its neighbours' average


def bowl_leaks(out, vol, eac):
    """Where water poured into the auricle could get out, other than over its rim or down the canal.

    Hold the head with the canal pointing down and pour water into the auricle. Seen from
    lateral, the auricle (its outline with the holes filled in) must be solid ear everywhere
    except over the canal: a hole elsewhere lets the water through. Also, inside the outline
    (2 mm in), an open edge of the ear that lateral sees must be the canal's mouth (meeting the
    canal lining). Returns (hole area mm2 not over the canal, open-edge points that leak).
    """
    import open3d as o3d
    from scipy import ndimage

    ov, ot = (np.asarray(x) for x in out["ear"])
    side = cm.SideMap(vol.frame, ov)
    he = side.heights(ov, ot)
    foot = np.isfinite(he)
    filled = ndimage.binary_fill_holes(foot)
    # see-through holes, except where the ray runs down the canal
    leak_px = int(cm.holes_through(side, (ov, ot), eac).sum())
    # open edges that lateral sees, inside the outline, not at the canal
    e = np.sort(np.concatenate([ot[:, [0, 1]], ot[:, [1, 2]], ot[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    p = ov[np.unique(u[c == 1])]
    inner = ndimage.binary_erosion(filled, iterations=int(np.ceil(2.0 / side.pix)))
    _ab, h = side.coords(p)
    seen = np.abs(h - side.sample(np.nan_to_num(he, nan=-1e3), p)) < 0.5
    inside = side.sample(inner, p, order=0) > 0.5
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.asarray(out["canal"][0], np.float32)), o3d.core.Tensor(np.asarray(out["canal"][1], np.uint32)))
    canal = s.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy() < 0.6
    return leak_px * side.pix**2, p[seen & inside & ~canal]


def test_open_meatus_on_the_saved_ct_ear():
    import open3d as o3d

    s = _scene()
    if s is None:
        print("  (no saved CT-ear placement: skipped)")
        return
    vol, env, layers, ear, th = s
    eac = cm.EacField(vol)
    others = {k: m for k, m in layers.items() if k != "skin"}
    out = cm.open_meatus(vol, env, ear, layers["skin"], th, eac=eac, log=lambda *_: None, others=others)
    ev_, et = (np.asarray(x) for x in out["ear"])
    assert len(et) > 10000

    # the canal is open: the opened ear keeps clear of the EAC's inside
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(ev_.astype(np.float32)), o3d.core.Tensor(et.astype(np.uint32)))
    inner = np.argwhere(vol.eac)[::25] * vol.mm
    inner = inner[eac(inner) < -0.8]
    gap = scene.compute_distance(o3d.core.Tensor(inner.astype(np.float32))).numpy()
    assert gap.min() > 0.3, gap.min()

    # tissue in bone is gone: the cut line may graze bone voxels, but nothing lies deep in bone
    from scipy import ndimage

    idx = np.round(ev_ / vol.mm).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)]  # the ear reaches past the volume
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    solid = vol.solid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    depth = ndimage.distance_transform_edt(solid, sampling=vol.mm)
    d = depth[tuple((idx - lo).T)]
    assert np.mean(d > 0.6) < 0.01, np.mean(d > 0.6)  # a few cut-edge points in porous bone

    # the canal is lined from the meatus down toward the eardrum, facing the lumen
    cv, ct = (np.asarray(x) for x in out["canal"])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cv), o3d.utility.Vector3iVector(ct))
    n = np.asarray(m.compute_vertex_normals().vertex_normals)
    assert np.mean(eac(cv + 0.3 * n) < eac(cv)) > 0.95
    t, _rad = eac.axial(cv)
    assert t.min() < -15.0 and t.max() > -6.0, (t.min(), t.max())

    # no CT tissue round the back of the bone (only the cut's own edge may touch it)
    e = np.sort(np.concatenate([et[:, [0, 1]], et[:, [1, 2]], et[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    edge = np.zeros(len(ev_), bool)
    edge[np.unique(u[c == 1])] = True
    assert not np.any(cm.behind_bone(vol, ev_) & ~edge)

    # the auricle holds water: no hole in the concha or any other bowl, only the canal
    hole_mm2, edges = bowl_leaks(out, vol, eac)
    assert hole_mm2 < 1.0, hole_mm2
    assert len(edges) < 5, (len(edges), edges[:5])
    for name, mesh in out["layers"].items():  # deeper layers trimmed alongside the scalp skin
        assert len(mesh[1]) > 0, name

    assert out["framework"] is None  # no painted auricle: no framework from the trim

    # the scalp skin gives way where the ear rises out of it, and nowhere far from the ear
    sv0 = np.asarray(layers["skin"][0])
    sv = np.asarray(out["scalp"][0])
    assert len(sv) < len(sv0)
    far = scene.compute_distance(o3d.core.Tensor(sv0.astype(np.float32))).numpy() > 8.0
    kept = o3d.t.geometry.RaycastingScene()
    kept.add_triangles(o3d.core.Tensor(sv.astype(np.float32)), o3d.core.Tensor(np.asarray(out["scalp"][1], np.uint32)))
    assert far.sum() > 1000
    assert kept.compute_distance(o3d.core.Tensor(sv0[far].astype(np.float32))).numpy().max() < 0.5


if __name__ == "__main__":
    test_smooth_max_rounds_only_the_crease()
    print("test_smooth_max_rounds_only_the_crease ok")
    test_auricle_ramp_thins_near_the_canal_only()
    print("test_auricle_ramp_thins_near_the_canal_only ok")
    test_flatten_loop_reaches_its_depth_past_the_first_surface()
    print("test_flatten_loop_reaches_its_depth_past_the_first_surface ok")
    test_envelope_does_not_wrap_behind_the_bone()
    print("test_envelope_does_not_wrap_behind_the_bone ok")
    test_blender_bundle_round_trip()
    print("test_blender_bundle_round_trip ok")
    test_sheet_ear_holds_water_and_keeps_its_cartilage_inside()
    print("test_sheet_ear_holds_water_and_keeps_its_cartilage_inside ok")
    test_edited_cartilage_gets_back_skin_and_a_cleft()
    print("test_edited_cartilage_gets_back_skin_and_a_cleft ok")
    test_painted_auricle_gets_cartilage_and_the_rest_goes_under()
    print("test_painted_auricle_gets_cartilage_and_the_rest_goes_under ok")
    test_open_meatus_on_the_saved_ct_ear()
    print("test_open_meatus_on_the_saved_ct_ear ok")
