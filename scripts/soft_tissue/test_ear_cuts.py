"""Hand-drawn mesh cuts: a closed mesh with the region inside the outline removed.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_ear_cuts.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ear_cuts as ec  # noqa: E402


def _slab():
    import trimesh

    m = trimesh.creation.box(extents=(20.0, 20.0, 4.0))  # a 4 mm plate, faces at z = +-2
    return np.asarray(m.vertices), np.asarray(m.faces)


def _square(c, half, z):
    return [[c[0] - half, c[1] - half, z], [c[0] + half, c[1] - half, z], [c[0] + half, c[1] + half, z], [c[0] - half, c[1] + half, z]]


def _inside(v, t, pts):
    import trimesh

    return trimesh.Trimesh(v, t, process=False).contains(pts)


def test_through_cut_makes_a_lined_hole():
    import trimesh

    v, t = _slab()
    cut = ec.MeshCut("skin", _square((0, 0), 3.0, 2.0), [0, 0, -1.0], 0.0)
    nv, nt = ec.apply_cuts(v, t, [cut])
    m = trimesh.Trimesh(nv, nt, process=False)
    assert m.is_watertight
    assert m.euler_number == 0  # one hole through: a torus-like solid
    probe = np.array([[0, 0, 0.0], [0, 0, 1.5], [6, 6, 0.0]])
    assert list(_inside(nv, nt, probe)) == [False, False, True]


def test_depth_cut_stops():
    v, t = _slab()
    cut = ec.MeshCut("cartilage", _square((0, 0), 3.0, 2.0), [0, 0, -1.0], 2.0)  # 2 mm into the 4 mm plate
    nv, nt = ec.apply_cuts(v, t, [cut])
    probe = np.array([[0, 0, 1.0], [0, 0, -1.2], [6, 6, 1.0]])
    assert list(_inside(nv, nt, probe)) == [False, True, True]


def test_no_cuts_is_unchanged():
    v, t = _slab()
    nv, nt = ec.apply_cuts(v, t, [])
    assert nv.shape == v.shape and np.array_equal(nt, t)


def test_join_to_scalp_meets_at_one_line():
    """A plate (the ear) pushed halfway through a sheet (the scalp skin): both are cut where they cross."""
    import types

    from scipy.spatial import cKDTree

    import trimesh

    sheet = trimesh.creation.box(extents=(40.0, 40.0, 0.01))  # thin box: its top face is the skin
    sv, st = trimesh.remesh.subdivide_to_size(sheet.vertices, sheet.faces, 0.5)
    top = st[(sv[st][:, :, 2] > 0).all(axis=1)]
    ear = trimesh.creation.box(extents=(10.0, 10.0, 10.0))  # straddles the skin plane z = 0.005
    ear = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(ear.vertices, ear.faces, 0.5))
    vol = types.SimpleNamespace(frame=types.SimpleNamespace(lateral=np.array([0, 0, 1.0])),
                                mm=np.ones(3), shape=(1, 1, 1), solid=np.zeros((1, 1, 1), bool))
    deep = ec.deep_to_skin((sv, top), vol)
    (ev_, et), (cv, ct) = ec.join_to_scalp((np.asarray(ear.vertices), np.asarray(ear.faces)), (sv, top), deep)
    assert ev_[:, 2].min() > -0.01  # nothing of the ear left under the skin
    assert not ((np.abs(cv[:, 0]) < 4.9) & (np.abs(cv[:, 1]) < 4.9)).any()  # skin under the ear removed

    def edge(v, t):
        e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        u, c = np.unique(e, axis=0, return_counts=True)
        return v[np.unique(u[c == 1])]

    ee = edge(ev_, et)
    ce = edge(cv, ct)
    inner = ce[(np.abs(ce[:, 0]) < 6) & (np.abs(ce[:, 1]) < 6)]  # the hole's rim, not the sheet's border
    assert np.median(cKDTree(inner).query(ee)[0]) < 0.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
