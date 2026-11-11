"""Cartilage from a skin sheet: cover under the skin, thickness, paint, cleaning.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_cartilage.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import auricle as au  # noqa: E402
import cartilage as ca  # noqa: E402


def _open_box(size=(20.0, 20.0, 6.0), res=60):
    """A box's surface without its bottom: one skin on top, normals out, open at the base."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh.create_box(*size)
    m = m.subdivide_midpoint(number_of_iterations=5)
    v = np.asarray(m.vertices)
    t = np.asarray(m.triangles)
    keep = ~(np.abs(v[t][:, :, 2]) < 1e-9).all(axis=1)  # drop the bottom face
    t = t[keep]
    used = np.unique(t)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return v[used], remap[t]


def test_single_skin_gives_a_plate_under_it():
    v, t = _open_box()
    w = np.ones(len(v))
    fv, ft = ca.framework(v, t, w, max_mm=1.5, log=lambda *_: None)
    assert len(ft) > 100
    mid = (np.abs(fv[:, 0] - 10) < 5) & (np.abs(fv[:, 1] - 10) < 5)
    z = fv[mid, 2]
    # under the top skin (z = 6): from the cover down to cover + 1.5 mm
    assert abs(z.max() - (6 - au.COVER_MM)) < 0.3, z.max()
    assert abs(z.min() - (6 - au.COVER_MM - 1.5)) < 0.3, z.min()


def test_cartilage_only_where_painted():
    v, t = _open_box()
    w = (v[:, 0] < 10).astype(float)  # half painted
    fv, _ft = ca.framework(v, t, w, log=lambda *_: None)
    assert np.mean(fv[:, 0] < 10.5) > 0.97


def test_clean_sheet_drops_islands_spikes_and_pinholes():
    v, t = _open_box()
    w = np.ones(len(v))
    # an island, a spike, a pinhole
    island = np.array([[50.0, 50, 50], [51, 50, 50], [50, 51, 50]])
    vi = len(v)
    v = np.concatenate([v, island])
    t = np.concatenate([t, [[vi, vi + 1, vi + 2]]])
    w = np.concatenate([w, np.ones(3)])
    top = np.nonzero(np.abs(v[:, 2] - 6) < 1e-9)[0]
    spike = top[np.argmin(np.linalg.norm(v[top, :2] - [5, 5], axis=1))]
    v[spike, 2] += 3.0
    hole_v = top[np.argmin(np.linalg.norm(v[top, :2] - [15, 15], axis=1))]
    t = t[~(t == hole_v).any(axis=1)]
    loops0 = ca.boundary_loops(t)
    v2, t2, w2 = ca.clean_sheet(v, t, w, log=lambda *_: None)
    assert len(v2) < len(v) + 2  # the island is gone (a pinhole centre may be added)
    assert np.abs(v2[:, 2]).max() < 6.5  # the spike is pulled in
    assert len(ca.boundary_loops(t2)) == len(loops0) - 1 - 1  # the island's and the pinhole's loops are gone


def test_medial_skin_covers_the_bare_face_down_to_the_envelope():
    """A cartilage plate with skin only on its top: the medial skin covers its underside and
    ends where it meets the envelope's skin (a plane under part of it)."""
    import open3d as o3d

    import sheet_skin as ss

    plate = o3d.geometry.TriangleMesh.create_box(20.0, 10.0, 1.5).subdivide_midpoint(3)
    fv, ft = np.asarray(plate.vertices), np.asarray(plate.triangles)
    top = o3d.geometry.TriangleMesh.create_box(20.0, 10.0, 0.01).subdivide_midpoint(4)
    sv = np.asarray(top.vertices) + [0, 0, 1.5 + au.COVER_MM]
    st = np.asarray(top.triangles)
    st = st[(sv[st][:, :, 2] > 1.5 + au.COVER_MM + 0.005).all(axis=1)]  # its upper face: skin over the top only
    # the envelope: a plane z = -2 for x < 10, sloping up to z = +3 at x = 20 (the plate goes into it there)
    gx, gy = np.meshgrid(np.linspace(-10, 30, 81), np.linspace(-10, 20, 61), indexing="ij")
    gz = np.where(gx < 10, -2.0, -2.0 + (gx - 10) * 0.5)
    ev_ = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    i = np.arange(81 * 61).reshape(81, 61)
    q = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1, 4)
    et = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    mv, mt = ss.medial_skin(fv, ft, sv, st, (ev_, et), (0, 0, 1), log=lambda *_: None)
    assert len(mt) > 50
    under = mv[:, 2] < 0.5
    assert under.mean() > 0.6  # mostly on the underside
    z_env = np.where(mv[:, 0] < 10, -2.0, -2.0 + (mv[:, 0] - 10) * 0.5)
    assert (mv[:, 2] > z_env - 0.3).all()  # never under the envelope's skin
    assert mv[under, 0].max() < 15.0  # it ends where the plate goes into the envelope (x = 14 + cover)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
