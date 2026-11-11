"""A CT-segmented right auricle (with some surrounding skin) as the ear for the envelope.

The segmentation came from a different CT than the RT143 volume, cut out with a box: its
lateral side is real skin (auricle, its back surface, the postauricular crease, skin around
the root); the medial and side faces are flat crop faces. It replaces ear.obj (a sculpted,
solid-backed left ear) as the ear's skin surface.

``clean`` keeps the main piece (the rest is segmentation specks), smooths the CT stair-steps
and thins the mesh. ``landmarks`` finds, in the CT's own millimetres:

  lateral  out of the head: across the flat medial crop face, pointing to -x (DICOM: +x is the
           patient's left, and this is a right ear)
  meatus   a first guess at where the canal opens into the concha: the deepest point seen
           from the lateral side
  (no canal: this segmentation has no canal lumen, so the canal's S-shaped course comes from
  the volume's EAC, built by ``ear_cuts.MeatusShape``)
  superior the scan's +z (DICOM patient frame: x to the left, y posterior, z superior)

``register`` places it on a FIVRS volume from two points: the CT meatus where the volume's
meatus belongs (``meatus_target``: lateral of the canal's end, swung down and forward), and
the deepest CT canal point on the line back to the volume's EAC opening; CT superior is kept
as close to the volume's superior as that allows. The canal is S-shaped, so these points
fix its chord, not a straight axis. With the CT's skin around the ear at or under the
scalp skin, the scalp join keeps the free auricle and its crease and drops the crop faces.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/ct_ear.py STL
Writes resources/soft_tissue/auricle/ct_right_ear.ply and its landmarks JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parents[1]
OUT = ROOT / "resources" / "soft_tissue" / "auricle"
NAME = "ct_right_ear"
TARGET_TRIS = 200_000


def clean(path, target_tris=TARGET_TRIS, smooth_iters=10):
    """Main piece of the STL, smoothed (Taubin, keeps volume) and thinned. Returns (verts, tris)."""
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    m.remove_duplicated_vertices()
    cl, n, _ = m.cluster_connected_triangles()
    cl, n = np.asarray(cl), np.asarray(n)
    m.remove_triangles_by_mask(cl != int(np.argmax(n)))
    m.remove_unreferenced_vertices()
    m = m.filter_smooth_taubin(number_of_iterations=smooth_iters)
    if len(m.triangles) > target_tris:
        m = m.simplify_quadric_decimation(target_tris)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    v, t = np.asarray(m.vertices, np.float64), np.asarray(m.triangles, np.int32)
    import auricle as au

    if au.signed_volume(v, t) < 0:
        t = t[:, [0, 2, 1]]
    return v, t


def _scene(v, t):
    import open3d as o3d

    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.asarray(v, np.float32)), o3d.core.Tensor(np.asarray(t, np.uint32)))
    return s


def _depth_map(scene, centre, look, a, b, half, step=0.3, back=60.0):
    import open3d as o3d

    A, B = np.meshgrid(np.arange(-half, half, step), np.arange(-half, half, step), indexing="ij")
    start = centre - look * back + A[..., None] * a + B[..., None] * b
    rays = np.concatenate([start.reshape(-1, 3), np.tile(look, (start.size // 3, 1))], axis=1).astype(np.float32)
    th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(A.shape)
    return start, th


def landmarks(v, t):
    """Lateral, superior, canal hole and meatus of the CT ear, in its own millimetres."""
    import open3d as o3d

    scene = _scene(v, t)
    c = v.mean(axis=0)
    # lateral: rays that start medial and travel lateral first meet the flat medial crop face.
    # Of the block's thinnest axis and its opposite, that is the direction whose first hits are
    # flattest; then the crop face's own plane refines it.
    _u, _s, vt = np.linalg.svd(v - c, full_matrices=False)
    half = np.ptp(v, axis=0).max() / 2 + 5
    best = None
    for look in (vt[2], -vt[2]):
        sup = np.array([0.0, 0.0, 1.0]) - look[2] * look
        sup /= np.linalg.norm(sup)
        start, th = _depth_map(scene, c, look, np.cross(sup, look), sup, half)
        d = th[np.isfinite(th)]
        hist, edges = np.histogram(d, bins=np.arange(d.min(), d.max() + 1.0, 1.0))
        mode = edges[np.argmax(hist)] + 0.5
        flat_share = np.mean(np.abs(d - mode) < 1.5)  # the crop face: most first hits at one depth
        if best is None or flat_share > best[0]:
            best = (flat_share, look, start, th)
    _share, look, start, th = best
    hit = np.isfinite(th)
    face_d = np.nanpercentile(np.where(hit, th, np.nan), 20)
    flat = hit & (np.abs(th - face_d) < 1.5)
    pts = (start + look[None, None] * np.where(hit, th, 0)[..., None])[flat]
    (a_, b_, c_, _d), _inl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).segment_plane(1.0, 3, 2000)
    lat = np.array([a_, b_, c_])
    # which way is out: DICOM patient space has x toward the patient's left, so a right ear's
    # lateral points toward -x (the flatness of first hits alone can't tell the crop face from
    # the broad skin around the ear)
    lat = lat if lat[0] < 0 else -lat
    # a first guess at the meatus: the deepest point of the ear seen from the lateral side, away
    # from the outline (the concha floor). The canal is taken to run straight medial from there.
    # Clicking the meatus and a deeper canal point in the viewer replaces both.
    sup = np.array([0.0, 0.0, 1.0]) - lat[2] * lat
    sup /= np.linalg.norm(sup)
    ant = np.cross(sup, lat)
    start, th = _depth_map(scene, c, -lat, ant, sup, half)
    hit = np.isfinite(th)
    inner = ndimage.binary_erosion(hit, iterations=int(4.0 / 0.3))
    smooth = ndimage.gaussian_filter(np.where(hit, th, 0.0), 2.0) / np.maximum(ndimage.gaussian_filter(hit.astype(float), 2.0), 1e-6)
    smooth[~inner] = -np.inf
    i, j = np.unravel_index(np.argmax(smooth), smooth.shape)
    meatus = start[i, j] - lat * th[i, j]
    return dict(lateral=lat, superior=sup, anterior=ant, meatus=meatus, center=c)


def meatus_target(vol, lateral_mm=14.0, inferior_mm=3.0, anterior_mm=2.0):
    """Where the meatus belongs on a volume: lateral of the canal's end, swung down and forward.

    The canal is S-shaped: going out from the EAC's lateral opening it swings down and anterior
    to the meatus (the same swing ``ear_cuts.MeatusShape`` uses).
    """
    f = vol.frame
    return f.canal_lateral_end + f.lateral * lateral_mm + f.anterior * anterior_mm - f.superior * inferior_mm


def register(vol, marks, target):
    """4x4 taking CT mm to volume mm, from two points and superior. Returns (T, info).

    marks["meatus"] (the CT meatus) goes to ``target`` (``meatus_target``); marks["hole"] (the
    deepest point of the CT's canal, where its cartilaginous canal ends) lies on the line from
    there to the volume's EAC opening. The canal between them is S-shaped, so this matches the
    chord, not a canal axis. CT superior is turned as close to the volume's superior as the
    chord allows. No scaling: the chord lengths are reported.
    """
    f = vol.frame
    m = np.asarray(marks["meatus"], float)

    def frame(axis, up):
        axis = axis / np.linalg.norm(axis)
        up = up - (up @ axis) * axis
        up /= np.linalg.norm(up)
        return np.stack([axis, up, np.cross(axis, up)], axis=1)
    chord_v = np.asarray(target, float) - f.canal_lateral_end
    if marks.get("hole") is not None:  # a clicked deep canal point: match the S-curve's chord
        chord_ct = m - np.asarray(marks["hole"], float)
        R = frame(chord_v / np.linalg.norm(chord_v), f.superior) @ frame(chord_ct / np.linalg.norm(chord_ct), np.asarray(marks["superior"], float)).T
        info = dict(ct_chord_mm=float(np.linalg.norm(chord_ct)), volume_chord_mm=float(np.linalg.norm(chord_v)), by="canal chord")
    else:  # no canal in the CT: its lateral and superior onto the volume's
        R = frame(f.lateral, f.superior) @ frame(np.asarray(marks["lateral"], float), np.asarray(marks["superior"], float)).T
        info = dict(by="lateral and superior")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(target, float) - R @ m
    return T, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stl")
    args = ap.parse_args()
    import envelope as ev

    v, t = clean(args.stl)
    marks = landmarks(v, t)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.write_ply(OUT / f"{NAME}.ply", v, t, (235, 150, 120))
    doc = {k: np.asarray(x).tolist() for k, x in marks.items()}
    doc["source"] = str(args.stl)
    (OUT / f"{NAME}.landmarks.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{len(t)} tris; meatus guess {np.round(marks['meatus'], 1)}, lateral {np.round(marks['lateral'], 2)}")


if __name__ == "__main__":
    main()
