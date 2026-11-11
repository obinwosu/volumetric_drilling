"""The auricle as a simulation model, built from the finished composite (``build_ear.py``).

The composite's meshes are for drawing (0.35 mm, the cartilage a thin solid). The model needs the
cartilage as one sheet, its mid-surface, at about 1.5 mm:

``mid_surface``
  1. the Blender-edited cartilage (one piece, before the composite clipped it), in volume mm;
  2. its thickness: from each triangle, a ray inward to the opposite face (1.5 mm typically);
  3. the mid-point of each such pair, oriented consistently over the whole plate (a spanning tree
     over the cloud: the plate curls, so no fixed direction works);
  4. a surface through that cloud (Poisson, trimmed to the cloud), the main piece kept, small holes
     closed, smoothed, remeshed to ``EDGE_MM``, and made manifold (every edge on at most two
     triangles, so the shell's hinges are well defined);
  5. normals turned to face laterally; each vertex keeps the local cartilage thickness.

Cached as PLY + NPZ next to the composite.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/auricle_sim.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import envelope as ev  # noqa: E402

EDGE_MM = 1.5
COMPOSITE = ev.ROOT / "resources" / "soft_tissue" / "composite" / "volume_256_RT143_ct_right_ear"
ADF = ev.ROOT / "ADF" / "volume_256_RT143.yaml"
EAR = ev.ROOT / "resources" / "soft_tissue" / "auricle" / "ct_right_ear.ply"
HOLE_MM = 5.0  # pinholes with a rim shorter than this are closed (larger fans would spike)


def edited_cartilage(vol, adf=ADF, ear=EAR):
    """The Blender-edited cartilage placed in volume mm (largest piece), and the placement matrix."""
    import trimesh

    import blender_edit as be
    import placement_viewer as pv

    saved = ev.ROOT / "resources" / "soft_tissue" / "envelope" / f"{pathlib.Path(adf).stem}_{pathlib.Path(ear).stem}_placement.json"
    doc = json.loads(saved.read_text())
    place = pv.Placement(vol.frame, np.zeros(3))
    place.load(doc["ear"], doc.get("frame"))
    T = place.matrix()
    cv, ct = be.read_ply(pathlib.Path(ear).with_name(pathlib.Path(ear).stem + "_cartilage_edited.ply"))[:2]
    cv = np.asarray(cv, float) @ T[:3, :3].T + T[:3, 3]
    ct = np.asarray(ct)
    if np.linalg.det(T[:3, :3]) < 0:
        ct = ct[:, [0, 2, 1]]
    m = trimesh.Trimesh(cv, ct, process=True)
    return max(m.split(only_watertight=False), key=lambda p: p.area), T


def _thickness(mm, max_mm=3.0, chunk=4000):
    """Per triangle: distance to the opposite face along the inward normal, and which way is outward."""
    import trimesh

    c, n = mm.triangles_center, mm.face_normals
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(mm)
    best = np.full(len(c), np.inf)
    side = np.zeros(len(c))
    for sgn in (1.0, -1.0):
        for s0 in range(0, len(c), chunk):
            idx = np.arange(s0, min(s0 + chunk, len(c)))
            dirs = -sgn * n[idx]
            loc, ri, ti = ray.intersects_location(c[idx] + dirs * 1e-3, dirs, multiple_hits=False)
            r = idx[ri]
            dist = np.linalg.norm(loc - c[r], axis=1)
            ok = (dist > 0.05) & (dist < max_mm) & ((n[ti] * n[r]).sum(1) < -0.5) & (dist < best[r])
            best[r[ok]] = dist[ok]
            side[r[ok]] = sgn
    return best, side


def split_bowties(v, t):
    """Give every vertex one copy per fan of triangles around it (two holes touching at a vertex
    become two simple rings). Copies sit at the same point."""
    from collections import defaultdict

    v, t = np.asarray(v, float), np.asarray(t).copy()
    around = defaultdict(list)
    for f, tri in enumerate(t):
        for c in range(3):
            around[int(tri[c])].append((f, c))
    new_v = [v]
    k = len(v)
    for vi, inc in around.items():
        if len(inc) < 2:
            continue
        # fans: triangles at vi joined when they share an edge through vi
        parent = list(range(len(inc)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        edge = {}
        for idx, (f, c) in enumerate(inc):
            for other in (t[f, (c + 1) % 3], t[f, (c + 2) % 3]):
                key = int(other)
                if key in edge:
                    parent[find(idx)] = find(edge[key])
                else:
                    edge[key] = idx
        roots = sorted({find(i) for i in range(len(inc))})
        for r in roots[1:]:
            new_v.append(v[vi][None])
            for idx, (f, c) in enumerate(inc):
                if find(idx) == r:
                    t[f, c] = k
            k += 1
    return np.concatenate(new_v), t


def _close_small_holes(v, t, max_rim_mm):
    """Fan-fill boundary loops whose rim is shorter than ``max_rim_mm`` (keeps the outer outline)."""
    import cartilage as ca

    v, t = np.asarray(v, float), np.asarray(t)
    new_v, new_t = [v], [t]
    k = len(v)
    for loop in ca.boundary_loops(t):
        order = ca._ordered_loop(t, loop)
        if order is None or len(order) < 3:
            continue
        p = v[order]
        rim = float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())
        if rim > max_rim_mm:
            continue
        new_v.append(p.mean(axis=0)[None])
        ring = np.array(order)
        new_t.append(np.stack([ring, np.roll(ring, -1), np.full(len(ring), k)], axis=1))
        k += 1
    return np.concatenate(new_v), np.concatenate(new_t)


def _despike(tm, factor=1.5, rounds=5, edge_mm=EDGE_MM):
    """Pull vertices that stick out far from their neighbours' average back onto it (in place)."""
    import cartilage as ca

    for _ in range(rounds):
        v = np.asarray(tm.vertices, float)
        nb, _deg = ca._neighbour_mean(v, np.asarray(tm.faces))
        off = np.linalg.norm(v - nb, axis=1)
        bad = off > factor * edge_mm * 0.5
        if not bad.any():
            break
        v[bad] = nb[bad]
        tm.vertices = v


FOLD_DEG = 100.0  # neighbouring triangles folded sharper than this are a remeshing flaw, not anatomy
NEEDLE_DEG = 12.0  # a corner sharper than this: a needle triangle (stiff, ill-conditioned element)


def _unfold(v, t, max_deg=FOLD_DEG, min_angle=NEEDLE_DEG, rounds=40):
    """Collapse the sliver triangles the simplifier leaves: folded back over a neighbour, or needles.

    Each round: for every neighbouring pair folded sharper than ``max_deg`` the smaller triangle, and
    every triangle with a corner under ``min_angle``, has its shortest edge collapsed to its midpoint
    (one collapse per vertex per round); triangles that lose a corner or would double an edge are
    dropped. Returns (v, t, flawed triangles left)."""
    import trimesh

    v, t = np.asarray(v, float).copy(), np.asarray(t, np.int64).copy()
    left = 0
    for _ in range(rounds):
        tm = trimesh.Trimesh(v, t, process=False)
        fa = tm.face_adjacency[np.degrees(tm.face_adjacency_angles) > max_deg]
        area = tm.area_faces
        needles = np.flatnonzero(np.degrees(tm.face_angles.min(axis=1)) < min_angle)
        small = np.unique(np.concatenate([np.where(area[fa[:, 0]] <= area[fa[:, 1]], fa[:, 0], fa[:, 1]), needles]))
        left = len(small)
        if not left:
            break
        small = small[np.argsort(area[small])]  # worst first
        touched = np.zeros(len(v), bool)
        for f in small:
            c = t[f]
            if touched[c].any():
                continue
            e = [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]
            a, b = min(e, key=lambda ab: np.linalg.norm(v[ab[0]] - v[ab[1]]))
            v[b] = 0.5 * (v[a] + v[b])
            t[t == a] = b
            touched[c] = True
        t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])]
        v, t, _ = _manifold(v, t)
    return v, t, left


def _largest_piece(v, t):
    """The connected piece (by shared vertices) with the most triangles, unused vertices dropped."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n = len(v)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]]])
    _k, lab = connected_components(sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    best = np.argmax(np.bincount(lab[t[:, 0]]))
    t = t[lab[t[:, 0]] == best]
    used = np.unique(t)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(v)[used], remap[t]


def _manifold(v, t):
    """Drop triangles on edges shared by more than two, then the vertices no triangle uses."""
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    key = e[:, 0] * (len(v) + 1) + e[:, 1]
    _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bad = (cnt[inv] > 2).reshape(3, -1).any(axis=0)
    t = t[~bad]
    used = np.unique(t)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[t], int(bad.sum())


def mid_surface(vol, edge_mm=EDGE_MM, log=print, lobule=False):
    """(vertices, triangles, thickness per vertex, cartilage under each vertex): the cartilage
    mid-surface, normals lateral; with ``lobule`` it carries on into the lobule (``_lobule_mid``),
    where there is no cartilage (flag False, thickness = the lobule's)."""
    import fast_simplification as fsimp
    import trimesh
    from scipy.spatial import cKDTree

    t0 = time.time()
    key = (id(vol), lobule)
    if key not in _DENSE:  # the costly part (thickness rays, Poisson) is the same at every edge size
        _DENSE[key] = _dense_mid(vol, log, t0, lobule=lobule)
    tm, mid, thick, is_cart = _DENSE[key]
    tm = tm.copy()
    target = int(tm.area / (0.433 * edge_mm**2))
    v2, f2 = fsimp.simplify(tm.vertices.astype(np.float32), tm.faces.astype(np.int32), target_count=target)
    tm = trimesh.Trimesh(v2, f2, process=True)
    tm = max(tm.split(only_watertight=False), key=lambda p: p.area)
    trimesh.smoothing.filter_taubin(tm, iterations=10)
    _despike(tm, edge_mm=edge_mm)
    v3, t3, dropped = _manifold(np.asarray(tm.vertices, float), np.asarray(tm.faces))
    v3, t3, folds = _unfold(v3, t3)
    tm = trimesh.Trimesh(v3, t3, process=True)
    tm = max(tm.split(only_watertight=False), key=lambda p: p.area)
    tm.fix_normals(multibody=False)
    L = np.asarray(vol.frame.lateral, float)
    if (tm.face_normals @ L * tm.area_faces).sum() < 0:  # face laterally
        tm.invert()
    v4, t4, folds = _unfold(tm.vertices, tm.faces)  # again: the clean-up above can fold a few back
    tm = trimesh.Trimesh(*_largest_piece(v4, t4), process=False)  # no vertex merging: it re-joins folds
    log(f"mid-surface: {folds} folded or needle triangles left (folds > {FOLD_DEG:g} deg, corners < {NEEDLE_DEG:g} deg)")
    # cartilage thickness at each vertex: nearest measured mid-point
    _d, near = cKDTree(mid).query(tm.vertices, k=8)
    th = np.median(thick[near], axis=1)
    cart = is_cart[near].mean(axis=1) >= 0.5
    log(f"mid-surface: {len(tm.vertices)} vertices, {len(tm.faces)} triangles, edge {tm.edges_unique_length.mean():.2f} mm, "
        f"area {tm.area:.0f} mm^2, euler {tm.euler_number}, {dropped} non-manifold triangles dropped ({time.time() - t0:.0f} s)")
    if lobule:
        log(f"mid-surface: {int((~cart).sum())} vertices in the lobule (no cartilage)")
    return np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int64), th, cart


_DENSE = {}
LOBULE_GRID_MM = 0.5


def _lobule_mid(vol, cart_mid, log):
    """Mid-points and thickness of the lobule (soft tissue with no cartilage), for the mid-surface.

    The edited skin has inner walls in the lobule, so pairing each face with the nearest opposite
    face fails there. Instead rays cross the lobule along its face normal (PCA of the skin below
    the cartilage, away from it) on a LOBULE_GRID_MM grid; the first and last hit are its two faces.
    Points within 2 mm of cartilage are left to the cartilage."""
    import trimesh
    from scipy.spatial import cKDTree

    skin = max(_load("auricle_skin").split(only_watertight=False), key=lambda p: p.area)
    S = np.asarray(vol.frame.superior, float)
    v = np.asarray(skin.vertices, float)
    top = (cart_mid @ S).min() + 1.0  # the lobule hangs below the lowest cartilage
    low = (cKDTree(cart_mid).query(v)[0] > 3.0) & (v @ S < top)
    pts = v[low]
    c = pts.mean(axis=0)
    _u, _s, W = np.linalg.svd(pts - c, full_matrices=False)
    a, b, n = W
    uv = np.stack([(pts - c) @ a, (pts - c) @ b], axis=1)
    g = np.mgrid[uv[:, 0].min():uv[:, 0].max():LOBULE_GRID_MM, uv[:, 1].min():uv[:, 1].max():LOBULE_GRID_MM].reshape(2, -1).T
    g = g[cKDTree(uv).query(g)[0] < 1.0]  # over the lobule's outline only
    far = 40.0
    origins = c + g[:, :1] * a + g[:, 1:] * b - far * n
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(skin)
    loc, ri, _ti = ray.intersects_location(origins, np.repeat(n[None], len(g), axis=0), multiple_hits=True)
    depth = ((loc - origins[ri]) @ n)
    first = np.full(len(g), np.inf)
    last = np.full(len(g), -np.inf)
    np.minimum.at(first, ri, depth)
    np.maximum.at(last, ri, depth)
    th = last - first
    ok = np.isfinite(th) & (th > 1.5) & (th < 14.0)
    mid = origins[ok] + (0.5 * (first[ok] + last[ok]))[:, None] * n
    th = th[ok]
    dc, jc = cKDTree(cart_mid).query(mid)
    keep = (dc > 2.0) & (mid @ S < top)
    mid, th, dc, jc = mid[keep], th[keep], dc[keep], jc[keep]
    # bridge to the cartilage: points every 0.4 mm on the way from the lobule's upper edge to the
    # nearest cartilage mid-point, so the fitted surface stays one piece
    near = np.flatnonzero(dc < 5.0)
    br, bt = [], []
    for i in near:
        k = int(dc[i] / 0.4)
        f = np.linspace(0, 1, k + 2)[1:-1, None]
        br.append(mid[i] + f * (cart_mid[jc[i]] - mid[i]))
        bt.append(np.full(k, th[i]))
    log(f"lobule: {len(mid)} mid-points, thickness median {np.median(th):.1f} mm; {sum(map(len, br))} bridging points")
    if br:
        mid, th = np.concatenate([mid] + br), np.concatenate([th] + bt)
    return mid, th


def _dense_mid(vol, log, t0, lobule=False):
    """The closed, smoothed mid-surface before remeshing, and the measured mid-points and thickness."""
    import fast_simplification as fsimp
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree

    cart, _T = edited_cartilage(vol)
    v, f = fsimp.simplify(cart.vertices.astype(np.float32), cart.faces.astype(np.int32), target_count=35000)
    mm = trimesh.Trimesh(v, f, process=True)
    best, side = _thickness(mm)
    good = np.isfinite(best)
    n = mm.face_normals * np.where(side == 0, 1.0, side)[:, None]  # outward where known
    mid = mm.triangles_center[good] - 0.5 * best[good, None] * n[good]
    thick = best[good]
    log(f"cartilage: thickness on {good.mean():.0%} of it, median {np.median(thick):.2f} mm ({time.time() - t0:.0f} s)")
    is_cart = np.ones(len(mid), bool)
    if lobule:
        lm, lt = _lobule_mid(vol, mid, log)
        mid, thick = np.concatenate([mid, lm]), np.concatenate([thick, lt])
        is_cart = np.concatenate([is_cart, np.zeros(len(lm), bool)])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mid))
    pc = pc.voxel_down_sample(0.5)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(15)
    pts = np.asarray(pc.points)
    surf, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, scale=1.1)
    dist, _ = cKDTree(pts).query(np.asarray(surf.vertices))
    surf.remove_vertices_by_mask(dist > 0.8)
    tm = trimesh.Trimesh(np.asarray(surf.vertices), np.asarray(surf.triangles), process=True)
    tm = max(tm.split(only_watertight=False), key=lambda p: p.area)
    sv, st = _close_small_holes(tm.vertices, tm.faces, HOLE_MM)
    tm = trimesh.Trimesh(sv, st, process=True)
    trimesh.smoothing.filter_taubin(tm, iterations=20)
    return tm, mid, thick, is_cart


def _edge_tag(edge_mm):
    """File suffix for a mid-surface remeshed to another edge size (none for the default)."""
    return "" if abs(edge_mm - EDGE_MM) < 1e-9 else f"_e{edge_mm:g}mm".replace(".", "p")


def cached_mid_surface(vol, rebuild=False, log=print, edge_mm=EDGE_MM, lobule=False, with_cart=False):
    """The mid-surface, cached: (v, t, thickness), and the cartilage flag per vertex with
    ``with_cart``. ``lobule``: the one that carries on into the lobule (sim_auricle_mid*), else the
    cartilage only (sim_cartilage_mid*, the first version)."""
    tag = _edge_tag(edge_mm)
    stem = "sim_auricle_mid" if lobule else "sim_cartilage_mid"
    path = COMPOSITE / f"{stem}{tag}.npz"
    if path.exists() and not rebuild:
        z = np.load(path)
        out = z["v"], z["t"], z["thickness"], (z["cart"] if "cart" in z else np.ones(len(z["v"]), bool))
    else:
        out = mid_surface(vol, edge_mm=edge_mm, log=log, lobule=lobule)
        np.savez(path, v=out[0], t=out[1], thickness=out[2], cart=out[3])
        ev.write_ply(COMPOSITE / f"{stem}{tag}.ply", out[0], out[1], (90, 150, 255))
    return out if with_cart else out[:3]


if __name__ == "__main__":
    vol = ev.Volume(ADF)
    cached_mid_surface(vol, rebuild="--rebuild" in sys.argv)


# -- the model: layers, base, fans --------------------------------------------------------------
BASE_MM = 2.0  # cartilage within this of the head's skin is held: the auricle's base (for now)
RING_MM = (2.0, 7.0)  # where the fans insert: just off the base
# The superior fan turned in the plane of the head's skin about the ear canal's axis (user,
# 2026-11-11: ~15 deg counterclockwise seen from the side, so its top leans back).
SUPERIOR_TURN_DEG = 15.0
SKIN_MM = 0.8  # auricular skin (Danter 1996), both sides
# The base sits on soft tissue (areolar / subcutaneous, a few mm over fascia and the canal): tied to
# rest by its shear stiffness per area, G / depth ~ (0.03 MPa / 3) / 2 mm. An estimate, the first
# knob for clinical tuning (how far the whole ear swings for a given pull).
BASE_K_N_PER_MM3 = 0.005


def _project(mesh, pts):
    import trimesh

    return trimesh.proximity.ProximityQuery(mesh).on_surface(np.asarray(pts, float))[0]


def _load(name):
    import trimesh

    return trimesh.load(COMPOSITE / f"{name}.ply", process=False)


FASCIA_MM = 3.0  # the muscles' origin lies this far under the head's skin (skin + fat over the fascia)


def head_depth(vol, pts):
    """Points moved (along lateral) to fascia depth under the head's skin.

    The skin's lateral position at each point's (anterior, superior) place, from the envelope skin
    (median of its nearest vertices in that plane). The scan stops ~33 mm above the ear, so beyond
    it the temporal scalp is taken to carry on straight up at the depth of its nearest scanned part
    (an assumption)."""
    from scipy.spatial import cKDTree

    L, A, S = (np.asarray(x, float) for x in (vol.frame.lateral, vol.frame.anterior, vol.frame.superior))
    env = _load("envelope_skin").vertices
    plane = np.stack([env @ A, env @ S], axis=1)
    pts = np.asarray(pts, float)
    q = np.stack([pts @ A, pts @ S], axis=1)
    _d, near = cKDTree(plane).query(q, k=12)
    level = np.median((env @ L)[near], axis=1) - FASCIA_MM
    return pts + (level - pts @ L)[:, None] * L


def base_mask(v, t, d_head, base_mm=None, rings=2, min_size=8):
    """Held base: cartilage near the head's skin, with small gaps closed and small islands dropped."""
    import cartilage as ca
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    m = d_head < (base_mm or BASE_MM)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    adj = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(v),) * 2).tocsr()
    grow = lambda x: x | (adj @ x.astype(float) > 0)
    shrink = lambda x: x & ~((adj @ (~x).astype(float)) > 0)
    for _ in range(rings):
        m = grow(m)
    for _ in range(rings):
        m = shrink(m)
    sub = adj[m][:, m]
    k, lab = connected_components(sub, directed=False)
    size = np.bincount(lab, minlength=k)
    keep = np.zeros(len(v), bool)
    keep[np.flatnonzero(m)[size[lab] >= min_size]] = True
    return keep


def place_fans(vol, v, base, d_head, log=print):
    """Fibres of every fan in tissue_params.FANS on the mid-surface (vertex ids into v).

    Each inserts on a patch (its insertion width) round the fan ring's extreme point in its
    direction; muscles start on a line (their origin width) at their length along that direction, at
    fascia depth; ligaments start on bone. Directions come from the volume's anatomical frame
    (anterior = toward this ear's face), which mirrors the left-ear reference figure."""
    import fibres as fb
    import tissue_params as tp

    L, A, S = (np.asarray(x, float) for x in (vol.frame.lateral, vol.frame.anterior, vol.frame.superior))
    ring = (~base) & (d_head >= RING_MM[0]) & (d_head <= RING_MM[1])
    ids = np.flatnonzero(ring)
    c = v[ring].mean(axis=0)
    a, s = (v - c) @ A, (v - c) @ S
    smed = np.median(s[ring])
    score = {"superior": s, "anterior": a + 0.5 * s, "anterior_ligament": a - 0.5 * s,
             "posterior": -a - 0.3 * np.abs(s - smed), "posterior_ligament": -a - 0.3 * np.abs(s - smed) - 0.5 * d_head}
    # counterclockwise seen from the side (looking medially): +angle about the lateral (canal) axis
    import placement_viewer as pv

    Rs = pv.rot(L, SUPERIOR_TURN_DEG)
    S_up, A_up = Rs @ S, Rs @ A
    score["superior"] = (v - c) @ S_up
    direction = {"superior": (S_up, A_up), "anterior": (A, S), "posterior": (-A, S),
                 "posterior_ligament": (-A - L, S), "anterior_ligament": (A - L, S)}
    bone = None
    sets = []
    for name, spec in tp.FANS.items():
        target = ids[np.argmax(score[name][ids])]
        r = max(0.5 * spec.insertion_width_mm.nominal, 2.0)
        ins = ids[np.linalg.norm(v[ids] - v[target], axis=1) <= r]
        d, perp = direction[name]
        d = d / np.linalg.norm(d)
        perp = perp - (perp @ d) * d
        perp /= np.linalg.norm(perp)
        p = v[ins].mean(axis=0)
        mid = p + spec.length_mm.nominal * d
        half = 0.5 * spec.origin_width_mm.nominal
        ends = np.array([mid - half * perp, mid + half * perp])
        if spec.material == "auricular_ligament":
            bone = bone if bone is not None else _load("bone")
            oa, ob = _project(bone, ends)
        else:
            oa, ob = head_depth(vol, ends)
        law = tp.fan_law(name)
        sets.append(fb.fan(ins, v, oa, ob, spec.thickness_mm.nominal, per_vertex=2, name=name, **law))
        log(f"fan {name}: {len(ins)} insertion vertices, origin {np.linalg.norm(ob - oa):.0f} mm wide, "
            f"fibres {sets[-1].rest_len.min():.0f}-{sets[-1].rest_len.max():.0f} mm")
    return sets


CARTILAGE_MM = 1.6  # the measured median thickness of the edited cartilage
LAYERS = ("lateral skin", "cartilage", "medial skin")


def head_distance(v, cache=True, edge_mm=EDGE_MM, lobule=False):
    """Distance from each mid-surface vertex to the head's skin (envelope), cached."""
    import trimesh

    path = COMPOSITE / f"sim_head_distance{'_lobule' if lobule else ''}{_edge_tag(edge_mm)}.npy"
    if cache and path.exists():
        d = np.load(path)
        if len(d) == len(v):
            return d
    _p, d, _t = trimesh.proximity.ProximityQuery(_load("envelope_skin")).on_surface(v)
    np.save(path, d)
    return d


class AuricleModel:
    """The auricle on its mid-surface: lateral skin / cartilage / medial skin bonded face to face,
    the base held, the auricular muscles and ligaments as fans on the cartilage."""

    def __init__(self, vol, base_k=BASE_K_N_PER_MM3, log=print, edge_mm=EDGE_MM, layers=3, lobule=False):
        """``edge_mm``: mid-surface edge size. ``layers``: 3 (skin / cartilage / skin bonded face to
        face) or 1 (one shell with the bonded laminate's stretch and bending stiffness, see
        ``laminate_shell``): the cheaper physics for the perception sweep (mesh_sweep.py).
        ``lobule``: the mid-surface carries on into the lobule, whose triangles get a soft core
        (skin / fibro-fatty tissue / skin, ``lobule_shell``) instead of cartilage."""
        import stack as sk
        import tissue_params as tp
        import wedge_test as wt
        import fibres as fb
        import dataclasses

        import peel_model as pm

        self.vol = vol
        v, t, th, has_cart = cached_mid_surface(vol, log=log, edge_mm=edge_mm, lobule=lobule, with_cart=True)
        self.mid_v, self.tris = v, t
        self.edge_mm = edge_mm
        self.has_cart = has_cart
        self.lobule_tris = ~has_cart[t].any(axis=1) if lobule else np.zeros(len(t), bool)
        self.d_head = head_distance(v, edge_mm=edge_mm, lobule=lobule)
        # close gaps over ~3 mm and drop islands under ~18 mm^2 whatever the edge size
        self.base = base_mask(v, t, self.d_head, rings=max(1, round(3.0 / edge_mm)),
                              min_size=max(2, round(8 * (EDGE_MM / edge_mm) ** 2)))
        skin = dataclasses.replace(tp.skin_params("skin", "skin_perichondrium"), thickness_mm=SKIN_MM, contact_n_per_mm3=0.0)
        cart = tp.shell_params(thickness_mm=CARTILAGE_MM)
        tight = tp.INTERFACES["skin_perichondrium"].opening.bond_law()
        self.fan_sets = place_fans(vol, v, self.base, self.d_head, log=log)
        n = len(v)
        d0, dc = 50.0, 100.0  # elastic over any pull here
        base_law = pm.BondLaw(0.5 * base_k * d0 * dc, d0, dc)
        self.L = layers
        if layers == 3:
            fibres = fb.Fibres.concat(self.fan_sets).shifted(n)  # on the cartilage layer
            base_ids = 2 * n + np.flatnonzero(self.base)  # the medial skin over the base, on the head
            self.stack = sk.Stack(v, t, [skin, cart, skin], [tight, tight], None, offset=True, curved=True,
                                  ref_layer=1, bending="shell", floor_z=-1e9, fibres=fibres, names=list(LAYERS),
                                  anchors=(base_ids, base_law))
        elif layers == 1:
            fibres = fb.Fibres.concat(self.fan_sets)
            self.shell = laminate_shell(skin, cart)
            self.stack = sk.Stack(v, t, [self.shell], [], None, offset=True, curved=True, ref_layer=0,
                                  bending="shell", floor_z=-1e9, fibres=fibres, names=["auricle (one shell)"],
                                  anchors=(np.flatnonzero(self.base), base_law))
        else:
            raise ValueError("layers: 3 or 1")
        if self.lobule_tris.any():
            lob_h = float(np.median(th[~has_cart])) if (~has_cart).any() else LOBULE_MM
            core = tp.skin_params("tpf")
            core = dataclasses.replace(core, thickness_mm=max(lob_h - 2 * SKIN_MM, 0.5), k1_mpa=0.0, k2=0.0)
            if layers == 1:
                self.lobule = lobule_shell(skin, core)
                set_triangle_material(self.stack.sheets[0], self.lobule_tris, self.lobule)
            else:  # the core layer is fibro-fatty in the lobule (the layer offsets stay the cartilage's)
                self.lobule = core
                set_triangle_material(self.stack.sheets[1], self.lobule_tris, core)
            log(f"lobule: {int(self.lobule_tris.sum())} triangles, {lob_h:.1f} mm thick, core E {core.E_mpa:g} MPa")
        self.n = n
        self.cart_layer = 1 if layers == 3 else 0
        self.held = np.zeros((layers * n, 3), bool)  # nothing held rigidly: the base is anchored
        log(f"model: {layers * n} vertices in {layers} layer(s), {len(self.stack.face_e) if layers > 1 else 0} face bonds, {len(fibres)} fibres, "
            f"{int(self.base.sum())} base vertices anchored at {base_k} N/mm^3 "
            f"({base_k * float(self.stack.weight.sum()):.2f} N/mm in all)")

    def grip(self, centre, radius=3.0):
        """Vertices of every layer within ``radius`` of ``centre`` (a pinch through the auricle)."""
        local = np.flatnonzero((np.linalg.norm(self.mid_v - centre, axis=1) <= radius) & ~self.base)
        return np.concatenate([self.stack.layer_vertices(k, local) for k in range(self.L)])

    def helix_top(self):
        """A point on the helix rim, up and back: the usual place to pull the auricle (otoscopy)."""
        L, A, S = (np.asarray(x, float) for x in (self.vol.frame.lateral, self.vol.frame.anterior, self.vol.frame.superior))
        v = self.mid_v
        free = ~self.base
        score = (v - v.mean(0)) @ (S - 0.4 * A) + 0.3 * self.d_head
        return v[np.flatnonzero(free)[np.argmax(score[free])]]


# Bending stiffness of the one-shell stand-in as a fraction of the fully bonded laminate's. A flat
# bonded strip gives 0.72 (laminate_test, REVIEW 6c: the perichondrium planes slip a little); on the
# curved ear under a helix pinch the 3-layer model is softer, and 0.45 matches the 1 mm 3-layer
# reference best (mesh_sweep.py, REVIEW 6f: drawn skin within 0.21 mm mean at 1.5 mm edges).
PARTIAL_BOND = 0.45


LOBULE_MM = 6.0  # used only if no lobule thickness was measured


def lobule_shell(skin, core):
    """One shell for the lobule: skin / soft fibro-fatty core / skin. Stretch as the sum of E h
    (the skins carry it), with the skins' J-curve; bending as the layers' own (a core this soft
    cannot carry the shear that would couple the two skins, so no sandwich effect)."""
    import dataclasses

    hs, hc = skin.thickness_mm, core.thickness_mm
    Eh_lin = 2 * skin.E_mpa * hs + core.E_mpa * hc
    EI = 2 * (skin.E_mpa + skin.k1_mpa) * hs**3 / 12 + core.E_mpa * hc**3 / 12
    h = np.sqrt(12 * EI / Eh_lin)
    return dataclasses.replace(skin, E_mpa=Eh_lin / h, thickness_mm=float(h), k1_mpa=2 * skin.k1_mpa * hs / h, k2=skin.k2,
                               nu=0.45)


def set_triangle_material(sheet, mask, params):
    """Give the triangles in ``mask`` of one sheet another material (fast solver only): per-triangle
    (thickness, mu, lam, k1, k2) in ``sheet.tri_material``, and their shell bending scaled by the
    ratio of bending stiffness D. The slow reference (peel_model) still uses the sheet's own."""
    import fast_solver as fs

    base = getattr(sheet, "tri_material", None)
    if base is None:
        base = fs.triangle_material(sheet.params, len(sheet.tris))
    new = fs.triangle_material(params, len(sheet.tris))
    sheet.tri_material = tuple(np.where(mask, b2, b1) for b1, b2 in zip(base, new))
    if sheet.shell_K is not None:
        sheet.shell_K = sheet.shell_K.copy()
        sheet.shell_K[mask] *= params.bending_D / sheet.params.bending_D


def laminate_shell(skin, cart, partial=None, j_curve=True):
    """One shell standing in for skin / cartilage / skin: the same stretch stiffness (sum E h) and
    bending stiffness (``partial`` x the fully bonded section's EI) at small strain, and, with
    ``j_curve``, the skin's collagen stiffening at large stretch (both skins' k1 h, the skin's k2).
    Cartilage stays linear (near-linear in tension to tearing; tissue_params). Returns
    pm.SkinParams with the equivalent E, thickness, k1 and k2.

    Bending uses only E (D = E h^3 / 12 (1 - nu^2)), so h is chosen for the linear part: E h =
    Ec hc + 2 Es hs and E h^3 / 12 = EI, then k1 = 2 k1_skin hs / h gives back the skins' small-strain
    k1 h. Without ``j_curve`` (the sweep's first version) the skins' k1 is folded into E."""
    import dataclasses

    partial = PARTIAL_BOND if partial is None else partial
    hs, hc, Ec = skin.thickness_mm, cart.thickness_mm, cart.E_mpa
    Es = skin.E_mpa + skin.k1_mpa  # small-strain skin modulus
    EI = partial * (Ec * hc**3 / 12 + 2 * Es * (hs**3 / 12 + hs * (0.5 * (hc + hs)) ** 2))
    if not j_curve or skin.k1_mpa <= 0:
        Eh = Ec * hc + 2 * Es * hs
        h = np.sqrt(12 * EI / Eh)
        return dataclasses.replace(cart, E_mpa=Eh / h, thickness_mm=float(h))
    Eh_lin = Ec * hc + 2 * skin.E_mpa * hs
    h = np.sqrt(12 * EI / Eh_lin)
    return dataclasses.replace(cart, E_mpa=Eh_lin / h, thickness_mm=float(h), k1_mpa=2 * skin.k1_mpa * hs / h,
                               k2=skin.k2)


def pull(model, grip, direction, distance, steps, fast=True, log=print):
    """Pull the grip ``distance`` mm along ``direction`` in ``steps``: (forces (N, 3), positions per step)."""
    import fast_solver as fs
    import peel_model as pm

    st = model.stack
    fixed = model.held.copy()
    fixed[grip] = True
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    x = st.rest.copy()
    forces, frames = [], []
    for k in range(1, steps + 1):
        t0 = time.perf_counter()
        x[grip] = st.rest[grip] + d * distance * k / steps
        if fast:
            x, info = fs.static_solve(st, x, fixed, gtol=1e-7, max_iter=100)
        else:
            x, info = pm.solve(st, x, fixed, gtol=1e-7, max_iter=100)
        g = st.energy(x)[1]
        f = g[grip].sum(axis=0)  # the force the hand must apply (balances the tissue)
        forces.append(f)
        frames.append(x.copy())
        log(f"  {'fast' if fast else 'slow'} pull {distance * k / steps:5.2f} mm: force {np.linalg.norm(f):7.4f} N "
            f"(along the pull {f @ d:7.4f} N), {info.nit} iterations, {1000 * (time.perf_counter() - t0):.0f} ms")
    return np.array(forces), frames
