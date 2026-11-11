"""Pull the auricle with the mouse: the first interactive test of the composite ear.

The model is ``auricle_sim.AuricleModel`` on the cartilage's mid-surface (1.5 mm edges): by default
one shell standing in for skin / cartilage / skin (``--layers 3`` for the full face-bonded laminate,
about 8x slower), the base on compliant soft tissue, the auricular muscles and ligaments as fans.
Solved by the fast C++ Newton, quasi-static, one Newton step a frame: the mesh sweep (REVIEW 6f)
found this within about 0.2 mm of the finest 3-layer model, and keeping up with a quick tug.

Physics mesh and render meshes are split: what you see is the finished composite (``build_ear.py``),
never solved, carried by the physics mesh each frame. The auricle's skin and cartilage ride on their
nearest mid-surface triangle at their own depth (``Embedding``); the head's skin near the ear moves
with the auricle's base, fading over ~8 mm (``HeadSkinFollow``, drawing only). The window (OpenGL)
keeps every mesh on the GPU and only rewrites its vertices; physics runs in its own thread.

The desktop icon (launch_auricle_pull.sh, --pick) first asks which model to load (MODELS).

Mouse
  Ctrl + left-drag   pinch the auricle where you click (a finger pad: springs over a ~3 mm patch,
                     every layer) and pull, in the plane of the screen; turn the view to pull another
                     way. Past SLIP_N (10 N) the fingers slip and the pull stops growing
  release            let go: the auricle springs back
  left-drag          orbit; right-drag: pan; scroll: zoom
Keys
  The legend (top left; G hides it) lists every layer, its colour and its key: 1 head skin, 2 auricle
  skin, 3 fat, 4 TPF / SMAS, 5 auricular muscles, 6 temporalis fascia, 7 temporalis, 8 periosteum,
  9 deep face tissue, 0 bone, C cartilage, F muscle fibres. L peels the head's layers one by one; the
  loose areolar plane (4) is a potential space (not drawn). R reset,
  [ ] base stiffness down / up (rebuilds), Esc quit
The title bar shows the pull, the force, physics and drawing times.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/auricle_pull_viewer.py [--layers 3]
    middle_ear_newton/.venv/bin/python scripts/soft_tissue/auricle_pull_viewer.py --selftest out_dir
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import auricle_sim as asim  # noqa: E402
import envelope as ev  # noqa: E402

GRIP_MM = 3.0
# The pinch is a finger pad, not a rigid clamp: springs (GRIP_N_PER_MM in all, shared over the pad by
# area x exp(-(r / GRIP_MM)^2), out to 1.5 GRIP_MM, every layer) pull the skin toward where the hand
# is. Fingertip pulp is about 1-5 N/mm under a firm pinch (estimate). The pinch holds by friction, so
# beyond SLIP_N the fingers slide and the pull stops growing (estimate: a firm two-finger pinch on
# skin, ~10 N squeeze x friction ~0.5 x 2 pads).
GRIP_N_PER_MM = 3.0
# contact with the head (bone and its tissue): stiffness per area of the ear's surface, the gap a free
# vertex keeps, and how much deeper than at rest a vertex that starts inside the head's tissue may go
CONTACT_N_PER_MM3 = 5.0
CONTACT_CLEAR_MM = 0.8
CONTACT_GIVE_MM = 0.3
# the ear against itself: clearance between non-neighbouring physics vertices (the drawn ear's
# thickness: cartilage 1.6 mm + skin 0.8 mm both sides), stiffness per pair, neighbours excluded
SELF_CLEAR_MM = 2.0
SELF_K_N_PER_MM = 5.0
SELF_RINGS = 4
SLIP_N = 10.0
NEWTON_PER_FRAME = 1  # enough to keep up (REVIEW 6f); more only costs time
LAYERS = 1  # one equivalent shell; 3 = the full laminate (--layers)
EDGE_MM = asim.EDGE_MM  # physics mid-surface edge (--edge)
LOBULE = True  # the physics carries on into the lobule (soft core); --no-lobule: cartilage only
SMOOTH_SKIN = True  # BlendEmbedding (smooth) for the drawn skin; False: Embedding (nearest triangle)
GPU_PLACE = True  # drawn meshes placed on the GPU with Warp (gpu_place.py); --cpu-place for numpy
SKIN_RGB = (240, 205, 190)  # the pinna's original tone for the pinna and the head's skin; real skin shading comes later (user)
COLOURS = {"auricle_skin": SKIN_RGB, "cartilage": (90, 150, 255), "envelope_skin": SKIN_RGB,
           "bone": (225, 222, 212), "canal_skin": (230, 120, 130), "atlas_temporalis_muscle": (178, 72, 66), "temporalis_muscle": (150, 50, 46), "auricular_muscles": (190, 62, 56),
           "tpf": (232, 205, 192), "face_deep": (205, 150, 140),
           "subcutaneous_fat": (240, 205, 120), "loose_areolar": (225, 222, 210), "temporalis_fascia": (215, 225, 240),
           "periosteum": (215, 195, 185),
           "muscles": (196, 74, 68)}


class Embedding:
    """Each drawn vertex on its nearest mid-surface triangle: barycentric place, and its whole offset
    from there in the triangle's own frame (edge, normal), so it turns with the triangle. Exact at rest,
    also for parts beyond the cartilage (lobule, rim, the skin closing underneath).

    The barycentric part is one sparse matrix; frames are computed once per physics triangle and
    shared by the drawn vertices on it."""

    def __init__(self, mid_v, tris, pts):
        import trimesh
        from scipy import sparse

        m = trimesh.Trimesh(mid_v, tris, process=False)
        q, _d, tid = trimesh.proximity.closest_point(m, pts)
        self.tid = tid
        self.tris = np.asarray(tris)
        self.bary = trimesh.triangles.points_to_barycentric(m.triangles[tid], q)
        n = len(pts)
        self.B = sparse.csr_matrix((self.bary.ravel(), (np.repeat(np.arange(n), 3), self.tris[tid].ravel())),
                                   shape=(n, len(mid_v)))
        R = self._frames(np.asarray(mid_v, float))[tid]
        self.local = np.einsum("nji,nj->ni", R, pts - q)  # R^T (p - q)

    def _frames(self, x):
        """(F, 3, 3) frame of every physics triangle, columns: edge, normal, normal x edge."""
        t = self.tris
        a, b, c = x[t[:, 0]], x[t[:, 1]], x[t[:, 2]]
        e1 = b - a
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return np.stack([e1, n, np.cross(n, e1)], axis=2)

    def place(self, x):
        R = self._frames(x)[self.tid]
        return self.B @ x + np.einsum("nij,nj->ni", R, self.local)


class BlendEmbedding:
    """Drawn vertices carried by a smooth blend of nearby physics vertices (the pull viewer's default).

    ``Embedding`` ties each drawn vertex to its one nearest physics triangle. Where the nearest
    triangle jumps (skin bridging two parts of the cartilage, rims, folds), neighbouring drawn
    vertices ride different parts and the skin between them tears into long strips under a big pull
    (REVIEW 6f). Here each drawn vertex P takes its ``k`` nearest physics vertices (by rest distance
    from P itself, so neighbours always get nearly the same blend) with weights
    (1 - (d / d_k+1)^2)^3, which reach zero at the (k+1)th neighbour, so the blend is continuous:

        p = sum_j w_j (x_j + Q_j (P - X_j))

    Q_j is each physics vertex's rotation: the polar part of the area-free sum of its triangles'
    rest-to-now frame rotations. Exact at rest."""

    K = 16

    def __init__(self, mid_v, tris, pts, k=K):
        from scipy import sparse
        from scipy.spatial import cKDTree

        X = np.asarray(mid_v, float)
        self.tris = np.asarray(tris)
        f = len(self.tris)
        self.VT = sparse.csr_matrix((np.ones(3 * f), (self.tris.ravel(), np.repeat(np.arange(f), 3))), shape=(len(X), f))
        self.R0 = frames(X, self.tris)
        d, j = cKDTree(X).query(pts, k=k + 1)
        w = np.clip(1 - (d[:, :k] / np.maximum(d[:, k:], 1e-9)) ** 2, 0, None) ** 3
        w[:, 0] += 1e-12
        self.w = w / w.sum(axis=1, keepdims=True)
        self.j = j[:, :k]
        self.arm = np.asarray(pts, float)[:, None, :] - X[self.j]

    def rotations(self, x):
        Q = np.einsum("fij,fkj->fik", frames(x, self.tris), self.R0)
        A = (self.VT @ Q.reshape(len(Q), 9)).reshape(-1, 3, 3)
        U, _s, Vt = np.linalg.svd(A)
        flip = np.linalg.det(U @ Vt) < 0
        U[flip, :, 2] *= -1
        return U @ Vt

    def place(self, x):
        Q = self.rotations(x)
        return np.einsum("nk,nkc->nc", self.w, x[self.j] + np.einsum("nkij,nkj->nki", Q[self.j], self.arm))


def frames(x, tris):
    """(F, 3, 3) frame of every triangle, columns: first edge, normal, normal x edge."""
    a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    e1 = b - a
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    n = np.cross(b - a, c - a)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.stack([e1, n, np.cross(n, e1)], axis=2)


class HeadSkinFollow:
    """The head's skin near the ear carried along by the auricle's base (drawing only, no forces).

    The base sits on the soft tissue that holds the head's skin, so the skin there moves with it,
    less with distance: each skin vertex within ``3 lam`` of the base takes the inverse-distance
    blend of its ``k`` nearest base vertices' motion, times exp(-(r / lam)^2), r its distance to the
    nearest one. ``lam`` (mm) is an estimate of how far the subcutis spreads a base movement."""

    LAM_MM = 8.0

    def __init__(self, mid_v, base, skin_v, lam=LAM_MM, k=6, out_dir=None):
        """``out_dir``: the outward direction, one vector or one per vertex of ``skin_v`` (away from the
        nearest bone). Given, the motion's inward part is dropped: the superficial layers glide over
        the deep ones or lift off, never sink into them (the temporalis fascia and periosteum stay)."""
        from scipy.spatial import cKDTree

        od = None if out_dir is None else np.asarray(out_dir, float)
        if od is not None:
            od = np.broadcast_to(od, np.asarray(skin_v).shape) / np.maximum(np.linalg.norm(od, axis=-1, keepdims=True), 1e-9)
        self.rest = np.asarray(skin_v, float)
        self.mid_rest = np.asarray(mid_v, float)
        ids = np.flatnonzero(base)
        d, j = cKDTree(self.mid_rest[ids]).query(self.rest, k=k)
        near = d[:, 0] < 3 * lam
        self.near = np.flatnonzero(near)
        d, j = d[near], ids[j[near]]
        w = 1.0 / np.maximum(d, 0.5)
        w /= w.sum(axis=1, keepdims=True)
        self.w = w * np.exp(-(d[:, :1] / lam) ** 2)
        self.j = j
        self.out_dir = None if od is None else np.ascontiguousarray(od[self.near])  # per near vertex

    def place(self, x_mid):
        out = self.rest.copy()
        disp = x_mid - self.mid_rest
        s = np.einsum("nk,nkc->nc", self.w, disp[self.j])
        if self.out_dir is not None:
            s -= np.minimum((s * self.out_dir).sum(1), 0.0)[:, None] * self.out_dir
        out[self.near] += s
        return out


class Puller:
    """The model, its fast solver and the hand: thread-safe enough for one GUI and one physics thread."""

    def __init__(self, base_k=asim.BASE_K_N_PER_MM3, log=print):
        import fast_solver as fs

        self.fs = fs
        self.vol = ev.Volume(asim.ADF)
        self.model = asim.AuricleModel(self.vol, base_k=base_k, log=log, layers=LAYERS, edge_mm=EDGE_MM, lobule=LOBULE)
        st = self.model.stack
        self.st = st
        self.m = fs.FastModel.from_stack(st)
        self.m.set_options(project_psd=True, threads=4)
        try:
            self.m.set_factor(fs.best_factor())
        except Exception:
            pass
        self.x = st.rest.copy()
        self.v = np.zeros_like(self.x)
        self.lock = threading.Lock()
        self.grip = None
        self.grip_rest = None
        self.offset = np.zeros(3)
        self.force = np.zeros(3)
        self.ms = 0.0
        self.base_fixed = self.model.held.copy()
        self.m.set_fixed(self.base_fixed)
        # build the sparse pattern and factor now (gtol 0 forces one Newton step), not on the first pull
        self.m.step(self.x.copy(), np.zeros_like(self.x), 0.0, fs.NEWTON, 1, gtol=0.0)
        n = st.n
        self.cart = st.layer_slice(self.model.cart_layer)
        self.contact = self._head_contact()
        self._self_contact()

    def _self_contact(self):
        """The ear cannot pass through itself (user): its physics vertices keep SELF_CLEAR_MM apart
        (the drawn ear's thickness: cartilage and the skin on both sides), parts closer at rest no
        closer than they start; mesh neighbours within SELF_RINGS rings never count. The 3-layer
        model uses its cartilage layer (its stacked layers are kept apart by their bonds)."""
        import os

        if os.environ.get("FIVRS_NO_CONTACT") or os.environ.get("FIVRS_NO_SELF"):
            return
        n, L = self.model.n, self.model.L
        on = np.zeros(len(self.st.rest), bool)
        on[self.cart] = True
        tris = np.concatenate([self.model.tris + k * n for k in range(L)])
        self.m.set_self_contact(SELF_K_N_PER_MM, SELF_CLEAR_MM, on, tris, rings=SELF_RINGS)

    def _head_contact(self):
        """Nothing passes through anything (user): the head (bone and its tissue up to the skin, the
        build's head_obstacle.npz) is an obstacle for every physics vertex. A free vertex keeps
        CONTACT_CLEAR_MM off it; one that starts inside it (the ear's root, the concha floor sitting
        in the head's soft tissue) may go no more than CONTACT_GIVE_MM deeper than it starts.
        Returns the number of vertices in contact range at rest, or None without the obstacle."""
        path = asim.COMPOSITE / "head_obstacle.npz"
        import os

        if not path.exists() or os.environ.get("FIVRS_NO_CONTACT") or os.environ.get("FIVRS_NO_HEAD"):
            return None
        from scipy import ndimage

        z = np.load(path)
        phi, o, h = z["phi"].astype(np.float32), z["origin"], z["h"]
        self.m.set_obstacle(phi, o, h)
        p0 = ndimage.map_coordinates(phi, ((self.st.rest - o) / h).T, order=1, mode="nearest")
        clear = np.minimum(CONTACT_CLEAR_MM, p0 - CONTACT_GIVE_MM)
        self.m.set_contact(CONTACT_N_PER_MM3 * self.st.vert_area, clear)
        return int((p0 < CONTACT_CLEAR_MM).sum())

    def cartilage_x(self):
        with self.lock:
            return self.x[self.cart].copy()

    def grab(self, point, radius=GRIP_MM):
        with self.lock:
            cx = self.x[self.cart]
            r = np.linalg.norm(cx - point, axis=1)
            local = np.flatnonzero((r <= 1.5 * radius) & ~self.model.base)
            if not len(local):
                local = np.array([int(np.argmin(r))])
            ids = np.concatenate([self.st.layer_vertices(k, local) for k in range(self.model.L)])
            w = self.st.vert_area[ids] * np.exp(-(r[local] / radius) ** 2)[np.tile(np.arange(len(local)), self.model.L)]
            self.grip = ids
            self.grip_k = np.zeros(len(self.x))
            self.grip_k[ids] = GRIP_N_PER_MM * w / w.sum()
            self.grip_rest = self.x[ids].copy()
            self.offset = np.zeros(3)
            self.slipping = False
            self.targets = self.x.copy()
            self.m.set_grip(self.grip_k, self.targets)
        return len(local)

    def drag(self, offset):
        # no lock: one assignment the physics reads at its next step (the window must never wait
        # for a solve: that capped drawing at the physics rate)
        self.offset = np.asarray(offset, float)

    def release(self):
        with self.lock:
            self.grip = None
            self.m.set_grip(None)

    def reset(self):
        with self.lock:
            self.grip = None
            self.m.set_grip(None)
            self.x = self.st.rest.copy()

    def _spring_force(self):
        g = self.grip
        return (self.grip_k[g, None] * (self.targets[g] - self.x[g])).sum(axis=0)

    def step(self):
        fs = self.fs
        with self.lock:
            if self.grip is not None:
                g = self.grip
                self.targets[g] = self.grip_rest + self.offset
                f = self._spring_force()
                self.slipping = np.linalg.norm(f) > SLIP_N
                if self.slipping:  # the fingers slide: hold the pull at SLIP_N
                    self.targets[g] = self.x[g] + (self.targets[g] - self.x[g]) * (SLIP_N / np.linalg.norm(f))
                self.m.set_grip_targets(self.targets)
            t0 = time.perf_counter()
            self.m.step(self.x, self.v, 0.0, fs.NEWTON, NEWTON_PER_FRAME, gtol=1e-6)
            self.v[:] = 0.0
            self.ms = 1000 * (time.perf_counter() - t0)
            self.force = self._spring_force() if self.grip is not None else np.zeros(3)
            self.x_all = self.x.copy()  # every layer, for drawing each layer on its own
            return self.x[self.cart].copy(), self.force.copy(), float(np.linalg.norm(self.offset)) if self.grip is not None else 0.0

    def sources(self):
        """Where the drawn meshes ride: {"core": (rest, tris, take)} with take(x_all) -> that rest's
        current positions. One shell: everything on it. Three layers: the drawn skin on the two
        skin layers together (each drawn point blends its nearest, so the lateral skin follows the
        lateral layer, the medial the medial, and the rim blends the two), the drawn cartilage and
        the head skin on the cartilage layer."""
        st, n, m = self.st, self.model.n, self.model
        core = (st.rest[self.cart], m.tris, lambda x: x[self.cart])
        if m.L == 1:
            return {"core": core, "skin": core}
        a, b = st.layer_slice(0), st.layer_slice(2)
        skin = (np.concatenate([st.rest[a], st.rest[b]]), np.concatenate([m.tris, m.tris + n]),
                lambda x: np.concatenate([x[a], x[b]]))
        return {"core": core, "skin": skin}


# L peels the head's layers off one by one, outermost first (drawing only; the deeper ones do not
# move with the ear): skin, fat, the loose areolar plane (the auricular muscles lie in it, under the
# fat), the temporalis fascia; under it the temporalis and the periosteum.
PEEL = ("envelope_skin", "subcutaneous_fat", "tpf", "auricular_muscles", "temporalis_fascia", "temporalis_muscle")
# drawn as the build's solids (build_ear.py: solid_<layer>.ply, stacked tight, cut flat at the scene's box)
SOLIDS = {"envelope_skin": "solid_skin", "bone": "solid_bone", "periosteum": "solid_periosteum", "face_deep": "solid_face_deep",
          **{n: f"solid_{n}" for n in PEEL[1:]}}
PEEL_START = 0  # --peel
# on-screen legend: (key, mesh, label), top to bottom = outside in; number keys toggle each, G hides it
LEGEND = (("1", "envelope_skin", "skin of the head (1)"), ("2", "auricle_skin", "skin of the auricle"),
          ("3", "subcutaneous_fat", "subcutaneous fat (2)"), ("4", "tpf", "temporoparietal fascia / SMAS (3)"),
          ("5", "auricular_muscles", "auricular muscles (3)"), ("6", "temporalis_fascia", "temporalis fascia"),
          ("7", "temporalis_muscle", "temporalis"), ("8", "periosteum", "periosteum"),
          ("9", "face_deep", "deep tissue of the face"), ("0", "bone", "bone"), ("C", "cartilage", "auricular cartilage"),
          ("F", "fans", "auricular muscle fibres (physics)"))
SEAM_MM = 3.0  # the auricle's skin this close to the head's follows the head's skin (fading out)
FIT_TO_LAYERS = False  # lay the bellies in the plane under the fat: off, the per-point fit crumples them (to redo)
PRESS_TEMPORALIS = False  # the layers are built around the placed temporalis, so it is drawn as placed
# (pressing it point by point under its fascia crumpled it)
# (the layers are built around the placed temporalis: build_ear.py)


def scene_temporalis(v, t, bone_v):
    """The placed atlas temporalis (atlas_placement.py) for this scene: the part inside the CT cube,
    fitted between the periosteum (under it) and the temporalis fascia's underside (the areolar
    surface minus the areolar and fascia thicknesses): the atlas is another person, its edges poke
    out. Drawing only: the temporalis and its fascia stay put when the ear is pulled (user)."""
    lo, hi = bone_v.min(0) - 3.0, bone_v.max(0) + 3.0
    t = t[((v[t] > lo) & (v[t] < hi)).all(axis=(1, 2))]
    if not PRESS_TEMPORALIS:
        return v, t
    try:
        import muscle_bellies as mb
        import trimesh

        L = np.asarray(ev.Volume(asim.ADF).frame.lateral, float)
        deep = {n: th for n, th, _s in ev.LAYERS}
        used = np.unique(t)
        v = v.copy()
        roof = asim.COMPOSITE / "temporalis_muscle.ply"  # the muscle layer's outer face: its fascia's underside
        if roof.exists():
            v[used] = mb.press_under(v[used], trimesh.load(roof, process=False), L, 0.05)
        else:
            v[used] = mb.press_under(v[used], trimesh.load(asim.COMPOSITE / "loose_areolar.ply", process=False), L,
                                     deep["loose_areolar"] + deep["temporalis_fascia"])
        v[used] = mb.press_under(v[used], trimesh.load(asim.COMPOSITE / "periosteum.ply", process=False), -L, 0.05)
    except Exception as e:
        print(f"temporalis not fitted under its fascia: {e}")
    return v, t


def load_drawn(names):
    import trimesh

    out = {}
    for n in names:
        p = asim.COMPOSITE / f"{n}.ply"
        if p.exists():
            m = trimesh.load(p, process=False)
            out[n] = (np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
    return out


# -- headless check: a scripted grab and pull, rendered offscreen --------------------------------
def selftest(out_dir, pull_mm=10.0, frames=12):
    import open3d as o3d
    from open3d.visualization import rendering
    from PIL import Image, ImageDraw

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = Puller()
    drawn = load_drawn(["auricle_skin", "cartilage", "envelope_skin"])
    emb = {k: BlendEmbedding(P.model.mid_v, P.model.tris, v) for k, (v, _t) in drawn.items() if k in ("auricle_skin", "cartilage")}
    emb["envelope_skin"] = HeadSkinFollow(P.model.mid_v, P.model.base, drawn["envelope_skin"][0])
    L, A, S = (np.asarray(x, float) for x in (P.vol.frame.lateral, P.vol.frame.anterior, P.vol.frame.superior))
    P.grab(P.model.helix_top())
    d = (S - A) / np.linalg.norm(S - A)
    E = P.model.mid_v.mean(axis=0)
    ims = []
    for k in range(frames + 1):
        P.drag(d * pull_mm * k / frames)
        for _ in range(6):
            cx, f, dist = P.step()
        r = rendering.OffscreenRenderer(640, 640)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        for name, (v, t) in drawn.items():
            vv = emb[name].place(cx) if name in emb else v
            g = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv), o3d.utility.Vector3iVector(t))
            g.compute_vertex_normals()
            g.paint_uniform_color([c / 255 for c in COLOURS[name]])
            r.scene.add_geometry(name, g, mat)
        r.scene.set_background([0.06, 0.06, 0.08, 1])
        r.setup_camera(40, E, E + 110 * (L + 0.15 * A) / np.linalg.norm(L + 0.15 * A), S)
        im = Image.fromarray(np.asarray(r.render_to_image())).convert("RGB")
        del r
        ImageDraw.Draw(im).text((10, 10), f"pull up and back {dist:4.1f} mm   force {np.linalg.norm(f):5.2f} N   "
                                          f"solver {P.ms:.0f} ms / frame", fill=(255, 255, 255))
        ims.append(im)
        print(f"selftest {dist:5.2f} mm: force {np.linalg.norm(f):6.3f} N, {P.ms:.0f} ms per frame", flush=True)
    ims[0].save(out / "auricle_pull_selftest.gif", save_all=True, append_images=ims[1:], duration=250, loop=0)
    print(f"wrote {out / 'auricle_pull_selftest.gif'}")


# -- the window --------------------------------------------------------------------------------
# A lean OpenGL window (GLFW): every drawn mesh stays on the GPU and only its vertex positions and
# normals are overwritten each frame (the physics mesh is never drawn; the dense render meshes ride
# on it). Physics runs in its own thread at its own rate; the window draws the newest result.
VERT = """
#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 nrm;
uniform mat4 mvp;
out vec3 n_world;
void main() { n_world = nrm; gl_Position = mvp * vec4(pos, 1.0); }
"""
FRAG = """
#version 330 core
in vec3 n_world;
uniform vec3 colour;
uniform vec3 view_dir;
uniform int lit;
out vec4 frag;
void main() {
    if (lit == 0) { frag = vec4(colour, 1.0); return; }
    vec3 n = normalize(n_world);
    if (!gl_FrontFacing) n = -n;
    vec3 l = normalize(-view_dir + vec3(0.0, 0.0, 0.0));
    float d = abs(dot(n, l));
    vec3 h = normalize(l - view_dir);
    float s = pow(max(dot(n, h), 0.0), 40.0);
    frag = vec4(colour * (0.25 + 0.75 * d) + 0.12 * s, 1.0);
}
"""


def vertex_normal_matrix(tris, n_vertices):
    """Sparse (V, F) matrix: vertex normal = sum of its triangles' area-weighted normals."""
    from scipy import sparse

    f = len(tris)
    return sparse.csr_matrix((np.ones(3 * f), (tris.ravel(), np.repeat(np.arange(f), 3))), shape=(n_vertices, f))


def normals(v, tris, M):
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    vn = M @ fn
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


LEGEND_VERT = """#version 330 core
layout(location=0) in vec2 pos; layout(location=1) in vec2 uv; out vec2 v_uv;
void main() { v_uv = uv; gl_Position = vec4(pos, 0.0, 1.0); }
"""
LEGEND_FRAG = """#version 330 core
in vec2 v_uv; out vec4 frag; uniform sampler2D tex;
void main() { frag = texture(tex, v_uv); }
"""


class Legend:
    """The legend drawn as an image over the 3D view (top left): a swatch, the layer's name, its key,
    and whether it is shown. Redrawn when something is toggled."""

    def __init__(self):
        from OpenGL import GL
        from OpenGL.GL import shaders

        self.GL = GL
        self.prog = shaders.compileProgram(shaders.compileShader(LEGEND_VERT, GL.GL_VERTEX_SHADER),
                                           shaders.compileShader(LEGEND_FRAG, GL.GL_FRAGMENT_SHADER))
        self.tex = GL.glGenTextures(1)
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.size = (1, 1)
        self.visible = True

    def update(self, rows, footer):
        """rows: (key, label, rgb, shown or None)."""
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        except Exception:
            font = bold = ImageFont.load_default()
        lh, w = 22, 430
        h = 30 + lh * len(rows) + 20 * len(footer) + 12
        img = Image.new("RGBA", (w, h), (12, 13, 16, 200))
        d = ImageDraw.Draw(img)
        d.text((10, 6), "Layers (key toggles)", font=bold, fill=(235, 235, 235, 255))
        y = 30
        for key, label, rgb, shown in rows:
            dim = shown is False
            sw = tuple(int(c * (0.35 if dim else 1.0)) for c in rgb) + (255,)
            d.rectangle([10, y + 3, 26, y + 17], fill=sw, outline=(200, 200, 200, 255))
            col = (120, 120, 120, 255) if dim else (235, 235, 235, 255)
            d.text((34, y + 1), f"{key}", font=bold, fill=(255, 210, 90, 255) if not dim else (120, 110, 80, 255))
            d.text((54, y + 1), label + ("  (off)" if dim else ""), font=font, fill=col)
            y += lh
        for line in footer:
            d.text((10, y + 4), line, font=font, fill=(190, 190, 190, 255))
            y += 20
        data = np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM), np.uint8)
        GL = self.GL
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        self.size = (w, h)

    def draw(self, fb_w, fb_h, scale=1.0):
        if not self.visible:
            return
        GL = self.GL
        w, h = self.size[0] * scale, self.size[1] * scale
        x0, y1 = -1.0 + 2 * 12 / fb_w, 1.0 - 2 * 12 / fb_h
        x1, y0 = x0 + 2 * w / fb_w, y1 - 2 * h / fb_h
        quad = np.array([[x0, y0, 0, 0], [x1, y0, 1, 0], [x1, y1, 1, 1], [x0, y0, 0, 0], [x1, y1, 1, 1], [x0, y1, 0, 1]],
                        np.float32)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, GL.ctypes.c_void_p(8))
        GL.glUseProgram(self.prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "tex"), 0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)


class GLMesh:
    def __init__(self, v, t, colour, dynamic, lines=False):
        from OpenGL import GL

        self.GL = GL
        self.t = np.ascontiguousarray(t, np.uint32)
        self.count = self.t.size
        self.colour = np.array(colour, np.float32) / (255.0 if max(colour) > 1 else 1.0)
        self.lines = lines
        self.visible = True
        self.M = None if lines else vertex_normal_matrix(np.asarray(t), len(v))
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        usage = GL.GL_DYNAMIC_DRAW if dynamic else GL.GL_STATIC_DRAW
        self.vbo = GL.glGenBuffers(2)
        nv = np.zeros_like(v) if lines else normals(np.asarray(v, float), np.asarray(t), self.M)
        for k, data in enumerate((v, nv)):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[k])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, np.ascontiguousarray(data, np.float32), usage)
            GL.glEnableVertexAttribArray(k)
            GL.glVertexAttribPointer(k, 3, GL.GL_FLOAT, False, 0, None)
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.t, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)

    def update(self, v, n=None):
        GL = self.GL
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[0])
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, np.ascontiguousarray(v, np.float32))
        if n is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo[1])
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, np.ascontiguousarray(n, np.float32))

    def draw(self, prog):
        GL = self.GL
        if not self.visible:
            return
        GL.glUniform3fv(GL.glGetUniformLocation(prog, "colour"), 1, self.colour)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "lit"), 0 if self.lines else 1)
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_LINES if self.lines else GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)


def _look_at(eye, centre, up):
    f = centre - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


class Viewer:
    """Keys: F fans, B bone, H head skin, C cartilage, R reset, [ and ] base stiffness (rebuild),
    Esc quit. Mouse: Ctrl + left-drag pinch and pull; left-drag orbit; right-drag pan; scroll zoom."""

    # the superficial layers (1-3: skin, fat, TPF with the auricular muscles) move together with the
    # auricle's base, gliding over the temporalis fascia and everything deeper, which stay (user,
    # 2026-11-11). The plane between them (layer 4, loose areolar) is a potential space.
    FOLLOW = ("envelope_skin", "subcutaneous_fat", "tpf", "auricular_muscles")
    DYNAMIC = ("auricle_skin", "cartilage", *FOLLOW)

    def __init__(self, layers=None):
        import glfw
        from OpenGL import GL
        from OpenGL.GL import shaders

        self.glfw, self.GL = glfw, GL
        # load everything slow before the window exists: a window that does not answer events for
        # a few seconds gets the desktop's "not responding" dialog
        drawn = load_drawn(["auricle_skin", "cartilage", "envelope_skin", "bone"])
        solids = load_drawn(list(SOLIDS.values()))
        for name, file in SOLIDS.items():  # the cutaway block's solids where the build made them
            if file in solids:
                drawn[name] = solids[file]
        if "bone" in drawn:  # lighter for drawing
            import open3d as o3d

            v, t = drawn["bone"]
            g = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t)).simplify_quadric_decimation(150000)
            drawn["bone"] = (np.asarray(g.vertices), np.asarray(g.triangles, np.int32))
        self.drawn = drawn
        self._camera_from(drawn["auricle_skin"][0])
        assert glfw.init(), "GLFW could not start"
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        # a fixed class (the title changes every frame) so the dock files it under the desktop icon
        glfw.window_hint_string(glfw.X11_CLASS_NAME, "fivrs-soft-ear")
        glfw.window_hint_string(glfw.X11_INSTANCE_NAME, "fivrs-soft-ear")
        self.win = glfw.create_window(1500, 950, "Auricle pull: building the model ...", None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        self.prog = shaders.compileProgram(shaders.compileShader(VERT, GL.GL_VERTEX_SHADER),
                                           shaders.compileShader(FRAG, GL.GL_FRAGMENT_SHADER))
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        self.legend = Legend()
        for cb, fn in ((glfw.set_mouse_button_callback, self._button), (glfw.set_cursor_pos_callback, self._motion),
                       (glfw.set_scroll_callback, self._scroll), (glfw.set_key_callback, self._key)):
            cb(self.win, fn)
        self.lock = threading.Lock()
        self.P = None
        self.meshes = {}
        self.latest, self.seq, self.drawn_seq = None, 0, -1
        self.mouse = None  # (button, x, y) while a drag is on
        self.grabbing = False
        self.base_k = asim.BASE_K_N_PER_MM3
        self.status = "building the model ..."
        self.fps, self.frames, self.t_fps = 0.0, 0, time.perf_counter()
        self.phys_ms = self.place_ms = 0.0
        self.phys_hz, self.phys_n, self.t_phys = 0.0, 0, time.perf_counter()
        self.show = {"fans": False, "bone": True, "envelope_skin": True, "cartilage": False, "auricle_skin": True,
                     "muscles": False, "atlas_temporalis_muscle": True}
        self.generation = 0
        self.peel = PEEL_START
        for i, name in enumerate(PEEL):
            self.show[name] = i >= self.peel
        for name, (v, t) in drawn.items():
            self.meshes[name] = GLMesh(v, t, COLOURS[name], dynamic=name in self.DYNAMIC)
            self.meshes[name].visible = self.show.get(name, True)
        threading.Thread(target=self._build, args=(self.base_k,), daemon=True).start()

    # -- model and physics thread ----------------------------------------------------------
    def _build(self, base_k):
        self.generation += 1
        gen = self.generation
        self.P = None
        self.status = f"building the model (base {base_k:.4f} N/mm^3) ..."
        P = Puller(base_k=base_k, log=lambda *a: None)
        src = P.sources()
        # the superficial layers follow the auricle's base with the inward part of its motion dropped
        # at the base itself (each base vertex: away from its nearest bone), once for all of them, so
        # skin, fat and TPF move by the same field and stay together (glide or lift, never sink)
        Xc, Tc, take = src["core"]
        nb = self._outward(Xc)

        def take_head(x_all, Xc=Xc, nb=nb, take=take):
            u = take(x_all) - Xc
            return Xc + u - np.minimum((u * nb).sum(1), 0.0)[:, None] * nb

        src["head"] = (Xc, Tc, take_head)
        rides = {"auricle_skin": "skin", "cartilage": "core", **{k: "head" for k in self.FOLLOW if k in self.drawn}}
        Emb = BlendEmbedding if SMOOTH_SKIN else Embedding
        tris = {k: v[1] for k, v in self.drawn.items()}
        placers = {}  # source -> (GPUPlacer or None, CPU placers by mesh)
        for key in ("core", "skin", "head"):
            if key == "skin" and src["skin"] is src["core"]:
                continue
            X, T, _take = src[key]
            names = [k for k, r in rides.items() if r == key or (key == "core" and r == "skin" and src["skin"] is src["core"])]
            emb = {k: Emb(X, T, self.drawn[k][0]) for k in names if k not in self.FOLLOW}
            head = {k: HeadSkinFollow(X, P.model.base, self.drawn[k][0]) for k in names if k in self.FOLLOW} or None
            gpu = None
            if GPU_PLACE:
                import gpu_place

                gpu = gpu_place.GPUPlacer.create(X, T, emb, tris, head, log=print)
            if gpu is None and Emb is BlendEmbedding:  # ~85 ms a frame on the CPU: use the nearest-triangle one
                emb = {k: Embedding(X, T, self.drawn[k][0]) for k in emb}
            cpu = dict(emb)
            if head is not None:
                cpu.update(head)
            placers[key] = (gpu, cpu, names)
        if gen != self.generation:
            return
        self.placers, self.src = placers, src
        self.seam = None
        try:  # the auricle's skin where it meets the head's: moves as the head's skin does there, so
            # the two stay one sheet (the sulcus stretches; no gap opens onto the fascia)
            import open3d as o3d

            av = self.drawn["auricle_skin"][0]
            hs = self.drawn["envelope_skin"]
            sc = o3d.t.geometry.RaycastingScene()  # (trimesh's proximity query takes ~25 GB here)
            sc.add_triangles(o3d.core.Tensor(np.asarray(hs[0], np.float32)), o3d.core.Tensor(np.asarray(hs[1], np.uint32)))
            d = sc.compute_distance(o3d.core.Tensor(np.asarray(av, np.float32))).numpy()
            idx = np.flatnonzero(d < SEAM_MM)
            s_ = np.clip(1.0 - d[idx] / SEAM_MM, 0.0, 1.0)
            self.seam = (idx, (s_ * s_ * (3 - 2 * s_))[:, None],
                         HeadSkinFollow(src["core"][0], P.model.base, av[idx]))
        except Exception as e:
            print(f"seam not joined: {e}")
        self.bellies = None
        try:  # the auricular muscles drawn as thin bellies on the fascia (muscle_bellies.py)
            import muscle_bellies as mb
            import trimesh

            roof = trimesh.load(asim.COMPOSITE / "loose_areolar.ply", process=False)  # the fat's underside
            skin = trimesh.load(asim.COMPOSITE / "envelope_skin.ply", process=False)
            allv = np.concatenate([self.drawn[k][0] for k in ("bone", "envelope_skin", "auricle_skin") if k in self.drawn])
            box = (allv.min(0) - 1.0, allv.max(0) + 1.0)  # drawn only inside the scene
            self.bellies = mb.Bellies(P.model, roof, skin, box=box) if FIT_TO_LAYERS else mb.Bellies(P.model, box=box)
            self.belly_M = vertex_normal_matrix(self.bellies.tris, len(self.bellies.rest))
        except Exception as e:
            print(f"muscle bellies not drawn: {e}")
        gpu = placers["core"][0]
        self.place_on = "GPU" if gpu is not None else "CPU"
        self.frame = np.asarray(P.vol.frame.superior, float)
        self.P = P
        self.status = (f"{MODEL_NAME}: {P.model.L} layer(s), {len(P.st.rest)} physics vertices, base {base_k:.4f} N/mm^3; "
                       f"drawn {len(self.drawn['auricle_skin'][0])} skin vertices")
        self._physics(P, gen)

    def _outward(self, pts):
        """Per point: the direction away from its nearest bone (the way out of the head there)."""
        from scipy.spatial import cKDTree

        if getattr(self, "_bone_tree", None) is None:
            self._bone_tree = cKDTree(self.drawn["bone"][0])
        _d, i = self._bone_tree.query(pts)
        out = np.asarray(pts, float) - self.drawn["bone"][0][i]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def _physics(self, P, gen):
        """Step, place the drawn meshes on the result, hand the newest frame to the window."""
        while gen == self.generation:
            cx, f, dist = P.step()
            t0 = time.perf_counter()
            show = [k for k in self.DYNAMIC if self.show.get(k, True) or self.latest is None]
            out = {}
            for key, (gpu, cpu, names) in self.placers.items():
                x = self.src[key][2](P.x_all)
                want = [k for k in names if k in show]
                if gpu is not None:
                    out.update(gpu.place(x, want))
                else:
                    for name in want:
                        v = cpu[name].place(x)
                        out[name] = (v, normals(v, self.drawn[name][1], self.meshes[name].M))
            if self.bellies is not None and self.show.get("muscles", True):
                bv = self.bellies.place(cx)
                out["muscles"] = (bv, normals(bv, self.bellies.tris, self.belly_M))
            if self.seam is not None and "auricle_skin" in out:
                idx, w, hf = self.seam
                v_, n_ = out["auricle_skin"]
                v_ = v_.copy()
                v_[idx] = (1 - w) * v_[idx] + w * hf.place(self.src["head"][2](P.x_all))
                out["auricle_skin"] = (v_, n_)
            F = P.st.fibres
            ins = cx[F.a - P.st.n * P.model.cart_layer]
            out["fans"] = np.concatenate([ins, F.origin])
            self.place_ms = 1000 * (time.perf_counter() - t0)
            with self.lock:
                self.latest = (out, f, dist, P.grip is not None)
                self.seq += 1
            self.phys_ms = P.ms
            self.phys_n += 1
            idle = P.grip is None and P.ms < 1.0
            time.sleep(0.02 if idle else 0.0)

    # -- camera ------------------------------------------------------------------------------
    def _camera_from(self, pts):
        vol = ev.Volume(asim.ADF)
        L, A, S = (np.asarray(x, float) for x in (vol.frame.lateral, vol.frame.anterior, vol.frame.superior))
        self.centre = pts.mean(axis=0)
        self.eye_dir = (L + 0.15 * A) / np.linalg.norm(L + 0.15 * A)
        import os

        view = os.environ.get("FIVRS_VIEW")  # "a,s,l": look from this direction (anterior, superior, lateral)
        if view:
            a_, s_, l_ = (float(x) for x in view.split(","))
            e = a_ * A + s_ * S + l_ * L
            self.eye_dir = e / np.linalg.norm(e)
        self.up = S
        self.dist = 150.0

    def _matrices(self):
        w, h = self.glfw.get_framebuffer_size(self.win)
        eye = self.centre + self.dist * self.eye_dir
        up = self.up - (self.up @ self.eye_dir) * self.eye_dir
        V = _look_at(eye, self.centre, up / np.linalg.norm(up))
        Pm = _perspective(35.0, w / max(h, 1), 1.0, 3000.0)
        return V, Pm, eye, w, h

    def _ray(self, x, y):
        V, Pm, eye, _w, _h = self._matrices()
        ww, wh = self.glfw.get_window_size(self.win)
        ndc = np.array([2 * x / ww - 1, 1 - 2 * y / wh, 1.0, 1.0])
        p = np.linalg.inv(Pm @ V) @ ndc
        d = p[:3] / p[3] - eye
        return eye, d / np.linalg.norm(d), -self.eye_dir

    # -- input -------------------------------------------------------------------------------
    def _pick(self, x, y):
        import open3d as o3d

        with self.lock:
            if self.latest is None or "auricle_skin" not in self.latest[0]:
                return None
            v = self.latest[0]["auricle_skin"][0]
        rs = o3d.t.geometry.RaycastingScene()
        rs.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(self.drawn["auricle_skin"][1].astype(np.uint32)))
        eye, d, _f = self._ray(x, y)
        t = float(rs.cast_rays(o3d.core.Tensor(np.concatenate([eye, d])[None].astype(np.float32)))["t_hit"].numpy()[0])
        return None if not np.isfinite(t) else eye + d * t

    def _button(self, win, button, action, mods):
        glfw = self.glfw
        x, y = glfw.get_cursor_pos(win)
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT and mods & glfw.MOD_CONTROL:
                self.mouse = ("pinch", x, y)
                p = self._pick(x, y) if self.P is not None else None
                if p is not None:
                    cx = self.P.cartilage_x()
                    self.P.grab(cx[int(np.argmin(np.linalg.norm(cx - p, axis=1)))])
                    self.drag_plane = (p, -self.eye_dir)
                    self.grabbing = True
            elif button == glfw.MOUSE_BUTTON_LEFT:
                self.mouse = ("orbit", x, y)
            elif button in (glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE):
                self.mouse = ("pan", x, y)
        elif action == glfw.RELEASE:
            if self.grabbing and self.P is not None:
                self.P.release()
            self.grabbing = False
            self.mouse = None

    def _motion(self, win, x, y):
        if self.mouse is None:
            return
        kind, x0, y0 = self.mouse
        dx, dy = x - x0, y - y0
        if kind == "pinch":
            if self.grabbing and self.P is not None:
                eye, d, _f = self._ray(x, y)
                c, n = self.drag_plane
                self.P.drag(eye + d * ((c - eye) @ n) / (d @ n) - c)
            return
        self.mouse = (kind, x, y)
        right = np.cross(self.up, self.eye_dir)
        right /= np.linalg.norm(right)
        if kind == "orbit":
            self.eye_dir = _rotate(self.eye_dir, self.up, -0.006 * dx)
            e = _rotate(self.eye_dir, right, -0.006 * dy)
            if abs(e @ self.up) < 0.98:
                self.eye_dir = e
        else:
            u = np.cross(self.eye_dir, right)
            self.centre = self.centre + 0.0012 * self.dist * (-dx * right + dy * u)

    def _scroll(self, win, sx, sy):
        self.dist *= 0.9 ** sy

    def _key(self, win, key, sc, action, mods):
        glfw = self.glfw
        if action != glfw.PRESS:
            return
        toggles = {glfw.KEY_F: "fans", glfw.KEY_B: "bone", glfw.KEY_H: "envelope_skin", glfw.KEY_C: "cartilage",
                   glfw.KEY_M: "auricular_muscles", glfw.KEY_T: "temporalis_muscle"}
        for k_, name, _label in LEGEND:  # number keys (and the letters) from the legend
            if k_.isdigit():
                toggles[getattr(glfw, f"KEY_{k_}")] = name
        if key == glfw.KEY_G:
            self.legend.visible = not self.legend.visible
            return
        if key == glfw.KEY_L:  # peel the next layer (wraps back to the skin)
            self.peel = (getattr(self, "peel", 0) + 1) % (len(PEEL) + 1)
            for i, name in enumerate(PEEL):
                self.show[name] = i >= self.peel
                if name in self.meshes:
                    self.meshes[name].visible = self.show[name]
            self._legend_dirty = True
            return
        if key in toggles:
            name = toggles[key]
            self.show[name] = not self.show.get(name, True)
            self._legend_dirty = True
            if name in self.meshes:
                self.meshes[name].visible = self.show[name]
        elif key == glfw.KEY_R and self.P is not None:
            self.P.reset()
        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
            self.base_k *= 1.5 if key == glfw.KEY_RIGHT_BRACKET else 1 / 1.5
            threading.Thread(target=self._build, args=(self.base_k,), daemon=True).start()
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    # -- drawing -----------------------------------------------------------------------------
    def _upload(self):
        with self.lock:
            if self.latest is None or self.seq == self.drawn_seq:
                return
            out, f, dist, held = self.latest
            self.drawn_seq = self.seq
        for name, (v, n) in ((k, val) for k, val in out.items() if k != "fans"):
            if name not in self.meshes:  # made once the model is built (the bellies come from its fans)
                self.meshes[name] = GLMesh(v, self.bellies.tris, COLOURS[name], dynamic=True)
                self.meshes[name].visible = self.show.get(name, True)
            self.meshes[name].update(v, n)
        k = len(out["fans"]) // 2
        if "fans" not in self.meshes:
            self.meshes["fans"] = GLMesh(out["fans"], np.stack([np.arange(k), np.arange(k) + k], 1), (1.0, 0.55, 0.1),
                                         dynamic=True, lines=True)
        else:
            self.meshes["fans"].update(out["fans"])
        self.meshes["fans"].visible = self.show["fans"]
        self.readout = (f, dist, held)

    def _update_legend(self):
        rows = []
        for key, name, label in LEGEND:
            if name not in self.meshes and name != "fans":
                continue
            rgb = COLOURS.get(name, (255, 140, 30) if name == "fans" else (160, 160, 160))
            rows.append((key, label, rgb, bool(self.show.get(name, True))))
        footer = ["L peel one layer   R reset   G hide this legend",
                  "Ctrl+drag pull   drag orbit   right-drag pan   wheel zoom"]
        self.legend.update(rows, footer)
        self._legend_dirty = False

    def _peel_label(self):
        k = getattr(self, "peel", 0)
        names = {"envelope_skin": "skin (1)", "subcutaneous_fat": "fat (2)", "tpf": "temporoparietal fascia / SMAS (3)",
                 "auricular_muscles": "auricular muscles (3)", "loose_areolar": "loose areolar plane (4)",
                 "temporalis_fascia": "temporalis fascia", "temporalis_muscle": "temporalis"}
        return "outermost: " + (names[PEEL[k]] if k < len(PEEL) else "periosteum / bone")

    def _demo_step(self, t_start):
        """Scripted pinch of the helix, pulled up and back 10 mm over 3 s; stats printed; then a
        screenshot and quit. Returns False when done."""
        P = self.P
        if P is None:
            return True
        if not hasattr(self, "_demo"):
            vol = P.vol
            S, A = np.asarray(vol.frame.superior, float), np.asarray(vol.frame.anterior, float)
            P.grab(P.model.helix_top())
            d = (S - A) / np.linalg.norm(S - A)
            import os

            pull = os.environ.get("FIVRS_PULL")  # "a,s,l": pull this way instead (anterior, superior, lateral)
            if pull:
                L_ = np.asarray(vol.frame.lateral, float)
                a_, s_, l_ = (float(x) for x in pull.split(","))
                d = a_ * A + s_ * S + l_ * L_
                d /= np.linalg.norm(d)
            self._demo = {"t0": time.perf_counter(), "d": d, "log": []}
        D = self._demo
        t = time.perf_counter() - D["t0"]
        P.drag(D["d"] * float(__import__("os").environ.get("FIVRS_PULL_MM", 10.0)) * min(1.0, t / 3.0))
        D["log"].append((t, self.phys_ms, self.place_ms))
        if t > 4.0:
            ms = np.array([r[1] for r in D["log"] if r[0] < 3.0])
            pl = np.array([r[2] for r in D["log"] if r[0] < 3.0])
            f, dist, _h = self.readout
            print(f"demo: while pulling, physics {np.mean(ms):.1f} ms (max {np.max(ms):.1f}), placing {np.mean(pl):.1f} ms, "
                  f"drawing {len(D['log']) / t:.0f} fps; at {dist:.1f} mm: {np.linalg.norm(f):.2f} N", flush=True)
            return False
        return True

    def screenshot(self, path):
        GL = self.GL
        w, h = self.glfw.get_framebuffer_size(self.win)
        data = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        from PIL import Image

        Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).save(path)

    def run(self, demo=None):
        glfw, GL = self.glfw, self.GL
        self.readout = (np.zeros(3), 0.0, False)
        last_title = 0.0
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            if demo and not self._demo_step(None):
                self._upload()
                self.screenshot(demo)
                print(f"wrote {demo}", flush=True)
                break
            self._upload()
            V, Pm, eye, w, h = self._matrices()
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.06, 0.06, 0.08, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glUseProgram(self.prog)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.prog, "mvp"), 1, True, (Pm @ V).astype(np.float32))
            GL.glUniform3fv(GL.glGetUniformLocation(self.prog, "view_dir"), 1, (-self.eye_dir).astype(np.float32))
            for m in self.meshes.values():
                m.draw(self.prog)
            if getattr(self, "_legend_dirty", True):
                self._update_legend()
            ww, _wh = glfw.get_window_size(self.win)
            self.legend.draw(w, h, scale=w / max(ww, 1))
            glfw.swap_buffers(self.win)
            self.frames += 1
            now = time.perf_counter()
            gap = now - getattr(self, "_last_frame", now)
            self._last_frame = now
            if self.P is None:
                self._build_gap = max(getattr(self, "_build_gap", 0.0), gap)
            elif getattr(self, "_build_gap", None) is not None:
                print(f"window: longest frame while the model was building {1000 * self._build_gap:.0f} ms", flush=True)
                self._build_gap = None
            if now - self.t_fps > 0.5:
                self.fps = self.frames / (now - self.t_fps)
                self.phys_hz = self.phys_n / (now - self.t_fps)
                self.frames, self.phys_n, self.t_fps = 0, 0, now
            if now - last_title > 0.2:
                last_title = now
                f, dist, held = self.readout
                slip = " (fingers slipping)" if held and getattr(self.P, "slipping", False) else ""
                state = f"PINCHED hand moved {dist:4.1f} mm, force {np.linalg.norm(f):5.2f} N{slip}" if held else "free"
                glfw.set_window_title(self.win, f"Auricle pull | {state} | physics {self.phys_ms:4.1f} ms "
                                                f"({self.phys_hz:3.0f}/s) + placing {self.place_ms:4.1f} ms on {getattr(self, 'place_on', '-')} | "
                                                f"drawing {self.fps:3.0f} fps | {self.status} | "
                                                f"{self._peel_label()} | layers: see the legend (G)")
        self.generation += 1  # stop the physics thread
        glfw.terminate()


def _rotate(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * (axis @ v) * (1 - np.cos(angle))


# Models the app can load (the picker at start). Times: physics per frame while pulling, mesh sweep
# (REVIEW 6f). Add new ones here (e.g. three layers with head-skin physics) as they become ready.
MODELS = [
    {"key": "shell_1.5_lobule", "name": "One shell, 1.5 mm, with lobule  (the current demo)", "layers": 1, "edge": 1.5,
     "lobule": True, "note": "skin / cartilage / skin as one shell; the lobule is skin / fat / skin; ~9 ms physics, 60 fps"},
    {"key": "shell_1.5", "name": "One shell, 1.5 mm, no lobule  (the 11 Nov demo)", "layers": 1, "edge": 1.5,
     "lobule": False, "note": "physics on the cartilage only; the lobule is drawn hanging from it"},
    {"key": "shell_2", "name": "One shell, 2 mm, no lobule", "layers": 1, "edge": 2.0, "lobule": False,
     "note": "coarser and faster (~5 ms); drawn skin ~0.35 mm further from the reference"},
    {"key": "layers3_2", "name": "Three layers, 2 mm, with lobule", "layers": 3, "edge": 2.0, "lobule": True,
     "note": "skin / cartilage (fat in the lobule) / skin bonded face to face, each skin drawn from its own layer; ~31 ms physics"},
    {"key": "layers3_1.5", "name": "Three layers, 1.5 mm, with lobule", "layers": 3, "edge": 1.5, "lobule": True,
     "note": "closest to the 1 mm reference; ~64 ms physics, ~15 updates a second"},
]
MODEL_NAME = MODELS[0]["name"]
LAST_MODEL = pathlib.Path.home() / ".cache" / "fivrs-auricle-pull" / "last_model.json"


def pick_model():
    """A small window before the viewer: choose the model and how the skin is drawn. Returns the
    chosen MODELS entry plus {"smooth": bool}, or None if closed. Remembers the last choice."""
    import json
    import tkinter as tk

    last = {}
    try:
        last = json.loads(LAST_MODEL.read_text())
    except Exception:
        pass
    keys = [m["key"] for m in MODELS]
    root = tk.Tk(className="fivrs-soft-ear")
    root.title("FIVRS Soft Ear: choose a model")
    root.resizable(False, False)
    pick = tk.StringVar(value=last.get("key") if last.get("key") in keys else keys[0])
    smooth = tk.BooleanVar(value=last.get("smooth", True))
    tk.Label(root, text="Which auricle model?", font=("Sans", 13, "bold")).pack(anchor="w", padx=16, pady=(14, 6))
    for m in MODELS:
        f = tk.Frame(root)
        f.pack(anchor="w", fill="x", padx=16, pady=3)
        tk.Radiobutton(f, text=m["name"], variable=pick, value=m["key"], font=("Sans", 11)).pack(anchor="w")
        tk.Label(f, text=m["note"], fg="#666", font=("Sans", 9)).pack(anchor="w", padx=26)
    tk.Checkbutton(root, text="Smooth skin drawing (blend nearby physics points; off: nearest triangle)",
                   variable=smooth).pack(anchor="w", padx=16, pady=(10, 4))
    out = {}

    def start(*_):
        out.update(next(m for m in MODELS if m["key"] == pick.get()), smooth=bool(smooth.get()))
        root.destroy()

    b = tk.Frame(root)
    b.pack(fill="x", padx=16, pady=(8, 14))
    tk.Button(b, text="Start", width=10, default="active", command=start).pack(side="right")
    tk.Button(b, text="Cancel", width=10, command=root.destroy).pack(side="right", padx=8)
    root.bind("<Return>", start)
    root.bind("<Escape>", lambda *_: root.destroy())
    root.mainloop()
    if not out:
        return None
    try:
        LAST_MODEL.parent.mkdir(parents=True, exist_ok=True)
        LAST_MODEL.write_text(json.dumps({"key": out["key"], "smooth": out["smooth"]}))
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", help="run a scripted pull offscreen and write a GIF to this folder")
    ap.add_argument("--demo", help="open the window, pull the helix by script, print timings, save a screenshot here")
    ap.add_argument("--layers", type=int, choices=(1, 3), default=LAYERS,
                    help="1: one equivalent shell (fast, default); 3: skin / cartilage / skin bonded face to face")
    ap.add_argument("--cpu-place", action="store_true", help="place the drawn meshes with numpy instead of on the GPU")
    ap.add_argument("--edge", type=float, default=EDGE_MM, help="physics mid-surface edge, mm")
    ap.add_argument("--no-lobule", action="store_true", help="physics on the cartilage only (the lobule hangs from it)")
    ap.add_argument("--rigid-skin", action="store_true", help="draw the skin on its nearest physics triangle (old way)")
    ap.add_argument("--peel", type=int, default=0, help="start with this many of the head's layers peeled off (L)")
    ap.add_argument("--pick", action="store_true", help="choose the model in a small window first (the desktop icon)")
    a = ap.parse_args()
    g = globals()
    g["LAYERS"], g["EDGE_MM"], g["SMOOTH_SKIN"], g["LOBULE"] = a.layers, a.edge, not a.rigid_skin, not a.no_lobule
    g["MODEL_NAME"] = f"{a.layers} layer(s), {a.edge:g} mm"
    g["GPU_PLACE"] = not a.cpu_place
    g["PEEL_START"] = max(0, min(a.peel, len(PEEL)))
    if a.pick:
        m = pick_model()
        if m is None:
            return
        g["LAYERS"], g["EDGE_MM"], g["SMOOTH_SKIN"], g["MODEL_NAME"] = m["layers"], m["edge"], m["smooth"], m["name"].split("  (")[0]
        g["LOBULE"] = m["lobule"]
        print(f"model: {m['name']} (smooth skin {m['smooth']})", flush=True)
    if a.selftest:
        selftest(a.selftest)
        return
    Viewer().run(demo=a.demo)


if __name__ == "__main__":
    main()
