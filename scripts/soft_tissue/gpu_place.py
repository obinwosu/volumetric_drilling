"""Drawn-mesh placement and normals on the GPU (NVIDIA Warp), for the pull viewer.

The physics (the CPU solver) gives the mid-surface; every frame the dense drawn meshes are put on it
and their normals recomputed. On the CPU (numpy / scipy, auricle_pull_viewer.Embedding,
HeadSkinFollow, normals) that is ~21 ms for the skin, cartilage and head skin (165k vertices); here
~0.3 ms including the copy back for OpenGL (REVIEW 6f). Same maths:

  embedded mesh   p = sum_k bary_k x_k + R(triangle) local, R = (edge, normal, normal x edge)
                  (``Embedding``), or p = sum_j w_j (x_j + Q_j arm_j) over nearby physics vertices
                  with Q_j each vertex's rotation (``BlendEmbedding``, the default)
  head skin       p = rest + sum_k w_k (x_j_k - x_rest_j_k)  (drawing only, no forces)
  normals         area-weighted sum of each vertex's triangle normals, normalised

    GPUPlacer.create(...)  returns None when Warp or CUDA is missing (the viewer then stays on the CPU)
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
except Exception:  # pragma: no cover - no Warp
    wp = None


if wp is not None:

    @wp.kernel
    def _place(x: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.vec3i), tid: wp.array(dtype=int),
               bary: wp.array(dtype=wp.vec3), local: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        t = tris[tid[i]]
        a = x[t[0]]
        b = x[t[1]]
        c = x[t[2]]
        e1 = wp.normalize(b - a)
        n = wp.normalize(wp.cross(b - a, c - a))
        w = bary[i]
        q = local[i]
        out[i] = w[0] * a + w[1] * b + w[2] * c + q[0] * e1 + q[1] * n + q[2] * wp.cross(n, e1)

    @wp.func
    def _frame(x: wp.array(dtype=wp.vec3), t: wp.vec3i):
        a = x[t[0]]
        b = x[t[1]]
        c = x[t[2]]
        e1 = wp.normalize(b - a)
        n = wp.normalize(wp.cross(b - a, c - a))
        return wp.matrix_from_cols(e1, n, wp.cross(n, e1))

    @wp.kernel
    def _vertex_rot_sum(x: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.vec3i), R0: wp.array(dtype=wp.mat33),
                        acc: wp.array(dtype=wp.mat33)):
        f = wp.tid()
        t = tris[f]
        Q = _frame(x, t) * wp.transpose(R0[f])  # this triangle's rest-to-now rotation
        wp.atomic_add(acc, t[0], Q)
        wp.atomic_add(acc, t[1], Q)
        wp.atomic_add(acc, t[2], Q)

    @wp.kernel
    def _polar(acc: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
        i = wp.tid()
        U = wp.mat33()
        s = wp.vec3()
        V = wp.mat33()
        wp.svd3(acc[i], U, s, V)
        R = U * wp.transpose(V)
        if wp.determinant(R) < 0.0:
            U = wp.matrix_from_cols(wp.vec3(U[0, 0], U[1, 0], U[2, 0]), wp.vec3(U[0, 1], U[1, 1], U[2, 1]),
                                    -wp.vec3(U[0, 2], U[1, 2], U[2, 2]))
            R = U * wp.transpose(V)
        out[i] = R

    @wp.kernel
    def _blend(x: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.mat33), j: wp.array2d(dtype=int),
               w: wp.array2d(dtype=float), arm: wp.array2d(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        p = wp.vec3()
        for k in range(j.shape[1]):
            v = j[i, k]
            p += w[i, k] * (x[v] + rot[v] * arm[i, k])
        out[i] = p

    @wp.kernel
    def _follow(x: wp.array(dtype=wp.vec3), x_rest: wp.array(dtype=wp.vec3), rest: wp.array(dtype=wp.vec3),
                near: wp.array(dtype=int), j: wp.array2d(dtype=int), w: wp.array2d(dtype=float),
                out_dir: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        s = wp.vec3()
        for k in range(j.shape[1]):
            s += w[i, k] * (x[j[i, k]] - x_rest[j[i, k]])
        od = out_dir[i]  # zero: no limit
        inward = wp.dot(s, od)
        if inward < 0.0:
            s = s - inward * od  # glide over the deep layers or lift off; never sink into them
        out[near[i]] = rest[near[i]] + s

    @wp.kernel
    def _face_normals(v: wp.array(dtype=wp.vec3), tris: wp.array(dtype=wp.vec3i), acc: wp.array(dtype=wp.vec3)):
        f = wp.tid()
        t = tris[f]
        n = wp.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]])
        wp.atomic_add(acc, t[0], n)
        wp.atomic_add(acc, t[1], n)
        wp.atomic_add(acc, t[2], n)

    @wp.kernel
    def _normalise(acc: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        a = acc[i]
        L = wp.length(a)
        out[i] = wp.vec3() if L < 1e-12 else a / L


def _f32(a):
    return np.ascontiguousarray(a, np.float32)


class GPUPlacer:
    """Places the embedded meshes (``Embedding``) and the head skin (``HeadSkinFollow``) on the GPU."""

    def __init__(self, mid_rest, mid_tris, embeddings, drawn_tris, head=None, head_name="envelope_skin",
                 device="cuda:0"):
        self.dev = device
        d = device
        self.x = wp.zeros(len(mid_rest), dtype=wp.vec3, device=d)
        self.x_rest = wp.array(_f32(mid_rest), dtype=wp.vec3, device=d)
        self.mid_tris = wp.array(np.asarray(mid_tris, np.int32), dtype=wp.vec3i, device=d)
        self.meshes = {}
        self.blend = False
        for name, e in embeddings.items():
            if hasattr(e, "arm"):  # BlendEmbedding
                self.blend = True
                g = self._buffers(drawn_tris[name], len(e.w))
                g.update(j=wp.array(e.j.astype(np.int32), dtype=int, device=d), w=wp.array(_f32(e.w), dtype=float, device=d),
                         arm=wp.array(_f32(e.arm), dtype=wp.vec3, device=d))
                self.R0 = wp.array(_f32(e.R0), dtype=wp.mat33, device=d)
            else:
                g = self._buffers(drawn_tris[name], len(e.tid))
                g.update(tid=wp.array(e.tid.astype(np.int32), dtype=int, device=d),
                         bary=wp.array(_f32(e.bary), dtype=wp.vec3, device=d),
                         local=wp.array(_f32(e.local), dtype=wp.vec3, device=d))
            self.meshes[name] = g
        if self.blend:
            self.rot_acc = wp.zeros(len(mid_rest), dtype=wp.mat33, device=d)
            self.rot = wp.zeros(len(mid_rest), dtype=wp.mat33, device=d)
        # meshes carried by the auricle's base (HeadSkinFollow): one, or a {name: follow} dict
        heads = head if isinstance(head, dict) else ({head_name: head} if head is not None else {})
        for name, hf in heads.items():
            g = self._buffers(drawn_tris[name], len(hf.rest))
            g.update(rest=wp.array(_f32(hf.rest), dtype=wp.vec3, device=d),
                     near=wp.array(hf.near.astype(np.int32), dtype=int, device=d),
                     j=wp.array(hf.j.astype(np.int32), dtype=int, device=d),
                     w=wp.array(_f32(hf.w), dtype=float, device=d),
                     out_dir=wp.array(_f32(hf.out_dir) if getattr(hf, "out_dir", None) is not None
                                      else np.zeros((len(hf.near), 3), np.float32), dtype=wp.vec3, device=d))
            wp.copy(g["v"], g["rest"])  # vertices far from the base never move
            self.meshes[name] = g

    def _buffers(self, tris, nv):
        d = self.dev
        return {"tris": wp.array(np.asarray(tris, np.int32), dtype=wp.vec3i, device=d), "nf": len(tris), "nv": nv,
                "v": wp.zeros(nv, dtype=wp.vec3, device=d), "acc": wp.zeros(nv, dtype=wp.vec3, device=d),
                "n": wp.zeros(nv, dtype=wp.vec3, device=d),
                "host_v": wp.zeros(nv, dtype=wp.vec3, device="cpu", pinned=True),
                "host_n": wp.zeros(nv, dtype=wp.vec3, device="cpu", pinned=True)}

    @classmethod
    def create(cls, *args, log=print, **kw):
        if wp is None:
            log("drawn meshes placed on the CPU (no Warp)")
            return None
        try:
            wp.config.quiet = True
            wp.init()
            if not wp.is_cuda_available():
                log("drawn meshes placed on the CPU (no CUDA device)")
                return None
            return cls(*args, **kw)
        except Exception as e:  # pragma: no cover - driver trouble
            log(f"drawn meshes placed on the CPU (Warp failed: {e})")
            return None

    def place(self, x_mid, names):
        """{name: (vertices, normals)} as new float32 numpy arrays, for each name in ``names``."""
        d = self.dev
        self.x.assign(_f32(x_mid))
        if self.blend:
            self.rot_acc.zero_()
            wp.launch(_vertex_rot_sum, self.mid_tris.shape[0], [self.x, self.mid_tris, self.R0, self.rot_acc], device=d)
            wp.launch(_polar, self.x.shape[0], [self.rot_acc, self.rot], device=d)
        for name in names:
            g = self.meshes[name]
            if "arm" in g:
                wp.launch(_blend, g["nv"], [self.x, self.rot, g["j"], g["w"], g["arm"], g["v"]], device=d)
            elif "tid" in g:
                wp.launch(_place, g["nv"], [self.x, self.mid_tris, g["tid"], g["bary"], g["local"], g["v"]], device=d)
            else:
                wp.launch(_follow, len(g["near"]), [self.x, self.x_rest, g["rest"], g["near"], g["j"], g["w"], g["out_dir"], g["v"]],
                          device=d)
            g["acc"].zero_()
            wp.launch(_face_normals, g["nf"], [g["v"], g["tris"], g["acc"]], device=d)
            wp.launch(_normalise, g["nv"], [g["acc"], g["n"]], device=d)
            wp.copy(g["host_v"], g["v"])
            wp.copy(g["host_n"], g["n"])
        wp.synchronize_device(d)
        # copies: the window may still be drawing the previous frame's arrays
        return {name: (self.meshes[name]["host_v"].numpy().copy(), self.meshes[name]["host_n"].numpy().copy())
                for name in names}
