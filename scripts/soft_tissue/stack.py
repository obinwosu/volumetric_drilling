"""Layered tissue stack for the slow reference solver.

Several sheets (skin, fascia, periosteum, ...) on the same in-plane mesh,
listed outside to inside. Neighbouring layers are joined vertex to vertex by
breakable bonds (one bilinear law on the full separation, per plane). The
innermost layer is attached to a rigid substrate (bone or cartilage) by bonds
with separate opening (peel) and sliding (shear) laws. An optional rigid
elevator blade (wedge_test.Blade) can be driven under the stack.

Layers share one mid-surface by default: bonded layers sit on top of each other
with zero rest separation. That keeps the bonds correct when a bonded flap turns
(no rest offset to rotate), at the price that a bonded stack bends like the sum
of its layers (no extra stiffness from layers offset from the stack's centre).
That is a lower bound on a real laminate's bending stiffness.

``offset=True`` gives each layer its real height instead (mid-surfaces apart by
half of each neighbour's thickness) and joins neighbours face to face: each
interior edge of a layer is bonded to the same edge of the next one, at points
pushed out from the two edge midpoints along the edges' mean triangle normals by
half of each layer's thickness (the faces where they touch). The offset turns with the
triangles, so a bonded flap still turns freely, and a fully bonded stack bends
as a laminate (transformed section); a weak bond lets the faces slip
(partial interaction). A plane's law is a ``pm.BondLaw`` (one law on the full
separation) or a ``wedge_test.Interface`` (opening along the inner triangle's
normal, sliding across it; compression resisted at the opening's stiffness):
slippery planes such as perichondrium on cartilage. Checked by
``laminate_test.py``.

Duck-types ``peel_model.Sheet`` for ``peel_model.solve``.

Units: mm, N, MPa.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import peel_model as pm  # noqa: E402
import wedge_test as wt  # noqa: E402


def vertex_normals(v, t):
    """Area-weighted unit vertex normals (triangle winding gives the side)."""
    v, t = np.asarray(v, float), np.asarray(t)
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    n = np.zeros_like(v)
    for c in range(3):
        np.add.at(n, t[:, c], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


class Stack:
    def __init__(self, rest, tris, layers, between, attachment, attach_mask=None, pair_masks=None, floor_z=0.0,
                 names=None, blade: wt.Blade | None = None, extra_pairs=(), offset=False, bending="hinge",
                 fibres=None, curved=False, ref_layer=None, anchors=None):
        """
        rest, tris   in-plane mesh shared by every layer (rest z = floor_z)
        layers       list of pm.SkinParams, outside to inside
        between      list of pm.BondLaw, one per plane between neighbouring layers
        attachment   wt.Interface (opening, sliding) of the innermost layer to the substrate
        attach_mask  (n,) bool: which vertices are attached to the substrate (default all)
        pair_masks   list of (n,) bool per plane: which vertices are bonded (default all)
        extra_pairs  (layer, local_a, local_b, law) bonds inside one layer, e.g. welding the two
                     sides of a mesh split back together in a layer the incision did not cut
        offset       give the layers their real heights and bond them face to face (see the module
                     docstring); ``between`` may then hold wt.Interface (mixed) laws too
        bending      "hinge" or "shell", or one per layer
        fibres       ``fibres.Fibres`` (vertex ids in the stack's numbering): tension-only fans
        curved       ``rest`` is a curved surface (with ``offset``): the layers sit along its vertex
                     normals (triangle winding: the outside), each a curved shell holding its shape
        ref_layer    with ``curved``: the layer whose mid-surface ``rest`` is (default the innermost)
        anchors      (vertex ids, pm.BondLaw): those vertices tied to their rest positions by one law
                     on the full separation, per area (a compliant base, e.g. the auricle on the head);
                     instead of a substrate ``attachment``
        """
        rest = np.asarray(rest, dtype=np.float64)
        self.n = n = len(rest)
        self.L = L = len(layers)
        assert len(between) == L - 1
        self.names = names or [f"layer{k}" for k in range(L)]
        bends = [bending] * L if isinstance(bending, str) else list(bending)
        self.offset = offset
        h = np.array([p.thickness_mm for p in layers])
        # mid-surface height of each layer: the innermost on floor_z, the others stacked on it
        self.height = np.array([h[k + 1:].sum() - 0.5 * h[-1] + 0.5 * h[k] for k in range(L)]) if offset else np.zeros(L)
        if curved and ref_layer is not None:
            self.height = self.height - self.height[ref_layer]
        self.curved = curved
        normal = vertex_normals(rest, tris) if curved else np.array([0.0, 0.0, 1.0])
        self.sheets = []
        for k, p in enumerate(layers):
            r = rest + self.height[k] * normal
            self.sheets.append(pm.Sheet(r, tris, np.zeros(0, dtype=np.int64), p, floor_z=floor_z, bending=bends[k],
                                        curved=curved))
        self.floor_z = floor_z
        self.rest = np.concatenate([s.rest for s in self.sheets])
        self.vert_area = np.tile(self.sheets[0].vert_area, L)
        self.layer = np.repeat(np.arange(L), n)
        # layer-to-layer bonds
        a, b, w, law = [], [], [], []
        for k in range(L - 1 if not offset else 0):
            mask = np.ones(n, dtype=bool) if pair_masks is None else np.asarray(pair_masks[k], dtype=bool)
            v = np.flatnonzero(mask)
            a.append(k * n + v)
            b.append((k + 1) * n + v)
            w.append(self.sheets[k].vert_area[v])
            law.append(np.full(len(v), k))
        self.laws = list(between)
        for layer, la, lb, wlaw in extra_pairs:
            la, lb = np.asarray(la), np.asarray(lb)
            a.append(layer * n + la)
            b.append(layer * n + lb)
            w.append(self.sheets[layer].vert_area[la])
            law.append(np.full(len(la), len(self.laws)))
            self.laws.append(wlaw)
        cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt))
        self.pair_a, self.pair_b = cat(a, np.int64), cat(b, np.int64)
        self.pair_w, self.pair_law = cat(w, np.float64), cat(law, np.int64)
        self.pair_hist = np.zeros(len(self.pair_a))
        self.pair_broken = np.zeros(len(self.pair_a), dtype=bool)
        self._setup_faces(between, pair_masks, h)
        # substrate attachment of the innermost layer
        self.iface = attachment
        mask = np.ones(n, dtype=bool) if attach_mask is None else np.asarray(attach_mask, dtype=bool)
        self.ids = (L - 1) * n + np.flatnonzero(mask)
        if attachment is None:  # a free stack, attached to nothing
            self.ids = np.zeros(0, dtype=np.int64)
        self.anchor_law = None
        if anchors is not None:
            assert attachment is None, "anchors replace the substrate attachment"
            self.ids = np.asarray(anchors[0], dtype=np.int64)
            self.anchor_law = anchors[1]
        self.weight = self.vert_area[self.ids]
        self.hist_n = np.zeros(len(self.ids))
        self.hist_s = np.zeros(len(self.ids))
        self.broken = np.zeros(len(self.ids), dtype=bool)
        # optional blade, acting on every layer
        self.blade = blade
        self.tip = 0.0
        self.fibres = fibres
        tris_all = np.concatenate([s.tris + k * n for k, s in enumerate(self.sheets)])
        hinges_all = np.concatenate([s.hinges + k * n for k, s in enumerate(self.sheets)])
        self.tris_all, self.hinges_all = tris_all, hinges_all
        stencils = [tris_all, hinges_all]
        stencils += [s.shell_stencil + k * n for k, s in enumerate(self.sheets) if s.bending == "shell"]
        if len(self.face_e):
            stencils.append(self.face_v)
        if fibres is not None and len(fibres.stencils()):
            stencils.append(fibres.stencils())
        if len(self.pair_a):
            stencils.append(np.stack([self.pair_a, self.pair_b], axis=1))
        self.color_groups = pm._colors(L * n, stencils)

    def _setup_faces(self, between, pair_masks, h):
        """Face-to-face bonds of an offset stack, one per interior mesh edge: edge e of layer k to edge e of k + 1.

        Each bonded point sits at the edge's midpoint, pushed along the mean of its (one or two)
        triangles' normals: a centred normal, so a uniformly bent laminate is matched exactly to first
        order (a triangle's own normal belongs to the middle of its span, not its centroid, and bonding
        there would lock a stiff bond)."""
        n, L = self.n, self.L
        tris = self.sheets[0].tris
        # edges with their one or two triangles
        e = np.concatenate([tris[:, [1, 2]], tris[:, [2, 0]], tris[:, [0, 1]]])
        owner = np.tile(np.arange(len(tris)), 3)
        key = np.sort(e, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        key, owner = key[order], owner[order]
        first = np.ones(len(key), dtype=bool)
        first[1:] = (key[1:] != key[:-1]).any(axis=1)
        start = np.flatnonzero(first)
        edges = key[start]
        t1 = owner[start]
        count = np.diff(np.append(start, len(key)))
        t2 = np.where(count == 2, owner[np.minimum(start + 1, len(owner) - 1)], -1)
        # Only interior edges are bonded: a boundary edge has one triangle, whose normal belongs to the
        # middle of that triangle, not to the edge, and a stiff bond would lock there. Each triangle's
        # area is shared among its interior edges, so the bonded area is still the whole area.
        inner = t2 >= 0
        edges, t1, t2 = edges[inner], t1[inner], t2[inner]
        area = self.sheets[0].area
        n_inner = np.bincount(np.concatenate([t1, t2]), minlength=len(tris)).astype(float)
        share = np.where(n_inner > 0, area / np.maximum(n_inner, 1.0), 0.0)
        e_area = share[t1] + share[t2]
        self.face_tris = tris  # local triangle corners (same in every layer)
        E, T, oa, ob, w, plane, law_n, law_s = [], [], [], [], [], [], [], []
        self.face_laws = []
        if self.offset:
            for k in range(L - 1):
                ok = np.ones(len(edges), dtype=bool)
                if pair_masks is not None:
                    ok = np.asarray(pair_masks[k], dtype=bool)[edges].all(axis=1)
                law = between[k]
                if isinstance(law, wt.Interface):
                    i_n, i_s = len(self.face_laws), len(self.face_laws) + 1
                    self.face_laws += [law.opening, law.sliding]
                else:
                    i_n, i_s = len(self.face_laws), -1
                    self.face_laws.append(law)
                m = int(ok.sum())
                E.append(np.stack([edges[ok, 0] + k * n, edges[ok, 1] + k * n, edges[ok, 0] + (k + 1) * n,
                                   edges[ok, 1] + (k + 1) * n], axis=1))
                T.append(np.stack([t1[ok], t2[ok]], axis=1))
                oa.append(np.full(m, -0.5 * h[k]))  # the outer layer's inner face
                ob.append(np.full(m, 0.5 * h[k + 1]))  # the inner layer's outer face
                w.append(e_area[ok])
                plane.append(np.full(m, k))
                law_n.append(np.full(m, i_n))
                law_s.append(np.full(m, i_s))
        cat = (lambda xs, dt, shape=(0,): np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dtype=dt))
        self.face_e = cat(E, np.int64, (0, 4))  # global: edge of layer k (2), the same edge of layer k + 1 (2)
        self.face_t = cat(T, np.int64, (0, 2))  # local triangles on the edge (second -1 on the boundary)
        self.face_oa, self.face_ob, self.face_w = cat(oa, np.float64), cat(ob, np.float64), cat(w, np.float64)
        self.face_plane, self.face_law_n, self.face_law_s = cat(plane, np.int64), cat(law_n, np.int64), cat(law_s, np.int64)
        F = len(self.face_e)
        self.face_hist_n = np.zeros(F)
        self.face_hist_s = np.zeros(F)
        self.face_broken = np.zeros(F, dtype=bool)
        # rest separation of each bond in its own frame (inner edge direction, its normal, their cross
        # product): zero for a flat stack; on a curved one the offset layers cannot meet every edge
        # exactly, and this keeps the rest shape free of force (and the bonds turn with the tissue)
        self.face_c0 = np.zeros((F, 3))
        if F and getattr(self, "curved", False):
            _pa, _pb, r0, nb = self.face_points(self.rest)
            t = self._edge_dir(self.rest)[0]
            self.face_c0 = np.stack([np.einsum("ij,ij->i", r0, t), np.einsum("ij,ij->i", r0, nb),
                                     np.einsum("ij,ij->i", r0, np.cross(t, nb))], axis=1)
        # each bond's stencil: the vertices of its triangles in both layers (8; a boundary edge repeats)
        if F:
            ta, tb = self.face_t[:, 0], np.where(self.face_t[:, 1] >= 0, self.face_t[:, 1], self.face_t[:, 0])
            la, lb = self.face_e[:, 0] // n, self.face_e[:, 2] // n
            sa = np.concatenate([tris[ta], tris[tb]], axis=1)
            st = []
            for row_a, row_e in zip(sa, self.face_e[:, :2]):
                u = [int(row_e[0] % n), int(row_e[1] % n)]
                for v_ in row_a:
                    if v_ not in u:
                        u.append(int(v_))
                while len(u) < 4:
                    u.append(u[-1])
                st.append(u[:4])
            st = np.array(st)
            self.face_v = np.concatenate([st + (la * n)[:, None], st + (lb * n)[:, None]], axis=1)
        else:
            self.face_v = np.zeros((0, 8), dtype=np.int64)

    # -- helpers --------------------------------------------------------------------
    def layer_slice(self, k):
        return slice(k * self.n, (k + 1) * self.n)

    def layer_vertices(self, k, local):
        return k * self.n + np.asarray(local)

    # -- energy ---------------------------------------------------------------------
    def energy(self, x, parts=False):
        total, grad = 0.0, np.zeros_like(x)
        out = {}
        for k, s in enumerate(self.sheets):
            e, g, p = s.energy(x[self.layer_slice(k)], parts=True)
            total += e
            grad[self.layer_slice(k)] = g
            for key, val in p.items():
                out[key] = out.get(key, 0.0) + val
        e, g = self._pair_energy(x)
        total += e
        grad += g
        out["pairs"] = e
        if len(self.face_e):
            e, g = self._face_energy(x)
            total += e
            grad += g
            out["faces"] = e
        e, g = self._attach_energy(x)
        total += e
        grad += g
        out["attachment"] = e
        if self.fibres is not None and len(self.fibres):
            e, g = self.fibres.energy(x)
            total += e
            grad += g
            out["fibres"] = e
        if self.blade is not None:
            e, g, _push = self._blade(x)
            total += e
            grad += g
            out["blade"] = e
        if parts:
            return total, grad, out
        return total, grad

    def _pair_energy(self, x):
        grad = np.zeros_like(x)
        if not len(self.pair_a):
            return 0.0, grad
        r = x[self.pair_a] - x[self.pair_b]
        d = np.linalg.norm(r, axis=1)
        energy = 0.0
        g = np.zeros_like(r)
        for k, law in enumerate(self.laws):
            sel = (self.pair_law == k) & ~self.pair_broken
            if not sel.any():
                continue
            psi, dpsi = wt._channel(d[sel], self.pair_hist[sel], law)
            m = np.maximum(self.pair_hist[sel], law.delta0_mm)
            inside = d[sel] <= m
            scale = np.where(inside, pm.traction(m, law) / m, dpsi / np.maximum(d[sel], 1e-12))
            energy += float(np.sum(self.pair_w[sel] * psi))
            g[sel] = (self.pair_w[sel] * scale)[:, None] * r[sel]
        np.add.at(grad, self.pair_a, g)
        np.add.at(grad, self.pair_b, -g)
        return energy, grad

    @staticmethod
    def _normals(x, t):
        e1, e2 = x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]
        N = np.cross(e1, e2)
        ln = np.linalg.norm(N, axis=1)
        return N / ln[:, None], ln, e1, e2

    def _edge_normals(self, x, layer_of):
        """Unit centred normal of every bond's edge in layer ``layer_of`` (per bond), and the face normal pieces."""
        n = self.n
        t = self.face_tris
        off = (layer_of * n)[:, None]
        t1, t2 = self.face_t[:, 0], self.face_t[:, 1]
        has2 = t2 >= 0
        n1, l1, a1, b1 = self._normals(x, t[t1] + off)
        n2, l2, a2, b2 = self._normals(x, t[np.maximum(t2, 0)] + off)
        Msum = n1 + np.where(has2[:, None], n2, 0.0)
        lm = np.linalg.norm(Msum, axis=1)
        return Msum / lm[:, None], lm, (n1, l1, a1, b1, t[t1] + off), (n2, l2, a2, b2, t[np.maximum(t2, 0)] + off), has2

    def _edge_dir(self, x):
        """Unit direction of each bond's inner edge, and its length."""
        e = x[self.face_e[:, 3]] - x[self.face_e[:, 2]]
        le = np.linalg.norm(e, axis=1)
        return e / le[:, None], le

    def face_points(self, x):
        """The two bonded points of every face bond, their separation r (less its rest separation,
        carried in the bond's frame), and the inner normal."""
        n = self.n
        la, lb = self.face_e[:, 0] // n, self.face_e[:, 2] // n
        na = self._edge_normals(x, la)[0]
        nb = self._edge_normals(x, lb)[0]
        pa = 0.5 * (x[self.face_e[:, 0]] + x[self.face_e[:, 1]]) + self.face_oa[:, None] * na
        pb = 0.5 * (x[self.face_e[:, 2]] + x[self.face_e[:, 3]]) + self.face_ob[:, None] * nb
        r = pa - pb
        c0 = getattr(self, "face_c0", None)
        if c0 is not None and np.any(c0):
            t = self._edge_dir(x)[0]
            r = r - (c0[:, :1] * t + c0[:, 1:2] * nb + c0[:, 2:] * np.cross(t, nb))
        return pa, pb, r, nb

    def _face_terms(self, x):
        """Per face bond: energy, dE/dr and dE/dn_b (the inner normal, mixed laws only)."""
        _pa, _pb, r, nb = self.face_points(x)
        F = len(r)
        psi = np.zeros(F)
        g_r = np.zeros((F, 3))
        g_n = np.zeros((F, 3))
        alive = ~self.face_broken
        for i, law in enumerate(self.face_laws):
            iso = alive & (self.face_law_n == i) & (self.face_law_s < 0)
            if iso.any():
                d = np.linalg.norm(r[iso], axis=1)
                p_, dp_ = wt._channel(d, self.face_hist_n[iso], law)
                psi[iso] = p_
                g_r[iso] = (dp_ / np.maximum(d, 1e-300))[:, None] * r[iso]
            mix = alive & (self.face_law_n == i) & (self.face_law_s >= 0)
            if mix.any():
                slide = self.face_laws[self.face_law_s[mix][0]]
                n_ = nb[mix]
                dn = np.einsum("ij,ij->i", r[mix], n_)
                sv = r[mix] - dn[:, None] * n_
                ds = np.linalg.norm(sv, axis=1)
                p_n, dp_n = wt._channel(np.maximum(dn, 0.0), self.face_hist_n[mix], law)
                k0 = law.k_bond
                p_n = np.where(dn < 0, 0.5 * k0 * dn**2, p_n)
                dp_n = np.where(dn < 0, k0 * dn, dp_n)
                p_s, dp_s = wt._channel(ds, self.face_hist_s[mix], slide)
                sc = dp_s / np.maximum(ds, 1e-300)
                psi[mix] = p_n + p_s
                g_r[mix] = dp_n[:, None] * n_ + sc[:, None] * sv
                g_n[mix] = dp_n[:, None] * r[mix] - (sc * dn)[:, None] * sv
        return psi, g_r, g_n

    def _face_energy(self, x):
        psi, g_r, g_n = self._face_terms(x)
        w = self.face_w[:, None]
        g_r, g_n = w * g_r, w * g_n
        grad = np.zeros_like(x)
        e = self.face_e
        for c in (0, 1):
            np.add.at(grad, e[:, c], 0.5 * g_r)
            np.add.at(grad, e[:, 2 + c], -0.5 * g_r)
        # through the normals: dE/dn_a = oa g_r, dE/dn_b = -ob g_r + g_n (- the rest frame's turn)
        n = self.n
        gn_b = -self.face_ob[:, None] * g_r + g_n
        c0 = self.face_c0
        if np.any(c0):
            t, le = self._edge_dir(x)
            nb = self._edge_normals(x, e[:, 2] // n)[0]
            gn_b = gn_b - (c0[:, 1:2] * g_r + c0[:, 2:] * np.cross(g_r, t))
            g_t = -(c0[:, :1] * g_r + c0[:, 2:] * np.cross(nb, g_r))
            g_e = (g_t - np.einsum("ij,ij->i", g_t, t)[:, None] * t) / le[:, None]
            np.add.at(grad, e[:, 3], g_e)
            np.add.at(grad, e[:, 2], -g_e)
        for lay, gn in ((e[:, 0] // n, self.face_oa[:, None] * g_r), (e[:, 2] // n, gn_b)):
            ne, lm, one, two, has2 = self._edge_normals(x, lay)
            gM = (gn - np.einsum("ij,ij->i", gn, ne)[:, None] * ne) / lm[:, None]  # dE / d(sum of unit normals)
            for (nf, ln, e1, e2, tv), use in ((one, np.ones(len(e), bool)), (two, has2)):
                q = (gM - np.einsum("ij,ij->i", gM, nf)[:, None] * nf) / ln[:, None]
                q = q * use[:, None]
                g1, g2 = np.cross(e2, q), np.cross(q, e1)
                np.add.at(grad, tv[:, 1], g1)
                np.add.at(grad, tv[:, 2], g2)
                np.add.at(grad, tv[:, 0], -g1 - g2)
        return float(np.sum(self.face_w * psi)), grad

    def _attach_energy(self, x):
        grad = np.zeros_like(x)
        if not len(self.ids):
            return 0.0, grad
        if self.anchor_law is not None:  # one law on the full separation from the rest position
            r = x[self.ids] - self.rest[self.ids]
            d = np.linalg.norm(r, axis=1)
            w = self.weight * ~self.broken
            psi, dpsi = wt._channel(d, self.hist_n, self.anchor_law)
            grad[self.ids] = (w * dpsi / np.maximum(d, 1e-300))[:, None] * r
            return float(np.sum(w * psi)), grad
        r = x[self.ids] - self.rest[self.ids]
        dn, ds = r[:, 2], np.linalg.norm(r[:, :2], axis=1)
        alive = ~self.broken
        psi_n, dpn = wt._channel(np.maximum(dn, 0.0), self.hist_n, self.iface.opening)
        psi_s, _dps = wt._channel(ds, self.hist_s, self.iface.sliding)
        w = self.weight * alive
        m = np.maximum(self.hist_s, self.iface.sliding.delta0_mm)
        inside = ds <= m
        scale_s = np.where(inside, pm.traction(m, self.iface.sliding) / m,
                           pm.traction(ds, self.iface.sliding) / np.maximum(ds, 1e-12))
        g = np.zeros_like(r)
        g[:, 2] = w * dpn * (dn > 0.0)
        g[:, :2] = (w * scale_s)[:, None] * r[:, :2]
        grad[self.ids] = g
        return float(np.sum(w * (psi_n + psi_s))), grad

    def _blade(self, x):
        b = self.blade
        k = b.contact_n_per_mm3 * self.vert_area
        rise, drise = b.rise(self.tip - x[:, 0])
        gap = x[:, 2] - self.floor_z - rise
        kg = np.where(gap < 0.0, k * gap, 0.0)
        grad = np.zeros_like(x)
        grad[:, 2] = kg
        grad[:, 0] = kg * drise
        return float(np.sum(0.5 * kg * gap)), grad, float(-np.sum(kg * drise))

    def push_force(self, x):
        return self._blade(x)[2] if self.blade is not None else 0.0

    # -- damage ---------------------------------------------------------------------
    def commit(self, x):
        if len(self.pair_a):
            d = np.linalg.norm(x[self.pair_a] - x[self.pair_b], axis=1)
            self.pair_hist = np.maximum(self.pair_hist, d)
            dc = np.array([L.delta_c_mm for L in self.laws])[self.pair_law]
            self.pair_broken |= self.pair_hist >= dc
        if len(self.face_e):
            _pa, _pb, r, nb = self.face_points(x)
            mixed = self.face_law_s >= 0
            dn = np.einsum("ij,ij->i", r, nb)
            open_ = np.where(mixed, np.maximum(dn, 0.0), np.linalg.norm(r, axis=1))
            self.face_hist_n = np.maximum(self.face_hist_n, open_)
            slide = np.linalg.norm(r - dn[:, None] * nb, axis=1)
            self.face_hist_s = np.where(mixed, np.maximum(self.face_hist_s, slide), 0.0)
            dc = np.array([L_.delta_c_mm for L_ in self.face_laws])
            self.face_broken |= self.face_hist_n >= dc[self.face_law_n]
            self.face_broken |= mixed & (self.face_hist_s >= dc[np.maximum(self.face_law_s, 0)])
        if len(self.ids) and self.anchor_law is not None:
            d = np.linalg.norm(x[self.ids] - self.rest[self.ids], axis=1)
            self.hist_n = np.maximum(self.hist_n, d)
            self.broken |= self.hist_n >= self.anchor_law.delta_c_mm
        elif len(self.ids):
            r = x[self.ids] - self.rest[self.ids]
            self.hist_n = np.maximum(self.hist_n, np.maximum(r[:, 2], 0.0))
            self.hist_s = np.maximum(self.hist_s, np.linalg.norm(r[:, :2], axis=1))
            self.broken |= (self.hist_n >= self.iface.opening.delta_c_mm) | (self.hist_s >= self.iface.sliding.delta_c_mm)

    def plane_broken_fraction(self, k, mask=None):
        """Fraction of plane k's bonds (between layer k and k+1) that are broken."""
        if self.offset:
            sel = np.flatnonzero(self.face_plane == k)
            if mask is not None:
                sel = sel[mask[self.face_e[sel, 0] - k * self.n]]
            return float(self.face_broken[sel].mean()) if len(sel) else 0.0
        sel = np.flatnonzero(self.pair_law == k)
        if mask is not None:
            sel = sel[mask[self.pair_a[sel] - k * self.n]]
        return float(self.pair_broken[sel].mean()) if len(sel) else 0.0

    def attach_broken_fraction(self, mask=None):
        sel = np.ones(len(self.ids), dtype=bool)
        if mask is not None:
            sel &= mask[self.ids - (self.L - 1) * self.n]
        return float(self.broken[sel].mean()) if sel.any() else 0.0


def from_table(stack_name, rest, tris, corner="nominal", **kw):
    """A Stack of one of tissue_params.STACKS on a given mesh."""
    import tissue_params as tp

    spec = tp.STACKS[stack_name]
    layers = [tp.skin_params(m, spec["attachment"], corner) for m in spec["layers"]]
    between = [tp.INTERFACES[i].opening.bond_law(corner) for i in spec["between"]]
    attachment = kw.pop("attachment", tp.wedge_interface(spec["attachment"], corner))
    return Stack(rest, tris, layers, between, attachment, names=spec["layers"], **kw)
