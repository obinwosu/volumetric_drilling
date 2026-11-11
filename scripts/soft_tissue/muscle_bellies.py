"""Thin muscle bellies drawn over the auricular fans (drawing only; the fans' fibres carry the force).

Each auricular muscle fan (``auricle_sim.place_fans``: fibres from an origin line on the head to an
insertion patch on the cartilage) is drawn as a thin muscle belly: a ruled surface from the origin
line to the insertion patch, laid in the loose areolar / temporoparietal plane, just under the
subcutaneous fat and superficial to the temporalis fascia (user, 2026-11-11), except near the ear,
where it rises to its insertion; given the fan's thickness (tissue_params.FANS), thickest over the
belly and thin at both attachments and toward its edges. The posterior one is two rounded fascicles.
Drawn only inside the scene (the CT cube); the fans' attachments beyond it still act in the physics. ``roof``: the underside of the subcutaneous fat (the
envelope's ``loose_areolar`` surface, its outer face).

As the ear moves, each belly point moves by its insertion's motion times how far along the muscle
it is (the origin stays, the insertion end goes with the cartilage).
"""

from __future__ import annotations

import numpy as np

N_ALONG = 28
RISE = 0.25  # last fraction of the length rises off the fascia to the insertion
CLEAR_MM = 0.1  # gap between the belly and the fat above it
FASCICLES = {"posterior": 2}  # 2-3 fascicles (posterior auricular muscle, cadaver reports)
FASCICLE_GAP = 0.12  # fraction of the width between fascicles
SKIN_CLEAR_MM = 0.5  # bellies stay this far under the head's skin
END_THICKNESS = 0.35  # at each attachment, as a fraction of the belly (as the atlas posterior auricular)


def _smooth(shift, sigma=(2.0, 3.0)):
    """A (k, n, 3) shift field smoothed across and along the belly."""
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(shift[..., c], sigma, mode="nearest") for c in range(3)], axis=-1)


def press_under(pts, layer, lateral, margin):
    """Points outside ``layer`` (a sheet; outward = its lateral-facing side), where the layer is straight
    below them, pressed back to ``margin`` mm under it. Others unchanged."""
    import trimesh

    q, d, tri = trimesh.proximity.closest_point(layer, pts)
    n = layer.face_normals[tri]
    n = n * np.where(n @ lateral < 0, -1.0, 1.0)[:, None]
    s = ((pts - q) * n).sum(1)
    under = np.abs(s) >= 0.9 * d - 0.3
    move = under & (s > -margin)
    out = pts.copy()
    out[move] = pts[move] - (s[move] + margin)[:, None] * n[move]
    return out


class Belly:
    def __init__(self, name, fibres, mid_rest, thickness_mm, fascia=None, lateral=None, n_along=N_ALONG, skin=None):
        """``fibres``: this fan's fibres (ordered along the origin line, as ``fibres.fan`` makes
        them); ``mid_rest``: the mid-surface they insert on; ``fascia``: trimesh to lie on."""
        self.name = name
        o = np.asarray(fibres.origin, float)
        a = np.asarray(fibres.a)
        k = len(a)
        # across the width: the origin line and a smoothed insertion line (fibres share vertices)
        self.ins_w = np.zeros((k, k))
        for i in range(k):
            lo, hi = max(0, i - 2), min(k, i + 3)
            self.ins_w[i, lo:hi] = 1.0 / (hi - lo)
        self.a = a
        self.origin = o
        t = np.linspace(0.0, 1.0, n_along)
        self.t = t
        ins_rest = self.ins_w @ mid_rest[a]
        P = o[:, None, :] + t[None, :, None] * (ins_rest - o)[:, None, :]  # (k, n, 3) straight
        if fascia is not None:  # into the plane under the fat, where the scene has it
            import trimesh

            q, d, tri = trimesh.proximity.closest_point(fascia, P.reshape(-1, 3))
            nrm = fascia.face_normals[tri]
            off = P.reshape(-1, 3) - q
            # where the plane is straight below or above (not past its rim at the CT's edge)
            ok = (np.abs((off * nrm).sum(1)) >= 0.9 * d - 0.3).reshape(k, n_along).astype(float)
            nrm = nrm * np.where(nrm @ lateral < 0, -1.0, 1.0)[:, None]  # outward: away from the head
            lay = (q - (CLEAR_MM + 0.5 * thickness_mm) * nrm).reshape(P.shape)  # just under the fat
            # a smooth shift toward that plane, fading out where the scene ends and toward the
            # insertion (the plane's own folds near the ear would crumple the belly point by point)
            s_ = np.clip((t - (1.0 - RISE)) / RISE, 0.0, 1.0)
            s_ = s_ * s_ * (3 - 2 * s_)
            wt = _smooth(np.repeat(ok[..., None], 3, axis=2))
            shift = _smooth((lay - P) * ok[..., None]) / np.maximum(wt, 1e-3) * np.clip(2.0 * wt, 0.0, 1.0)
            P = P + (1 - s_)[None, :, None] * shift
        if skin is not None:  # stay under the head's skin: the insertion runs inside the ear's root
            flat = P.reshape(-1, 3)
            P = P + _smooth((press_under(flat, skin, lateral, SKIN_CLEAR_MM + 0.5 * thickness_mm) - flat).reshape(P.shape))
        self.rest = P
        self.ins_rest = ins_rest
        self.h = float(thickness_mm)
        groups = self._groups(k, FASCICLES.get(name, 1))
        self.profile = self._profile(k, groups, t, fascicles=FASCICLES.get(name, 1) > 1)
        self._build_tris(k, n_along, groups)

    @staticmethod
    def _groups(k, fascicles):
        cols = np.arange(k)
        if fascicles == 1:
            return [cols]
        edges = np.linspace(0, k, fascicles + 1)
        gap = FASCICLE_GAP * k / fascicles
        return [cols[(cols >= e0 + (gap / 2 if j else 0)) & (cols < e1 - (gap / 2 if j < fascicles - 1 else 0))]
                for j, (e0, e1) in enumerate(zip(edges[:-1], edges[1:]))]

    @staticmethod
    def _profile(k, groups, t, fascicles):
        """Thickness as a fraction of the nominal, (k, n): a muscle belly, thickest in its middle.
        Across: a rounded (elliptical) fascicle, or a flat sheet thinning toward its edges. Along:
        thin at both attachments (tendinous / aponeurotic ends, END_THICKNESS), full over the belly."""
        across = np.zeros(k)
        for g in groups:
            u = np.linspace(-1.0, 1.0, len(g)) if len(g) > 1 else np.zeros(1)
            e = np.sqrt(np.clip(1.0 - (0.95 * u) ** 2, 0.0, 1.0))
            across[g] = e if fascicles else 0.35 + 0.65 * e
        along = END_THICKNESS + (1 - END_THICKNESS) * np.sin(np.pi * t) ** 0.6
        return across[:, None] * along[None, :]

    def _build_tris(self, k, n, groups):
        """Slab triangles: top and bottom grids, closed on all four sides, per fascicle."""
        tris = []
        nv = 2 * k * n  # vertex (side, i, j) -> side * k * n + i * n + j

        def vid(side, i, j):
            return side * k * n + i * n + j

        for g in groups:
            g = [int(x) for x in g]
            for side, flip in ((0, False), (1, True)):
                for i0, i1 in zip(g[:-1], g[1:]):
                    for j in range(n - 1):
                        a, b, c, d = vid(side, i0, j), vid(side, i1, j), vid(side, i1, j + 1), vid(side, i0, j + 1)
                        tris += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]
            for i in (g[0], g[-1]):  # long edges
                for j in range(n - 1):
                    a, b, c, d = vid(0, i, j), vid(0, i, j + 1), vid(1, i, j + 1), vid(1, i, j)
                    tris += [(a, b, c), (a, c, d)] if i == g[0] else [(a, c, b), (a, d, c)]
            for j in (0, n - 1):  # ends
                for i0, i1 in zip(g[:-1], g[1:]):
                    a, b, c, d = vid(0, i0, j), vid(0, i1, j), vid(1, i1, j), vid(1, i0, j)
                    tris += [(a, c, b), (a, d, c)] if j == 0 else [(a, b, c), (a, c, d)]
        self.tris = np.asarray(tris, np.int64)
        self.nv = nv

    def place(self, x_mid):
        """Slab vertices (2 k n, 3) for the mid-surface positions ``x_mid``."""
        disp = self.ins_w @ (x_mid[self.a]) - self.ins_rest  # (k, 3) insertion motion
        P = self.rest + self.t[None, :, None] * disp[:, None, :]
        du = np.gradient(P, axis=0)
        dv = np.gradient(P, axis=1)
        nrm = np.cross(dv, du)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-12)
        half = 0.5 * self.h * self.profile[..., None] * nrm
        return np.concatenate([(P + half).reshape(-1, 3), (P - half).reshape(-1, 3)])


class Bellies:
    """All the auricular muscle bellies of a model, as one drawn mesh."""

    def __init__(self, model, fascia=None, skin=None, box=None):
        """``box``: (lo, hi) of the scene; triangles outside it are not drawn (the muscles and their
        attachments still reach beyond it in the physics)."""
        import tissue_params as tp

        lateral = np.asarray(model.vol.frame.lateral, float)
        self.items = []
        for fs in model.fan_sets:
            name = fs.names[0]
            if tp.FANS[name].material != "auricular_muscle":
                continue
            self.items.append(Belly(name, fs, model.mid_v, tp.FANS[name].thickness_mm.nominal, fascia, lateral,
                                    skin=skin))
        offs = np.cumsum([0] + [b.nv for b in self.items])
        self.tris = np.concatenate([b.tris + o for b, o in zip(self.items, offs[:-1])])
        self.rest = self.place(model.mid_v)
        if box is not None:
            lo, hi = box
            v = self.rest[self.tris]
            self.tris = self.tris[((v > lo) & (v < hi)).all(axis=(1, 2))]

    def place(self, x_mid):
        return np.concatenate([b.place(x_mid) for b in self.items])
