"""Tension-only fibres: the auricular muscles and ligaments as fans holding the auricle on the head.

Each fibre runs from a vertex of the model (the insertion, on cartilage) to either a fixed origin
point (fascia or mastoid, held for now) or another vertex (once the origin layer moves too). It is a
cable with the same J-shaped law as the sheets (``stretch_test``): along the fibre, Green strain
e = (l^2 / L0^2 - 1) / 2 and energy density W(e) = E e^2 / 2 + k1 / (2 k2) (exp(k2 e^2) - 1),
stored in its volume A L0; nothing when shorter than its rest length (tension only), so a fan
resists a pull away from its origin and goes slack when pushed. Force along the fibre:
A lambda W'(e).

A fan (``fan``) is many fibres spread from an origin line to an insertion patch: the shape of the
auricularis superior (about 4.9 cm wide at its origin, 4.5 cm long, converging on the top of the
auricle), anterior and posterior. They are not drawn.

Units: mm, N, MPa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import peel_model as pm


@dataclass
class Fibres:
    a: np.ndarray  # (F,) insertion vertex per fibre
    b: np.ndarray  # (F,) origin vertex, or -1 for a fixed origin point
    origin: np.ndarray  # (F, 3) fixed origin point (used where b == -1)
    rest_len: np.ndarray  # (F,) rest length L0
    area: np.ndarray  # (F,) cross-section A (mm^2) carried by this fibre
    E_mpa: float | np.ndarray = 0.0  # linear part (one value, or one per fibre)
    k1_mpa: float | np.ndarray = 0.0  # collagen recruitment (J-curve); off when 0
    k2: float | np.ndarray = 0.0
    names: list = field(default_factory=list)  # optional: which fan each fibre belongs to (per fibre)

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=np.int64)
        self.b = np.asarray(self.b, dtype=np.int64)
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(-1, 3)
        self.rest_len = np.asarray(self.rest_len, dtype=np.float64)
        self.area = np.asarray(self.area, dtype=np.float64)
        n = len(self.a)
        self.E_arr = np.broadcast_to(np.asarray(self.E_mpa, float), (n,)).copy()
        self.k1_arr = np.broadcast_to(np.asarray(self.k1_mpa, float), (n,)).copy()
        self.k2_arr = np.broadcast_to(np.asarray(self.k2, float), (n,)).copy()

    def __len__(self):
        return len(self.a)

    def law(self, e):
        """W, W', W'' of the fibre law at Green strain e (zero in compression)."""
        e = np.asarray(e, dtype=np.float64)
        on = e > 0
        ep = np.where(on, e, 0.0)
        if e.ndim == 0:  # one strain: the first fibre's material (a set of one material)
            E, k1, k2 = (np.asarray(a[0]) for a in (self.E_arr, self.k1_arr, self.k2_arr))
            ep, on = np.asarray(ep), np.asarray(on)
            w, dw, d2w = 0.5 * E * ep**2, E * ep, np.where(on, E, 0.0)
            if k1 > 0 and k2 > 0:
                cw, cdw, cd2w = pm.collagen(ep, k1, k2)
                w, dw, d2w = w + cw, dw + cdw, d2w + np.where(on, cd2w, 0.0)
            return w, dw, d2w
        E, k1, k2 = (np.broadcast_to(a, e.shape) for a in (self.E_arr, self.k1_arr, self.k2_arr))
        w, dw, d2w = 0.5 * E * ep**2, E * ep, np.where(on, E, 0.0)
        j = (k1 > 0) & (k2 > 0)
        if j.any():
            cw, cdw, cd2w = pm.collagen(ep[j], k1[j], k2[j])
            w, dw, d2w = w.copy(), dw.copy(), d2w.copy()
            w[j] += cw
            dw[j] += cdw
            d2w[j] += np.where(on[j], cd2w, 0.0)
        return w, dw, d2w

    def vectors(self, x):
        """Fibre vectors insertion - origin."""
        far = np.where((self.b >= 0)[:, None], x[np.maximum(self.b, 0)], self.origin)
        return x[self.a] - far

    def strain(self, x):
        d = self.vectors(x)
        return 0.5 * ((d * d).sum(axis=1) / self.rest_len**2 - 1.0)

    def tension(self, x):
        """Force along each fibre (N): A lambda W'(e)."""
        d = self.vectors(x)
        lam = np.sqrt((d * d).sum(axis=1)) / self.rest_len
        return self.area * lam * self.law(0.5 * (lam**2 - 1.0))[1]

    def energy(self, x):
        d = self.vectors(x)
        e = 0.5 * ((d * d).sum(axis=1) / self.rest_len**2 - 1.0)
        w, dw, _ = self.law(e)
        vol = self.area * self.rest_len
        g = (vol * dw / self.rest_len**2)[:, None] * d
        grad = np.zeros_like(x)
        np.add.at(grad, self.a, g)
        moving = self.b >= 0
        np.add.at(grad, self.b[moving], -g[moving])
        return float((vol * w).sum()), grad

    def stencils(self):
        """Pairs of moving vertices coupled by a fibre (for the slow solver's colouring)."""
        m = self.b >= 0
        return np.stack([self.a[m], self.b[m]], axis=1)

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if p is not None and len(p)]
        cat = lambda key: np.concatenate([getattr(p, key) for p in parts])
        return Fibres(cat("a"), cat("b"), cat("origin"), cat("rest_len"), cat("area"), cat("E_arr"), cat("k1_arr"),
                      cat("k2_arr"), sum((list(p.names) or [""] * len(p) for p in parts), []))

    def shifted(self, offset):
        """The same fibres with vertex ids moved by ``offset`` (to a layer of a stack)."""
        return Fibres(self.a + offset, np.where(self.b >= 0, self.b + offset, -1), self.origin, self.rest_len, self.area,
                      self.E_arr, self.k1_arr, self.k2_arr, list(self.names))


def fan(insertion_vertices, x_rest, origin_a, origin_b, thickness_mm, E_mpa, k1_mpa=0.0, k2=0.0, per_vertex=3,
        name="fan"):
    """A flat fan: fibres from points spread along the origin line a-b to the insertion vertices.

    Each insertion vertex gets ``per_vertex`` fibres to evenly spread origin points (the fan's fibres
    converge), and the fan's cross-section (origin width x thickness) is shared among its fibres.
    """
    ins = np.asarray(insertion_vertices, dtype=np.int64)
    oa, ob = np.asarray(origin_a, float), np.asarray(origin_b, float)
    n = len(ins) * per_vertex
    t = (np.arange(n) + 0.5) / n
    origins = oa + t[:, None] * (ob - oa)
    # insertions in order along the origin line, so the fibres do not cross
    a = np.repeat(ins[np.argsort(_along(x_rest[ins], oa, ob))], per_vertex)
    L0 = np.linalg.norm(x_rest[a] - origins, axis=1)
    area = np.full(n, np.linalg.norm(ob - oa) * thickness_mm / n)
    return Fibres(a, -np.ones(n, dtype=np.int64), origins, L0, area, E_mpa, k1_mpa, k2, [name] * n)


def _along(p, oa, ob):
    u = (ob - oa) / np.linalg.norm(ob - oa)
    return (p - oa) @ u
