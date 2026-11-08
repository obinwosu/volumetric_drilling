"""Thin skin sheet bonded to a rigid surface by breakable peel bonds.

Reference model for the canal skin, solved quasi-statically by minimising
energy. It is a check on the physics, not the real-time solver.

Units: mm, N, MPa (N/mm^2). Energy in N*mm. Adhesion energy in N/mm
(1 N/mm = 1000 J/m^2).

Energy terms
  membrane  St Venant-Kirchhoff on each triangle, plane stress, times thickness
  bending   discrete hinge on each interior edge, scaled so a cylinder of
            curvature k stores D/2 k^2 per area, D = E h^3 / (12 (1 - nu^2))
  bonds     one per bonded vertex, to its rest point on the surface. Bilinear
            traction-separation: stiff rise to Tmax at delta0, softening to zero
            at delta_c. The area under it is the adhesion energy G. Damage is
            kept: a bond that has opened unloads along its secant, never heals.
  contact   one-sided penalty keeping vertices on the surface side (z >= 0)

All material numbers are placeholders until bench data replaces them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SkinParams:
    E_mpa: float = 1.0  # skin modulus, placeholder
    nu: float = 0.45
    thickness_mm: float = 1.0
    G_n_per_mm: float = 0.05  # adhesion (peel) energy, 50 J/m^2, placeholder
    delta0_mm: float = 0.05  # separation at peak traction
    delta_c_mm: float = 0.4  # separation at full release
    contact_n_per_mm3: float = 200.0
    # Collagen recruitment (J-shaped curve): extra stress k1 E exp(k2 E^2) along
    # each rest axis in tension. Off when k1 = 0.
    k1_mpa: float = 0.0
    k2: float = 0.0

    @property
    def bending_D(self) -> float:
        return self.E_mpa * self.thickness_mm**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def T_max(self) -> float:
        return 2.0 * self.G_n_per_mm / self.delta_c_mm

    @property
    def k_bond(self) -> float:
        return self.T_max / self.delta0_mm


@dataclass
class Sheet:
    rest: np.ndarray  # (n, 3) rest positions on the surface, z = 0
    tris: np.ndarray  # (m, 3)
    bonded: np.ndarray  # (b,) vertex ids with a bond
    params: SkinParams
    floor_z: float = 0.0  # contact plane height for this sheet's mid-surface
    dm_inv: np.ndarray = field(init=False)
    area: np.ndarray = field(init=False)
    vert_area: np.ndarray = field(init=False)
    hinges: np.ndarray = field(init=False)  # (h, 4): opposite a, opposite b, edge v3, edge v4
    hinge_k: np.ndarray = field(init=False)

    def __post_init__(self):
        rest, tris = self.rest, self.tris
        d1 = rest[tris[:, 1], :2] - rest[tris[:, 0], :2]
        d2 = rest[tris[:, 2], :2] - rest[tris[:, 0], :2]
        dm = np.stack([d1, d2], axis=2)  # (m, 2, 2), columns are edges
        det = np.linalg.det(dm)
        flip = det < 0
        if flip.any():
            self.tris = tris = tris.copy()
            tris[flip] = tris[flip][:, [0, 2, 1]]
            dm[flip] = dm[flip][:, :, [1, 0]]
            det = np.abs(det)
        self.dm_inv = np.linalg.inv(dm)
        self.area = 0.5 * det
        self.vert_area = np.zeros(len(rest))
        for corner in range(3):
            np.add.at(self.vert_area, tris[:, corner], self.area / 3.0)
        self.hinges = _hinges(tris)
        self.hinge_k = np.zeros(len(self.hinges))
        self.hinge_k = self._calibrated_hinge_k()
        self.history = np.zeros(len(self.bonded))  # largest separation reached per bond
        self.color_groups = _colors(len(rest), [self.tris, self.hinges])

    # -- bending calibration -------------------------------------------------
    def _hinge_weight(self):
        h = self.hinges
        e = np.linalg.norm(self.rest[h[:, 3]] - self.rest[h[:, 2]], axis=1)
        tri_area = _hinge_areas(self.rest, h)
        return e**2 / tri_area

    def _calibrated_hinge_k(self):
        """Scale hinges so a cylinder about y stores D/2 k^2 per area."""
        if len(self.hinges) == 0:
            return np.zeros(0)
        radius = 50.0 * max(np.ptp(self.rest[:, 0]), 1.0)
        rolled = roll_about_y(self.rest, radius)
        theta = _dihedral(rolled, self.hinges)[0]
        weight = self._hinge_weight()
        discrete = float(np.sum(weight * theta**2))
        if discrete <= 0.0:  # no hinge bends when rolled about y (a single cell)
            return np.zeros(len(self.hinges))
        target = 0.5 * (1.0 / radius) ** 2 * float(self.area.sum())
        return self.params.bending_D * (target / discrete) * weight

    # -- energy ---------------------------------------------------------------
    def energy(self, x, parts=False):
        e_m, g_m = self._membrane(x)
        e_b, g_b = self._bending(x)
        e_c, g_c = self._bonds(x)
        e_k, g_k = self._contact(x)
        total = e_m + e_b + e_c + e_k
        grad = g_m + g_b + g_c + g_k
        if parts:
            return total, grad, {"membrane": e_m, "bending": e_b, "bonds": e_c, "contact": e_k}
        return total, grad

    def _membrane(self, x):
        p = self.params
        mu = p.E_mpa / (2.0 * (1.0 + p.nu))
        lam = p.E_mpa * p.nu / (1.0 - p.nu**2)  # plane stress
        t = self.tris
        ds = np.stack([x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]], axis=2)  # (m, 3, 2)
        F = ds @ self.dm_inv
        C = np.einsum("mki,mkj->mij", F, F)
        Eg = 0.5 * (C - np.eye(2))
        tr = Eg[:, 0, 0] + Eg[:, 1, 1]
        vol = self.area * p.thickness_mm
        energy = vol * (mu * np.einsum("mij,mij->m", Eg, Eg) + 0.5 * lam * tr**2)
        S = 2.0 * mu * Eg + lam * tr[:, None, None] * np.eye(2)
        if p.k1_mpa > 0.0 and p.k2 > 0.0:
            for a in range(2):
                ea = np.maximum(Eg[:, a, a], 0.0)
                grow = np.exp(p.k2 * ea**2)
                energy = energy + vol * p.k1_mpa / (2.0 * p.k2) * (grow - 1.0)
                S[:, a, a] += p.k1_mpa * ea * grow
        H = vol[:, None, None] * (F @ S @ np.transpose(self.dm_inv, (0, 2, 1)))
        grad = _scatter(len(x), (t[:, 1], H[:, :, 0]), (t[:, 2], H[:, :, 1]), (t[:, 0], -H[:, :, 0] - H[:, :, 1]))
        return float(energy.sum()), grad

    def _bending(self, x):
        theta, u = _dihedral(x, self.hinges, with_grad=True)
        energy = self.hinge_k * theta**2
        coef = 2.0 * self.hinge_k * theta
        grad = _scatter(len(x), *((self.hinges[:, slot], coef[:, None] * u[slot]) for slot in range(4)))
        return float(energy.sum()), grad

    def _bond_terms(self, x):
        p = self.params
        r = x[self.bonded] - self.rest[self.bonded]
        delta = np.linalg.norm(r, axis=1)
        m_eff = np.maximum(self.history, p.delta0_mm)
        t_m = traction(m_eff, p)
        ks = np.where(m_eff > 0, t_m / m_eff, 0.0)
        a = self.vert_area[self.bonded]
        return r, delta, m_eff, ks, a

    def _bonds(self, x):
        p = self.params
        r, delta, m_eff, ks, a = self._bond_terms(x)
        inside = delta <= m_eff
        psi = np.where(
            inside,
            0.5 * ks * delta**2,
            0.5 * ks * m_eff**2 + envelope_energy(delta, p) - envelope_energy(m_eff, p),
        )
        safe = np.maximum(delta, 1e-12)
        scale = np.where(inside, ks, traction(delta, p) / safe)
        grad = np.zeros_like(x)
        grad[self.bonded] = (a * scale)[:, None] * r
        return float(np.sum(a * psi)), grad

    def _contact(self, x):
        k = self.params.contact_n_per_mm3 * self.vert_area
        below = np.minimum(x[:, 2] - self.floor_z, 0.0)
        grad = np.zeros_like(x)
        grad[:, 2] = k * below
        return float(np.sum(0.5 * k * below**2)), grad

    def commit(self, x):
        """Record bond damage after a converged step."""
        r = x[self.bonded] - self.rest[self.bonded]
        self.history = np.maximum(self.history, np.linalg.norm(r, axis=1))

    def broken(self):
        return self.history >= self.params.delta_c_mm


def _scatter(n, *pairs):
    """Sum (index, (k, 3) value) pairs into an (n, 3) array."""
    index = np.concatenate([i for i, _v in pairs])
    values = np.concatenate([v for _i, v in pairs])
    return np.stack([np.bincount(index, values[:, k], minlength=n) for k in range(3)], axis=1)


def _colors(n, stencils):
    """Vertex groups whose gradient stencils never overlap, for FD Hessians.

    ``stencils`` are (k, s) arrays of vertex ids that interact (a triangle or a
    hinge). Two vertices share a group only if no stencil-neighbour is shared.
    """
    from scipy import sparse

    rows, cols = [], []
    for st in stencils:
        for a in range(st.shape[1]):
            for b in range(st.shape[1]):
                rows.append(st[:, a])
                cols.append(st[:, b])
    rows = np.concatenate(rows + [np.arange(n)])
    cols = np.concatenate(cols + [np.arange(n)])
    adj = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    adj.data[:] = 1
    two = (adj @ adj).tocsr()
    color = -np.ones(n, dtype=np.int64)
    for v in range(n):
        taken = set(color[two.indices[two.indptr[v] : two.indptr[v + 1]]].tolist())
        c = 0
        while c in taken:
            c += 1
        color[v] = c
    groups = []
    for c in range(color.max() + 1):
        members = np.flatnonzero(color == c)
        # rows touched by each member: its stencil neighbours
        owner_rows = [adj.indices[adj.indptr[v] : adj.indptr[v + 1]] for v in members]
        r = np.concatenate(owner_rows)
        o = np.repeat(members, [len(x) for x in owner_rows])
        groups.append((members, r, o))
    return groups


def hessian(sheet: "Sheet", x, eps=1e-6):
    """Sparse Hessian of the sheet energy by coloured central differences."""
    from scipy import sparse

    n = len(x)
    rows, cols, vals = [], [], []
    for members, r, o in sheet.color_groups:
        for k in range(3):
            step = np.zeros_like(x)
            step[members, k] = eps
            d = (sheet.energy(x + step)[1] - sheet.energy(x - step)[1]) / (2.0 * eps)
            for a in range(3):
                rows.append(3 * r + a)
                cols.append(3 * o + k)
                vals.append(d[r, a])
    h = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n))
    return (0.5 * (h + h.T)).tocsc()


def solve(sheet: "Sheet", x0, fixed, gtol=1e-9, max_iter=200):
    """Minimise the sheet energy with ``fixed`` (n, 3) bool entries held at x0.

    Damped Newton on an assembled sparse Hessian. The damping grows until the
    step lowers the energy, so buckling and bond softening (where the energy is
    not convex) are handled.
    """
    from types import SimpleNamespace

    from scipy import sparse
    from scipy.sparse.linalg import splu

    free = ~fixed.ravel()
    x = x0.copy()
    energy, grad = sheet.energy(x)
    mu = 1e-6
    it = 0
    for it in range(1, max_iter + 1):
        g = grad.ravel()[free]
        if np.abs(g).max() < gtol:
            break
        H = hessian(sheet, x)[free][:, free]
        eye = sparse.identity(H.shape[0], format="csc")
        accepted = False
        for _attempt in range(30):
            try:
                d = -splu((H + mu * eye).tocsc()).solve(g)
            except RuntimeError:
                mu *= 10.0
                continue
            if g @ d >= 0.0:
                mu *= 10.0
                continue
            step = np.zeros(x.size)
            step[free] = d
            alpha = 1.0
            while alpha > 1e-4:
                trial = x + alpha * step.reshape(-1, 3)
                e_new, g_new = sheet.energy(trial)
                if e_new <= energy + 1e-4 * alpha * (g @ d):
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                x, energy, grad = trial, e_new, g_new
                mu = max(mu / 4.0, 1e-9) if alpha == 1.0 else mu
                break
            mu *= 10.0
        if not accepted:
            break
    return x, SimpleNamespace(nit=it, energy=energy, max_grad=float(np.abs(grad.ravel()[free]).max()))


@dataclass
class BondLaw:
    """One bilinear traction-separation channel. Area under it is G."""

    G_n_per_mm: float
    delta0_mm: float
    delta_c_mm: float

    @property
    def T_max(self) -> float:
        return 2.0 * self.G_n_per_mm / self.delta_c_mm

    @property
    def k_bond(self) -> float:
        return self.T_max / self.delta0_mm


def traction(delta, p: SkinParams):
    delta = np.asarray(delta, dtype=np.float64)
    rise = p.k_bond * delta
    soft = p.T_max * (p.delta_c_mm - delta) / (p.delta_c_mm - p.delta0_mm)
    return np.where(delta <= p.delta0_mm, rise, np.where(delta < p.delta_c_mm, soft, 0.0))


def envelope_energy(delta, p: SkinParams):
    d0, dc, tm = p.delta0_mm, p.delta_c_mm, p.T_max
    delta = np.asarray(delta, dtype=np.float64)
    rise = 0.5 * p.k_bond * delta**2
    mid = np.minimum(delta, dc)
    soft = 0.5 * tm * d0 + tm / (dc - d0) * (dc * (mid - d0) - 0.5 * (mid**2 - d0**2))
    return np.where(delta <= d0, rise, soft)


def roll_about_y(rest, radius):
    """Wrap a flat sheet onto a cylinder about the y axis (curvature along x)."""
    out = rest.copy()
    phi = rest[:, 0] / radius
    out[:, 0] = radius * np.sin(phi)
    out[:, 2] = radius * (1.0 - np.cos(phi))
    return out


def _hinges(tris):
    edges = {}
    for t_index, (a, b, c) in enumerate(tris):
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(u, v), max(u, v))
            edges.setdefault(key, []).append(w)
    rows = [(opp[0], opp[1], key[0], key[1]) for key, opp in edges.items() if len(opp) == 2]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def _hinge_areas(x, h):
    a = 0.5 * np.linalg.norm(np.cross(x[h[:, 0]] - x[h[:, 2]], x[h[:, 0]] - x[h[:, 3]]), axis=1)
    b = 0.5 * np.linalg.norm(np.cross(x[h[:, 1]] - x[h[:, 2]], x[h[:, 1]] - x[h[:, 3]]), axis=1)
    return a + b


def _dihedral(x, h, with_grad=False):
    """Signed bend angle at each hinge and its gradient (Bridson et al. 2003).

    Slots: 0, 1 the opposite vertices, 2, 3 the shared edge.
    """
    x1, x2, x3, x4 = x[h[:, 0]], x[h[:, 1]], x[h[:, 2]], x[h[:, 3]]
    e = x4 - x3
    e_len = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
    n1 = np.cross(x1 - x3, x1 - x4)
    n2 = np.cross(x2 - x4, x2 - x3)
    n1_sq = np.maximum(np.einsum("ij,ij->i", n1, n1), 1e-24)
    n2_sq = np.maximum(np.einsum("ij,ij->i", n2, n2), 1e-24)
    n1_hat = n1 / np.sqrt(n1_sq)[:, None]
    n2_hat = n2 / np.sqrt(n2_sq)[:, None]
    sin = np.einsum("ij,ij->i", np.cross(n1_hat, n2_hat), e / e_len[:, None])
    cos = np.einsum("ij,ij->i", n1_hat, n2_hat)
    theta = np.arctan2(sin, cos)
    if not with_grad:
        return theta, None
    u1 = (e_len / n1_sq)[:, None] * n1
    u2 = (e_len / n2_sq)[:, None] * n2
    e_hat = e / e_len[:, None]
    w1 = n1 / n1_sq[:, None]
    w2 = n2 / n2_sq[:, None]
    u3 = np.einsum("ij,ij->i", x1 - x4, e_hat)[:, None] * w1 + np.einsum("ij,ij->i", x2 - x4, e_hat)[:, None] * w2
    u4 = -np.einsum("ij,ij->i", x1 - x3, e_hat)[:, None] * w1 - np.einsum("ij,ij->i", x2 - x3, e_hat)[:, None] * w2
    # Bridson's bending modes are the gradient of -theta for this sign of theta.
    return theta, (-u1, -u2, -u3, -u4)


def strip_mesh(length_mm, width_mm, edge_mm):
    """Flat strip on z = 0, x along the length. Alternating diagonals."""
    nx = max(2, int(round(length_mm / edge_mm)) + 1)
    ny = max(2, int(round(width_mm / edge_mm)) + 1)
    xs = np.linspace(0.0, length_mm, nx)
    ys = np.linspace(0.0, width_mm, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    rest = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            if (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    return rest, np.asarray(tris, dtype=np.int64), idx
