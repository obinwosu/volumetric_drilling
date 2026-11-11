"""Thin skin sheet bonded to a rigid surface by breakable peel bonds.

Reference model for the canal skin, solved quasi-statically by minimising
energy. It is a check on the physics, not the real-time solver.

Units: mm, N, MPa (N/mm^2). Energy in N*mm. Adhesion energy in N/mm
(1 N/mm = 1000 J/m^2).

Energy terms
  membrane  St Venant-Kirchhoff on each triangle, plane stress, times thickness
  bending   discrete hinge on each interior edge, scaled so a cylinder of
            curvature k stores D/2 k^2 per area, D = E h^3 / (12 (1 - nu^2)).
            A curved rest shape (``curved=True``: a shell, e.g. cartilage) bends
            from its rest angle, k (theta - theta0)^2, and stretches from its rest
            triangles in their own planes.
            ``bending="shell"`` replaces the hinges by a shell with the plate's
            Poisson ratio (cartilage): each triangle's shape operator from its three
            edges' bend angles (Grinspun et al. 2006), S = sum_i c_i dtheta_i t_i t_i^T,
            energy area D/2 ((1 - nu) tr(S^2) + nu tr(S)^2). Hinges alone charge only
            the mean curvature, so a strip curls into a saddle almost for free.
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


SHELL_C = 1.0  # shell scale on a curved mesh with no flat layout (set from the flat calibration below)
BEND_C = 0.513  # hinge stiffness / (D * |e|^2 / (A1 + A2)) that bends a regular mesh like a plate (flat calibration)


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
    # each rest axis in tension. Off when k1 = 0. Past k2 E^2 = COLLAGEN_CAP (far
    # beyond rupture) the curve continues as a smooth quadratic (see ``collagen``).
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
    curved: bool = False  # the rest shape is a curved shell (not a flat sheet on z = 0)
    calib_rest: np.ndarray | None = None  # with ``curved``: the same mesh laid flat, to calibrate bending (else BEND_C)
    bending: str = "hinge"  # "hinge" (skin) or "shell" (plate with its Poisson ratio: cartilage)
    dm_inv: np.ndarray = field(init=False)
    area: np.ndarray = field(init=False)
    vert_area: np.ndarray = field(init=False)
    hinges: np.ndarray = field(init=False)  # (h, 4): opposite a, opposite b, edge v3, edge v4
    hinge_k: np.ndarray = field(init=False)
    hinge_rest: np.ndarray = field(init=False)  # rest bend angle per hinge (0 for a flat sheet)
    shell_K: np.ndarray | None = field(init=False, default=None)  # (T, 3, 3) shell bending, with bending="shell"
    tri_hinge: np.ndarray | None = field(init=False, default=None)
    shell_stencil: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        rest, tris = self.rest, self.tris
        if self.curved:  # each rest triangle in its own plane: edges in a local 2D frame (always positive)
            e1_ = rest[tris[:, 1]] - rest[tris[:, 0]]
            e2_ = rest[tris[:, 2]] - rest[tris[:, 0]]
            ax = e1_ / np.linalg.norm(e1_, axis=1, keepdims=True)
            nrm = np.cross(e1_, e2_)
            ay = np.cross(nrm / np.linalg.norm(nrm, axis=1, keepdims=True), ax)
            d1 = np.stack([np.einsum("ij,ij->i", e1_, ax), np.einsum("ij,ij->i", e1_, ay)], axis=1)
            d2 = np.stack([np.einsum("ij,ij->i", e2_, ax), np.einsum("ij,ij->i", e2_, ay)], axis=1)
        else:
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
        self.hinge_rest = _dihedral(rest, self.hinges)[0] if (self.curved and len(self.hinges)) else np.zeros(len(self.hinges))
        self.hinge_k = np.zeros(len(self.hinges))
        self.hinge_k = self._calibrated_hinge_k()
        if self.bending == "shell":
            self._setup_shell()
            # a shell element couples its triangle with the vertices across its three edges
            st = np.concatenate([self.tris, self.tris], axis=1).copy()  # (T, 6); a missing one repeats a corner
            for i in range(3):
                hk = self.tri_hinge[:, i]
                ok = hk >= 0
                a_, b_ = self.hinges[hk[ok], 0], self.hinges[hk[ok], 1]
                a_in = (a_[:, None] == self.tris[ok]).any(axis=1)
                st[ok, 3 + i] = np.where(a_in, b_, a_)
            self.shell_stencil = st
        self.history = np.zeros(len(self.bonded))  # largest separation reached per bond
        self.color_groups = _colors(len(rest), [self.tris, self.hinges] + ([self.shell_stencil] if self.bending == "shell" else []))

    # -- bending calibration -------------------------------------------------
    def _hinge_weight(self):
        h = self.hinges
        e = np.linalg.norm(self.rest[h[:, 3]] - self.rest[h[:, 2]], axis=1)
        tri_area = _hinge_areas(self.rest, h)
        return e**2 / tri_area

    def _calibrated_hinge_k(self):
        """Scale hinges so a cylinder about y stores D/2 k^2 per area. A curved shell is
        calibrated on its flat layout (``calib_rest``) when given, else with ``BEND_C``."""
        if len(self.hinges) == 0:
            return np.zeros(0)
        if self.curved:
            if self.calib_rest is not None:
                flat = Sheet(np.asarray(self.calib_rest, float), self.tris, np.zeros(0, np.int64), self.params)
                c = flat.hinge_k / np.maximum(self.params.bending_D * flat._hinge_weight(), 1e-300)
                return self.params.bending_D * float(np.median(c)) * self._hinge_weight()
            return self.params.bending_D * BEND_C * self._hinge_weight()
        radius = 50.0 * max(np.ptp(self.rest[:, 0]), 1.0)
        rolled = roll_about_y(self.rest, radius)
        theta = _dihedral(rolled, self.hinges)[0]
        weight = self._hinge_weight()
        discrete = float(np.sum(weight * theta**2))
        if discrete <= 0.0:  # no hinge bends when rolled about y (a single cell)
            return np.zeros(len(self.hinges))
        target = 0.5 * (1.0 / radius) ** 2 * float(self.area.sum())
        return self.params.bending_D * (target / discrete) * weight

    # -- shell bending (plate with Poisson ratio) ------------------------------
    def _setup_shell(self):
        """Per triangle: its three edges' hinges (-1 on the boundary) and the constant 3x3
        matrix K with energy dtheta^T K dtheta (see the module doc)."""
        rest, tris, h = self.rest, self.tris, self.hinges
        edge_h = {}
        for k, (_a, _b, u, v) in enumerate(h):
            edge_h[(min(u, v), max(u, v))] = k
        th = -np.ones((len(tris), 3), np.int64)
        for t, (a, b, c) in enumerate(tris):
            for i, (u, v) in enumerate(((b, c), (c, a), (a, b))):  # edge opposite corner i
                th[t, i] = edge_h.get((min(u, v), max(u, v)), -1)
        self.tri_hinge = th
        # each hinge's angle measured against the mesh's own normal: +1 when the triangle holding
        # the hinge's first opposite vertex runs (opp, v3, v4) in its winding, else -1
        sign = np.ones(len(h))
        tri_of = {}
        for t, tri in enumerate(tris):
            tri_of[frozenset(int(q) for q in tri)] = t
        for k, (a, _b, u, v) in enumerate(h):
            tri = tris[tri_of[frozenset((int(a), int(u), int(v)))]]
            i = int(np.nonzero(tri == a)[0][0])
            sign[k] = 1.0 if (tri[(i + 1) % 3] == u and tri[(i + 2) % 3] == v) else -1.0
        self.hinge_sign = sign
        # rest geometry: edge lengths, in-plane edge normals t_i
        P = [rest[tris[:, k]] for k in range(3)]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        area = 0.5 * np.linalg.norm(n, axis=1)
        nh = n / np.maximum(2 * area, 1e-300)[:, None]
        tv, ln = [], []
        for i in range(3):
            e = P[(i + 2) % 3] - P[(i + 1) % 3]
            ln.append(np.linalg.norm(e, axis=1))
            t = np.cross(e, nh)
            tv.append(t / np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None])
        cvec = np.stack([ln[i] / (2.0 * area) for i in range(3)], axis=1)  # (T, 3)
        dots = np.stack([np.stack([np.einsum("ij,ij->i", tv[i], tv[j]) for j in range(3)], axis=1) for i in range(3)], axis=1)
        nu = self.params.nu
        G = (1.0 - nu) * dots**2 + nu  # (T, 3, 3)
        K = 0.5 * self.params.bending_D * area[:, None, None] * cvec[:, :, None] * cvec[:, None, :] * G
        K[th < 0, :] = 0.0
        K[np.repeat((th < 0)[:, None, :], 3, axis=1)] = 0.0
        self.shell_K = K
        self.shell_K = K * self._shell_scale()

    def _shell_scale(self):
        """Scale so a cylinder stores D/2 k^2 per area (as the hinges are calibrated)."""
        if not len(self.hinges):
            return 1.0
        if self.curved:
            if self.calib_rest is None:
                return SHELL_C
            flat = Sheet(np.asarray(self.calib_rest, float), self.tris, np.zeros(0, np.int64), self.params, bending="shell")
            return flat._shell_scale_flat()
        return self._shell_scale_flat()

    def _shell_scale_flat(self):
        radius = 50.0 * max(np.ptp(self.rest[:, 0]), 1.0)
        rolled = roll_about_y(self.rest, radius)
        theta = _dihedral(rolled, self.hinges)[0]
        e = self._shell_energy_of(theta)
        if e <= 0.0:
            return 1.0
        return 0.5 * (1.0 / radius) ** 2 * float(self.area.sum()) * self.params.bending_D / e * (1.0 if self.shell_K is None else 1.0)

    def _shell_energy_of(self, theta):
        d = np.where(self.tri_hinge >= 0, (self.hinge_sign * _wrap(theta - self.hinge_rest))[np.maximum(self.tri_hinge, 0)], 0.0)
        return float(np.einsum("ti,tij,tj->", d, self.shell_K, d))

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
                w, dw, _d2w = collagen(ea, p.k1_mpa, p.k2)
                energy = energy + vol * w
                S[:, a, a] += dw
        H = vol[:, None, None] * (F @ S @ np.transpose(self.dm_inv, (0, 2, 1)))
        grad = _scatter(len(x), (t[:, 1], H[:, :, 0]), (t[:, 2], H[:, :, 1]), (t[:, 0], -H[:, :, 0] - H[:, :, 1]))
        return float(energy.sum()), grad

    def _bending(self, x):
        theta, u = _dihedral(x, self.hinges, with_grad=True)
        dth = _wrap(theta - self.hinge_rest)
        if self.bending == "shell":
            valid = self.tri_hinge >= 0
            sd = self.hinge_sign * dth
            d = np.where(valid, sd[np.maximum(self.tri_hinge, 0)], 0.0)
            kd = np.einsum("tij,tj->ti", self.shell_K, d)
            energy = np.einsum("ti,ti->t", d, kd)
            coef = np.zeros(len(self.hinges))
            np.add.at(coef, self.tri_hinge[valid], 2.0 * kd[valid])
            coef *= self.hinge_sign
            grad = _scatter(len(x), *((self.hinges[:, slot], coef[:, None] * u[slot]) for slot in range(4)))
            return float(energy.sum()), grad
        energy = self.hinge_k * dth**2
        coef = 2.0 * self.hinge_k * dth
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


COLLAGEN_CAP = 8.0  # k2 e^2 where the exponential hands over to a quadratic


def collagen(e, k1, k2):
    """Collagen energy density W(e) = k1/(2 k2) (exp(k2 e^2) - 1) for e >= 0, and W', W''.

    Past e_c (k2 e_c^2 = COLLAGEN_CAP, about three times any rupture strain) W
    continues as its second-order Taylor expansion at e_c, so a wildly
    overstretched element (a fast hand, one solver frame) has a finite energy
    the solver can recover from. Identical in fast/soft_fast.cpp.
    """
    e = np.maximum(np.asarray(e, dtype=np.float64), 0.0)
    ec = np.sqrt(COLLAGEN_CAP / k2)
    m = np.minimum(e, ec)
    grow = np.exp(k2 * m * m)
    w = k1 / (2.0 * k2) * (grow - 1.0)
    dw = k1 * m * grow
    d2w = k1 * grow * (1.0 + 2.0 * k2 * m * m)
    d = e - m  # > 0 only past the cap
    return w + dw * d + 0.5 * d2w * d * d, dw + d2w * d, np.where(e > 0, d2w, 0.0)


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


def _wrap(a):
    """Angle into (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


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
