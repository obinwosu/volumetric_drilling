"""Python side of the fast C++ soft-tissue solvers (fast/soft_fast.cpp).

Builds the C++ model from the same mesh, bending calibration and bonds as the
Python reference (``peel_model.Sheet``), so the reference can referee it.
Compiles the library on first use (needs g++ and Eigen headers).

Units: mm, ms, g, N.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess

import numpy as np

import peel_model as pm

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "fast" / "soft_fast.cpp"
BUILD = HERE / "fast" / "_build"
# SuiteSparse: a from-source install in ~/.local/suitesparse wins over the system one.
_CHOLMOD_ROOTS = [pathlib.Path.home() / ".local" / "suitesparse", pathlib.Path("/usr")]
CHOLMOD_ROOT = next((r for r in _CHOLMOD_ROOTS if (r / "include" / "suitesparse" / "cholmod.h").exists()), None)

DENSITY_G_PER_MM3 = 1e-3  # water-like soft tissue
NEWTON, XPBD = 0, 1
# Newton's sparse factorization
EIGEN_LDLT, CHOLMOD_SUPERNODAL, CHOLMOD_SIMPLICIAL = 0, 1, 2

_dbl = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_int = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_lib = None


def build(force=False):
    """Compile the library; with CHOLMOD when SuiteSparse headers are installed."""
    cholmod = CHOLMOD_ROOT is not None
    library = BUILD / ("libsoft_fast_cholmod.so" if cholmod else "libsoft_fast.so")
    if force or not library.exists() or library.stat().st_mtime < SOURCE.stat().st_mtime:
        BUILD.mkdir(parents=True, exist_ok=True)
        cmd = ["g++", "-O3", "-march=native", "-std=c++17", "-fopenmp", "-shared", "-fPIC", "-I/usr/include/eigen3"]
        if cholmod:
            cmd += ["-DSF_CHOLMOD", f"-I{CHOLMOD_ROOT / 'include' / 'suitesparse'}"]
        cmd += [str(SOURCE), "-o", str(library)]
        if cholmod:
            libdir = CHOLMOD_ROOT / "lib"
            cmd += [f"-L{libdir}", f"-Wl,-rpath,{libdir}", "-lcholmod", "-lsuitesparseconfig"]
        subprocess.run(cmd, check=True)
    return library


def lib():
    global _lib
    if _lib is None:
        # CHOLMOD's dense kernels are small; one BLAS thread is as fast as four
        # and leaves the cores to haptics and graphics.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        L = ctypes.CDLL(str(build()))
        L.sf_create.restype = ctypes.c_void_p
        L.sf_create.argtypes = [ctypes.c_int, _dbl, _dbl]
        L.sf_destroy.argtypes = [ctypes.c_void_p]
        d = ctypes.c_double
        L.sf_set_triangles.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _dbl, _dbl, _dbl, _dbl, _dbl, _dbl, _dbl]
        L.sf_set_pairs.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _int, _dbl, _int, ctypes.c_int, _dbl]
        L.sf_pair_state.argtypes = [ctypes.c_void_p, _dbl, _u8]
        L.sf_set_faces.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _int, _dbl, _dbl, _dbl, _int, _int, ctypes.c_int, _dbl, _dbl]
        L.sf_face_state.argtypes = [ctypes.c_void_p, _dbl, _dbl, _u8]
        L.sf_set_fibres.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _int, _dbl, _dbl, _dbl, _dbl, _dbl, _dbl]
        L.sf_set_blade.argtypes = [ctypes.c_void_p, _dbl, d, d, d, d]
        L.sf_set_blade_tip.argtypes = [ctypes.c_void_p, d]
        L.sf_blade_push.restype = d
        L.sf_blade_push.argtypes = [ctypes.c_void_p, _dbl]
        L.sf_set_bond_broken.argtypes = [ctypes.c_void_p, _u8]
        L.sf_set_hinges.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _dbl]
        L.sf_set_hinge_rest.argtypes = [ctypes.c_void_p, _dbl]
        L.sf_set_shell.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _int, _dbl, _dbl]
        L.sf_set_tets.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _dbl, _dbl, d, d]
        L.sf_set_bonds.argtypes = [ctypes.c_void_p, ctypes.c_int, _int, _dbl, _dbl, _dbl, ctypes.c_int]
        L.sf_set_floor.argtypes = [ctypes.c_void_p, d, _dbl]
        L.sf_set_fixed.argtypes = [ctypes.c_void_p, _u8]
        L.sf_set_external.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.sf_set_grip.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.sf_set_sdf.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.sf_set_contact.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.sf_set_self_contact.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.sf_set_grip_targets.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.sf_set_options.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.sf_set_factor.restype = ctypes.c_int
        L.sf_set_factor.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.sf_have_cholmod.restype = ctypes.c_int
        L.sf_set_hessian.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.sf_elem_count.restype = ctypes.c_int
        L.sf_elem_count.argtypes = [ctypes.c_void_p]
        L.sf_elem_hessian.restype = ctypes.c_int
        L.sf_elem_hessian.argtypes = [ctypes.c_void_p, ctypes.c_int, _dbl, ctypes.c_int, ctypes.c_int, _int, _dbl]
        L.sf_energy.restype = d
        L.sf_energy.argtypes = [ctypes.c_void_p, _dbl, _dbl]
        L.sf_commit.restype = ctypes.c_int
        L.sf_commit.argtypes = [ctypes.c_void_p, _dbl]
        L.sf_bond_state.argtypes = [ctypes.c_void_p, _dbl, _u8]
        L.sf_step.argtypes = [ctypes.c_void_p, _dbl, _dbl, d, ctypes.c_int, ctypes.c_int, d, d, _dbl]
        _lib = L
    return _lib


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i32(a):
    return np.ascontiguousarray(a, dtype=np.int32)


class FastModel:
    """A C++ model. ``x`` and ``v`` are (n, 3) float64 arrays owned by Python."""

    def __init__(self, rest, mass):
        self.rest = _f64(rest)
        self.n = len(self.rest)
        self.mass = _f64(mass)
        self.handle = lib().sf_create(self.n, self.rest.ravel(), self.mass)
        self.fixed = np.zeros((self.n, 3), dtype=bool)
        self.n_bonds = 0
        self.n_pairs = 0
        self.set_factor(best_factor())

    def __del__(self):
        if getattr(self, "handle", None) and _lib is not None:
            _lib.sf_destroy(self.handle)
            self.handle = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_sheet(cls, sheet: pm.Sheet, density=DENSITY_G_PER_MM3):
        p = sheet.params
        mass = density * sheet.vert_area * p.thickness_mm
        m = cls(sheet.rest, np.maximum(mass, 1e-12))
        m.set_triangles(sheet.tris, sheet.dm_inv, sheet.area, *triangle_material(p, len(sheet.tris)))
        if len(sheet.hinges):
            lib().sf_set_hinges(m.handle, len(sheet.hinges), _i32(sheet.hinges).ravel(), _f64(sheet.hinge_k))
            rest_angle = getattr(sheet, "hinge_rest", None)
            if rest_angle is not None and np.any(rest_angle != 0.0):  # a curved shell
                lib().sf_set_hinge_rest(m.handle, _f64(rest_angle))
            if getattr(sheet, "bending", "hinge") == "shell":  # plate bending with its Poisson ratio
                lib().sf_set_shell(m.handle, len(sheet.tris), _i32(sheet.tris).ravel(), _i32(sheet.tri_hinge).ravel(),
                                   _f64(sheet.shell_K).ravel(), _f64(sheet.hinge_sign))
        if len(getattr(sheet, "bonded", ())):
            law = _f64([p.G_n_per_mm, p.delta0_mm, p.delta_c_mm, 0.0, 1.0, 2.0])
            m.set_bonds(sheet.bonded, sheet.rest[sheet.bonded], sheet.vert_area[sheet.bonded], law, mixed=False)
        if p.contact_n_per_mm3 > 0 and sheet.floor_z > -1e8:
            lib().sf_set_floor(m.handle, sheet.floor_z, _f64(p.contact_n_per_mm3 * sheet.vert_area))
        return m

    @classmethod
    def from_stack(cls, stack, density=DENSITY_G_PER_MM3):
        """The same layered stack as ``stack.Stack``: per-layer materials, layer bonds, substrate bonds."""
        n, L = stack.n, stack.L
        mass = np.concatenate([density * s.vert_area * s.params.thickness_mm for s in stack.sheets])
        m = cls(stack.rest, np.maximum(mass, 1e-12))
        tris = np.concatenate([s.tris + k * n for k, s in enumerate(stack.sheets)])
        dm_inv = np.concatenate([s.dm_inv for s in stack.sheets])
        area = np.concatenate([s.area for s in stack.sheets])
        mats = [getattr(s, "tri_material", None) or triangle_material(s.params, len(s.tris)) for s in stack.sheets]
        m.set_triangles(tris, dm_inv, area, *(np.concatenate([mt[c] for mt in mats]) for c in range(5)))
        hinges = np.concatenate([s.hinges + k * n for k, s in enumerate(stack.sheets)])
        lib().sf_set_hinges(m.handle, len(hinges), _i32(hinges).ravel(),
                            _f64(np.concatenate([s.hinge_k for s in stack.sheets])))
        rest_angle = np.concatenate([s.hinge_rest for s in stack.sheets])
        if np.any(rest_angle != 0.0):
            lib().sf_set_hinge_rest(m.handle, _f64(rest_angle))
        shells = [k for k, s in enumerate(stack.sheets) if s.bending == "shell"]
        if shells:  # shell layers replace their own hinges; the others keep them
            h0 = np.cumsum([0] + [len(s.hinges) for s in stack.sheets])
            sign = np.ones(len(hinges))
            st_tris, st_th, st_K = [], [], []
            for k in shells:
                s = stack.sheets[k]
                sign[h0[k]:h0[k + 1]] = s.hinge_sign
                st_tris.append(s.tris + k * n)
                st_th.append(np.where(s.tri_hinge >= 0, s.tri_hinge + h0[k], -1))
                st_K.append(s.shell_K)
            t_ = np.concatenate(st_tris)
            lib().sf_set_shell(m.handle, len(t_), _i32(t_).ravel(), _i32(np.concatenate(st_th)).ravel(),
                               _f64(np.concatenate(st_K)).ravel(), _f64(sign))
        if len(getattr(stack, "face_e", ())):
            m.set_faces(stack)
        if getattr(stack, "fibres", None) is not None and len(stack.fibres):
            m.set_fibres(stack.fibres)
        if len(stack.pair_a):
            m.set_pairs(stack.pair_a, stack.pair_b, stack.pair_w, stack.pair_law, stack.laws)
        if len(stack.ids) and getattr(stack, "anchor_law", None) is not None:
            a = stack.anchor_law
            m.set_bonds(stack.ids, stack.rest[stack.ids], stack.weight,
                        _f64([a.G_n_per_mm, a.delta0_mm, a.delta_c_mm, 0.0, 1.0, 2.0]), mixed=False)
        elif len(stack.ids):
            iface = stack.iface
            law = _f64([iface.opening.G_n_per_mm, iface.opening.delta0_mm, iface.opening.delta_c_mm,
                        iface.sliding.G_n_per_mm, iface.sliding.delta0_mm, iface.sliding.delta_c_mm])
            m.set_bonds(stack.ids, stack.rest[stack.ids], stack.weight, law, mixed=True)
        k_floor = np.concatenate([s.params.contact_n_per_mm3 * s.vert_area for s in stack.sheets])
        lib().sf_set_floor(m.handle, stack.floor_z, _f64(k_floor))
        if stack.blade is not None:
            m.set_blade(stack.blade.contact_n_per_mm3 * stack.vert_area, stack.blade, stack.floor_z)
            m.set_blade_tip(stack.tip)
        return m

    @classmethod
    def from_tets(cls, rest, tets, E_mpa, nu, density=DENSITY_G_PER_MM3):
        rest = _f64(rest)
        tets = _i32(tets)
        Dm = np.stack([rest[tets[:, k]] - rest[tets[:, 0]] for k in (1, 2, 3)], axis=2)
        vol = np.linalg.det(Dm) / 6.0
        flip = vol < 0
        if flip.any():
            tets = tets.copy()
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            Dm = np.stack([rest[tets[:, k]] - rest[tets[:, 0]] for k in (1, 2, 3)], axis=2)
            vol = np.abs(vol)
        vmass = np.zeros(len(rest))
        for k in range(4):
            np.add.at(vmass, tets[:, k], density * vol / 4.0)
        m = cls(rest, np.maximum(vmass, 1e-12))
        mu = E_mpa / (2.0 * (1.0 + nu))
        lam = E_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lib().sf_set_tets(m.handle, len(tets), tets.ravel(), _f64(np.linalg.inv(Dm)).ravel(), _f64(vol), mu, lam)
        m.tets, m.tet_vol = tets, vol
        return m

    def set_triangles(self, tris, dm_inv, area, thick, mu, lam, k1, k2):
        """Triangles with a material each (arrays of length len(tris))."""
        lib().sf_set_triangles(self.handle, len(tris), _i32(tris).ravel(), _f64(dm_inv).ravel(), _f64(area),
                               _f64(thick), _f64(mu), _f64(lam), _f64(k1), _f64(k2))

    def set_pairs(self, a, b, weights, law_index, laws):
        """Layer-to-layer bonds a[i]-b[i]. ``laws``: list of pm.BondLaw; ``law_index`` picks one per pair."""
        self.n_pairs = len(a)
        flat = _f64([[L.G_n_per_mm, L.delta0_mm, L.delta_c_mm] for L in laws]).ravel()
        lib().sf_set_pairs(self.handle, len(a), _i32(a), _i32(b), _f64(weights), _i32(law_index), len(laws), flat)

    def set_faces(self, stack):
        """Face bonds of an offset stack (``stack.Stack(offset=True)``)."""
        n, t = stack.n, stack.face_tris
        e = stack.face_e
        la, lb = e[:, 0] // n, e[:, 2] // n
        t1, t2 = stack.face_t[:, 0], stack.face_t[:, 1]
        corners = np.concatenate([t[t1] + (la * n)[:, None], t[t2] + (la * n)[:, None],
                                  t[t1] + (lb * n)[:, None], t[t2] + (lb * n)[:, None]], axis=1)
        flat = _f64([[L.G_n_per_mm, L.delta0_mm, L.delta_c_mm] for L in stack.face_laws]).ravel()
        self.n_faces = len(e)
        lib().sf_set_faces(self.handle, len(e), _i32(corners).ravel(), _i32(e).ravel(), _f64(stack.face_oa),
                           _f64(stack.face_ob), _f64(stack.face_w), _i32(stack.face_law_n), _i32(stack.face_law_s),
                           len(stack.face_laws), flat, _f64(getattr(stack, "face_c0", np.zeros((len(e), 3)))).ravel())

    def set_fibres(self, fb):
        """Tension-only fibres (``fibres.Fibres``)."""
        n = len(fb)
        lib().sf_set_fibres(self.handle, n, _i32(fb.a), _i32(fb.b), _f64(fb.origin).ravel(), _f64(fb.rest_len),
                            _f64(fb.area), _f64(fb.E_arr), _f64(fb.k1_arr), _f64(fb.k2_arr))

    def face_state(self):
        hn, hs = np.zeros(self.n_faces), np.zeros(self.n_faces)
        broken = np.zeros(self.n_faces, dtype=np.uint8)
        lib().sf_face_state(self.handle, hn, hs, broken)
        return hn, hs, broken.astype(bool)

    def pair_state(self):
        hist = np.zeros(self.n_pairs)
        broken = np.zeros(self.n_pairs, dtype=np.uint8)
        lib().sf_pair_state(self.handle, hist, broken)
        return hist, broken.astype(bool)

    def set_blade(self, k_per_vertex, blade, floor_z):
        """The wedge test's rigid elevator (``wedge_test.Blade``) on bone at ``floor_z``."""
        lib().sf_set_blade(self.handle, _f64(k_per_vertex), blade.height_mm, blade.slope, blade.heel_mm, floor_z)

    def set_blade_tip(self, tip):
        lib().sf_set_blade_tip(self.handle, float(tip))

    def blade_push(self, x):
        return lib().sf_blade_push(self.handle, _f64(x).ravel())

    def set_bond_broken(self, mask):
        lib().sf_set_bond_broken(self.handle, np.ascontiguousarray(mask, dtype=np.uint8))

    def set_bonds(self, verts, anchors, weights, law, mixed):
        self.n_bonds = len(verts)
        lib().sf_set_bonds(self.handle, len(verts), _i32(verts), _f64(anchors).ravel(), _f64(weights), _f64(law),
                           int(mixed))

    def set_options(self, project_psd=True, threads=1):
        """Newton options: per-element stabilisation and assembly threads."""
        lib().sf_set_options(self.handle, int(project_psd), int(threads))

    def set_factor(self, kind=EIGEN_LDLT):
        """Newton's sparse factorization. Raises if CHOLMOD is asked for but not built in."""
        if not lib().sf_set_factor(self.handle, int(kind)):
            raise RuntimeError("CHOLMOD not available: install SuiteSparse to ~/.local/suitesparse")

    def set_hessian(self, analytic=True):
        """Element Hessians: analytic (default) or finite differences of the exact gradient."""
        lib().sf_set_hessian(self.handle, int(analytic))

    def element_hessians(self, x, psd=True, analytic=True):
        """Yield (kind, vertices, block) for every element, for testing."""
        xf = _f64(x).ravel()
        info = np.zeros(5, dtype=np.int32)
        H = np.zeros(144)
        for k in range(lib().sf_elem_count(self.handle)):
            n = lib().sf_elem_hessian(self.handle, k, xf, int(psd), int(analytic), info, H)
            yield int(info[0]), info[1:1 + n].copy(), H[:9 * n * n].reshape(3 * n, 3 * n).copy()

    def set_fixed(self, mask):
        self.fixed = np.asarray(mask, dtype=bool)
        lib().sf_set_fixed(self.handle, np.ascontiguousarray(self.fixed.ravel(), dtype=np.uint8))

    def set_external(self, f):
        """A set (dead) force per vertex (N, 3) in N, added as energy -f.x; None removes it."""
        if f is None:
            self._fext = None
            lib().sf_set_external(self.handle, None)
            return
        self._fext = np.ascontiguousarray(f, dtype=np.float64).reshape(-1)
        assert len(self._fext) == 3 * self.n
        lib().sf_set_external(self.handle, self._fext.ctypes.data)

    def set_grip(self, k, targets=None):
        """Springs pulling vertices to targets: energy 1/2 k_i |x_i - t_i|^2, ``k`` (N,) in N/mm
        (0 = not gripped), ``targets`` (N, 3). None removes them. Unlike set_fixed this keeps the
        sparsity pattern, so a grab costs nothing extra."""
        if k is None:
            self._grip = None
            lib().sf_set_grip(self.handle, None, None)
            return
        k = np.ascontiguousarray(k, dtype=np.float64).reshape(-1)
        t = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1)
        assert len(k) == self.n and len(t) == 3 * self.n
        self._grip = (k, t)
        lib().sf_set_grip(self.handle, k.ctypes.data, t.ctypes.data)

    def set_obstacle(self, grid, origin, spacing):
        """A static obstacle as a signed distance grid (nx, ny, nz; phi > 0 outside it, mm), its
        first node's position and the node spacing (mm). None removes it. Vertices with contact
        (``set_contact``) are kept outside it."""
        if grid is None:
            self._sdf = None
            lib().sf_set_sdf(self.handle, None, None, None, None)
            return
        g = np.ascontiguousarray(grid, dtype=np.float32)
        n = np.ascontiguousarray(g.shape, dtype=np.int32)
        o, h = _f64(np.asarray(origin, float)), _f64(np.broadcast_to(np.asarray(spacing, float), (3,)))
        self._sdf = (g, n, o, h)
        lib().sf_set_sdf(self.handle, n.ctypes.data, o.ctypes.data, h.ctypes.data, g.ctypes.data)

    def set_contact(self, k, clearance=0.0):
        """Contact with the obstacle: stiffness per vertex (N/mm, 0 = none) and clearance (mm): a
        one-sided penalty 1/2 k (phi - clearance)^2 where phi < clearance. None removes it."""
        if k is None:
            self._contact = None
            lib().sf_set_contact(self.handle, None, None)
            return
        k = _f64(np.broadcast_to(np.asarray(k, float), (self.n,)))
        c = _f64(np.broadcast_to(np.asarray(clearance, float), (self.n,)))
        self._contact = (k, c)
        lib().sf_set_contact(self.handle, k.ctypes.data, c.ctypes.data)

    def set_self_contact(self, k, clearance, on, tris, rings=2, give=0.3):
        """Self-contact: vertices in ``on`` (bool per vertex) keep ``clearance`` mm apart (parts
        already closer at rest: their rest distance less ``give``), stiffness ``k`` N/mm per pair;
        vertices within ``rings`` rings of each other on ``tris`` never count. k = 0 turns it off."""
        if not k:
            lib().sf_set_self_contact(self.handle, 0.0, 0.0, 0.0, None, None, None)
            self._self = None
            return
        ptr, idx = mesh_rings(tris, self.n, rings)
        on = np.ascontiguousarray(on, dtype=np.uint8)
        self._self = (on, ptr, idx)
        lib().sf_set_self_contact(self.handle, float(k), float(clearance), float(give), on.ctypes.data, ptr.ctypes.data,
                                  idx.ctypes.data)

    def set_grip_targets(self, targets):
        t = np.ascontiguousarray(targets, dtype=np.float64).reshape(-1)
        assert len(t) == 3 * self.n
        self._grip = (self._grip[0], t) if self._grip is not None else None
        lib().sf_set_grip_targets(self.handle, t.ctypes.data)

    # -- evaluation ---------------------------------------------------------
    def energy(self, x):
        g = np.zeros(self.n * 3)
        e = lib().sf_energy(self.handle, _f64(x).ravel(), g)
        return e, g.reshape(-1, 3)

    def commit(self, x):
        return lib().sf_commit(self.handle, _f64(x).ravel())

    def bond_state(self):
        hist = np.zeros(self.n_bonds)
        broken = np.zeros(self.n_bonds, dtype=np.uint8)
        lib().sf_bond_state(self.handle, hist, broken)
        return hist, broken.astype(bool)

    def step(self, x, v, h, solver=NEWTON, iters=1, gtol=1e-9, damping=0.0):
        """One frame in place. Fixed dofs of ``x`` must hold this frame's targets."""
        xf = _f64(x).ravel().copy()
        vf = _f64(v).ravel().copy()
        stats = np.zeros(7)
        lib().sf_step(self.handle, xf, vf, h, solver, iters, gtol, damping, stats)
        x[...] = xf.reshape(-1, 3)
        v[...] = vf.reshape(-1, 3)
        return {"ms": float(stats[0]), "iterations": int(stats[1]), "max_grad": float(stats[2]),
                "assemble_ms": float(stats[3]), "factor_ms": float(stats[4]), "search_ms": float(stats[5]),
                "factorizations": int(stats[6])}


_static_models = {}


def static_solve(sheet, x0, fixed, gtol=1e-9, max_iter=200):
    """Drop-in for ``peel_model.solve`` on a single ``pm.Sheet``, using the fast Newton solver.

    The C++ model is cached per sheet object (fixtures rebuild the sheet when
    they cut or break it). Bond damage stays in the sheet: it is pushed into
    the model before each solve.
    """
    from types import SimpleNamespace

    key = id(sheet)
    entry = _static_models.get(key)
    if entry is None or entry[0] is not sheet:
        if len(_static_models) > 8:
            _static_models.clear()
        entry = (sheet, FastModel.from_stack(sheet) if hasattr(sheet, "sheets") else FastModel.from_sheet(sheet))
        _static_models[key] = entry
    m = entry[1]
    if len(getattr(sheet, "bonded", ())):
        opened = sheet.rest.copy()
        opened[sheet.bonded, 2] += sheet.history
        m.commit(opened)
    m.set_fixed(fixed)
    x = np.array(x0, dtype=np.float64, copy=True)
    r = m.step(x, np.zeros_like(x), 0.0, NEWTON, max_iter, gtol=gtol)
    e, g = m.energy(x)
    return x, SimpleNamespace(nit=r["iterations"], energy=e, max_grad=float(np.abs(g[~fixed]).max()) if (~fixed).any() else 0.0)


def triangle_material(p, count):
    """Per-triangle arrays (thick, mu, lam, k1, k2) for one material."""
    mu = p.E_mpa / (2.0 * (1.0 + p.nu))
    lam = p.E_mpa * p.nu / (1.0 - p.nu**2)  # plane stress
    return tuple(np.full(count, v, dtype=np.float64) for v in (p.thickness_mm, mu, lam, p.k1_mpa, p.k2))


def mesh_rings(tris, n, rings=2):
    """Each vertex's neighbours within ``rings`` rings on the mesh (CSR: ptr, sorted idx; int32)."""
    from scipy import sparse

    t = np.asarray(tris)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1
    R = A.copy()
    for _ in range(rings - 1):
        R = (R + R @ A).tocsr()
        R.data[:] = 1
    R = R.tocsr()
    R.sort_indices()
    return np.ascontiguousarray(R.indptr, np.int32), np.ascontiguousarray(R.indices, np.int32)


def have_cholmod():
    return bool(lib().sf_have_cholmod())


def best_factor():
    """CHOLMOD supernodal when built in (2-2.5x faster factoring with OpenBLAS), else Eigen."""
    return CHOLMOD_SUPERNODAL if have_cholmod() else EIGEN_LDLT


def box_tets(lx, ly, lz, edge):
    """Box [0,lx]x[0,ly]x[0,lz] split into cubes of about ``edge``, six tets per cube."""
    n = [max(1, int(round(L / edge))) for L in (lx, ly, lz)]
    xs = [np.linspace(0, L, k + 1) for L, k in zip((lx, ly, lz), n)]
    gx, gy, gz = np.meshgrid(*xs, indexing="ij")
    rest = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    idx = np.arange(rest.shape[0]).reshape(n[0] + 1, n[1] + 1, n[2] + 1)
    tets = []
    # Six tets around the main diagonal of each cube (conforming across faces).
    paths = [(1, 3, 7), (1, 5, 7), (2, 3, 7), (2, 6, 7), (4, 5, 7), (4, 6, 7)]
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = [idx[i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)] for b in range(8)]
                for a, b, d in paths:
                    tets.append((c[0], c[a], c[b], c[d]))
    return rest, np.asarray(tets, dtype=np.int64)
