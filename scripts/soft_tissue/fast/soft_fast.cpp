// Fast soft-tissue solvers for the FIVRS soft-tissue work.
//
// Same energies as the Python reference (scripts/soft_tissue/peel_model.py,
// wedge_test.py), so the reference can referee them:
//   membrane   St Venant-Kirchhoff triangles, plane stress, plus collagen
//              recruitment along the rest axes (J-shaped curve)
//   bending    discrete hinges, energy k (theta - theta0)^2 (theta0: rest angle,
//              0 for a flat sheet; a curved shell such as cartilage keeps its own)
//   volume     St Venant-Kirchhoff tetrahedra
//   bonds      breakable bilinear bonds to fixed anchors, either one law on
//              the full separation, or separate opening (z) and sliding laws
//   floor      one-sided penalty, z >= floor
//   pairs      breakable bilinear bonds between two vertices (layer to layer);
//              layers share one mid-surface, so the rest separation is zero
//   blade      a rigid elevator: ramp of the wedge test, penalty contact
//   shell      plate bending with its Poisson ratio (cartilage), replacing the
//              hinges of its layer: per triangle, its three edges' signed bend angles d,
//              energy d^T K d (K from the reference, peel_model.Sheet bending="shell")
//   faces      layers at their real heights bonded face to face (stack.Stack offset=True):
//              per interior edge, the edge midpoint of each layer pushed along the edge's
//              mean triangle normal by half the layer's thickness; one law on the
//              separation, or opening along the inner normal and sliding across it
//   fibres     tension-only cables (fibres.py: the auricular muscles as fans), insertion
//              vertex to a fixed origin point or another vertex; Green strain along the
//              fibre, W = E e^2/2 + collagen (J-curve), energy A L0 W, nothing when slack
//
// Each frame minimises the implicit-Euler incremental potential
//   m/(2 h^2) |x - x_pred|^2 + E(x)
// (or E(x) alone when h <= 0, quasi-static) with one of two solvers:
//   Newton   sparse Hessian (element blocks projected to positive
//            semi-definite, hinges Gauss-Newton), LDLT, Armijo line search
//   XPBD     Gauss-Seidel over elements, each element energy treated as one
//            constraint C = sqrt(2 E_e) with compliance 1
//
// Units: mm, ms, g, N (consistent: 1 g mm / ms^2 = 1 N).
// Plain C interface for ctypes. Needs Eigen; when built with -DSF_CHOLMOD the
// Newton step can also factor with CHOLMOD's supernodal Cholesky (SuiteSparse).

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#ifdef SF_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

double now_ms_impl();
inline double now_ms() { return now_ms_impl(); }

struct Law {
    double G = 0, d0 = 1, dc = 2;
    double Tmax() const { return 2.0 * G / dc; }
    double k() const { return Tmax() / d0; }
};

double traction(double d, const Law& L) {
    if (d <= L.d0) return L.k() * d;
    if (d < L.dc) return L.Tmax() * (L.dc - d) / (L.dc - L.d0);
    return 0.0;
}

double envelope(double d, const Law& L) {
    if (d <= L.d0) return 0.5 * L.k() * d * d;
    const double m = std::min(d, L.dc), tm = L.Tmax();
    return 0.5 * tm * L.d0 + tm / (L.dc - L.d0) * (L.dc * (m - L.d0) - 0.5 * (m * m - L.d0 * L.d0));
}

// One damaged bilinear channel: energy and slope at separation s.
void channel(double s, double hist, const Law& L, double& psi, double& dpsi) {
    const double m = std::max(hist, L.d0);
    const double ks = traction(m, L) / m;
    if (s <= m) {
        psi = 0.5 * ks * s * s;
        dpsi = ks * s;
    } else {
        psi = 0.5 * ks * m * m + envelope(s, L) - envelope(m, L);
        dpsi = traction(s, L);
    }
}

using SpMat = Eigen::SparseMatrix<double>;

// Sparse factorization behind one interface, so the solver can be swapped.
struct Factor {
    virtual ~Factor() = default;
    virtual void analyze(const SpMat& A) = 0;
    virtual bool factorize(const SpMat& A) = 0;  // false: not positive definite
    virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;
};

template <class S>
struct FactorT : Factor {
    S s;
    void analyze(const SpMat& A) override { s.analyzePattern(A); }
    bool factorize(const SpMat& A) override {
        s.factorize(A);
        return s.info() == Eigen::Success;
    }
    Eigen::VectorXd solve(const Eigen::VectorXd& b) override { return s.solve(b); }
};

enum FactorKind { F_EIGEN_LDLT = 0, F_CHOLMOD_SUPER = 1, F_CHOLMOD_SIMPL = 2 };

bool have_cholmod() {
#ifdef SF_CHOLMOD
    return true;
#else
    return false;
#endif
}

std::unique_ptr<Factor> make_factor(int kind) {
#ifdef SF_CHOLMOD
    // print = 0: a non-positive-definite try is normal (Newton raises its damping and retries).
    if (kind == F_CHOLMOD_SUPER) {
        auto f = std::make_unique<FactorT<Eigen::CholmodSupernodalLLT<SpMat, Eigen::Lower>>>();
        f->s.cholmod().print = 0;
        return f;
    }
    if (kind == F_CHOLMOD_SIMPL) {
        auto f = std::make_unique<FactorT<Eigen::CholmodSimplicialLDLT<SpMat, Eigen::Lower>>>();
        f->s.cholmod().print = 0;
        return f;
    }
#endif
    (void)kind;
    return std::make_unique<FactorT<Eigen::SimplicialLDLT<SpMat>>>();
}

// Collagen energy density for Green strain e >= 0, and its first two derivatives.
// Past k2 e^2 = kCollagenCap it continues as its quadratic Taylor expansion, so an
// overstretched element has a finite energy. Identical to peel_model.collagen.
constexpr double kCollagenCap = 8.0;
void collagen(double e, double k1, double k2, double& w, double& dw, double& d2w) {
    w = dw = d2w = 0;
    if (e <= 0 || k1 <= 0 || k2 <= 0) return;
    const double ec = std::sqrt(kCollagenCap / k2), m = std::min(e, ec), grow = std::exp(k2 * m * m);
    const double w0 = k1 / (2.0 * k2) * (grow - 1.0), dw0 = k1 * m * grow, d2 = k1 * grow * (1.0 + 2.0 * k2 * m * m);
    const double d = e - m;
    w = w0 + dw0 * d + 0.5 * d2 * d * d;
    dw = dw0 + d2 * d;
    d2w = d2;
}

enum Kind { TRI = 0, HINGE = 1, TET = 2, BOND = 3, FLOOR = 4, PAIR = 5, BLADE = 6, SHELL = 7, FACE = 8, FIBRE = 9, CONTACT = 10 };

constexpr int MAXV = 8;  // largest element stencil (FACE)

struct Elem {
    int kind;
    int index;  // into the kind's arrays
    int n;      // vertices used
    int v[MAXV];
};

struct Model {
    int nv = 0;
    std::vector<double> rest, mass;
    // triangles
    std::vector<int> tri;
    std::vector<double> tri_dminv, tri_area;
    std::vector<double> tri_thick, tri_mu, tri_lam, tri_k1, tri_k2;  // per triangle (layers)
    // hinges
    std::vector<int> hinge;
    std::vector<double> hinge_k, hinge_rest;
    // shells (replace the hinges when present)
    std::vector<int> shell_hinge, shell_n, shell_v, shell_map;  // per triangle: 3 hinges, stencil size, 6 verts, 3x4 slots
    std::vector<double> shell_K, hinge_sign;
    std::vector<uint8_t> hinge_in_shell;  // hinges replaced by shells (the rest stay hinges)
    // face bonds (offset layers): per bond its 4 triangles' corners (layer a: 2, layer b: 2; global),
    // its edge (a: 2, b: 2), offsets, area, laws (sliding -1: one law), damage
    std::vector<int> face_tri, face_edge, face_ln, face_ls, face_v, face_slot;  // slot: 18 tri corners + 4 edge verts
    std::vector<double> face_oa, face_ob, face_w, face_hn, face_hs, face_c0;  // c0: rest separation in the bond frame
    std::vector<uint8_t> face_broken;
    std::vector<Law> face_laws;
    // fibres: insertion a, origin b (-1: fixed point), origin point, rest length, area, material
    std::vector<int> fib_a, fib_b;
    std::vector<double> fib_origin, fib_L0, fib_A, fib_E, fib_k1, fib_k2;
    // tets
    std::vector<int> tet;
    std::vector<double> tet_dminv, tet_vol;
    double tmu = 0, tlam = 0;
    // bonds
    std::vector<int> bond_v;
    std::vector<double> bond_anchor, bond_w, hist_n, hist_s;
    std::vector<uint8_t> broken;
    bool mixed = false;
    Law opening, sliding;
    // layer-to-layer bonds
    std::vector<int> pair_a, pair_b, pair_law;
    std::vector<double> pair_w, pair_hist;
    std::vector<uint8_t> pair_broken;
    std::vector<Law> pair_laws;
    // blade (rigid elevator)
    std::vector<double> blade_k;
    double blade_tip = 0, blade_height = 0.5, blade_slope = 1, blade_heel = 0.5, blade_floor = 0;
    // floor
    std::vector<double> floor_k;
    double floor_z = -1e30;
    // contact with a static obstacle given as a signed distance grid (phi > 0 outside it): one-sided
    // penalty 1/2 k (phi(x) - off)^2 where phi(x) < off, per vertex (k 0: no contact)
    std::vector<float> sdf;
    int sdf_n[3] = {0, 0, 0};
    double sdf_o[3] = {0, 0, 0}, sdf_h[3] = {1, 1, 1};
    std::vector<double> contact_k, contact_off;
    // self-contact: vertex pairs (found each frame by a spatial hash) closer than their clearance
    // are pushed apart, 1/2 k (clear - d)^2; clear = min(d0, rest distance - give), so parts already
    // close at rest may come no closer than they were less ``give``. Pairs within a few rings of each
    // other on the mesh (self_ring CSR) never count. The Hessian is the diagonal k I per vertex in
    // contact (like the grip), so contacts coming and going never change the sparsity pattern.
    std::vector<uint8_t> self_on;
    double self_k = 0, self_d0 = 0, self_give = 0.3;
    std::vector<int> self_ring_ptr, self_ring_idx, self_pi, self_pj;
    std::vector<double> self_clear;
    // fixed degrees of freedom
    std::vector<uint8_t> fixed;
    // applied (dead) load per dof: energy - f.x (a set force at a grip; empty: none)
    std::vector<double> fext;
    // grip springs: energy 1/2 k_i |x_i - t_i|^2 per vertex (a finger pad; empty: none). Only the
    // Hessian diagonal changes, so grabbing does not rebuild the sparsity pattern.
    std::vector<double> grip_k, grip_t;
    // derived
    std::vector<Elem> elems;
    std::vector<double> lambda;  // XPBD multipliers
    std::vector<double> xlast;   // positions at the end of the last frame
    bool project_psd = true;     // project element Hessians to positive semi-definite
    bool analytic = true;        // analytic element Hessians (else finite differences)
    int threads = 1;             // threads for Hessian assembly
    double t_assemble = 0, t_factor = 0, t_search = 0;  // ms in the last frame
    int n_factor = 0;                                    // factorizations in the last frame
    // Newton
    std::vector<int> dof_map;  // full dof -> free index or -1
    int nfree = 0;
    bool pattern_ready = false;
    int factor_kind = F_EIGEN_LDLT;
    std::unique_ptr<Factor> factor;
    // Fixed sparsity pattern: where each element-Hessian entry lands.
    Eigen::SparseMatrix<double> K;
    std::vector<int> elem_off;   // offset of each element's block in hbuf / slot
    std::vector<int> slot;       // value index in K, or -1 for a fixed dof
    std::vector<int> diag_slot;  // value index of each free dof's diagonal
    std::vector<double> hbuf;
    // Gather lists: K value v sums hbuf[gather_idx[gather_ptr[v] .. gather_ptr[v+1])]
    std::vector<int> gather_ptr, gather_idx;
};

inline Vec3 P(const double* x, int i) { return Vec3(x[3 * i], x[3 * i + 1], x[3 * i + 2]); }

// ---- element energies: energy, and gradient into g (n*3 entries) ------------

double tri_eg(const Model& M, int t, const double* x, double* g) {
    const int* v = &M.tri[3 * t];
    Eigen::Matrix<double, 3, 2> Ds;
    Ds.col(0) = P(x, v[1]) - P(x, v[0]);
    Ds.col(1) = P(x, v[2]) - P(x, v[0]);
    Eigen::Map<const Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> Dinv(&M.tri_dminv[4 * t]);
    const Eigen::Matrix<double, 3, 2> F = Ds * Dinv;
    const Eigen::Matrix2d E = 0.5 * (F.transpose() * F - Eigen::Matrix2d::Identity());
    const double tr = E.trace();
    const double vol = M.tri_area[t] * M.tri_thick[t];
    const double mu = M.tri_mu[t], lam = M.tri_lam[t], k1 = M.tri_k1[t], k2 = M.tri_k2[t];
    double energy = vol * (mu * (E.array() * E.array()).sum() + 0.5 * lam * tr * tr);
    Eigen::Matrix2d S = 2.0 * mu * E + lam * tr * Eigen::Matrix2d::Identity();
    if (k1 > 0 && k2 > 0) {
        for (int a = 0; a < 2; ++a) {
            const double ea = std::max(E(a, a), 0.0);
            double w, dw, d2w;
            collagen(ea, k1, k2, w, dw, d2w);
            energy += vol * w;
            S(a, a) += dw;
        }
    }
    if (g) {
        const Eigen::Matrix<double, 3, 2> H = vol * (F * S * Dinv.transpose());
        for (int k = 0; k < 3; ++k) {
            g[3 + k] = H(k, 0);
            g[6 + k] = H(k, 1);
            g[k] = -H(k, 0) - H(k, 1);
        }
    }
    return energy;
}

// Signed hinge angle and d(theta)/dx for slots (opp a, opp b, edge 3, edge 4).
double dihedral(const double* x, const int* h, Vec3* dtheta) {
    const Vec3 x1 = P(x, h[0]), x2 = P(x, h[1]), x3 = P(x, h[2]), x4 = P(x, h[3]);
    const Vec3 e = x4 - x3;
    const double el = std::max(e.norm(), 1e-12);
    const Vec3 n1 = (x1 - x3).cross(x1 - x4);
    const Vec3 n2 = (x2 - x4).cross(x2 - x3);
    const double n1s = std::max(n1.squaredNorm(), 1e-24), n2s = std::max(n2.squaredNorm(), 1e-24);
    const Vec3 n1h = n1 / std::sqrt(n1s), n2h = n2 / std::sqrt(n2s);
    const Vec3 eh = e / el;
    const double theta = std::atan2(n1h.cross(n2h).dot(eh), n1h.dot(n2h));
    if (dtheta) {
        const Vec3 w1 = n1 / n1s, w2 = n2 / n2s;
        dtheta[0] = -(el * w1);
        dtheta[1] = -(el * w2);
        dtheta[2] = -((x1 - x4).dot(eh) * w1 + (x2 - x4).dot(eh) * w2);
        dtheta[3] = -(-(x1 - x3).dot(eh) * w1 - (x2 - x3).dot(eh) * w2);
    }
    return theta;
}

double hinge_eg(const Model& M, int h, const double* x, double* g) {
    Vec3 d[4];
    double th = dihedral(x, &M.hinge[4 * h], g ? d : nullptr);
    if (!M.hinge_rest.empty()) {  // from the rest angle, wrapped into (-pi, pi]
        th -= M.hinge_rest[h];
        th = std::fmod(th + M_PI, 2.0 * M_PI);
        if (th < 0) th += 2.0 * M_PI;
        th -= M_PI;
    }
    const double k = M.hinge_k[h];
    if (g)
        for (int s = 0; s < 4; ++s)
            for (int c = 0; c < 3; ++c) g[3 * s + c] = 2.0 * k * th * d[s](c);
    return k * th * th;
}

// Shell bending of one triangle: d_i = sign * (theta_i - theta0_i) on its edges' hinges, E = d^T K d.
inline double wrap_angle(double a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0) a += 2.0 * M_PI;
    return a - M_PI;
}

double shell_eg(const Model& M, int t, const double* x, double* g, Vec3 (*dout)[4] = nullptr, double* dsign = nullptr) {
    double d[3] = {0, 0, 0};
    Vec3 dd[3][4];
    for (int i = 0; i < 3; ++i) {
        const int h = M.shell_hinge[3 * t + i];
        if (h < 0) continue;
        double th = dihedral(x, &M.hinge[4 * h], dd[i]);
        if (!M.hinge_rest.empty()) th -= M.hinge_rest[h];
        d[i] = M.hinge_sign[h] * wrap_angle(th);
    }
    const double* K = &M.shell_K[9 * t];
    double kd[3];
    for (int i = 0; i < 3; ++i) kd[i] = K[3 * i] * d[0] + K[3 * i + 1] * d[1] + K[3 * i + 2] * d[2];
    const double energy = d[0] * kd[0] + d[1] * kd[1] + d[2] * kd[2];
    const int n = M.shell_n[t];
    if (g) {
        std::fill(g, g + 3 * n, 0.0);
        for (int i = 0; i < 3; ++i) {
            const int h = M.shell_hinge[3 * t + i];
            if (h < 0) continue;
            const double coef = 2.0 * kd[i] * M.hinge_sign[h];
            for (int sl = 0; sl < 4; ++sl) {
                const int pos = M.shell_map[12 * t + 4 * i + sl];
                for (int c = 0; c < 3; ++c) g[3 * pos + c] += coef * dd[i][sl](c);
            }
        }
    }
    if (dout)
        for (int i = 0; i < 3; ++i)
            for (int sl = 0; sl < 4; ++sl) dout[i][sl] = dd[i][sl];
    if (dsign)
        for (int i = 0; i < 3; ++i) {
            const int h = M.shell_hinge[3 * t + i];
            dsign[i] = h < 0 ? 0.0 : M.hinge_sign[h];
        }
    return energy;
}

double tet_eg(const Model& M, int t, const double* x, double* g) {
    const int* v = &M.tet[4 * t];
    Mat3 Ds;
    for (int c = 0; c < 3; ++c) Ds.col(c) = P(x, v[c + 1]) - P(x, v[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Dinv(&M.tet_dminv[9 * t]);
    const Mat3 F = Ds * Dinv;
    const Mat3 E = 0.5 * (F.transpose() * F - Mat3::Identity());
    const double tr = E.trace();
    const double vol = M.tet_vol[t];
    const double energy = vol * (M.tmu * (E.array() * E.array()).sum() + 0.5 * M.tlam * tr * tr);
    if (g) {
        const Mat3 S = 2.0 * M.tmu * E + M.tlam * tr * Mat3::Identity();
        const Mat3 H = vol * (F * S * Dinv.transpose());
        for (int k = 0; k < 3; ++k) {
            double sum = 0;
            for (int c = 0; c < 3; ++c) {
                g[3 * (c + 1) + k] = H(k, c);
                sum += H(k, c);
            }
            g[k] = -sum;
        }
    }
    return energy;
}

double bond_eg(const Model& M, int b, const double* x, double* g) {
    if (M.broken[b]) {
        if (g) g[0] = g[1] = g[2] = 0;
        return 0.0;
    }
    const Vec3 r = P(x, M.bond_v[b]) - Vec3(M.bond_anchor[3 * b], M.bond_anchor[3 * b + 1], M.bond_anchor[3 * b + 2]);
    const double w = M.bond_w[b];
    if (!M.mixed) {
        const double d = r.norm();
        double psi, dpsi;
        channel(d, M.hist_n[b], M.opening, psi, dpsi);
        if (g) {
            // inside the history the slope is ks*d, so dpsi/d is finite at d = 0
            const double m = std::max(M.hist_n[b], M.opening.d0);
            const double scale = (d <= m) ? traction(m, M.opening) / m : dpsi / std::max(d, 1e-12);
            for (int c = 0; c < 3; ++c) g[c] = w * scale * r(c);
        }
        return w * psi;
    }
    const double dn = r(2), open = std::max(dn, 0.0);
    const double s = std::hypot(r(0), r(1));
    double pn, dpn, ps, dps;
    channel(open, M.hist_n[b], M.opening, pn, dpn);
    channel(s, M.hist_s[b], M.sliding, ps, dps);
    if (g) {
        const double m = std::max(M.hist_s[b], M.sliding.d0);
        const double scale = (s <= m) ? traction(m, M.sliding) / m : dps / std::max(s, 1e-12);
        g[0] = w * scale * r(0);
        g[1] = w * scale * r(1);
        g[2] = (dn > 0) ? w * dpn : 0.0;
    }
    return w * (pn + ps);
}

// trilinear phi and its gradient at p; past the grid's edge phi carries on linearly (its edge value
// plus the gradient times the distance out), so a vertex that overshoots still feels the obstacle
double sdf_sample(const Model& M, const double* p, double* grad) {
    int i0[3];
    double f[3], out[3];
    for (int a = 0; a < 3; ++a) {
        const double u_raw = (p[a] - M.sdf_o[a]) / M.sdf_h[a];
        double u = std::min(std::max(u_raw, 0.0), double(M.sdf_n[a] - 1) - 1e-9);
        out[a] = (u_raw - u) * M.sdf_h[a];
        i0[a] = std::min(int(u), M.sdf_n[a] - 2);
        f[a] = u - i0[a];
    }
    auto at = [&](int x, int y, int z) {
        return double(M.sdf[(size_t(x) * M.sdf_n[1] + y) * M.sdf_n[2] + z]);
    };
    double c[2][2][2];
    for (int dx = 0; dx < 2; ++dx)
        for (int dy = 0; dy < 2; ++dy)
            for (int dz = 0; dz < 2; ++dz) c[dx][dy][dz] = at(i0[0] + dx, i0[1] + dy, i0[2] + dz);
    const double fx = f[0], fy = f[1], fz = f[2];
    auto lerp = [](double a, double b, double t) { return a + (b - a) * t; };
    const double c00 = lerp(c[0][0][0], c[1][0][0], fx), c10 = lerp(c[0][1][0], c[1][1][0], fx);
    const double c01 = lerp(c[0][0][1], c[1][0][1], fx), c11 = lerp(c[0][1][1], c[1][1][1], fx);
    const double c0 = lerp(c00, c10, fy), c1 = lerp(c01, c11, fy);
    if (grad) {
        const double dx00 = c[1][0][0] - c[0][0][0], dx10 = c[1][1][0] - c[0][1][0];
        const double dx01 = c[1][0][1] - c[0][0][1], dx11 = c[1][1][1] - c[0][1][1];
        grad[0] = lerp(lerp(dx00, dx10, fy), lerp(dx01, dx11, fy), fz) / M.sdf_h[0];
        grad[1] = lerp(c10 - c00, c11 - c01, fz) / M.sdf_h[1];
        grad[2] = (c1 - c0) / M.sdf_h[2];
    }
    double g_[3];
    if (!grad) {  // the extrapolation needs the gradient
        double tmp[3];
        const double dx00 = c[1][0][0] - c[0][0][0], dx10 = c[1][1][0] - c[0][1][0];
        const double dx01 = c[1][0][1] - c[0][0][1], dx11 = c[1][1][1] - c[0][1][1];
        tmp[0] = lerp(lerp(dx00, dx10, fy), lerp(dx01, dx11, fy), fz) / M.sdf_h[0];
        tmp[1] = lerp(c10 - c00, c11 - c01, fz) / M.sdf_h[1];
        tmp[2] = (c1 - c0) / M.sdf_h[2];
        g_[0] = tmp[0], g_[1] = tmp[1], g_[2] = tmp[2];
    } else {
        g_[0] = grad[0], g_[1] = grad[1], g_[2] = grad[2];
    }
    return lerp(c0, c1, fz) + g_[0] * out[0] + g_[1] * out[1] + g_[2] * out[2];
}

double contact_eg(const Model& M, int i, const double* x, double* g) {
    double gr[3];
    const double gap = sdf_sample(M, x + 3 * i, gr) - M.contact_off[i];
    const double k = M.contact_k[i];
    if (gap >= 0) {
        if (g) g[0] = g[1] = g[2] = 0;
        return 0;
    }
    if (g)
        for (int c = 0; c < 3; ++c) g[c] = k * gap * gr[c];
    return 0.5 * k * gap * gap;
}

double floor_eg(const Model& M, int i, const double* x, double* g) {
    const double below = std::min(x[3 * i + 2] - M.floor_z, 0.0);
    const double k = M.floor_k[i];
    if (g) {
        g[0] = g[1] = 0;
        g[2] = k * below;
    }
    return 0.5 * k * below * below;
}

double pair_eg(const Model& M, int p, const double* x, double* g) {
    if (M.pair_broken[p]) {
        if (g) std::fill(g, g + 6, 0.0);
        return 0.0;
    }
    const Law& L = M.pair_laws[M.pair_law[p]];
    const Vec3 r = P(x, M.pair_a[p]) - P(x, M.pair_b[p]);
    const double d = r.norm(), w = M.pair_w[p];
    double psi, dpsi;
    channel(d, M.pair_hist[p], L, psi, dpsi);
    if (g) {
        const double m = std::max(M.pair_hist[p], L.d0);
        const double scale = (d <= m) ? traction(m, L) / m : dpsi / std::max(d, 1e-12);
        for (int c = 0; c < 3; ++c) {
            g[c] = w * scale * r(c);
            g[3 + c] = -g[c];
        }
    }
    return w * psi;
}

// ---- face bonds -------------------------------------------------------------
double channel_stiffness(double s, double hist, const Law& L);
template <int N>
void radial_hessian(const Eigen::Matrix<double, N, 1>& r, double hist, const Law& L, double w, bool psd,
                    Eigen::Matrix<double, N, N>& H);
// Unit normal of triangle (corners c), with its raw length and edges.
struct TriN {
    Vec3 n, e1, e2;
    double len;
};
inline TriN tri_normal(const double* x, const int* c) {
    TriN t;
    t.e1 = P(x, c[1]) - P(x, c[0]);
    t.e2 = P(x, c[2]) - P(x, c[0]);
    const Vec3 N = t.e1.cross(t.e2);
    t.len = N.norm();
    t.n = N / t.len;
    return t;
}
inline Mat3 skew(const Vec3& v) {
    Mat3 S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}

// Separation r of bond f and the inner layer's edge normal; side normals for the derivatives.
struct FaceGeo {
    TriN t[4];         // a1, a2, b1, b2
    Vec3 na, nb, Ma, Mb;
    double la, lb;     // |sum of unit normals|
    Vec3 te;           // unit direction of the inner edge
    double le;         // its length
    Vec3 r;            // separation less the rest separation (in the bond frame te, nb, te x nb)
};
FaceGeo face_geo(const Model& M, int f, const double* x) {
    FaceGeo G;
    for (int k = 0; k < 4; ++k) G.t[k] = tri_normal(x, &M.face_tri[12 * f + 3 * k]);
    G.Ma = G.t[0].n + G.t[1].n;
    G.Mb = G.t[2].n + G.t[3].n;
    G.la = G.Ma.norm();
    G.lb = G.Mb.norm();
    G.na = G.Ma / G.la;
    G.nb = G.Mb / G.lb;
    const int* e = &M.face_edge[4 * f];
    const Vec3 pa = 0.5 * (P(x, e[0]) + P(x, e[1])) + M.face_oa[f] * G.na;
    const Vec3 pb = 0.5 * (P(x, e[2]) + P(x, e[3])) + M.face_ob[f] * G.nb;
    const Vec3 ev = P(x, e[3]) - P(x, e[2]);
    G.le = ev.norm();
    G.te = ev / G.le;
    const double* c0 = &M.face_c0[3 * f];
    G.r = pa - pb - (c0[0] * G.te + c0[1] * G.nb + c0[2] * G.te.cross(G.nb));
    return G;
}

// Energy (unweighted) and dE/dr, dE/dnb of bond f at separation r.
double face_law(const Model& M, int f, const Vec3& r, const Vec3& nb, Vec3& gr, Vec3& gn) {
    gr.setZero();
    gn.setZero();
    const Law& Ln = M.face_laws[M.face_ln[f]];
    if (M.face_ls[f] < 0) {
        const double d = r.norm();
        double psi, dpsi;
        channel(d, M.face_hn[f], Ln, psi, dpsi);
        if (d > 0) gr = dpsi / d * r;
        return psi;
    }
    const Law& Ls = M.face_laws[M.face_ls[f]];
    const double dn = r.dot(nb);
    const Vec3 sv = r - dn * nb;
    const double ds = sv.norm();
    double pn, dpn, ps, dps;
    if (dn < 0) {
        const double k0 = Ln.Tmax() / Ln.d0;
        pn = 0.5 * k0 * dn * dn;
        dpn = k0 * dn;
    } else {
        channel(dn, M.face_hn[f], Ln, pn, dpn);
    }
    channel(ds, M.face_hs[f], Ls, ps, dps);
    const double sc = ds > 0 ? dps / ds : 0.0;
    gr = dpn * nb + sc * sv;
    gn = dpn * r - sc * dn * sv;
    return pn + ps;
}

// Scatter dE/d(unit sum normal) of one side through its two triangles into the stencil gradient.
void face_normal_grad(const Model& M, int f, int side, const FaceGeo& G, const Vec3& gnormal, double* g) {
    const Vec3 nE = side == 0 ? G.na : G.nb;
    const double lm = side == 0 ? G.la : G.lb;
    const Vec3 gM = (gnormal - gnormal.dot(nE) * nE) / lm;
    for (int k = 2 * side; k < 2 * side + 2; ++k) {
        const TriN& T = G.t[k];
        const Vec3 q = (gM - gM.dot(T.n) * T.n) / T.len;
        const Vec3 g1 = T.e2.cross(q), g2 = q.cross(T.e1);
        const int* sl = &M.face_slot[22 * f + 3 * k];
        for (int c = 0; c < 3; ++c) {
            g[3 * sl[1] + c] += g1(c);
            g[3 * sl[2] + c] += g2(c);
            g[3 * sl[0] + c] -= g1(c) + g2(c);
        }
    }
}

double face_eg(const Model& M, int f, const double* x, double* g) {
    const int n = 8;
    if (M.face_broken[f]) {
        if (g) std::fill(g, g + 3 * n, 0.0);
        return 0.0;
    }
    const FaceGeo G = face_geo(M, f, x);
    Vec3 gr, gn;
    const double w = M.face_w[f];
    const double psi = face_law(M, f, G.r, G.nb, gr, gn);
    if (g) {
        std::fill(g, g + 3 * n, 0.0);
        gr *= w;
        gn *= w;
        const int* sl = &M.face_slot[22 * f + 12];
        for (int c = 0; c < 3; ++c) {
            g[3 * sl[0] + c] += 0.5 * gr(c);
            g[3 * sl[1] + c] += 0.5 * gr(c);
            g[3 * sl[2] + c] -= 0.5 * gr(c);
            g[3 * sl[3] + c] -= 0.5 * gr(c);
        }
        const double* c0 = &M.face_c0[3 * f];
        face_normal_grad(M, f, 0, G, M.face_oa[f] * gr, g);
        face_normal_grad(M, f, 1, G, -M.face_ob[f] * gr + gn - (c0[1] * gr + c0[2] * gr.cross(G.te)), g);
        if (c0[0] != 0 || c0[2] != 0) {  // the rest frame turns with the inner edge
            const Vec3 gt = -(c0[0] * gr + c0[2] * G.nb.cross(gr));
            const Vec3 ge = (gt - gt.dot(G.te) * G.te) / G.le;
            for (int c = 0; c < 3; ++c) {
                g[3 * sl[3] + c] += ge(c);
                g[3 * sl[2] + c] -= ge(c);
            }
        }
    }
    return w * psi;
}

// Gauss-Newton: J^T H_law J, J = dr/dx on the stencil (the inner normal's turning in the law's
// split is left out). H_law: one law radial; mixed: opening along nb, sliding radial across it.
void face_hessian(const Model& M, int f, const double* x, bool psd, Eigen::MatrixXd& Hout) {
    const int n = 8;
    Hout.setZero(3 * n, 3 * n);
    if (M.face_broken[f]) return;
    const FaceGeo G = face_geo(M, f, x);
    Eigen::Matrix<double, 3, 24> J = Eigen::Matrix<double, 3, 24>::Zero();
    const int* sl = &M.face_slot[22 * f];
    for (int k = 0; k < 4; ++k) {
        J.block<3, 3>(0, 3 * sl[12 + k]) += (k < 2 ? 0.5 : -0.5) * Mat3::Identity();
    }
    const double* c0 = &M.face_c0[3 * f];
    for (int side = 0; side < 2; ++side) {
        const Vec3 nE = side == 0 ? G.na : G.nb;
        const double lm = side == 0 ? G.la : G.lb;
        // dr/dn of this side: oa I, or -(ob + c0y) I - c0z [te]x (the rest frame turns with nb)
        const Mat3 O = side == 0 ? Mat3(M.face_oa[f] * Mat3::Identity())
                                 : Mat3(-(M.face_ob[f] + c0[1]) * Mat3::Identity() - c0[2] * skew(G.te));
        const Mat3 PE = (Mat3::Identity() - nE * nE.transpose()) / lm;
        for (int k = 2 * side; k < 2 * side + 2; ++k) {
            const TriN& T = G.t[k];
            const Mat3 A = O * PE * (Mat3::Identity() - T.n * T.n.transpose()) / T.len;
            const Mat3 d1 = -A * skew(T.e2), d2 = A * skew(T.e1);
            J.block<3, 3>(0, 3 * sl[3 * k + 1]) += d1;
            J.block<3, 3>(0, 3 * sl[3 * k + 2]) += d2;
            J.block<3, 3>(0, 3 * sl[3 * k]) -= d1 + d2;
        }
    }
    if (c0[0] != 0 || c0[2] != 0) {  // and with the inner edge's direction
        const Mat3 Dt = -(c0[0] * Mat3::Identity() - c0[2] * skew(G.nb)) * (Mat3::Identity() - G.te * G.te.transpose()) / G.le;
        J.block<3, 3>(0, 3 * sl[15]) += Dt;
        J.block<3, 3>(0, 3 * sl[14]) -= Dt;
    }
    Mat3 H;
    const double w = M.face_w[f];
    const Law& Ln = M.face_laws[M.face_ln[f]];
    if (M.face_ls[f] < 0) {
        radial_hessian<3>(G.r, M.face_hn[f], Ln, w, psd, H);
    } else {
        const Vec3 nb = G.nb;
        const double dn = G.r.dot(nb);
        Vec3 t1 = nb.unitOrthogonal(), t2 = nb.cross(t1);
        Eigen::Matrix<double, 3, 2> Tm;
        Tm.col(0) = t1;
        Tm.col(1) = t2;
        Eigen::Matrix2d Hs;
        radial_hessian<2>(Eigen::Vector2d(G.r.dot(t1), G.r.dot(t2)), M.face_hs[f], M.face_laws[M.face_ls[f]], w, psd, Hs);
        double kn = dn < 0 ? Ln.Tmax() / Ln.d0 : channel_stiffness(dn, M.face_hn[f], Ln);
        if (psd) kn = std::max(kn, 0.0);
        H = Tm * Hs * Tm.transpose() + w * kn * nb * nb.transpose();
    }
    Hout = J.transpose() * H * J;
}

// ---- fibres -----------------------------------------------------------------
Vec3 fibre_vec(const Model& M, int f, const double* x) {
    const Vec3 far = M.fib_b[f] >= 0 ? P(x, M.fib_b[f])
                                      : Vec3(M.fib_origin[3 * f], M.fib_origin[3 * f + 1], M.fib_origin[3 * f + 2]);
    return P(x, M.fib_a[f]) - far;
}

// W, W', W'' of fibre f at Green strain e (zero in compression).
void fibre_law(const Model& M, int f, double e, double& w, double& dw, double& d2w) {
    w = dw = d2w = 0;
    if (e <= 0) return;
    double cw, cdw, cd2;
    collagen(e, M.fib_k1[f], M.fib_k2[f], cw, cdw, cd2);
    w = 0.5 * M.fib_E[f] * e * e + cw;
    dw = M.fib_E[f] * e + cdw;
    d2w = M.fib_E[f] + cd2;
}

double fibre_eg(const Model& M, int f, const double* x, double* g) {
    const Vec3 d = fibre_vec(M, f, x);
    const double L0 = M.fib_L0[f], e = 0.5 * (d.squaredNorm() / (L0 * L0) - 1.0);
    double w, dw, d2w;
    fibre_law(M, f, e, w, dw, d2w);
    const double vol = M.fib_A[f] * L0;
    if (g) {
        const Vec3 gv = vol * dw / (L0 * L0) * d;
        for (int c = 0; c < 3; ++c) {
            g[c] = gv(c);
            if (M.fib_b[f] >= 0) g[3 + c] = -gv(c);
        }
    }
    return vol * w;
}

void fibre_hessian(const Model& M, int f, const double* x, Eigen::MatrixXd& Hout) {
    const int n = M.fib_b[f] >= 0 ? 2 : 1;
    Hout.setZero(3 * n, 3 * n);
    const Vec3 d = fibre_vec(M, f, x);
    const double L0 = M.fib_L0[f], e = 0.5 * (d.squaredNorm() / (L0 * L0) - 1.0);
    double w, dw, d2w;
    fibre_law(M, f, e, w, dw, d2w);
    if (e <= 0) return;
    const double c = M.fib_A[f] / L0;
    const Mat3 H = c * (d2w / (L0 * L0) * d * d.transpose() + dw * Mat3::Identity());  // both terms >= 0 in tension
    Hout.block<3, 3>(0, 0) = H;
    if (n == 2) {
        Hout.block<3, 3>(3, 3) = H;
        Hout.block<3, 3>(0, 3) = -H;
        Hout.block<3, 3>(3, 0) = -H;
    }
}

// Blade top face: height a distance b behind the tip, its slope and curvature.
void blade_rise(const Model& M, double b, double& h, double& dh, double& d2h) {
    h = dh = d2h = 0;
    if (b <= 0) return;
    const double s = M.blade_slope, L = M.blade_height / s;
    const double f = std::max(std::min(M.blade_heel, L), 1e-12);
    const double b1 = L - 0.5 * f, b2 = L + 0.5 * f;
    if (b < b1) {
        h = s * b, dh = s;
    } else if (b < b2) {
        h = s * b - s * (b - b1) * (b - b1) / (2.0 * f), dh = s - s * (b - b1) / f, d2h = -s / f;
    } else {
        h = M.blade_height;
    }
}

// gap = z - floor - rise(tip - x); energy k gap^2 / 2 where gap < 0.
double blade_eg(const Model& M, int i, const double* x, double* g) {
    double h, dh, d2h;
    blade_rise(M, M.blade_tip - x[3 * i], h, dh, d2h);
    const double gap = x[3 * i + 2] - M.blade_floor - h;
    const double k = M.blade_k[i];
    if (g) {
        const double kg = gap < 0 ? k * gap : 0.0;
        g[0] = kg * dh;
        g[1] = 0;
        g[2] = kg;
    }
    return gap < 0 ? 0.5 * k * gap * gap : 0.0;
}

double elem_eg(const Model& M, const Elem& e, const double* x, double* g) {
    switch (e.kind) {
        case PAIR: return pair_eg(M, e.index, x, g);
        case BLADE: return blade_eg(M, e.index, x, g);
        case TRI: return tri_eg(M, e.index, x, g);
        case HINGE: return hinge_eg(M, e.index, x, g);
        case SHELL: return shell_eg(M, e.index, x, g);
        case FACE: return face_eg(M, e.index, x, g);
        case FIBRE: return fibre_eg(M, e.index, x, g);
        case TET: return tet_eg(M, e.index, x, g);
        case BOND: return bond_eg(M, e.index, x, g);
        case CONTACT: return contact_eg(M, e.index, x, g);
        default: return floor_eg(M, e.index, x, g);
    }
}

void build_elems(Model& M) {
    M.elems.clear();
    for (int t = 0; t < (int)M.tri.size() / 3; ++t)
        M.elems.push_back({TRI, t, 3, {M.tri[3 * t], M.tri[3 * t + 1], M.tri[3 * t + 2], -1}});
    for (int h = 0; h < (int)M.hinge.size() / 4; ++h)
        if (M.hinge_in_shell.empty() || !M.hinge_in_shell[h])
            M.elems.push_back({HINGE, h, 4, {M.hinge[4 * h], M.hinge[4 * h + 1], M.hinge[4 * h + 2], M.hinge[4 * h + 3], -1, -1}});
    {
        for (int t = 0; t < (int)M.shell_n.size(); ++t) {
            Elem el{SHELL, t, M.shell_n[t], {-1, -1, -1, -1, -1, -1}};
            for (int a = 0; a < M.shell_n[t]; ++a) el.v[a] = M.shell_v[6 * t + a];
            M.elems.push_back(el);
        }
    }
    for (int t = 0; t < (int)M.tet.size() / 4; ++t)
        M.elems.push_back({TET, t, 4, {M.tet[4 * t], M.tet[4 * t + 1], M.tet[4 * t + 2], M.tet[4 * t + 3]}});
    for (int b = 0; b < (int)M.bond_v.size(); ++b) M.elems.push_back({BOND, b, 1, {M.bond_v[b], -1, -1, -1}});
    if (!M.floor_k.empty())
        for (int i = 0; i < M.nv; ++i)
            if (M.floor_k[i] > 0) M.elems.push_back({FLOOR, i, 1, {i, -1, -1, -1}});
    if (!M.contact_k.empty() && !M.sdf.empty())
        for (int i = 0; i < M.nv; ++i)
            if (M.contact_k[i] > 0) M.elems.push_back({CONTACT, i, 1, {i, -1, -1, -1}});
    for (int p = 0; p < (int)M.pair_a.size(); ++p)
        M.elems.push_back({PAIR, p, 2, {M.pair_a[p], M.pair_b[p], -1, -1}});
    if (!M.blade_k.empty())
        for (int i = 0; i < M.nv; ++i)
            if (M.blade_k[i] > 0) M.elems.push_back({BLADE, i, 1, {i, -1, -1, -1}});
    for (int f = 0; f < (int)M.fib_a.size(); ++f)
        M.elems.push_back({FIBRE, f, M.fib_b[f] >= 0 ? 2 : 1, {M.fib_a[f], M.fib_b[f] >= 0 ? M.fib_b[f] : -1}});
    for (int f = 0; f < (int)M.face_w.size(); ++f) {
        Elem el{FACE, f, 8, {}};
        for (int a = 0; a < 8; ++a) el.v[a] = M.face_v[8 * f + a];
        M.elems.push_back(el);
    }
    M.lambda.assign(M.elems.size(), 0.0);
    M.pattern_ready = false;
}

constexpr double SELF_FACING = 0.6;  // |cos| between a pair's separation and each vertex's normal

void build_self_pairs(Model& M, const double* x) {
    M.self_pi.clear(), M.self_pj.clear(), M.self_clear.clear();
    if (M.self_k <= 0 || M.self_on.empty()) return;
    const double cell = 1.5 * M.self_d0;
    auto key = [&](const double* p) {
        const long long a = (long long)std::floor(p[0] / cell), b = (long long)std::floor(p[1] / cell),
                        c = (long long)std::floor(p[2] / cell);
        return (a * 73856093LL) ^ (b * 19349663LL) ^ (c * 83492791LL);
    };
    std::unordered_map<long long, std::vector<int>> grid;
    for (int i = 0; i < M.nv; ++i)
        if (M.self_on[i]) grid[key(x + 3 * i)].push_back(i);
    // vertex normals of the current shape (area-weighted): a real self-contact is two parts of the
    // sheet facing each other, apart along their normals; vertices squeezed together within the
    // sheet (in its own plane) are not touching
    std::vector<double> vn(3 * M.nv, 0.0);
    for (size_t t = 0; t + 2 < M.tri.size(); t += 3) {
        const int a = M.tri[t], b = M.tri[t + 1], c = M.tri[t + 2];
        const double u[3] = {x[3 * b] - x[3 * a], x[3 * b + 1] - x[3 * a + 1], x[3 * b + 2] - x[3 * a + 2]};
        const double w[3] = {x[3 * c] - x[3 * a], x[3 * c + 1] - x[3 * a + 1], x[3 * c + 2] - x[3 * a + 2]};
        const double nx = u[1] * w[2] - u[2] * w[1], ny = u[2] * w[0] - u[0] * w[2], nz = u[0] * w[1] - u[1] * w[0];
        for (int v : {a, b, c}) vn[3 * v] += nx, vn[3 * v + 1] += ny, vn[3 * v + 2] += nz;
    }
    for (int i = 0; i < M.nv; ++i) {
        const double l = std::sqrt(vn[3 * i] * vn[3 * i] + vn[3 * i + 1] * vn[3 * i + 1] + vn[3 * i + 2] * vn[3 * i + 2]);
        if (l > 0) vn[3 * i] /= l, vn[3 * i + 1] /= l, vn[3 * i + 2] /= l;
    }
    for (int i = 0; i < M.nv; ++i) {
        if (!M.self_on[i]) continue;
        const double* p = x + 3 * i;
        const long long a0 = (long long)std::floor(p[0] / cell), b0 = (long long)std::floor(p[1] / cell),
                        c0 = (long long)std::floor(p[2] / cell);
        for (int da = -1; da <= 1; ++da)
            for (int db = -1; db <= 1; ++db)
                for (int dc = -1; dc <= 1; ++dc) {
                    const long long k_ = ((a0 + da) * 73856093LL) ^ ((b0 + db) * 19349663LL) ^ ((c0 + dc) * 83492791LL);
                    auto it = grid.find(k_);
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        if (j <= i) continue;
                        const double dx = x[3 * j] - p[0], dy = x[3 * j + 1] - p[1], dz = x[3 * j + 2] - p[2];
                        const double dd = dx * dx + dy * dy + dz * dz;
                        if (dd >= cell * cell) continue;
                        const double d_ = std::sqrt(dd) + 1e-12;
                        const double ci = std::abs(dx * vn[3 * i] + dy * vn[3 * i + 1] + dz * vn[3 * i + 2]) / d_;
                        const double cj = std::abs(dx * vn[3 * j] + dy * vn[3 * j + 1] + dz * vn[3 * j + 2]) / d_;
                        if (ci < SELF_FACING || cj < SELF_FACING) continue;  // not facing each other
                        const int* lo = M.self_ring_idx.data() + M.self_ring_ptr[i];
                        const int* hi = M.self_ring_idx.data() + M.self_ring_ptr[i + 1];
                        if (std::binary_search(lo, hi, j)) continue;  // mesh neighbours
                        const double rx = M.rest[3 * j] - M.rest[3 * i], ry = M.rest[3 * j + 1] - M.rest[3 * i + 1],
                                     rz = M.rest[3 * j + 2] - M.rest[3 * i + 2];
                        const double clear = std::min(M.self_d0, std::sqrt(rx * rx + ry * ry + rz * rz) - M.self_give);
                        if (clear <= 0.05) continue;
                        M.self_pi.push_back(i), M.self_pj.push_back(j), M.self_clear.push_back(clear);
                    }
                }
    }
}

double self_eg(const Model& M, const double* x, double* grad) {
    double e = 0;
    for (size_t q = 0; q < M.self_pi.size(); ++q) {
        const int i = M.self_pi[q], j = M.self_pj[q];
        double r[3] = {x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]};
        const double d = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        const double pen = M.self_clear[q] - d;
        if (pen <= 0 || d < 1e-12) continue;
        e += 0.5 * M.self_k * pen * pen;
        if (grad)
            for (int c = 0; c < 3; ++c) {
                const double gc = -M.self_k * pen * r[c] / d;
                grad[3 * i + c] += gc;
                grad[3 * j + c] -= gc;
            }
    }
    return e;
}

double total_eg(const Model& M, const double* x, double* grad) {
    if (grad) std::fill(grad, grad + 3 * M.nv, 0.0);
    double e = 0, gl[3 * MAXV];
    for (const Elem& el : M.elems) {
        e += elem_eg(M, el, x, grad ? gl : nullptr);
        if (grad)
            for (int a = 0; a < el.n; ++a)
                for (int c = 0; c < 3; ++c) grad[3 * el.v[a] + c] += gl[3 * a + c];
    }
    if (!M.fext.empty())
        for (int d = 0; d < 3 * M.nv; ++d) {
            e -= M.fext[d] * x[d];
            if (grad) grad[d] -= M.fext[d];
        }
    if (!M.self_pi.empty()) e += self_eg(M, x, grad);
    if (!M.grip_k.empty())
        for (int d = 0; d < 3 * M.nv; ++d) {
            const double k = M.grip_k[d / 3];
            if (k == 0) continue;
            const double r = x[d] - M.grip_t[d];
            e += 0.5 * k * r * r;
            if (grad) grad[d] += k * r;
        }
    return e;
}

// ---- analytic element Hessians ---------------------------------------------
//
// St Venant-Kirchhoff (plus J-curve) on a simplex with F = sum_a x_a beta_a^T:
//   H = vol * ( J^T C J  +  (beta_a^T S beta_b) I3 )
// J: Voigt strain derivatives, dE_ij/dx_a = (beta_a,j F_i + beta_a,i F_j) / 2
// C: material stiffness in Voigt form (positive), S: second Piola stress.
// The material term is always positive semi-definite; the geometric term is
// when S is. Only when S has a negative eigenvalue (compression) is the
// element block projected with an eigen-decomposition.

template <int N>
void project_block(Eigen::Matrix<double, N, N>& H) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es(H);
    const auto ev = es.eigenvalues().cwiseMax(0.0);
    H = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

// D = 2 (triangle, 3 vertices) or 3 (tet, 4 vertices).
template <int D>
void simplex_hessian(const Eigen::Matrix<double, 3, D>& F, const Eigen::Matrix<double, D, D + 1>& beta,
                     double mu, double lam, double k1, double k2, double vol, bool psd,
                     Eigen::Matrix<double, 3 * (D + 1), 3 * (D + 1)>& H) {
    constexpr int NV = D + 1, N = 3 * NV, NS = D * (D + 1) / 2;
    using MatD = Eigen::Matrix<double, D, D>;
    const MatD E = 0.5 * (F.transpose() * F - MatD::Identity());
    const double tr = E.trace();
    MatD S = 2.0 * mu * E + lam * tr * MatD::Identity();
    // Voigt ordering: diagonals, then (0,1), (0,2), (1,2)
    int vi[NS], vj[NS];
    int q = 0;
    for (int a = 0; a < D; ++a) vi[q] = vj[q] = a, ++q;
    for (int a = 0; a < D; ++a)
        for (int b = a + 1; b < D; ++b) vi[q] = a, vj[q] = b, ++q;
    Eigen::Matrix<double, NS, NS> C = Eigen::Matrix<double, NS, NS>::Zero();
    for (int a = 0; a < D; ++a)
        for (int b = 0; b < D; ++b) C(a, b) = lam + (a == b ? 2.0 * mu : 0.0);
    for (int r = D; r < NS; ++r) C(r, r) = 4.0 * mu;  // shear: psi has 2 mu (E_ij^2 + E_ji^2)
    if (k1 > 0 && k2 > 0)
        for (int a = 0; a < D; ++a) {
            const double ea = std::max(E(a, a), 0.0);
            double w, dw, d2w;
            collagen(ea, k1, k2, w, dw, d2w);
            S(a, a) += dw;
            C(a, a) += d2w;
        }
    Eigen::Matrix<double, NS, N> J;
    for (int r = 0; r < NS; ++r)
        for (int v = 0; v < NV; ++v) {
            const Eigen::Vector3d d = 0.5 * (beta(vj[r], v) * F.col(vi[r]) + beta(vi[r], v) * F.col(vj[r]));
            J.template block<1, 3>(r, 3 * v) = d.transpose();
        }
    H.noalias() = J.transpose() * C * J;
    const Eigen::Matrix<double, NV, NV> G = beta.transpose() * S * beta;
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b) H.template block<3, 3>(3 * a, 3 * b).diagonal().array() += G(a, b);
    H *= vol;
    if (!psd) return;
    Eigen::SelfAdjointEigenSolver<MatD> es;
    es.computeDirect(S, Eigen::EigenvaluesOnly);
    if (es.eigenvalues().minCoeff() < 0) project_block<N>(H);
}

void tri_hessian(const Model& M, int t, const double* x, bool psd, Eigen::MatrixXd& Hout) {
    const int* v = &M.tri[3 * t];
    Eigen::Matrix<double, 3, 2> Ds;
    Ds.col(0) = P(x, v[1]) - P(x, v[0]);
    Ds.col(1) = P(x, v[2]) - P(x, v[0]);
    Eigen::Map<const Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> Dinv(&M.tri_dminv[4 * t]);
    const Eigen::Matrix<double, 3, 2> F = Ds * Dinv;
    Eigen::Matrix<double, 2, 3> beta;  // beta(j, a): dF_:j / dx_a
    for (int j = 0; j < 2; ++j) {
        beta(j, 1) = Dinv(0, j);
        beta(j, 2) = Dinv(1, j);
        beta(j, 0) = -Dinv(0, j) - Dinv(1, j);
    }
    Eigen::Matrix<double, 9, 9> H;
    simplex_hessian<2>(F, beta, M.tri_mu[t], M.tri_lam[t], M.tri_k1[t], M.tri_k2[t], M.tri_area[t] * M.tri_thick[t], psd, H);
    Hout = H;
}

void tet_hessian(const Model& M, int t, const double* x, bool psd, Eigen::MatrixXd& Hout) {
    const int* v = &M.tet[4 * t];
    Mat3 Ds;
    for (int c = 0; c < 3; ++c) Ds.col(c) = P(x, v[c + 1]) - P(x, v[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Dinv(&M.tet_dminv[9 * t]);
    const Mat3 F = Ds * Dinv;
    Eigen::Matrix<double, 3, 4> beta;
    for (int j = 0; j < 3; ++j) {
        double sum = 0;
        for (int c = 0; c < 3; ++c) {
            beta(j, c + 1) = Dinv(c, j);
            sum += Dinv(c, j);
        }
        beta(j, 0) = -sum;
    }
    Eigen::Matrix<double, 12, 12> H;
    simplex_hessian<3>(F, beta, M.tmu, M.tlam, 0.0, 0.0, M.tet_vol[t], psd, H);
    Hout = H;
}

// Slope of a damaged channel's force at separation s (second derivative of psi).
double channel_stiffness(double s, double hist, const Law& L) {
    const double m = std::max(hist, L.d0);
    if (s <= m) return traction(m, L) / m;
    if (s < L.dc) return -L.Tmax() / (L.dc - L.d0);
    return 0.0;
}

// Radial bond in n dimensions: psi(|r|). Eigenvalues psi'' (along r) and psi'/|r| (across).
template <int N>
void radial_hessian(const Eigen::Matrix<double, N, 1>& r, double hist, const Law& L, double w, bool psd,
                    Eigen::Matrix<double, N, N>& H) {
    const double d = r.norm();
    const double m = std::max(hist, L.d0);
    if (d <= m) {  // inside the history: psi = ks d^2 / 2, isotropic
        H = w * traction(m, L) / m * Eigen::Matrix<double, N, N>::Identity();
        return;
    }
    double psi, dpsi;
    channel(d, hist, L, psi, dpsi);
    double along = channel_stiffness(d, hist, L), across = dpsi / d;
    if (psd) along = std::max(along, 0.0), across = std::max(across, 0.0);
    const Eigen::Matrix<double, N, 1> u = r / d;
    H = w * (along * u * u.transpose() + across * (Eigen::Matrix<double, N, N>::Identity() - u * u.transpose()));
}

void bond_hessian(const Model& M, int b, const double* x, bool psd, Eigen::MatrixXd& Hout) {
    Mat3 H = Mat3::Zero();
    if (!M.broken[b]) {
        const Vec3 r = P(x, M.bond_v[b]) -
                       Vec3(M.bond_anchor[3 * b], M.bond_anchor[3 * b + 1], M.bond_anchor[3 * b + 2]);
        const double w = M.bond_w[b];
        if (!M.mixed) {
            radial_hessian<3>(r, M.hist_n[b], M.opening, w, psd, H);
        } else {
            Eigen::Matrix2d Hs;
            radial_hessian<2>(Eigen::Vector2d(r(0), r(1)), M.hist_s[b], M.sliding, w, psd, Hs);
            H.topLeftCorner<2, 2>() = Hs;
            if (r(2) > 0) {
                double kn = channel_stiffness(r(2), M.hist_n[b], M.opening);
                H(2, 2) = w * (psd ? std::max(kn, 0.0) : kn);
            }
        }
    }
    Hout = H;
}

void pair_hessian(const Model& M, int p, const double* x, bool psd, Eigen::MatrixXd& Hout) {
    Hout.setZero(6, 6);
    if (M.pair_broken[p]) return;
    const Vec3 r = P(x, M.pair_a[p]) - P(x, M.pair_b[p]);
    Mat3 H;
    radial_hessian<3>(r, M.pair_hist[p], M.pair_laws[M.pair_law[p]], M.pair_w[p], psd, H);
    Hout.block<3, 3>(0, 0) = H;
    Hout.block<3, 3>(3, 3) = H;
    Hout.block<3, 3>(0, 3) = -H;
    Hout.block<3, 3>(3, 0) = -H;
}

void blade_hessian(const Model& M, int i, const double* x, bool psd, Eigen::MatrixXd& Hout) {
    double h, dh, d2h;
    blade_rise(M, M.blade_tip - x[3 * i], h, dh, d2h);
    const double gap = x[3 * i + 2] - M.blade_floor - h;
    Mat3 H = Mat3::Zero();
    if (gap < 0) {
        const double k = M.blade_k[i];
        H(0, 0) = k * (dh * dh - gap * d2h);
        H(0, 2) = H(2, 0) = k * dh;
        H(2, 2) = k;
        if (psd) project_block<3>(H);
    }
    Hout = H;
}

// Local Hessian of one element (n*3 square), positive semi-definite if psd.
// xw is a working copy of x; finite differences perturb and restore it in place.
void elem_hessian_fd(const Model& M, const Elem& el, double* xw, Eigen::MatrixXd& H, bool psd) {
    const int n = 3 * el.n;
    H.setZero(n, n);
    double gp[3 * MAXV], gm[3 * MAXV];
    for (int a = 0; a < el.n; ++a)
        for (int c = 0; c < 3; ++c) {
            const int dof = 3 * el.v[a] + c;
            const double x0 = xw[dof];
            const double h = 1e-7 * (1.0 + std::abs(x0));
            xw[dof] = x0 + h;
            elem_eg(M, el, xw, gp);
            xw[dof] = x0 - h;
            elem_eg(M, el, xw, gm);
            xw[dof] = x0;
            for (int r = 0; r < n; ++r) H(r, 3 * a + c) = (gp[r] - gm[r]) / (2.0 * h);
        }
    H = 0.5 * (H + H.transpose()).eval();
    if (!psd) return;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    Eigen::VectorXd ev = es.eigenvalues().cwiseMax(0.0);
    H = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

void elem_hessian(const Model& M, const Elem& el, double* xw, Eigen::MatrixXd& H, bool psd, bool analytic) {
    const double* x = xw;
    if (el.kind == HINGE) {  // Gauss-Newton: 2k grad(theta) grad(theta)^T
        Vec3 d[4];
        dihedral(x, &M.hinge[4 * el.index], d);
        Eigen::Matrix<double, 12, 1> j;
        for (int s = 0; s < 4; ++s) j.segment<3>(3 * s) = d[s];
        H = 2.0 * M.hinge_k[el.index] * j * j.transpose();
        return;
    }
    if (el.kind == SHELL) {  // Gauss-Newton: J^T 2K J, J the signed angle gradients on the stencil
        Vec3 dd[3][4];
        double sg[3];
        shell_eg(M, el.index, x, nullptr, dd, sg);
        const int n = 3 * el.n;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, n);
        for (int i = 0; i < 3; ++i) {
            if (sg[i] == 0.0) continue;
            for (int sl = 0; sl < 4; ++sl) {
                const int pos = M.shell_map[12 * el.index + 4 * i + sl];
                J.block<1, 3>(i, 3 * pos) += sg[i] * dd[i][sl].transpose();
            }
        }
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(&M.shell_K[9 * el.index]);
        H = 2.0 * J.transpose() * K * J;
        return;
    }
    if (el.kind == CONTACT) {  // Gauss-Newton: k grad grad^T where in contact (positive semi-definite)
        H.setZero(3, 3);
        double gr[3];
        const double gap = sdf_sample(M, xw + 3 * el.index, gr) - M.contact_off[el.index];
        if (gap < 0)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) H(a, b) = M.contact_k[el.index] * gr[a] * gr[b];
        return;
    }
    if (el.kind == FLOOR) {
        H.setZero(3, 3);
        if (x[3 * el.index + 2] < M.floor_z) H(2, 2) = M.floor_k[el.index];
        return;
    }
    if (!analytic) return elem_hessian_fd(M, el, xw, H, psd);
    switch (el.kind) {
        case PAIR: return pair_hessian(M, el.index, x, psd, H);
        case FACE: return face_hessian(M, el.index, x, psd, H);
        case FIBRE: return fibre_hessian(M, el.index, x, H);
        case BLADE: return blade_hessian(M, el.index, x, psd, H);
        case TRI: return tri_hessian(M, el.index, x, psd, H);
        case TET: return tet_hessian(M, el.index, x, psd, H);
        default: return bond_hessian(M, el.index, x, psd, H);
    }
}

void ensure_dofs(Model& M) {
    if (!M.dof_map.empty()) return;
    M.dof_map.assign(3 * M.nv, -1);
    M.nfree = 0;
    for (int d = 0; d < 3 * M.nv; ++d)
        if (!M.fixed[d]) M.dof_map[d] = M.nfree++;
}

double incremental(const Model& M, const double* x, const double* pred, double inv_h2, double* grad) {
    double e = total_eg(M, x, grad);
    if (inv_h2 > 0)
        for (int d = 0; d < 3 * M.nv; ++d) {
            const double m = M.mass[d / 3] * inv_h2, dx = x[d] - pred[d];
            e += 0.5 * m * dx * dx;
            if (grad) grad[d] += m * dx;
        }
    return e;
}

void build_pattern(Model& M) {
    std::vector<Eigen::Triplet<double>> trip;
    M.elem_off.assign(M.elems.size() + 1, 0);
    for (size_t k = 0; k < M.elems.size(); ++k) {
        const Elem& el = M.elems[k];
        const int n = 3 * el.n;
        M.elem_off[k + 1] = M.elem_off[k] + n * n;
        for (int a = 0; a < n; ++a) {
            const int ra = M.dof_map[3 * el.v[a / 3] + a % 3];
            if (ra < 0) continue;
            for (int b = 0; b < n; ++b) {
                const int cb = M.dof_map[3 * el.v[b / 3] + b % 3];
                if (cb >= 0) trip.emplace_back(ra, cb, 0.0);
            }
        }
    }
    for (int i = 0; i < M.nfree; ++i) trip.emplace_back(i, i, 0.0);
    M.K = Eigen::SparseMatrix<double>(M.nfree, M.nfree);
    M.K.setFromTriplets(trip.begin(), trip.end());
    M.K.makeCompressed();
    auto find = [&](int r, int c) {
        const int* inner = M.K.innerIndexPtr();
        const int lo = M.K.outerIndexPtr()[c], hi = M.K.outerIndexPtr()[c + 1];
        return int(std::lower_bound(inner + lo, inner + hi, r) - inner);
    };
    M.slot.assign(M.elem_off.back(), -1);
    for (size_t k = 0; k < M.elems.size(); ++k) {
        const Elem& el = M.elems[k];
        const int n = 3 * el.n;
        for (int a = 0; a < n; ++a) {
            const int ra = M.dof_map[3 * el.v[a / 3] + a % 3];
            if (ra < 0) continue;
            for (int b = 0; b < n; ++b) {
                const int cb = M.dof_map[3 * el.v[b / 3] + b % 3];
                if (cb >= 0) M.slot[M.elem_off[k] + a * n + b] = find(ra, cb);
            }
        }
    }
    M.diag_slot.resize(M.nfree);
    for (int i = 0; i < M.nfree; ++i) M.diag_slot[i] = find(i, i);
    M.hbuf.assign(M.elem_off.back(), 0.0);
    const int nnz = int(M.K.nonZeros());
    M.gather_ptr.assign(nnz + 1, 0);
    for (int s : M.slot)
        if (s >= 0) ++M.gather_ptr[s + 1];
    for (int v = 0; v < nnz; ++v) M.gather_ptr[v + 1] += M.gather_ptr[v];
    M.gather_idx.resize(M.gather_ptr[nnz]);
    std::vector<int> fill(M.gather_ptr.begin(), M.gather_ptr.end() - 1);
    for (int i = 0; i < int(M.slot.size()); ++i)
        if (M.slot[i] >= 0) M.gather_idx[fill[M.slot[i]]++] = i;
    if (!M.factor) M.factor = make_factor(M.factor_kind);
    M.factor->analyze(M.K);
    M.pattern_ready = true;
}

// Newton iterations on the incremental potential. Returns iterations used.
int newton(Model& M, double* x, const double* pred, double h, int iters, double gtol, double* max_grad) {
    ensure_dofs(M);
    build_self_pairs(M, x);
    const double inv_h2 = (h > 0) ? 1.0 / (h * h) : 0.0;
    std::vector<double> g(3 * M.nv), trial(3 * M.nv), gt(3 * M.nv);
    double e = incremental(M, x, pred, inv_h2, g.data());
    double mu = 1e-8;
    int it = 0;
    for (; it < iters; ++it) {
        double gmax = 0;
        Eigen::VectorXd gf(M.nfree);
        for (int d = 0; d < 3 * M.nv; ++d)
            if (M.dof_map[d] >= 0) {
                gf(M.dof_map[d]) = g[d];
                gmax = std::max(gmax, std::abs(g[d]));
            }
        *max_grad = gmax;
        if (gmax < gtol) break;
        double tick = now_ms();
        if (!M.pattern_ready) build_pattern(M);
        const int nt = std::max(1, M.threads);
        const int ne = int(M.elems.size());
#pragma omp parallel num_threads(nt)
        {
            std::vector<double> xw(x, x + 3 * M.nv);  // each thread perturbs its own copy
            Eigen::MatrixXd Hl;
#pragma omp for schedule(static)
            for (int k = 0; k < ne; ++k) {
                elem_hessian(M, M.elems[k], xw.data(), Hl, M.project_psd, M.analytic);
                const int n = 3 * M.elems[k].n;
                double* dst = &M.hbuf[M.elem_off[k]];
                for (int a = 0; a < n; ++a)
                    for (int b = 0; b < n; ++b) dst[a * n + b] = Hl(a, b);
            }
        }
        double* kv = M.K.valuePtr();
        const int nnz = int(M.K.nonZeros());
        const int* gp = M.gather_ptr.data();
        const int* gi = M.gather_idx.data();
        const double* hb = M.hbuf.data();
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int v = 0; v < nnz; ++v) {
            double sum = 0;
            for (int q = gp[v]; q < gp[v + 1]; ++q) sum += hb[gi[q]];
            kv[v] = sum;
        }
        for (int d = 0; d < 3 * M.nv; ++d)
            if (M.dof_map[d] >= 0)
                kv[M.diag_slot[M.dof_map[d]]] += M.mass[d / 3] * inv_h2 + (M.grip_k.empty() ? 0.0 : M.grip_k[d / 3]);
        if (!M.self_pi.empty()) {  // self-contact: k on the diagonal of each vertex in contact
            for (size_t q = 0; q < M.self_pi.size(); ++q) {
                const int i = M.self_pi[q], j = M.self_pj[q];
                const double dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
                if (std::sqrt(dx * dx + dy * dy + dz * dz) >= M.self_clear[q]) continue;
                for (int v : {i, j})
                    for (int c = 0; c < 3; ++c)
                        // 2k I on each vertex bounds the pair's own stiffness k [P, -P; -P, P] (P = n n^T)
                        // from above, so the step never overshoots into the contact (with k it did, and
                        // the line search kept failing)
                        if (M.dof_map[3 * v + c] >= 0) kv[M.diag_slot[M.dof_map[3 * v + c]]] += 2.0 * M.self_k;
            }
        }
        const Eigen::SparseMatrix<double>& K = M.K;
        M.t_assemble += now_ms() - tick;
        bool accepted = false;
        for (int attempt = 0; attempt < 30 && !accepted; ++attempt) {
            Eigen::SparseMatrix<double> A = K;
            for (int i = 0; i < M.nfree; ++i) A.valuePtr()[M.diag_slot[i]] += mu;
            tick = now_ms();
            ++M.n_factor;
            if (!M.factor->factorize(A)) {
                M.t_factor += now_ms() - tick;
                mu *= 10;
                continue;
            }
            const Eigen::VectorXd dfree = -M.factor->solve(gf);
            M.t_factor += now_ms() - tick;
            tick = now_ms();
            const double slope = gf.dot(dfree);
            if (!(slope < 0)) {
                M.t_search += now_ms() - tick;
                mu *= 10;
                continue;
            }
            double alpha = 1.0;
            while (alpha > 1e-4) {
                for (int d = 0; d < 3 * M.nv; ++d)
                    trial[d] = x[d] + ((M.dof_map[d] >= 0) ? alpha * dfree(M.dof_map[d]) : 0.0);
                const double et = incremental(M, trial.data(), pred, inv_h2, gt.data());
                if (et <= e + 1e-4 * alpha * slope) {
                    std::copy(trial.begin(), trial.end(), x);
                    g.swap(gt);
                    e = et;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            M.t_search += now_ms() - tick;
            if (accepted)
                mu = (alpha == 1.0) ? std::max(mu * 0.25, 1e-10) : mu;
            else
                mu *= 10;
        }
        if (!accepted) break;
    }
    double gmax = 0;
    for (int d = 0; d < 3 * M.nv; ++d)
        if (M.dof_map[d] >= 0) gmax = std::max(gmax, std::abs(g[d]));
    *max_grad = gmax;
    return it;
}

// One XPBD Gauss-Seidel sweep. Each element energy E_e is one constraint
// C = sqrt(2 E_e) with compliance 1 (so E_e = C^2 / 2).
void xpbd_sweep(Model& M, double* x, double h) {
    const double alpha = 1.0 / (h * h);
    std::fill(M.lambda.begin(), M.lambda.end(), 0.0);
    double gl[3 * MAXV];
    {
        for (size_t k = 0; k < M.elems.size(); ++k) {
            const Elem& el = M.elems[k];
            const double e = elem_eg(M, el, x, gl);
            if (e < 1e-24) continue;
            const double C = std::sqrt(2.0 * e);
            double denom = alpha;
            for (int a = 0; a < el.n; ++a)
                for (int c = 0; c < 3; ++c) {
                    const int d = 3 * el.v[a] + c;
                    if (M.fixed[d]) continue;
                    const double gc = gl[3 * a + c] / C;
                    denom += gc * gc / M.mass[el.v[a]];
                }
            const double dl = (-C - alpha * M.lambda[k]) / denom;
            M.lambda[k] += dl;
            for (int a = 0; a < el.n; ++a)
                for (int c = 0; c < 3; ++c) {
                    const int d = 3 * el.v[a] + c;
                    if (!M.fixed[d]) x[d] += gl[3 * a + c] / C * dl / M.mass[el.v[a]];
                }
        }
    }
}

// XPBD with small steps: the frame is split into ``substeps``, each a
// prediction and one sweep. Fixed dofs move linearly to their targets.
void xpbd(Model& M, double* x, double* v, const double* x_start, double h, int substeps, double damping) {
    const int n3 = 3 * M.nv;
    const double hs = h / substeps;
    const double keep = std::pow(1.0 - damping, 1.0 / substeps);
    std::vector<double> target(x, x + n3), prev(n3);
    std::copy(x_start, x_start + n3, x);
    for (int s = 1; s <= substeps; ++s) {
        std::copy(x, x + n3, prev.begin());
        const double f = double(s) / substeps;
        for (int d = 0; d < n3; ++d) {
            if (M.fixed[d])
                x[d] = x_start[d] + f * (target[d] - x_start[d]);
            else {
                v[d] *= keep;
                x[d] += hs * v[d];
            }
        }
        xpbd_sweep(M, x, hs);
        for (int d = 0; d < n3; ++d) v[d] = M.fixed[d] ? 0.0 : (x[d] - prev[d]) / hs;
    }
}

size_t x0_size(const Model& M) { return size_t(3 * M.nv); }

double now_ms_impl() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

}  // namespace

extern "C" {

void* sf_create(int nv, const double* rest, const double* mass) {
    Model* M = new Model;
    M->nv = nv;
    M->rest.assign(rest, rest + 3 * nv);
    M->mass.assign(mass, mass + nv);
    M->fixed.assign(3 * nv, 0);
    return M;
}

void sf_destroy(void* m) { delete static_cast<Model*>(m); }

// Material per triangle: thickness, Lame mu and lambda (plane stress), collagen k1, k2.
void sf_set_triangles(void* m, int count, const int* tris, const double* dminv, const double* area,
                      const double* thick, const double* mu, const double* lam, const double* k1, const double* k2) {
    Model& M = *static_cast<Model*>(m);
    M.tri.assign(tris, tris + 3 * count);
    M.tri_dminv.assign(dminv, dminv + 4 * count);
    M.tri_area.assign(area, area + count);
    M.tri_thick.assign(thick, thick + count);
    M.tri_mu.assign(mu, mu + count);
    M.tri_lam.assign(lam, lam + count);
    M.tri_k1.assign(k1, k1 + count);
    M.tri_k2.assign(k2, k2 + count);
    build_elems(M);
}

// Layer-to-layer bonds: vertex a[i] to b[i], weight (area), law index into laws {G, d0, dc}.
void sf_set_pairs(void* m, int count, const int* a, const int* b, const double* weight, const int* law, int n_laws,
                  const double* laws) {
    Model& M = *static_cast<Model*>(m);
    M.pair_a.assign(a, a + count);
    M.pair_b.assign(b, b + count);
    M.pair_w.assign(weight, weight + count);
    M.pair_law.assign(law, law + count);
    M.pair_hist.assign(count, 0.0);
    M.pair_broken.assign(count, 0);
    M.pair_laws.clear();
    for (int i = 0; i < n_laws; ++i) M.pair_laws.push_back({laws[3 * i], laws[3 * i + 1], laws[3 * i + 2]});
    build_elems(M);
}

// Face bonds of offset layers (stack.Stack offset=True). Per bond: tris 12 global corners (layer a's
// two triangles on the edge, then layer b's), edge 4 (a's two ends, b's two ends), offsets oa, ob
// (signed, along the edge normal), weight (area), opening law and sliding law (-1: one law on the
// full separation) as indices into laws {G, d0, dc}.
void sf_set_faces(void* m, int count, const int* tris, const int* edge, const double* oa, const double* ob,
                  const double* weight, const int* law_n, const int* law_s, int n_laws, const double* laws,
                  const double* c0) {
    Model& M = *static_cast<Model*>(m);
    M.face_tri.assign(tris, tris + 12 * count);
    M.face_edge.assign(edge, edge + 4 * count);
    M.face_oa.assign(oa, oa + count);
    M.face_ob.assign(ob, ob + count);
    M.face_w.assign(weight, weight + count);
    M.face_ln.assign(law_n, law_n + count);
    M.face_ls.assign(law_s, law_s + count);
    M.face_hn.assign(count, 0.0);
    M.face_hs.assign(count, 0.0);
    M.face_broken.assign(count, 0);
    M.face_c0.assign(c0, c0 + 3 * count);
    M.face_laws.clear();
    for (int i = 0; i < n_laws; ++i) M.face_laws.push_back({laws[3 * i], laws[3 * i + 1], laws[3 * i + 2]});
    // stencil: the distinct vertices (8 for an interior edge), and each corner / edge end's place in it
    M.face_v.assign(8 * count, -1);
    M.face_slot.assign(22 * count, 0);
    for (int f = 0; f < count; ++f) {
        int* v = &M.face_v[8 * f];
        int nv = 0;
        auto place = [&](int q) {
            for (int a = 0; a < nv; ++a)
                if (v[a] == q) return a;
            v[nv] = q;
            return nv++;
        };
        for (int i = 0; i < 12; ++i) M.face_slot[22 * f + i] = place(tris[12 * f + i]);
        for (int i = 0; i < 4; ++i) M.face_slot[22 * f + 12 + i] = place(edge[4 * f + i]);
        for (int a = nv; a < 8; ++a) v[a] = v[0];  // never: interior edges have 8
    }
    build_elems(M);
}

// Tension-only fibres (fibres.py): insertion a, origin b (-1: the fixed point origin[3i..]), rest
// length, area and material (E, k1, k2) per fibre.
void sf_set_fibres(void* m, int count, const int* a, const int* b, const double* origin, const double* L0,
                   const double* area, const double* E, const double* k1, const double* k2) {
    Model& M = *static_cast<Model*>(m);
    M.fib_a.assign(a, a + count);
    M.fib_b.assign(b, b + count);
    M.fib_origin.assign(origin, origin + 3 * count);
    M.fib_L0.assign(L0, L0 + count);
    M.fib_A.assign(area, area + count);
    M.fib_E.assign(E, E + count);
    M.fib_k1.assign(k1, k1 + count);
    M.fib_k2.assign(k2, k2 + count);
    build_elems(M);
}

void sf_face_state(void* m, double* hist_n, double* hist_s, uint8_t* broken) {
    Model& M = *static_cast<Model*>(m);
    std::copy(M.face_hn.begin(), M.face_hn.end(), hist_n);
    std::copy(M.face_hs.begin(), M.face_hs.end(), hist_s);
    std::copy(M.face_broken.begin(), M.face_broken.end(), broken);
}

void sf_pair_state(void* m, double* hist, uint8_t* broken) {
    Model& M = *static_cast<Model*>(m);
    std::copy(M.pair_hist.begin(), M.pair_hist.end(), hist);
    std::copy(M.pair_broken.begin(), M.pair_broken.end(), broken);
}

// Blade: per-vertex contact stiffness (0 = no contact), shape and the bone height.
void sf_set_blade(void* m, const double* k_per_vertex, double height, double slope, double heel, double floor_z) {
    Model& M = *static_cast<Model*>(m);
    M.blade_k.assign(k_per_vertex, k_per_vertex + M.nv);
    M.blade_height = height, M.blade_slope = slope, M.blade_heel = heel, M.blade_floor = floor_z;
    build_elems(M);
}

void sf_set_blade_tip(void* m, double tip) { static_cast<Model*>(m)->blade_tip = tip; }

// Force pushing the blade along +x (the reaction of the tissue on its ramp).
double sf_blade_push(void* m, const double* x) {
    Model& M = *static_cast<Model*>(m);
    double push = 0;
    for (int i = 0; i < (int)M.blade_k.size(); ++i) {
        if (M.blade_k[i] <= 0) continue;
        double h, dh, d2h;
        blade_rise(M, M.blade_tip - x[3 * i], h, dh, d2h);
        const double gap = x[3 * i + 2] - M.blade_floor - h;
        if (gap < 0) push -= M.blade_k[i] * gap * dh;
    }
    return push;
}

// Mark substrate bonds broken (a sharp edge severing them).
void sf_set_bond_broken(void* m, const uint8_t* broken) {
    Model& M = *static_cast<Model*>(m);
    for (size_t b = 0; b < M.broken.size(); ++b) M.broken[b] = M.broken[b] || broken[b];
}

void sf_set_hinges(void* m, int count, const int* hinges, const double* k) {
    Model& M = *static_cast<Model*>(m);
    M.hinge.assign(hinges, hinges + 4 * count);
    M.hinge_k.assign(k, k + count);
    M.hinge_rest.clear();
    build_elems(M);
}

// Shell bending (after sf_set_hinges / sf_set_hinge_rest): per triangle its corners, the hinges on
// its edges (opposite corners 0, 1, 2; -1 on the boundary) and K (row-major 3x3); per hinge its sign
// (its angle measured against the mesh's normal). Replaces the hinge energy.
void sf_set_shell(void* m, int count, const int* tris, const int* tri_hinge, const double* K, const double* sign) {
    Model& M = *static_cast<Model*>(m);
    const int nh = (int)M.hinge_k.size();
    M.shell_hinge.assign(tri_hinge, tri_hinge + 3 * count);
    M.shell_K.assign(K, K + 9 * count);
    M.hinge_sign.assign(sign, sign + nh);
    M.shell_n.assign(count, 0);
    M.shell_v.assign(6 * count, -1);
    M.shell_map.assign(12 * count, 0);
    M.hinge_in_shell.assign(nh, 0);
    for (int i = 0; i < 3 * count; ++i)
        if (tri_hinge[i] >= 0) M.hinge_in_shell[tri_hinge[i]] = 1;
    for (int t = 0; t < count; ++t) {
        int* v = &M.shell_v[6 * t];
        int n = 0;
        for (int c = 0; c < 3; ++c) v[n++] = tris[3 * t + c];
        for (int i = 0; i < 3; ++i) {
            const int h = tri_hinge[3 * t + i];
            if (h < 0) continue;
            for (int sl = 0; sl < 4; ++sl) {
                const int q = M.hinge[4 * h + sl];
                int pos = -1;
                for (int a = 0; a < n; ++a)
                    if (v[a] == q) pos = a;
                if (pos < 0) v[pos = n++] = q;
                M.shell_map[12 * t + 4 * i + sl] = pos;
            }
        }
        M.shell_n[t] = n;
    }
    build_elems(M);
}

// Rest bend angle per hinge (after sf_set_hinges): a curved shell bends from its own shape.
void sf_set_hinge_rest(void* m, const double* theta0) {
    Model& M = *static_cast<Model*>(m);
    M.hinge_rest.assign(theta0, theta0 + M.hinge_k.size());
}

void sf_set_tets(void* m, int count, const int* tets, const double* dminv, const double* vol, double mu, double lam) {
    Model& M = *static_cast<Model*>(m);
    M.tet.assign(tets, tets + 4 * count);
    M.tet_dminv.assign(dminv, dminv + 9 * count);
    M.tet_vol.assign(vol, vol + count);
    M.tmu = mu, M.tlam = lam;
    build_elems(M);
}

// law: {G, d0, dc} for opening and sliding. mixed = 0: one law on |r| (opening).
void sf_set_bonds(void* m, int count, const int* verts, const double* anchor, const double* weight, const double* law,
                  int mixed) {
    Model& M = *static_cast<Model*>(m);
    M.bond_v.assign(verts, verts + count);
    M.bond_anchor.assign(anchor, anchor + 3 * count);
    M.bond_w.assign(weight, weight + count);
    M.hist_n.assign(count, 0.0);
    M.hist_s.assign(count, 0.0);
    M.broken.assign(count, 0);
    M.opening = {law[0], law[1], law[2]};
    M.sliding = {law[3], law[4], law[5]};
    M.mixed = mixed != 0;
    build_elems(M);
}

// Obstacle: a signed distance grid (n[0] x n[1] x n[2], x slowest; phi > 0 outside), its first
// node's position and spacing (mm). Null grid removes it.
void sf_set_sdf(void* m, const int* n, const double* origin, const double* h, const float* grid) {
    Model& M = *static_cast<Model*>(m);
    if (!grid) {
        M.sdf.clear();
    } else {
        for (int a = 0; a < 3; ++a) M.sdf_n[a] = n[a], M.sdf_o[a] = origin[a], M.sdf_h[a] = h[a];
        M.sdf.assign(grid, grid + size_t(n[0]) * n[1] * n[2]);
    }
    build_elems(M);
}

// Self-contact: stiffness (N/mm per pair, 0 = off), clearance d0 and give (mm), which vertices take
// part (nv flags), and each vertex's excluded mesh neighbours (CSR, sorted).
void sf_set_self_contact(void* m, double k, double d0, double give, const uint8_t* on, const int* ring_ptr,
                         const int* ring_idx) {
    Model& M = *static_cast<Model*>(m);
    M.self_k = k, M.self_d0 = d0, M.self_give = give;
    if (k <= 0 || !on) {
        M.self_on.clear();
        M.self_pi.clear(), M.self_pj.clear(), M.self_clear.clear();
        return;
    }
    M.self_on.assign(on, on + M.nv);
    M.self_ring_ptr.assign(ring_ptr, ring_ptr + M.nv + 1);
    M.self_ring_idx.assign(ring_idx, ring_idx + ring_ptr[M.nv]);
}

// Contact stiffness (N/mm, 0 = none) and clearance (mm, phi kept >= it) per vertex.
void sf_set_contact(void* m, const double* k, const double* off) {
    Model& M = *static_cast<Model*>(m);
    if (!k) {
        M.contact_k.clear();
        M.contact_off.clear();
    } else {
        M.contact_k.assign(k, k + M.nv);
        M.contact_off.assign(off, off + M.nv);
    }
    build_elems(M);
}

void sf_set_floor(void* m, double z, const double* k_per_vertex) {
    Model& M = *static_cast<Model*>(m);
    M.floor_z = z;
    M.floor_k.assign(k_per_vertex, k_per_vertex + M.nv);
    build_elems(M);
}

void sf_set_fixed(void* m, const uint8_t* mask) {
    Model& M = *static_cast<Model*>(m);
    M.fixed.assign(mask, mask + 3 * M.nv);
    M.dof_map.clear();
    M.pattern_ready = false;
}

// Applied force per vertex (3 nv values, N), or null to remove it.
void sf_set_external(void* m, const double* f) {
    Model& M = *static_cast<Model*>(m);
    if (f)
        M.fext.assign(f, f + 3 * M.nv);
    else
        M.fext.clear();
}

// Grip springs: stiffness per vertex (nv values, N/mm) and targets (3 nv), or null k to remove them.
void sf_set_grip(void* m, const double* k, const double* targets) {
    Model& M = *static_cast<Model*>(m);
    if (!k) {
        M.grip_k.clear();
        M.grip_t.clear();
        return;
    }
    M.grip_k.assign(k, k + M.nv);
    M.grip_t.assign(targets, targets + 3 * M.nv);
}

// Move the grip targets only (3 nv values).
void sf_set_grip_targets(void* m, const double* targets) {
    Model& M = *static_cast<Model*>(m);
    if (!M.grip_k.empty()) M.grip_t.assign(targets, targets + 3 * M.nv);
}

void sf_set_options(void* m, int project_psd, int threads) {
    Model& M = *static_cast<Model*>(m);
    M.project_psd = project_psd != 0;
    M.threads = std::max(1, threads);
}

// Sparse factorization for Newton: 0 Eigen LDLT, 1 CHOLMOD supernodal LLT,
// 2 CHOLMOD simplicial LDLT. Returns 0 if CHOLMOD was asked for but not built in.
int sf_set_factor(void* m, int kind) {
    Model& M = *static_cast<Model*>(m);
    if (kind != F_EIGEN_LDLT && !have_cholmod()) return 0;
    M.factor_kind = kind;
    M.factor = make_factor(kind);
    if (M.pattern_ready) M.factor->analyze(M.K);
    return 1;
}

int sf_have_cholmod() { return have_cholmod() ? 1 : 0; }

// Element Hessians: 1 analytic (default), 0 finite differences of the gradient.
void sf_set_hessian(void* m, int analytic) { static_cast<Model*>(m)->analytic = analytic != 0; }

int sf_elem_count(void* m) {
    Model& M = *static_cast<Model*>(m);
    if (M.elems.empty()) build_elems(M);
    return int(M.elems.size());
}

// One element's Hessian for testing: kind and vertices into info[0..4], the
// (3n)^2 block into H (row-major). Returns n, the number of vertices.
int sf_elem_hessian(void* m, int k, const double* x, int psd, int analytic, int* info, double* H) {
    Model& M = *static_cast<Model*>(m);
    if (M.elems.empty()) build_elems(M);
    const Elem& el = M.elems[k];
    std::vector<double> xw(x, x + 3 * M.nv);
    Eigen::MatrixXd Hl;
    elem_hessian(M, el, xw.data(), Hl, psd != 0, analytic != 0);
    info[0] = el.kind;
    for (int a = 0; a < 4; ++a) info[1 + a] = el.v[a];
    const int n = 3 * el.n;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) H[r * n + c] = Hl(r, c);
    return el.n;
}

double sf_energy(void* m, const double* x, double* grad) { return total_eg(*static_cast<Model*>(m), x, grad); }

// Record bond damage after a frame. Returns bonds broken so far.
int sf_commit(void* m, const double* x) {
    Model& M = *static_cast<Model*>(m);
    int broken = 0;
    for (size_t b = 0; b < M.bond_v.size(); ++b) {
        const Vec3 r = P(x, M.bond_v[b]) - Vec3(M.bond_anchor[3 * b], M.bond_anchor[3 * b + 1], M.bond_anchor[3 * b + 2]);
        if (M.mixed) {
            M.hist_n[b] = std::max(M.hist_n[b], std::max(r(2), 0.0));
            M.hist_s[b] = std::max(M.hist_s[b], std::hypot(r(0), r(1)));
            if (M.hist_n[b] >= M.opening.dc || M.hist_s[b] >= M.sliding.dc) M.broken[b] = 1;
        } else {
            M.hist_n[b] = std::max(M.hist_n[b], r.norm());
            if (M.hist_n[b] >= M.opening.dc) M.broken[b] = 1;
        }
        broken += M.broken[b];
    }
    for (size_t f = 0; f < M.face_w.size(); ++f) {
        const FaceGeo G = face_geo(M, int(f), x);
        if (M.face_ls[f] < 0) {
            M.face_hn[f] = std::max(M.face_hn[f], G.r.norm());
        } else {
            const double dn = G.r.dot(G.nb);
            M.face_hn[f] = std::max(M.face_hn[f], std::max(dn, 0.0));
            M.face_hs[f] = std::max(M.face_hs[f], (G.r - dn * G.nb).norm());
            if (M.face_hs[f] >= M.face_laws[M.face_ls[f]].dc) M.face_broken[f] = 1;
        }
        if (M.face_hn[f] >= M.face_laws[M.face_ln[f]].dc) M.face_broken[f] = 1;
    }
    for (size_t p = 0; p < M.pair_a.size(); ++p) {
        const double d = (P(x, M.pair_a[p]) - P(x, M.pair_b[p])).norm();
        M.pair_hist[p] = std::max(M.pair_hist[p], d);
        if (M.pair_hist[p] >= M.pair_laws[M.pair_law[p]].dc) M.pair_broken[p] = 1;
    }
    return broken;
}

void sf_bond_state(void* m, double* hist_n, uint8_t* broken) {
    Model& M = *static_cast<Model*>(m);
    std::copy(M.hist_n.begin(), M.hist_n.end(), hist_n);
    std::copy(M.broken.begin(), M.broken.end(), broken);
}

// One frame. x, v (3*nv) are updated in place; fixed dofs of x must already
// hold their targets for this frame. solver 0 = Newton (iters = Newton
// iterations), 1 = XPBD (iters = substeps, one sweep each).
// h <= 0: quasi-static (Newton only). stats: [ms, iterations, max_grad].
void sf_step(void* m, double* x, double* v, double h, int solver, int iters, double gtol, double damping,
             double* stats) {
    Model& M = *static_cast<Model*>(m);
    const double t0 = now_ms();
    M.t_assemble = M.t_factor = M.t_search = 0;
    M.n_factor = 0;
    if (M.xlast.size() != x0_size(M)) M.xlast.assign(x, x + 3 * M.nv);
    if (solver == 1 && h > 0) {
        // Start from last frame's positions; fixed dofs travel to their targets.
        std::vector<double> start(M.xlast);
        for (int d = 0; d < 3 * M.nv; ++d)
            if (!M.fixed[d]) start[d] = x[d];
        xpbd(M, x, v, start.data(), h, iters, damping);
        M.xlast.assign(x, x + 3 * M.nv);
        stats[0] = now_ms() - t0;
        stats[1] = iters;
        stats[2] = 0;
        return;
    }
    std::vector<double> x0(x, x + 3 * M.nv), pred(3 * M.nv);
    for (int d = 0; d < 3 * M.nv; ++d) {
        if (M.fixed[d]) {
            pred[d] = x[d];
            continue;
        }
        const double vel = (h > 0) ? v[d] * (1.0 - damping) : 0.0;
        pred[d] = x[d] + ((h > 0) ? h * vel : 0.0);
        x[d] = pred[d];
    }
    double max_grad = 0;
    int used = iters;
    used = newton(M, x, pred.data(), h, iters, gtol, &max_grad);
    M.xlast.assign(x, x + 3 * M.nv);
    if (h > 0)
        for (int d = 0; d < 3 * M.nv; ++d) v[d] = M.fixed[d] ? 0.0 : (x[d] - x0[d]) / h;
    stats[0] = now_ms() - t0;
    stats[1] = used;
    stats[2] = max_grad;
    stats[3] = M.t_assemble;
    stats[4] = M.t_factor;
    stats[5] = M.t_search;
    stats[6] = M.n_factor;
}

}  // extern "C"
