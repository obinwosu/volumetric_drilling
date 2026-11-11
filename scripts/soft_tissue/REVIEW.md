# Soft-tissue simulation: review notes

Status as of 2026-11-08. Branch `feature/graphics-compare`.

**Commit status**
- Committed and pushed in `081db89`: the reference models and bench tests (peel, wedge, cut, stretch, canal skin).
- Not committed: the fast solvers, the solver race, the CHOLMOD/OpenBLAS work, the material table, the layered stack, the per-material and composite tests, the GIFs, and this file.

This covers six things:
1. How the soft tissue is checked: the slow Python reference models and their bench tests.
2. The fast C++ solvers that will run inside the AMBF plugin.
3. The race between the fast solvers.
4. **The material and interface table**, with sources, and every fixture run on every material (§5).
5. **Layered (composite) tissue**: which plane lets go, and grabbing one layer at a time (§6).
6. Every package installed or needed, and how (§7).

---

## 1. Approach in one paragraph

The approach is one framework with many laws.
- Each tissue layer (skin, periosteum, canal skin, and so on) has its own **bulk material law**.
- Each interface between layers (the loose areolar plane, periosteum-to-bone, canal skin-to-bone) has its own **separation law**, with separate opening (peel) and sliding (shear) strengths.

The bench tests are *fixtures* that check each law against an answer known from engineering theory. They are not the tissue model itself. The same energy is then solved fast in C++ and checked against the slow solver.

**Verification vs validation.** Everything here is *verification*: the code solves its equations correctly. *Validation*, meaning the numbers match real tissue, needs bench or cadaver data. **Every material number below is a placeholder until then.**

---

## 2. Slow reference models (Python, offline)

Files are in `scripts/soft_tissue/`. Python 3.12.3 in `middle_ear_newton/.venv` (numpy 2.5.4, scipy 1.18.1, PyYAML).

### The shared model: `peel_model.py`

A thin skin sheet of triangles. Its total energy is the sum of:

| Term | What it is |
|---|---|
| membrane | St Venant–Kirchhoff, plane stress, times thickness. Optional **J-curve** collagen stiffening along each rest axis: extra energy k1/(2k2)·(exp(k2 E²) − 1) |
| bending | discrete hinge per interior edge, calibrated so a cylinder stores D/2·κ² per area, D = E h³ / (12(1 − ν²)) |
| bonds | breakable bilinear traction–separation bond per bonded vertex. The area under the law = adhesion energy G. Damage is permanent (unloads along its secant, never heals) |
| contact | one-sided penalty keeping the sheet above the floor (`floor_z`) |

Solver: damped Newton, quasi-static, on an assembled sparse Hessian with scipy `splu`. Damping grows until the energy drops, which handles buckling and bond softening. Exact and slow, which is the point.

Default parameters (`SkinParams`, all placeholders):
- E 1 MPa, ν 0.45, thickness 1 mm
- G 0.05 N/mm (50 J/m²), δ0 0.05 mm, δc 0.4 mm
- contact 200 N/mm³
- J-curve off (k1 = k2 = 0); the stretch test turns it on

### The bench tests

Each test has a known answer from engineering theory.

| Test | Surgical meaning | Known answer | Tests (time) |
|---|---|---|---|
| **Peel** (`peel_strip.py`) | pulling tissue off a surface (middle-ear band lysis) | Kendall: (F/w)(1 − cos θ) + (F/w)²/(2Eh) = G. Front curl √(2G/D) | `test_peel_model.py`, 5 tests (57 s) |
| **Wedge** (`wedge_test.py`) | elevating a flap with a round knife / Rosen (TM flap is wedge-like, not traction) | energy balance; push = G·width for a thin blade; Boeing wedge test lift-off a = (9DΔ²/2G)^¼ with elastic-foundation correction | `test_wedge.py`, 4 tests (110 s) |
| **Cut** (`cut_test.py`) | knife through skin/fascia; incision weakening the auricle's hold | Griffith/Inglis: crack opening 4σa/E, energy released πσ²a²h/E (finite width sec(πa/W)); blade force (Jc − G)·h | `test_cut.py`, 5 tests (22 s) |
| **Stretch** (`stretch_test.py`) | auricle retraction force | exact J-curve F = w·h·s·(E e + k1 e exp(k2 e²)); initial stiffness (E + k1)·w·h; rupture at the weak row | `test_stretch.py`, 3 tests (5 s) |
| Canal skin (`canal_skin.py`) | builds the EAC skin sheet from the volume's EAC label | regions, normals, annulus taper, lumen stays open | `test_canal_skin.py`, 4 tests (2 s) |

What each test checks:
- **Peel:**
  - every gradient matches finite differences
  - the assembled Hessian matches a dense one
  - a bent cylinder stores the plate bending energy
  - the area under the bond law is G, and damage doesn't heal
  - a short peel reaches the Kendall force within 1%
- **Wedge:**
  - before anything breaks, push force equals the energy slope (±1%)
  - a thin blunt blade pushes with G × width (±5%)
  - a sharp edge frees tissue at its cutting cost (±12%)
  - a thick blade lifts tissue where beam theory says (within 15%; energy balance 0.99–1.06; push ±8%)
- **Cut:**
  - splitting keeps tips joined and the two sides apart
  - the incision gapes and gives up energy like Griffith
  - a finer mesh moves toward theory. The coarse mesh releases 0.77 of theory; an independent linear FE confirmed this is the triangle elements converging slowly at the crack tip, not a bug.
  - release scales with tension²
  - blade force = toughness minus released energy
- **Stretch:**
  - a uniform pull follows the exact J-curve
  - initial stiffness is E + k1
  - the strip snaps at its weak row and then carries nothing

Postponed by your decision: **hold** and **press**.

Run them all (about 3.5 min):
```
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_peel_model.py
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_wedge.py
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_cut.py
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_stretch.py
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_canal_skin.py
```

---

## 3. Fast solvers (C++, for the AMBF plugin)

Files:
- `fast/soft_fast.cpp`: the solver library, with a plain C interface so both Python (ctypes) and the AMBF plugin can call it. Built on first use into `fast/_build/` (git-ignored).
- `fast_solver.py`: Python wrapper. `FastModel.from_sheet(...)` builds the C++ model from the *same* mesh and parameters as the reference, so the reference can referee it. `FastModel.from_tets(...)` handles volumes.
- `test_fast_solver.py`: 7 tests, about 5 s (plus about 15 s the first time, to compile).

**Same energies as the reference:**
- triangles (St Venant–Kirchhoff + J-curve), with a **material per triangle**, so one model holds several layers
- hinges
- tetrahedra (St Venant–Kirchhoff volume)
- bonds to a fixed substrate (one law, or separate opening/sliding)
- **layer-to-layer bonds** (`PAIR`): breakable bilinear bonds between a vertex and its partner in the next layer
- the **elevator blade** (`BLADE`): the wedge test's rigid ramp
- the floor

**J-curve cap.** Past k2·e² = 8 (about 3× any material's rupture strain) the collagen term continues as its quadratic Taylor expansion instead of the exponential. Nothing below rupture changes. Without the cap, a fast hand could over-stretch one row of a stiff, thin layer in a single frame, the energy reached 1e26, and the solver never recovered. The same law is in the reference, the C++ solver and the exact formulas (`peel_model.collagen`).

**Units:** mm, ms, g, N, which are consistent (1 g·mm/ms² = 1 N). Tissue density is 1e-3 g/mm³.

**Each frame** minimises the implicit-Euler incremental potential m/(2h²)·|x − x_pred|² + E(x), warm-started from the last frame.

### Newton (the one we keep)
1. Builds each element's stiffness block **analytically**:
   - **Triangles and tets** (St Venant–Kirchhoff + J-curve): material part Jᵀ·C·J, where J is the strain derivative and C the Voigt stiffness, plus geometric part (βₐᵀ S β_b)·I, where S is the stress. In tension S ≥ 0, so the block is already positive semi-definite and no eigen-decomposition is needed. Only compressed elements are projected.
   - **Bonds:** eigenvalues are ψ″ along the separation and ψ′/d across it. Projection clamps ψ″ ≥ 0, which is exact.
   - **Hinges:** Gauss–Newton. **Floor:** exact.
   - The old finite-difference path is kept behind `set_hessian(analytic=False)` so a test can compare the two.
2. Assembles the blocks into a sparse matrix with a fixed pattern (no re-sorting each frame). Element blocks are computed in parallel; each matrix entry then sums its own precomputed list of contributions, so the add step is parallel too.
3. Factors it (Eigen LDLT, or **CHOLMOD supernodal** when available; see §7), and takes the step with an Armijo line search. Damping grows if the matrix isn't positive definite.

### XPBD (tested, rejected for skin)
Gauss–Seidel over elements, each element's energy treated as one constraint C = √(2E) with compliance 1, plus substeps and optional mass scaling.

### Tests (`test_fast_solver.py`)
- energy and gradient match the reference to 1e-12
- tetrahedron energy is exact for uniform stretch, and the gradient matches finite differences
- Newton matches the exact force (sheet and volume) within 0.1% at real-time frame steps
- Newton peels at the Kendall force (within 8%)
- CHOLMOD and Eigen take the same Newton step (to 1e-9)
- analytic element Hessians match finite differences, raw and projected, on a stretched, crumpled J-curve sheet with damaged bonds, on tets, and on mixed opening/sliding bonds. About half the triangles and bonds in that test are indefinite, so the projection path is exercised. Agreement is about 1e-9; the test allows 1e-5.
- XPBD only converges with mass scaling: ×10⁴ is within 1%, true mass is off by more than 50%

### Tests (`test_stack.py`, 3 tests, about 25 s)
- a layered mastoid stack (with blade, damaged layer bonds, some broken) has the same energy and gradient in C++ as in the reference (1e-11), and the same blade push
- the new layer-bond and blade stiffness blocks match finite differences, including in their non-convex ranges
- an 8-step ramped lift gives the same positions (1e-6) and the same broken bonds on both solvers

---

## 4. Solver race (`solver_race.py`)

A grip moves a little each frame at a real-time rate; each solver gets a fixed amount of work per frame. Frames of 1 ms (haptic rate) and 5 ms (200 Hz). 8 assembly threads.

### Scenarios
- **sheet:** J-curve skin strip, 10 × 2 mm, 0.25 mm mesh, pulled to 30% (exact force)
- **volume:** tetrahedral block, 6 × 2 × 2 mm, 0.5 mm mesh, pulled to 20% (exact force)
- **peel:** 25 × 5 mm strip peeled at 90°, 0.25 mm mesh, 2121 vertices (Kendall 244 mN)
- **peel_coarse:** the same peel on a 0.5 mm mesh, 561 vertices

### Accuracy (error vs the exact force or Kendall)

| Scenario | Newton 1 it, 1 ms / 5 ms | Newton 3 it | Best XPBD |
|---|---|---|---|
| sheet | 0.09% / 2.6% | exact | blows up (J-curve overflows); 9% at best |
| volume | 0.01% / 0.12% | exact | 5–6% at best, up to 100% |
| peel 0.25 mm | 1.3% / 4.3% | 1% | 12% at best, otherwise 2.6–4× off |
| peel 0.5 mm | 3.5% / 12.5% | 2–3% | 10% at best, otherwise 1.5–14× off |

**Why XPBD loses:** skin is extremely stiff for its tiny mass at millimetre mesh sizes. XPBD only converges if the tissue is made 100–10,000× heavier, which would feel sluggish, and even then it's unreliable. XPBD suits soft, coarse, game-like objects, not forces that must match measurements.

### Speed: Newton, 1 iteration, 5 ms frames, median solve time per frame

Each column adds one change to the one before it.

| Scenario | Eigen LDLT, finite-difference Hessians | + CHOLMOD, reference BLAS | + OpenBLAS | **+ analytic Hessians, parallel add (now)** |
|---|---|---|---|---|
| peel 0.25 mm | 18.7 ms (factor 11.5, assemble 6.4) | 18.4 ms | 11.1 ms (factor 4.5, assemble 5.9) | **7.8 ms** (factor 4.5, assemble 2.6) |
| peel 0.5 mm | 2.9 ms (factor 1.0, assemble 1.7) | 3.2 ms | 2.6 ms | **1.5 ms** (factor 0.7, assemble 0.6) |
| volume | 2.5 ms (factor 1.3, assemble 1.1) | 2.5 ms | 1.7 ms | **0.84 ms** (factor 0.6, assemble 0.2) |
| sheet | 1.1 ms | n/a | 1.2 ms | **0.86 ms** |

Overall: **2.4× faster on the fine peel, 3× on the volume**, with identical accuracy. Newton with 3 iterations on the 0.5 mm peel now fits a 5 ms (200 Hz) frame: 4.9 ms.

Findings:
- CHOLMOD alone gained nothing. Its supernodal method gets its speed from BLAS, and the system only had the slow reference BLAS.
- With OpenBLAS, factoring is **2.5× faster** on the fine mesh and 2× on the volume. Answers are identical.
- **One** OpenBLAS thread is as fast as four, so the wrapper sets `OPENBLAS_NUM_THREADS=1` by default. That leaves cores for haptics and graphics.
- Each frame does exactly one factorization; the line search isn't rejecting steps.
- Analytic Hessians cut assembly from 5.9 to 2.6 ms on the fine mesh. The rest is mostly memory traffic (about 10 MB of element blocks per frame, largely the 144-entry hinge blocks). More threads beyond 4 don't help.
- **Factoring is the bigger cost again** (4.5 of 7.8 ms). Further gains would come from reusing a factorization across frames, a coarser mesh away from the working area, or a cheaper hinge block. All three are optional at this point.

Results:
- `resources/soft_tissue/solver_race/results.json`: full race, Eigen
- `resources/soft_tissue/solver_race/results_cholmod.json`: Newton, CHOLMOD + OpenBLAS + analytic Hessians (latest)

Run:
```
middle_ear_newton/.venv/bin/python scripts/soft_tissue/solver_race.py                      # all contenders, best factor
middle_ear_newton/.venv/bin/python scripts/soft_tissue/solver_race.py --factor eigen --solvers "Newton 1 it"
```

### Plan for the plugin
- Newton runs in its own soft-tissue thread at about 200 Hz, with 1–3 iterations per frame. The mesh is about 0.5 mm near the working area and coarser elsewhere.
- Cap its threads so drilling haptics (1 kHz) and graphics (40–60 FPS) aren't starved.
- Newton's stiffness matrix gives the 1 kHz haptic loop a local linear force model between soft-tissue frames.

---

## 5. Materials: table and per-material tests

### The table (`tissue_params.py`)
Every number has a **nominal value, a low–high range and a source**: a citation, "clinical judgement (user)", or "estimate: reason". **57 of 86 values are estimates.** `python tissue_params.py` prints the regime report below; `tissue_params.sources()` lists every value with its source.

**Materials**
- skin: also cartilaginous canal skin
- bony canal skin
- TPF: subcutis + temporoparietal fascia, soft
- temporalis fascia
- periosteum
- perichondrium: periosteum × 0.7, your clinical note

**Interfaces**
- skin–subcutis: firm
- areolar: the loose dissection plane beneath TPF
- periosteum–bone
- perichondrium–cartilage: slippery in shear
- skin–perichondrium
- temporalis fascia on a rigid muscle stand-in
- canal skin–periosteum: fused
- annulus

**Stacks**, outside to inside:
- **mastoid:** skin =firm= TPF ~areolar~ periosteum | bone
- **temporal:** skin =firm= TPF ~areolar~ temporalis fascia | rigid muscle stand-in
- **bony canal:** canal skin =fused= periosteum | bone
- **auricle:** skin = perichondrium | cartilage (rigid)

**Sourced anchors:**
- **Skin:** linear modulus 83 MPa, failure strain 54% (Ní Annaidh 2012).
- **Temporalis fascia:** E 36 MPa, UTS 3.6 MPa, thickness 1.46 mm (Zwirner 2021).
- **Periosteum:** toe 1.9 MPa, linear 18–70 MPa (McBride 2011, Kadlub 2020).
- **Periosteum peel off bone:** 324–695 J/m² (Hamma 2025, porcine).
- **Tearing:** dermis 17 kJ/m², adipose 4.1 kJ/m² (Comley & Fleck 2010).

**Unsourced:**
- every shear (mode II) energy
- the areolar and skin–subcutis planes
- perichondrium and its adhesion
- canal skin and annulus
- friction
- periosteum thickness and strength

**How the J-curve is fitted:** a soft "toe" modulus E, plus collagen k1·e·exp(k2·e²). k2 is fitted so the stiffness reaches the published linear-region modulus at the failure strain, separately for each corner of the range.

### Regime report (nominal values)

| Material | Attached by | Bending zone (mm) | Bond lets go after (mm) | Mesh needed (mm) | Peel force (N/mm) | Tissue strength (N/mm) |
|---|---|---|---|---|---|---|
| skin | skin–subcutis | 1.14 | 8.0 | 0.38 | 0.193 | 13.0 |
| bony canal skin | periosteum–bone | 0.08 | 3.3 | **0.027** | 0.396 | 0.87 |
| TPF | areolar | 0.60 | 4.0 | 0.20 | 0.014 | 0.017 |
| temporalis fascia | muscle stand-in | 1.51 | 4.0 | 0.50 | 0.099 | 3.4 |
| periosteum | periosteum–bone | 0.22 | 3.3 | **0.072** | 0.435 | 0.93 |
| perichondrium | perichondrium–cartilage | 0.15 | 3.3 | **0.049** | 0.296 | 0.39 |

What it says:
- **Thin, stiff layers need a much finer mesh at their peel front** than skin: periosteum about 0.07 mm, bony canal skin about 0.03 mm.
- **Every tissue peels rather than tears at nominal values.** Perichondrium and TPF are close, with peel force over half their strength. A perichondrium that tears while being raised fits that.
- **Every attachment lets go only after 3–8 mm of stretch** (the failure separation, 2G/T). That follows from the *estimated* peak strengths, and it decides how a plane feels: stringy or crisp. It is the most important number to measure.
- **Consequence:** a blunt elevator lifting 0.1–1 mm can never free these tissues; only a sharp edge can. This matches surgery, where Sharpey's fibres are cut or scraped.

### Per-material tests (`material_bench.py`, `test_materials.py`)
Every fixture runs on every material. Each run is sized from that material's regime, so its breaking zone is resolved and steady state is reached.
- **quick:** the fast solver against theory; cut and stretch also on the slow reference. About 2 min.
- **full:** adds the slow reference for peel and wedge. About 25–100 min per case; results in "Full tier" below.

Fast solver, converged 5 ms frames (Newton iterates to convergence, up to 20), hand at 10 mm/s:

| Material | Peel / Kendall (ms per frame) | Peel, 3 iterations / Kendall (ms) | Sharp elevator push / edge cost | Cut release / Griffith (allowed) | Stretch: J-curve error / snap stretch |
|---|---|---|---|---|---|
| skin | 1.000 (5.9) | 1.00 (3.7) | 0.990 | 0.394 (0.273–0.728) | 2e-9 / 1.3751 |
| bony canal skin | 1.000 (26.8) | **88** (10.1) | 0.999 | 0.394 (0.273–0.728) | 4e-8 / 1.3751 |
| TPF | 1.000 (3.9) | 1.00 (2.4) | 0.996 | 0.728 (0.728) | 1e-6 / 1.7263 |
| temporalis fascia | 1.000 (4.4) | 1.00 (2.6) | 0.993 | 0.475 (0.364–0.728) | 3e-9 / 1.1646 |
| periosteum | 1.000 (9.2) | 1.02 (5.4) | 0.999 | 0.570 (0.485–0.728) | 2e-8 / 1.2625 |
| perichondrium | 1.000 (13.5) | **57** (6.5) | 1.000 | 0.570 (0.485–0.728) | 2e-8 / 1.2625 |

How each column is checked:
- **Peel:** against the general Kendall energy balance G = f(1 − cos θ + ε) − h·W(ε). It reduces to Kendall's formula for a linear arm and handles the J-curve arm exactly.
- **Cut:** the release must lie between Griffith at the stiffened modulus E + k1 and at the soft modulus E. Each bound is multiplied by the mesh's own crack-tip undershoot, measured on the same plate without stiffening. Fast equals slow to 1e-6.
- **Stretch:** each strip snaps exactly at its deliberately weakened row (99% of the failure strain), on both solvers.

**Finding: 3 iterations per frame are not enough for the thinnest layers.** Skin, TPF, temporalis fascia and periosteum peel correctly at the real-time budget. Bony canal skin and perichondrium, on 0.03–0.05 mm meshes, read 57–88× too high: the hand moves one or two mesh cells per frame, and 3 iterations can't settle a stiff film that far from equilibrium. Solved to convergence they are exact, but at 13–27 ms per frame. Either a slower hand (2 mm/s converges with 3 iterations) or more iterations fixes it. For the plugin this means:
- a fine mesh **only at the working front** (adaptive refinement)
- **iterate until converged** each frame, with a cap

---

## 6. Composite (layered) tissue

### Model (`stack.py`)
- Several layers on one in-plane mesh, outside to inside, each with its own material.
- Neighbouring layers are joined vertex to vertex by breakable bonds, one law per plane (your decision).
- The innermost layer is attached to a rigid substrate with separate peel and shear laws.
- An optional sharp elevator can be driven under the stack.
- `FastModel.from_stack` builds the identical C++ model; `test_stack.py` referees it.

**Simplifications to know:**
- **Layers share one mid-surface** (zero rest offset). That keeps layer bonds correct when a bonded flap turns. But a bonded stack bends like the *sum* of its layers, without the extra stiffness of layers offset from its centre, which is a lower bound.
- **Separated layers do not collide yet.** See the grasp-periosteum GIF, where the periosteum passes through the elevated flap.

### Results (`composite_bench.py`, `test_composites.py`; fast solver, converged frames)

| Case | What happened | Check |
|---|---|---|
| Mastoid, grip skin | skin + TPF peel off periosteum together: areolar plane 100% broken, skin–TPF 0%, bone 0%. Skin rose 29.5 mm, periosteum 0.002 mm | force / two-layer Kendall (areolar G) = **1.001** |
| Mastoid, grip all three | all come off bone together; no layer plane broken | force between G_I and G_I + G_II bounds (see below) |
| Mastoid, sharp elevator under periosteum | flap comes up full thickness; layer planes 0% broken | push / edge cost = **0.993** |
| Mastoid, incision through skin + TPF only, under 0.2% stretch | grip force drops 12.7% | less than with the planes unbonded (13.8%) or cut full thickness (17.4%): the intact periosteum bridges the cut |
| Grasp periosteum only (skin flap already raised) | periosteum lifts 1.5 mm; skin and TPF move 0.0 mm | layer selectivity |
| Temporal, grip skin | skin + TPF peel off temporalis fascia; fascia rises 0.016 mm | force / Kendall = **1.000** |
| Bony canal, sharp elevator | canal skin and periosteum lift as one; fused plane intact | push / edge cost = **1.000** |
| Auricle, lap shear (slide along cartilage) | perichondrium slides off at 0.65 N (0.8 mm strip) | / Kendall at 0° = **1.000** |
| Mastoid, lap shear (periosteum along bone) | 4.09 N, about 6× harder than perichondrium on cartilage, as the table sets | / Kendall at 0° = **1.000** |
| Auricle, grip skin | skin lets go of perichondrium (the weaker plane); perichondrium stays on cartilage | force / Kendall = **1.000** |

**Finding: the full-thickness peel takes 1.48× the pure-peel force.** At a 90° peel, the bone bonds still hanging under the flap are dragged sideways as well as lifted. The attachment law tracks peel and shear separately and lets each absorb its full energy until either breaks, so mixed loading costs extra. The result lies inside the bound G_I … G_I + G_II (0.40 of the upper bound). A proper mixed-mode law, with one damage variable on an effective separation and a mixed-mode criterion such as Benzeggagh–Kenane, would make it predictable. Recommended before the elevation demo.

### GIFs (`composite_demo.py`, side view; layers drawn offset by thickness)
- `resources/soft_tissue/composite_demo/mastoid_plane_selection.gif`
- `resources/soft_tissue/composite_demo/mastoid_full_thickness.gif`
- `resources/soft_tissue/composite_demo/grasp_periosteum.gif`: shows the missing contact between released layers
- `resources/soft_tissue/peel_demo/peel_slow_vs_fast.gif`: single skin strip, slow vs fast

### Full tier: slow reference vs theory and vs fast (2026-11-08)
Every check passes (`test_materials.py` checks on `*_full.json`; `test_composites.py full files`). Each case ran in its own process; times are single-thread wall clock.

Per material, test-sized strips:

| Material | Peel slow / Kendall | Peel fast / slow | Peel slow time | Wedge energy balance | Wedge slow push / edge cost | Wedge fast / slow | Wedge slow time |
|---|---|---|---|---|---|---|---|
| skin | 1.000 | 1.000 | 1202 s | 1.000 | 1.000 | 0.997 | 186 s |
| bony canal skin | 1.000 | 1.000 | 1527 s | 1.000 | 0.999 | 1.001 | 296 s |
| TPF | 1.000 | 1.000 | 1339 s | 1.000 | 1.007 | 0.992 | 1287 s |
| temporalis fascia | 1.000 | 1.000 | 722 s | 1.000 | 1.002 | 0.997 | 120 s |
| periosteum | 1.000 | 1.000 | 371 s | 1.000 | 1.000 | 1.000 | 604 s |
| perichondrium | 1.000 | 1.000 | 627 s | 1.000 | 1.000 | 1.001 | 926 s |

Composites:

| Case | Slow result | Fast / slow | Fast time | Slow time |
|---|---|---|---|---|
| mastoid plane selection | force / Kendall 1.000; areolar plane broken, periosteum rises 0.0016 mm | 1.001 | 24 s | 4938 s |
| mastoid full thickness | 1.503 × G_I-Kendall (inside the G_I … G_I+G_II bound; known mixed-mode gap) | 0.993 | 15 s | 2179 s |
| mastoid elevator | push / edge cost 0.999; layers intact | 0.997 | 4 s | 3723 s |
| bony canal elevator | push / edge cost 1.000; skin + periosteum lift as one | 1.000 | 12 s | 3755 s |
| mastoid incision | drops 0.127 < 0.138 < 0.174 | identical to 1e-4 | 3.5 s | 80 s |
| grasp periosteum | periosteum 1.51 mm up, skin 3e-11 mm | 0.990 (grasp force) | 1.2 s | 84 s |
| temporal plane selection | force / Kendall 1.000; temporalis fascia rises 0.016 mm | 1.001 | 23 s | 6348 s |
| mastoid lap shear | force / √(2EhG_II) 1.000 | 1.000 | 16 s | 835 s |
| auricle lap shear | 1.000 | 1.000 | 11 s | 428 s |
| auricle plane selection | 1.000; skin–perichondrium plane lets go first | 1.000 | 10 s | 948 s |

The slow reference agrees with theory to 0.7% or better everywhere, and the fast solver agrees with it to 1% or better (worst: TPF wedge 0.8%, grasp force 1.0%). The fast solver is 30–500× faster at these sizes. Results: `resources/soft_tissue/materials/*_full.json`, `resources/soft_tissue/composites/composites_full.json`.

Run:
```
middle_ear_newton/.venv/bin/python scripts/soft_tissue/tissue_params.py          # regime report
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_stack.py
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_materials.py quick   # about 2 min
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_composites.py quick  # about 1 min
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_materials.py full    # long
middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_composites.py full   # long
```

---

## 6b. Cartilage as a curved shell (step 1 of the auricle grasp plan, 2026-11-10)

The auricle's cartilage is a curved plate. It now bends in both solvers, checked against exact answers.

**What changed**
- **Curved rest shapes** (`peel_model.Sheet(curved=True)`): the sheet stretches from its rest triangles in their own planes, and bends from each hinge's rest angle. Before, the rest shape had to be flat (z = 0).
- **Shell bending** (`bending="shell"`): hinges alone charge only mean curvature, so a strip curls into a saddle almost for free. Measured: a cantilever 19–20% too soft, curling the wrong way (+0.18 instead of −ν). The shell gives each triangle a shape operator from its three edges' signed bend angles (Grinspun et al. 2006) and charges the plate energy D/2·((1 − ν)·tr ΔS² + ν·(tr ΔS)²). It is calibrated on a cylinder like the hinges. Skin keeps the hinges (the default).
- **Fast solver:** rest angles per hinge (`sf_set_hinge_rest`) and a 6-vertex `SHELL` element per triangle (`sf_set_shell`), with energy, gradient and a Gauss–Newton Hessian. Element stencils grew from 4 to 6 vertices.
- **Tetrahedra were tried and rejected** for the plate. Linear tets lock in bending: 1.93× too stiff with 2 layers through the thickness, 1.26× with 4 (21k tets for a 20 × 4 mm strip).
- **Material:** `tissue_params.SHELL_MATERIALS["auricular_cartilage"]`, with E 5.0 MPa (2.8–5.8, tension) and thickness 1.0 mm (0.8–1.25). Poisson's ratio 0.3 and density are estimates. Sources are in the table and in `AURICLE_LITERATURE.md`.

**Tests** (`plate_test.py`, `test_plate.py`: 7 tests, about 6 s). Slow and fast agree to rounding: energy 1e-15, gradient 5e-15.

| Fixture | Known answer | Slow | Fast |
|---|---|---|---|
| curved shell at rest, or moved rigidly | no force | 4e-14 N | 4e-14 N |
| cylinder re-bent from R 8 to 5 mm | D/2 (1/R1 − 1/R0)² per area | 0.975 | 0.975 |
| narrow cantilever, small deflection | 3 EI d / L³ | 1.021 | 1.021 |
| sideways curl of the bent strip | −ν (−0.3) | −0.283 | −0.283 |
| large deflection (half the arm) | exact elastica | 1.017 | 1.017 |
| narrow ring (1 mm) pinched across a diameter (curved rest) | exact ring elastica | 1.032 | 1.032 |

Wider strips stiffen from the beam value toward the plate value D·w, as theory says. A curved ring shows it most: a 3 mm ring on R 8 mm cannot curl freely across its width and comes out 5% over the beam elastica (2% of that is the width; the 1 mm ring isolates the element's own ~3% overshoot, the same as the cantilever's 2%). The earlier 1.010 for the ring compared against the small-squeeze formula at a squeeze where the exact ring is already 4% softer, so it hid that.

**GIFs** (`plate_demo.py`, every frame a solved state, slow and fast drawn together over the exact answer): `resources/soft_tissue/plate_demo/` — `cantilever_elastica.gif`, `poisson_curl.gif`, `pinched_ring.gif`, `rebend.gif`.

**Next:** offset layer bonds (done, §6c), fascial bands, contact with friction, and pincer pads (the auricle grasp-and-pull plan, each solved slow and fast first).

## 6c. Bonded layers at their real thickness (step 2 of the auricle grasp plan, 2026-11-10)

Until now, layers in a stack shared one mid-surface, so a bonded stack bent like the sum of its layers. Now layers can sit at their real heights and be bonded face to face, so a bonded stack bends as a laminate. Peel and slip go through the same bonds.

**What changed**
- **`stack.Stack(offset=True)`:** each layer's mid-surface sits at its real height. One bond per interior mesh edge joins a layer to the next, at the faces where they touch:
  - Each bonded point is the edge midpoint, pushed along the mean normal of the edge's two triangles by half the layer's thickness.
  - That normal is centred, so a uniformly bent laminate matches exactly to first order.
  - Two other choices were tried and locked: bonding at triangle centroids with the triangle's own normal (+57% at a stiff bond), and including boundary edges (+5%). A triangle's normal belongs to the middle of its span, not its centroid, and a boundary edge has only one triangle.
  - Each triangle's area is shared among its interior edges, so the bonded area is the whole area.
- **Laws:** one law on the full separation (`pm.BondLaw`), or a mixed interface (`wedge_test.Interface`). A mixed interface has opening along the inner layer's normal and sliding across it, and resists compression at the opening's stiffness. That keeps perichondrium on cartilage slippery now that cartilage is a deforming layer.
- **`bending` per layer:** "shell" (cartilage) and "hinge" (skin) can be mixed in one stack.
- **Fast solver:**
  - New `FACE` element, an 8-vertex stencil (the stencil limit is now 8), with energy, gradient and a Gauss-Newton Hessian; `sf_set_faces` and `sf_face_state`.
  - Shells now replace only their own layer's hinges.
  - `static_solve` accepts stacks.
- **Unchanged:** the default (`offset=False`) and every existing fixture.

**Tests** (`laminate_test.py`, `test_laminate.py`). Slow and fast agree to rounding: energy and gradient 1e-15, tip forces to 1e-4.

| Fixture | Known answer | Slow | Fast |
|---|---|---|---|
| bonded three-layer stack at rest, or moved rigidly | no force | 1e-14 N | 1e-14 N |
| two layers (0.4 mm at 15 MPa on 1 mm cartilage), slip parameter αL 0 → 100 | partial interaction (Newmark 1951) | 1.007–1.021 | 1.007–1.020 |
| same at αL 1000 (nearly rigid bond) | partial interaction | 1.032 | 1.032 |
| one law on the full separation, αL 30 | partial interaction | 1.008 | 1.008 |
| three layers, stiff bonds | transformed section (5× the layers bending apart) | 1.025 | 1.025 |
| outer layer peeled at 90° off the inner, G 0.2 N/mm | Kendall | 0.996 | 0.996 |

The remaining ~1–2% is the shell's own overshoot on this mesh (the bare cantilever is 1.021).

- **Test fixtures:** the cantilever pushes only the inner layer's tip, and the bonds carry the load. Pushing every layer's tip to the same height over-constrains a rotated bonded end; that alone read 12% stiff.
- **Peel:** the grip is held in height only, so it follows the front. Holding it in place lowers the peel angle and read 2.5× Kendall.
- **Slip near the clamp:** next to the clamp, the slip between faces converges to Newmark's only with mesh refinement. At 1 mm from the clamp it reads 1.44×, 1.18× and 1.05× at elements of 0.25, 0.125 and 0.0625 mm; the stiffness is unaffected.

**The auricle as a laminate** (skin 0.8 / perichondrium 0.15 / cartilage 1.0 / perichondrium / skin, with the table's interfaces, small strain, fast solver): bending stiffness **EI 2.6 N·mm² per mm width, 6.2× the bare cartilage** (0.42). That is 72% of fully bonded (3.6), because the perichondrium planes slip a little. Skin is soft (1.6 MPa at small strain), but it sits about 1 mm from the middle, so it adds most of that.
- Skin's small-strain stiffness is an estimate in the table, so this number is the first thing to check against the whole-auricle measurement (Zopf: 0.194 N/mm at the helix) once the auricle mesh runs.

**GIFs** (`laminate_demo.py`, every frame a solved state, slow and fast over the exact answer), in `resources/soft_tissue/laminate_demo/`:
- `partial_interaction.gif`: the bond swept from free slip to rigid; slip along the strip and the stiffness against Newmark.
- `layer_peel.gif`: the outer layer peeled off, face bonds breaking; force against Kendall.

**Next:** the auricular muscles and ligaments as fans (done, §6d).

## 6d. Auricular muscles and ligaments as fans (step 3, 2026-11-10)

The auricle is held on the head by the auricular muscles (anterior, superior, posterior) and ligaments. Per your decision, they are not modelled as muscle tissue and are not drawn. Each is a **fan of tension-only fibres** from an origin line on the head to an insertion patch on the cartilage; the superior one is about 4.9 cm wide at its origin, narrowing onto the top of the auricle (your figure; Chon 2021: 4.7 × 5.0 cm). The reference figure is a left ear and must be mirrored for RT143.

**What changed**
- **`fibres.py`:**
  - `Fibres`: one cable per fibre, insertion vertex to a fixed origin point (or to another vertex, once the origin layer moves).
  - The fibre law is the sheets' J-curve along the fibre: Green strain, W = E e²/2 + collagen, energy A·L₀·W. A fibre has no energy when slack.
  - `fan(...)` spreads fibres from an origin line to the insertion vertices without crossing, and shares the fan's section (width × thickness) among them.
- **`stack.Stack(fibres=...)`:** fibres attach to any layer.
- **Fast solver:** a `FIBRE` element (1 or 2 vertices) with an analytic Hessian (both terms ≥ 0 in tension, zero when slack); `sf_set_fibres`.
- **Material table** (`tissue_params.FAN_MATERIALS`, `FANS`, `fan_law`), from `AURICULAR_MUSCLES_LITERATURE.md`:
  - **Muscle:** passive muscle along its fibres, linear region 0.45 MPa (rabbit, Morrow 2010; human temporalis up to 1.58), failure Green strain 0.64; toe and recruitment are estimates.
  - **Ligaments:** fascia stand-in, 26 MPa (deep temporal fascia).
  - **Fan geometry:** superior from your figure and Chon 2021; posterior length 28 mm (Donofrio 2024); the anterior muscle and both ligaments have no published dimensions (labelled estimates).
  - **Small-pull stiffness:** superior 33, posterior 21, anterior 16 mN/mm (the toe of their J-curves); posterior ligament 1.6 N/mm and anterior ligament 2.3 N/mm. In a gentle pull the ligaments hold the ear; the muscles take over far up their curve.
  - **Whole-ear target:** the auricle folded forward recoils with 84 ± 10 N intact, 53 N without the posterior muscle, 40 N also without the posterior ligament (Chi 2021). That is an end-of-range fold with no displacement given, so it is an upper bound for the real-ear pull.

**Tests** (`fibre_test.py`, `test_fibres.py`: 5 tests, about 10 s). Slow and fast agree to rounding (energy 1e-16, gradient 3e-18).

| Fixture | Known answer | Slow | Fast |
|---|---|---|---|
| fibres at rest | no force | 7e-17 N | 7e-17 N |
| two unequal fibres in series, stretched ×1.05, ×1.2, ×1.4 | one tension, solved exactly on the J-curve | 1.00000 | 1.00000 |
| fan on a cartilage cantilever tip, pushed down 0.02 mm | plate + Σ A(E+k₁)/L₀ cos²θ | 1.0014 | 1.0014 |
| same, lifted 0.02 mm | fan slack: plate alone | +0 N | +0 N |

**GIF:** `resources/soft_tissue/fan_demo/fan_on_plate.gif`, from `fan_demo.py`. The tip is lifted (the fan goes slack and its fibres turn grey) and then pushed down (the fan pulls taut, coloured by tension). The force kinks at zero and stiffens along the J-curve. Slow and fast are drawn together. The demo fan is drawn thin so the plate's own force is visible.

**Next (your order):** the first pull on the real composite ear with the mouse; then your tracked hands through WiVRn; then contact, friction and the pincer grasp.

## 6e. First pull on the real ear (2026-11-10)

The composite ear from `build_ear.py` now runs as a model you can pull with the mouse:

```
scripts/soft_tissue/launch_auricle_pull.sh
```

Ctrl + left-drag pinches the auricle where you click and pulls it in the screen plane; releasing lets it spring back. Log: `~/.cache/fivrs-auricle-pull/viewer.log`.

**How the model is built** (`auricle_sim.py`)
- **Cartilage mid-surface:**
  - Source: your Blender-edited cartilage, before the composite clipped it. The clipped version is in about 5,000 fragments.
  - Thickness: from ray pairs across the plate, median 1.6 mm.
  - Shape: mid-points oriented over the whole plate, a Poisson surface through them, then cleaned and remeshed to 1.5 mm. Result: 1,868 vertices, 3,031 mm². Cached as `sim_cartilage_mid.npz` / `.ply` next to the composite.
- **Layers:** lateral skin 0.8 mm / cartilage 1.6 mm / medial skin 0.8 mm, face-bonded along the curved mid-surface (§6c). Skin is bonded to the cartilage with the skin-perichondrium law. 5,604 vertices.
- **Curved offset layers needed two changes:**
  - `stack.Stack(curved=True)` places the layers along vertex normals.
  - Each face bond keeps its rest separation in its own frame (inner edge, normal, their cross product). On a coarse, curved mesh the offset layers cannot meet every edge exactly: without this the ear held 3.8 N at rest; now 1e-13 N, and rigid motion is still free. Both solvers; the flat tests are unchanged.
- **Base:**
  - **What:** the medial skin over the cartilage's base (within 2 mm of the head's skin: concha back, tragus; 393 vertices) is tied to its rest position by elastic soft tissue.
  - **Stiffness:** 0.005 N/mm³ per area, an estimate of the areolar/subcutaneous layer's shear stiffness over its depth. About 2.8 N/mm in all. **This is the main knob** (a slider in the viewer).
  - **Why not clamped:** a rigidly clamped base made the ear about 3.5 N/mm and Newton failed past 4 mm. The cartilage had to stretch in-plane (25% strain) because the ear could not swing.
- **Fans (§6d):**
  - **Directions:** from the volume's anatomical frame, which mirrors the left-ear figure.
  - **Insertions:** each is a patch round the base's extreme point in its direction.
  - **Muscle origins:** at their length along that direction, 3 mm under the head's skin. The scan ends 33 mm above the ear, so the scalp is assumed to carry on straight up beyond it.
  - **Ligament origins:** on bone.
- **Drawing:** the composite's auricle skin and cartilage ride on the model. Each drawn vertex keeps its offset from its nearest mid-surface triangle, in that triangle's frame.

**First result:** pulling the helix up and back 10 mm (a 3 mm pinch through all layers) takes **0.9-1.1 N per mm, 11 N at 10 mm**, rising slightly with the pull.
- **Where the energy goes** at 5 mm: soft-tissue base 53%, cartilage in-plane 24%, cartilage bending 8%, skins 10%, face bonds 4%, fans 1%. The whole ear swings about 2 mm at its base.
- **Slow vs fast:** at 1 mm the force agrees to 4.5e-8. The slow solve took 12½ min; fast takes 0.1-0.3 s.
- **Speed:** 90-170 ms per Newton iteration (4 threads, CHOLMOD), 4 iterations per frame.

**Known limits of this first version**
- No contact: the auricle can pass through the head.
- The lobule has no cartilage, so it is carried by the nearest cartilage.
- Perichondrium is part of the skin-cartilage bond.
- Drawn skin beyond the cartilage, near the canal entrance, stretches into a band when the ear moves.
- The fans barely load in this pull, because their insertions sit near the base, which hardly moves.
- Nothing is calibrated against a whole-ear measurement yet. Chi 2021's 84 N is an end-of-range fold, not this.

## 6f. How much physics the ear needs to look right (2026-11-10)

The pull viewer ran at 300-500 ms a frame. The question: are we computing more than the eye can see? `mesh_sweep.py` applies the same set force to the ear built several ways and judges each on what is drawn, not on the physics mesh.

```
middle_ear_newton/.venv/bin/python scripts/soft_tissue/mesh_sweep.py      # ~5 min
/usr/bin/python3 scripts/soft_tissue/mesh_sweep_plot.py
```

Output: `resources/soft_tissue/mesh_sweep/` (`results.json`, `shapes.npz`, `look_vs_speed.png`, `error_maps.png`).

**Set-up**
- **Load:** 3 N on a 3 mm pinch at the top of the helix, up and back. It is a set (dead) force, new in the fast solver (`FastModel.set_external`, test: `test_set_force_stretches_two_fibres_in_series_to_that_tension`).
- **Models:**
  - mid-surface edge 1, 1.5, 2, 3, 4 and 6 mm;
  - each as the 3-layer laminate (§6c) and as one equivalent shell (`laminate_shell`: the same stretch stiffness, bending stiffness fitted).
- **Look:** the drawn auricle skin (72,000 vertices, the same for every model) is carried on each model and compared with the 1 mm 3-layer answer. The reference skin moves 2.3 mm on average, 9.9 mm at most.
- **Live:** the force comes on over 6 frames and holds for 24, at 1, 2 or 4 Newton steps a frame. Measured: physics ms while the force changes, ms to place the drawn skin, and lag behind the settled shape.

**Mesh flaw found first.** Every mid-surface had 50-160 sliver triangles folded back over a neighbour (up to 180°), plus needles with corners down to 0.04°, left by the simplifier.
- **Effect:** at a fold, a face bond's direction (the mean of two opposite normals) is undefined. One coarse solve jammed at 4e7 N, and needles made Newton retry its factorization hundreds of times.
- **Fix:** `auricle_sim._unfold` collapses such triangles until no fold is over 100° and no corner under 12°. Every cached mid-surface was rebuilt, including the viewer's 1.5 mm one, now 1,642 vertices.
- **After the fix:** every model converges from rest in 16-21 Newton steps.

**Results** (1 Newton step a frame; placing the drawn skin adds about 9 ms)

| Model | Vertices | Physics ms/frame | Drawn skin off by (mean / 95%) |
|---|---|---|---|
| 1 mm, 3 layers (reference) | 11,016 | 182 | 0 |
| 1.5 mm, 3 layers (the viewer) | 4,926 | 64 | 0.13 / 0.28 mm |
| 2 mm, 3 layers | 2,658 | 29 | 0.21 / 0.52 mm |
| 1.5 mm, 1 shell | 1,642 | 8.3 | 0.21 / 0.38 mm |
| 2 mm, 1 shell | 886 | 4.7 | 0.35 / 0.56 mm |
| 3 mm, 1 shell | 364 | 2.7 | 0.37 / 0.85 mm |
| 4 mm, 1 shell | 186 | 1.3 | 0.49 / 1.05 mm |
| 6 mm (either) | 52-156 | ~1 | 2.4 / 6.5 mm: the mesh loses half the ear |

**What it says**
- **One Newton step a frame is enough.** Under a 6-frame tug, every usable model stays within 0.1 mm of its settled shape (1 mm 3-layer: 0.24 mm). The viewer's 4 steps a frame (258 ms) bought nothing visible.
- **The layers cost far more than the mesh.** One shell at 1.5 mm is 8× cheaper than 3 layers at 1.5 mm, as good as 3 layers at 2 mm, and fits a 60 fps frame with the skin placement.
- **The shell's stiffness had to be fitted.** With the flat-strip slip factor (0.72) the shell is 8% stiffer than the 3 layers here, and its error stopped at 0.27 mm whatever the mesh. Fitted to the reference (`PARTIAL_BOND = 0.45`), it falls to 0.21 mm. So the curved ear slips more between layers than a flat strip does.
- **The 3-layer model is still the reference.** It is what we fit cheaper models to, and it is needed where layers separate: peeling skin, elevating perichondrium.
- **Errors are largest at the free rim and lobule,** far from the pinch, where small differences in bending add up (`error_maps.png`). The 4-6 mm "worst" values are a few drawn vertices, near the canal, that change nearest triangle.

**The viewer now uses it.** `auricle_pull_viewer.py` defaults to the 1.5 mm shell at 1 Newton step a frame: 8-9 ms a frame in the scripted 10 mm helix pull, against about 320 ms before, with the same force (11.0 N at 10 mm). `--layers 3` brings back the full laminate.

**Then the drawing was the bottleneck.** With 9 ms physics the Open3D window still lagged while pulling:
- it re-created the 141,000-triangle skin and cartilage every frame (38-84 ms to draw them as new geometry);
- it queued a redraw every 10 ms whether or not the last one was drawn.

The viewer is now a plain OpenGL window (GLFW):
- **Meshes stay on the GPU;** only vertex positions and normals are rewritten each frame.
- **Physics and drawing run apart:** physics in its own thread, drawing the newest result at the display's rate.
- **Placement is faster:** frames per physics triangle and a sparse barycentric matrix take it from 9 to 2 ms per mesh.
- **Measured** in a scripted helix pull (`--demo`): physics 9.4 ms plus placing 12.9 ms while pulling, drawing at 60 fps. The title bar shows these live.

**Head skin follows the base (drawing only).** The head's skin near the ear moves with the auricle's base: an inverse-distance blend of the nearest base vertices' motion, fading as exp(-(r/8 mm)²) (`HeadSkinFollow`).
- **8 mm is an estimate** of how far the subcutis spreads a movement of the base.
- **It feeds no force back.** A physical version would make the head skin a layer bonded to the subcutis.


**Big pulls over-stretched; drawing moved to the GPU (2026-11-11).** Pulling the helix 10 / 20 / 30 mm with the old rigid 3 mm pinch: 11 / 27 / 63 N, and the most-stretched edge 23 / 41 / 77% longer (median under 3%), always at the rim of the pinch; the base slid 4-14 mm. Three causes, three changes:
- **The one shell had dropped the skin's J-curve.** `laminate_shell` now keeps it: the linear part (cartilage E h + both skins' E h) and bending EI as before, plus collagen k1 = 2 k1_skin h_skin / h with the skin's k2. Small-strain behaviour (what the sweep fitted) is unchanged; at ~30% stretch the skins' stiffness overtakes the cartilage's. Cartilage stays linear: the sourced data say it is near-linear up to tearing (~40%).
- **The pinch was a rigid clamp.** It is now a finger pad: grip springs in the fast solver (`sf_set_grip`, ½ k |x - t|², diagonal only, so a grab no longer rebuilds the sparsity pattern), 3 N/mm in all, shared by area × exp(-(r/3 mm)²). Test: `test_grip_spring_and_fibre_in_series_share_one_tension`.
- **Nothing limited the pull.** A pinch holds by friction: past `SLIP_N` = 10 N (estimate) the fingers slip and the pull stops growing (shown in the title bar).

| Hand moved | Force | Ear lags the hand | Most-stretched edge | Base slid (mean) |
|---|---|---|---|---|
| 5 mm | 3.2 N | 1.0 mm | 9% | 1.2 mm |
| 10 mm | 7.1 N | 2.2 mm | 17% | 2.6 mm |
| 20 mm | 10 N (slipping) | 9.3 mm | 22% | 3.7 mm |
| 30 mm | 10 N (slipping) | 19 mm | 23% | 3.7 mm |

A stiffer base made the stretch worse (the ear is pulled instead of moved), so the base was left as it is pending clinical tuning. The solver's pattern is now built at load (a first pull cost ~44 ms; now ~10 ms).

**Placement and normals on the GPU** (`gpu_place.py`, NVIDIA Warp kernels; the physics stays on the CPU and sends 1,642 points a frame). Skin, cartilage and head skin, 165k vertices: CPU numpy/scipy 20.8 ms; GPU 0.26 ms copied back for OpenGL, 0.10 ms if left on the GPU. Positions agree within 1e-4 mm. In the viewer (`--demo`): physics 8.7 ms (max 10.4), placing 0.5 ms (was 13), drawing 60 fps. `--cpu-place` keeps the numpy path.


**Long rough strips under big pulls were a drawing artefact (2026-11-11).** In the user's video (helix pulled far) the skin showed long parallel strips: a fluted band near the helix root, streaks on the lower helix, a sliver at the pinch. The physics surface was smooth (no physics edge bent more than 18°, stretch at most 23%), but drawn skin edges stretched up to 50× (99.9th percentile 14×). `Embedding` carries each drawn vertex on its one nearest physics triangle; where the nearest triangle jumps (skin bridging two parts of the cartilage, rims; half the drawn skin is ≥ 1.8 mm and 10% is 7-12 mm off the mid-surface), neighbouring drawn vertices ride parts 2.6-5.6 mm apart and the skin between them tears into strips.

| Drawing (20 mm hand pull) | Drawn edges bending > 20° | Worst drawn-edge stretch |
|---|---|---|
| nearest triangle (old) | 5,488 | 50× |
| per-vertex smooth rotations | 2,686 | (same seams) |
| harmonic fill along seams | (seams are everywhere: 10k edges) | — |
| **blend of 16 nearest physics vertices (`BlendEmbedding`)** | **822** | **9.9×** |
| blend of 32 | 373 | 5.5× |

`BlendEmbedding`: p = Σ w_j (x_j + Q_j (P − X_j)) over the 16 physics vertices nearest the drawn vertex P at rest, weights (1 − (d/d₁₇)²)³ (zero at the 17th, so the blend is continuous), Q_j each physics vertex's rotation (polar part of its triangles' rotations). Exact at rest. On the GPU (`gpu_place.py`, `wp.svd3`) all three drawn meshes take 0.53 ms; on the CPU it is ~85 ms, so without a GPU the viewer falls back to `Embedding`. The remaining red band (skin bridging the helix root) is now an even stretch, not strips. `mesh_sweep.py` still uses `Embedding`.

**Model picker.** The desktop icon now opens a small window first (`--pick`): one shell 1.5 mm (the current demo, default), one shell 2 mm, three layers 2 mm, three layers 1.5 mm, and smooth vs nearest-triangle skin drawing; the last choice is remembered (`~/.cache/fivrs-auricle-pull/last_model.json`). New models (three layers with head-skin physics) are added to `MODELS` in `auricle_pull_viewer.py`. Three layers at 2 mm: physics 29.7 ms, but drawing fell to 41 fps (to look at in the 60 fps sweep).


**Lobule pull bent the antitragus sideways: the lobule had no physics (2026-11-11).** The mid-surface covered the cartilage only, so a pinch on the lobule (10.9 mm from the nearest cartilage) fell back to one cartilage vertex at the antitragus tip, with the whole 10 N on it, and the drawn lobule hung off it on a 10 mm lever and swung sideways. The mid-surface now carries on into the lobule (`AuricleModel(lobule=True)`, cache `sim_auricle_mid*`): the edited skin has inner walls there, so the lobule's mid-plane comes from rays straight through it (first and last hit, 0.5 mm grid; median 6.3 mm thick), joined to the cartilage by bridging points. Its 249 triangles get a soft shell (`lobule_shell`: skin / fibro-fatty core 0.01 MPa (estimate, the subcutis value) / skin; no sandwich coupling) through per-triangle materials in the fast solver (`set_triangle_material`; the slow reference still uses one material per sheet). Now the pinch grips 12 lobule vertices and the lobule stretches itself. Left: the skin over the antitragus still draws as a stretched band, and small flaps appear where the lobule meets the cheek (drawn skin squeezed between the moving lobule and the anchored base). Viewer: 60 fps, physics 9.3 ms. The old cartilage-only model stays in the picker.

**Three layers drawn from their own layers.** In 3-layer mode the drawn skin now rides the two skin layers (blend over both, so each side follows its own layer and the rim blends them) and the drawn cartilage rides the cartilage layer. 2 mm, with lobule: physics 31 ms, drawing 43 fps.

**What the layers add during a pull: almost nothing visible.** Helix pinch, 3 layers at 2 mm: the skin layers slide over the cartilage by 0.08 mm mean (0.36 mm max) at a 5 mm pull and 0.15 mm mean (0.79 mm max, at the pinch) at 10 mm, while the cartilage moves ~4 mm; the one shell is within 0.21 mm of the 1 mm 3-layer reference (above). The layers matter where they separate (incision, elevation in a plane, peel), which suggests: one shell for deformation everywhere, the layers drawn, and the interlayer bonds as live physics only where tissue has been cut or is being elevated.


**The surgical scene as a cutaway block (2026-11-11).** The user's review found coarse muscles, auricular muscles in the wrong plane, and layers that looked loose and fanned out. Changes:
- **Layers 3 and 4:** the envelope now has loose areolar (layer 4, 0.4 mm, the dissection plane) on the temporalis fascia / periosteum, then the auricular muscles (a thickness map) and the temporoparietal fascia / SMAS (layer 3, 0.6 mm), under the fat and skin. Older saved thicknesses are split 40 / 60 (`envelope.upgrade_thickness`).
- **Temporalis:** the layers are built around the user's placed atlas temporalis (its thickness measured across it along the canal axis, only where it is the outermost tissue over the bone, smoothed 2 mm, edge tapered 3 mm). It is drawn as that layer's solid, sitting on the periosteum, with its fascia and everything above draped over it.
- **Auricular muscles** (`auricular_muscles.py`): footprints run from each physics fan's origin line to the arc of the auricle's base outline nearest its insertion, with a belly profile (posterior: two fascicles). The physics fans are unchanged.
- **Solids** (`build_ear.py`): each layer's outer surface, pushed in by its own thickness along the bone distance gradient (`Envelope.extruded_solid`), so each inner face lands on the layer beneath (median gap 0.02-0.17 mm). Cut exactly at the scene box: the CT cube inset 1.5 mm on its cut faces, out to the skin laterally. The bone, and the face's deep tissue in front of the ear (the build's face fill, which the layers lie on there), are solids too.
- **Pinholes:** the layer tracing's pinholes are closed (bowtie vertices split first).
- **Build memory:** the build blew up to 25 GB once (trimesh rays on the CT bone) and the kernel killed it twice. It now uses Open3D's ray caster and peaks at about 3.5 GB; builds run under a 16 GB cap.
- **Viewer:** draws the solids. L peels: skin, fat, TPF, auricular muscles, loose areolar, temporalis fascia, temporalis, then bone.


**Layers glide, not clear (2026-11-11, user review).**
- **Temporalis fascia:** an even 1.0 mm over the muscle. The user says it's submillimetre (about 0.6 mm), drawn 1.0 mm; Zwirner 2021 measured 1.46.
- **Loose areolar (layer 4):** a potential space, 0 at rest and not drawn. Layer 3 (TPF, 1.0 mm) lies straight on the fascia / periosteum.
- **What moves when the ear is pulled:**
  - **Before:** only the skin followed the ear's base. Pulling the posterosuperior helix toward the zygoma root slid it off the still fat, and its inward motion sank the superficial layers through the fixed fascia, which showed through.
  - **Now:** skin, fat, TPF and the auricular muscles move together. Any inward part of the motion is dropped, using each vertex's direction away from its nearest bone, so they glide over the deep layers or lift off them, never sink.
  - **Under the base:** the skin, fat and TPF solids are built from the untrimmed layers, so they run on under the auricle's base.
  - **The seam:** the auricle's skin within 3 mm of the head's skin follows it.
  - **Result:** at an 18 mm pull nothing deep is exposed.
- **Memory:** the viewer once took about 25 GB in trimesh's proximity query (72k points against 92k triangles) and was killed. It now uses Open3D. Builds and demos run under a systemd memory cap (MemoryMax).


**Tessellation under traction (2026-11-11).** The user saw faceted patches, mostly in the concha, under a strong pull. Causes found:
- **Hidden surfaces:**
  - **Concha tiles:** small separate pieces inside the edited auricle skin (inner walls in the lobule and concha; 500, 116 and 29 mm² against the 6,132 mm² outer skin). They were embedded on the deforming ear and poked through it. Now only the largest piece is drawn.
  - **Head skin under the ear:** a short-lived change ran the head's superficial layers on under the auricle, and they poked through the concha. Reverted: the head skin stops at the ear's base (user), and the auricle's skin within 3 mm of it follows the head's skin.
- **Layers sliding apart:** the no-sink limit was applied per layer vertex, so skin and fat over the same spot moved differently (3.4 vs 1.7 mm) and the skin slid off the fat. Now the inward part is dropped once, at the ear's base vertices, and every superficial layer follows that one field.
- **Still open:** pulled hard forward and down, the concha floor reaches the bony canal wall and the grey bone shows through. The physics has no ear-bone contact yet.


**Nothing passes through anything (2026-11-11, user; also needed in AMBF).**
- **Contact element:** the fast solver now has one against a static obstacle given as a signed distance grid: a one-sided penalty ½ k (φ − clearance)², Gauss-Newton Hessian k ∇φ∇φᵀ, and φ extrapolated past the grid edge. Test: `test_obstacle_contact_holds_a_pushed_sheet_at_force_over_stiffness` (rests exactly F/k in, gradient matches finite differences).
- **Obstacle:** `build_ear.write_obstacle` saves the head (bone, the face's deep tissue, and the envelope up to the skin) as `head_obstacle.npz`.
- **The ear's vertices:** 5 N/mm³ × vertex area. Free ones keep 0.8 mm off; the 150 that start inside the head's tissue (the root, the concha floor) may go no more than 0.3 mm deeper than at rest.
- **Result:** pulled hard toward the zygoma root, the concha no longer enters the canal's bone; pressed back, the ear lies on the head. Physics about 11 ms.
- **A frame-rate stall it exposed:** `Puller.drag` waited for the physics lock, so drawing ran at the physics rate whenever a solve passed 16.7 ms. It no longer waits: 60 fps.
- **Not yet covered:** contact between layers once they separate (the ear's self-contact: below).


**The ear against itself (2026-11-11).**
- **How it works:**
  - Self-contact in the fast solver, with vertex pairs found each Newton call by a spatial hash.
  - A pair is pushed apart (½ k (clear − d)²) only if it is separated mostly along both vertices' current normals (|cos| ≥ 0.6): two parts of the sheet facing each other.
  - Mesh neighbours within 4 rings never count.
  - Clearance is min(2.0 mm, rest distance − 0.3); stiffness 5 N/mm per pair.
  - The stiffness goes on each vertex's diagonal (2k: an upper bound on the pair's), so contacts coming and going never change the sparsity pattern.
- **Test:** `test_self_contact_holds_two_sheets_apart_by_the_clearance_less_force_over_stiffness`.
- **What I tried first:**
  - Plain vertex-to-vertex pairs without the facing test caught vertices squeezed together within the sheet near the pinch. A rim vertex held by one triangle collapsed it, and Newton fell into retries: 17-20 factorizations and about 140 ms a step.
  - The facing test fixed that (1 factorization, about 11 ms).
  - A clearance of the drawn ear's thickness (3.2 mm, or 2.5) brings the retries back on this 1.5 mm mesh, because the rolled helix rim's facing parts sit closer than that. 2.0 mm, about the physics shell's thickness, is stable. The drawn skin can still overlap up to about 1 mm in tight folds.
- **Result:** with the helix folded down 22 mm it lies over the concha instead of sinking into it. Physics 11.4 ms mean (single-frame peaks about 50 ms), 60 fps.
- **Diagnostics:** `FIVRS_NO_CONTACT`, `FIVRS_NO_HEAD`, `FIVRS_NO_SELF`.

**Open questions**
- **What error can a viewer see?** 0.2 mm on an ear seen through a headset is probably invisible, but that is a guess. It needs a perceptual threshold (side-by-side or forced-choice), not more bench numbers.
- **Is the reference right?** It has never been compared with a real ear. The sweep only shows how close each cheap model is to our best model.
- **Measuring real ears:** stereo or depth video of an ear under a known force at a known point, with point clouds fitted against the model, could calibrate and validate the model in living people, not only in cadavers.
- **Other loads:** one load case only. Folding the ear forward and pressing the tragus should be checked before choosing a model for good.

## 7. Packages

| Package | Version | Why | How installed | Scope |
|---|---|---|---|---|
| Python venv `middle_ear_newton/.venv` | Python 3.12.3, numpy 2.5.4, scipy 1.18.1, PyYAML | reference models, tests, race | already present | project |
| g++ | 13.3 | builds `soft_fast.cpp` | already present | system |
| Eigen (`libeigen3-dev`) | 3.4.0 | dense/sparse math, LDLT, CHOLMOD wrapper | already present | system |
| OpenMP (`libgomp`, part of gcc) | — | parallel assembly | already present | system |
| **SuiteSparse / CHOLMOD** | CHOLMOD 5.3.5 (GitHub `DrTimothyAldenDavis/SuiteSparse`, commit `0f41a77`, 2026-10-23) | supernodal sparse Cholesky | **built from source this session**, no sudo | user: `~/.local/suitesparse` |
| **OpenBLAS (`libopenblas-dev`)** | 0.3.26 | fast BLAS that CHOLMOD's supernodal factor needs | **`sudo apt install -y libopenblas-dev`** (you approved it in the terminal) | system |
| CUDA, METIS, Fortran | — | not used | — | — |

### How SuiteSparse was built
Only CHOLMOD and its ordering libraries, no GPU:
```
git clone --depth 1 https://github.com/DrTimothyAldenDavis/SuiteSparse.git ~/src/SuiteSparse
cd ~/src/SuiteSparse
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/suitesparse \
  -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;colamd;ccolamd;cholmod" \
  -DSUITESPARSE_USE_CUDA=OFF -DSUITESPARSE_USE_FORTRAN=OFF -DSUITESPARSE_DEMOS=OFF -DBUILD_TESTING=OFF \
  -DCMAKE_C_FLAGS="-O3 -march=native"
cmake --build build -j 16
cmake --install build
```
- Source stays in `~/src/SuiteSparse`; libraries and headers are in `~/.local/suitesparse/{lib,include/suitesparse}`.
- `fast_solver.build()` looks in `~/.local/suitesparse` first, then `/usr`. It links with an rpath, so no `LD_LIBRARY_PATH` is needed.
- Without SuiteSparse it builds Eigen-only (`libsoft_fast.so`); with it, it builds `libsoft_fast_cholmod.so`.

### Things to know about the OpenBLAS install
- apt switched the **system-wide** `libblas.so.3` / `liblapack.so.3` to OpenBLAS through `update-alternatives`. Anything on this machine that links the generic BLAS now gets OpenBLAS, which is usually faster.
- numpy/scipy in the venv ship their own BLAS and are unaffected. AMBF and CHAI3D use Eigen and don't link BLAS, as far as I know. Worth a quick drilling session to confirm nothing changed.
- To undo: `sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu` (and the `liblapack` one) to pick reference BLAS, or `sudo apt remove libopenblas-dev`.

### For the AMBF plugin later
Its CMake will need:
- `-DSF_CHOLMOD`
- `-I$HOME/.local/suitesparse/include/suitesparse`
- links to `-lcholmod -lsuitesparseconfig`, with an rpath to `$HOME/.local/suitesparse/lib`

Alternatively, `find_package(CHOLMOD)` with `CMAKE_PREFIX_PATH=$HOME/.local/suitesparse`, since SuiteSparse installs CMake config files.

---

## 8. Open items

1. **Material numbers:** literature values are in for the bulk tissues (§5). The **peak strengths of every interface** (which set how stringy each plane is), every shear energy, and perichondrium are still estimates. Cadaver measurements are needed for validation.
2. ~~Analytic element Hessians~~: done (2.4–3× overall with CHOLMOD/OpenBLAS). Next: **adaptive mesh refinement** at the working front, and **iterate-to-convergence** per frame with a cap (thin layers need it; §5). Factor reuse is optional.
3. **Mixed-mode attachment law** (one damage variable, mode-mixity criterion) for bone and cartilage attachments (§6).
4. **Contact between released layers** (§6).
5. **Bone as the floor:** skin must rest on the real voxel bone surface, not a flat plane.
6. **Auricle:** place `~/Downloads/ear.obj` on RT143 (Volume A in the study GUI) at the EAC opening; decimate a simulation copy; grow the skin and periosteum layers over the lateral mastoid.
7. ~~Layered stack and plane selection~~: done (§6). Temporalis muscle as a deformable volume is deferred.
8. **Plugin integration:** soft-tissue thread, thread cap, haptic coupling, and Instruments / Corridors in the existing pause menu.
9. **Haptic dropouts** (known issue, uninvestigated) should be understood before any haptic soft-tissue demo.
