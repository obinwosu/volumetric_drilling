"""Every bench fixture on every material in tissue_params, slow and fast.

quick (default): the fast solver against theory, every fixture; cut and stretch
                 also against the slow reference. About 2 minutes.
full:            adds the slow reference for peel and wedge, and fast-vs-slow
                 agreement. Tens of minutes.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_materials.py [quick|full]
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import material_bench as mb  # noqa: E402
import tissue_params as tp  # noqa: E402


def check_peel(name, r):
    # Converged frames peel at the general Kendall force (arm with its own J-curve).
    assert abs(r["fast"]["force_ratio"] - 1.0) < 0.05, (name, r["fast"])
    assert r["fast"]["steady_rows"] >= 20, (name, r["fast"])
    if "slow" in r:
        assert abs(r["slow"]["force_ratio"] - 1.0) < 0.06, (name, r["slow"])
        assert abs(r["fast_over_slow"] - 1.0) < 0.05, (name, r)


def check_wedge(name, r):
    # A sharp elevator frees tissue at its edge: steady push = cutting cost x width.
    assert abs(r["fast"]["steady_push_n"] / r["edge_cost_push_n"] - 1.0) < 0.05, (name, r["fast"])
    if "slow" in r:
        assert 0.98 <= r["slow"]["energy_balance_ratio"] < 1.08, (name, r["slow"])
        assert abs(r["slow"]["steady_push_n"] / r["edge_cost_push_n"] - 1.0) < 0.08, (name, r["slow"])
        assert abs(r["fast_over_slow"] - 1.0) < 0.05, (name, r)


def check_cut(name, r):
    # Release lies between Griffith for the stiffened and the soft modulus, each
    # times the mesh's own crack-tip undershoot (measured without stiffening).
    lo, hi = r["mesh_factor"] * r["stiffened_over_soft"], r["mesh_factor"]
    assert lo * 0.98 <= r["slow"]["ratio_to_soft"] <= hi * 1.02, (name, r)
    assert r["slow"]["min_opening_mm"] > 0.0
    assert abs(r["fast_over_slow"] - 1.0) < 1e-6, (name, r)


def check_stretch(name, r):
    for label in ("slow", "fast"):
        s = r[label]
        assert s["max_force_error_before_break"] < 1e-5, (name, label, s)
        assert s["separated"] and abs(s["force_after_n"]) < 1e-9, (name, label, s)
        # Snaps at the weak row (1% weaker), within one stretch step.
        weak = (1.0 + 2.0 * 0.99 * (r["fail_stretch"] ** 2 - 1.0) / 2.0) ** 0.5
        assert weak - 1e-9 <= s["broke_at_stretch"] <= weak + r["step"] + 1e-9, (name, label, s)
        assert abs(s["small_strain_stiffness_n"] / s["exact_small_strain_stiffness_n"] - 1.0) < 0.05, (name, label, s)
    assert abs(r["fast"]["broke_at_stretch"] - r["slow"]["broke_at_stretch"]) < 1e-6, (name, r)


CHECKS = {"peel": check_peel, "wedge": check_wedge, "cut": check_cut, "stretch": check_stretch}


def test_every_material(tier="quick"):
    results = {}
    for name in tp.MATERIALS:
        results[name] = mb.run_material(name, tier=tier, verbose=True)
        for f, check in CHECKS.items():
            check(name, results[name][f])
    mb.OUT.mkdir(parents=True, exist_ok=True)
    (mb.OUT / f"all_{tier}.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    return results


def test_regime_flags():
    """What the table predicts, in words a reviewer can check against experience."""
    reg = {n: tp.regime(n) for n in tp.MATERIALS}
    for n, r in reg.items():
        assert r.fails_first == "interface", (n, r)  # peeling, not tearing, at nominal values
    # Thin, stiff layers need a much finer mesh at their peel front than skin.
    assert reg["periosteum"].mesh_mm < 0.1 and reg["bony_canal_skin"].mesh_mm < 0.05
    assert reg["skin"].mesh_mm >= 0.25
    # Perichondrium is weaker than periosteum, and closer to tearing when peeled.
    assert reg["perichondrium"].kendall_n_per_mm < reg["periosteum"].kendall_n_per_mm
    assert (reg["perichondrium"].kendall_n_per_mm / reg["perichondrium"].strength_n_per_mm
            > reg["periosteum"].kendall_n_per_mm / reg["periosteum"].strength_n_per_mm)


if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "quick"
    test_regime_flags()
    test_every_material(tier)
    print(f"material tests passed ({tier})")
