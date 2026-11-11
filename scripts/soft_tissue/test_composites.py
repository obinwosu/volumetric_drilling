"""Composite tissue: plane selection, layer selectivity, elevation, incision, slipperiness.

quick (default): fast solver, about a minute. full: also the slow reference, and
fast-vs-slow agreement.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_composites.py [quick|full]
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import composite_bench as cb  # noqa: E402

SELECTIVE = 0.02  # a layer that should stay put moves less than 2% of the grip travel


def check(results):
    for solver, r in results["mastoid_plane_selection"].items():
        # Grip the skin: skin and TPF come up together off the periosteum (areolar plane).
        assert abs(r["force_ratio"] - 1.0) < 0.05, (solver, r)
        assert r["planes_broken_in_peeled_region"][1] > 0.95 and r["planes_broken_in_peeled_region"][0] < 0.05, r
        assert r["attachment_broken_in_peeled_region"] == 0.0, r
        assert r["layer_rise_mm"][2] < SELECTIVE * r["lift_mm"], r
        assert r["layer_rise_mm"][1] > 0.9 * r["layer_rise_mm"][0], r  # TPF rides with the skin

    for solver, r in results["temporal_plane_selection"].items():
        assert abs(r["force_ratio"] - 1.0) < 0.05, (solver, r)
        assert r["planes_broken_in_peeled_region"][1] > 0.95 and r["attachment_broken_in_peeled_region"] == 0.0, r
        assert r["layer_rise_mm"][2] < SELECTIVE * r["lift_mm"], r  # temporalis fascia stays down

    for solver, r in results["mastoid_full_thickness"].items():
        # All three come off bone together; energy per area between G_I and G_I + G_II.
        assert r["attachment_broken_in_peeled_region"] > 0.95 and max(r["planes_broken_in_peeled_region"]) < 0.05, r
        assert r["force_ratio"] > 0.95 and r["upper_ratio"] < 1.0, (solver, r)

    for case in ("mastoid_elevator", "bony_canal_elevator"):
        for solver, r in results[case].items():
            # A sharp elevator under the deepest layer lifts the whole flap, layers intact.
            assert abs(r["push_over_edge_cost"] - 1.0) < 0.05, (case, solver, r)
            assert r["attachment_broken_passed"] > 0.95 and max(r["planes_broken_passed"]) == 0.0, (case, r)
            assert min(r["layer_rise_mm"]) > 0.0, (case, r)

    for solver, r in results["mastoid_incision"].items():
        # Cutting skin and TPF over intact periosteum: the hold weakens, but the periosteum
        # still bridges the cut through the areolar plane.
        assert 0.0 < r["skin_tpf_cut_drop"] < r["skin_tpf_cut_unbonded_drop"] < r["full_thickness_cut_drop"], r

    for solver, r in results["grasp_periosteum"].items():
        # Grasp only the periosteum (skin flap already elevated): nothing else moves.
        assert r["periosteum_rise_mm"] > 0.99 * r["lift_mm"], r
        assert max(r["skin_rise_mm"], r["tpf_rise_mm"]) < SELECTIVE * r["lift_mm"], r
        assert r["grasp_force_n"] > 0.0

    for case in ("auricle_lap_shear", "mastoid_lap_shear"):
        for solver, r in results[case].items():
            assert abs(r["force_ratio"] - 1.0) < 0.05, (case, solver, r)
    for solver in results["auricle_lap_shear"]:
        # Perichondrium slides off cartilage more easily than periosteum off bone.
        a = results["auricle_lap_shear"][solver]["steady_force_n"] / results["auricle_lap_shear"][solver]["expected_force_n"]
        assert a > 0  # (ratio checked above)
        auricle_pw = results["auricle_lap_shear"][solver]["expected_force_n"]
        mastoid_pw = results["mastoid_lap_shear"][solver]["expected_force_n"]
        assert auricle_pw < mastoid_pw, (auricle_pw, mastoid_pw)

    for solver, r in results["auricle_plane_selection"].items():
        assert abs(r["force_ratio"] - 1.0) < 0.05, (solver, r)
        assert r["weaker"] == "skin_perichondrium"
        assert r["planes_broken_in_peeled_region"][0] > 0.95 and r["attachment_broken_in_peeled_region"] == 0.0, r

    # Fast and slow agree wherever both ran.
    for case, by_solver in results.items():
        if "slow" not in by_solver:
            continue
        f, s = by_solver["fast"], by_solver["slow"]
        for key in ("steady_force_n", "steady_push_n", "grasp_force_n"):
            if key in f:
                assert abs(f[key] / s[key] - 1.0) < 0.06, (case, key, f[key], s[key])
        for key in ("skin_tpf_cut_drop", "skin_tpf_cut_unbonded_drop", "full_thickness_cut_drop"):
            if key in f:
                assert abs(f[key] - s[key]) < 1e-4, (case, key, f[key], s[key])


def test_composites(tier="quick"):
    results = cb.run(tier=tier)
    cb.OUT.mkdir(parents=True, exist_ok=True)
    (cb.OUT / f"composites_{tier}.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    check(results)
    return results


def check_files(tier="full"):
    """Check results from runs split across processes (composite_bench.py --name <case>)."""
    results = {}
    for f in sorted(cb.OUT.glob(f"composites_{tier}_*.json")):
        results.update(json.loads(f.read_text()))
    missing = set(cb.CASES) - set(results)
    assert not missing, missing
    (cb.OUT / f"composites_{tier}.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    check(results)
    return results


if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if len(sys.argv) > 2 and sys.argv[2] == "files":
        check_files(tier)
        print(f"composite tests passed ({tier}, from split runs)")
        sys.exit(0)
    test_composites(tier)
    print(f"composite tests passed ({tier})")
