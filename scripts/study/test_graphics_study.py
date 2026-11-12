"""Graphics study: order balance, result files, and the per-arm session options.

    python3 scripts/study/test_graphics_study.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graphics_study as gs  # noqa: E402
import study_session as ss  # noqa: E402
from test_study_session import FakeIO, advance, run  # noqa: E402


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_order_is_balanced_within_each_comparison():
    folder = pathlib.Path(tempfile.mkdtemp())
    rng = random.Random(3)
    for comparison in gs.COMPARISONS:
        for k in range(10):
            sid = gs.next_study_id(folder)
            gs.add_to_key(folder, sid, f"doe_jane{k}")
            arm1, arm2, how = gs.choose_order(folder, comparison, rng)
            assert {arm1, arm2} == set(gs.COMPARISONS[comparison])
            gs.record_assignment(folder, sid, comparison, arm1, arm2, how)
            # After every pair the two orders are used equally often.
            done = [r for r in rows(folder / gs.ASSIGNMENTS_FILE) if r["comparison"] == comparison]
            firsts = [r["arm1"] for r in done]
            assert abs(firsts.count("rtx") - (len(done) - firsts.count("rtx"))) <= 1
            if k % 2 == 1:
                assert how == "balanced"
    ids = [r["study_id"] for r in rows(folder / gs.KEY_FILE)]
    assert ids[0] == "G001" and len(set(ids)) == len(ids) == 20


def test_tlx_scores_and_files():
    folder = pathlib.Path(tempfile.mkdtemp())
    answers = {key: 50 for key, *_ in gs.TLX_SCALES}
    answers["frustration"] = 80
    assert gs.raw_tlx(answers) == 55.0
    score = gs.record_tlx(folder, "G001", "chai_vs_rtx", 1, "rtx", answers, 42.0)
    assert score == 55.0
    row = rows(folder / gs.TLX_FILE)[0]
    assert row["graphics"] == "rtx" and row["frustration"] == "80" and row["raw_tlx"] == "55.0"
    assert len(gs.TLX_SCALES) == 6 and gs.TLX_STEPS[0] == 0 and gs.TLX_STEPS[-1] == 100 and len(gs.TLX_STEPS) == 21


def test_preferences_name_the_graphics():
    folder = pathlib.Path(tempfile.mkdtemp())
    row = gs.record_preferences(folder, "G001", "chai_vs_rtx", ["chai", "rtx"], {"realism": "2", "training": "1"})
    assert row["realism_graphics"] == "rtx" and row["training_graphics"] == "chai"
    try:
        gs.record_preferences(folder, "G002", "chai_vs_rtx", ["chai", "rtx"], {"realism": "3", "training": "1"})
    except ValueError:
        pass
    else:
        raise AssertionError("bad answer accepted")
    assert len(rows(folder / gs.PREFERENCES_FILE)) == 1


def test_arm_without_tutorial_or_rtx_ends_on_incus():
    folder = pathlib.Path(tempfile.mkdtemp())
    io = FakeIO()
    rec = folder / "arm2_chai"
    session = ss.StudySession(io, "G001", rec, {"graphics_study": {"graphics": "chai"}}, trial_seconds=600,
                              tutorial=False, rtx_reload=False, text=gs.briefing(2))
    run(session, io, 0.2)
    assert session.phase == "ready"
    assert session.hud(io.t)["panel"]["title"] == "Session 2 of 2"
    advance(session, io)  # left pedal
    assert session.phase == "briefing"
    assert session.hud(io.t)["panel"]["title"] == "Session 2 of 2" and session.hud(io.t)["panel"]["cover"]
    run(session, io, 0.5)
    assert not io.reload_requested  # no RTX window to reload
    assert io.recorder == "on"
    advance(session, io)  # read the briefing
    assert session.phase == "armed"
    session.incus_identified()  # only counts once the trial is running
    assert session.phase == "armed"
    run(session, io, 0.2, removed_voxels=10)
    assert session.phase == "running"
    run(session, io, 95.0, removed_voxels=900)
    session.incus_identified()
    assert session.phase == "done" and io.recorder == "stopped" and io.locked
    meta = json.loads((rec / "metadata.json").read_text())
    assert meta["trial"]["ended_by"] == "incus_identified"
    assert 94.0 < meta["trial"]["incus_identified_drilling_s"] < 96.0
    assert meta["tutorial_run"] is False
    row = gs.record_trial(folder, "G001", "chai_vs_rtx", 2, "chai", session, rec)
    assert row["incus_identified"] == 1 and row["ended_by"] == "incus_identified"
    assert rows(folder / gs.TRIALS_FILE)[0]["tutorial_in_this_arm"] == "0"


def test_time_limit_is_recorded():
    io = FakeIO()
    session = ss.StudySession(io, "G002", pathlib.Path(tempfile.mkdtemp()) / "a", {}, trial_seconds=30,
                              text=gs.briefing(1, 30))
    assert session.phase == "tutorial"
    session.begin_trial()
    assert session.hud(io.t)["panel"]["title"] == "Session 1 of 2"
    run(session, io, 0.5)
    assert io.reload_requested  # RTX arm: waits for the RTX window's reload
    advance(session, io)
    run(session, io, 0.2, removed_voxels=5)
    run(session, io, 31.0)
    assert session.phase == "done" and session.ended_by == "time_up" and session.incus_drilling_s is None


if __name__ == "__main__":
    test_order_is_balanced_within_each_comparison()
    test_tlx_scores_and_files()
    test_preferences_name_the_graphics()
    test_arm_without_tutorial_or_rtx_ends_on_incus()
    test_time_limit_is_recorded()
    print("ok")
