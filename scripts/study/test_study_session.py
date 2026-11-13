"""Study session flow with a fake simulator.

    python3 scripts/study/test_study_session.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import study_session as ss  # noqa: E402


class FakeIO:
    def __init__(self):
        self.t = 1000.0
        self.state = {"px": 0.0, "py": 0.0, "pz": 0.0, "zoom_in": 0.0, "zoom_out": 0.0, "recenter_count": 0,
                      "removed_voxels": 0, "paused": False, "drill": "4mm", "style": "cutting",
                      "advance_count": 0, "skip_count": 0, "hx": 0.0, "hy": 0.0, "hz": 0.0,
                      "cx": 0.0, "cy": 0.0, "cz": 1.0, "lx": 0.0, "ly": 0.0, "lz": -1.0,
                      "ux": 0.0, "uy": 1.0, "uz": 0.0, "fov": 0.2}
        self.hud = None
        self.locked = None
        self.restore_requested = self.reload_requested = False
        self.reload_wall = 0.0
        self.recorder = "off"
        self.recorder_dir = None

    def now(self): return self.t
    def read_state(self): return dict(self.state)
    def write_hud(self, hud): self.hud = json.loads(json.dumps(hud))
    def clear_hud(self): self.hud = None
    def set_drill_lock(self, locked): self.locked = locked
    def set_trial_active(self, active): self.trial_active = active
    def write_headset_place(self, seq, kind, tilt_deg=None): self.place, self.tilt = (seq, kind), tilt_deg
    def request_restore(self): self.restore_requested = True
    def restore_done(self): return self.restore_requested
    def request_burr_reset(self): self.state.update(drill="6mm", style="cutting")
    def request_reload(self):
        self.reload_requested = True
        self.reload_wall = self.t
    def restore_reload_wall(self): return self.reload_wall
    def start_recorder(self, d): self.recorder, self.recorder_dir = "on", d
    def recorder_running(self): return self.recorder == "on"
    def recorder_ready(self): return self.recorder == "on"
    def stop_recorder(self, timeout_s=60.0):
        self.recorder = "stopped"
        return 0


def run(session, io, seconds, hz=10, **state):
    io.state.update(state)
    for _ in range(int(seconds * hz)):
        io.t += 1.0 / hz
        session.tick()


def advance(session, io):
    io.state["advance_count"] += 1
    run(session, io, 0.2)


def tutorial(session, io):
    run(session, io, 0.2)
    assert session.hud(io.t)["panel"]["title"] == "Welcome"
    assert session.hud(io.t)["panel"]["advance"]  # nothing to do: advance straight away
    advance(session, io)
    advance(session, io)  # a tap before the step is done does nothing
    assert session.step == 1 and not session.progress.step_done
    for k in range(10):  # drill_move
        run(session, io, 0.1, px=0.06 * k)
    assert session.progress.step_done and session.step == 1  # done, still practising
    assert session.hud(io.t)["panel"]["done"]
    advance(session, io)
    run(session, io, 2.0, device_clutch=True)  # held without moving: not done
    assert not session.progress.step_done
    for k in range(5):
        run(session, io, 0.1, hx=0.005 * k)
    run(session, io, 0.1, device_clutch=False)
    assert session.progress.step_done
    advance(session, io)
    run(session, io, 0.1, recenter_count=1); advance(session, io)
    assert ss.TUTORIAL[session.step].key == "camera"
    run(session, io, 2.0, cam_clutch=True)  # clutch without moving the view: not done
    look = session.hud(io.t)["panel"]["look"]
    assert not session.progress.step_done
    target = np.array(look["target"]); here = np.array([0.0, 0.0, 1.0])
    d = (target - here) / np.linalg.norm(target - here)
    run(session, io, 0.5, lx=d[0], ly=d[1], lz=d[2])
    assert session.progress.step_done
    advance(session, io)
    run(session, io, 0.1, zoom_in=1.0)
    assert not session.progress.step_done  # in only
    run(session, io, 0.1, zoom_out=0.7); advance(session, io)
    run(session, io, 0.1, removed_voxels=500); advance(session, io)
    assert session.hud(io.t)["panel"]["title"].endswith("The pause menu")
    run(session, io, 0.5, paused=True)
    assert not session.progress.step_done
    run(session, io, 0.2, paused=False)
    assert session.progress.step_done
    advance(session, io)  # the last step: no burr change is asked for
    assert session.phase == "ready", session.phase


def test_full_session():
    with tempfile.TemporaryDirectory() as tmp:
        io = FakeIO()
        rec = pathlib.Path(tmp) / "S001"
        session = ss.StudySession(io, "S001", rec, {"volume_adf": "ADF/volume_256_RT143.yaml"}, trial_seconds=30)
        assert io.locked is False
        session.tick()
        assert io.hud["panel"]["title"] == "Welcome"
        tutorial(session, io)
        assert session.phase == "ready", session.phase
        assert all(session.progress.done.get(s.key) for s in ss.TUTORIAL)

        assert io.hud["panel"]["advance"].endswith("begin your study session")
        advance(session, io)  # the resident's left pedal
        assert session.phase == "briefing" and io.locked is True and io.restore_requested
        assert io.state["drill"] == "6mm" and io.hud["panel"]["cover"]
        assert io.hud["panel"]["footer"].startswith("Tap the left foot pedal")
        run(session, io, 0.5)
        assert io.reload_requested and io.recorder == "on" and session.phase == "briefing"
        advance(session, io)  # read the introduction
        assert session.phase == "armed"
        assert io.locked is False and io.recorder_dir == rec

        run(session, io, 5.0)  # idle, practice count still 500: the timer has not started
        assert session.phase == "armed" and session.remaining() == 30
        run(session, io, 0.1, removed_voxels=512)
        assert session.phase == "running"
        run(session, io, 10.0)
        run(session, io, 5.0, paused=True)  # pause time is not drilling time
        assert 19.0 < session.remaining() < 21.0
        run(session, io, 25.0, paused=False)
        assert session.phase == "done" and io.locked is True and io.recorder == "stopped"
        assert io.hud["banner"].startswith("Session complete")
        meta = json.loads((rec / "metadata.json").read_text())
        assert meta["study_id"] == "S001" and meta["volume_adf"].endswith("volume_256_RT143.yaml")
        assert abs(meta["trial"]["drilling_seconds"] - 30) < 0.2 and meta["trial"]["paused_seconds"] >= 4.9
        assert "participant_name" not in meta and "name" not in meta
        assert [e["event"] for e in meta["events"]].count("tutorial_step_done") == len(ss.TUTORIAL)


def test_skip_and_reset_failure_returns_to_ready():
    io = FakeIO()
    session = ss.StudySession(io, "S002", pathlib.Path(tempfile.mkdtemp()) / "S002", {}, trial_seconds=30)
    for _ in ss.TUTORIAL:
        session.skip_step()
    assert session.phase == "ready"
    io.restore_done = lambda: False
    session.begin_trial()
    run(session, io, ss.RESET_TIMEOUT_S + 1)
    assert session.phase == "ready" and session.error and io.locked is False
    run(session, io, 1.0)
    assert session.phase == "ready"  # no stray pedal count restarts it


def test_resident_pedal_waits_for_the_bone_and_proctor_can_skip():
    io = FakeIO()
    session = ss.StudySession(io, "S004", pathlib.Path(tempfile.mkdtemp()) / "S004", {}, trial_seconds=30)
    for _ in ss.TUTORIAL:
        session.skip_step()
    run(session, io, 0.2)
    io.recorder_ready = lambda: False
    advance(session, io); advance(session, io)  # begin, then "ready" at once
    assert session.phase == "briefing"
    assert session.hud(io.t)["panel"]["footer"].startswith("Thank you")
    io.recorder_ready = lambda: io.recorder == "on"
    run(session, io, 0.3)
    assert session.phase == "armed"
    io2 = FakeIO()
    s2 = ss.StudySession(io2, "S005", pathlib.Path(tempfile.mkdtemp()) / "S005", {}, trial_seconds=30)
    s2.begin_trial()
    run(s2, io2, 0.5)
    assert s2.phase == "briefing"
    s2.begin_trial()  # the proctor, again: do not wait for the pedal
    run(s2, io2, 0.2)
    assert s2.phase == "armed"


def test_one_minute_warning():
    io = FakeIO()
    session = ss.StudySession(io, "S003", pathlib.Path(tempfile.mkdtemp()) / "S003", {}, trial_seconds=70)
    session.phase, session.first_contact_wall = "running", io.t
    run(session, io, 11.0)
    assert session.hud(io.t)["panel"]["title"] == "1 minute remaining"
    run(session, io, 10.0)
    assert session.hud(io.t)["panel"] is None


def test_reload_from_an_earlier_run_does_not_count_and_a_new_one_does():
    # The rig hang: mesh_ready.json from an earlier launch said generation 1, the
    # new window's restore reload also said 1, and the trial never armed.
    runtime = pathlib.Path(tempfile.mkdtemp())
    io = ss.SessionIO(runtime=runtime)
    started = 2000.0
    (runtime / "mesh_ready.json").write_text(json.dumps({"generation": 1, "wall": started - 600, "reason": "restore"}))
    assert io.restore_reload_wall() < started
    (runtime / "mesh_ready.json").write_text(json.dumps({"generation": 1, "wall": started + 2, "reason": None}))
    assert io.restore_reload_wall() == 0.0  # a cut-driven reload is not the restore
    (runtime / "mesh_ready.json").write_text(json.dumps({"generation": 1, "wall": started + 2, "reason": "restore"}))
    assert io.restore_reload_wall() >= started



def test_right_arrow_skips_steps_and_begins():
    io = FakeIO()
    session = ss.StudySession(io, "S006", pathlib.Path(tempfile.mkdtemp()) / "S006", {}, trial_seconds=30)
    run(session, io, 0.2)
    for k in range(len(ss.TUTORIAL)):
        assert session.phase == "tutorial" and session.step == k
        run(session, io, 0.2, skip_count=k + 1)  # nothing done, still moves on
    assert session.phase == "ready" and not session.progress.done.get("drill")
    run(session, io, 0.2, paused=True, skip_count=20)
    assert session.phase == "ready"  # arrows belong to the pause menu while it is open
    run(session, io, 0.2, paused=False)
    run(session, io, 0.2, skip_count=21)
    assert session.phase == "briefing"
    run(session, io, 0.5, skip_count=22)
    assert session.phase == "armed"


def test_welcome_ignores_the_last_runs_tap_count():
    """The simulator restarts its tap count at 0; an old count must not block the welcome page."""
    io = FakeIO()
    io.state["advance_count"] = 37  # what the previous simulator left behind
    session = ss.StudySession(io, "S001", pathlib.Path(tempfile.mkdtemp()), {})
    run(session, io, 0.2)
    assert session.progress.step_done and session.step == 0
    io.state["advance_count"] = 0  # the new simulator starts writing
    run(session, io, 0.2)
    advance(session, io)
    assert session.step == 1


def test_state_file_from_an_earlier_simulator_is_ignored():
    import os
    import time
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "drill_state.json"
        path.write_text(json.dumps({"advance_count": 37}))
        old = time.time() - 600
        os.utime(path, (old, old))
        io = ss.SessionIO(runtime=folder)
        assert io.read_state() is None
        path.write_text(json.dumps({"advance_count": 0}))
        assert io.read_state() == {"advance_count": 0}


def armed_session(io, name, end_trial_count=0):
    io.state["end_trial_count"] = end_trial_count  # left over from practice or a restart
    session = ss.StudySession(io, name, pathlib.Path(tempfile.mkdtemp()) / name, {}, trial_seconds=30)
    for _ in ss.TUTORIAL:
        session.skip_step()
    session.begin_trial()
    run(session, io, 0.5)
    session.begin_trial()
    run(session, io, 0.2)
    assert session.phase == "armed" and io.trial_active is True
    return session


def test_resident_ends_the_trial_from_the_pause_menu():
    io = FakeIO()
    session = armed_session(io, "S006", end_trial_count=2)
    assert any("End trial" in line for line in session.hud(io.t)["panel"]["lines"])
    run(session, io, 1.0)  # the count from before arming does not end it
    assert session.phase == "armed"
    run(session, io, 0.1, removed_voxels=900)
    run(session, io, 5.0)
    assert session.phase == "running"
    run(session, io, 0.2, end_trial_count=3)
    assert session.phase == "done" and io.locked is True and io.recorder == "stopped"
    assert io.trial_active is False
    meta = json.loads((session.recording_dir / "metadata.json").read_text())
    assert meta["trial"]["ended_early"] and meta["trial"]["ended_by"] == "resident"
    assert "End trial" in " ".join(ss.BRIEFING_LINES)


def test_end_trial_before_first_bone_contact_and_after_a_simulator_restart():
    io = FakeIO()
    session = armed_session(io, "S007", end_trial_count=4)
    run(session, io, 0.2, end_trial_count=0)  # simulator restarted: count back to zero
    assert session.phase == "armed"
    run(session, io, 0.2, end_trial_count=1)
    assert session.phase == "done"
    meta = json.loads((session.recording_dir / "metadata.json").read_text())
    assert meta["trial"]["ended_by"] == "resident"


def test_proctor_end_is_recorded_as_proctor():
    io = FakeIO()
    session = armed_session(io, "S008")
    session.end_early()
    meta = json.loads((session.recording_dir / "metadata.json").read_text())
    assert meta["trial"]["ended_by"] == "proctor" and io.trial_active is False


def test_headset_view_is_placed_at_the_welcome_page_and_the_introduction():
    io = FakeIO()
    session = ss.StudySession(io, "S010", pathlib.Path(tempfile.mkdtemp()) / "S010", {}, trial_seconds=30)
    run(session, io, 0.2)
    assert io.place == (0, "look")  # welcome page
    advance(session, io)
    run(session, io, 3.0)
    assert io.place == (0, "look")  # the tutorial keeps the same placement
    session.recenter_headset()
    run(session, io, 0.1)
    assert io.place == (1, "look")
    for _ in ss.TUTORIAL:
        session.skip_step()
    session.begin_trial()
    run(session, io, 0.2)
    assert session.phase == "briefing" and io.place == (1, "look")  # not moved on its own
    assert io.hud["panel"]["compact"]
    session.begin_trial()
    run(session, io, 0.5)
    assert session.phase == "armed" and io.place == (1, "look")
    assert io.tilt is None
    session.set_headset_tilt(22)
    run(session, io, 0.1)
    assert io.place == (2, "look") and io.tilt == 22.0
    session.set_headset_tilt(90)
    assert session.headset_tilt == 60.0

if __name__ == "__main__":
    test_welcome_ignores_the_last_runs_tap_count()
    test_state_file_from_an_earlier_simulator_is_ignored()
    test_reload_from_an_earlier_run_does_not_count_and_a_new_one_does()
    test_full_session()
    test_skip_and_reset_failure_returns_to_ready()
    test_one_minute_warning()
    test_resident_pedal_waits_for_the_bone_and_proctor_can_skip()
    test_right_arrow_skips_steps_and_begins()
    test_resident_ends_the_trial_from_the_pause_menu()
    test_end_trial_before_first_bone_contact_and_after_a_simulator_restart()
    test_proctor_end_is_recorded_as_proctor()
    test_headset_view_is_placed_at_the_welcome_page_and_the_introduction()
    print("ok")
