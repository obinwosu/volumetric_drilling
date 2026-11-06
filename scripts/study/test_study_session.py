"""Study session flow with a fake simulator.

    python3 scripts/study/test_study_session.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import study_session as ss  # noqa: E402


class FakeIO:
    def __init__(self):
        self.t = 1000.0
        self.state = {"px": 0.0, "py": 0.0, "pz": 0.0, "zoom_travel": 0.0, "recenter_count": 0,
                      "removed_voxels": 0, "paused": False, "drill": "4mm", "style": "cutting"}
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
    def request_restore(self): self.restore_requested = True
    def restore_done(self): return self.restore_requested
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


def tutorial(session, io):
    for k in range(10):  # drill_move
        run(session, io, 0.1, px=0.06 * k)
    assert session.progress.done["drill_move"]
    run(session, io, 1.3, cam_clutch=True); run(session, io, 0.1, cam_clutch=False)
    run(session, io, 0.1, zoom_travel=1.2)
    run(session, io, 1.3, device_clutch=True); run(session, io, 0.1, device_clutch=False)
    run(session, io, 0.1, recenter_count=1)
    run(session, io, 1.3, orbit=True); run(session, io, 0.1, orbit=False)
    run(session, io, 0.1, removed_voxels=500)
    assert session.phase == "tutorial" and session.hud(io.t)["panel"]["title"].endswith("Pause menu and burrs")
    run(session, io, 0.5, paused=True, drill="4mm"); run(session, io, 0.2, paused=False)
    assert not session.progress.done.get("pause")  # resumed without changing the burr
    run(session, io, 0.5, paused=True); run(session, io, 0.2, drill="2mm")
    run(session, io, 0.2, paused=False)


def test_full_session():
    with tempfile.TemporaryDirectory() as tmp:
        io = FakeIO()
        rec = pathlib.Path(tmp) / "S001"
        session = ss.StudySession(io, "S001", rec, {"volume_adf": "ADF/volume_256_RT143.yaml"}, trial_seconds=30)
        assert io.locked is False
        session.tick()
        assert io.hud["panel"]["title"].startswith("Tutorial 1 of 8")
        tutorial(session, io)
        assert session.phase == "ready", session.phase
        assert all(session.progress.done.get(s.key) for s in ss.TUTORIAL)

        session.begin_trial()
        assert io.locked is True and io.restore_requested
        run(session, io, 0.5)
        assert io.reload_requested and session.phase == "armed" and io.recorder == "on"
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


if __name__ == "__main__":
    test_reload_from_an_earlier_run_does_not_count_and_a_new_one_does()
    test_full_session()
    test_skip_and_reset_failure_returns_to_ready()
    test_one_minute_warning()
    print("ok")
