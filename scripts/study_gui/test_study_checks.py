"""Launch checks on the Study tab: headset connection and live stall detection.

Run: middle_ear_newton/.venv/bin/python scripts/study_gui/test_study_checks.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import study_checks as sc  # noqa: E402

READY_VIEW_LOG = "Initializing RTX View\nRTX view ready\nLoading bone 100%\nRTX bone ready\n"
HEADSET_READY = "INFO! OpenXR eyepiece session created (2476x2594 per eye)\nINFO! OpenXR eyepiece session READY / begun\n"


def _runtime(tmp: pathlib.Path):
    sc.RUNTIME = tmp
    sc.EYES = tmp / "eyes"
    sc.PROBLEM_LOG = tmp / "launch_problems.log"


def _write(path: pathlib.Path, text: str = "", age: float = 0.0):
    path.write_text(text)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


def _rows(started_ago: float, headset=True, rtx_running=True):
    rows = sc.launch_status("running", rtx_running, time.time() - started_ago, headset=headset)
    return {row.name: row for row in rows}


def test_headset_connected_follows_the_last_wivrn_event():
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "server_logs_x.txt"
        log.write_text(
            "[2026-11-08T21:33:31.357] Client connected\n"
            "[2026-11-08T21:39:17.325]  INFO [client_connected] Client 1 connected\n")
        assert sc.headset_connected(log) is True
        log.write_text(log.read_text() + "[2026-11-08T21:38:53.299]  WARN [run_net] Network exception: Socket shutdown\n")
        assert sc.headset_connected(log) is False
        log.write_text(log.read_text() + "[2026-11-08T21:39:03.032] Client connected\n")
        assert sc.headset_connected(log) is True
        # An app connecting is not the headset.
        log.write_text("[x] Server exited, exit status 0\n[y]  INFO [client_connected] Client 2 connected\n")
        assert sc.headset_connected(log) is False


def test_simulator_that_never_draws_says_to_show_its_window():
    """AMBF up 30 s with no graphics heartbeat: the 2026-11-08 hang (window minimized too early)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", "INFO! OpenXR eyepiece will create the WiVRn session on the first graphics frame\n")
        _write(tmp / "status.json", age=600)  # left by an earlier launch
        _write(tmp / "view.log", READY_VIEW_LOG)
        _write(tmp / "summary.json")
        rows = _rows(30)
        assert rows["Simulator (AMBF)"].state == sc.FAIL
        assert "has not drawn a frame in 30 s" in rows["Simulator (AMBF)"].detail
        assert "dock" in rows["Simulator (AMBF)"].detail
        assert rows["Headset link (WiVRn)"].state == sc.WAIT  # not WiVRn's fault
        headline = sc.waiting_on(list(rows.values()))
        assert headline.startswith("Problem: Simulator (AMBF)"), headline


def test_slow_wivrn_while_ambf_draws_says_to_restart_wivrn():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", "INFO! OpenXR eyepiece connecting to WiVRn (AMBF keeps running meanwhile)\n")
        _write(tmp / "status.json")
        _write(tmp / "view.log", READY_VIEW_LOG)
        _write(tmp / "summary.json")
        slow = _rows(30)
        assert slow["Simulator (AMBF)"].state == sc.OK
        assert slow["Headset link (WiVRn)"].state == sc.WAIT
        assert "slow" in slow["Headset link (WiVRn)"].detail
        rows = _rows(50)
        assert rows["Headset link (WiVRn)"].state == sc.FAIL
        assert "quit WiVRn" in rows["Headset link (WiVRn)"].detail
        assert sc.waiting_on(list(rows.values())).startswith("Problem: Headset link")


def test_early_launch_is_waiting_not_failing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", "")
        rows = _rows(5)
        assert rows["Headset link (WiVRn)"].state == sc.WAIT
        assert rows["Simulator (AMBF)"].state == sc.WAIT


def test_healthy_launch_is_all_ok():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", HEADSET_READY)
        _write(tmp / "status.json")
        _write(tmp / "view.log", READY_VIEW_LOG)
        _write(tmp / "summary.json")
        _write(tmp / "drill_state.json")
        _write(tmp / "eyes")
        rows = _rows(60)
        assert all(row.state == sc.OK for row in rows.values()), {n: (r.state, r.detail) for n, r in rows.items()}
        assert sc.waiting_on(list(rows.values())) == "Everything is ready."


def test_rtx_view_that_stops_writing_is_a_problem():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", HEADSET_READY)
        _write(tmp / "status.json")
        _write(tmp / "view.log", READY_VIEW_LOG)
        _write(tmp / "summary.json", age=12)
        rows = _rows(60)
        assert rows["RTX view"].state == sc.FAIL
        assert "stopped responding 12 s ago" in rows["RTX view"].detail


def test_closed_rtx_window_is_named():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", HEADSET_READY)
        _write(tmp / "status.json")
        _write(tmp / "view.log", READY_VIEW_LOG + "RTX window closed; the RTX view is exiting\n")
        _write(tmp / "summary.json")
        rows = _rows(60)
        assert rows["RTX view"].state == sc.FAIL
        assert "window was closed" in rows["RTX view"].detail


def test_simulator_that_stops_drawing_is_a_problem():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        _write(tmp / "ambf.log", HEADSET_READY)
        _write(tmp / "status.json", age=9)
        _write(tmp / "view.log", READY_VIEW_LOG)
        _write(tmp / "summary.json")
        rows = _rows(60)
        assert rows["Simulator (AMBF)"].state == sc.FAIL
        assert "stopped drawing 9 s ago" in rows["Simulator (AMBF)"].detail
        desktop = _rows(60, headset=False)
        assert "Headset link (WiVRn)" not in desktop
        assert "Close Simulation" in desktop["Simulator (AMBF)"].detail


def test_problem_log_records_thread_waits():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _runtime(tmp)
        sc.log_problem("Headset link (WiVRn): test", [("this test", os.getpid())])
        text = sc.PROBLEM_LOG.read_text()
        assert "Headset link (WiVRn): test" in text
        assert f"this test (pid {os.getpid()}) threads:" in text
        assert len(text.splitlines()) >= 3


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    sys.exit(1 if failures else 0)
