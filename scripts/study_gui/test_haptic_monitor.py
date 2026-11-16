"""Haptic monitor window, its Advanced-tab toggle, device hand-over, and health log events.
No Touch needed: the device side is stand-in JSON files and a stand-in lock holder.

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_haptic_monitor.py
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "haptic_monitor"))

import yaml  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtCore import QProcess  # noqa: E402

import haptic_monitor as hm  # noqa: E402
import haptic_monitor_toggle as toggle  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def device(role="drill", servo=1000, force_output=True, buttons=0, encoders=(1, -659, -911, 2052, 2653, 2487)):
    return {
        "role": role, "model": "Touch", "vendor": "3D Systems Inc.", "serial": "23059000020",
        "driver": "Dev_Sep 19 2024", "max_force_n": 3.3, "max_continuous_force_n": 0.88, "valid": True,
        "encoders": list(encoders), "joint_deg": [0.0, 15.4, -21.2], "gimbal_deg": [0.1, 45.6, 4.0],
        "temperature": [0.003, 0.01, 0.016], "position_mm": [0, 0, 0], "velocity_mm_s": [0, 0, 0],
        "device_force_n": [0.0, 0.0, 1.2], "buttons": buttons, "inkwell": True, "safety_switch": True,
        "servo_hz": servo, "calibration": "OK", "force_output": force_output,
    }


def sim_state(**overrides):
    state = {
        "source": "sim", "wall_time": time.time(), "status": "ok", "error": "", "physics_hz": 998.0,
        "physics_gap_ms": 1.2, "paused": False, "contact_suppressed": False, "drill_in_contact": True,
        "drill_on": True, "safety_tripped": False, "safety_trips": 0, "suction_tracked": True,
        "drill_requested_n": [0.0, 0.0, 1.5], "drill_sent_n": [0.0, 0.0, 1.2], "suction_sent_n": [0.3, 0.0, 0.4],
        "devices": [device("drill"), device("suction")],
    }
    state.update(overrides)
    return state


def write(path: pathlib.Path, state: dict):
    path.write_text(json.dumps(state))


# ---------------------------------------------------------------- helpers

def test_encoder_range_and_health_levels():
    rng = hm.EncoderRange()
    rng.update([1, 2, 3, 4, 5, 6])
    rng.update([-1, 5, 3, 4, 9, 6])
    assert rng.rows()[0] == (-1, 1, 2) and rng.rows()[4] == (5, 9, 4)

    levels = {label: level for label, _, level in hm.health_summary(sim_state(), device(), "sim")}
    assert levels["Device"] == "ok" and levels["Servo"] == "ok" and levels["Force output"] == "ok"
    levels = {label: level for label, _, level in hm.health_summary(
        sim_state(safety_tripped=True), device(servo=600, force_output=False), "sim")}
    assert levels["Servo"] == "bad" and levels["Force output"] == "bad" and levels["Safety"] == "bad"
    # Direct mode never sends force: "off" is expected, not a fault.
    levels = {label: level for label, _, level in hm.health_summary(
        {"status": "ok", "wall_time": time.time()}, device(force_output=False), "direct")}
    assert levels["Force output"] == "info"
    assert hm.health_summary(None, None, "sim") == [("Device", "no data", "bad")]
    bad = hm.health_summary({"status": "no_device", "error": "could not open the Touch"}, None, "direct")
    assert bad == [("Device", "could not open the Touch", "bad")]


def test_release_waits_for_the_probe_to_let_go():
    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        assert hm.release_device_for_sim(0.2, folder)  # nobody holds it
        # Stand-in probe: holds the lock until the release file appears.
        holder = subprocess.Popen([sys.executable, "-c", f"""
import fcntl, os, pathlib, time
d = pathlib.Path({str(folder)!r})
(d / 'haptic_probe.release').unlink(missing_ok=True)
h = open(d / 'haptic_probe.lock', 'a+'); fcntl.flock(h, fcntl.LOCK_EX)
print('held', flush=True)
while not (d / 'haptic_probe.release').exists(): time.sleep(0.01)
time.sleep(0.1)
"""], stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "held"
        assert hm.device_locked(folder)
        start = time.monotonic()
        assert hm.release_device_for_sim(2.0, folder, settle_s=0.3)
        took = time.monotonic() - start
        assert 0.3 <= took < 1.5, took  # waits for the hand-over, plus the settle time
        holder.wait(2)
        # A holder that never lets go: False after the timeout.
        stuck = subprocess.Popen([sys.executable, "-c", f"""
import fcntl, time
h = open({str(folder / 'haptic_probe.lock')!r}, 'a+'); fcntl.flock(h, fcntl.LOCK_EX)
print('held', flush=True); time.sleep(5)
"""], stdout=subprocess.PIPE, text=True)
        assert stuck.stdout.readline().strip() == "held"
        assert not hm.release_device_for_sim(0.3, folder)
        stuck.kill()
        stuck.wait()
        assert hm.release_requested(time.time(), folder)


# ---------------------------------------------------------------- window

def make_window(folder: pathlib.Path, sim: list):
    log = folder / "haptic_health.log"
    log.write_text("2026-11-14 10:00:00.000 [sim] drill (SN 1): reading: servo 1000 Hz\n")
    window = hm.HapticMonitor(state_dir=folder, health_log=log, start_probe=False, sim_running=lambda: sim[0])
    return window, log


def test_window_shows_simulator_feed():
    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        sim = [True]
        window, log = make_window(folder, sim)
        assert (folder / hm.ALIVE_FILE).exists()  # tells the plugin to write at 20 Hz
        write(folder / hm.SIM_STATE, sim_state())
        window.tick()
        assert "Simulator running" in window.mode.text()
        assert window.tabs.count() == 2 and window.tabs.tabText(1) == "Suction Touch"
        drill = window.panels["drill"]
        assert drill.joints.item(0, 1).text() == "-659" and drill.joints.item(1, 1).text() == "15.4°"
        assert drill.buttons.item(0, 2).text() == "ON"  # inkwell
        assert "asked by contact <b>1.50 N</b>, sent <b>1.20 N</b>" in drill.forces.text()
        assert "frazier_suction" in window.panels["suction"].forces.text()
        assert drill.temp_bars[2].value() == 2
        assert "Servo" in window.health.text() and "1000 Hz" in window.health.text()
        assert "Physics" in window.health.text()

        # Buttons, min/max follow the next sample.
        state = sim_state()
        state["devices"][0] = device(buttons=3, encoders=(5, -600, -911, 2052, 2653, 2487))
        write(folder / hm.SIM_STATE, state)
        window.tick()
        assert drill.buttons.item(0, 0).text() == "ON" and drill.buttons.item(0, 1).text() == "ON"
        assert drill.minmax.item(0, 0).text() == "1" and drill.minmax.item(1, 0).text() == "5"
        assert drill.minmax.item(2, 1).text() == "59"
        assert len(drill.arm_graph.samples) == 2 and len(drill.force_graph.samples) == 2

        # A stale feed is not shown as live.
        write(folder / hm.SIM_STATE, sim_state(wall_time=time.time() - 10))
        window.tick()
        assert "no feed yet" in window.mode.text() and "no data" in window.health.text()

        # Health log tail, then new lines as they are appended.
        window.slow_tick()
        assert "reading: servo 1000 Hz" in window.log_view.toPlainText()
        with open(log, "a") as handle:
            handle.write("2026-11-14 10:00:01.000 [sim] drill (SN 1): FORCE OUTPUT OFF\n")
        window.slow_tick()
        assert window.log_view.toPlainText().count("\n") == 1 and "FORCE OUTPUT OFF" in window.log_view.toPlainText()
        window.close()
        assert not (folder / hm.ALIVE_FILE).exists()


def test_window_reads_direct_when_simulator_closed():
    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        sim = [False]
        window, _ = make_window(folder, sim)
        direct = {"source": "direct", "wall_time": time.time(), "status": "ok", "error": "",
                  "devices": [device(force_output=False)]}
        write(folder / hm.DIRECT_STATE, direct)
        window.tick()
        assert "Simulator closed" in window.mode.text()
        assert window.tabs.count() == 1
        assert "force output off" in window.panels["drill"].forces.text()
        assert "off (monitor sends none)" in window.health.text()
        # The simulator starts: the window switches to its feed (graphs restart).
        sim[0] = True
        write(folder / hm.SIM_STATE, sim_state())
        window.slow_tick()
        window.tick()
        assert "Simulator running" in window.mode.text()
        assert len(window.panels["drill"].arm_graph.samples) == 1
        # Device missing in direct mode: the error is shown, not hidden.
        sim[0] = False
        window.sim_seen_at = 0.0
        (folder / hm.SIM_STATE).unlink()
        write(folder / hm.DIRECT_STATE, {"source": "direct", "wall_time": time.time(), "status": "no_device",
                                         "error": "could not open the Touch: HD error 0x302", "devices": []})
        window.tick()
        assert "could not open the Touch" in window.health.text()
        window.close()


# ---------------------------------------------------------------- Advanced tab toggle

class FakeConfiguration:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.yaml_data = yaml.safe_load(path.read_text())

    def save_value(self, key, value):
        self.yaml_data[key] = value
        data = yaml.safe_load(self.path.read_text())
        data[key] = value
        self.path.write_text(yaml.safe_dump(data))


def wait_for(predicate, seconds=3.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_toggle_opens_saves_and_follows_the_window():
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "gui_setup.yaml"
        path.write_text("study_choice: resident\n")
        configuration = FakeConfiguration(path)
        messages = []
        stand_in = [sys.executable, "-c", "import time; time.sleep(30)"]
        parent = QtWidgets.QWidget()
        control = toggle.HapticMonitorToggle(configuration, messages.append, command=stand_in, parent=parent)
        box = control.build_group(parent)
        assert box.title() == "Haptic device" and not control.checkbox.isChecked()

        control.checkbox.setChecked(True)
        assert wait_for(control.is_open)
        assert yaml.safe_load(path.read_text())["haptic_monitor"] is True
        assert yaml.safe_load(path.read_text())["study_choice"] == "resident"

        # Closing the window by hand unticks the box and saves that.
        control.process.kill()
        assert wait_for(lambda: not control.checkbox.isChecked())
        assert yaml.safe_load(path.read_text())["haptic_monitor"] is False

        # Saved true reopens it at launcher start; quitting the launcher keeps the setting.
        configuration.save_value("haptic_monitor", True)
        again = toggle.HapticMonitorToggle(configuration, messages.append, command=stand_in, parent=parent)
        again.build_group(parent)
        assert again.checkbox.isChecked()
        again.restore()
        assert wait_for(again.is_open)
        again.shutdown()
        assert wait_for(lambda: again.process.state() == QProcess.NotRunning)
        assert yaml.safe_load(path.read_text())["haptic_monitor"] is True
        assert again.checkbox.isChecked()

        # Unticking closes it.
        control.checkbox.setChecked(True)
        assert wait_for(control.is_open)
        control.checkbox.setChecked(False)
        assert wait_for(lambda: not control.is_open())
        assert yaml.safe_load(path.read_text())["haptic_monitor"] is False


def test_launcher_frees_the_device_before_ambf_starts():
    source = (HERE / "study_gui.py").read_text()
    release = source.index("self.haptic_monitor.release_for_simulator()")
    start = source.index("self._ambf_process.start(ambf_bin, self._current_args)")
    assert release < start and source.count("self._ambf_process.start(") == 1
    assert "self.haptic_monitor.shutdown()" in source


# ---------------------------------------------------------------- C++ health events

def test_health_tracker_events():
    with tempfile.TemporaryDirectory() as folder:
        binary = pathlib.Path(folder) / "test_health_tracker"
        build = subprocess.run(["g++", "-std=c++17", "-O0", "-o", str(binary),
                                str(REPO / "scripts/haptic_monitor/test_health_tracker.cpp"), "-ldl"],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(binary), str(pathlib.Path(folder) / "health.log")], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr + run.stdout
        assert "health tracker ok" in run.stdout


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
