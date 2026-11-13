"""RTX view start-up progress: the headset card's status file and the study panels' log.

Replays an RTX view start-up log through the launcher (its window built offscreen).

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_rtx_loading.py [view.log]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = pathlib.Path(__file__).resolve().parent
os.chdir(HERE)  # gui_setup.yaml paths are relative to this folder
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "study"))
from PyQt5 import QtWidgets  # noqa: E402

QtWidgets.QApplication.exec_ = lambda self: 0  # build the launcher window, do not block
import study_gui  # noqa: E402

SAMPLE = """Initializing RTX View
Loading Newton
Waiting for AMBF volume_info before opening the RTX window
Opening RTX window
Loading intact volume
Closed occupancy
Meshing union of 12 structures
RTX view ready
Loading bone 25%
Loading bone 61%
Loading bone 100%
RTX bone ready
"""


def replay(lines):
    ui = study_gui.window
    tmp = pathlib.Path(tempfile.mkdtemp())
    study_gui.RTX_LOADING_PATH = tmp / "rtx_loading.json"
    said = []
    ui.print_info = said.append
    ui.study_mode = {"study_id": "G001", "graphics": "rtx"}
    ui._rtx_loading = {"heading": "Initializing RTX View", "label": "Starting", "percent": 0, "lines": []}
    ui._rtx_loading_logged = None
    ui._rtx_loading_card = False
    ui._rtx_loading_update()
    seen = []
    for line in lines:
        ui._rtx_loading_line(line)
        if study_gui.RTX_LOADING_PATH.exists():
            seen.append(json.loads(study_gui.RTX_LOADING_PATH.read_text()))
        if "RTX bone ready" in line:  # the splash closes on this line
            ui._rtx_loading_done("ready")
    return ui, seen, said


def test_progress_reaches_the_headset_file_and_the_log():
    ui, seen, said = replay(SAMPLE.splitlines())
    percents = [s["percent"] for s in seen]
    assert percents == sorted(percents), "progress never goes back"
    assert seen[-1]["label"] == "Ready" and seen[-1]["percent"] == 100
    assert seen[-1]["lines"][-1] == "RTX bone ready" and len(seen[-1]["lines"]) <= 6
    assert any(s["label"] == "Loading the bone" for s in seen)
    assert not study_gui.RTX_LOADING_PATH.exists(), "removed once RTX is ready"
    assert said[0] == "RTX view: Starting (0%)" and said[-1] == "RTX view: Ready (100%)"
    assert "RTX view: Opening RTX window (34%)" in said
    bone = [s for s in said if s.startswith("RTX view: Loading the bone")]
    assert 2 <= len(bone) <= 6, bone  # every 10%, not every line
    assert len(said) == len(set(said)), "each step logged once"


def test_simulator_closing_mid_load_clears_the_card():
    ui, _seen, _said = replay(SAMPLE.splitlines()[:5])
    assert study_gui.RTX_LOADING_PATH.exists()
    ui._rtx_loading_done("closed")
    assert not study_gui.RTX_LOADING_PATH.exists() and ui._rtx_loading is None


def test_real_log(path):
    lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    _ui, seen, said = replay(lines)
    print(f"{path}: {len(lines)} lines -> {len(said)} log entries")
    for entry in said:
        print("   ", entry)
    assert seen and said[-1] == "RTX view: Ready (100%)"


if __name__ == "__main__":
    test_progress_reaches_the_headset_file_and_the_log()
    test_simulator_closing_mid_load_clears_the_card()
    if len(sys.argv) > 1:
        test_real_log(sys.argv[1])
    print("ok")
