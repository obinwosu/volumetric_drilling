"""Graphics study tab, end to end with a stand-in simulator (no AMBF, RTX or recorder).

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_graphics_study_mode.py
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import pathlib
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "study"))

from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtCore import QProcess  # noqa: E402

import graphics_study as gs  # noqa: E402
import graphics_study_mode as gsm  # noqa: E402
import study_rehearsal as sr  # noqa: E402
import study_session  # noqa: E402

sr.RESTORE_S = sr.RELOAD_S = sr.RECORDER_S = 0.0
APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
QtWidgets.QMessageBox.exec_ = lambda self: 0
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)


def _warning(_parent, title, text, *rest):
    raise AssertionError(f"warning shown: {title}: {text}")


QtWidgets.QMessageBox.warning = staticmethod(_warning)
# Cycled: each test that runs both sessions answers the questionnaire twice.
TLX_SCORES = itertools.cycle([[10, 20, 30, 40, 50, 60], [70, 70, 70, 70, 70, 70]])


def fill_tlx(dialog):
    scores = next(TLX_SCORES)
    dialog._groups["mental"].button(scores[0]).click()
    dialog.submit.click()  # too early: not saved, the five missing scales are flagged
    assert dialog.result() != QtWidgets.QDialog.Accepted
    assert dialog.missing() == [key for key, *_ in gs.TLX_SCALES][1:]
    assert dialog._boxes["physical"].styleSheet() and not dialog._boxes["mental"].styleSheet()
    assert "Physical Demand" in dialog.remaining.text()
    for (key, *_), value in zip(gs.TLX_SCALES, scores):
        dialog._groups[key].button(value).click()
    assert not dialog.missing() and not dialog._boxes["physical"].styleSheet()
    dialog.submit.click()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    return QtWidgets.QDialog.Accepted


gsm.TlxDialog.exec_ = fill_tlx


def fill_preferences(dialog):
    assert not dialog.submit.isEnabled()
    dialog._groups["realism"]["2"].click()
    assert not dialog.submit.isEnabled()
    dialog._groups["training"]["none"].click()
    assert dialog.submit.isEnabled()
    return QtWidgets.QDialog.Accepted


gsm.PreferenceDialog.exec_ = fill_preferences


def fill_background(dialog):
    assert not dialog.submit.isEnabled()
    buttons = dialog._buttons
    buttons["role"]["fellow"].click()
    shown = [y for y, b in buttons["pgy"].items() if b.isVisibleTo(dialog)]
    assert shown == [6, 7], shown  # fellows: PGY 6-7 only
    buttons["role"]["attending"].click()
    assert dialog.years.isVisibleTo(dialog) and not buttons["pgy"][1].isVisibleTo(dialog)
    buttons["role"]["resident"].click()
    assert [y for y, b in buttons["pgy"].items() if b.isVisibleTo(dialog)] == [1, 2, 3, 4, 5]
    buttons["pgy"][3].click()
    buttons["cortical_mastoidectomies"]["6_10"].click()
    for key in ("vr_experience", "vr_simulation_experience"):
        buttons[key]["minimal"].click()
    assert not dialog.submit.isEnabled()  # one experience question left
    buttons["drilling_simulator_experience"]["none"].click()
    assert dialog.submit.isEnabled()
    dialog.submit.click()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    return QtWidgets.QDialog.Accepted


gsm.BackgroundDialog.exec_ = fill_background


def fill_reflection(dialog):
    assert not dialog.submit.isEnabled()  # practice interest is required, comments are not
    dialog._buttons["yes"].click()
    assert dialog.submit.isEnabled()
    dialog.comments.setPlainText("Felt real,\nbut the light was dim.")
    return QtWidgets.QDialog.Accepted


gsm.ReflectionDialog.exec_ = fill_reflection


def pump(panel, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        APP.processEvents()
        panel._tick()
        time.sleep(0.02)


class FakeProcess:
    def __init__(self):
        self.running = False

    def state(self):
        return QProcess.Running if self.running else QProcess.NotRunning


class FakeSetting:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeUi:
    """Just what the tab touches on the launcher."""

    def __init__(self, base):
        self.study_mode = None
        self.study_panel = None
        self.launches = []
        self._ambf_process = FakeProcess()
        self._rtx_process = FakeProcess()
        self.gui_configuration = type("C", (), {"recording_base_path": FakeSetting(base),
                                                "footpedal_device": FakeSetting("")})()
        self._volume_button = button = QtWidgets.QRadioButton()  # kept: the group does not own it
        button.volume_adf = "../../ADF/volume_256_RT143.yaml"
        button.volume_name = "Anatomy A"
        self.anatomy_button_group = QtWidgets.QButtonGroup()
        self.anatomy_button_group.addButton(button)
        self.check_rtx_view = QtWidgets.QCheckBox()
        self.check_openxr_eyepiece = QtWidgets.QCheckBox()
        self.check_openvr_hmd = self.check_video_record = None

    def pressed_start_simulation(self):
        self.launches.append((dict(self.study_mode), self.check_rtx_view.isChecked(),
                              self.check_openxr_eyepiece.isChecked()))
        self._ambf_process.running = True

    def close_simulation(self):
        self._ambf_process.running = False

    def get_starting_burr_mm(self):
        return 6

    def print_info(self, text):
        pass


def drive_arm(panel, io, tutorial):
    if tutorial:
        assert panel.session.phase == "tutorial"
        panel.begin_trial()  # the proctor skips the tutorial here
    else:
        pump(panel, 0.2)
        assert panel.session.phase == "ready"
        assert io.hud["panel"]["title"] == "Session 2 of 2"
        io.state["advance_count"] += 1  # left pedal
        pump(panel, 0.2)
    assert panel.session.phase == "briefing", panel.session.phase
    pump(panel, 0.3)
    io.state["advance_count"] += 1  # read the briefing
    pump(panel, 0.3)
    assert panel.session.phase == "armed", panel.session.phase
    io.state["removed_voxels"] += 50
    pump(panel, 0.3)
    assert panel.session.phase == "running"
    assert panel.incus_button.isEnabled()
    panel.incus_button.click()
    pump(panel, 0.1)


def test_full_study_with_files():
    base = tempfile.mkdtemp()
    ui = FakeUi(base)
    io = sr.RehearsalIO()
    study_session.SessionIO = lambda recorder_cmd=None: io  # the tab builds one per arm
    panel = gsm.GraphicsStudyPanel(ui)
    panel.comparison.setCurrentIndex(panel.comparison.findData("upstream_vs_rtx"))
    panel.name_edit.setText("doe_jane")
    assert panel.start_button.isEnabled()
    panel.start_button.click()
    order = panel.plan["order"]
    assert sorted(order) == ["rtx", "upstream"]
    mode, rtx_box, headset_box = ui.launches[0]
    assert mode == {"study_id": "G001", "graphics": order[0]} and rtx_box == (order[0] == "rtx") and headset_box
    drive_arm(panel, io, tutorial=True)
    assert panel.stage == "between", panel.stage  # TLX filled by fill_tlx
    assert not ui._ambf_process.running  # simulator closed for the questionnaire
    assert panel.next_button.isEnabled()
    panel.next_button.click()
    assert ui.launches[1][0]["graphics"] == order[1]
    drive_arm(panel, io, tutorial=False)
    assert panel.stage == "complete"
    folder = pathlib.Path(base) / "graphics_study"
    with open(folder / gs.TLX_FILE, newline="") as handle:
        tlx = list(csv.DictReader(handle))
    assert [r["graphics"] for r in tlx] == order and [r["raw_tlx"] for r in tlx] == ["35.0", "70.0"]
    with open(folder / gs.TRIALS_FILE, newline="") as handle:
        trials = list(csv.DictReader(handle))
    assert [r["ended_by"] for r in trials] == ["incus_identified"] * 2
    assert [r["tutorial_in_this_arm"] for r in trials] == ["1", "0"]
    meta = json.loads((pathlib.Path(trials[1]["recording_dir"]) / "metadata.json").read_text())
    assert meta["graphics_study"]["graphics"] == order[1] and meta["graphics_study"]["arm"] == 2
    with open(folder / gs.PREFERENCES_FILE, newline="") as handle:
        pref = list(csv.DictReader(handle))[0]
    assert pref["realism_session"] == "2" and pref["realism_graphics"] == order[1]
    assert pref["training_session"] == "none" and pref["training_graphics"] == "none"
    with open(folder / gs.BACKGROUND_FILE, newline="") as handle:
        background = list(csv.DictReader(handle))
    assert len(background) == 1 and background[0]["study_id"] == "G001"
    assert (background[0]["role"], background[0]["training_stage_unit"], background[0]["training_stage_value"]) \
        == ("resident", "pgy", "3")
    assert background[0]["drilling_simulator_experience"] == "none"
    assert background[0]["survey_version"] == gs.resident_surveys().SURVEY_VERSION
    with open(folder / gs.REFLECTION_FILE, newline="") as handle:
        reflection = list(csv.DictReader(handle))
    assert [(r["practice_interest"], r["comments"]) for r in reflection] == [("yes", "Felt real,\nbut the light was dim.")]
    with open(folder / gs.KEY_FILE, newline="") as handle:
        assert list(csv.DictReader(handle))[0]["name"] == "doe_jane"
    with open(folder / gs.ASSIGNMENTS_FILE, newline="") as handle:
        row = list(csv.DictReader(handle))[0]
    assert [row["arm1"], row["arm2"]] == order


def test_questionnaire_reopens_after_closing_without_saving():
    panel = gsm.GraphicsStudyPanel(FakeUi(tempfile.mkdtemp()), rehearsal_only=True)
    panel.start_button.click()
    drive_arm(panel, panel._io, tutorial=True)
    panel.next_button.click()
    original = gsm.BackgroundDialog.exec_
    gsm.BackgroundDialog.exec_ = lambda dialog: QtWidgets.QDialog.Rejected
    try:
        drive_arm(panel, panel._io, tutorial=False)
    finally:
        gsm.BackgroundDialog.exec_ = original
    assert panel.stage == "background" and panel.tlx_button.isEnabled()
    panel.tlx_button.click()  # Open questionnaire
    assert panel.stage == "complete"


def test_headset_practice_run_is_not_a_participant():
    base = tempfile.mkdtemp()
    ui = FakeUi(base)
    io = sr.RehearsalIO()
    study_session.SessionIO = lambda recorder_cmd=None: io
    panel = gsm.GraphicsStudyPanel(ui)
    assert not panel.start_button.isEnabled()  # no name, not a practice run
    panel.practice_box.setChecked(True)
    assert panel.start_button.isEnabled()  # the name is optional
    panel.start_button.click()
    assert ui.launches[0][2], "a practice run uses the headset"
    drive_arm(panel, io, tutorial=True)
    panel.next_button.click()
    drive_arm(panel, io, tutorial=False)
    assert panel.stage == "complete"
    dev = pathlib.Path(base) / "graphics_study_dev"
    assert not (pathlib.Path(base) / "graphics_study").exists()
    with open(dev / gs.KEY_FILE, newline="") as handle:
        assert list(csv.DictReader(handle))[0]["name"] == "dev_practice"
    assert (dev / gs.BACKGROUND_FILE).is_file() and (dev / gs.REFLECTION_FILE).is_file()


def test_rehearsal_runs_without_files():
    panel = gsm.GraphicsStudyPanel(FakeUi(tempfile.mkdtemp()), rehearsal_only=True)
    panel.start_button.click()
    io = panel._io
    drive_arm(panel, io, tutorial=True)
    panel.next_button.click()
    drive_arm(panel, io, tutorial=False)
    assert panel.stage == "complete"


if __name__ == "__main__":
    test_full_study_with_files()
    test_questionnaire_reopens_after_closing_without_saving()
    test_headset_practice_run_is_not_a_participant()
    test_rehearsal_runs_without_files()
    print("ok")
