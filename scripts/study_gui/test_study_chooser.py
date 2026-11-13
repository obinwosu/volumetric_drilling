"""The master Study tab: picking a study shows its panel, and the choice is locked while
a study runs. Stand-in simulator only (no AMBF, RTX or recorder).

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_study_chooser.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "study"))

from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtCore import QProcess  # noqa: E402

import test_graphics_study_mode as tg  # noqa: E402  (message boxes, NASA-TLX answers, FakeUi)
import graphics_study_mode as gsm  # noqa: E402
import gui_setup  # noqa: E402
import study_chooser  # noqa: E402
import study_mode  # noqa: E402

APP = tg.APP


def make(choice="resident"):
    ui = tg.FakeUi(tempfile.mkdtemp())
    resident = study_mode.StudyPanel(ui, rehearsal_only=True)
    graphics = gsm.GraphicsStudyPanel(ui, rehearsal_only=True)
    ui.study_panel, ui.graphics_panel = resident, graphics
    saved = []
    chooser = study_chooser.StudyChooser(ui, resident, graphics, choice=choice, save=saved.append)
    return ui, chooser, resident, graphics, saved


def shown(chooser):
    return chooser.stack.currentWidget().widget()


def locked(chooser):
    chooser.refresh_lock()
    return not chooser.buttons["resident"].isEnabled()


def test_switching_shows_the_right_panel_and_remembers_it():
    ui, chooser, resident, graphics, saved = make("graphics")
    assert shown(chooser) is graphics and chooser.buttons["graphics"].isChecked()
    assert not locked(chooser) and not chooser.end_button.isEnabled()
    chooser.buttons["resident"].click()
    assert shown(chooser) is resident and chooser.choice == "resident"
    chooser.buttons["graphics"].click()
    assert shown(chooser) is graphics
    assert saved == ["resident", "graphics"]
    _, chooser, resident, _, _ = make("nonsense")  # an unknown saved value falls back
    assert shown(chooser) is resident


def test_locked_while_the_graphics_study_runs():
    ui, chooser, resident, graphics, saved = make("graphics")
    graphics.start_button.click()  # rehearsal: stand-in simulator
    assert locked(chooser) and chooser.lock_label.text().startswith("Locked: ")
    assert chooser.choose("resident") is False
    chooser.buttons["resident"].click()  # disabled: nothing happens
    assert shown(chooser) is graphics and chooser.buttons["graphics"].isChecked() and saved == []
    io = graphics._io
    tg.drive_arm(graphics, io, tutorial=True)
    assert graphics.stage == "between" and locked(chooser)  # NASA-TLX saved, session 2 to come
    graphics.next_button.click()
    tg.drive_arm(graphics, io, tutorial=False)
    assert graphics.stage == "complete"
    assert not locked(chooser)
    assert chooser.choose("resident") and shown(chooser) is resident


def test_proctor_ends_the_graphics_study():
    ui, chooser, resident, graphics, _ = make("graphics")
    graphics.start_button.click()
    assert locked(chooser) and chooser.end_button.isEnabled()
    chooser.end_now()
    assert graphics.stage == "idle" and graphics.stand_in is None
    assert graphics.start_button.isEnabled()
    assert not locked(chooser) and chooser.choose("resident")


def test_locked_while_the_simulator_is_open():
    ui, chooser, resident, graphics, _ = make("resident")
    ui._ambf_process.running = True
    assert locked(chooser) and "simulator" in chooser.lock_label.text()
    assert not chooser.end_button.isEnabled()  # close the simulator first
    assert chooser.choose("graphics") is False and shown(chooser) is resident
    ui._ambf_process.running = False
    assert not locked(chooser)


def test_locked_during_a_resident_session_and_its_questionnaire():
    ui, chooser, resident, graphics, _ = make("resident")
    resident.start_button.click()  # rehearsal_only: the stand-in simulator
    assert resident.session is not None and locked(chooser)
    assert chooser.choose("graphics") is False and shown(chooser) is resident
    resident.stand_in.close()  # the "simulator" closes: the session stops
    APP.processEvents()
    assert not locked(chooser)

    class Server:  # the questionnaire web app
        def __init__(self):
            self.running = True

        def state(self):
            return QProcess.Running if self.running else QProcess.NotRunning

        def terminate(self):
            self.running = False

        def waitForFinished(self, _ms):
            return True

    resident._feedback_server = Server()
    resident._resident_returned = False
    assert locked(chooser) and "questionnaire" in chooser.lock_label.text()
    resident._resident_returned = True  # the resident pressed "Return to the simulator study window"
    assert not locked(chooser)
    resident._resident_returned = False
    assert locked(chooser) and chooser.end_button.isEnabled()
    chooser.end_now()  # the proctor ends it: the questionnaire server stops
    assert not resident._feedback_server.running and not locked(chooser)


def test_a_running_study_is_never_hidden():
    ui, chooser, resident, graphics, _ = make("resident")
    graphics.start_button.click()  # started while the resident panel was shown
    chooser.refresh_lock()
    assert shown(chooser) is graphics and locked(chooser)
    chooser.end_now()


def test_choice_saved_in_gui_setup_without_touching_the_rest():
    folder = pathlib.Path(tempfile.mkdtemp())
    path = folder / "gui_setup.yaml"
    lines = [line for line in (HERE / "gui_setup.yaml").read_text().splitlines(keepends=True)
             if not line.startswith(study_chooser.SETTING + ":")]
    original = "".join(lines)
    path.write_text(original)
    config = gui_setup.GUIConfiguration(str(path))
    config.sim_options["bounces"] = 99  # an unsaved change on the launcher: must not be written
    config.save_value(study_chooser.SETTING, "resident")
    assert path.read_text() == original + "study_choice: resident\n"  # one line added, nothing else
    config.save_value(study_chooser.SETTING, "graphics")
    assert path.read_text() == original + "study_choice: graphics\n"  # that line replaced
    assert yaml.safe_load(path.read_text())[study_chooser.SETTING] == "graphics"
    assert gui_setup.GUIConfiguration(str(path)).yaml_data[study_chooser.SETTING] == "graphics"
    config.save()  # a later full save keeps the choice
    assert yaml.safe_load(path.read_text())[study_chooser.SETTING] == "graphics"


if __name__ == "__main__":
    test_switching_shows_the_right_panel_and_remembers_it()
    test_locked_while_the_graphics_study_runs()
    test_proctor_ends_the_graphics_study()
    test_locked_while_the_simulator_is_open()
    test_locked_during_a_resident_session_and_its_questionnaire()
    test_a_running_study_is_never_hidden()
    test_choice_saved_in_gui_setup_without_touching_the_rest()
    print("ok")
