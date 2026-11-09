"""Rehearsal stand-in drives the real study controller end to end.

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_study_rehearsal.py
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import study_rehearsal as sr  # noqa: E402
import study_session as ss  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t


def run(session, io, clock, seconds, **state):
    io.state.update(state)
    for _ in range(int(seconds * 10)):
        clock.t += 0.1
        session.tick()


def test_pause_menu_stylus_moves_pedals_select_and_back():
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    io = sr.RehearsalIO()
    sim = sr.SimulatorStandIn(io)
    sim._timer.stop()
    sim._double_tap_1()                            # open
    assert io.state["paused"] and "FIVRS Pause Menu" in io.pause_menu.text()
    assert "Left pedal" in io.pause_menu.text() and "Middle pedal" in io.pause_menu.text()
    sim._pedal_c()                                 # right pedal does nothing in the menu
    assert io.pause_menu.row == 0 and io.pause_menu.sub is None
    sim._pedal_a()                                 # select: open Saved camera views
    assert io.pause_menu.sub == "views" and "        > Home" in io.pause_menu.text()
    sim._pedal_b()                                 # back to the top, still paused
    assert io.state["paused"] and io.pause_menu.sub is None
    sim._tap_2(); sim._tap_2(); sim._pedal_a()     # down twice: Burr size, on the current 6mm
    assert "        > 6mm" in io.pause_menu.text()
    sim._tap_2(); sim._pedal_a()                   # down wraps to 1mm
    assert io.state["drill"] == "1mm"
    sim._tap_1(); sim._pedal_a()                   # up from Burr size: Burr type list
    assert io.pause_menu.sub == "type"
    sim._tap_2(); sim._pedal_a()                   # down to Diamond
    assert io.state["style"] == "diamond"
    sim._pedal_b()                                 # at the top: exit
    assert not io.state["paused"]
    sim.close()


def test_full_rehearsal():
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841 (kept alive)
    clock = Clock()
    io = sr.RehearsalIO()
    io.now = clock
    sim = sr.SimulatorStandIn(io)
    sim._timer.stop()
    held = sim._held
    rec = sr.rehearsal_dir() / "R001"
    session = ss.StudySession(io, "R001", rec, {"rehearsal": True}, trial_seconds=30)

    def tick(seconds=0.3):
        for _ in range(int(seconds * 10)):
            clock.t += 0.1
            sim._tick()
            session.tick()

    def advance():
        sim._pedal_a(); tick()

    tick(); advance()                                   # welcome
    for _ in range(3):
        sim._move(); tick()
    assert session.progress.step_done; advance()        # moving the drill
    held["device_clutch"] = True
    for _ in range(3):
        sim._move(); tick()
    held["device_clutch"] = False; tick()
    assert session.progress.step_done; advance()        # repositioning the device
    sim._recenter(); tick(); advance()                  # recentering
    assert ss.TUTORIAL[session.step].key == "camera"
    sim._turn(-1, 0); tick()
    assert io.state["lx"] == 0.0                        # no turn without the microscope button
    held["scope_button"] = True; tick()
    for _ in range(60):                                 # turn the head towards the target
        if session.progress.step_done:
            break
        spot = sr.headset_hud.project(session.progress.look_target, sr.headset_hud.camera_from_state(io.state))
        sim._turn(-1 if spot[0] > 0.05 else (1 if spot[0] < -0.05 else 0), 1 if spot[1] < -0.05 else (-1 if spot[1] > 0.05 else 0))
        tick(0.1)
    tick(0.5)
    assert session.progress.step_done, spot
    held["scope_button"] = False; tick()
    advance()                                           # moving the microscope
    held["scope_button"] = True; tick()
    sim._pedal_a(); sim._pedal_a(); sim._pedal_c(); sim._pedal_c(); tick()
    held["scope_button"] = False; tick()
    assert session.progress.step_done, (io.state["zoom_in"], io.state["zoom_out"])
    assert io.state["advance_count"] == 5  # the pedal presses with the microscope button zoomed instead
    advance()                                           # zooming
    io.state["removed_voxels"] = 300; tick(); advance() # drilling
    sim._double_tap_1(); tick()                         # open the pause menu
    count = io.state["advance_count"]
    sim._pedal_a(); tick()                              # left pedal selects in the menu ...
    assert session.phase == "tutorial" and io.state["advance_count"] == count  # ... and does not advance
    assert io.pause_menu.sub == "views"
    sim._pedal_b(); tick()                              # back out of the list
    held["drilling"] = True; sim._pedal_b(); tick(1.0)  # close with the middle pedal, keep it down
    assert not io.state["paused"]
    assert io.state["removed_voxels"] == 300, "the drill waits until the pedal is released"
    held["drilling"] = False; tick()
    assert session.progress.step_done; advance()        # pause menu, the last step
    assert session.phase == "ready", (session.phase, session.step)
    assert session.hud(clock.t)["panel"]["advance"].endswith("begin your study session")

    advance()                                           # the resident begins
    assert session.phase == "briefing" and io.state["drill_locked"]
    assert io.state["drill"] == "6mm" and io.state["style"] == "cutting"
    assert session.hud(clock.t)["panel"]["cover"]
    tick(sr.RESTORE_S + sr.RELOAD_S + sr.RECORDER_S + 1.0)
    assert session.phase == "briefing"                  # ready behind the card, waiting for the pedal
    advance()
    assert session.phase == "armed" and io.state["removed_voxels"] == 0 and not io.state["drill_locked"]
    io.state["removed_voxels"] = 10; tick(0.1)
    assert session.phase == "running"
    tick(31.0)
    assert session.phase == "done" and io.state["drill_locked"] and (rec / "metadata.json").is_file()
    sim.close()


def test_windows_build_and_preview_shows_text():
    from PyQt5 import QtWidgets
    import study_mode

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Ui:
        study_mode = None
        def print_info(self, text): pass

    panel = study_mode.StudyPanel(Ui(), rehearsal_only=True)
    panel.start()
    assert panel.stand_in is not None and panel.session is not None
    panel._tick()
    panel.stand_in._tick()
    image = sr.preview_image(panel.stand_in.hud, 960, 540)
    top_band = image[: 540 // 3]
    assert (top_band.max(axis=2) > 200).any(), "tutorial text missing from the top of the preview"
    panel.stand_in._pedal_a(); panel._tick(); panel.stand_in._tick()  # welcome -> step 1, with pictures
    image = sr.preview_image(panel.stand_in.hud, 960, 540)
    assert (image[: 540 // 3].max(axis=2) > 200).any()
    panel.stand_in._toggle_pause()
    image = sr.preview_image(panel.stand_in.hud, 960, 540)
    assert (image[200:340, 380:580] != (200, 186, 160)).any(), "pause menu missing over the centre"
    panel.stand_in.close()
    assert panel.stand_in is None and not panel._timer.isActive()
    app.processEvents()


if __name__ == "__main__":
    test_pause_menu_stylus_moves_pedals_select_and_back()
    test_full_rehearsal()
    test_windows_build_and_preview_shows_text()
    print("ok")
