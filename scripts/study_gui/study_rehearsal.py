"""Rehearse the study flow without the simulator.

    python3 scripts/study_gui/study_rehearsal.py           # Study tab on its own
    (or tick "Rehearse without the simulator" on the launcher's Study tab)

A stand-in window replaces AMBF, the RTX view and the recorder. Its buttons
produce the same state the plugin writes to drill_state.json (drill travel,
clutches, zoom, recenter, orbit, voxels removed, pause menu and burr), and a
preview shows the headset text exactly as headset_hud draws it. The real
study controller (scripts/study/study_session.py) runs unchanged, so the
tutorial, Begin trial, reset, timed trial and end popup behave as on the rig.

Nothing here touches /tmp/fivrs-rtx, the participant key or the study
recordings, so a rehearsal cannot disturb a real session.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import time

import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "study"))
sys.path.insert(0, str(REPO / "scripts" / "rtx_view"))
import headset_hud  # noqa: E402

RESTORE_S = 1.0       # stand-in time for AMBF to restore the bone
RELOAD_S = 2.0        # and for the RTX window to remesh it
RECORDER_S = 1.0      # and for the recorder's ROS node to come up
BURR_SIZES = ["1mm", "2mm", "4mm", "6mm"]
BURR_TYPES = ["Cutting", "Diamond"]
PREVIEW_SIZE = (960, 540)


class PauseMenu:
    """The plugin's pause menu (volumetric_drilling.cpp pauseMenuListText): saved camera
    views, burr type and burr size, each opening a list that ends in Back, then Resume.
    Hold the left pedal to open it; in it a left-pedal tap moves down (wrapping) and
    the middle pedal selects. Holding the left pedal in the menu asks "Close pause
    menu and return to simulator?": a tap toggles Yes / No, the middle pedal selects.
    The stylus does nothing in the menu."""

    VIEWS = ["Home", "View 1  empty", "View 2  empty"]
    ROWS = 4              # 0 saved views, 1 burr type, 2 burr size, 3 Resume

    def __init__(self, state: dict):
        self.state = state
        self.row = 0
        self.sub = None       # None, "views", "type" or "size"
        self.sub_index = 0
        self.notice = ""
        self.confirm = None   # None, or True / False for the highlighted Yes / No

    def open(self):
        self.row, self.sub, self.notice, self.confirm = 0, None, "", None

    def ask_close(self):
        if self.confirm is None:
            self.confirm = True

    def options(self):
        return {"views": self.VIEWS, "type": BURR_TYPES, "size": BURR_SIZES}[self.sub]

    def next(self, step: int = 1):
        if self.confirm is not None:
            self.confirm = not self.confirm
        elif self.sub:
            self.sub_index = (self.sub_index + step) % (len(self.options()) + 1)  # + Back
        else:
            self.row = (self.row + step) % self.ROWS

    def select(self) -> bool:
        """Choose the highlighted row. True means Resume: leave the menu."""
        if self.confirm is not None:
            yes, self.confirm = self.confirm, None
            return yes
        if self.sub and self.sub_index == len(self.options()):
            self.sub = None                                   # Back
        elif self.sub == "views":
            self.notice, self.sub = f"View: {self.VIEWS[self.sub_index].split()[0]}", None
        elif self.sub == "type":
            self.state["style"] = BURR_TYPES[self.sub_index].lower()
            self.notice, self.sub = f"Burr type: {BURR_TYPES[self.sub_index]}", None
        elif self.sub == "size":
            self.state["drill"] = BURR_SIZES[self.sub_index]
            self.notice, self.sub = f"Burr size: {BURR_SIZES[self.sub_index]}", None
        elif self.row == 0:
            self.sub, self.sub_index = "views", 0
        elif self.row == 1:
            self.sub = "type"
            self.sub_index = BURR_TYPES.index(self.state["style"].capitalize())
        elif self.row == 2:
            self.sub = "size"
            self.sub_index = BURR_SIZES.index(self.state["drill"])
        else:
            return True
        return False

    def text(self) -> str:
        """Same layout as the plugin's pause_hud.txt."""
        def row(i, label, value=""):
            return ("  > " if i == self.row and not self.sub else "    ") + label.ljust(20) + value

        def options(name, items):
            if self.sub != name:
                return []
            items = list(items) + ["Back"]
            return [("        > " if k == self.sub_index else "          ") + n for k, n in enumerate(items)]

        lines = ["FIVRS Pause Menu", ""]
        if self.confirm is not None:
            lines += ["Close pause menu and return to simulator?", "",
                      ("  > Yes" if self.confirm else "    Yes"),
                      ("    No" if self.confirm else "  > No"),
                      "", "pause menu navigation",
                      "Yes / No             Left pedal",
                      "Select               Middle pedal"]
            return "\n".join(lines) + "\n"
        lines += [row(0, "Saved camera views")] + options("views", self.VIEWS)
        lines += [row(1, "Burr type", self.state["style"].capitalize())] + options("type", BURR_TYPES)
        lines += [row(2, "Burr size", self.state["drill"])] + options("size", BURR_SIZES)
        lines += [row(3, "Resume")]
        if self.notice:
            lines += ["", self.notice]
        lines += ["", "pause menu navigation",
                  "Down                 Tap left pedal",
                  "Select               Middle pedal",
                  ("Back                 Select Back" if self.sub else
                   "Resume               Select Resume"),
                  "Close menu           Hold left pedal"]
        return "\n".join(lines) + "\n"


class RehearsalIO:
    """SessionIO with the simulator, RTX window and recorder replaced by timers."""

    def __init__(self):
        # Same fields as the plugin's drill_state.json. The camera looks down at the
        # drill tip from one unit away.
        self.state = {"px": 0.0, "py": 0.0, "pz": 0.0, "hx": 0.0, "hy": 0.0, "hz": 0.0,
                      "cam_clutch": False, "device_clutch": False, "orbit": False, "scope_button": False,
                      "zoom_travel": 0.0, "zoom_in": 0.0, "zoom_out": 0.0, "advance_count": 0, "skip_count": 0,
                      "recenter_count": 0, "removed_voxels": 0,
                      "cx": 0.0, "cy": 0.0, "cz": 1.0, "lx": 0.0, "ly": 0.0, "lz": -1.0,
                      "ux": 0.0, "uy": 1.0, "uz": 0.0, "fov": 0.18,
                      "paused": False, "drill": "6mm", "style": "cutting", "drill_locked": False}
        self.pause_menu = PauseMenu(self.state)
        self.hud = None
        self._restore_at = self._reload_at = self._recorder_at = None
        self.recorder = "off"

    def now(self) -> float:
        return time.time()

    def read_state(self):
        return dict(self.state)

    def write_hud(self, hud: dict) -> None:
        self.hud = hud

    def clear_hud(self) -> None:
        self.hud = None

    def set_drill_lock(self, locked: bool) -> None:
        self.state["drill_locked"] = bool(locked)

    def set_trial_active(self, active: bool) -> None:
        self.state["trial_active"] = bool(active)

    def write_headset_place(self, seq: int, kind: str, tilt_deg: float | None = None) -> None:
        self.state["headset_place"] = (int(seq), kind, tilt_deg)

    def request_restore(self) -> None:
        self._restore_at = self.now() + RESTORE_S
        self._reload_at = None

    def restore_done(self) -> bool:
        if self._restore_at is not None and self.now() >= self._restore_at:
            self.state["removed_voxels"] = 0  # the plugin zeroes the count on restore
            return True
        return False

    def request_reload(self) -> None:
        self._reload_at = self.now() + RELOAD_S

    def request_burr_reset(self) -> None:
        self.state.update(drill="6mm", style="cutting")

    def restore_reload_wall(self) -> float:
        if self._reload_at is not None and self.now() >= self._reload_at:
            return self._reload_at
        return 0.0

    def start_recorder(self, output_dir: pathlib.Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = "starting"
        self._recorder_at = self.now() + RECORDER_S

    def recorder_running(self) -> bool:
        return self.recorder in ("starting", "on")

    def recorder_ready(self) -> bool:
        if self.recorder == "starting" and self.now() >= self._recorder_at:
            self.recorder = "on"
        return self.recorder == "on"

    def stop_recorder(self, timeout_s: float = 60.0):
        was = self.recorder
        self.recorder = "stopped"
        return 0 if was == "on" else None


class PreviewHud(headset_hud.HeadsetHud):
    """headset_hud's drawing, fed from the rehearsal instead of /tmp/fivrs-rtx."""

    def __init__(self, io: RehearsalIO):
        super().__init__()
        self.io = io

    def _poll(self, now: float):
        self._study = self.io.hud or {}
        self._state = self.io.state
        self._paused = bool(self.io.state["paused"])
        self._pause_text = self.io.pause_menu.text() if self._paused else None


def preview_image(hud: PreviewHud, width: int, height: int) -> np.ndarray:
    """A stand-in eye image (dark field, bone disc in the middle) with the headset text on it."""
    image = np.empty((height, width, 3), np.uint8)
    image[...] = (24, 18, 16)
    yy, xx = np.ogrid[:height, :width]
    bone = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= (height * 0.3) ** 2
    image[bone] = (200, 186, 160)
    hud.draw(image, image.copy())
    return image


class SimulatorStandIn(QtWidgets.QWidget):
    """Buttons for what the resident does in the headset, and the headset preview."""

    closed = pyqtSignal()

    def __init__(self, io: RehearsalIO):
        super().__init__()
        self.io = io
        self.hud = PreviewHud(io)
        self._held = {}
        self._last = time.time()
        self.setWindowTitle("Study rehearsal: simulator stand-in")
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---- layout ----
    def _build(self):
        layout = QtWidgets.QHBoxLayout(self)
        controls = QtWidgets.QVBoxLayout()
        layout.addLayout(controls)

        stylus = QtWidgets.QGroupBox("Stylus")
        grid = QtWidgets.QGridLayout(stylus)
        self._button(grid, 0, 0, "Move the stylus", "Moves the drill about 10 mm, or only the stylus "
                     "while the action button is held", click=self._move)
        self._button(grid, 0, 1, "Hold action button (reposition)", toggle="device_clutch")
        self._button(grid, 1, 0, "Double-tap action button", "Recenter (not in the pause menu)",
                     click=self._double_tap_2)
        self._button(grid, 1, 1, "Hold Ctrl + action button (orbit)", toggle="orbit")
        controls.addWidget(stylus)

        scope = QtWidgets.QGroupBox("Microscope and headset")
        grid = QtWidgets.QGridLayout(scope)
        self._button(grid, 0, 0, "Hold microscope button", "Camera clutch; with a pedal, zoom",
                     toggle="scope_button")
        arrows = QtWidgets.QHBoxLayout()
        for text, yaw, pitch in (("Turn head \u2190", 1, 0), ("\u2192", -1, 0), ("\u2191", 0, 1), ("\u2193", 0, -1)):
            button = QtWidgets.QPushButton(text)
            button.setAutoRepeat(True)
            button.setToolTip("Moves the view only while the microscope button is held")
            button.clicked.connect(lambda _=False, y=yaw, p=pitch: self._turn(y, p))
            arrows.addWidget(button)
        grid.addLayout(arrows, 0, 1)
        controls.addWidget(scope)

        feet = QtWidgets.QGroupBox("Foot pedals")
        row = QtWidgets.QHBoxLayout(feet)
        self.pedal_a = self._button(row, None, None, "Left (a): tap", "Advance; down in the pause menu",
                                    click=self._pedal_a)
        self._button(row, None, None, "Left (a): hold", "Open the pause menu; in it, close (Yes / No)", click=self._hold_a)
        self.pedal_b = self._button(row, None, None, "Middle (b)",
                                    "Select in the pause menu; zoom out with the microscope button",
                                    click=self._pedal_b)
        self.pedal_c = self._button(row, None, None, "Right (c): hold to drill", hold="drilling")
        controls.addWidget(feet)
        self._button(controls, None, None, "Right arrow key: skip tutorial step",
                     "Moves on without doing the step; elsewhere acts like the left pedal",
                     click=self._right_arrow)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        note = QtWidgets.QLabel(
            "\"Hold\" buttons stay down until clicked again, so they can be combined with a pedal. "
            "Left pedal: a tap advances the tutorial, a hold opens the pause menu, and with the "
            "microscope button it zooms in. Middle pedal: zooms out with the microscope button. Right "
            "pedal: hold to drill. In the pause menu (pedals only): left tap moves down, middle selects; "
            "each list ends in Back and Resume closes the menu.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        controls.addWidget(note)
        controls.addStretch(1)

        self.preview = QtWidgets.QLabel()
        self.preview.setMinimumSize(PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2)
        self.preview.setAlignment(Qt.AlignCenter)
        box = QtWidgets.QGroupBox("Headset view (text only)")
        QtWidgets.QVBoxLayout(box).addWidget(self.preview)
        layout.addWidget(box, 1)

    def _button(self, layout, r, c, text, tip="", click=None, hold=None, toggle=None):
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tip)
        if click:
            button.clicked.connect(click)
        if hold:
            button.pressed.connect(lambda: self._held.__setitem__(hold, True))
            button.released.connect(lambda: self._held.__setitem__(hold, False))
        if toggle:
            button.setCheckable(True)
            button.setStyleSheet("QPushButton:checked { background-color: #b8860b; color: white; }")
            button.toggled.connect(lambda on: self._held.__setitem__(toggle, on))
        if r is None:
            layout.addWidget(button)
        else:
            layout.addWidget(button, r, c)
        return button

    # ---- actions ----
    def _move(self):
        state = self.io.state
        state["hx"] += 0.006  # metres of stylus travel
        if not state["device_clutch"]:
            state["px"] += 0.2  # simulator units of drill travel

    def _recenter(self):
        self.io.state["recenter_count"] += 1

    def _zoom(self, doublings):
        state = self.io.state
        fov = state["fov"]
        state["fov"] = min(0.60, max(0.035, fov * 2.0 ** doublings))
        change = abs(np.log2(state["fov"] / fov))
        state["zoom_travel"] += change
        state["zoom_in" if doublings < 0 else "zoom_out"] += change

    def _turn(self, yaw, pitch):
        """Head moved while the microscope button is held: turn the view half a degree."""
        state = self.io.state
        if not state["cam_clutch"] or state["paused"]:
            return
        look = np.array([state["lx"], state["ly"], state["lz"]])
        up = np.array([state["ux"], state["uy"], state["uz"]])
        right = np.cross(look, up)
        angle = np.radians(0.5)
        if yaw:
            look = look * np.cos(angle) - right * yaw * np.sin(angle)
        if pitch:
            look, up = look * np.cos(angle) + up * pitch * np.sin(angle), up * np.cos(angle) - look * pitch * np.sin(angle)
        look /= np.linalg.norm(look)
        up /= np.linalg.norm(up)
        state.update(lx=look[0], ly=look[1], lz=look[2], ux=up[0], uy=up[1], uz=up[2])

    def _toggle_pause(self):
        paused = not self.io.state["paused"]
        self.io.state["paused"] = paused
        if paused:
            self.io.pause_menu.open()

    def _double_tap_2(self):
        if not self.io.state["paused"]:  # the stylus does nothing in the pause menu
            self._recenter()

    def _right_arrow(self):
        if not self.io.state["paused"]:
            self.io.state["skip_count"] += 1

    def _hold_a(self):
        """Left pedal held down: opens the pause menu, or in it asks to close it
        (not with the microscope button)."""
        state = self.io.state
        if state["scope_button"]:
            return
        if not state["paused"]:
            self._toggle_pause()
        else:
            self.io.pause_menu.ask_close()

    def _pedal_a(self):
        """Left pedal tapped."""
        state = self.io.state
        if state["paused"]:
            self.io.pause_menu.next()
        elif state["scope_button"]:
            self._zoom(-0.25)
        else:
            state["advance_count"] += 1

    def _pedal_b(self):
        """Middle pedal: select in the pause menu (Resume leaves it); with the microscope
        button, zoom out."""
        state = self.io.state
        if state["paused"]:
            if self.io.pause_menu.select():
                state["paused"] = False
        elif state["scope_button"]:
            self._zoom(0.25)

    # ---- loop ----
    def _tick(self):
        now = time.time()
        dt, self._last = now - self._last, now
        state = self.io.state
        paused = state["paused"]
        state["scope_button"] = bool(self._held.get("scope_button"))
        state["cam_clutch"] = state["scope_button"] and not paused
        for key in ("device_clutch", "orbit"):
            state[key] = bool(self._held.get(key)) and not paused
        drilling = self._held.get("drilling")
        if not drilling:
            self._drill_suppress = False
        elif paused:
            # As on the rig: a right pedal held through the pause menu waits for release.
            self._drill_suppress = True
        drilling = drilling and not getattr(self, "_drill_suppress", False)
        if drilling and not paused and not state["drill_locked"]:
            state["removed_voxels"] += int(400 * dt)  # about 400 voxels a second
        self.status.setText(
            f"Drill {'LOCKED' if state['drill_locked'] else 'free'}  ·  {'PAUSED' if paused else 'running'}  ·  "
            f"burr {state['drill']} {state['style']}\nvoxels removed {state['removed_voxels']}  ·  "
            f"zoom in {state['zoom_in']:.2f} / out {state['zoom_out']:.2f}  ·  left-pedal taps {state['advance_count']}"
            f"  ·  recorder {self.io.recorder}")
        image = preview_image(self.hud, *PREVIEW_SIZE)
        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                              QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self._timer.stop()
        self.closed.emit()
        super().closeEvent(event)


def rehearsal_dir() -> pathlib.Path:
    """Throwaway folder for the rehearsal's metadata.json (never the study recordings)."""
    return pathlib.Path(tempfile.mkdtemp(prefix="fivrs_rehearsal_"))


class _StandaloneUi:
    """What StudyPanel needs from the launcher, for running the Study tab on its own."""

    def __init__(self, log: QtWidgets.QPlainTextEdit):
        self._log = log
        self.study_mode = None

    def print_info(self, text):
        self._log.appendPlainText(text)


def main():
    import study_mode

    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    window.setWindowTitle("FIVRS study (rehearsal)")
    layout = QtWidgets.QVBoxLayout(window)
    log = QtWidgets.QPlainTextEdit()
    log.setReadOnly(True)
    panel = study_mode.StudyPanel(_StandaloneUi(log), rehearsal_only=True)
    layout.addWidget(panel)
    layout.addWidget(log)
    window.resize(720, 520)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
