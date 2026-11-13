"""Proctor pause window.

While the participant's pause menu is open in the simulator, this window shows
the same choices as buttons on the monitor, so the proctor can make them
instead of talking the participant through the pedal menu. It closes when the
menu closes.

The simulator writes /tmp/fivrs-rtx/pause_options.json while the menu is open
and reads one command at a time from /tmp/fivrs-rtx/proctor_menu.txt
(serviceProctorPauseCommand in volumetric_drilling.cpp).
"""
from __future__ import annotations

import json
import pathlib
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
OPTIONS_PATH = RUNTIME / "pause_options.json"
COMMAND_PATH = RUNTIME / "proctor_menu.txt"
DRILL_STATE_PATH = RUNTIME / "drill_state.json"
# drill_state.json is rewritten every frame; older than this means the simulator is gone.
STALE_SEC = 3.0


def read_options(now: float | None = None) -> dict | None:
    """The open menu's options, or None when no live simulator has its menu open."""
    now = time.time() if now is None else now
    try:
        if now - DRILL_STATE_PATH.stat().st_mtime > STALE_SEC:
            return None
        options = json.loads(OPTIONS_PATH.read_text())
    except (OSError, ValueError):
        return None
    return options if isinstance(options, dict) and options.get("open") else None


def send(verb: str, arg: str = "") -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    partial = COMMAND_PATH.with_suffix(".txt.part")
    partial.write_text(f"{verb} {arg}".strip() + "\n")
    partial.replace(COMMAND_PATH)


class ProctorPauseWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Participant pause menu")
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("QPushButton:checked { background: #2e7d32; color: white; font-weight: bold; }")
        self._layout_key = None
        self._options = {}
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.poll)
        self._timer.start()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Simulator paused: the participant's pause menu is open")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel(
            "Make the choice here; the headset menu follows. Press Resume to return the participant to drilling."))

        self.box_type = QtWidgets.QGroupBox("Burr type")
        self.row_type = QtWidgets.QHBoxLayout(self.box_type)
        self.type_buttons = {}
        for key, label in (("cutting", "Cutting"), ("diamond", "Diamond")):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, k=key: send("burr_type", k))
            self.row_type.addWidget(button)
            self.type_buttons[key] = button
        layout.addWidget(self.box_type)

        self.box_size = QtWidgets.QGroupBox("Burr size")
        self.row_size = QtWidgets.QHBoxLayout(self.box_size)
        self.size_buttons = []
        layout.addWidget(self.box_size)

        self.box_views = QtWidgets.QGroupBox("Saved camera views")
        self.row_views = QtWidgets.QGridLayout(self.box_views)
        layout.addWidget(self.box_views)

        self.menu_label = QtWidgets.QLabel()
        self.menu_label.setFont(QFont("Monospace", 9))
        self.menu_label.setStyleSheet("color: #555;")
        menu_box = QtWidgets.QGroupBox("What the participant sees in the headset")
        QtWidgets.QVBoxLayout(menu_box).addWidget(self.menu_label)
        layout.addWidget(menu_box)

        buttons = QtWidgets.QHBoxLayout()
        self.button_end = QtWidgets.QPushButton("End trial")
        self.button_end.setStyleSheet("color: #b00020;")
        self.button_end.clicked.connect(self._end_trial)
        buttons.addWidget(self.button_end)
        buttons.addStretch(1)
        self.button_resume = QtWidgets.QPushButton("Resume")
        self.button_resume.setMinimumHeight(44)
        self.button_resume.setMinimumWidth(180)
        self.button_resume.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.button_resume.clicked.connect(lambda: send("resume"))
        buttons.addWidget(self.button_resume)
        layout.addLayout(buttons)

    @staticmethod
    def _clear(grid):
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _rebuild(self, options):
        """Size and view buttons, only when the lists change."""
        sizes = [str(s) for s in options.get("sizes") or []]
        views = [(str(v.get("id", "")), str(v.get("label", "")), bool(v.get("valid")))
                 for v in options.get("views") or []]
        key = (tuple(sizes), tuple(views))
        if key == self._layout_key:
            return
        self._layout_key = key
        self._clear(self.row_size)
        self.size_buttons = []
        for k, name in enumerate(sizes):
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, i=k: send("burr_size", str(i)))
            self.row_size.addWidget(button)
            self.size_buttons.append(button)
        self._clear(self.row_views)
        for k, (view_id, label, valid) in enumerate(views):
            button = QtWidgets.QPushButton(label if valid else f"{label} (empty)")
            button.setEnabled(valid)
            button.clicked.connect(lambda _=False, v=view_id: send("view", v))
            self.row_views.addWidget(button, k // 3, k % 3)
        self.box_views.setVisible(bool(views))

    def _apply(self, options):
        self._options = options
        self._rebuild(options)
        style = options.get("style")
        for key, button in self.type_buttons.items():
            button.setChecked(key == style)
        for k, button in enumerate(self.size_buttons):
            button.setChecked(k == options.get("size_index"))
        self.button_end.setVisible(bool(options.get("trial_active")))
        self.menu_label.setText(str(options.get("menu", "")).rstrip())

    def _end_trial(self):
        answer = QtWidgets.QMessageBox.question(
            self, "End trial",
            "End the timed trial now? The participant moves on to the questionnaires and feedback.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            send("end_trial")

    def poll(self):
        options = read_options()
        if options is None:
            if self.isVisible():
                self.hide()
            return
        self._apply(options)
        if not self.isVisible():
            self.adjustSize()
            self.show()
            self.raise_()
