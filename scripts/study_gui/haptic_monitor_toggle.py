"""Advanced tab: "Haptic device" box that opens / closes the haptic monitor window.

The monitor (scripts/haptic_monitor/haptic_monitor.py) is its own process and
window, so it keeps running beside the simulator. The checkbox state is saved as
`haptic_monitor: true/false` in gui_setup.yaml; a saved true reopens it when the
launcher starts. Closing the monitor window yourself unticks the box.

Before AMBF starts, the launcher calls release_for_simulator(): with the
simulator closed the monitor owns the Touch, and only one process can.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QProcess

REPO = pathlib.Path(__file__).resolve().parents[2]
MONITOR_DIR = REPO / "scripts" / "haptic_monitor"
if str(MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(MONITOR_DIR))

import haptic_monitor  # noqa: E402

CONFIG_KEY = "haptic_monitor"


class HapticMonitorToggle(QtCore.QObject):
    def __init__(self, configuration, print_info=print, command=None, parent=None):
        """configuration: GUIConfiguration (yaml_data + save_value). command: argv for the
        monitor process (tests pass a stand-in)."""
        super().__init__(parent)
        self.configuration = configuration
        self.print_info = print_info
        self.command = command or [sys.executable, str(MONITOR_DIR / "haptic_monitor.py")]
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ForwardedChannels)
        self.process.finished.connect(self._finished)
        self._shutting_down = False
        self.checkbox: QtWidgets.QCheckBox | None = None

    # --- widgets

    def build_group(self, parent) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Haptic device", parent)
        row = QtWidgets.QHBoxLayout(box)
        self.checkbox = QtWidgets.QCheckBox("Haptic monitor window", box)
        self.checkbox.setToolTip(
            "Live Touch data (encoders, angles, buttons, temperature, servo rate, forces) and a "
            "health log. Works with the simulator running (reads its feed) or closed (opens the "
            "Touch itself, no force).")
        self.checkbox.setChecked(self.saved())
        self.checkbox.toggled.connect(self.set_open)
        log_button = QtWidgets.QPushButton("Open health log", box)
        log_button.setToolTip(str(haptic_monitor.HEALTH_LOG))
        log_button.clicked.connect(self.open_log)
        row.addWidget(self.checkbox)
        row.addWidget(log_button)
        row.addWidget(QtWidgets.QLabel(f"Log: {haptic_monitor.HEALTH_LOG}", box), 1)
        return box

    def saved(self) -> bool:
        return bool((self.configuration.yaml_data or {}).get(CONFIG_KEY, False))

    def _save(self, value: bool):
        if self.saved() != value:
            self.configuration.save_value(CONFIG_KEY, bool(value))

    def restore(self):
        """Launcher start: reopen the monitor if it was open last time."""
        if self.saved():
            self.set_open(True)

    # --- process

    def is_open(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def set_open(self, value: bool):
        self._save(value)
        if self.checkbox is not None and self.checkbox.isChecked() != value:
            self.checkbox.blockSignals(True)
            self.checkbox.setChecked(value)
            self.checkbox.blockSignals(False)
        if value and not self.is_open():
            self.process.start(self.command[0], self.command[1:])
            self.print_info("Haptic monitor opened")
        elif not value and self.is_open():
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()

    def _finished(self, *_):
        if self._shutting_down:
            return
        # The window was closed by hand (or crashed): show it as closed.
        if self.checkbox is not None and self.checkbox.isChecked():
            self.checkbox.blockSignals(True)
            self.checkbox.setChecked(False)
            self.checkbox.blockSignals(False)
            self._save(False)

    def open_log(self):
        path = haptic_monitor.HEALTH_LOG
        if not path.exists():
            self.print_info(f"No haptic health log yet ({path})")
            return
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def release_for_simulator(self) -> bool:
        """Free the Touch before AMBF starts. False if the monitor still holds it."""
        if haptic_monitor.release_device_for_sim():
            return True
        self.print_info("WARNING! The haptic monitor still holds the Touch; AMBF may not get the device.")
        return False

    def shutdown(self):
        """Launcher closing: close the monitor but keep the saved setting."""
        self._shutting_down = True
        if self.is_open():
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
