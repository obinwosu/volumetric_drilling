#!/usr/bin/env python3
"""Qt loading splash for RTX Tuner and RTX View startup.

Newton has no PyQt5, so this process is system Python. The child writes a log;
the splash tails that log and closes when a ready line appears. The child is
started in its own session so it keeps running after this window goes away.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import signal
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
STYLE = """
QDialog, QWidget { background: #0B1C2C; color: #E6EEF5; font-size: 16px; }
QLabel { color: #E6EEF5; }
QProgressBar {
    background: #0F2433; border: 1px solid #1E3A4F; border-radius: 4px;
    min-height: 22px; text-align: center; color: #E6EEF5;
}
QProgressBar::chunk { background: #1F6F8B; }
QPushButton {
    background: #1F6F8B; color: #E6EEF5; border: 1px solid #2F8FBF;
    border-radius: 4px; padding: 8px 16px; font-size: 16px; min-height: 32px;
}
QPushButton:hover { background: #2589A8; }
"""

# First matching prefix wins. 100 closes the splash.
STAGES = (
    ("Tuner ready", 100, "Tuner ready"),
    ("RTX view ready", 100, "RTX view ready"),
    ("Initializing RTX View", 6, "Initializing RTX View"),
    ("Tuner anatomy", 8, "Starting tuner"),
    ("Loading Newton", 16, "Loading Newton"),
    ("Loading anatomy slices", 16, "Loading anatomy"),
    ("Waiting for AMBF", None, "Waiting for simulator"),
    ("Opening RTX window", 34, "Opening RTX window"),
    ("Loading intact volume", 42, "Loading volume"),
    ("Closed occupancy", 48, "Preparing structures"),
    ("Fitting the EAC", 52, "Fitting the canal"),
    ("Meshing union", 62, "Meshing surfaces"),
    ("Initial surface", 62, "Meshing bone"),
    ("closed ", 72, "Meshing structures"),
    ("Classifying lateral", 80, "Cortex masks"),
    ("Fitting the lateral cortex", 80, "Fitting cortex"),
    ("Tuner burr", 88, "Loading instruments"),
    ("Fluted burr", 88, "Loading instruments"),
    ("Tuner sheet", 94, "Priming irrigation"),
)


def stage_from_line(line: str):
    """Return (percent or None, label) or None if the line is not a stage."""
    text = line.strip()
    if not text:
        return None
    for prefix, percent, label in STAGES:
        if text.startswith(prefix) or prefix in text:
            return percent, label
    return None


def kill_matching(script_name: str) -> None:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return
    mine = os.getpid()
    for raw in out.splitlines():
        parts = raw.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == mine:
            continue
        line = parts[1]
        if "python" in line.split()[0] and script_name in line:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def add_loading_widgets(qt_widgets, layout, heading, cancelable=False, on_cancel=None):
    """Heading, status, and bar. Returns (status_label, bar)."""
    title = qt_widgets.QLabel(heading)
    title.setStyleSheet("font-size: 20px; font-weight: 600;")
    layout.addWidget(title)
    status = qt_widgets.QLabel("Starting…")
    status.setWordWrap(True)
    layout.addWidget(status)
    bar = qt_widgets.QProgressBar()
    bar.setRange(0, 0)
    bar.setValue(0)
    layout.addWidget(bar)
    if cancelable:
        buttons = qt_widgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = qt_widgets.QPushButton("Cancel")
        if on_cancel is not None:
            cancel.clicked.connect(on_cancel)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
    return status, bar


def apply_stage(status, bar, line: str) -> bool:
    """Update status widgets from one log line. True when startup is finished."""
    hit = stage_from_line(line)
    if hit is None:
        return False
    percent, label = hit
    status.setText(label)
    if percent is None:
        bar.setRange(0, 0)
    else:
        current = bar.value() if bar.maximum() == 100 else 0
        bar.setRange(0, 100)
        bar.setValue(max(current, percent))
    return percent == 100


class Splash:
    """Standalone loading window."""

    def __init__(self, qt_widgets, heading, parent=None, cancelable=True):
        from PyQt5.QtCore import Qt

        self.QtWidgets = qt_widgets
        self.ready = False
        self.dialog = qt_widgets.QDialog(parent)
        self.dialog.setWindowTitle(heading)
        self.dialog.resize(520, 200)
        self.dialog.setStyleSheet(STYLE)
        layout = qt_widgets.QVBoxLayout(self.dialog)
        self.status, self.bar = add_loading_widgets(
            qt_widgets,
            layout,
            heading,
            cancelable=cancelable,
            on_cancel=self.dialog.reject,
        )
        self.dialog.setWindowModality(Qt.NonModal)

    def show(self):
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def close(self):
        self.dialog.close()

    def apply_line(self, line: str) -> bool:
        """Update from one log line. True when the process is ready."""
        if apply_stage(self.status, self.bar, line):
            self.ready = True
        return self.ready


def _drain_log(path: pathlib.Path, offset: int) -> tuple[int, list[str]]:
    if not path.is_file():
        return offset, []
    data = path.read_bytes()
    if len(data) <= offset:
        return offset, []
    chunk = data[offset:]
    text = chunk.decode("utf-8", errors="replace")
    return offset + len(chunk), text.splitlines()


def run_command(heading, ready, command, log_path, cancelable=True):
    """Show a splash, start ``command``, close when ``ready`` is printed."""
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QTimer

    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    splash = Splash(QtWidgets, heading, cancelable=cancelable)
    splash.show()
    handle = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(REPO),
    )
    state = {"offset": 0, "failed": False}

    def tick():
        state["offset"], lines = _drain_log(log_path, state["offset"])
        for line in lines:
            if splash.apply_line(line) or ready.lower() in line.lower():
                splash.ready = True
                splash.dialog.accept()
                return
        code = proc.poll()
        if code is not None and not splash.ready:
            state["failed"] = True
            splash.status.setText("Startup failed. See the log.")
            splash.bar.setRange(0, 100)
            splash.bar.setValue(0)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(100)
    result = splash.dialog.exec_()
    timer.stop()
    if splash.ready:
        return 0
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
    return 1 if result == QtWidgets.QDialog.Rejected or state["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description="RTX loading splash")
    parser.add_argument("--heading", default="Opening RTX Tuner")
    parser.add_argument("--ready", default="Tuner ready")
    parser.add_argument("--log", required=True, type=pathlib.Path)
    parser.add_argument("--no-cancel", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("rtx_splash.py needs a command after --")
    return run_command(
        args.heading,
        args.ready,
        command,
        args.log,
        cancelable=not args.no_cancel,
    )


def _self_check():
    assert stage_from_line("Tuner ready. Views canonical_views.yaml")[0] == 100
    assert stage_from_line("RTX view ready")[0] == 100
    assert stage_from_line("Initializing RTX View")[1] == "Initializing RTX View"
    assert stage_from_line("Waiting for AMBF volume_info before opening the RTX window")[0] is None
    assert stage_from_line("Meshing union occupancy for bone")[1] == "Meshing surfaces"
    assert stage_from_line("random chatter") is None


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        print("ok")
    else:
        sys.exit(main())
