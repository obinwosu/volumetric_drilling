#!/usr/bin/env python3
"""Qt loading splash for RTX Tuner and RTX View startup.

Newton has no PyQt5, so this process is system Python. The child writes a log;
the splash tails that log and closes when a ready line appears. The child is
started in its own session so it keeps running after this window goes away.
"""

from __future__ import annotations

import argparse
import os
import re
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
    # The live view opens its window hidden and reports the bone upload; it is
    # ready only when the whole bone is in and the window is shown.
    ("RTX bone ready", 100, "Ready"),
    ("RTX view ready", 50, "Loading the bone"),
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


# "Loading bone 37%" from the live view maps onto this part of the bar.
BONE_RANGE = (50, 99)


def bone_fraction(line: str):
    """0..1 from a "Loading bone NN%" line, else None."""
    match = re.search(r"Loading bone (\d+)%", line)
    return None if match is None else min(1.0, int(match.group(1)) / 100.0)


def stage_from_line(line: str):
    """Return (percent or None, label) or None if the line is not a stage."""
    text = line.strip()
    if not text:
        return None
    fraction = bone_fraction(text)
    if fraction is not None:
        low, high = BONE_RANGE
        return int(low + (high - low) * fraction), "Loading the bone"
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


def make_spinner(qt_widgets, size=44):
    """A turning arc, so the window reads as busy even while the bar waits."""
    from PyQt5.QtCore import QRectF, Qt, QTimer
    from PyQt5.QtGui import QColor, QPainter, QPen

    class Spinner(qt_widgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedSize(size, size)
            self._angle = 0
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._turn)
            self._timer.start(16)

        def _turn(self):
            self._angle = (self._angle + 6) % 360
            self.update()

        def paintEvent(self, _event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            width = max(3.0, size / 9.0)
            rect = QRectF(width, width, size - 2 * width, size - 2 * width)
            track = QPen(QColor("#1E3A4F"), width)
            painter.setPen(track)
            painter.drawEllipse(rect)
            arc = QPen(QColor("#2F8FBF"), width)
            arc.setCapStyle(Qt.RoundCap)
            painter.setPen(arc)
            painter.drawArc(rect, -self._angle * 16, 100 * 16)

    return Spinner()


def add_loading_widgets(qt_widgets, layout, heading, cancelable=False, on_cancel=None):
    """Heading with a spinner, status, and bar. Returns (status_label, bar)."""
    header = qt_widgets.QHBoxLayout()
    header.addWidget(make_spinner(qt_widgets))
    title = qt_widgets.QLabel(heading)
    title.setStyleSheet("font-size: 20px; font-weight: 600;")
    title.setWordWrap(True)
    header.addWidget(title, 1)
    layout.addLayout(header)
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
        self.detail = qt_widgets.QLabel("This may take a minute.")
        self.detail.setStyleSheet("color: #9FB4C6;")
        layout.addWidget(self.detail)
        self.dialog.setWindowModality(Qt.NonModal)
        # Above the simulator windows, which open while this one is up.
        self.dialog.setWindowFlags(self.dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        self._bone_started = None

    def show(self):
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def close(self):
        self.dialog.close()

    def apply_line(self, line: str, now=None) -> bool:
        """Update from one log line. True when the process is ready."""
        import time

        if apply_stage(self.status, self.bar, line):
            self.ready = True
        fraction = bone_fraction(line)
        if fraction is not None:
            now = time.monotonic() if now is None else now
            self.detail.setText(_bone_detail(fraction, self._bone_started, now))
            if self._bone_started is None:
                self._bone_started = (now, fraction)
        return self.ready


def _bone_detail(fraction, started, now):
    """Percent and an estimate of the time left, from the upload rate so far."""
    pct = int(fraction * 100)
    if started is None or fraction >= 1.0:
        return f"{pct}%  ·  This may take a minute." if fraction < 1.0 else "Bone loaded. Opening the view…"
    start_time, start_fraction = started
    done = fraction - start_fraction
    elapsed = now - start_time
    if done <= 0.02 or elapsed <= 0.5:
        return f"{pct}%  ·  This may take a minute."
    left = elapsed / done * (1.0 - fraction)
    if left < 60:
        estimate = f"about {max(1, int(round(left / 5.0)) * 5)} s left"
    else:
        estimate = f"about {int(round(left / 60.0))} min left"
    return f"{pct}%  ·  {estimate}"


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
    assert stage_from_line("RTX bone ready")[0] == 100
    assert stage_from_line("RTX view ready")[0] < 100
    assert stage_from_line("Loading bone 0%")[0] == BONE_RANGE[0]
    assert stage_from_line("Loading bone 100%")[0] == BONE_RANGE[1]
    assert _bone_detail(0.5, (0.0, 0.0), 10.0) == "50%  ·  about 10 s left"
    assert _bone_detail(0.1, (0.0, 0.0), 40.0).endswith("about 6 min left")
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
