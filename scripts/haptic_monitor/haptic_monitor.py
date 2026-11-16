#!/usr/bin/env python3
"""Haptic monitor: a small window with live Touch data and a device health log.

Two sources, picked automatically:
  * Simulator running: AMBF owns the Touch, so the window reads what the drilling
    plugin publishes (/tmp/fivrs-rtx/haptic_state.json, haptic_monitor_feed.h):
    drill and suction forces (asked by contact / sent after the safety guard),
    physics rate, plus the device data below.
  * Simulator closed: the window starts bin/haptic_probe, which opens the Touch
    itself (force output off) and writes /tmp/fivrs-rtx/haptic_direct.json.
    It lets go of the device when the study GUI starts the simulator
    (release_device_for_sim), when an ambf_simulator process appears, or when
    this window closes.

Device data (both sources, from OpenHaptics, like the Touch Diagnostic app):
encoders and angles (turret / thigh / shin, yaw / pitch / roll), buttons and
inkwell, min / max / range since the window opened, motor temperature (the
driver's heat model, 0-100 %), calibration, servo loop rate, force output on/off,
OpenHaptics errors, and 10 s rolling graphs. Firmware version is not readable
through OpenHaptics on this box, so it is not shown.

Health log (always written while a device is read, window open or not):
~/.local/share/fivrs/haptic_health.log

Run:  python3 scripts/haptic_monitor/haptic_monitor.py
"""
from __future__ import annotations

import collections
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets

HERE = pathlib.Path(__file__).resolve().parent
STATE_DIR = pathlib.Path("/tmp/fivrs-rtx")
SIM_STATE = "haptic_state.json"
DIRECT_STATE = "haptic_direct.json"
ALIVE_FILE = "haptic_monitor.alive"
RELEASE_FILE = "haptic_probe.release"
LOCK_FILE = "haptic_probe.lock"
HEALTH_LOG = pathlib.Path.home() / ".local/share/fivrs/haptic_health.log"
PROBE = HERE / "bin" / "haptic_probe"
# Same OpenHaptics environment the launcher gives AMBF (GTDD_HOME device configs, libraries).
HAPTIC_ENV = HERE.parent / "setup_haptic_env.sh"
PROBE_SOURCES = (HERE / "haptic_probe.cpp",
                 HERE.parent.parent / "plugin/volumetric_drilling/haptic_device_probe.h")

HISTORY_S = 10.0
STALE_S = 1.5
SIM_GRACE_S = 3.0          # after the simulator quits, wait this long before opening the device
RELEASE_FRESH_S = 20.0     # a release request keeps the probe off this long
ARM = ("Turret", "Thigh", "Shin")
STYLUS = ("Yaw", "Pitch", "Roll")
BUTTONS = ("Stylus 1", "Stylus 2")
LOW_SERVO_HZ = 900


# ---------------------------------------------------------------- helpers (no Qt)

def simulator_running() -> bool:
    """An ambf_simulator process exists."""
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/comm") as comm:
                if comm.read().startswith("ambf_simulator"):
                    return True
        except OSError:
            continue
    return False


def read_state(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def is_fresh(state: dict | None, now: float, limit: float = STALE_S) -> bool:
    return bool(state) and now - float(state.get("wall_time", 0.0)) < limit


def device_locked(state_dir: pathlib.Path = STATE_DIR) -> bool:
    """True while a haptic_probe holds the Touch."""
    try:
        with open(state_dir / LOCK_FILE, "a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def release_device_for_sim(timeout_s: float = 2.0, state_dir: pathlib.Path = STATE_DIR,
                           settle_s: float = 1.0) -> bool:
    """Ask a running haptic_probe to let go of the Touch and wait until it has.
    The study GUI calls this right before it starts AMBF. True when the device is free.
    If the probe held it, wait settle_s more before AMBF opens it: one of five test
    launches (2026-11-14) segfaulted inside OpenHaptics' device open 0.1 s after the
    probe closed the Touch (not reproduced in three more)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / RELEASE_FILE).write_text(f"{time.time():.3f}\n")
    held = device_locked(state_dir)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not device_locked(state_dir):
            if held:
                time.sleep(settle_s)
            return True
        time.sleep(0.02)
    return not device_locked(state_dir)


def release_requested(now: float, state_dir: pathlib.Path = STATE_DIR) -> bool:
    try:
        return now - (state_dir / RELEASE_FILE).stat().st_mtime < RELEASE_FRESH_S
    except OSError:
        return False


def probe_needs_build() -> bool:
    if not PROBE.exists():
        return True
    built = PROBE.stat().st_mtime
    return any(src.exists() and src.stat().st_mtime > built for src in PROBE_SOURCES)


def build_probe() -> str:
    """Build bin/haptic_probe; returns '' or the compiler's error text."""
    result = subprocess.run(["bash", str(HERE / "build_probe.sh")], capture_output=True, text=True)
    return "" if result.returncode == 0 else (result.stderr or result.stdout)[-2000:]


def norm(vector) -> float:
    return sum(float(v) ** 2 for v in (vector or (0.0, 0.0, 0.0))) ** 0.5


class EncoderRange:
    """Min / max / range of the six encoders since the window opened (or Reset)."""

    def __init__(self):
        self.low = [None] * 6
        self.high = [None] * 6

    def update(self, encoders):
        for i, value in enumerate(encoders[:6]):
            self.low[i] = value if self.low[i] is None else min(self.low[i], value)
            self.high[i] = value if self.high[i] is None else max(self.high[i], value)

    def rows(self):
        return [(lo, hi, None if lo is None else hi - lo) for lo, hi in zip(self.low, self.high)]


def health_summary(state: dict | None, device: dict | None, source: str) -> list[tuple[str, str, str]]:
    """(label, value, level) for the health strip; level is ok / warn / bad / info."""
    if not state:
        return [("Device", "no data", "bad")]
    items = []
    if state.get("status") != "ok" or not device:
        items.append(("Device", state.get("error") or state.get("status", "no device"), "bad"))
        return items
    items.append(("Device", "connected", "ok"))
    hz = int(device.get("servo_hz", 0))
    items.append(("Servo", f"{hz} Hz", "ok" if hz >= LOW_SERVO_HZ else "bad"))
    calibration = device.get("calibration", "unknown")
    items.append(("Calibration", calibration, "ok" if calibration == "OK" else "warn"))
    if source == "direct":
        items.append(("Force output", "off (monitor sends none)", "info"))
    else:
        items.append(("Force output", "on" if device.get("force_output") else "OFF", "ok" if device.get("force_output") else "bad"))
    temps = device.get("temperature") or []
    if temps:
        hottest = max(temps)
        items.append(("Hottest motor", f"{hottest * 100:.0f}%", "ok" if hottest < 0.8 else "bad"))
    if source == "sim":
        physics = float(state.get("physics_hz", 0.0))
        items.append(("Physics", f"{physics:.0f} Hz", "ok" if physics >= 500 else "warn"))
        if state.get("safety_tripped"):
            items.append(("Safety", "TRIPPED (force faded)", "bad"))
        elif state.get("paused"):
            items.append(("Sim", "paused (no force)", "info"))
    return items


# ---------------------------------------------------------------- widgets

class RollingGraph(QtWidgets.QWidget):
    """A few lines over the last HISTORY_S seconds, autoscaled. Painted by hand
    so the window needs nothing beyond PyQt5."""

    COLORS = ("#3b78c4", "#e07b39", "#3a9a5b", "#b03a8a")

    def __init__(self, title: str, names, unit: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.names = list(names)
        self.unit = unit
        self.samples = collections.deque()
        self.setMinimumHeight(90)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def add(self, t: float, values):
        self.samples.append((t, [float(v) for v in values]))
        while self.samples and t - self.samples[0][0] > HISTORY_S:
            self.samples.popleft()

    def clear(self):
        self.samples.clear()
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 16, -4, -4)
        painter.fillRect(self.rect(), QtGui.QColor("#f4f5f7"))
        painter.setPen(QtGui.QColor("#cfd3da"))
        painter.drawRect(rect)
        painter.setPen(QtGui.QColor("#333"))
        font = painter.font()
        font.setPointSizeF(8.0)
        painter.setFont(font)
        painter.drawText(6, 12, f"{self.title} ({HISTORY_S:.0f} s)")
        x = 6 + painter.fontMetrics().horizontalAdvance(f"{self.title} ({HISTORY_S:.0f} s)") + 10
        for index, name in enumerate(self.names):
            painter.setPen(QtGui.QColor(self.COLORS[index % len(self.COLORS)]))
            painter.drawText(x, 12, name)
            x += painter.fontMetrics().horizontalAdvance(name) + 10
        if len(self.samples) < 2:
            return
        values = [v for _, row in self.samples for v in row]
        low, high = min(values), max(values)
        if high - low < 1e-6:
            low, high = low - 1.0, high + 1.0
        pad = (high - low) * 0.08
        low, high = low - pad, high + pad
        t_end = self.samples[-1][0]
        painter.setPen(QtGui.QColor("#777"))
        painter.drawText(rect.right() - 90, rect.top() + 11, f"{high:.4g} {self.unit}".strip())
        painter.drawText(rect.right() - 90, rect.bottom() - 2, f"{low:.4g} {self.unit}".strip())

        def point(t, v):
            px = rect.left() + rect.width() * (1.0 - (t_end - t) / HISTORY_S)
            py = rect.bottom() - rect.height() * (v - low) / (high - low)
            return QtCore.QPointF(px, py)

        for index in range(len(self.names)):
            path = QtGui.QPainterPath()
            for n, (t, row) in enumerate(self.samples):
                p = point(t, row[index])
                if n == 0:
                    path.moveTo(p)
                else:
                    path.lineTo(p)
            painter.setPen(QtGui.QPen(QtGui.QColor(self.COLORS[index % len(self.COLORS)]), 1.4))
            painter.drawPath(path)


def _table(rows, cols, row_labels, col_labels):
    table = QtWidgets.QTableWidget(rows, cols)
    table.setVerticalHeaderLabels(row_labels)
    table.setHorizontalHeaderLabels(col_labels)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.setFocusPolicy(QtCore.Qt.NoFocus)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    row_height = table.fontMetrics().height() + 6
    table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
    table.verticalHeader().setMinimumSectionSize(row_height)
    table.verticalHeader().setDefaultSectionSize(row_height)
    table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    for r in range(rows):
        for c in range(cols):
            item = QtWidgets.QTableWidgetItem("–")
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(r, c, item)
    header = table.horizontalHeader().sizeHint().height()
    table.setFixedHeight(header + rows * row_height + 2 * table.frameWidth() + 2)
    return table


def fit_table(table):
    """Once shown, size the table to its real header and rows (no clipped last row)."""
    height = table.horizontalHeader().height() + table.verticalHeader().length() + 2 * table.frameWidth()
    if height > 0 and table.height() != height:
        table.setFixedHeight(height)
        # Box layouts cache their minimum size; without this the panel keeps the
        # old one and the next widget is drawn over the table's last row.
        widget = table.parentWidget()
        while widget is not None:
            if widget.layout() is not None:
                widget.layout().invalidate()
            widget = widget.parentWidget()


class DevicePanel(QtWidgets.QWidget):
    """One Touch: info, encoders and angles, buttons, min/max, temperature, forces, graphs."""

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.range = EncoderRange()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        left = QtWidgets.QVBoxLayout()
        self.info = QtWidgets.QLabel("–")
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.info.setWordWrap(True)
        left.addWidget(self.info)
        self.joints = _table(2, 6, ["Encoders", "Angles"], [*ARM, *STYLUS])
        left.addWidget(self.joints)
        self.buttons = _table(1, 3, ["State"], [*BUTTONS, "Inkwell"])
        left.addWidget(self.buttons)
        self.minmax = _table(3, 6, ["Min", "Max", "Range"], [*ARM, *STYLUS])
        left.addWidget(self.minmax)
        reset = QtWidgets.QPushButton("Reset min / max")
        reset.clicked.connect(self.reset_range)
        left.addWidget(reset)

        left.addWidget(QtWidgets.QLabel("Motor temperature (OpenHaptics motor heat model, 0-100%):"))
        temps = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(temps)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(3)
        self.temp_bars = []
        for i, name in enumerate(ARM):
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            bar.setFixedHeight(bar.fontMetrics().height() + 4)
            grid.addWidget(QtWidgets.QLabel(name), i, 0)
            grid.addWidget(bar, i, 1)
            self.temp_bars.append(bar)
        left.addWidget(temps)
        self.forces = QtWidgets.QLabel("–")
        self.forces.setWordWrap(True)
        left.addWidget(self.forces)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        self.arm_graph = RollingGraph("Arm encoders", ARM)
        self.stylus_graph = RollingGraph("Stylus encoders", STYLUS)
        self.force_graph = RollingGraph("Force |F|", ["sent", "asked"] if role == "drill" else ["sent"], "N")
        for graph in (self.arm_graph, self.stylus_graph, self.force_graph):
            right.addWidget(graph)
        layout.addLayout(left, 0)
        layout.addLayout(right, 1)
        for i in range(left.count()):
            widget = left.itemAt(i).widget()
            if widget is not None:
                widget.setFixedWidth(450)
        self.setMinimumWidth(800)

    def reset_range(self):
        self.range = EncoderRange()

    def clear_graphs(self):
        for graph in (self.arm_graph, self.stylus_graph, self.force_graph):
            graph.clear()

    def show_device(self, device: dict, state: dict, source: str, now: float):
        info = (f"<b>{device.get('model') or '?'}</b> · SN {device.get('serial') or '?'} · "
                f"driver {device.get('driver') or '?'} · calibration <b>{device.get('calibration')}</b> · "
                f"servo {device.get('servo_hz', 0)} Hz · max {device.get('max_force_n', 0):.1f} N "
                f"({device.get('max_continuous_force_n', 0):.2f} N continuous) · firmware: not readable")
        self.info.setText(info)
        encoders = list(device.get("encoders") or [0] * 6)
        joints = list(device.get("joint_deg") or [0] * 3)
        gimbal = list(device.get("gimbal_deg") or [0] * 3)
        angles = joints[:3] + gimbal[:3]
        for c in range(6):
            self.joints.item(0, c).setText(str(encoders[c]))
            self.joints.item(1, c).setText(f"{angles[c]:.1f}°")
        buttons = int(device.get("buttons", 0))
        states = [bool(buttons & 1), bool(buttons & 2), bool(device.get("inkwell"))]
        for c, on in enumerate(states):
            item = self.buttons.item(0, c)
            item.setText("ON" if on else "off")
            item.setBackground(QtGui.QColor("#7a7a7a" if on else "#ffffff"))
            item.setForeground(QtGui.QColor("#ffffff" if on else "#000000"))
        self.range.update(encoders)
        for c, (lo, hi, span) in enumerate(self.range.rows()):
            for r, value in enumerate((lo, hi, span)):
                self.minmax.item(r, c).setText("–" if value is None else str(value))
        temps = list(device.get("temperature") or [])
        for i, bar in enumerate(self.temp_bars):
            if i < len(temps):
                bar.setEnabled(True)
                bar.setValue(int(round(temps[i] * 100)))
            else:
                bar.setEnabled(False)
                bar.setFormat("not reported")
        device_force = norm(device.get("device_force_n"))
        if source == "sim" and self.role == "drill":
            asked, sent = norm(state.get("drill_requested_n")), norm(state.get("drill_sent_n"))
            text = (f"Drill force: asked by contact <b>{asked:.2f} N</b>, sent <b>{sent:.2f} N</b>, "
                    f"at the device {device_force:.2f} N · "
                    f"{'in contact' if state.get('drill_in_contact') else 'no contact'} · "
                    f"drill {'ON' if state.get('drill_on') else 'off'} · safety trips {state.get('safety_trips', 0)}")
            self.force_graph.add(now, [sent, asked])
        elif source == "sim":
            sent = norm(state.get("suction_sent_n"))
            text = (f"Suction (frazier_suction) force: sent <b>{sent:.2f} N</b>, at the device {device_force:.2f} N"
                    + ("" if state.get("suction_tracked") else " · suction not tracked"))
            self.force_graph.add(now, [sent])
        else:
            text = "Simulator closed: no forces. The monitor opened the device with force output off."
            self.force_graph.add(now, [device_force, 0.0] if self.role == "drill" else [device_force])
        self.forces.setText(text)
        for table in (self.joints, self.buttons, self.minmax):
            fit_table(table)
        self.arm_graph.add(now, encoders[:3])
        self.stylus_graph.add(now, encoders[3:6])
        for graph in (self.arm_graph, self.stylus_graph, self.force_graph):
            graph.update()


class HapticMonitor(QtWidgets.QWidget):
    def __init__(self, state_dir: pathlib.Path = STATE_DIR, health_log: pathlib.Path = HEALTH_LOG,
                 start_probe: bool = True, sim_running=simulator_running, parent=None):
        super().__init__(parent)
        self.state_dir = pathlib.Path(state_dir)
        self.health_log = pathlib.Path(health_log)
        self.start_probe_enabled = start_probe
        self.sim_running = sim_running
        self.probe: subprocess.Popen | None = None
        self.probe_error = ""
        self.next_probe_try = 0.0
        self.sim_seen_at = 0.0
        self.source = "none"
        self.log_offset = 0
        self.setWindowTitle("Haptic monitor")
        self.resize(1000, 740)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.mode = QtWidgets.QLabel("–")
        self.mode.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.mode)
        self.health = QtWidgets.QLabel("–")
        self.health.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.health)
        self.tabs = QtWidgets.QTabWidget()
        self.panels: dict[str, DevicePanel] = {}
        layout.addWidget(self.tabs, 1)

        log_box = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_title = QtWidgets.QLabel(f"Health log (always written while a Touch is read): {self.health_log}")
        log_title.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        log_layout.addWidget(log_title)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setFixedHeight(120)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_box)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)
        self.slow_timer = QtCore.QTimer(self)
        self.slow_timer.timeout.connect(self.slow_tick)
        self.slow_timer.start(1000)
        self.slow_tick()

    # --- source

    def _probe_running(self) -> bool:
        return self.probe is not None and self.probe.poll() is None

    def stop_probe(self):
        if self._probe_running():
            self.probe.terminate()
            try:
                self.probe.wait(2.0)
            except subprocess.TimeoutExpired:
                self.probe.kill()
        self.probe = None

    def _start_probe(self, now: float):
        if not self.start_probe_enabled or now < self.next_probe_try:
            return
        self.next_probe_try = now + 3.0
        if probe_needs_build():
            self.probe_error = build_probe()
            if self.probe_error:
                return
        try:
            (self.state_dir / RELEASE_FILE).unlink(missing_ok=True)
            command = [str(PROBE)]
            if HAPTIC_ENV.exists():
                command = ["bash", "-c", 'source "$1"; exec "$2"', "haptic_probe", str(HAPTIC_ENV), str(PROBE)]
            self.probe = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.probe_error = ""
        except OSError as error:
            self.probe_error = str(error)

    def slow_tick(self):
        now = time.time()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ALIVE_FILE).write_text(f"{now:.3f}\n")
        sim = self.sim_running()
        if sim:
            self.sim_seen_at = now
            self.stop_probe()
        elif (now - self.sim_seen_at > SIM_GRACE_S and not release_requested(now, self.state_dir)
              and not self._probe_running()):
            self._start_probe(now)
        if self.probe is not None and self.probe.poll() is not None:
            err = (self.probe.stderr.read() or b"").decode(errors="replace").strip() if self.probe.stderr else ""
            if err:
                self.probe_error = err.splitlines()[-1]
            self.probe = None
        self._read_log()

    def _read_log(self):
        try:
            size = self.health_log.stat().st_size
        except OSError:
            return
        if size < self.log_offset:
            self.log_offset = 0
        partial = False
        if self.log_offset == 0 and size > 20000:
            self.log_offset = size - 20000
            partial = True
        if size == self.log_offset:
            return
        with open(self.health_log, "rb") as handle:
            handle.seek(self.log_offset)
            chunk = handle.read()
        self.log_offset += len(chunk)
        lines = chunk.decode(errors="replace").splitlines()
        if partial:
            lines = lines[1:]
        for line in lines:
            self.log_view.appendPlainText(line)

    def current_state(self, now: float) -> tuple[str, dict | None]:
        sim_state = read_state(self.state_dir / SIM_STATE)
        if self.sim_running() or now - self.sim_seen_at <= SIM_GRACE_S:
            return "sim", sim_state if is_fresh(sim_state, now) else None
        if is_fresh(sim_state, now):
            return "sim", sim_state
        direct = read_state(self.state_dir / DIRECT_STATE)
        return "direct", direct if is_fresh(direct, now) else None

    # --- display

    def _panel(self, role: str) -> DevicePanel:
        if role not in self.panels:
            panel = DevicePanel(role)
            self.panels[role] = panel
            self.tabs.addTab(panel, f"{role.capitalize()} Touch")
        return self.panels[role]

    def _fit_window(self):
        """Panels and their final table sizes arrive after the window is shown:
        grow it so nothing overlaps."""
        self.layout().activate()
        need = self.layout().totalMinimumSize().height()
        if need != self.minimumHeight():
            self.setMinimumHeight(need)
            if self.height() < need:
                self.resize(self.width(), need)

    def tick(self):
        now = time.time()
        source, state = self.current_state(now)
        if source != self.source:
            self.source = source
            for panel in self.panels.values():
                panel.clear_graphs()
        if source == "sim":
            mode = "Simulator running: reading the simulator's feed (AMBF owns the Touch)"
            if state is None:
                mode += " · no feed yet (simulator starting, or a plugin built before the monitor)"
        else:
            mode = "Simulator closed: the monitor reads the Touch directly (force output off)"
            if state is None:
                mode += " · " + (self.probe_error or "opening the device…")
        if state and state.get("status") == "ok" and state.get("error"):
            mode += " · " + state["error"]
        self.mode.setText(mode)
        devices = (state or {}).get("devices") or []
        first = devices[0] if devices else None
        parts = []
        for label, value, level in health_summary(state, first, source):
            color = {"ok": "#2e7d32", "warn": "#b26a00", "bad": "#c62828"}.get(level, "#455a64")
            parts.append(f"{label}: <b style='color:{color}'>{value}</b>")
        for extra in devices[1:]:
            hz = extra.get("servo_hz", 0)
            parts.append(f"{extra.get('role')}: <b>{hz} Hz</b>, force output {'on' if extra.get('force_output') else 'off'}")
        self.health.setText(" &nbsp;·&nbsp; ".join(parts))
        for device in devices:
            if device.get("valid"):
                self._panel(device.get("role", "drill")).show_device(device, state, source, now)
        self._fit_window()

    def closeEvent(self, event):
        self.timer.stop()
        self.slow_timer.stop()
        self.stop_probe()
        try:
            (self.state_dir / ALIVE_FILE).unlink(missing_ok=True)
        except OSError:
            pass
        super().closeEvent(event)


def apply_style(app: QtWidgets.QApplication):
    """Compact: one 9 pt font for the whole window."""
    font = app.font()
    font.setPointSizeF(9.0)
    app.setFont(font)


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_style(app)
    app.setApplicationName("Haptic monitor")
    window = HapticMonitor()
    window.show()
    for sig in ("SIGTERM", "SIGINT"):
        import signal
        signal.signal(getattr(signal, sig), lambda *_: window.close())
    # Let Python see signals while Qt runs.
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
