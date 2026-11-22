#!/usr/bin/env python3
"""Haptics tuner: drill a bone cube in the RTX view and change the force live.

    python3 scripts/haptics_tuner/haptics_tuner.py

Build first with scripts/haptics_tuner/build.sh. Settings reach the physics loop
through shared memory, so a change is felt on the next tick.
"""
import json
import os
import pathlib
import random
import shlex
import subprocess
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_bone_cube  # noqa: E402
import tuning_shm as shm  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKSPACE = pathlib.Path(os.environ.get("ROS_AMBF_WS", pathlib.Path.home() / "ros_ambf_ws"))
PRESET_FILE = pathlib.Path(__file__).with_name("presets.json")
RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
PLUGIN = REPO / ".haptics" / "install" / "volumetric_plugin" / "lib" / "libvolumetric_drilling.so"
FOOTPEDAL = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
VOXEL_MM3 = make_bone_cube.VOXEL_MM ** 3
PHYSICS_HZ = 1000.0
PLOT_SECONDS = 4.0


def find_rtx_python():
    for root in (REPO, WORKSPACE / "src" / "volumetric_drilling"):
        python = root / "middle_ear_newton" / ".venv" / "bin" / "python"
        if python.is_file():
            return python
    return None


def simulator_running():
    return subprocess.run(["pgrep", "-x", "ambf_simulator"], stdout=subprocess.DEVNULL).returncode == 0


def ambf_command():
    adf = shlex.quote(str(make_bone_cube.ADF))
    return (
        "set +u; source /opt/ros/jazzy/setup.bash; "
        f"source {shlex.quote(str(WORKSPACE / 'install' / 'setup.bash'))}; "
        f"source {shlex.quote(str(REPO / '.haptics' / 'install' / 'local_setup.bash'))}; "
        f"source {shlex.quote(str(REPO / 'scripts' / 'setup_haptic_env.sh'))}; "
        "export FIVRS_HAPTIC_TUNER=1 FIVRS_SUCTION_DEVICE=0 FIVRS_IRRIGATION=0; "
        f"exec {shlex.quote(str(WORKSPACE / 'install' / 'AMBF' / 'bin' / 'ambf_simulator'))} "
        f"--launch_file {shlex.quote(str(REPO / 'launch_haptics.yaml'))} -l 0,7 -a {adf} --nt 2 "
        f"--fp {shlex.quote(FOOTPEDAL)} --sb 4 --anatomy_adf {adf} "
        "--session_user 'Haptics tuner' --session_anatomy 'Bone cube' --startup_view default --rtx_view 1"
    )


def rtx_command(python):
    return (
        "set +u; source /opt/ros/jazzy/setup.bash; "
        f"source {shlex.quote(str(WORKSPACE / 'install' / 'setup.bash'))}; "
        f"source {shlex.quote(str(REPO / '.haptics' / 'install' / 'local_setup.bash'))}; "
        "unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_PLATFORMTHEME QTDIR "
        "__NV_PRIME_RENDER_OFFLOAD __VK_LAYER_NV_optimus; "
        'export DISPLAY="${DISPLAY:-:1}" PYTHONUNBUFFERED=1 '
        '__GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}" '
        'VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" '
        "FIVRS_IRRIGATION=0 FIVRS_SUCTION_DEVICE=0 FIVRS_DLSS=performance; "
        f"exec {shlex.quote(str(python))} {shlex.quote(str(REPO / 'scripts' / 'rtx_view' / 'fivrs_rtx_view.py'))} "
        f"--anatomy-adf {shlex.quote(str(make_bone_cube.ADF))}"
    )


class Chart(QtWidgets.QWidget):
    """A scrolling trace, drawn as a min/max band per pixel so a fast hum stays visible."""

    def __init__(self, title, unit, color, maximum, parent=None):
        super().__init__(parent)
        self.title, self.unit, self.color, self.maximum = title, unit, QtGui.QColor(color), maximum
        self.values = np.zeros(0)
        self.markers = []
        self.setMinimumHeight(112)

    def show_values(self, values, markers=()):
        self.values = values
        self.markers = list(markers)
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        palette = self.palette()
        rect = self.rect().adjusted(46, 22, -8, -6)
        painter.fillRect(self.rect(), palette.base())
        painter.setPen(palette.text().color())
        now = float(self.values[-1]) if len(self.values) else 0.0
        painter.drawText(8, 16, f"{self.title}: {now:.2f} {self.unit}")
        painter.setPen(palette.mid().color())
        painter.drawRect(rect)
        painter.setPen(palette.text().color())
        painter.drawText(4, rect.top() + 10, f"{self.maximum:g}")
        painter.drawText(4, rect.bottom(), "0")

        def y_at(value):
            share = min(max(value / self.maximum, 0.0), 1.0)
            return int(rect.bottom() - share * rect.height())

        for value, label in self.markers:
            pen = QtGui.QPen(palette.mid().color(), 1, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(rect.left(), y_at(value), rect.right(), y_at(value))
            painter.setPen(palette.text().color())
            painter.drawText(rect.left() + 4, y_at(value) - 2, label)

        if len(self.values) > 1 and rect.width() > 2:
            painter.setPen(QtGui.QPen(self.color, 1))
            samples = int(PLOT_SECONDS * PHYSICS_HZ)
            offset = samples - len(self.values)
            edges = np.linspace(0, samples, rect.width() + 1).astype(int)
            for x in range(rect.width()):
                low, high = max(edges[x] - offset, 0), edges[x + 1] - offset
                if high <= low:
                    continue
                part = self.values[low:high]
                painter.drawLine(rect.left() + x, y_at(float(part.min())), rect.left() + x, y_at(float(part.max())))
        painter.end()


class Control(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        self.setToolTip(spec.tip)
        if spec.kind == "toggle":
            self.box = QtWidgets.QCheckBox(spec.label)
            self.box.toggled.connect(lambda _on: self.changed.emit())
            layout.addWidget(self.box)
            return
        name = QtWidgets.QLabel(spec.label)
        name.setMinimumWidth(150)
        layout.addWidget(name)
        if spec.kind == "choice":
            self.combo = QtWidgets.QComboBox()
            self.combo.addItems(spec.choices)
            self.combo.currentIndexChanged.connect(lambda _index: self.changed.emit())
            layout.addWidget(self.combo, 1)
            return
        self.steps = 1000
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.steps)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(spec.low, spec.high)
        self.spin.setDecimals(spec.decimals)
        self.spin.setSingleStep(10 ** -spec.decimals * (10 if spec.decimals > 1 else 1))
        self.spin.setSuffix(f" {spec.unit}" if spec.unit else "")
        self.spin.setMinimumWidth(104)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

    def _from_slider(self, position):
        value = self.spec.low + (self.spec.high - self.spec.low) * position / self.steps
        blocked = self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(blocked)
        self.changed.emit()

    def _from_spin(self, value):
        share = (value - self.spec.low) / (self.spec.high - self.spec.low)
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(round(share * self.steps)))
        self.slider.blockSignals(blocked)
        self.changed.emit()

    def value(self):
        if self.spec.kind == "toggle":
            return 1.0 if self.box.isChecked() else 0.0
        if self.spec.kind == "choice":
            return float(self.combo.currentIndex())
        return float(self.spin.value())

    def set_value(self, value):
        widgets = [w for w in (getattr(self, n, None) for n in ("box", "combo", "slider", "spin")) if w is not None]
        states = [w.blockSignals(True) for w in widgets]
        if self.spec.kind == "toggle":
            self.box.setChecked(value >= 0.5)
        elif self.spec.kind == "choice":
            self.combo.setCurrentIndex(int(value))
        else:
            self.spin.setValue(value)
            share = (value - self.spec.low) / (self.spec.high - self.spec.low)
            self.slider.setValue(int(round(share * self.steps)))
        for widget, state in zip(widgets, states):
            widget.blockSignals(state)


class Tuner(QtWidgets.QMainWindow):
    def __init__(self, link=None):
        super().__init__()
        self.setWindowTitle("Haptics Tuner")
        self.link = link or shm.Link()
        self.controls = {}
        self.slots = {"A": shm.stock(), "B": shm.clean(shm.BUILT_IN["Firm cut, strong hum (starting point)"])}
        self.blind = None  # {"1": "A", "2": "B"} during a blind comparison
        self.user_presets = self._load_presets()
        self.ambf = QtCore.QProcess(self)
        self.rtx = QtCore.QProcess(self)
        self.ambf.finished.connect(self._sim_finished)
        self.rtx.finished.connect(self._update_buttons)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.addLayout(self._build_top_row())
        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, 1)
        self.settings_area = QtWidgets.QScrollArea()
        self.settings_area.setWidgetResizable(True)
        self.settings_area.setWidget(self._build_settings())
        self.settings_area.setMinimumWidth(520)
        self.blind_note = QtWidgets.QLabel("Blind comparison: settings are hidden.\nPlay 1 and 2, drill, then press Reveal.")
        self.blind_note.setAlignment(QtCore.Qt.AlignCenter)
        self.blind_note.setMinimumWidth(520)
        self.blind_note.hide()
        body.addWidget(self.settings_area, 5)
        body.addWidget(self.blind_note, 5)
        body.addLayout(self._build_live_panel(), 6)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self._apply(shm.stock())
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(40)
        self._update_buttons()
        self.resize(1240, 820)

    # ---- layout ----
    def _build_top_row(self):
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start simulator")
        self.start_button.clicked.connect(self.start_sim)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_sim)
        self.restore_button = QtWidgets.QPushButton("Restore cube")
        self.restore_button.clicked.connect(self.restore_cube)
        for button in (self.start_button, self.stop_button, self.restore_button):
            row.addWidget(button)
        row.addSpacing(18)
        row.addWidget(QtWidgets.QLabel("Preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setMinimumWidth(270)
        self._fill_presets()
        self.preset_combo.activated[str].connect(self.load_preset)
        row.addWidget(self.preset_combo)
        save = QtWidgets.QPushButton("Save as…")
        save.clicked.connect(self.save_preset)
        row.addWidget(save)
        row.addStretch(1)
        return row

    def _build_settings(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        groups = {}
        for spec in shm.SPECS:
            if spec.group not in groups:
                box = QtWidgets.QGroupBox(spec.group)
                groups[spec.group] = QtWidgets.QVBoxLayout(box)
                layout.addWidget(box)
            control = Control(spec)
            control.changed.connect(self._control_changed)
            groups[spec.group].addWidget(control)
            self.controls[spec.key] = control
        layout.addStretch(1)
        return panel

    def _build_live_panel(self):
        column = QtWidgets.QVBoxLayout()
        self.force_chart = Chart("Burr push on bone", "N", "#2a78c8", 3.3)
        self.hum_chart = Chart("Hum", "N", "#d9822b", 0.6)
        self.cut_chart = Chart("Bone removal", "mm³/s", "#3f9b57", 100.0)
        for chart in (self.force_chart, self.hum_chart, self.cut_chart):
            column.addWidget(chart, 1)
        self.live_label = QtWidgets.QLabel("Simulator not running")
        column.addWidget(self.live_label)
        self.motor_label = QtWidgets.QLabel("Touch motors: simulator not running")
        column.addWidget(self.motor_label)

        compare = QtWidgets.QGroupBox("A/B comparison")
        grid = QtWidgets.QGridLayout(compare)
        self.play_a = QtWidgets.QPushButton("Play A")
        self.play_b = QtWidgets.QPushButton("Play B")
        self.play_a.clicked.connect(lambda: self.play("1" if self.blind else "A"))
        self.play_b.clicked.connect(lambda: self.play("2" if self.blind else "B"))
        self.store_a = QtWidgets.QPushButton("Store current as A")
        self.store_b = QtWidgets.QPushButton("Store current as B")
        self.store_a.clicked.connect(lambda: self.store("A"))
        self.store_b.clicked.connect(lambda: self.store("B"))
        self.blind_button = QtWidgets.QPushButton("Start blind comparison")
        self.blind_button.clicked.connect(self.toggle_blind)
        grid.addWidget(self.play_a, 0, 0)
        grid.addWidget(self.play_b, 0, 1)
        grid.addWidget(self.store_a, 1, 0)
        grid.addWidget(self.store_b, 1, 1)
        grid.addWidget(self.blind_button, 2, 0, 1, 2)
        column.addWidget(compare)
        return column

    # ---- settings ----
    def current(self):
        return {key: control.value() for key, control in self.controls.items()}

    def _apply(self, values):
        values = self.link.write(values)
        for key, control in self.controls.items():
            control.set_value(values[key])
        self._sync_enabled(values)

    def _control_changed(self):
        values = self.link.write(self.current())
        self.controls["cut_full_n"].set_value(values["cut_full_n"])
        self._sync_enabled(values)

    def _sync_enabled(self, values):
        gated = values["cut_gate"] >= 0.5
        for key in ("cut_start_n", "cut_full_n", "cut_curve", "rate_1", "rate_2", "rate_4", "rate_6"):
            self.controls[key].setEnabled(gated)

    # ---- presets ----
    def _load_presets(self):
        try:
            data = json.loads(PRESET_FILE.read_text())
        except (OSError, ValueError):
            return {}
        return {name: shm.clean(values) for name, values in data.items() if isinstance(values, dict)}

    def _fill_presets(self):
        self.preset_combo.clear()
        self.preset_combo.addItems(list(shm.BUILT_IN) + sorted(self.user_presets))

    def load_preset(self, name):
        values = self.user_presets.get(name) or shm.BUILT_IN.get(name)
        if values:
            self._apply(values)
            self.preset_combo.setCurrentText(name)
            self._say(f"Loaded {name}.")

    def save_preset(self, _checked=False, name=None):
        if name is None:
            name, accepted = QtWidgets.QInputDialog.getText(self, "Save preset", "Name")
            if not accepted:
                return
        name = name.strip()
        if not name or name in shm.BUILT_IN:
            self._say("Choose a name that is not a built-in preset.")
            return
        self.user_presets[name] = self.current()
        temporary = PRESET_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.user_presets, indent=2, sort_keys=True) + "\n")
        temporary.replace(PRESET_FILE)
        self._fill_presets()
        self.preset_combo.setCurrentText(name)
        self._say(f"Saved {name}.")

    # ---- A/B ----
    def store(self, slot):
        self.slots[slot] = self.current()
        self._say(f"Stored the current settings as {slot}.")

    def play(self, label):
        slot = self.blind[label] if self.blind else label
        self._apply(self.slots[slot])
        self.play_a.setDown(label in ("A", "1"))
        self.play_b.setDown(label in ("B", "2"))
        self._say(f"Playing {label}.")

    def toggle_blind(self):
        if self.blind:
            self._say(f"1 was {self.blind['1']}, 2 was {self.blind['2']}.")
            self.blind = None
        else:
            order = ["A", "B"]
            random.shuffle(order)
            self.blind = {"1": order[0], "2": order[1]}
            self._apply(self.slots[order[0]])
            self.play_a.setDown(True)
            self.play_b.setDown(False)
            self._say("Blind comparison started. Playing 1.")
        blind = self.blind is not None
        self.settings_area.setVisible(not blind)
        self.blind_note.setVisible(blind)
        self.preset_combo.setEnabled(not blind)
        self.store_a.setEnabled(not blind)
        self.store_b.setEnabled(not blind)
        self.play_a.setText("Play 1" if blind else "Play A")
        self.play_b.setText("Play 2" if blind else "Play B")
        self.blind_button.setText("Reveal" if blind else "Start blind comparison")

    # ---- simulator ----
    def _sim_alive(self):
        return self.ambf.state() != QtCore.QProcess.NotRunning

    def start_sim(self):
        if self._sim_alive():
            return
        if simulator_running():
            self._say("Another simulator is running. Close it first: only one can hold the Touch.")
            return
        if not PLUGIN.is_file():
            self._say("The tuner's plugin is not built. Run scripts/haptics_tuner/build.sh.")
            return
        python = find_rtx_python()
        if python is None:
            self._say("The RTX Python environment (middle_ear_newton/.venv) was not found.")
            return
        if not make_bone_cube.ready():
            self._say("Making the bone cube…")
            QtWidgets.QApplication.processEvents()
            make_bone_cube.build()
        RUNTIME.mkdir(parents=True, exist_ok=True)
        # The haptic monitor holds the Touch while the simulator is closed.
        if subprocess.run(["pgrep", "-x", "haptic_probe"], stdout=subprocess.DEVNULL).returncode == 0:
            (RUNTIME / "haptic_probe.release").write_text("release\n")
            QtCore.QThread.msleep(1000)
        # Same as the main simulator: force is on from launch. libhdPhantom's warm-up
        # and motor self-check run by themselves; their result is only displayed.
        self.link.set_force_enable(True)
        (RUNTIME / "haptic_selfcheck.json").unlink(missing_ok=True)
        self.link.write(self.current())
        for process, command, log in ((self.ambf, ambf_command(), "haptics_ambf.log"),
                                      (self.rtx, rtx_command(python), "haptics_rtx.log")):
            process.setWorkingDirectory(str(REPO))
            process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            process.setStandardOutputFile(str(RUNTIME / log))
            process.start("bash", ["-c", command])
        self._say("Starting. Hold the right pedal to run the drill.")
        self._update_buttons()

    def stop_sim(self):
        self.link.set_force_enable(False)
        for process in (self.rtx, self.ambf):
            if process.state() != QtCore.QProcess.NotRunning:
                process.terminate()
                if not process.waitForFinished(9000):
                    process.kill()
                    process.waitForFinished(2000)
        self._update_buttons()

    def _sim_finished(self, *_args):
        self.link.set_force_enable(False)
        self.motor_label.setText("Touch motors: simulator not running")
        if self.rtx.state() != QtCore.QProcess.NotRunning:
            self.rtx.terminate()
        self._say(f"Simulator closed. Log: {RUNTIME / 'haptics_ambf.log'}")
        self._update_buttons()

    def restore_cube(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        (RUNTIME / "restore.request").write_text("restore\n")
        self._say("Cube restored.")

    def _update_buttons(self, *_args):
        alive = self._sim_alive()
        self.start_button.setEnabled(not alive)
        self.stop_button.setEnabled(alive)
        self.restore_button.setEnabled(alive)

    def _say(self, text):
        self.status.setText(text)

    # ---- live plots ----
    def refresh(self):
        if self._sim_alive():
            _checked, text = shm.motor_state(RUNTIME / "haptic_selfcheck.json", int(self.ambf.processId()))
            self.motor_label.setText(text)
        rows = self.link.recent(int(PLOT_SECONDS * PHYSICS_HZ))
        if not len(rows):
            return
        force, hum, removed = rows[:, 1], rows[:, 2], rows[:, 5]
        window = 100
        kernel = np.ones(window) / window
        rate = np.convolve(removed, kernel, mode="same") * VOXEL_MM3 * PHYSICS_HZ
        values = self.current()
        markers = []
        if values["cut_gate"] >= 0.5:
            markers = [(values["cut_start_n"], "starts"), (values["cut_full_n"], "full")]
        self.force_chart.show_values(force, markers)
        self.hum_chart.show_values(hum)
        self.cut_chart.maximum = max(100.0, float(rate.max()) * 1.15)
        self.cut_chart.show_values(rate)
        flags = int(rows[-1, 6])
        state = [
            "drill ON" if flags & shm.FLAG_DRILL_ON else "drill off",
            "touching bone" if flags & shm.FLAG_CONTACT else "free air",
            f"burr {rows[-1, 7]:.0f} mm",
        ]
        if flags & shm.FLAG_FORCE_HELD:
            state.append("force held at zero")
        if flags & shm.FLAG_LIMITED:
            state.append("sustained force limited")
        if flags & shm.FLAG_TRIPPED:
            state.append("SAFETY GUARD TRIPPED: lift off the bone")
        self.live_label.setText("  |  ".join(state))

    def closeEvent(self, event):
        self.stop_sim()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Tuner()
    window.show()
    if "--start" in sys.argv:
        QtCore.QTimer.singleShot(300, window.start_sim)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
