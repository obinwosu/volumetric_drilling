"""Launcher tab "UR5": record expert drillings of the cadaveric specimens.

Lists the specimens of scripts/expert_drilling/specimens.yaml, takes the expert's first and
last name, and offers the Main tab's launch options. Defaults: Fastest rendering, drill only.

Every session is recorded. Start simulator keeps the drill locked until the recorder holds the
starting volume, so no cut is missed. Closing the RTX view (or the simulator) ends the session and
asks Save or Discard; Discard moves the session to the system Trash.

Each recording is one folder:
    <recording base>/expert_drillings/<Last_First>/<specimen>/session_NNN/
and one row in <recording base>/expert_drillings/sessions.csv. The same person can drill the
same specimen any number of times; the number counts that person's sessions on that specimen.

"Development session" (on while the pipeline is being built) saves to expert_drillings_dev
instead, with its own numbering and sessions.csv, so test runs never mix with expert data.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import time

import numpy as np
import yaml
from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, Qt, QTimer
from PyQt5.QtGui import QPixmap

import gui_setup
import ur5_ct_dialog

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = REPO / "scripts" / "expert_drilling" / "specimens.yaml"
RECORDER = REPO / "scripts" / "expert_drilling" / "expert_recorder.py"
RECORDER_PYTHON = REPO / "feedback_system" / ".venv" / "bin" / "python"
SETTING = "expert_drilling"
DATA_FOLDER = "expert_drillings"
DEV_DATA_FOLDER = "expert_drillings_dev"
REVIEW_APP = REPO / "scripts" / "session_review" / "review_app.py"
VIDEO_HZ = 30
DRILL_LOCK = pathlib.Path("/tmp/fivrs-rtx/study_drill_lock")  # the plugin stops the drill while it exists
LOCK_TEXT = "ur5\n"
RECORDER_START_TIMEOUT_S = 45.0
NO_RTX_SETTLE_S = 4.0  # without the RTX view there is no "ready" signal
CSV_COLUMNS = ("session_id", "last_name", "first_name", "specimen", "session_number", "started", "video", "folder")
DEFAULTS = {
    "rtx": True, "openxr": False, "openvr": False, "stream_stereo": False, "stream_depth": False,
    "video": True, "pupil": False, "burr": 4, "view": "default", "development": True,
    "preset": "fastest", **gui_setup.PRESETS["fastest"], "suction": False,
}


def load_specimens(path=MANIFEST) -> list[dict]:
    if not pathlib.Path(path).is_file():
        return []
    return (yaml.safe_load(pathlib.Path(path).read_text()) or {}).get("specimens") or []


def name_part(text: str) -> str:
    """A name as a folder part: letters (any language), digits, hyphens."""
    cleaned = re.sub(r"[^\w-]+", "-", text.strip(), flags=re.UNICODE).strip("-_")
    return cleaned


def person_folder(first: str, last: str) -> str:
    return "_".join(part for part in (name_part(last), name_part(first)) if part)


def specimen_folder(specimen: dict) -> str:
    return name_part(specimen["name"]).replace("-", "_")


def create_session_folder(root, first: str, last: str, specimen: dict) -> tuple[pathlib.Path, int]:
    """The next session of this person on this specimen. Case differences are the same person."""
    root = pathlib.Path(root)
    wanted = person_folder(first, last)
    root.mkdir(parents=True, exist_ok=True)
    person = next((p for p in root.iterdir() if p.is_dir() and p.name.casefold() == wanted.casefold()), root / wanted)
    parent = person / specimen_folder(specimen)
    parent.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for m in (re.fullmatch(r"session_(\d+)", p.name) for p in parent.iterdir()) if m]
    number = max(numbers, default=0) + 1
    while True:
        folder = parent / f"session_{number:03d}"
        try:
            folder.mkdir()  # fails if it exists, so a number is never reused
            return folder, number
        except FileExistsError:
            number += 1


def append_session_row(root, row: dict) -> None:
    path = pathlib.Path(root) / "sessions.csv"
    new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def move_to_trash(folder) -> bool:
    """The desktop's own Trash (QFile.moveToTrash put folders in the working directory)."""
    try:
        done = subprocess.run(["gio", "trash", str(folder)], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and not pathlib.Path(folder).exists()


def discard_session(folder) -> str:
    """Move a session to the system Trash. Never deletes outright: if there is no Trash, it is
    renamed beside the others (discarded_...), which the session lists ignore."""
    folder = pathlib.Path(folder)
    where = "the Trash"
    if not move_to_trash(folder):
        target = folder.with_name(f"discarded_{folder.name}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        folder.rename(target)
        where = str(target)
    for parent in (folder.parent, folder.parent.parent):  # no empty person / specimen folders
        try:
            parent.rmdir()
        except OSError:
            break
    return where


def volume_pose(adf: pathlib.Path) -> dict:
    """World pose and size of the volume, from its ADF (the volume never moves)."""
    from scipy.spatial.transform import Rotation

    doc = yaml.safe_load(adf.read_text())
    block = doc[doc["volumes"][0]]
    location = block["location"]
    rpy = location["orientation"]
    rotation = Rotation.from_euler("ZYX", [rpy["y"], rpy["p"], rpy["r"]]).as_matrix()
    return {
        "p_world_volume": [float(location["position"][a]) for a in "xyz"],
        "R_world_volume": rotation.tolist(),
        "rpy_world_volume": [float(rpy[a]) for a in "rpy"],
        "dimensions_sim": [float(block["dimensions"][a]) for a in "xyz"],
        "index_to_local": "local = index * dimensions_sim / voxel_shape - dimensions_sim / 2",
    }


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


class ExpertDrillingPanel(QtWidgets.QWidget):
    def __init__(self, ui, save=None):
        super().__init__()
        self.ui = ui
        self.save = save
        self.specimens = load_specimens()
        self.recorder = None
        self.session = None
        self.last_session = None
        self._recorder_log = None
        self._stop_deadline = None
        self._launched = False       # this tab started the running simulator
        self._awaiting_start = False  # simulator loading; the recorder starts when it is ready
        self._launch_time = 0.0
        self._launch_mode = "none"
        self._launch_rtx = False
        self._decision = None         # "save" / "discard", once the user has answered
        self._finished = None         # the recorder has exited: (session, exit code)
        self._ask_box = None
        self._ct_dialog = None
        if DRILL_LOCK.is_file() and DRILL_LOCK.read_text() == LOCK_TEXT:
            DRILL_LOCK.unlink(missing_ok=True)  # left by a launcher that crashed
        saved = ui.gui_configuration.yaml_data.get(SETTING) if ui is not None else None
        self.settings = {**DEFAULTS, **(saved if isinstance(saved, dict) else {})}
        self._build()
        self._poll = QTimer(self)
        self._poll.setInterval(250)
        self._poll.timeout.connect(self._tick)
        self._poll.start()
        self._refresh()

    # ---------------------------------------------------------------- layout
    def _build(self):
        outer = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        outer.addLayout(left, 2)
        outer.addLayout(right, 3)

        who = QtWidgets.QGroupBox("Expert")
        form = QtWidgets.QFormLayout(who)
        self.first_name = QtWidgets.QLineEdit(str(self.settings.get("first_name", "")))
        self.last_name = QtWidgets.QLineEdit(str(self.settings.get("last_name", "")))
        self.first_name.setPlaceholderText("First name")
        self.last_name.setPlaceholderText("Last name")
        form.addRow("First name:", self.first_name)
        form.addRow("Last name:", self.last_name)
        self.check_development = QtWidgets.QCheckBox("Development session (saved separately)")
        self.check_development.setToolTip(
            "On: saved to expert_drillings_dev with its own session numbers. Off: real expert data.")
        self.check_development.setChecked(bool(self.settings.get("development", True)))
        self.check_development.toggled.connect(self._refresh)
        form.addRow(self.check_development)
        left.addWidget(who)

        box = QtWidgets.QGroupBox("Specimens")
        specimen_row = QtWidgets.QHBoxLayout(box)
        column = QtWidgets.QVBoxLayout()
        self.specimen_group = QtWidgets.QButtonGroup(self)
        self.specimen_column = column
        for specimen in self.specimens:
            self._add_specimen_radio(specimen)
        column.addStretch(1)
        self.button_add_ct = QtWidgets.QPushButton("Add specimens from CT...")
        self.button_add_ct.setToolTip("Split a CT into bones, segment bone + inner ear, and add the ones you accept.")
        self.button_add_ct.clicked.connect(self.open_add_from_ct)
        column.addWidget(self.button_add_ct)
        specimen_row.addLayout(column, 1)
        preview = QtWidgets.QVBoxLayout()
        self.preview_icon = QtWidgets.QLabel("No preview")
        self.preview_icon.setFixedSize(230, 190)
        self.preview_icon.setAlignment(Qt.AlignCenter)
        self.preview_text = QtWidgets.QLabel("")
        self.preview_text.setWordWrap(True)
        preview.addWidget(self.preview_icon)
        preview.addWidget(self.preview_text)
        preview.addStretch(1)
        specimen_row.addLayout(preview, 2)
        left.addWidget(box, 1)

        self.sessions_label = QtWidgets.QLabel("")
        self.sessions_label.setWordWrap(True)
        left.addWidget(self.sessions_label)

        start = QtWidgets.QGroupBox("Start")
        grid = QtWidgets.QGridLayout(start)
        grid.addWidget(QtWidgets.QLabel("Starting burr:"), 0, 0)
        self.combo_burr = QtWidgets.QComboBox()
        for mm in (1, 2, 4, 6):
            self.combo_burr.addItem(f"{mm} mm", mm)
        self.combo_burr.setCurrentIndex(max(0, self.combo_burr.findData(self.settings["burr"])))
        grid.addWidget(self.combo_burr, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Saved view:"), 0, 2)
        self.combo_view = QtWidgets.QComboBox()
        grid.addWidget(self.combo_view, 0, 3)
        grid.setColumnStretch(3, 1)
        right.addWidget(start)

        display = QtWidgets.QGroupBox("Display and recording")
        grid = QtWidgets.QGridLayout(display)
        self.check_rtx = QtWidgets.QCheckBox("RTX bone view")
        self.check_openxr = QtWidgets.QCheckBox("OpenXR eyepiece (WiVRn)")
        self.check_openvr = QtWidgets.QCheckBox("OpenVR HMD (legacy)")
        self.check_stream_stereo = QtWidgets.QCheckBox("Stream stereo")
        self.check_stream_depth = QtWidgets.QCheckBox("Stream depth")
        self.check_video = QtWidgets.QCheckBox("Video record")
        self.check_pupil = QtWidgets.QCheckBox("Pupil capture")
        for row, widgets in enumerate(((self.check_rtx, self.check_openxr), (self.check_openvr, self.check_video),
                                       (self.check_stream_stereo, self.check_stream_depth), (self.check_pupil,))):
            for col, widget in enumerate(widgets):
                grid.addWidget(widget, row, col)
        self.video_label = QtWidgets.QLabel("")
        self.video_label.setWordWrap(True)
        grid.addWidget(self.video_label, 4, 0, 1, 2)
        right.addWidget(display)

        render = QtWidgets.QGroupBox("Rendering / Instruments")
        grid = QtWidgets.QGridLayout(render)
        self.combo_preset = self._combo((("fastest", "Fastest"), ("balanced", "Balanced"),
                                         ("beautiful", "Most beautiful"), ("custom", "Custom")))
        self.combo_dlss = self._combo((("quality", "Quality"), ("balanced", "Balanced"),
                                       ("performance", "Performance"), ("ultra_performance", "Ultra performance")))
        self.combo_samples = self._combo(tuple((n, str(n)) for n in gui_setup.SAMPLE_CHOICES))
        self.combo_bounces = self._combo(tuple((n, "Tuner value" if n == 0 else str(n)) for n in gui_setup.BOUNCE_CHOICES))
        self.check_irrigation = QtWidgets.QCheckBox("Irrigation")
        self.check_dust = QtWidgets.QCheckBox("Bone dust")
        self.check_suction = QtWidgets.QCheckBox("Suction (second Touch)")
        for col, (text, widget) in enumerate((("Preset:", self.combo_preset), ("RTX DLSS:", self.combo_dlss),
                                              ("Samples / pixel:", self.combo_samples),
                                              ("Light bounces:", self.combo_bounces))):
            grid.addWidget(QtWidgets.QLabel(text), col // 2 * 2, col % 2 * 2)
            grid.addWidget(widget, col // 2 * 2, col % 2 * 2 + 1)
        grid.addWidget(self.check_irrigation, 4, 0, 1, 2)
        grid.addWidget(self.check_dust, 4, 2)
        grid.addWidget(self.check_suction, 5, 0, 1, 3)
        right.addWidget(render)

        self.button_sim = QtWidgets.QPushButton("Start simulator")
        self.button_sim.setMinimumHeight(60)
        self.button_sim.clicked.connect(self.pressed_sim)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.button_sim)
        right.addLayout(buttons)
        self.record_badge = QtWidgets.QLabel("")
        self.record_badge.setStyleSheet("color: #F0A8A8; font-weight: bold;")
        right.addWidget(self.record_badge)
        review = QtWidgets.QHBoxLayout()
        self.button_review = QtWidgets.QPushButton("Review sessions")
        self.button_review_last = QtWidgets.QPushButton("Review last session")
        self.button_review.clicked.connect(lambda: self.open_review())
        self.button_review_last.clicked.connect(lambda: self.open_review(self.last_session))
        self.button_review_last.setEnabled(False)
        review.addWidget(self.button_review)
        review.addWidget(self.button_review_last)
        right.addLayout(review)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        right.addStretch(1)

        self._show_settings()
        self.combo_preset.currentIndexChanged.connect(self._preset_chosen)
        for widget in (self.combo_dlss, self.combo_samples, self.combo_bounces):
            widget.currentIndexChanged.connect(self._options_changed)
        for widget in (self.check_irrigation, self.check_dust, self.check_suction):
            widget.toggled.connect(self._options_changed)
        for widget in (self.check_rtx, self.check_openxr, self.check_openvr, self.check_video):
            widget.toggled.connect(self._refresh)
        self.check_openxr.toggled.connect(lambda on: on and self.check_openvr.setChecked(False))
        self.check_openvr.toggled.connect(lambda on: on and self.check_openxr.setChecked(False))
        self.first_name.textChanged.connect(self._refresh)
        self.last_name.textChanged.connect(self._refresh)
        wanted = str(self.settings.get("specimen", ""))
        buttons = self.specimen_group.buttons()
        chosen = next((b for b in buttons if b.specimen["name"] == wanted), buttons[0] if buttons else None)
        if chosen is not None:
            chosen.setChecked(True)

    def _add_specimen_radio(self, specimen):
        radio = QtWidgets.QRadioButton(specimen["name"])
        radio.specimen = specimen
        radio.toggled.connect(self._specimen_changed)
        self.specimen_group.addButton(radio)
        at = len(self.specimen_group.buttons()) - 1
        self.specimen_column.insertWidget(at, radio)
        return radio

    def tbone_dir(self):
        configured = self.ui.gui_configuration.yaml_data.get("tbone_seg_dir")
        return pathlib.Path(configured).expanduser() if configured else ur5_ct_dialog.DEFAULT_TBONE_DIR

    def open_add_from_ct(self):
        python = self.tbone_dir() / ".venv" / "bin" / "python"
        if not python.is_file():
            self.activity("The segmentation model is not installed. Run scripts/segmentation/setup_tbone_seg.sh")
            return
        if self._ct_dialog is None:
            self._ct_dialog = ur5_ct_dialog.AddFromCtDialog(self, python)
        self._ct_dialog.show()
        self._ct_dialog.raise_()
        self._ct_dialog.activateWindow()

    def add_specimen(self, specimen):
        """A specimen just accepted in the CT dialog: list it and select it."""
        self.specimens.append(specimen)
        radio = self._add_specimen_radio(specimen)
        radio.setEnabled(not self.sim_running())
        if not self.sim_running():
            radio.setChecked(True)
        self.activity(f"Added {specimen['name']} ({specimen['laterality']} ear).")

    @staticmethod
    def _combo(items):
        widget = QtWidgets.QComboBox()
        for value, label in items:
            widget.addItem(label, value)
        return widget

    def _show_settings(self):
        s = self.settings
        for widget, key in ((self.check_rtx, "rtx"), (self.check_openxr, "openxr"), (self.check_openvr, "openvr"),
                            (self.check_stream_stereo, "stream_stereo"), (self.check_stream_depth, "stream_depth"),
                            (self.check_video, "video"), (self.check_pupil, "pupil"),
                            (self.check_irrigation, "irrigation"), (self.check_dust, "bone_dust"),
                            (self.check_suction, "suction")):
            widget.blockSignals(True)
            widget.setChecked(bool(s[key]))
            widget.blockSignals(False)
        for widget, key in ((self.combo_preset, "preset"), (self.combo_dlss, "dlss"),
                            (self.combo_samples, "samples"), (self.combo_bounces, "bounces")):
            widget.blockSignals(True)
            widget.setCurrentIndex(max(0, widget.findData(s[key])))
            widget.blockSignals(False)

    def _preset_chosen(self, *_):
        preset = self.combo_preset.currentData()
        if preset in gui_setup.PRESETS:
            self.settings.update(gui_setup.PRESETS[preset], preset=preset)
            self._show_settings()

    def _options_changed(self, *_):
        self.settings.update(dlss=self.combo_dlss.currentData(), samples=self.combo_samples.currentData(),
                             bounces=self.combo_bounces.currentData(), irrigation=self.check_irrigation.isChecked(),
                             bone_dust=self.check_dust.isChecked(), suction=self.check_suction.isChecked())
        self.settings["preset"] = gui_setup.matching_preset(self.settings)
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(self.combo_preset.findData(self.settings["preset"]))
        self.combo_preset.blockSignals(False)

    # ---------------------------------------------------------------- state
    def specimen(self):
        button = self.specimen_group.checkedButton()
        return button.specimen if button is not None else None

    def _specimen_changed(self, checked):
        if not checked:
            return
        specimen = self.specimen()
        pixmap = QPixmap(str(REPO / specimen["icon_path"]))
        if pixmap.isNull():
            self.preview_icon.setText("(no icon)")
        else:
            self.preview_icon.setPixmap(pixmap.scaled(self.preview_icon.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        shape = "x".join(str(v) for v in specimen["voxel_shape_xyz"])
        self.preview_text.setText(f"{specimen['laterality'].capitalize()} ear (from the inner ear's shape)\n"
                                  f"{specimen['voxel_mm']:.2f} mm voxels, {shape}\n"
                                  f"Bone {specimen['voxels']['bone']:,}, inner ear {specimen['voxels']['inner_ear']:,} voxels")
        self._fill_views(specimen["laterality"])
        self._refresh()

    def _fill_views(self, laterality):
        path = REPO / "ADF" / "world" / "canonical_views.yaml"
        views = (yaml.safe_load(path.read_text()) or {}).get("views") or [] if path.is_file() else []
        current = self.combo_view.currentData() or self.settings.get("view", "default")
        self.combo_view.clear()
        self.combo_view.addItem("Default", "default")
        self.combo_view.addItem("Home", "home")
        for view in views:
            side = str(view.get("laterality", "")).lower()
            if view.get("valid") and side in ("", laterality):
                self.combo_view.addItem(str(view.get("label", view.get("id"))), view.get("id"))
        self.combo_view.setCurrentIndex(max(0, self.combo_view.findData(current)))

    def video_mode(self) -> str:
        if not self.check_video.isChecked() or not self.check_rtx.isChecked():
            return "none"
        return "stereo" if self.check_openxr.isChecked() else "mono"

    def root(self) -> pathlib.Path:
        base = pathlib.Path(str(self.ui.gui_configuration.recording_base_path.get()))
        return base / (DEV_DATA_FOLDER if self.check_development.isChecked() else DATA_FOLDER)

    def open_review(self, folder=None):
        """The launcher's Review tab (the standalone app if there is no such tab)."""
        panel = getattr(self.ui, "review_panel", None)
        if panel is not None:
            panel.open_session(folder)
            return
        base = str(self.ui.gui_configuration.recording_base_path.get())
        args = ["--base", base] + ([str(folder)] if folder else [])
        if not QProcess.startDetached(str(REVIEW_APP.with_name("launch_review_app.sh")), args):
            self.activity("ERROR! The review app did not start.")
    def _names(self):
        return self.first_name.text().strip(), self.last_name.text().strip()

    def sim_running(self) -> bool:
        return self.ui._ambf_process.state() != QProcess.NotRunning

    def recording(self) -> bool:
        return self.recorder is not None

    def _refresh(self, *_):
        mode = self.video_mode()
        self.video_label.setText({
            "none": "No video: ground truth only." + ("" if self.check_rtx.isChecked() else " Video needs the RTX bone view."),
            "mono": f"Video: mono, {VIDEO_HZ} fps, from the RTX window.",
            "stereo": f"Video: stereo (both headset eyes), up to {VIDEO_HZ} fps.",
        }[mode])
        first, last = self._names()
        specimen = self.specimen()
        if first and last and specimen and name_part(first) and name_part(last):
            person = next((p for p in self.root().glob("*") if p.is_dir()
                           and p.name.casefold() == person_folder(first, last).casefold()), None) if self.root().is_dir() else None
            done = 0
            if person is not None:
                done = len(list((person / specimen_folder(specimen)).glob("session_[0-9]*")))
            kind = "development" if self.check_development.isChecked() else "expert"
            active = self.session or (self._finished[0] if self._finished else None)
            if active is not None:
                self.sessions_label.setText(f"{first} {last}: {kind} session {active['number']} on {specimen['name']}.")
            else:
                self.sessions_label.setText(f"{first} {last} has {done} {kind} session{'s' if done != 1 else ''} on "
                                            f"{specimen['name']}. The next is session {done + 1}.")
        else:
            self.sessions_label.setText("Enter the first and last name.")
        running = self.sim_running()
        in_session = self.recording() or self._awaiting_start
        ending = self._stop_deadline is not None or self._finished is not None
        self.button_sim.setText("End session" if in_session and self._launched else
                                "Close simulator" if running else "Start simulator")
        self.button_sim.setEnabled(not ending)
        for widget in (self.first_name, self.last_name, self.check_development):
            widget.setEnabled(not (running or in_session or ending))
        for button in self.specimen_group.buttons():
            button.setEnabled(not (running or ending))
        if ending:
            self.record_badge.setText("Finishing the recording...")
        elif self.recording() and self.session["ready"]:
            self.record_badge.setText(f"● Recording {self.session['kind']}session {self.session['number']}. "
                                      "Close the RTX view to finish.")
        elif in_session:
            self.record_badge.setText("Getting the recording ready. The drill is locked until then.")
        else:
            self.record_badge.setText("")

    def activity(self, text):
        self.status.setText(text)
        self.ui.print_info(f"UR5: {text}")

    # ---------------------------------------------------------------- launch
    def launch_request(self) -> dict:
        options = {key: self.settings[key] for key in ("dlss", "samples", "bounces", "irrigation", "bone_dust", "suction")}
        options["preset"] = gui_setup.matching_preset(options)
        mode = self.video_mode()
        if self.check_openxr.isChecked() and self.check_rtx.isChecked():
            options["eye_pipeline"] = True  # as the studies do: frame rate only, same image
        if mode == "mono":
            options["record_hz"] = VIDEO_HZ
            options["record_width"] = 1280
        first, last = self._names()
        specimen = self.specimen()
        return {
            "checks": {"openxr": self.check_openxr.isChecked(), "openvr": self.check_openvr.isChecked(),
                       "rtx": self.check_rtx.isChecked(), "stream_stereo": self.check_stream_stereo.isChecked(),
                       "stream_depth": self.check_stream_depth.isChecked(), "video": False,
                       "pupil": self.check_pupil.isChecked()},
            "options": options, "burr": self.combo_burr.currentData(), "view": self.combo_view.currentData() or "default",
            "user": f"{first} {last}".strip(), "adf": str(REPO / specimen["adf_path"]), "name": specimen["name"],
            "info": gui_setup.VolumeInfo(specimen["name"], REPO / specimen["adf_path"], REPO / specimen["icon_path"],
                                         specimen.get("description", "")),
        }

    def _save_settings(self):
        first, last = self._names()
        specimen = self.specimen()
        self.settings.update(
            rtx=self.check_rtx.isChecked(), openxr=self.check_openxr.isChecked(), openvr=self.check_openvr.isChecked(),
            stream_stereo=self.check_stream_stereo.isChecked(), stream_depth=self.check_stream_depth.isChecked(),
            video=self.check_video.isChecked(), pupil=self.check_pupil.isChecked(),
            burr=self.combo_burr.currentData(), view=self.combo_view.currentData() or "default",
            first_name=first, last_name=last, specimen=specimen["name"] if specimen else "",
            development=self.check_development.isChecked())
        if self.save is not None:
            self.save(SETTING, dict(self.settings))

    def pressed_sim(self):
        if self.sim_running():
            if self._launched and (self.recording() or self._awaiting_start):
                self.end_session()
            else:
                self.ui.close_simulation()
            self._refresh()
            return
        if self.ui.study_mode:
            self.activity("A study is in progress; the simulator was not started.")
            return
        if self.specimen() is None:
            self.activity("No specimens found. Run prepare_specimens.py.")
            return
        first, last = self._names()
        if not name_part(first) or not name_part(last):
            self.activity("Enter the first and last name: every session is recorded under them.")
            return
        self._save_settings()
        self._lock_drill()
        self._launch_mode = self.video_mode()
        self._launch_rtx = self.check_rtx.isChecked()
        self.ui.start_expert_simulation(self.launch_request())
        if not self.sim_running():
            self._unlock_drill()
            self.activity("The simulator did not start.")
            self._refresh()
            return
        self._launched = True
        self._awaiting_start = True
        self._launch_time = time.monotonic()
        self.activity(f"Starting the simulator on {self.specimen()['name']}. Recording starts when it is up.")
        self._refresh()

    @staticmethod
    def _lock_drill():
        DRILL_LOCK.parent.mkdir(parents=True, exist_ok=True)
        DRILL_LOCK.write_text(LOCK_TEXT)

    @staticmethod
    def _unlock_drill():
        if DRILL_LOCK.is_file() and DRILL_LOCK.read_text() == LOCK_TEXT:
            DRILL_LOCK.unlink(missing_ok=True)

    def _simulator_ready(self) -> bool:
        if self._launch_rtx:
            return (self.ui._rtx_process.state() == QProcess.Running
                    and getattr(self.ui, "_rtx_loading", True) is None)
        return time.monotonic() - self._launch_time >= NO_RTX_SETTLE_S

    # ---------------------------------------------------------------- recording
    def _spawn_recorder(self, command, folder):
        self._recorder_log = open(folder / "recorder.log", "w")
        return subprocess.Popen(["bash", "-lc", command], stdout=self._recorder_log,
                                stderr=subprocess.STDOUT, start_new_session=True)

    def _signal_recorder(self, number):
        os.killpg(self.recorder.pid, number)

    def start_recording(self):
        first, last = self._names()
        specimen = self.specimen()
        adf = REPO / specimen["adf_path"]
        folder, number = create_session_folder(self.root(), first, last, specimen)
        mode = self._launch_mode
        request = self.ui.expert_launch or self.launch_request()
        started = datetime.datetime.now().isoformat(timespec="seconds")
        metadata = {
            "format": "fivrs_expert_drilling_v1",
            "development": self.check_development.isChecked(),
            "participant": {"first_name": first, "last_name": last},
            "specimen": {key: specimen.get(key) for key in (
                "name", "bone_id", "laterality", "laterality_source", "segmentation", "source_bone", "voxel_mm", "voxel_shape_xyz", "labels", "adf_path",
                "source_file", "source_sha256", "T_lps_voxel")},
            "session_number": number,
            "started": started,
            "video": {"mode": mode, "rate_hz": VIDEO_HZ if mode != "none" else 0,
                      "stereo_base_mm": 23.0 if mode == "stereo" else None,
                      "stereo_base_sim": 23.0 * 0.01007 if mode == "stereo" else None},
            "launch": {"checks": request["checks"], "render": request["options"],
                       "starting_burr_mm": request["burr"], "startup_view": request["view"]},
            "volume": volume_pose(adf),
            "units": {"mm_per_sim_unit_anatomy": 49.664, "sim_units_per_nominal_burr_mm": 0.01007,
                      "note": "A burr's nominal mm and the anatomy's mm are different scales: "
                              "effective diameter on the anatomy = nominal x 0.01007 x 49.664 (about 0.5)."},
            "software": {"commit": git_commit()},
        }
        (folder / "session.json").write_text(json.dumps(metadata, indent=2))
        command = (
            "set +u && source /opt/ros/jazzy/setup.bash && "
            'source "$HOME/ros_ambf_ws/install/setup.bash" && '
            f"exec {shlex.quote(str(RECORDER_PYTHON))} -u {shlex.quote(str(RECORDER))} "
            f"--output_dir {shlex.quote(str(folder))} --volume_adf {shlex.quote(str(adf))} "
            f"--video {mode} --video_hz {VIDEO_HZ}"
        )
        self.recorder = self._spawn_recorder(command, folder)
        self.session = {"folder": folder, "number": number, "first": first, "last": last, "specimen": specimen,
                        "started": started, "video": mode, "started_at": time.monotonic(), "ready": False,
                        "kind": "development " if self.check_development.isChecked() else ""}
        self.session["root"] = self.root()
        if request["checks"].get("pupil") and self.ui._pupil_process.state() == QProcess.Running:
            try:
                self.ui.study_manager.pupil_manager.start_recording(str(folder / "pupil"))
                self.session["pupil"] = True
            except Exception as exc:
                self.activity(f"Pupil recording did not start: {exc}")
        self.activity(f"Preparing session {number} of {first} {last} on {specimen['name']}...")
        self._refresh()

    def stop_recording(self):
        if self.recorder is None or self._stop_deadline is not None:
            return
        self._lock_drill()  # the final volume and the last removals describe the same moment
        if self.session.get("pupil"):
            try:
                self.ui.study_manager.pupil_manager.stop_recoding()
            except Exception as exc:
                self.activity(f"Pupil recording did not stop cleanly: {exc}")
        if self.recorder.poll() is None:
            self._signal_recorder(signal.SIGINT)
        self._stop_deadline = time.monotonic() + 120.0
        self._refresh()

    def end_session(self):
        """The session is over: finish the recording and ask whether to keep it."""
        if self._awaiting_start:
            self._awaiting_start = False
            self._launched = False
            self._unlock_drill()
            self.activity("Closed before recording began; nothing was saved.")
            if self.sim_running():
                self.ui.close_simulation()
            return
        if self.recorder is None or self._stop_deadline is not None:
            return
        session = self.session
        self.stop_recording()
        self.activity("Session ended. Save or discard the recording.")
        self._ask(session)

    def _ask(self, session):
        minutes = (time.monotonic() - session["started_at"]) / 60.0
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle("Save this recording?")
        box.setText("Save this recording?")
        box.setInformativeText(f"{session['first']} {session['last']}, {session['specimen']['name']}, "
                               f"{session['kind']}session {session['number']} ({minutes:.1f} min).\n\n"
                               "Discard moves it to the Trash.")
        save = box.addButton("Save", QtWidgets.QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
        box.setDefaultButton(save)
        box.setEscapeButton(save)  # closing the box never loses a session
        box.buttonClicked.connect(lambda button: self._answer(button is not discard))
        self._ask_box = box
        box.setModal(True)
        box.open()
        window = self.window()
        window.raise_()
        window.activateWindow()

    def _answer(self, keep: bool):
        self._decision = "save" if keep else "discard"
        self._ask_box = None
        if self._finished is not None:
            self._apply_decision()
        else:
            self.activity("Finishing the recording...")

    def _tick(self):
        if self._awaiting_start:
            if not self.sim_running():
                self._awaiting_start = False
                self._launched = False
                self._unlock_drill()
            elif self._simulator_ready():
                self._awaiting_start = False
                self.start_recording()
        if self.recorder is not None:
            session = self.session
            if not session["ready"] and (session["folder"] / "recording.started").exists():
                session["ready"] = True
                self._unlock_drill()
                self.activity(f"RECORDING {session['kind']}session {session['number']} of {session['first']} {session['last']} "
                              f"on {session['specimen']['name']}.")
                self._refresh()
            code = self.recorder.poll()
            if code is None and not session["ready"] and self._stop_deadline is None \
                    and time.monotonic() - session["started_at"] > RECORDER_START_TIMEOUT_S:
                self._signal_recorder(signal.SIGKILL)
                code = self.recorder.wait()
            if code is None and self._stop_deadline is not None and time.monotonic() > self._stop_deadline:
                self._signal_recorder(signal.SIGKILL)
                code = self.recorder.wait()
            if code is not None:
                self._recording_ended(code)
        if self.button_sim.text().startswith("Start") == self.sim_running():
            self._refresh()

    def _recording_ended(self, code):
        session = self.session
        expected = self._stop_deadline is not None
        self.recorder = None
        self.session = None
        self._stop_deadline = None
        if self._recorder_log:
            self._recorder_log.close()
            self._recorder_log = None
        if not expected:
            # The recorder died mid-session. Keep what it wrote and let nothing go unrecorded.
            self._lock_drill()
            self._log_session(session)
            text = (f"Recording of session {session['number']} stopped unexpectedly (code {code}). "
                    f"The drill is locked. Close the simulator and start again. Log: {session['folder']}/recorder.log")
            self.activity(f"ERROR! {text}")
            QtWidgets.QMessageBox.warning(self, "Recording stopped", text)
            self._refresh()
            return
        self._finished = (session, code)
        if self._decision is not None:
            self._apply_decision()
        self._refresh()

    @staticmethod
    def _log_session(session):
        append_session_row(session["root"], {
            "session_id": f"{person_folder(session['first'], session['last'])}__"
                          f"{specimen_folder(session['specimen'])}__{session['number']:03d}",
            "last_name": session["last"], "first_name": session["first"], "specimen": session["specimen"]["name"],
            "session_number": session["number"], "started": session["started"], "video": session["video"],
            "folder": str(session["folder"])})

    def _apply_decision(self):
        session, code = self._finished
        decision, self._decision, self._finished = self._decision, None, None
        if decision == "discard":
            where = discard_session(session["folder"])
            self.activity(f"Discarded session {session['number']} (moved to {where}).")
        else:
            self._log_session(session)
            self._report_saved(session, code)
        self._launched = False
        if self.sim_running():
            self.ui.close_simulation()
        else:
            self._unlock_drill()
        self._refresh()

    def _report_saved(self, session, code):
        summary = None
        try:
            summary = json.loads((session["folder"] / "session.json").read_text()).get("summary")
        except (OSError, ValueError):
            pass
        if summary is None:
            self.activity(f"ERROR! Session {session['number']} did not finish (code {code}). See {session['folder']}/recorder.log")
            return
        check = summary["replay_check"]
        verdict = {"exact": "replay matches the simulator exactly",
                   "mismatch": (f"WARNING! replay differs from the simulator by "
                                f"{check.get('voxels_ambf_removed_not_recorded', 0) + check.get('voxels_recorded_but_still_in_ambf', 0)} voxels"),
                   }.get(check["status"], "replay not checked (the simulator had already closed)")
        self.last_session = session["folder"]
        self.button_review_last.setEnabled(True)
        self.activity(f"Saved {session['kind']}session {session['number']}: {summary['duration_s']:.0f} s, "
                      f"{summary['voxels_removed']:,} voxels, {summary['strokes']} strokes; {verdict}.")

    def rtx_view_closed(self):
        """The RTX view has exited. AMBF is still up, so the final volume can be checked."""
        if self._launched and (self.recording() or self._awaiting_start):
            if self._awaiting_start:
                self.activity("The RTX view closed before the session began.")
            self.end_session()
            self._refresh()

    def simulator_closed(self):
        if self._awaiting_start:
            self._awaiting_start = False
            self._launched = False
            self._unlock_drill()
        elif self.recording():
            self.end_session()
        elif self._finished is None and self._decision is None:
            self._launched = False
            self._unlock_drill()
        self._refresh()

    def shutdown(self):
        """The launcher is closing: a session still open is saved, never lost."""
        self._awaiting_start = False
        if self._ask_box is not None:
            self._ask_box.done(0)
            self._ask_box = None
        if self.recorder is not None:
            self._decision = self._decision or "save"
            self.stop_recording()
            try:
                code = self.recorder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self._signal_recorder(signal.SIGKILL)
                code = self.recorder.wait()
            self._recording_ended(code)
        elif self._finished is not None:
            self._decision = self._decision or "save"
            self._apply_decision()
        self._unlock_drill()
