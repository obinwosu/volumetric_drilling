"""Launcher tab "Register MRI": line an MRI up with a temporal bone CT, automatically.

1. Choose the CT and the MRI (DICOM folders; one folder holding both is fine).
2. Register: scripts/registration/mri_ct_worker.py runs in the tbone-seg venv (it has
   SimpleITK and torch) and finds the rigid transform on the GPU in about half a minute.
   Why it works the way it does: scripts/registration/rigid_register.py.
3. Check: the three views show the MRI over the CT (blend, checkerboard or colour);
   mouse wheel moves through the slices, click to move the crosshair. "Open in 3D Slicer"
   shows the same result in Slicer, where it can be adjusted by hand and saved back.

Results go to resources/intermediate_volumes/registration/<date_time>/ (git-ignored):
ct.nii.gz, mri.nii.gz, mri_to_ct.tfm (ITK; Slicer reads it), mri_on_ct.nii.gz (the MRI
on the CT grid) and registration.json.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import pathlib
import shutil

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap

import tumor_panel

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKER = REPO / "scripts" / "registration" / "mri_ct_worker.py"
SLICER_SCRIPT = REPO / "scripts" / "registration" / "slicer_show.py"
DEFAULT_TBONE_DIR = REPO.parent / "tbone-seg-nnunetv2"
WORK_ROOT = REPO / "resources" / "intermediate_volumes" / "registration"  # git-ignored
SLICER_SETTING = "slicer_path"
CT_WINDOW = (-400.0, 2000.0)
CHECKER_MM = 8.0


def find_slicer(configured: str | None = None) -> str | None:
    """3D Slicer's launcher: the configured path, then PATH, then the usual unpack places."""
    candidates = []
    if configured:
        candidates.append(os.path.expanduser(configured))
    found = shutil.which("Slicer")
    if found:
        candidates.append(found)
    home = os.path.expanduser("~")
    for pattern in (f"{home}/Slicer*/Slicer", f"{home}/*/Slicer*/Slicer", f"{home}/Downloads/Slicer*/Slicer",
                    "/opt/Slicer*/Slicer", "/usr/local/Slicer*/Slicer"):
        candidates.extend(sorted(glob.glob(pattern), reverse=True))  # newest version first
    return next((c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)), None)


def confidence(result: dict) -> tuple[bool, str]:
    """(sure, text) from the worker's record: several starts ending on the same pose, and
    that pose clearly ahead of the best different one."""
    score = result.get("score_1mm") or 0.0
    runner = result.get("runner_up_score_1mm")
    agree = result.get("starts_agreeing", 0)
    ratio = (runner / score) if runner and score else 0.0
    sure = agree >= 2 and ratio <= 0.8
    text = (f"{agree} of {result.get('starts_tried', '?')} starts ended here; next-best different pose scores "
            f"{ratio:.0%} of this one" if runner else f"{agree} starts ended here")
    return sure, text


def fuse(ct_plane: np.ndarray, mri_plane: np.ndarray, mode: str, mix: float, spacing_mm: float,
         labels: np.ndarray | None = None) -> np.ndarray:
    """RGB uint8 of one plane. ct_plane in HU; mri_plane uint8 (0 = no MRI there);
    labels: tumor parts (1, 2, ...) tinted in tumor_panel.PART_COLORS."""
    low, high = CT_WINDOW
    ct = np.clip((ct_plane.astype(np.float32) - low) / (high - low), 0, 1)
    mri = mri_plane.astype(np.float32) / 255.0
    has_mri = mri_plane > 0
    if mode == "checker":
        n = max(1, int(round(CHECKER_MM / spacing_mm)))
        rows, cols = np.indices(ct.shape)
        use_mri = (((rows // n) + (cols // n)) % 2 == 0) & has_mri
        gray = np.where(use_mri, mri, ct)
        rgb = np.stack([gray] * 3, -1)
    elif mode == "color":  # CT in gray, MRI added in green
        rgb = np.stack([ct * (1 - 0.5 * mix * mri), ct, ct * (1 - 0.5 * mix * mri)], -1)
        rgb[..., 1] = np.clip(ct * (1 - mix * 0.4) + mix * mri, 0, 1)
    else:  # blend
        gray = np.where(has_mri, (1 - mix) * ct + mix * mri, ct * (1 - mix * 0.6))
        rgb = np.stack([gray] * 3, -1)
    if labels is not None and labels.any():
        for n in np.unique(labels[labels > 0]):
            color = np.array(tumor_panel.PART_COLORS[(int(n) - 1) % len(tumor_panel.PART_COLORS)]) / 255.0
            inside = labels == n
            rgb[inside] = rgb[inside] * 0.55 + color * 0.45
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


class FusedView(QtWidgets.QLabel):
    """One plane with a crosshair. Click moves the crosshair; the wheel moves the slice."""

    clicked = pyqtSignal(float, float)  # column, row in the plane's pixels (press or drag)
    pressed = pyqtSignal(float, float, bool)  # a press only: column, row, Shift held
    scrolled = pyqtSignal(int)

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.setMinimumSize(240, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: black;")
        self.image = None
        self.cross = None
        self.marks = ("", "")
        self._scale, self._offset = 1.0, (0, 0)

    def show_rgb(self, rgb: np.ndarray, cross, marks=("", "")):
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.cross, self.marks = cross, marks
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self):
        if self.image is None:
            return
        target = self.size()
        pix = QPixmap.fromImage(self.image).scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = pix.width() / self.image.width()
        self._offset = ((target.width() - pix.width()) / 2, (target.height() - pix.height()) / 2)
        painter = QPainter(pix)
        if self.cross is not None:
            col, row = (v * self._scale for v in self.cross)
            painter.setPen(QPen(QColor(255, 200, 0, 150), 1, Qt.DashLine))
            painter.drawLine(int(col), 0, int(col), pix.height())
            painter.drawLine(0, int(row), pix.width(), int(row))
        painter.setPen(QColor(120, 200, 255))
        painter.drawText(6, 16, self.title)
        if self.marks[0]:
            painter.drawText(6, pix.height() // 2, self.marks[0])
        if self.marks[1]:
            painter.drawText(pix.width() - 14, pix.height() // 2, self.marks[1])
        painter.end()
        self.setPixmap(pix)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() // 120
        if steps:
            self.scrolled.emit(int(steps))

    def _where(self, event):
        if self.image is None:
            return None
        col = (event.x() - self._offset[0]) / self._scale
        row = (event.y() - self._offset[1]) / self._scale
        return (col, row) if 0 <= col < self.image.width() and 0 <= row < self.image.height() else None

    def mouseMoveEvent(self, event):
        where = self._where(event) if event.buttons() & Qt.LeftButton else None
        if where:
            self.clicked.emit(*where)

    def mousePressEvent(self, event):
        where = self._where(event)
        if where:
            self.pressed.emit(where[0], where[1], bool(event.modifiers() & Qt.ShiftModifier))
            self.clicked.emit(*where)


class RegisterMriPanel(QtWidgets.QWidget):
    def __init__(self, ui=None, tbone_dir=None, slicer_path=None, save=None, tumor_python=None):
        """save(key, value): stores a setting in gui_setup.yaml (the Slicer path).
        tumor_python: Python of the nnInteractive venv (default ../nninteractive/.venv)."""
        super().__init__()
        self.ui = ui
        self.tbone_dir = pathlib.Path(tbone_dir or DEFAULT_TBONE_DIR).expanduser()
        self.python = self.tbone_dir / ".venv" / "bin" / "python"
        self.slicer_setting = slicer_path
        self.save = save
        self.series = {"ct": [], "mri": []}
        self.result = None
        self.preview_ct = self.preview_mri = None
        self.cursor = None  # (k, j, i) in the preview grid
        self._job = None
        self._stdout = ""
        self._result = self._error = None
        self._scan_queue = []
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self.tumor = tumor_panel.TumorPanel(tumor_python, say=self._say)
        self.tumor.labels_changed.connect(self._update_views)
        self._build()
        if not self.python.is_file():
            self._say(f"Python with SimpleITK + torch not found: {self.python}\n"
                      "Set tbone_seg_dir in gui_setup.yaml (scripts/segmentation/setup_tbone_seg.sh installs it).")
        self._update_slicer()

    # ------------------------------------------------------------ layout

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)

        scans = QtWidgets.QGroupBox("1. Scans (DICOM folders; one folder may hold both)")
        grid = QtWidgets.QGridLayout(scans)
        self.combos, self.folder_labels = {}, {}
        for row, (kind, label) in enumerate((("ct", "CT"), ("mri", "MRI"))):
            button = QtWidgets.QPushButton(f"{label} folder...")
            button.clicked.connect(lambda _=False, k=kind: self._choose_folder(k))
            setattr(self, f"{kind}_button", button)
            self.folder_labels[kind] = QtWidgets.QLabel("No folder chosen")
            self.folder_labels[kind].setWordWrap(True)
            self.combos[kind] = QtWidgets.QComboBox()
            self.combos[kind].setMinimumWidth(460)
            self.combos[kind].currentIndexChanged.connect(self._set_enabled)
            grid.addWidget(button, row, 0)
            grid.addWidget(self.combos[kind], row, 1)
            grid.addWidget(self.folder_labels[kind], row, 2)
        grid.setColumnStretch(2, 1)
        hint = QtWidgets.QLabel("MRI: a heavily T2-weighted CISS / FIESTA or a post-contrast T1 both work. "
                                "The CT may hold one ear or both.")
        hint.setStyleSheet("color: gray;")
        grid.addWidget(hint, 2, 0, 1, 3)
        layout.addWidget(scans)

        run = QtWidgets.QGroupBox("2. Register")
        row = QtWidgets.QHBoxLayout(run)
        self.go_button = QtWidgets.QPushButton("Register MRI to CT")
        self.go_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.go_button.clicked.connect(self._run)
        row.addWidget(self.go_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        row.addWidget(self.cancel_button)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("")
        row.addWidget(self.progress, 1)
        self.open_button = QtWidgets.QPushButton("Open a result...")
        self.open_button.setToolTip("Show an earlier registration (a folder in resources/intermediate_volumes/"
                                    "registration).")
        self.open_button.clicked.connect(self._open_result)
        row.addWidget(self.open_button)
        layout.addWidget(run)

        check = QtWidgets.QGroupBox("3. Check (mouse wheel: move through the slices; click: move the crosshair)")
        check_layout = QtWidgets.QVBoxLayout(check)
        controls = QtWidgets.QHBoxLayout()
        self.result_label = QtWidgets.QLabel("No result yet.")
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(40)
        controls.addWidget(self.result_label, 1)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Blend", "blend")
        self.mode_combo.addItem("Checkerboard", "checker")
        self.mode_combo.addItem("MRI in green", "color")
        self.mode_combo.currentIndexChanged.connect(self._update_views)
        controls.addWidget(self.mode_combo)
        controls.addWidget(QtWidgets.QLabel("CT"))
        self.mix_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(50)
        self.mix_slider.setFixedWidth(140)
        self.mix_slider.valueChanged.connect(self._update_views)
        controls.addWidget(self.mix_slider)
        controls.addWidget(QtWidgets.QLabel("MRI"))
        check_layout.addLayout(controls)
        views = QtWidgets.QHBoxLayout()
        self.axial, self.coronal, self.sagittal = FusedView("Axial"), FusedView("Coronal"), FusedView("Sagittal")
        for view in (self.axial, self.coronal, self.sagittal):
            views.addWidget(view)
        self.axial.clicked.connect(lambda c, r: self._clicked("axial", c, r))
        self.coronal.clicked.connect(lambda c, r: self._clicked("coronal", c, r))
        self.sagittal.clicked.connect(lambda c, r: self._clicked("sagittal", c, r))
        self.axial.pressed.connect(lambda c, r, shift: self._pressed("axial", c, r, shift))
        self.coronal.pressed.connect(lambda c, r, shift: self._pressed("coronal", c, r, shift))
        self.sagittal.pressed.connect(lambda c, r, shift: self._pressed("sagittal", c, r, shift))
        self.axial.scrolled.connect(lambda n: self._move(0, n))
        self.coronal.scrolled.connect(lambda n: self._move(1, -n))
        self.sagittal.scrolled.connect(lambda n: self._move(2, n))
        check_layout.addLayout(views, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.ear_buttons = []
        for index, text in enumerate(("Right ear", "Left ear")):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _=False, i=index: self._go_to_ear(i))
            buttons.addWidget(button)
            self.ear_buttons.append(button)
        buttons.addStretch(1)
        self.slicer_button = QtWidgets.QPushButton("Open in 3D Slicer")
        self.slicer_button.clicked.connect(self._open_slicer)
        buttons.addWidget(self.slicer_button)
        self.find_slicer_button = QtWidgets.QPushButton("Find 3D Slicer...")
        self.find_slicer_button.clicked.connect(self._choose_slicer)
        buttons.addWidget(self.find_slicer_button)
        self.reload_button = QtWidgets.QPushButton("Reload after Slicer")
        self.reload_button.setToolTip("After adjusting the transform in Slicer and saving it over mri_to_ct.tfm:\n"
                                      "redo the MRI on the CT grid and these views.")
        self.reload_button.clicked.connect(self._reload)
        buttons.addWidget(self.reload_button)
        self.folder_button = QtWidgets.QPushButton("Results folder")
        self.folder_button.clicked.connect(self._open_folder)
        buttons.addWidget(self.folder_button)
        check_layout.addLayout(buttons)
        layout.addWidget(check, 1)
        layout.addWidget(self.tumor)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        layout.addWidget(self.log)
        self._set_enabled()

    # ------------------------------------------------------------ worker process

    def _start(self, job, args):
        self._job = job
        self._stdout = ""
        self._result = self._error = None
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(REPO))
        self._process.start(str(self.python), ["-I", str(WORKER), *args])
        self._set_enabled()

    def _read_output(self):
        self._stdout += bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        *lines, self._stdout = self._stdout.replace("\r", "\n").split("\n")
        for line in lines:
            self._handle_line(line)

    def _handle_line(self, line):
        if line.startswith("@progress "):
            _, fraction, text = line.split(" ", 2)
            self.progress.setValue(int(float(fraction) * 1000))
            self.progress.setFormat(text)
        elif line.startswith("@result "):
            self._result = json.loads(line[len("@result "):])
        elif line.startswith("@error "):
            self._error = line[len("@error "):]
        elif line.strip() and "WARNING" not in line and "No Series were found" not in line:
            self.log.appendPlainText(line)

    def _finished(self, code, _status):
        if self._stdout:
            self._handle_line(self._stdout)
            self._stdout = ""
        job, self._job = self._job, None
        result, error = self._result, self._error
        if error or code != 0 or result is None:
            self._say(f"Failed: {error or f'exit code {code}'}")
            self.progress.setFormat("Failed")
            self._scan_queue = []
        elif job[0] == "scan":
            self._scanned(job[1], job[2], result["series"])
        elif job[0] in ("run", "preview"):
            self.show_result(result)
        if self._scan_queue and self._job is None:
            self._start(*self._scan_queue.pop(0))
        self._set_enabled()

    def _cancel(self):
        if self._process.state() != QProcess.NotRunning:
            self._scan_queue = []
            self._process.kill()
            self._say("Cancelled.")

    def _busy(self):
        return self._process.state() != QProcess.NotRunning

    def _set_enabled(self, *_):
        busy = self._busy()
        for kind in ("ct", "mri"):
            getattr(self, f"{kind}_button").setEnabled(not busy)
            self.combos[kind].setEnabled(not busy and bool(self.series[kind]))
        self.go_button.setEnabled(not busy and self._chosen("ct") is not None and self._chosen("mri") is not None)
        self.cancel_button.setEnabled(busy)
        self.open_button.setEnabled(not busy)
        have = self.result is not None
        self.slicer_button.setEnabled(have and self._slicer() is not None)
        self.reload_button.setEnabled(have and not busy)
        self.folder_button.setEnabled(have)
        for button in self.ear_buttons:
            button.setEnabled(have and self.preview_ct is not None)

    def _say(self, text):
        self.log.appendPlainText(text)

    # ------------------------------------------------------------ 1. scans

    def _choose_folder(self, kind):
        start = self.folder_labels[kind].text()
        if not os.path.isdir(start):
            start = str(pathlib.Path.home() / "Downloads")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, f"{kind.upper()} DICOM folder", start)
        if folder:
            self.open_folder(kind, folder)

    def open_folder(self, kind, folder):
        self.folder_labels[kind].setText(folder)
        self.progress.setFormat("Looking for DICOM series...")
        job = (("scan", kind, folder), ["scan", folder])
        if self._busy():
            self._scan_queue.append(job)
        else:
            self._start(*job)

    def _scanned(self, kind, folder, series):
        """Fill this kind's list; a folder with both CT and MR series also fills the other."""
        modality = {"ct": "CT", "mri": "MR"}
        found = {k: [s for s in series if s.get("modality") == modality[k]] for k in ("ct", "mri")}
        if not found[kind]:  # unknown modality: offer everything
            found[kind] = series
        fill = [kind] + [k for k in ("ct", "mri") if k != kind and found[k] and not self.series[k]]
        for k in fill:
            items = sorted(found[k], key=lambda s: -s["slices"])  # thin reconstructions have the most slices
            self.series[k] = items
            combo = self.combos[k]
            combo.blockSignals(True)
            combo.clear()
            for s in items:
                text = f"{s.get('description') or '(no name)'}  |  {s['slices']} slices"
                if s.get("slice_thickness"):
                    text += f", {float(s['slice_thickness']):g} mm"
                combo.addItem(text)
            combo.blockSignals(False)
            if k != kind:
                self.folder_labels[k].setText(folder)
        self.progress.setFormat("")
        self.tumor.offer_series(found["mri"])
        self._say(f"{folder}: {len(found['ct'])} CT and {len(found['mri'])} MR series.")
        self._set_enabled()

    def _chosen(self, kind):
        index = self.combos[kind].currentIndex()
        items = self.series[kind]
        return items[index] if 0 <= index < len(items) else None

    # ------------------------------------------------------------ 2. register

    def _run(self):
        ct, mri = self._chosen("ct"), self._chosen("mri")
        if ct is None or mri is None:
            return
        work = WORK_ROOT / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work.mkdir(parents=True, exist_ok=True)
        self._say(f"Registering {mri.get('description') or 'MRI'} to {ct.get('description') or 'CT'}...")
        self._start(("run",), ["run", "--ct", ct["folder"], "--ct-uid", ct["uid"],
                               "--mri", mri["folder"], "--mri-uid", mri["uid"], "--work", str(work)])

    def _open_result(self):
        start = str(WORK_ROOT if WORK_ROOT.is_dir() else REPO)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Registration folder", start)
        if folder:
            path = pathlib.Path(folder) / "registration.json"
            if not path.is_file():
                self._say(f"No registration.json in {folder}")
                return
            result = json.loads(path.read_text())
            result["work"] = folder
            self.show_result(result)

    def _reload(self):
        if self.result:
            self._start(("preview",), ["preview", self.result["work"]])

    # ------------------------------------------------------------ 3. check

    def show_result(self, result):
        self.result = result
        work = pathlib.Path(result["work"])
        self.preview_ct = np.load(work / "preview_ct.npy")
        self.preview_mri = np.load(work / "preview_mri.npy")
        ears = result.get("temporal_bones_mm") or []
        self.cursor = self._index_of(ears[0]) if ears else [n // 2 for n in self.preview_ct.shape]
        pitch, roll, yaw = result.get("pitch_roll_yaw_deg", (0, 0, 0))
        sure, why = confidence(result)
        verdict = ("<b style='color:#3a3'>Looks right.</b>" if sure else
                   "<b style='color:#c60'>Not sure: check it here or in 3D Slicer.</b>")
        edited = f" Transform edited {result['transform_edited']}." if result.get("transform_edited") else ""
        self.result_label.setText(
            f"{verdict} {why}.<br>MRI tilted {pitch:+.1f}° nose up/down, {roll:+.1f}° roll, "
            f"{yaw:+.1f}° turn relative to the CT; {result.get('seconds', 0):.0f} s on "
            f"{result.get('device', '?')}.{edited}")
        self.result_label.setTextFormat(Qt.RichText)
        self.ear_buttons[0].setVisible(len(ears) > 1)
        self.ear_buttons[1].setText("Left ear" if len(ears) > 1 else "Ear")
        self.tumor.set_work(str(work))
        self._update_views()
        self._set_enabled()
        self._say(f"Result: {work}")

    def _direction(self):
        return np.array(self.result.get("preview_direction") or np.eye(3).ravel()).reshape(3, 3)

    def _index_of(self, point):
        """LPS mm -> preview (k, j, i). The preview grid has the CT's axes."""
        step = self.result["preview_spacing_mm"]
        origin = np.array(self.result["preview_origin_mm"])
        i, j, k = np.round(self._direction().T @ (np.array(point) - origin) / step).astype(int)
        shape = self.preview_ct.shape
        return [int(np.clip(k, 0, shape[0] - 1)), int(np.clip(j, 0, shape[1] - 1)), int(np.clip(i, 0, shape[2] - 1))]

    def _point_of(self, index_kji):
        """Preview (k, j, i) -> LPS mm."""
        k, j, i = index_kji
        step = self.result["preview_spacing_mm"]
        return np.array(self.result["preview_origin_mm"]) + self._direction() @ (np.array([i, j, k], float) * step)

    def _pressed(self, plane, col, row, shift):
        """A press (not a drag): in the tumor box's click mode, segment there."""
        if self.preview_ct is None:
            return
        self._clicked(plane, col, row)
        self.tumor.click(self._point_of(self.cursor), positive=not shift)

    def _go_to_ear(self, index):
        ears = (self.result or {}).get("temporal_bones_mm") or []
        if ears:  # [right, left], or one ear
            self.cursor = self._index_of(ears[min(index, len(ears) - 1)])
            self._update_views()

    def _move(self, axis, steps):
        """axis 0: through the axial slices (wheel up = superior), 1: coronal, 2: sagittal."""
        if self.preview_ct is None:
            return
        self.cursor[axis] = int(np.clip(self.cursor[axis] + steps, 0, self.preview_ct.shape[axis] - 1))
        self._update_views()

    def _clicked(self, plane, col, row):
        if self.preview_ct is None:
            return
        depth = self.preview_ct.shape[0]
        k, j, i = self.cursor
        if plane == "axial":
            i, j = col, row
        elif plane == "coronal":
            i, k = col, depth - 1 - row
        else:
            j, k = col, depth - 1 - row
        shape = self.preview_ct.shape
        self.cursor = [int(np.clip(k, 0, shape[0] - 1)), int(np.clip(j, 0, shape[1] - 1)),
                       int(np.clip(i, 0, shape[2] - 1))]
        self._update_views()

    def _update_views(self, *_):
        if self.preview_ct is None:
            return
        mode = self.mode_combo.currentData()
        mix = self.mix_slider.value() / 100.0
        step = self.result["preview_spacing_mm"]
        k, j, i = self.cursor
        depth = self.preview_ct.shape[0]
        ct, mri = self.preview_ct, self.preview_mri
        tumor = self.tumor.preview if (self.tumor.preview is not None
                                       and self.tumor.preview.shape == ct.shape) else np.zeros(ct.shape, np.uint8)
        # Radiological: the patient's right on the left of the screen (LPS i grows to the left).
        self.axial.show_rgb(fuse(ct[k], mri[k], mode, mix, step, tumor[k]), (i, j), ("R", "L"))
        self.coronal.show_rgb(fuse(ct[::-1, j], mri[::-1, j], mode, mix, step, tumor[::-1, j]),
                              (i, depth - 1 - k), ("R", "L"))
        self.sagittal.show_rgb(fuse(ct[::-1, :, i], mri[::-1, :, i], mode, mix, step, tumor[::-1, :, i]),
                               (j, depth - 1 - k), ("A", "P"))

    # ------------------------------------------------------------ 3D Slicer and files

    def _slicer(self):
        return find_slicer(self.slicer_setting)

    def _update_slicer(self):
        path = self._slicer()
        self.slicer_button.setToolTip(path or "3D Slicer not found: use Find 3D Slicer... (slicer.org, Linux build).")
        self.find_slicer_button.setVisible(path is None)
        self._set_enabled()

    def _choose_slicer(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "3D Slicer's Slicer launcher", str(pathlib.Path.home()))
        if path:
            self.slicer_setting = path
            if self.save is not None:
                self.save(SLICER_SETTING, path)
            self._update_slicer()

    def _open_slicer(self):
        slicer = self._slicer()
        if not (slicer and self.result):
            return
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("FIVRS_REGISTRATION_DIR", self.result["work"])
        process.setProcessEnvironment(environment)
        process.setProgram(slicer)
        process.setArguments(["--python-script", str(SLICER_SCRIPT), self.result["work"]])
        ok, _pid = process.startDetached()
        self._say("Opening 3D Slicer..." if ok else f"Could not start {slicer}")

    def _open_folder(self):
        if self.result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.result["work"]))
