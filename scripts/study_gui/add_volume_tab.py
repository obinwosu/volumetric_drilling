"""Launcher tab "Add a volume": CT DICOM -> temporal bone segmentation -> new anatomy.

1. Choose a DICOM folder. Every series below it is listed; picking one loads it.
2. The ear: laterality comes from the series name ("Left Dehiscence 0.4") or, for a
   single-ear scan, from where it sits in the patient. It is only asked when neither
   says. A box (50 mm, the size of the model's training scans) is placed on the ear;
   click any of the three views to move it.
3. Segment: the box is cropped and resampled to 0.098 mm, segmented by the
   tbone-seg-nnunetv2 model (scripts/segmentation/tbone_worker.py, run in that
   repo's venv), and written as PNG slices + ADF + icon. The new volume is added to
   gui_setup.yaml and to the anatomy list on the Launch tab. With "Then start the
   simulator on it" ticked, the launcher then starts the simulator on that volume.

Only the box is ever segmented: a whole head at 0.1 mm would not fit in GPU memory.
Folds: one is the default. Each extra fold is another model in the ensemble; it adds
time and host memory (the fold outputs are summed in RAM).
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKER = REPO / "scripts" / "segmentation" / "tbone_worker.py"
DEFAULT_TBONE_DIR = REPO.parent / "tbone-seg-nnunetv2"
WORK_ROOT = REPO / "resources" / "intermediate_volumes" / "segmentation"  # git-ignored
MODEL_SPACING_MM = 0.098
CROP_LOG = REPO / "scripts" / "segmentation" / "crop_log.jsonl"  # no patient data: for an auto-crop later
WINDOW = (-400.0, 2000.0)  # HU shown black .. white (bone window)


# ---------------------------------------------------------------- gui_setup.yaml


def register_volume(yaml_path: pathlib.Path, name: str, adf: pathlib.Path, icon: pathlib.Path,
                    description: str) -> str:
    """Add a volumeN entry and list it under volumes:, editing only those lines so the rest of
    the file keeps its order and comments. Returns the key. Re-adding the same ADF reuses its key."""
    text = yaml_path.read_text()
    lines = text.splitlines()
    base = yaml_path.parent

    def rel(path):
        """Paths as the other entries write them: ./assets/x.png, ../../ADF/x.yaml."""
        path = pathlib.Path(path).resolve()
        if path.is_relative_to(base.resolve()):
            return "./" + path.relative_to(base.resolve()).as_posix()
        if path.is_relative_to(REPO):
            return "../../" + path.relative_to(REPO).as_posix()
        return str(path)

    adf_rel, icon_rel = rel(adf), rel(icon)
    keys = [int(m.group(1)) for m in re.finditer(r"^volume(\d+):", text, re.M)]
    existing = re.search(rf"^(volume\d+):\n  adf_path: {re.escape(adf_rel)}\n", text, re.M)
    key = existing.group(1) if existing else f"volume{max(keys, default=0) + 1}"
    quoted = description.replace("'", "''")
    block = [f"{key}:", f"  adf_path: {adf_rel}", f"  icon_path: {icon_rel}", f"  name: {json.dumps(name)}",
             f"  description: '{quoted}'"]
    if existing:  # replace the old block
        start = lines.index(f"{key}:")
        end = start + 1
        while end < len(lines) and lines[end].startswith("  "):
            end += 1
        lines[start:end] = block
    else:
        list_start = lines.index("volumes:")
        insert_at = list_start + 1
        while insert_at < len(lines) and lines[insert_at].startswith("- "):
            insert_at += 1
        lines[insert_at:insert_at] = [f"- {key}"]
        lines[list_start:list_start] = block  # entry just above the list
    yaml_path.write_text("\n".join(lines) + "\n")
    return key


def rename_volume(yaml_path: pathlib.Path, adf: pathlib.Path, name: str) -> str | None:
    """Change the name: line of the volumeN entry whose adf_path is adf. Only that line changes.
    Returns the key, or None if no entry uses that ADF."""
    yaml_path = pathlib.Path(yaml_path).resolve()
    lines = yaml_path.read_text().splitlines()
    target = pathlib.Path(adf).resolve()
    key = None
    for index, line in enumerate(lines):
        match = re.match(r"^(volume\d+):\s*$", line)
        if match:
            key, block_start = match.group(1), index
            continue
        if not line.startswith("  "):
            key = None  # left the volumeN block
            continue
        if key is not None and line.startswith("  adf_path:"):
            path = pathlib.Path(line.split(":", 1)[1].strip()).expanduser()
            if not path.is_absolute():
                path = yaml_path.parent / path
            if path.resolve() != target:
                continue
            end = block_start + 1
            while end < len(lines) and lines[end].startswith("  "):
                end += 1
            for row in range(block_start + 1, end):
                if lines[row].startswith("  name:"):
                    lines[row] = f"  name: {json.dumps(name)}"
                    break
            else:
                lines.insert(block_start + 1, f"  name: {json.dumps(name)}")
            yaml_path.write_text("\n".join(lines) + "\n")
            return key
    return None


# ---------------------------------------------------------------- slice views


class SliceView(QtWidgets.QLabel):
    """One plane of the preview with the crop box. Clicking moves the box center in this plane."""

    clicked = pyqtSignal(float, float)  # column, row in preview voxels of this plane
    scrolled = pyqtSignal(int)  # wheel steps: move the box through this plane

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.setMinimumSize(260, 260)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: black;")
        self.image = None
        self.box = None  # (col, row, half_cols, half_rows)
        self.marks = ("", "")  # left / right edge letters
        self._scale = 1.0
        self._offset = (0, 0)

    def show_plane(self, plane: np.ndarray, box, marks=("", "")):
        low, high = WINDOW
        gray = (np.clip((plane.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        gray = np.ascontiguousarray(gray)
        self.image = QImage(gray.data, gray.shape[1], gray.shape[0], gray.shape[1], QImage.Format_Grayscale8).copy()
        self.box, self.marks = box, marks
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
        if self.box:
            col, row, half_c, half_r = (v * self._scale for v in self.box)
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.drawRect(int(col - half_c), int(row - half_r), int(2 * half_c), int(2 * half_r))
            painter.setPen(QPen(QColor(255, 200, 0, 140), 1, Qt.DashLine))
            painter.drawLine(int(col), 0, int(col), pix.height())
            painter.drawLine(0, int(row), pix.width(), int(row))
        painter.setPen(QColor(120, 200, 255))
        painter.drawText(6, 18, self.title)
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

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:  # drag the box
            self.mousePressEvent(event)

    def mousePressEvent(self, event):
        if self.image is None:
            return
        col = (event.x() - self._offset[0]) / self._scale
        row = (event.y() - self._offset[1]) / self._scale
        if 0 <= col < self.image.width() and 0 <= row < self.image.height():
            self.clicked.emit(col, row)


# ---------------------------------------------------------------- the tab


class AddVolumePanel(QtWidgets.QWidget):
    def __init__(self, ui, tbone_dir=None):
        """ui: the launcher (study_gui.Ui). It gets the new volume through ui.add_volume(vinfo)."""
        super().__init__()
        self.ui = ui
        self.tbone_dir = pathlib.Path(tbone_dir or DEFAULT_TBONE_DIR).expanduser()
        self.python = self.tbone_dir / ".venv" / "bin" / "python"
        self.series = []
        self.meta = None
        self.preview = None
        self.center = None  # LPS mm
        self.work = None
        self._job = None
        self._stdout = ""
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self._build()
        self._check_install()

    # ------------------------------------------------------------ layout

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)

        scan = QtWidgets.QGroupBox("1. CT scan (DICOM)")
        grid = QtWidgets.QGridLayout(scan)
        self.folder_button = QtWidgets.QPushButton("Choose DICOM folder...")
        self.folder_button.clicked.connect(self._choose_folder)
        self.folder_label = QtWidgets.QLabel("No folder chosen")
        self.folder_label.setWordWrap(True)
        self.series_combo = QtWidgets.QComboBox()
        self.series_combo.activated.connect(self._load_series)
        grid.addWidget(self.folder_button, 0, 0)
        grid.addWidget(self.folder_label, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Series"), 1, 0)
        grid.addWidget(self.series_combo, 1, 1)
        grid.setColumnStretch(1, 1)
        layout.addWidget(scan)

        ear = QtWidgets.QGroupBox("2. Ear and crop box (click or drag to move it; mouse wheel moves it through the slices)")
        ear_layout = QtWidgets.QVBoxLayout(ear)
        row = QtWidgets.QHBoxLayout()
        self.laterality_label = QtWidgets.QLabel("Ear: load a scan")
        self.laterality_label.setStyleSheet("font-weight: bold;")
        row.addWidget(self.laterality_label)
        self.side_group = QtWidgets.QButtonGroup(self)
        self.radio_left = QtWidgets.QRadioButton("Left")
        self.radio_right = QtWidgets.QRadioButton("Right")
        for index, radio in enumerate((self.radio_left, self.radio_right)):
            self.side_group.addButton(radio, index)
            row.addWidget(radio)
        self.side_group.buttonClicked.connect(self._side_changed)
        self.change_side = QtWidgets.QPushButton("Change")
        self.change_side.setToolTip("Only if the series name or scan position got the side wrong.")
        self.change_side.clicked.connect(lambda: self._show_side_choice(True))
        row.addWidget(self.change_side)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Box"))
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(30, 60)
        self.size_spin.setValue(50)
        self.size_spin.setSuffix(" mm")
        self.size_spin.setToolTip("The model was trained on 48-53 mm cubes around one temporal bone.\n"
                                  "Smaller uses less GPU memory but may cut off the mastoid tip or sinus.")
        self.size_spin.valueChanged.connect(self._update_views)
        row.addWidget(self.size_spin)
        self.box_label = QtWidgets.QLabel("")
        self.reset_box = QtWidgets.QPushButton("Reset box")
        self.reset_box.setToolTip("Back to the automatic placement.")
        self.reset_box.clicked.connect(self._reset_box)
        row.addWidget(self.reset_box)
        ear_layout.addLayout(row)
        self.position_label = QtWidgets.QLabel("")
        info = QtWidgets.QHBoxLayout()
        info.addWidget(self.position_label, 1)
        info.addWidget(self.box_label)
        ear_layout.addLayout(info)
        views = QtWidgets.QHBoxLayout()
        self.axial = SliceView("Axial")
        self.coronal = SliceView("Coronal")
        self.sagittal = SliceView("Sagittal")
        for view in (self.axial, self.coronal, self.sagittal):
            views.addWidget(view)
        self.axial.clicked.connect(lambda c, r: self._clicked("axial", c, r))
        self.coronal.clicked.connect(lambda c, r: self._clicked("coronal", c, r))
        self.sagittal.clicked.connect(lambda c, r: self._clicked("sagittal", c, r))
        self.axial.scrolled.connect(lambda n: self._nudge(2, n))  # up = superior
        self.coronal.scrolled.connect(lambda n: self._nudge(1, -n))  # up = anterior
        self.sagittal.scrolled.connect(lambda n: self._nudge(0, n * self._lateral_sign()))  # up = lateral
        for view in (self.axial, self.coronal, self.sagittal):
            view.setToolTip("Click or drag: move the box in this plane.\n"
                            "Mouse wheel: move it (and the slice shown) through this plane, 1 mm a step.")
        ear_layout.addLayout(views, 1)
        layout.addWidget(ear, 1)
        self._show_side_choice(False)

        seg = QtWidgets.QGroupBox("3. Segment and add to the anatomy list")
        grid = QtWidgets.QGridLayout(seg)
        grid.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Shown in the anatomy list, e.g. Patient scan L")
        grid.addWidget(self.name_edit, 0, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Folds"), 1, 0)
        self.folds_combo = QtWidgets.QComboBox()
        self.folds_combo.addItem("1 fold (recommended)", [0])
        self.folds_combo.addItem("3 folds (ensemble, more memory)", [0, 1, 2])
        self.folds_combo.addItem("5 folds (full ensemble, most memory)", [0, 1, 2, 3, 4])
        self.folds_combo.setToolTip("The model author recommends all 5 folds. Each fold is another network\n"
                                    "whose output is summed in host memory, so more folds take longer and\n"
                                    "can run out of memory. One fold was used on Windows and looks fine.")
        grid.addWidget(self.folds_combo, 1, 1)
        self.mirror_check = QtWidgets.QCheckBox("Test-time mirroring (more accurate, ~8x slower)")
        self.mirror_check.setChecked(True)
        grid.addWidget(self.mirror_check, 1, 2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Simulator voxels"), 2, 0)
        self.factor_combo = QtWidgets.QComboBox()
        self.factor_combo.addItem("0.2 mm (about 256 per side, like Anatomy A-I)", 2)
        self.factor_combo.addItem("0.1 mm (about 512 per side, full model resolution)", 1)
        grid.addWidget(self.factor_combo, 2, 1)
        self.go_button = QtWidgets.QPushButton("Segment and add volume")
        self.go_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.go_button.clicked.connect(self._run)
        grid.addWidget(self.go_button, 2, 2)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        grid.addWidget(self.cancel_button, 2, 3)
        self.start_check = QtWidgets.QCheckBox("Then start the simulator on it")
        self.start_check.setToolTip("When the volume is ready, select it and press Start simulation on\n"
                                    "the Main tab with that tab's settings (headset, RTX view, ...).")
        self.start_check.toggled.connect(self._start_check_toggled)
        grid.addWidget(self.start_check, 3, 2, 1, 2)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("")
        grid.addWidget(self.progress, 4, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        layout.addWidget(seg)

        bottom = QtWidgets.QHBoxLayout()
        self.result_icon = QtWidgets.QLabel()
        self.result_icon.setFixedSize(128, 128)
        bottom.addWidget(self.result_icon)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        bottom.addWidget(self.log, 1)
        layout.addLayout(bottom)
        self._set_enabled()

    def _start_check_toggled(self, checked):
        self.go_button.setText("Segment and start the simulator" if checked else "Segment and add volume")

    def _check_install(self):
        model = self.tbone_dir / "nnUNet_results" / "Dataset101_TemporalBone"
        if not self.python.is_file():
            self._say(f"Segmentation Python not found: {self.python}\n"
                      "Set tbone_seg_dir in gui_setup.yaml to the tbone-seg-nnunetv2 checkout.")
        elif not model.is_dir():
            self._say(f"Model not found in {model}.")

    # ------------------------------------------------------------ worker process

    def _start(self, job, args):
        self._job = job
        self._stdout = ""
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)
        self._process.setWorkingDirectory(str(WORK_ROOT.parent if WORK_ROOT.parent.is_dir() else REPO))
        self._process.start(str(self.python), ["-I", str(WORKER), *args, "--tbone-dir", str(self.tbone_dir)]
                            if job == "run" else ["-I", str(WORKER), *args])
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
        elif line.strip() and "it/s]" not in line and "s/it]" not in line:
            self.log.appendPlainText(line)

    def _finished(self, code, _status):
        if self._stdout:
            self._handle_line(self._stdout)
            self._stdout = ""
        job, self._job = self._job, None
        result, error = getattr(self, "_result", None), getattr(self, "_error", None)
        self._result = self._error = None
        if error or code != 0 or result is None:
            self._say(f"Failed: {error or f'exit code {code}'}")
            self.progress.setFormat("Failed")
        elif job == "scan":
            self._scanned(result["series"])
        elif job == "load":
            self._loaded(result)
        elif job == "run":
            self._done(result)
        self._set_enabled()

    def _cancel(self):
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._say("Cancelled.")

    def _set_enabled(self):
        busy = self._process.state() != QProcess.NotRunning
        self.folder_button.setEnabled(not busy)
        self.series_combo.setEnabled(not busy and bool(self.series))
        self.go_button.setEnabled(not busy and self.meta is not None and self._laterality() != "")
        self.cancel_button.setEnabled(busy)

    def _say(self, text):
        self.log.appendPlainText(text)

    # ------------------------------------------------------------ 1. scan

    def _choose_folder(self):
        start = str(pathlib.Path.home() / "Downloads")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "DICOM folder", start)
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder):
        self.folder_label.setText(folder)
        self.series, self.meta = [], None
        self.series_combo.clear()
        self.progress.setFormat("Looking for DICOM series...")
        self._start("scan", ["scan", folder])

    def _scanned(self, series):
        # Largest CT series first: usually the thin bone reconstruction.
        self.series = sorted(series, key=lambda s: (s.get("modality") != "CT", -s["slices"]))
        for s in self.series:
            label = f"{s.get('description') or '(no name)'}  |  {s['slices']} slices"
            if s.get("slice_thickness"):
                label += f", {float(s['slice_thickness']):g} mm"
            if s.get("modality") and s["modality"] != "CT":
                label += f"  [{s['modality']}]"
            self.series_combo.addItem(label)
        if not self.series:
            self._say("No DICOM series found in that folder.")
            self.progress.setFormat("")
            return
        self._say(f"Found {len(self.series)} series.")
        self._load_series(0)

    def _load_series(self, index):
        if not (0 <= index < len(self.series)):
            return
        s = self.series[index]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work = WORK_ROOT / stamp
        self.work.mkdir(parents=True, exist_ok=True)
        self.meta = None
        self._start("load", ["load", s["folder"], s["uid"], str(self.work)])

    # ------------------------------------------------------------ 2. ear and box

    def _loaded(self, meta):
        self.meta = meta
        self.preview = np.load(pathlib.Path(meta["work"]) / "preview.npy")
        side = meta.get("laterality") or ""
        self.radio_left.setChecked(side == "left")
        self.radio_right.setChecked(side == "right")
        if side:
            self.laterality_label.setText(f"Ear: {side.upper()} (from the {meta['laterality_source']})")
            self._show_side_choice(False)
        else:
            self.side_group.setExclusive(False)
            self.radio_left.setChecked(False)
            self.radio_right.setChecked(False)
            self.side_group.setExclusive(True)
            self.laterality_label.setText("Ear: which side? The scan does not say.")
            self._show_side_choice(True)
        self.center = list(meta["ear_centers"].get(side or "right"))
        if not self.name_edit.text().strip():
            self.name_edit.setText(self._default_name(meta, side))
        self._update_views()
        self._set_enabled()

    @staticmethod
    def _default_name(meta, side):
        words = re.sub(r"\s+", " ", meta.get("description") or "CT").strip()
        return f"{words} ({side[0].upper()})" if side else words

    def _show_side_choice(self, show):
        self.radio_left.setVisible(show)
        self.radio_right.setVisible(show)
        self.change_side.setVisible(not show and self.meta is not None)

    def _laterality(self):
        return "left" if self.radio_left.isChecked() else "right" if self.radio_right.isChecked() else ""

    def _side_changed(self, *_):
        side = self._laterality()
        self.laterality_label.setText(f"Ear: {side.upper()} (chosen)")
        if self.meta:
            self.center = list(self.meta["ear_centers"][side])
            self._update_views()
        self._set_enabled()

    def _to_index(self, point):
        """LPS mm -> preview (i, j, k)."""
        direction = np.array(self.meta["preview_direction"]).reshape(3, 3)
        offset = np.array(point) - np.array(self.meta["preview_origin"])
        return (direction.T @ offset) / np.array(self.meta["preview_spacing"])

    def _to_point(self, index):
        direction = np.array(self.meta["preview_direction"]).reshape(3, 3)
        return (np.array(self.meta["preview_origin"]) +
                direction @ (np.array(index) * np.array(self.meta["preview_spacing"]))).tolist()

    def _lateral_sign(self):
        return -1 if self._laterality() == "right" else 1  # LPS x grows toward the patient's left

    def _nudge(self, axis, steps):
        """Move the box steps mm along LPS axis (0 x, 1 y, 2 z)."""
        if self.meta is None:
            return
        self.center[axis] += float(steps)
        self._update_views()

    def _auto_center(self):
        side = self._laterality() or "right"
        return list(self.meta["ear_centers"][side])

    def _reset_box(self):
        if self.meta is not None:
            self.center = self._auto_center()
            self._update_views()

    def _clicked(self, plane, col, row):
        if self.meta is None:
            return
        i, j, k = self._to_index(self.center)
        depth = self.preview.shape[0]
        if plane == "axial":
            i, j = col, row
        elif plane == "coronal":
            i, k = col, depth - 1 - row
        else:
            j, k = col, depth - 1 - row
        self.center = self._to_point([i, j, k])
        self._update_views()

    def _update_views(self, *_):
        if self.meta is None or self.preview is None:
            return
        depth, rows, cols = self.preview.shape
        i, j, k = self._to_index(self.center)
        i, j, k = (int(np.clip(round(v), 0, n - 1)) for v, n in zip((i, j, k), (cols, rows, depth)))
        size = self.size_spin.value()
        half = np.array([size / 2.0 / s for s in self.meta["preview_spacing"]])  # i, j, k
        flipped = depth - 1 - k
        # Radiological convention: the patient's right on the left of the screen (LPS i grows to the left).
        self.axial.show_plane(self.preview[k], (i, j, half[0], half[1]), ("R", "L"))
        self.coronal.show_plane(self.preview[::-1, j, :], (i, flipped, half[0], half[2]), ("R", "L"))
        self.sagittal.show_plane(self.preview[::-1, :, i], (j, flipped, half[1], half[2]), ("A", "P"))
        voxels = int(round(size / MODEL_SPACING_MM))
        self.box_label.setText(f"Model input: {voxels}\u00b3 voxels at {MODEL_SPACING_MM} mm")
        moved = np.array(self.center) - np.array(self._auto_center())
        lateral = moved[0] * self._lateral_sign()
        self.position_label.setText(
            "Box moved from the automatic spot: "
            f"{abs(lateral):.0f} mm {'lateral' if lateral >= 0 else 'medial'}, "
            f"{abs(moved[1]):.0f} mm {'posterior' if moved[1] >= 0 else 'anterior'}, "
            f"{abs(moved[2]):.0f} mm {'superior' if moved[2] >= 0 else 'inferior'}")

    # ------------------------------------------------------------ 3. run

    def _run(self):
        name = self.name_edit.text().strip()
        side = self._laterality()
        if not name:
            self._say("Give the volume a name first.")
            return
        if not side:
            self._say("Choose the ear first.")
            return
        folds = self.folds_combo.currentData()
        args = ["run", self.meta["work"], "--center", *[f"{v:.3f}" for v in self.center],
                "--size", str(self.size_spin.value()), "--folds", *map(str, folds),
                "--name", name, "--laterality", side, "--factor", str(self.factor_combo.currentData())]
        if not self.mirror_check.isChecked():
            args.append("--no-mirroring")
        self._say(f"Segmenting {name} ({side} ear, {len(folds)} fold{'s' if len(folds) > 1 else ''})...")
        self.result_icon.clear()
        self._start("run", args)

    def _log_crop(self, result):
        """One line per segmentation: where the automatic box was, where the proctor put it, and
        where the ossicles, labyrinth, facial nerve etc. landed. Learning data for an automatic crop."""
        import hashlib

        meta = self.meta or {}
        entry = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "volume": result["name"],
            "adf": pathlib.Path(result["adf"]).name,
            "series": hashlib.sha256(meta.get("series_uid", "").encode()).hexdigest()[:16],
            "description": meta.get("description", ""),
            "laterality": result["laterality"],
            "laterality_source": meta.get("laterality_source") or "chosen",
            "scan_size": meta.get("size"),
            "scan_spacing_mm": meta.get("spacing"),
            "scan_origin_mm": meta.get("origin"),
            "auto_center_mm": meta.get("ear_centers", {}).get(result["laterality"]),
            "folds": result["folds"],
            "mirroring": result.get("mirroring"),
            **result.get("crop", {}),
        }
        try:
            with CROP_LOG.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError as exc:
            self._say(f"Could not write the crop log: {exc}")

    def _done(self, result):
        name = result["name"]
        folds = ", ".join(map(str, result["folds"]))
        description = (f"{name}: {result['laterality']} ear, segmented from CT by tbone nnU-Net "
                       f"(fold {folds}, {result['voxel_mm']:.3f} mm), "
                       f"{datetime.date.today().isoformat()}.")
        yaml_path = HERE / "gui_setup.yaml"
        key = register_volume(yaml_path, name, pathlib.Path(result["adf"]), pathlib.Path(result["icon"]),
                              description)
        pix = QPixmap(result["icon"])
        if not pix.isNull():
            self.result_icon.setPixmap(pix.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._say(f"Done in {result['seconds']:.0f} s (GPU peak {result['peak_gpu_gib']:.1f} GiB). "
                  f"Added '{name}' as {key}: {pathlib.Path(result['adf']).name}")
        self._log_crop(result)
        add = getattr(self.ui, "add_volume", None)
        if add is not None:
            add(name, result["adf"], result["icon"], description)
        start = getattr(self.ui, "start_simulation_on_selected_volume", None)
        if self.start_check.isChecked() and start is not None:
            message = start()
            if message:
                self._say(message)
