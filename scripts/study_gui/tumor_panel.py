"""The "Tumor" box of the Register MRI tab: automatic + click-to-segment on a registered MRI.

Drives scripts/registration/tumor_worker.py, which stays running in ../nninteractive/.venv
(scripts/segmentation/setup_tumor_seg.sh installs it) so the models load once. The
tumor series may be any series from the registered MRI's session (same DICOM frame of
reference), e.g. the C+ T1 next to the C+ FIESTA used for the registration: a contrast T1
is the better input for both models.

* Find tumor automatically: the Kujawa et al. 2024 vestibular schwannoma nnU-Net
  (contrast T1, or T2 for CISS / FIESTA) -> part 1.
* Click mode: click a view to add to the current part (Shift+click removes), via
  nnInteractive. "New part" starts another (e.g. a middle fossa extension the automatic
  model missed); Undo goes back one click at a time.
The tab shows the parts over its CT views; Save writes them on the CT grid
(<registration>/tumor/tumor_on_ct.nii.gz), which "Open in 3D Slicer" also loads.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, QProcessEnvironment, pyqtSignal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
TUMOR_WORKER = REPO / "scripts" / "registration" / "tumor_worker.py"
DEFAULT_PYTHON = REPO.parent / "nninteractive" / ".venv" / "bin" / "python"
PYTHON_SETTING = "nninteractive_python"
PART_COLORS = [(255, 128, 26), (255, 217, 38), (64, 200, 255), (255, 80, 200), (120, 255, 120)]


def tumor_series_order(series: list) -> list:
    """MR series, a post-contrast T1 first (the better input), then by slice count."""
    def rank(s):
        text = (s.get("description") or "").lower()
        contrast = bool(re.search(r"c\+|post|gad|gd|contrast|\+c", text))
        t1 = "t1" in text or "mprage" in text or "bravo" in text or "spgr" in text
        return (not (contrast and t1), not t1, -s.get("slices", 0))

    return sorted([s for s in series if s.get("modality") in ("MR", "", None)], key=rank)


class TumorPanel(QtWidgets.QGroupBox):
    labels_changed = pyqtSignal()  # preview labels (self.preview) changed: redraw

    def __init__(self, python=None, say=print):
        super().__init__("4. Tumor (optional): automatic, then click to add (Shift+click removes)")
        self.python = pathlib.Path(python or DEFAULT_PYTHON).expanduser()
        self.say = say
        self.work = None
        self.series = []
        self.preview = None  # uint8 labels on the tab's preview grid
        self.parts = []
        self.current = -1
        self.opened_uid = None
        self._queue = []
        self._waiting = None  # command name the worker is busy with
        self._buffer = ""
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read)
        self._process.finished.connect(self._exited)
        self._build()
        if not self.python.is_file():
            self.status.setText(f"Not installed: {self.python} (run scripts/segmentation/setup_tumor_seg.sh)")
        self._set_enabled()

    # ------------------------------------------------------------ layout

    def _build(self):
        grid = QtWidgets.QGridLayout(self)
        self.folder_button = QtWidgets.QPushButton("Tumor MRI folder...")
        self.folder_button.setToolTip("A series from the registered MRI's session, e.g. the C+ T1.\n"
                                      "The MRI folder chosen above is listed already.")
        self.folder_button.clicked.connect(self._choose_folder)
        grid.addWidget(self.folder_button, 0, 0)
        self.series_combo = QtWidgets.QComboBox()
        self.series_combo.setMinimumWidth(360)
        self.series_combo.activated.connect(self._open_chosen)
        grid.addWidget(self.series_combo, 0, 1, 1, 2)
        self.auto_button = QtWidgets.QPushButton("Find tumor automatically")
        self.auto_button.setToolTip("Vestibular schwannoma nnU-Net (Kujawa et al. 2024). Typical CPA / IAC tumors;\n"
                                    "an unusual extension may need clicks.")
        self.auto_button.clicked.connect(lambda: self._send({"cmd": "auto"}))
        grid.addWidget(self.auto_button, 0, 3)
        self.click_check = QtWidgets.QCheckBox("Click mode")
        self.click_check.setToolTip("Click in a view: add to the current part. Shift+click: remove.")
        grid.addWidget(self.click_check, 1, 0)
        self.parts_combo = QtWidgets.QComboBox()
        self.parts_combo.activated.connect(lambda i: self._send({"cmd": "select", "index": i}))
        grid.addWidget(self.parts_combo, 1, 1)
        row = QtWidgets.QHBoxLayout()
        self.new_button = QtWidgets.QPushButton("New part")
        self.new_button.clicked.connect(lambda: self._send({"cmd": "new_part"}))
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.clicked.connect(lambda: self._send({"cmd": "undo"}))
        self.delete_button = QtWidgets.QPushButton("Delete part")
        self.delete_button.clicked.connect(lambda: self._send({"cmd": "delete", "index": self.current}))
        self.save_button = QtWidgets.QPushButton("Save on CT")
        self.save_button.clicked.connect(lambda: self._send({"cmd": "save"}))
        for button in (self.new_button, self.undo_button, self.delete_button, self.save_button):
            row.addWidget(button)
        grid.addLayout(row, 1, 2, 1, 2)
        self.status = QtWidgets.QLabel("Register first, then choose the tumor series.")
        self.status.setWordWrap(True)
        grid.addWidget(self.status, 2, 0, 1, 4)
        grid.setColumnStretch(1, 1)

    def _set_enabled(self):
        idle = self._waiting is None and not self._queue
        installed = self.python.is_file()
        opened = self.opened_uid is not None
        self.folder_button.setEnabled(installed and idle)
        self.series_combo.setEnabled(installed and idle and bool(self.series) and self.work is not None)
        self.auto_button.setEnabled(installed and idle and opened)
        self.click_check.setEnabled(installed and opened)
        self.parts_combo.setEnabled(idle and bool(self.parts))
        self.new_button.setEnabled(idle and opened)
        has_part = 0 <= self.current < len(self.parts)
        self.undo_button.setEnabled(idle and has_part and self.parts[self.current].get("undo", 0) > 0)
        self.delete_button.setEnabled(idle and has_part)
        self.save_button.setEnabled(idle and bool(self.parts))

    # ------------------------------------------------------------ the tab's calls

    def set_work(self, work: str | None):
        """A registration was shown: its folder. Reopens the series chosen before, if any."""
        changed = work != self.work
        self.work = work
        self.preview = None
        if changed:
            self.opened_uid, self.parts, self.current = None, [], -1
            self._update_parts()
        if work:
            preview = pathlib.Path(work) / "tumor" / "preview_tumor.npy"
            if preview.is_file():
                self.preview = np.load(preview)
            meta = pathlib.Path(work) / "tumor" / "tumor.json"
            if meta.is_file() and changed:
                saved = json.loads(meta.read_text()).get("series", {})
                if saved.get("uid"):
                    self.offer_series([{"uid": saved["uid"], "folder": saved["folder"], "modality": "MR",
                                        "description": saved.get("description", ""), "slices": 0}])
                    index = next(i for i, s in enumerate(self.series) if s["uid"] == saved["uid"])
                    self.series_combo.setCurrentIndex(index)
                    self._open_chosen(index)  # restores the saved parts
        self.labels_changed.emit()
        self._set_enabled()

    def offer_series(self, series: list):
        """Add MR series (from the tab's scans) to the list; a contrast T1 goes first."""
        known = {s["uid"] for s in self.series}
        self.series = tumor_series_order(self.series + [s for s in series if s["uid"] not in known])
        self.series_combo.clear()
        for s in self.series:
            text = s.get("description") or "(no name)"
            if s.get("slices"):
                text += f"  |  {s['slices']} slices"
            self.series_combo.addItem(text)
        self._set_enabled()

    def click(self, lps, positive: bool):
        """The tab's views were clicked in click mode: CT-space point (LPS mm)."""
        if self.click_check.isChecked() and self.opened_uid is not None:
            self._send({"cmd": "click", "lps": [float(v) for v in lps], "positive": positive})
            return True
        return False

    def shutdown(self):
        if self._process.state() != QProcess.NotRunning:
            self._process.write(b'{"cmd": "quit"}\n')
            self._process.waitForFinished(2000)
            self._process.kill()

    # ------------------------------------------------------------ series

    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Tumor MRI DICOM folder",
                                                            str(pathlib.Path.home() / "Downloads"))
        if folder:
            self._send({"cmd": "scan", "folder": folder})

    def _open_chosen(self, index=None):
        index = self.series_combo.currentIndex() if index is None else index
        if self.work and 0 <= index < len(self.series):
            s = self.series[index]
            self._send({"cmd": "open", "work": self.work, "folder": s["folder"], "uid": s["uid"]})

    # ------------------------------------------------------------ worker

    def _send(self, command):
        if not self.python.is_file():
            return
        if self._process.state() == QProcess.NotRunning:
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONUNBUFFERED", "1")
            self._process.setProcessEnvironment(environment)
            self._process.setWorkingDirectory(str(REPO))
            self._process.start(str(self.python), ["-u", str(TUMOR_WORKER)])
        self._queue.append(command)
        self._pump()

    def _pump(self):
        if self._waiting is None and self._queue:
            command = self._queue.pop(0)
            self._waiting = command["cmd"]
            self.status.setText({"auto": "Finding the tumor (about a minute the first time)...",
                                 "click": "Segmenting...", "open": "Opening the series...",
                                 "save": "Writing on the CT grid...", "scan": "Looking for series..."}
                                .get(self._waiting, "Working..."))
            self._process.write((json.dumps(command) + "\n").encode())
        self._set_enabled()

    def _read(self):
        self._buffer += bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if line.startswith("@result "):
                self._handle(json.loads(line[len("@result "):]))
            elif line.startswith("@error "):
                error = json.loads(line[len("@error "):])
                self.status.setText(f"Failed: {error.get('error')}")
                self.say(f"Tumor: {error.get('error')}")
                self._done()
            elif line.startswith("@progress "):
                self.status.setText(line.split(" ", 2)[2])

    def _handle(self, result):
        name = result.get("cmd")
        if name == "scan":
            self.offer_series(result.get("series", []))
            self.status.setText(f"{len(result.get('series', []))} series found.")
            if self.opened_uid is None and self.work and self.series:
                self.series_combo.setCurrentIndex(0)  # a contrast T1 if there is one
                self._open_chosen(0)
        else:
            if name == "open":
                index = self.series_combo.currentIndex()
                self.opened_uid = self.series[index]["uid"] if 0 <= index < len(self.series) else "?"
            self.parts, self.current = result.get("parts", []), result.get("current", -1)
            if result.get("preview") and os.path.isfile(result["preview"]):
                self.preview = np.load(result["preview"])
                self.labels_changed.emit()
            self._update_parts()
            total = sum(p["ml"] for p in self.parts)
            text = result.get("message", "")
            if self.parts:
                text += f". Total {total:.2f} mL."
            if result.get("ct_file") and name in ("save", "auto"):
                text += " Saved on the CT grid."
            self.status.setText(text)
        self._done()

    def _done(self):
        self._waiting = None
        self._pump()

    def _exited(self, code, _status):
        if self._waiting is not None:
            self.status.setText(f"The tumor worker stopped (exit code {code}).")
            self.say(f"Tumor worker exited with code {code}.")
        self._waiting, self._queue = None, []
        self._set_enabled()

    def _update_parts(self):
        self.parts_combo.clear()
        for n, part in enumerate(self.parts):
            self.parts_combo.addItem(f"Part {n + 1}: {part['ml']:.2f} mL ({part['source']})")
        if 0 <= self.current < len(self.parts):
            self.parts_combo.setCurrentIndex(self.current)
