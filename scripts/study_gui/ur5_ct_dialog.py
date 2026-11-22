"""UR5 tab: "Add specimens from CT".

Choose a CT (NRRD or NIfTI). The worker (scripts/segmentation/ur5_ct_worker.py) splits it into
bones and segments bone + inner ear. Each bone is shown over the CT; the ones ticked and named
here become UR5 specimens. The worker runs under a hard memory cap, so a run that goes over it is
stopped on its own and cannot take the desktop down.
"""
from __future__ import annotations

import datetime
import json
import pathlib

from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QPixmap

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKER = REPO / "scripts" / "segmentation" / "ur5_ct_worker.py"
DEFAULT_TBONE_DIR = REPO.parent / "tbone-seg-nnunetv2"
WORK_ROOT = REPO / "resources" / "intermediate_volumes" / "ur5_ct"
MEMORY_MAX = "12G"
SWAP_MAX = "1G"


def capped_command(python, arguments):
    """(program, args) that run the worker in its own memory-limited scope."""
    return "systemd-run", ["--user", "--scope", "--quiet", "-p", f"MemoryMax={MEMORY_MAX}",
                           "-p", f"MemorySwapMax={SWAP_MAX}", str(python), "-u", str(WORKER)] + list(arguments)


class BoneCard(QtWidgets.QGroupBox):
    def __init__(self, info, default_name):
        super().__init__(f"Bone {info['number']}")
        self.info = info
        layout = QtWidgets.QHBoxLayout(self)
        picture = QtWidgets.QLabel("No preview")
        picture.setAlignment(Qt.AlignCenter)
        if info.get("preview"):
            pixmap = QPixmap(info["preview"])
            if not pixmap.isNull():
                picture.setPixmap(pixmap.scaledToHeight(min(pixmap.height(), 320), Qt.SmoothTransformation))
        layout.addWidget(picture, 3)
        side = QtWidgets.QVBoxLayout()
        self.check = QtWidgets.QCheckBox("Add as a specimen")
        self.check.setChecked(bool(info["usable"]) and not info.get("warning"))
        self.check.setEnabled(bool(info["usable"]))
        side.addWidget(self.check)
        self.name = QtWidgets.QLineEdit(default_name)
        side.addWidget(QtWidgets.QLabel("Name:"))
        side.addWidget(self.name)
        where = info["side"].capitalize() + " ear" if info.get("side") else "Side unknown"
        text = QtWidgets.QLabel(f"{where}\nBone {info['bone_ml']} mL\nInner ear {info['inner_ear_ml']} mL\n"
                                f"{info['voxel_mm']:.2f} mm voxels")
        side.addWidget(text)
        if info.get("warning"):
            warning = QtWidgets.QLabel(info["warning"])
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #F0B35E;")
            side.addWidget(warning)
        side.addStretch(1)
        layout.addLayout(side, 1)


class AddFromCtDialog(QtWidgets.QDialog):
    def __init__(self, panel, python):
        super().__init__(panel)
        self.panel = panel
        self.python = pathlib.Path(python)
        self.result = None
        self.cards = []
        self.queue = []
        self.current = None
        self.added = []
        self.error = ""
        self._buffer = ""
        self.setWindowTitle("Add specimens from CT")
        self.resize(1120, 820)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._build()
        self._update()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit()
        self.path.setPlaceholderText("CT file (.nrrd, .nii, .nii.gz)")
        self.path.textChanged.connect(self._update)
        self.button_browse = QtWidgets.QPushButton("Choose...")
        self.button_browse.clicked.connect(self.choose)
        self.button_run = QtWidgets.QPushButton("Segment")
        self.button_run.clicked.connect(self.start)
        self.button_cancel = QtWidgets.QPushButton("Cancel")
        self.button_cancel.clicked.connect(self.cancel)
        for widget in (self.path, self.button_browse, self.button_run, self.button_cancel):
            row.addWidget(widget, 1 if widget is self.path else 0)
        layout.addLayout(row)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        layout.addWidget(self.bar)
        self.status = QtWidgets.QLabel("Choose a CT. A scan holding several bones is split automatically.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.area = QtWidgets.QScrollArea()
        self.area.setWidgetResizable(True)
        self.holder = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.holder)
        self.cards_layout.addStretch(1)
        self.area.setWidget(self.holder)
        layout.addWidget(self.area, 1)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.button_add = QtWidgets.QPushButton("Add selected specimens")
        self.button_add.clicked.connect(self.add_selected)
        self.button_close = QtWidgets.QPushButton("Close")
        self.button_close.clicked.connect(self.close)
        bottom.addWidget(self.button_add)
        bottom.addWidget(self.button_close)
        layout.addLayout(bottom)

    def busy(self):
        return self.process.state() != QProcess.NotRunning

    def _update(self, *_):
        busy = self.busy()
        path = pathlib.Path(self.path.text().strip()).expanduser() if self.path.text().strip() else None
        self.button_run.setEnabled(not busy and path is not None and path.is_file())
        self.button_browse.setEnabled(not busy)
        self.path.setEnabled(not busy)
        self.button_cancel.setEnabled(busy)
        self.button_add.setEnabled(not busy and any(c.check.isChecked() and c.isEnabled() for c in self.cards))
        self.button_close.setEnabled(not busy)

    def choose(self):
        start = str(pathlib.Path.home() / "Downloads")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a CT", start, "CT (*.nrrd *.nii *.nii.gz);;All files (*)")
        if path:
            self.path.setText(path)

    def _clear_cards(self):
        for card in self.cards:
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self.cards = []

    def start(self):
        if self.panel.sim_running():
            self.status.setText("Close the simulator first: segmentation needs the GPU.")
            return
        source = pathlib.Path(self.path.text().strip()).expanduser()
        work = WORK_ROOT / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{source.name.split('.')[0]}"
        self._clear_cards()
        self.result = None
        self.error = ""
        self.current = "run"
        self.bar.setValue(0)
        self.status.setText("Starting...")
        program, args = capped_command(self.python, ["run", str(source), "--work", str(work),
                                                     "--tbone_dir", str(self.panel.tbone_dir())])
        self.process.start(program, args)
        self._update()

    def cancel(self):
        if self.busy():
            self.queue = []
            self.current = "cancelled"
            self.process.terminate()
            self.status.setText("Cancelling...")

    def add_selected(self):
        names = [c.name.text().strip() for c in self.cards if c.check.isChecked() and c.isEnabled()]
        known = {s["name"].casefold() for s in self.panel.specimens}
        problem = next((n for n in names if not n or n.casefold() in known or
                        sum(m.casefold() == n.casefold() for m in names) > 1), None)
        if problem is not None:
            self.status.setText(f"Each specimen needs its own new name ({problem or 'empty name'}).")
            return
        self.queue = [c for c in self.cards if c.check.isChecked() and c.isEnabled()]
        self._next_add()

    def _next_add(self):
        if not self.queue:
            self.current = None
            if self.added:
                self.status.setText("Added: " + ", ".join(self.added) + ". They are in the UR5 specimen list.")
            self._update()
            return
        card = self.queue[0]
        self.current = "add"
        self.status.setText(f"Adding {card.name.text().strip()}...")
        program, args = capped_command(self.python, ["add", self.result["work"], "--bone", str(card.info["number"]),
                                                     "--name", card.name.text().strip()])
        self.process.start(program, args)
        self._update()

    # ---- worker output ----
    def _read(self):
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.handle_line(line)

    def handle_line(self, line):
        if line.startswith("@progress "):
            parts = line.split(" ", 2)
            try:
                self.bar.setValue(int(float(parts[1]) * 1000))
            except (ValueError, IndexError):
                return
            if len(parts) > 2:
                self.status.setText(parts[2])
        elif line.startswith("@error "):
            self.error = line[7:]
        elif line.startswith("@result "):
            payload = json.loads(line[8:])
            if self.current == "run":
                self.show_result(payload)
            elif self.current == "add":
                card = self.queue.pop(0)
                card.check.setChecked(False)
                card.setEnabled(False)
                self.added.append(payload["name"])
                self.panel.add_specimen(payload)

    def show_result(self, result):
        self.result = result
        stem = result["source_file"].split(".")[0].replace("_CT", "")
        for info in result["bones"]:
            card = BoneCard(info, f"{stem} bone {info['number']}")
            card.check.toggled.connect(self._update)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
            self.cards.append(card)
        count = len(result["bones"])
        self.status.setText(f"Found {count} bone{'s' if count != 1 else ''}. Check each one, then add the ones you want.")
        self._update()

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self.error = "The segmentation could not start (systemd-run or the tbone-seg environment is missing)."

    def _finished(self, code, status):
        self._read()
        cancelled = self.current == "cancelled"
        was_add = self.current == "add"
        if cancelled:
            self.current = None
            self.queue = []
            self.bar.setValue(0)
            self.status.setText("Cancelled.")
        elif self.error:
            self.current = None
            self.queue = []
            self.status.setText(f"ERROR! {self.error}")
        elif code != 0 or status != QProcess.NormalExit:
            self.current = None
            self.queue = []
            self.status.setText(f"ERROR! The segmentation stopped (code {code}). It may have reached the "
                                f"{MEMORY_MAX} memory cap; the rest of the computer is unaffected.")
        elif was_add:
            self._next_add()
            return
        else:
            self.current = None
        self._update()

    def closeEvent(self, event):
        if self.busy():
            self.status.setText("Cancel the running job first.")
            event.ignore()
            return
        super().closeEvent(event)
