#!/usr/bin/env python3
"""Pick an anatomy and a named view, then return tuner launch arguments.

The Newton venv has no PyQt5. The desktop launcher and ``pick_launch()``
run this file with system Python so Qt never loads in the tuner process.
The study GUI can skip the menu and pass --anatomy-adf / --view itself.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
SETUP = REPO / "scripts" / "study_gui" / "gui_setup.yaml"
VIEWS = REPO / "ADF" / "world" / "canonical_views.yaml"


def _laterality(adf_path: pathlib.Path):
    try:
        data = yaml.safe_load(adf_path.read_text())
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    for value in data.values():
        if isinstance(value, dict) and "laterality" in value:
            lat = str(value["laterality"]).strip().lower()
            if lat in ("left", "right"):
                return lat
    return ""


def load_volumes():
    data = yaml.safe_load(SETUP.read_text()) or {}
    volumes = []
    for key in data.get("volumes") or []:
        item = data.get(key) or {}
        adf = (SETUP.parent / str(item.get("adf_path", ""))).expanduser().resolve()
        if not adf.is_file():
            continue
        icon = (SETUP.parent / str(item.get("icon_path", ""))).expanduser().resolve()
        volumes.append(
            {
                "name": str(item.get("name") or adf.stem),
                "adf": adf,
                "icon": icon,
                "description": str(item.get("description") or ""),
                "laterality": _laterality(adf),
            }
        )
    return volumes


def load_views():
    data = yaml.safe_load(VIEWS.read_text()) or {}
    views = []
    for item in data.get("views") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        views.append(
            {
                "id": str(item["id"]),
                "label": str(item.get("label") or item["id"]),
                "laterality": str(item.get("laterality") or "").strip().lower(),
                "valid": bool(item.get("valid")),
            }
        )
    return views


def volume_title(adf):
    """Study-GUI name for an anatomy ADF, or the file stem."""
    path = pathlib.Path(adf).expanduser().resolve()
    for volume in load_volumes():
        if volume["adf"] == path:
            return volume["name"]
    return path.stem.replace("volume_", "")


def _parse_choice(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    mapping = {}
    index = 0
    while index < len(lines) - 1:
        if lines[index].startswith("--"):
            mapping[lines[index]] = lines[index + 1]
            index += 2
        else:
            index += 1
    adf = mapping.get("--anatomy-adf")
    if not adf:
        return None
    path = pathlib.Path(adf).expanduser().resolve()
    return {
        "adf": path,
        "view": mapping.get("--view") or "default",
        "name": volume_title(path),
        "close_all": mapping.get("--close-all") in ("1", "true", "yes"),
    }


def pick_launch():
    """Show the menu in system Python. Returns a choice dict, or None if cancelled."""
    dialog_python = os.environ.get("DIALOG_PYTHON", "python3")
    try:
        proc = subprocess.run(
            [dialog_python, str(HERE / "tuner_launch_dialog.py")],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"Could not open the volume picker: {exc}", flush=True)
        return None
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        return None
    return _parse_choice(proc.stdout)


def _tuner_command(adf, view, close_all=False):
    python = REPO / "middle_ear_newton" / ".venv" / "bin" / "python"
    script = HERE / "fivrs_rtx_tuner.py"
    command = [str(python), str(script), "--anatomy-adf", str(adf), "--view", str(view)]
    if close_all:
        command.append("--close-all")
    return command


class Picker:
    """Qt dialog. Built only when this file is the process entry point."""

    def __init__(self, qt_widgets, volumes, views, launch=False):
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap

        self.QtWidgets = qt_widgets
        self.Qt = Qt
        self.QPixmap = QPixmap
        self.volumes = volumes
        self.views = views
        self.launch = launch
        self.chosen = None
        self.dialog = qt_widgets.QDialog()
        self.dialog.setWindowTitle("FIVRS RTX Tuner")
        self.dialog.resize(680, 640)
        self.dialog.setStyleSheet(
            """
            QDialog { background: #0B1C2C; color: #E6EEF5; font-size: 16px; }
            QLabel { color: #E6EEF5; }
            QListWidget, QComboBox {
                background: #0F2433; color: #E6EEF5; border: 1px solid #1E3A4F;
                padding: 6px; font-size: 16px;
            }
            QCheckBox { color: #E6EEF5; font-size: 15px; spacing: 8px; }
            QPushButton {
                background: #1F6F8B; color: #E6EEF5; border: 1px solid #2F8FBF;
                border-radius: 4px; padding: 10px 18px; font-size: 16px; min-height: 36px;
            }
            QPushButton:hover { background: #2589A8; }
            """
        )
        layout = qt_widgets.QVBoxLayout(self.dialog)
        self._form_page = qt_widgets.QWidget()
        form = qt_widgets.QVBoxLayout(self._form_page)
        title = qt_widgets.QLabel("Choose a volume and a view")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        form.addWidget(title)

        form.addWidget(qt_widgets.QLabel("Anatomy"))
        self.list = qt_widgets.QListWidget()
        for volume in volumes:
            side = volume["laterality"].upper() if volume["laterality"] else "—"
            item = qt_widgets.QListWidgetItem(f"{volume['name']}   ({side})")
            item.setData(Qt.UserRole, volume)
            self.list.addItem(item)
        self.list.currentRowChanged.connect(self._refresh_views)
        self.list.itemDoubleClicked.connect(lambda *_: self._accept())
        form.addWidget(self.list, 1)

        self.preview = qt_widgets.QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(120)
        form.addWidget(self.preview)

        self.description = qt_widgets.QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color: #9DB4C4; font-size: 14px;")
        form.addWidget(self.description)

        form.addWidget(qt_widgets.QLabel("View"))
        self.view_combo = qt_widgets.QComboBox()
        form.addWidget(self.view_combo)
        self.close_all = qt_widgets.QCheckBox("Close every structure")
        form.addWidget(self.close_all)
        close_hint = qt_widgets.QLabel(
            "Each color is meshed as its own solid, so unnamed hexes can be named. Opening takes longer."
        )
        close_hint.setWordWrap(True)
        close_hint.setStyleSheet("color: #9DB4C4; font-size: 13px;")
        form.addWidget(close_hint)

        buttons = qt_widgets.QHBoxLayout()
        launch = qt_widgets.QPushButton("Open Tuner")
        cancel = qt_widgets.QPushButton("Cancel")
        launch.clicked.connect(self._accept)
        cancel.clicked.connect(self.dialog.reject)
        buttons.addWidget(cancel)
        buttons.addWidget(launch)
        form.addLayout(buttons)
        layout.addWidget(self._form_page)

        self._load_page = qt_widgets.QWidget()
        self._load_page.hide()
        load = qt_widgets.QVBoxLayout(self._load_page)
        import rtx_splash

        self._load_status, self._load_bar = rtx_splash.add_loading_widgets(
            qt_widgets, load, "Opening RTX Tuner", cancelable=True, on_cancel=self.dialog.reject
        )
        load.addStretch(1)
        layout.addWidget(self._load_page)

        preferred = next(
            (i for i, volume in enumerate(volumes) if volume["name"] == "AMBF Delta"),
            next((i for i, volume in enumerate(volumes) if "Delta" in volume["name"]), 0),
        )
        if volumes:
            self.list.setCurrentRow(preferred)

    def exec_(self):
        return self.dialog.exec_()

    def _current_volume(self):
        item = self.list.currentItem()
        return item.data(self.Qt.UserRole) if item else None

    def _refresh_views(self):
        volume = self._current_volume()
        self.view_combo.clear()
        self.view_combo.addItem("Default (laterality)", "default")
        self.view_combo.addItem("Home", "home")
        lat = (volume or {}).get("laterality") or ""
        for view in self.views:
            vlat = view["laterality"]
            if lat in ("left", "right") and vlat in ("left", "right") and vlat != lat:
                continue
            suffix = "" if view["valid"] else " [empty]"
            self.view_combo.addItem(view["label"] + suffix, view["id"])
        if volume and "Delta" in volume["name"]:
            index = self.view_combo.findData("right_transtemporal_2")
            if index >= 0:
                self.view_combo.setCurrentIndex(index)
        if volume and volume["icon"].is_file():
            pixmap = self.QPixmap(str(volume["icon"]))
            self.preview.setPixmap(
                pixmap.scaled(240, 160, self.Qt.KeepAspectRatio, self.Qt.SmoothTransformation)
            )
        else:
            self.preview.setPixmap(self.QPixmap())
        self.description.setText((volume or {}).get("description") or "")

    def _accept(self):
        volume = self._current_volume()
        if volume is None:
            return
        self.chosen = {
            "adf": volume["adf"],
            "view": self.view_combo.currentData() or "default",
            "name": volume["name"],
            "close_all": bool(self.close_all.isChecked()),
        }
        if self.launch:
            self._start_tuner()
            return
        self.dialog.accept()

    def _start_tuner(self):
        from PyQt5.QtCore import QTimer

        import rtx_splash

        rtx_splash.kill_matching("fivrs_rtx_tuner.py")
        heading = f"Opening RTX Tuner — {self.chosen['name']}"
        self.dialog.setWindowTitle(heading)
        self._form_page.hide()
        self._load_page.show()
        self._load_status.setText("Starting…")
        log_path = pathlib.Path.home() / ".cache" / "fivrs-rtx-tuner" / "tuner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"")
        handle = open(log_path, "ab", buffering=0)
        env = os.environ.copy()
        env["FIVRS_SPLASH_ACTIVE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            _tuner_command(self.chosen["adf"], self.chosen["view"], self.chosen.get("close_all")),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(REPO),
            env=env,
        )
        self._log_path = log_path
        self._log_offset = 0
        self._ready = False
        self._timer = QTimer(self.dialog)
        self._timer.timeout.connect(self._poll_log)
        self._timer.start(100)
        self.dialog.rejected.connect(self._cancel_tuner)

    def _poll_log(self):
        import rtx_splash

        self._log_offset, lines = rtx_splash._drain_log(self._log_path, self._log_offset)
        for line in lines:
            if rtx_splash.apply_stage(self._load_status, self._load_bar, line):
                self._ready = True
                self._timer.stop()
                self.dialog.accept()
                return
        if self._proc.poll() is not None and not self._ready:
            self._timer.stop()
            self._load_status.setText("Startup failed. See ~/.cache/fivrs-rtx-tuner/tuner.log")
            self._load_bar.setRange(0, 100)
            self._load_bar.setValue(0)

    def _cancel_tuner(self):
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, 15)
        except OSError:
            proc.terminate()


def show_picker(launch=False):
    from PyQt5 import QtWidgets

    volumes = load_volumes()
    if not volumes:
        raise SystemExit(f"No volumes listed in {SETUP}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FIVRS RTX Tuner")
    dialog = Picker(QtWidgets, volumes, load_views(), launch=launch)
    if dialog.exec_() != QtWidgets.QDialog.Accepted or not dialog.chosen:
        return None
    return dialog.chosen


def main():
    launch = "--launch" in sys.argv
    choice = show_picker(launch=launch)
    if not choice:
        return 1
    if launch:
        return 0
    lines = f"--anatomy-adf\n{choice['adf']}\n--view\n{choice['view']}\n"
    if choice.get("close_all"):
        lines += "--close-all\n1\n"
    sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
