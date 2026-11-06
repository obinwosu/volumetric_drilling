"""Study tab for the FIVRS launcher.

The proctor types the resident's name and presses Start. The name is stored
only in the study key file (participant_key.csv next to the recordings); the
recording, the simulator and the headset use the Study ID.

Start launches the simulator in study mode: Volume A, RTX view in the headset,
haptic device 1 only, no irrigation. The resident then works through the
tutorial in the headset (scripts/study/study_session.py) and the proctor
presses Begin trial when they are ready. At the end the launcher asks to
continue to feedback and starts the analysis.
"""
from __future__ import annotations

import csv
import datetime
import pathlib
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, Qt, QTimer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "study"))
import study_session  # noqa: E402

STUDY_VOLUME = "volume_256_RT143.yaml"  # Anatomy A
KEY_FILE = "participant_key.csv"


def next_study_id(key_path: pathlib.Path) -> str:
    used = set()
    if key_path.is_file():
        with open(key_path, newline="") as handle:
            used = {row["study_id"] for row in csv.DictReader(handle)}
    number = 1
    while f"S{number:03d}" in used:
        number += 1
    return f"S{number:03d}"


def add_to_key(key_path: pathlib.Path, study_id: str, name: str) -> None:
    new = not key_path.is_file()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(key_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["study_id", "name", "created"])
        writer.writerow([study_id, name, datetime.datetime.now().isoformat(timespec="seconds")])


def recorder_command(volume_adf: pathlib.Path):
    python = REPO / "feedback_system" / ".venv" / "bin" / "python"
    script = REPO / "scripts" / "study" / "fivrs_recorder.py"

    def command(output_dir: pathlib.Path) -> str:
        return (
            "set +u && source /opt/ros/jazzy/setup.bash && "
            'source "$HOME/ros_ambf_ws/install/setup.bash" && '
            f'exec "{python}" -u "{script}" --output_dir "{output_dir}" --volume_adf "{volume_adf}"'
        )

    return command


class StudyPanel(QtWidgets.QWidget):
    PHASE_TEXT = {
        "tutorial": "Tutorial",
        "ready": "Tutorial complete: press Begin trial when the resident is ready",
        "resetting": "Restoring the bone and starting the recorder...",
        "armed": "Trial armed: the timer starts at the first bone removed",
        "running": "Timed trial running",
        "done": "Session complete",
    }

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.session = None
        self.recording_dir = None
        self._done_shown = False
        self._analysis = QProcess(self)
        self._analysis.setProcessChannelMode(QProcess.MergedChannels)
        self._analysis.readyReadStandardOutput.connect(self._analysis_output)
        self._analysis.finished.connect(self._analysis_finished)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._build()
        # A lock left by a crashed study session would stop the drill in normal use.
        study_session.SessionIO().set_drill_lock(False)
        study_session.SessionIO().clear_hud()

    # ---- layout ----
    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        form = QtWidgets.QGroupBox("Participant")
        row = QtWidgets.QHBoxLayout(form)
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Resident's name (kept only in the study key)")
        row.addWidget(self.name_edit, 1)
        self.start_button = QtWidgets.QPushButton("Start study session")
        self.start_button.clicked.connect(self.start)
        row.addWidget(self.start_button)
        layout.addWidget(form)

        info = QtWidgets.QLabel(
            "Study mode: Anatomy A, RTX view in the headset, haptic device 1 only, no irrigation. "
            "The tutorial runs in the headset; the bone is restored before the 10-minute trial."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        status = QtWidgets.QGroupBox("Session")
        grid = QtWidgets.QGridLayout(status)
        self.id_label = QtWidgets.QLabel("Study ID: -")
        self.phase_label = QtWidgets.QLabel("Not started")
        self.phase_label.setWordWrap(True)
        self.phase_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        self.step_label = QtWidgets.QLabel("")
        self.step_label.setWordWrap(True)
        self.clock_label = QtWidgets.QLabel("10:00")
        self.clock_label.setStyleSheet("font-size: 40px; font-weight: bold;")
        grid.addWidget(self.id_label, 0, 0)
        grid.addWidget(self.clock_label, 0, 1, 2, 1, Qt.AlignRight)
        grid.addWidget(self.phase_label, 1, 0)
        grid.addWidget(self.step_label, 2, 0, 1, 2)
        layout.addWidget(status)

        buttons = QtWidgets.QHBoxLayout()
        self.skip_button = QtWidgets.QPushButton("Skip tutorial step")
        self.skip_button.clicked.connect(lambda: self.session and self.session.skip_step())
        self.begin_button = QtWidgets.QPushButton("Begin trial")
        self.begin_button.clicked.connect(self.begin_trial)
        self.end_button = QtWidgets.QPushButton("End trial early")
        self.end_button.clicked.connect(self.end_early)
        for button in (self.skip_button, self.begin_button, self.end_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._set_buttons(None)

    def _set_buttons(self, phase):
        self.start_button.setEnabled(phase in (None, "done"))
        self.name_edit.setEnabled(phase in (None, "done"))
        self.skip_button.setEnabled(phase == "tutorial")
        self.begin_button.setEnabled(phase in ("tutorial", "ready"))
        self.end_button.setEnabled(phase in ("armed", "running"))

    # ---- actions ----
    def _study_volume(self):
        for button in self.ui.anatomy_button_group.buttons():
            if pathlib.Path(button.volume_adf).name == STUDY_VOLUME:
                return button
        return None

    def start(self):
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Study", "Enter the resident's name first.")
            return
        if self.ui._ambf_process.state() != QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, "Study", "Close the running simulation first.")
            return
        volume = self._study_volume()
        if volume is None:
            QtWidgets.QMessageBox.warning(self, "Study", f"{STUDY_VOLUME} is not in gui_setup.yaml.")
            return
        base = pathlib.Path(str(self.ui.gui_configuration.recording_base_path.get())).expanduser() / "study"
        key = base / KEY_FILE
        study_id = next_study_id(key)
        add_to_key(key, study_id, name)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.recording_dir = base / "recordings" / f"{study_id}_{stamp}"
        self.name_edit.clear()

        volume.setChecked(True)
        volume_adf = (HERE / volume.volume_adf).resolve()
        for box, on in ((self.ui.check_rtx_view, True), (self.ui.check_openxr_eyepiece, True),
                        (self.ui.check_openvr_hmd, False), (self.ui.check_video_record, False)):
            if box is not None:
                box.setChecked(on)
        self.ui.study_mode = {"study_id": study_id}
        self.ui.print_info(f"Study session {study_id}: starting the simulator")
        self.ui.pressed_start_simulation()
        if self.ui._ambf_process.state() == QProcess.NotRunning:
            self.ui.study_mode = None
            self.phase_label.setText("The simulator did not start. See the Info pane.")
            return

        metadata = {
            "volume_adf": str(volume_adf),
            "volume_name": volume.volume_name,
            "starting_burr_mm": self.ui.get_starting_burr_mm(),
            "study_mode": {"irrigation": False, "haptic_devices": 1, "display": "headset RTX"},
        }
        io = study_session.SessionIO(recorder_cmd=recorder_command(volume_adf))
        self.session = study_session.StudySession(io, study_id, self.recording_dir, metadata)
        self._done_shown = False
        self.id_label.setText(f"Study ID: {study_id}")
        self._timer.start()

    def begin_trial(self):
        if self.session is None:
            return
        if self.session.phase == "tutorial":
            answer = QtWidgets.QMessageBox.question(
                self, "Begin trial", "The tutorial is not finished. Begin the timed trial anyway?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.session.error = None
        self.session.begin_trial()

    def end_early(self):
        if self.session is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "End trial", "Stop the timed trial now?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.session.end_early()

    def simulator_closed(self):
        """Called by the launcher when AMBF exits."""
        if self.session is not None and self.session.phase in ("armed", "running"):
            self.session.end_early()
            self.ui.print_info("Simulator closed during the trial; recording stopped.")
        self._stop_session()

    def _stop_session(self):
        self._timer.stop()
        self.ui.study_mode = None
        if self.session is not None:
            self.session.io.clear_hud()
            self.session.io.set_drill_lock(False)
            self._set_buttons("done" if self.session.phase == "done" else None)

    # ---- loop ----
    def _tick(self):
        session = self.session
        if session is None:
            return
        phase = session.tick()
        self._set_buttons(phase)
        self.phase_label.setText(session.error or self.PHASE_TEXT.get(phase, phase))
        if phase == "tutorial":
            step = study_session.TUTORIAL[session.step]
            self.step_label.setText(f"Step {session.step + 1} of {len(study_session.TUTORIAL)}: "
                                    f"{step.title} - {' '.join(step.lines)}")
        else:
            self.step_label.setText("")
        remaining = session.remaining()
        self.clock_label.setText(f"{int(remaining) // 60}:{int(remaining) % 60:02d}")
        if phase == "done" and not self._done_shown:
            self._done_shown = True
            self._finished()

    def _finished(self):
        self.ui.print_info(f"Study session {self.session.study_id} complete: {self.recording_dir}")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Session complete")
        box.setText("The timed trial is complete.\n\nHave the resident remove the headset, "
                    "then continue to the questionnaire and feedback.")
        proceed = box.addButton("Proceed to feedback", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Later", QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        self.ui._ambf_process.close()
        if box.clickedButton() is proceed:
            self.start_analysis()

    # ---- analysis ----
    def start_analysis(self):
        feedback = REPO / "feedback_system"
        python = feedback / ".venv" / "bin" / "python"
        out = self.recording_dir.parent.parent / "results" / self.recording_dir.name
        self._analysis_log = open(self.recording_dir.parent / (self.recording_dir.name + "_analysis.log"), "w")
        self._analysis.setWorkingDirectory(str(feedback))
        self._analysis.start(str(python), ["run.py", "process-hdf5", "--input-dir", str(self.recording_dir),
                                           "--output-dir", str(out)])
        self.phase_label.setText("Preparing feedback (analysis running)...")
        self.ui.print_info(f"Feedback analysis started for {self.session.study_id}")

    def _analysis_output(self):
        text = bytes(self._analysis.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._analysis_log.write(text)
        self._analysis_log.flush()

    def _analysis_finished(self, code, _status):
        self._analysis_log.close()
        if code == 0:
            self.phase_label.setText("Feedback is ready.")
            self.ui.print_info("Feedback analysis finished")
        else:
            self.phase_label.setText(f"Feedback analysis failed (exit {code}). See the analysis log.")
            self.ui.print_info(f"ERROR! Feedback analysis failed with exit {code}")
