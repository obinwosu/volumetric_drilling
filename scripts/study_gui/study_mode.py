"""Study tab for the FIVRS launcher.

The proctor types the resident's name and presses Start. The name is stored
only in the study key file (participant_key.csv next to the recordings); the
recording, the simulator and the headset use the Study ID.

Start launches the simulator in study mode: Volume A, RTX view in the headset,
haptic device 1 only, no irrigation. The resident then works through the
tutorial in the headset (scripts/study/study_session.py) and the proctor
presses Begin trial when they are ready. At the end the launcher asks to
continue to feedback and starts the analysis.

"Rehearse without the simulator" runs the same flow against a stand-in window
(study_rehearsal.py) instead of AMBF, the RTX view and the recorder.
"""
from __future__ import annotations

import csv
import datetime
import re
import pathlib
import sys
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "study"))
import study_session  # noqa: E402
import study_rehearsal  # noqa: E402
import study_checks  # noqa: E402

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
        "tutorial": "Tutorial in progress in the headset",
        "ready": "Tutorial complete: the resident presses the left pedal to begin the study session",
        "briefing": "Study introduction on screen; the bone resets and the recorder starts behind it",
        "armed": "Trial armed: the timer starts at the first bone removed",
        "running": "Timed trial running",
        "done": "Session complete",
    }

    def __init__(self, ui, rehearsal_only=False):
        super().__init__()
        self.ui = ui
        self.rehearsal_only = rehearsal_only
        self.session = None
        self.recording_dir = None
        self.stand_in = None
        self._done_shown = False
        self._analysis = QProcess(self)
        self._analysis.setProcessChannelMode(QProcess.MergedChannels)
        self._analysis.readyReadStandardOutput.connect(self._analysis_output)
        self._analysis.finished.connect(self._analysis_finished)
        self._analysis_out = None
        # The questionnaire and feedback web app, started after the analysis.
        self._feedback_server = QProcess(self)
        self._feedback_server.setProcessChannelMode(QProcess.MergedChannels)
        self._feedback_server.readyReadStandardOutput.connect(self._feedback_server_output)
        self._feedback_url = None
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.stop_feedback_server)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._launch_started = None
        self._build()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self.refresh_status)
        self._status_timer.start()
        self.refresh_status()
        if not rehearsal_only:
            # A lock left by a crashed study session would stop the drill in normal use.
            study_session.SessionIO().set_drill_lock(False)
            study_session.SessionIO().clear_hud()

    # ---- layout ----
    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # 1. Participant: the only thing the proctor fills in before the resident starts.
        form = QtWidgets.QGroupBox("1. Participant")
        rows = QtWidgets.QGridLayout(form)
        rows.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("lastname_firstname  (not used in study analysis)")
        self.name_edit.setToolTip("Stored only in the study key file. The recording, the headset and "
                                  "the analysis use the Study ID (S001, S002, ...).")
        self.name_edit.textChanged.connect(self._name_changed)
        self.name_edit.returnPressed.connect(self.start)
        rows.addWidget(self.name_edit, 0, 1)
        self.name_hint = QtWidgets.QLabel("Format: lastname_firstname, for example smith_jane. "
                                          "Kept only in the study key; the session gets a Study ID.")
        self.name_hint.setWordWrap(True)
        self.name_hint.setStyleSheet("color: gray;")
        rows.addWidget(self.name_hint, 1, 1)
        layout.addWidget(form)

        self.start_button = QtWidgets.QPushButton("Start study session")
        self.start_button.setMinimumHeight(48)
        self.start_button.setToolTip("Opens the simulator in study mode and starts the tutorial in the headset.")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #1fae4b; color: white; font-size: 18px; font-weight: bold;"
            " border-radius: 6px; }"
            " QPushButton:hover { background-color: #23c455; }"
            " QPushButton:disabled { background-color: #3c4a40; color: #8a9a8e; }")
        self.start_button.clicked.connect(self.start)
        layout.addWidget(self.start_button)

        info = QtWidgets.QLabel(
            "Opens Anatomy A with the RTX view in the headset, haptic device 1 only, no irrigation. "
            "The resident works through a short tutorial in the headset, then you begin the 10-minute "
            "timed trial below.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray;")
        layout.addWidget(info)

        rehearse = QtWidgets.QHBoxLayout()
        self.rehearse_box = QtWidgets.QCheckBox("Rehearse without the simulator (development)")
        self.rehearse_box.setToolTip("Run the study flow against a stand-in window: no AMBF, RTX, "
                                     "recording or participant key. Name not needed.")
        self.rehearse_box.setChecked(self.rehearsal_only)
        self.rehearse_box.setEnabled(not self.rehearsal_only)
        self.rehearse_box.toggled.connect(lambda _on: self._name_changed())
        self.rehearse_box.toggled.connect(lambda _on: self.refresh_status())
        self.trial_minutes = QtWidgets.QDoubleSpinBox()
        self.trial_minutes.setRange(0.25, 10.0)
        self.trial_minutes.setSingleStep(0.25)
        self.trial_minutes.setValue(1.0)
        self.trial_minutes.setSuffix(" min trial")
        self.trial_minutes.setToolTip("Length of the timed trial in a rehearsal (the study uses 10 min)")
        self.rehearse_box.toggled.connect(self.trial_minutes.setVisible)
        self.trial_minutes.setVisible(self.rehearsal_only)
        rehearse.addWidget(self.rehearse_box)
        rehearse.addWidget(self.trial_minutes)
        rehearse.addStretch(1)
        layout.addLayout(rehearse)
        self.desktop_box = QtWidgets.QCheckBox("Desktop view instead of the headset (development)")
        self.desktop_box.setToolTip(
            "Runs the real simulator with the RTX view in a normal window and the headset text drawn in it. "
            "No Quest needed. Move the view by holding Stylus 1 and moving the stylus.")
        self.desktop_box.setVisible(not self.rehearsal_only)
        self.desktop_box.toggled.connect(lambda _on: self.refresh_status())
        self.desktop_box.toggled.connect(lambda _on: self._name_changed())
        self.rehearse_box.toggled.connect(lambda on: self.desktop_box.setEnabled(not on))
        layout.addWidget(self.desktop_box)

        # System status: what Start needs, then what the launch is doing.
        system = QtWidgets.QGroupBox("System status")
        box = QtWidgets.QVBoxLayout(system)
        top = QtWidgets.QHBoxLayout()
        self.waiting_label = QtWidgets.QLabel()
        self.waiting_label.setWordWrap(True)
        self.waiting_label.setStyleSheet("font-weight: bold;")
        top.addWidget(self.waiting_label, 1)
        recheck = QtWidgets.QPushButton("Recheck")
        recheck.clicked.connect(self.refresh_status)
        top.addWidget(recheck, 0, Qt.AlignTop)
        box.addLayout(top)
        self.checks_label = QtWidgets.QLabel()
        self.checks_label.setTextFormat(Qt.RichText)
        self.checks_label.setWordWrap(True)
        box.addWidget(self.checks_label)
        self.activity = QtWidgets.QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumBlockCount(200)
        self.activity.setPlaceholderText("Launcher messages appear here.")
        self.activity.setFixedHeight(90)
        box.addWidget(self.activity)
        layout.addWidget(system)

        # 2. Proctor controls: on the desktop, never in the headset.
        status = QtWidgets.QGroupBox("2. Proctor controls  (desktop only; the resident does not see this)")
        grid = QtWidgets.QGridLayout(status)
        grid.setColumnStretch(1, 1)
        self.id_label = QtWidgets.QLabel("Study ID: -")
        self.phase_label = QtWidgets.QLabel("Not started")
        self.phase_label.setWordWrap(True)
        self.phase_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.step_label = QtWidgets.QLabel("")
        self.step_label.setWordWrap(True)
        self.clock_label = QtWidgets.QLabel("10:00")
        self.clock_label.setStyleSheet("font-size: 40px; font-weight: bold;")
        self.clock_label.setToolTip("Drilling time left in the timed trial (pauses do not count)")
        grid.addWidget(self.id_label, 0, 0, 1, 2)
        grid.addWidget(self.clock_label, 0, 2, 2, 1, Qt.AlignRight | Qt.AlignTop)
        grid.addWidget(self.phase_label, 1, 0, 1, 2)
        grid.addWidget(self.step_label, 2, 0, 1, 3)

        self.skip_button = QtWidgets.QPushButton("Skip tutorial step")
        self.skip_button.clicked.connect(lambda: self.session and self.session.skip_step())
        self.begin_button = QtWidgets.QPushButton("Begin timed trial")
        self.begin_button.clicked.connect(self.begin_trial)
        self.end_button = QtWidgets.QPushButton("End trial early")
        self.end_button.clicked.connect(self.end_early)
        self.feedback_button = QtWidgets.QPushButton("Proceed to feedback")
        self.feedback_button.clicked.connect(self.start_analysis)
        explain = [
            (self.skip_button, "During the tutorial: move the resident past a step they are stuck on."),
            (self.begin_button, "Normally the resident does this with the left pedal. Shows the study "
                                "introduction while the bone resets and the recording starts; pressed "
                                "during the introduction, starts the trial without waiting for their pedal."),
            (self.end_button, "During the trial: stop it now (hardware fault, or the resident stops)."),
            (self.feedback_button, "After the trial: prepare the questionnaire and feedback reports again "
                                   "(for example after a failed analysis)."),
        ]
        for row, (button, text) in enumerate(explain, start=3):
            button.setMinimumWidth(170)
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            grid.addWidget(button, row, 0)
            grid.addWidget(label, row, 1, 1, 2)
        layout.addWidget(status)
        self._set_buttons(None)
        self._name_changed()

    NAME_FORMAT = re.compile(r"^[A-Za-z][A-Za-z' -]*_[A-Za-z][A-Za-z' -]*$")

    def _name_ok(self) -> bool:
        return bool(self.NAME_FORMAT.match(self.name_edit.text().strip()))

    def _desktop(self) -> bool:
        box = getattr(self, "desktop_box", None)
        return bool(box is not None and box.isChecked())

    def _name_changed(self, *_):
        text = self.name_edit.text().strip()
        bad = bool(text) and not self._name_ok() and not self.rehearsing and not self._desktop()
        self.name_hint.setStyleSheet("color: #d9534f;" if bad else "color: gray;")
        if self.session is None or self.session.phase == "done":
            self._set_buttons(None if self.session is None else "done")

    def _set_buttons(self, phase):
        idle = phase in (None, "done")
        self.start_button.setEnabled(idle and (self.rehearsing or self._desktop() or self._name_ok()))
        self.name_edit.setEnabled(idle)
        self.rehearse_box.setEnabled(idle and not self.rehearsal_only)
        if hasattr(self, "desktop_box"):
            self.desktop_box.setEnabled(idle and not self.rehearsing)
        self.skip_button.setEnabled(phase == "tutorial")
        self.begin_button.setEnabled(phase in ("tutorial", "ready", "briefing"))
        self.end_button.setEnabled(phase in ("armed", "running"))
        if hasattr(self, "feedback_button"):
            self.feedback_button.setEnabled(
                phase == "done" and self.recording_dir is not None
                and not isinstance(getattr(self.session, "io", None), study_rehearsal.RehearsalIO)
                and self._analysis.state() == QProcess.NotRunning)
        if phase is None and self.session is None:
            self.phase_label.setText("Not started. Enter the name and press Start study session.")

    # ---- system status ----
    def activity_line(self, text: str):
        """Launcher messages (print_info), so errors show on this tab, not only on Info."""
        self.activity.appendPlainText(text.rstrip())

    def _process_state(self, name):
        process = getattr(self.ui, name, None)
        if process is None:
            return "none"
        return {QProcess.NotRunning: "stopped", QProcess.Starting: "starting",
                QProcess.Running: "running"}[process.state()]

    def _preflight(self):
        pedal = ""
        config = getattr(self.ui, "gui_configuration", None)
        if config is not None:
            pedal = str(config.footpedal_device.get())
        return study_checks.preflight(pedal, sim_running=self._process_state("_ambf_process") == "running",
                                      headset=not self.desktop_box.isChecked())

    def refresh_status(self):
        if self.rehearsing:
            rows = [study_checks.Check("Rehearsal", study_checks.INFO,
                                       "no hardware needed: the stand-in window replaces the simulator", False)]
            headline = "Rehearsal: no checks needed." if self.session is None or self.session.phase == "done" \
                else "Rehearsal running in the stand-in window."
        elif self._launch_started is not None and self._process_state("_ambf_process") != "stopped":
            rows = study_checks.launch_status(self._process_state("_ambf_process"),
                                              self._process_state("_rtx_process") == "running",
                                              self._launch_started,
                                              headset=not self.desktop_box.isChecked())
            self._note_problems(rows)
            headline = study_checks.waiting_on(rows)
            if headline == "Everything is ready." and self.session is not None:
                headline = "Simulator running. " + self.PHASE_TEXT.get(self.session.phase, "")
        else:
            if self._launch_started is not None:
                self._launch_started = None  # the simulator closed
            rows = self._preflight()
            missing = [r for r in rows if r.required and r.state == study_checks.FAIL]
            headline = ("Ready to start." if not missing else
                        "Not ready to start: " + ", ".join(r.name for r in missing) + ".")
        self.waiting_label.setText(headline)
        problem = headline.startswith("Problem:")
        self.waiting_label.setStyleSheet(
            f"font-weight: bold; color: {study_checks.COLOUR[study_checks.FAIL]};" if problem
            else "font-weight: bold;")
        cells = "".join(
            f"<tr><td style='color:{study_checks.COLOUR[r.state]}; font-weight:bold; padding-right:6px'>"
            f"{study_checks.ICON[r.state]}</td><td style='padding-right:10px'>{r.name}</td>"
            f"<td style='color:gray'>{r.detail}</td></tr>" for r in rows)
        self.checks_label.setText(f"<table>{cells}</table>")

    def _note_problems(self, rows):
        """Report each new launch problem once: on this tab, and in launch_problems.log
        with where AMBF and the RTX view were waiting, so a hang can be traced later."""
        seen = self.__dict__.setdefault("_problems_seen", set())
        current = {r.name for r in rows if r.state == study_checks.FAIL}
        for row in rows:
            if row.state != study_checks.FAIL or row.name in seen:
                continue
            text = f"{row.name}: {row.detail}"
            self.ui.print_info(f"PROBLEM {text}")
            pids = [(label, int(getattr(self.ui, name).processId() or 0))
                    for label, name in (("AMBF", "_ambf_process"), ("RTX view", "_rtx_process"))
                    if getattr(self.ui, name, None) is not None]
            study_checks.log_problem(text, pids)
        for name in seen - current:
            self.ui.print_info(f"Recovered: {name}")
            study_checks.log_problem(f"Recovered: {name}")
        self._problems_seen = current

    # ---- actions ----
    def _study_volume(self):
        for button in self.ui.anatomy_button_group.buttons():
            if pathlib.Path(button.volume_adf).name == STUDY_VOLUME:
                return button
        return None

    @property
    def rehearsing(self) -> bool:
        return self.rehearse_box.isChecked()

    def start(self):
        if self.rehearsing:
            self.start_rehearsal()
            return
        name = self.name_edit.text().strip()
        if self._desktop() and not self._name_ok():
            name = name or "dev_desktop"  # development run: any name, or none
        elif not self._name_ok():
            QtWidgets.QMessageBox.warning(self, "Study", "Enter the name as lastname_firstname, for example smith_jane.")
            return
        if self.ui._ambf_process.state() != QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, "Study", "Close the running simulation first.")
            return
        volume = self._study_volume()
        if volume is None:
            QtWidgets.QMessageBox.warning(self, "Study", f"{STUDY_VOLUME} is not in gui_setup.yaml.")
            return
        missing = [r for r in self._preflight() if r.required and r.state == study_checks.FAIL]
        if missing:
            text = "\n".join(f"\u2022 {r.name}: {r.detail}" for r in missing)
            answer = QtWidgets.QMessageBox.question(
                self, "Not ready", f"These are not ready yet:\n\n{text}\n\nTry to start anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.activity.clear()
        self.activity_line("Starting the study session...")
        # Desktop runs are development: their own key and recordings, never real Study IDs.
        folder = "study_dev" if self.desktop_box.isChecked() else "study"
        base = pathlib.Path(str(self.ui.gui_configuration.recording_base_path.get())).expanduser() / folder
        key = base / KEY_FILE
        study_id = next_study_id(key)
        add_to_key(key, study_id, name)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.recording_dir = base / "recordings" / f"{study_id}_{stamp}"
        self.name_edit.clear()

        volume.setChecked(True)
        volume_adf = (HERE / volume.volume_adf).resolve()
        desktop = self.desktop_box.isChecked()
        for box, on in ((self.ui.check_rtx_view, True), (self.ui.check_openxr_eyepiece, not desktop),
                        (self.ui.check_openvr_hmd, False), (self.ui.check_video_record, False)):
            if box is not None:
                box.setChecked(on)
        self.ui.study_mode = {"study_id": study_id}
        self.ui.print_info(f"Study session {study_id}: starting the simulator")
        self._launch_started = time.time()
        self._problems_seen = set()
        self.ui.pressed_start_simulation()
        if self.ui._ambf_process.state() == QProcess.NotRunning:
            self.ui.study_mode = None
            self._launch_started = None
            self.phase_label.setText("The simulator did not start. See System status above.")
            self.activity_line("The simulator did not start (see the messages above).")
            self.refresh_status()
            return

        metadata = {
            "volume_adf": str(volume_adf),
            "volume_name": volume.volume_name,
            "starting_burr_mm": self.ui.get_starting_burr_mm(),
            "study_mode": {"irrigation": False, "haptic_devices": 1,
                           "display": "desktop RTX (development)" if desktop else "headset RTX"},
        }
        io = study_session.SessionIO(recorder_cmd=recorder_command(volume_adf))
        self.session = study_session.StudySession(io, study_id, self.recording_dir, metadata)
        self._done_shown = False
        self.id_label.setText(f"Study ID: {study_id}")
        self._timer.start()

    def start_rehearsal(self):
        """The same session with a stand-in window instead of the simulator."""
        if self.stand_in is not None:
            self.stand_in.close()
        self.recording_dir = study_rehearsal.rehearsal_dir() / "R001"
        io = study_rehearsal.RehearsalIO()
        self.stand_in = study_rehearsal.SimulatorStandIn(io)
        self.stand_in.closed.connect(self._stand_in_closed)
        self.stand_in.resize(1300, 560)
        self.stand_in.show()
        metadata = {"volume_adf": STUDY_VOLUME, "rehearsal": True}
        self.session = study_session.StudySession(io, "R001", self.recording_dir, metadata,
                                                  trial_seconds=self.trial_minutes.value() * 60.0)
        self._done_shown = False
        self.id_label.setText("Study ID: R001 (rehearsal)")
        self.ui.print_info(f"Rehearsal started (no simulator). Metadata goes to {self.recording_dir}")
        self._timer.start()

    def _stand_in_closed(self):
        self.stand_in = None
        self.simulator_closed()

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
        """Called by the launcher when AMBF exits (or the rehearsal stand-in closes)."""
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
            status = ("done; waiting for the resident to press the left pedal"
                      if session.progress.step_done else "in progress")
            self.step_label.setText(f"Tutorial page {session.step + 1} of {len(study_session.TUTORIAL)}: "
                                    f"{step.title} ({status})\n{' '.join(step.lines)}")
        else:
            self.step_label.setText("")
        remaining = session.remaining()
        if phase == "running" and session.io.read_state().get("paused"):
            self.phase_label.setText("Timed trial paused (pause menu open)")
        self.clock_label.setText(f"{int(remaining) // 60}:{int(remaining) % 60:02d}")
        if phase == "done" and not self._done_shown:
            self._done_shown = True
            self._finished()

    def _finished(self):
        self.ui.print_info(f"Study session {self.session.study_id} complete: {self.recording_dir}")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Session complete")
        box.setText("Timed trial is complete!\n\nClick below to proceed to the background/demographic "
                    "questionnaire and feedback reports.")
        box.addButton("Proceed", QtWidgets.QMessageBox.AcceptRole)
        box.exec_()
        if isinstance(self.session.io, study_rehearsal.RehearsalIO):
            self.phase_label.setText("Rehearsal complete. A real session would start the feedback analysis now.")
            self.ui.print_info(f"Rehearsal metadata: {self.recording_dir / 'metadata.json'}")
            return
        self.ui.close_simulation()
        self.start_analysis()

    # ---- analysis ----
    def start_analysis(self):
        if self.recording_dir is None or self._analysis.state() != QProcess.NotRunning:
            return
        problem = study_checks.edt_problem()
        if problem:
            text = ("The feedback analysis needs the distance maps for Anatomy A, and they are "
                    f"{problem}.\n\nThe recording is saved. Once the folder is in place, press "
                    "Proceed to feedback on the Study tab.")
            self.phase_label.setText("Feedback waiting for the distance maps (EDT). See System status.")
            self.ui.print_info("ERROR! Feedback analysis needs the EDT grids: " + problem)
            QtWidgets.QMessageBox.warning(self, "Feedback not ready", text)
            self._set_buttons("done")
            return
        feedback = REPO / "feedback_system"
        python = feedback / ".venv" / "bin" / "python"
        out = self.recording_dir.parent.parent / "results" / self.recording_dir.name
        attempt = 2
        while out.exists():  # the analysis never overwrites; a retry gets its own folder
            out = out.with_name(f"{self.recording_dir.name}_try{attempt}")
            attempt += 1
        self._analysis_out = out
        self._analysis_log = open(self.recording_dir.parent / (self.recording_dir.name + "_analysis.log"), "w")
        self._analysis.setWorkingDirectory(str(feedback))
        self._analysis.start(str(python), ["run.py", "process-hdf5", "--input-dir", str(self.recording_dir),
                                           "--output-dir", str(out)])
        self.phase_label.setText("Preparing feedback (analysis running). The background questionnaire is open.")
        self.ui.print_info(f"Feedback analysis started for {self.session.study_id}")
        self._set_buttons("done")
        # The resident answers the background questions while the analysis runs.
        try:
            self.open_intake()
        except OSError as exc:
            self.ui.print_info(f"ERROR! Could not open the background questionnaire: {exc}")

    # ---- questionnaire and feedback reports ----
    def _feedback_address(self):
        """(port, url, first view) for this resident's questionnaire. One port for the
        whole visit, so the browser tab stays the same when the full server replaces
        the background-only one."""
        import socket

        study_id = self.session.study_id
        number = int(re.sub(r"\D", "", study_id) or 0)
        # The order of the two feedback views alternates by Study ID number.
        first_view = "event-based" if number % 2 else "summary"
        port = getattr(self, "_feedback_port", None)
        if port is None:
            with socket.socket() as probe:  # any free local port
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            self._feedback_port = port
        url = f"http://127.0.0.1:{port}/?participant_id={study_id}&first_review_view={first_view}"
        return port, url, first_view

    def _feedback_state(self) -> pathlib.Path:
        return self.recording_dir.parent.parent / "study_state.json"

    def _start_feedback_server(self, args, log_path: pathlib.Path):
        self.stop_feedback_server()
        if getattr(self, "_feedback_log", None) is not None:
            self._feedback_log.close()
        feedback = REPO / "feedback_system"
        self._feedback_log = open(log_path, "w")
        self._feedback_server.setWorkingDirectory(str(feedback))
        self._feedback_server.start(str(feedback / ".venv" / "bin" / "python"), args)

    def open_intake(self):
        """Serve the background questionnaire while the analysis runs, and open it.

        Its answers go to the resident's profile in study_state.json; the full server
        started by open_feedback skips the background step and the page reloads into
        the feedback on its own.
        """
        self._feedback_port = None
        port, url, _first = self._feedback_address()
        study_id = self.session.study_id
        args = ["run.py", "serve-intake", "--study-state", str(self._feedback_state()),
                "--display-name", study_id, "--port", str(port)]
        self._feedback_url = url
        self._feedback_opened = False
        self._start_feedback_server(args, self.recording_dir.parent / (self.recording_dir.name + "_intake_server.log"))
        self.ui.print_info(f"Background questionnaire starting for {study_id}")

    def open_feedback(self, results: pathlib.Path):
        """Serve the questionnaire and reports for this recording.

        Replaces the background-only server on the same port when there is one: the
        resident's open tab notices and reloads. Otherwise opens the browser. The
        Study ID goes in the address, so the resident never types it.
        """
        import json

        manifest = json.loads((results / "recommendations_manifest.json").read_text())
        case = manifest["cases"][0]
        study_id = self.session.study_id
        port, url, first_view = self._feedback_address()
        args = ["run.py", "serve", "--mode", "expert-study", "--run-id", case["run_id"],
                "--db-path", case["evidence_database"], "--study-state", str(self._feedback_state()),
                "--display-name", study_id, "--port", str(port)]
        for flag, key in (("--video", "review_video"), ("--video-manifest", "review_video_manifest"),
                          ("--ray-v3-dir", "ray_v3_directory")):
            if case.get(key) and pathlib.Path(case[key]).exists():
                args += [flag, case[key]]
        already_open = bool(getattr(self, "_feedback_opened", False))
        self._feedback_url = url
        self._start_feedback_server(args, results / "feedback_server.log")
        self._feedback_opened = already_open  # the open tab reloads itself
        self.ui.print_info(f"Questionnaire server starting for {study_id} ({first_view} first)")

    def _feedback_server_output(self):
        text = bytes(self._feedback_server.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._feedback_log.write(text)
        self._feedback_log.flush()
        if "FIVRS_RETURN_TO_SIMULATOR" in text:
            # The resident pressed "Return to the simulator study window" on the end page.
            window = self.window()
            window.showNormal()
            window.raise_()
            window.activateWindow()
            self.phase_label.setText("Session complete: the resident finished the questionnaires and feedback.")
            self.ui.print_info("Resident finished the feedback and returned to the study window")
        if "app ready" in text:
            intake = "intake app ready" in text
            if not self._feedback_opened:
                self._feedback_opened = True
                QDesktopServices.openUrl(QUrl(self._feedback_url))
                self.ui.print_info(f"Questionnaire opened: {self._feedback_url}")
            self.phase_label.setText(
                "Background questionnaire open in the browser; the feedback follows when the analysis finishes."
                if intake else f"Feedback open in the browser: {self._feedback_url}")

    def stop_feedback_server(self):
        if self._feedback_server.state() != QProcess.NotRunning:
            self._feedback_server.terminate()
            if not self._feedback_server.waitForFinished(3000):
                self._feedback_server.kill()

    def _analysis_output(self):
        text = bytes(self._analysis.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._analysis_log.write(text)
        self._analysis_log.flush()

    def _analysis_finished(self, code, _status):
        self._analysis_log.close()
        self._set_buttons("done")
        if code == 0:
            self.phase_label.setText("Feedback is ready. Opening the questionnaire...")
            self.ui.print_info("Feedback analysis finished")
            try:
                self.open_feedback(self._analysis_out)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                self.phase_label.setText(f"Feedback is ready but could not be opened: {exc}")
                self.ui.print_info(f"ERROR! Could not open the feedback: {exc}")
        else:
            self.phase_label.setText(f"Feedback analysis failed (exit {code}). See the analysis log.")
            self.ui.print_info(f"ERROR! Feedback analysis failed with exit {code}")
