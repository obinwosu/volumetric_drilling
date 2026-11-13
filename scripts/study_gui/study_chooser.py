"""The launcher's single "Study" tab: pick the resident study or the graphics study.

Each study keeps its own panel (study_mode.StudyPanel, graphics_study_mode.GraphicsStudyPanel);
this only chooses which one is shown. The choice is locked while a study is running so a
participant's session cannot be switched midway:
  - the simulator (or the rehearsal stand-in) is open,
  - resident study: a session is in progress, the feedback analysis is running, or the
    questionnaire is open and the resident has not returned from it,
  - graphics study: a participant is between "Start" and the final comparison questions.
It unlocks when the study finishes, or when the proctor presses "End study".
"""
from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, QTimer

CHOICES = (("resident", "Resident study"), ("graphics", "Graphics study"))
SETTING = "study_choice"  # key in gui_setup.yaml
BUSY_STAGES = {"session": "a session", "tlx": "the NASA-TLX", "between": "the break between sessions",
               "background": "the background questionnaire",
               "preferences": "the comparison questions", "reflection": "Post Sim Reflection"}


def _scroll(widget):
    # Scrolls when the launcher window is shorter than the panel.
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _running(process) -> bool:
    return process is not None and process.state() != QProcess.NotRunning


class StudyChooser(QtWidgets.QWidget):
    def __init__(self, ui, resident_panel, graphics_panel, choice: str = "resident", save=None):
        """save(choice) is called when the proctor picks a study (writes gui_setup.yaml)."""
        super().__init__()
        self.ui = ui
        self.panels = {"resident": resident_panel, "graphics": graphics_panel}
        self._save = save
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QtWidgets.QGroupBox("Which study?")
        row = QtWidgets.QHBoxLayout(top)
        self.group = QtWidgets.QButtonGroup(self)
        self.buttons = {}
        for index, (key, label) in enumerate(CHOICES):
            button = QtWidgets.QRadioButton(label)
            button.setStyleSheet("font-size: 15px; font-weight: bold;")
            self.group.addButton(button, index)
            self.buttons[key] = button
            row.addWidget(button)
        row.addSpacing(20)
        self.lock_label = QtWidgets.QLabel("")
        self.lock_label.setWordWrap(True)
        row.addWidget(self.lock_label, 1)
        self.end_button = QtWidgets.QPushButton("End study")
        self.end_button.setToolTip("Stop the participant's study here, so the other study can be chosen. "
                                   "What is already saved is kept.")
        self.end_button.clicked.connect(self.end_study)
        row.addWidget(self.end_button)
        layout.addWidget(top)

        self.stack = QtWidgets.QStackedWidget()
        for key, _label in CHOICES:
            self.stack.addWidget(_scroll(self.panels[key]))
        layout.addWidget(self.stack, 1)

        self.choice = choice if choice in self.panels else "resident"
        self.buttons[self.choice].setChecked(True)
        self.stack.setCurrentIndex(self._index(self.choice))
        self.group.buttonClicked.connect(self._picked)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh_lock)
        self._timer.start()
        self.refresh_lock()

    @staticmethod
    def _index(key) -> int:
        return [k for k, _ in CHOICES].index(key)

    def current_panel(self):
        return self.panels[self.choice]

    def choose(self, key: str) -> bool:
        """Show this study's panel. Refused (False) while a study is running."""
        if key not in self.panels:
            raise ValueError(key)
        if key != self.choice and self.busy_reason() is not None:
            self.buttons[self.choice].setChecked(True)
            return False
        changed = key != self.choice
        self.choice = key
        self.buttons[key].setChecked(True)
        self.stack.setCurrentIndex(self._index(key))
        if changed and self._save is not None:
            try:
                self._save(key)
            except OSError as exc:
                if hasattr(self.ui, "print_info"):
                    self.ui.print_info(f"ERROR! Could not remember the study choice: {exc}")
        self.refresh_lock()
        return True

    def _picked(self, button):
        self.choose(CHOICES[self.group.id(button)][0])

    # ---- lock ----
    def _simulator_open(self) -> bool:
        return _running(getattr(self.ui, "_ambf_process", None))

    def _panel_reason(self, key):
        panel = self.panels[key]
        if panel is None:
            return None
        if key == "resident":
            if panel._timer.isActive() and panel.session is not None:
                if panel.session.phase == "done":
                    return f"{panel.session.study_id} is finishing"
                return f"{panel.session.study_id} is in progress ({panel.session.phase})"
            if _running(panel._analysis):
                return "the feedback analysis is running"
            if panel.questionnaire_pending():
                return "the resident's questionnaire is open"
        elif panel.stage in BUSY_STAGES:
            study_id = (panel.plan or {}).get("study_id", "")
            return f"{study_id} is in {BUSY_STAGES[panel.stage]}".strip()
        if getattr(panel, "stand_in", None) is not None:
            return "the rehearsal simulator is open"
        return None

    def busy_key(self):
        """The study that is running, or None."""
        for key in (self.choice,) + tuple(k for k, _ in CHOICES if k != self.choice):
            if self._panel_reason(key) is not None:
                return key
        return self.choice if self._simulator_open() else None

    def busy_reason(self):
        key = self.busy_key()
        if key is None:
            return None
        return self._panel_reason(key) or "the simulator is open"

    def refresh_lock(self):
        key = self.busy_key()
        reason = self.busy_reason()
        if key is not None and key != self.choice:
            # Never hide a running study (for example, one started before this tab existed).
            self.choice = key
            self.buttons[key].setChecked(True)
            self.stack.setCurrentIndex(self._index(key))
        for button in self.buttons.values():
            button.setEnabled(reason is None)
        if reason is None:
            self.lock_label.setText("")
            self.end_button.setEnabled(False)
            return
        self.lock_label.setText(f"Locked: {reason}.")
        self.lock_label.setStyleSheet("color: #b36b00;")
        self.end_button.setEnabled(self._can_end() is None)

    def _can_end(self):
        """None if End study may be pressed now, else why not."""
        if self._simulator_open():
            return "Close the simulator first."
        panel = self.current_panel()
        if self.choice == "resident" and _running(panel._analysis):
            return "Wait for the feedback analysis to finish."
        return None

    def end_study(self):
        reason = self.busy_reason()
        if reason is None:
            return
        why_not = self._can_end()
        if why_not is not None:
            QtWidgets.QMessageBox.information(self, "End study", why_not)
            return
        label = dict(CHOICES)[self.choice]
        answer = QtWidgets.QMessageBox.question(
            self, "End study", f"{label}: {reason}.\n\nEnd it now? What is already saved is kept.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.end_now()

    def end_now(self):
        """End the running study without asking (End study, after the confirmation)."""
        panel = self.current_panel()
        if self.choice == "graphics":
            panel.end_participant()
        else:
            if panel.stand_in is not None:
                panel.stand_in.close()  # stops the rehearsal session like a simulator close
            if panel._timer.isActive():
                panel.simulator_closed()
            if panel.questionnaire_pending():
                panel.stop_feedback_server()
            if hasattr(self.ui, "print_info"):
                self.ui.print_info("Resident study ended by the proctor from the Study tab")
        self.refresh_lock()
