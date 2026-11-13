"""Graphics study tab for the FIVRS launcher.

Same task, two renderers, NASA-TLX after each (scripts/study/graphics_study.py).
The proctor types the participant's name, picks which two graphics to compare
and presses Start. The order is chosen at random (balanced within each
comparison). Arm 1 starts with the resident-study tutorial in its own graphics;
after its trial the simulator closes and the NASA-TLX opens on this monitor.
The proctor then starts arm 2 (the simulator reopens in the other graphics,
no tutorial), and a second NASA-TLX follows.

After the second NASA-TLX: the resident study's background questionnaire, two
questions comparing the sessions (more realistic, rather train on; Session 1 /
Session 2 / No difference), and the resident study's last page, Post Sim
Reflection. The participant can end a session early from the pause menu (End
trial), as in the resident study.

"Practice run in the headset" and "Desktop" are development runs: saved under
graphics_study_dev with their own IDs, never a participant.

The proctor presses "Incus identified" when the participant says they see the
short process of the incus; that ends the arm. Otherwise it ends at the time
limit (10 min of drilling, pauses not counted).

"Rehearse without the simulator" runs the whole flow against the stand-in
window from study_rehearsal.py: no AMBF, RTX, recorder or study files.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, Qt, QTimer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "study"))
import graphics_study as gs  # noqa: E402
import study_checks  # noqa: E402
import study_mode  # noqa: E402
import study_rehearsal  # noqa: E402
import study_session  # noqa: E402

NAME_FORMAT = re.compile(r"^[A-Za-z][A-Za-z' -]*_[A-Za-z][A-Za-z' -]*$")


class TlxDialog(QtWidgets.QDialog):
    """Raw NASA-TLX: six scales, 21 marks each (0-100 in steps of 5). Every scale must be answered.

    Submit with a scale unanswered outlines the missing scales in amber and scrolls to the first.
    """

    BOX_W, BOX_H, BOX_H_TICK = 44, 40, 52  # the 0 / 50 / 100 marks are taller, as on the paper form
    BOX_STYLE = (
        "QPushButton { border: 1px solid #777; border-radius: 0; background: #f4f4f4; }"
        " QPushButton:hover { background: #bfe3dc; border-color: #0e7c6b; }"
        " QPushButton:checked { background: #0e7c6b; border-color: #0e7c6b; }")
    # Dark amber: the launcher's theme draws all text light, so a pale fill would hide the labels.
    MISSING_STYLE = "QGroupBox { border: 3px solid #e0a000; border-radius: 4px; background: #3a2c08; }"

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle("Workload questionnaire (NASA-TLX)")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.started = time.monotonic()
        self.answers = {}
        self._groups = {}
        self._boxes = {}
        self._tried = False
        outer = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(title)
        heading.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(heading)
        intro = QtWidgets.QLabel("Think about the drilling session you just finished. For each question, "
                                 "click the mark that best describes your experience.")
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 15px;")
        outer.addWidget(intro)

        # The scales scroll if the screen is short; the heading and Submit stay in view.
        inner = QtWidgets.QWidget()
        scales = QtWidgets.QVBoxLayout(inner)
        scales.setContentsMargins(0, 0, 0, 0)
        middle = len(gs.TLX_STEPS) // 2
        for key, name, question, low, high in gs.TLX_SCALES:
            box = QtWidgets.QGroupBox()
            grid = QtWidgets.QGridLayout(box)
            grid.setHorizontalSpacing(0)
            grid.setVerticalSpacing(2)
            label = QtWidgets.QLabel(f"<b>{name}</b> &nbsp; {question}")
            label.setStyleSheet("font-size: 15px;")
            grid.addWidget(label, 0, 0, 1, len(gs.TLX_STEPS))
            group = QtWidgets.QButtonGroup(self)
            group.setExclusive(True)
            for column, value in enumerate(gs.TLX_STEPS):
                button = QtWidgets.QPushButton()
                button.setCheckable(True)
                button.setCursor(Qt.PointingHandCursor)
                button.setFixedSize(self.BOX_W, self.BOX_H if value % 50 else self.BOX_H_TICK)
                button.setToolTip(str(value))
                button.setStyleSheet(self.BOX_STYLE)
                group.addButton(button, value)
                grid.addWidget(button, 1, column, Qt.AlignBottom)
            tick = QtWidgets.QFrame()
            tick.setFixedSize(2, 10)
            tick.setStyleSheet("background: #555;")
            grid.addWidget(tick, 2, middle, Qt.AlignHCenter | Qt.AlignTop)
            grid.addWidget(QtWidgets.QLabel(low), 2, 0, 1, 5, Qt.AlignLeft)
            grid.addWidget(QtWidgets.QLabel(high), 2, len(gs.TLX_STEPS) - 5, 1, 5, Qt.AlignRight)
            group.idClicked.connect(lambda value, key=key: self._answered(key, value))
            self._groups[key] = group
            self._boxes[key] = box
            scales.addWidget(box)
        scales.addStretch(1)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # only on a very narrow screen
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, 1)

        self.submit = QtWidgets.QPushButton("Submit")
        self.submit.setMinimumSize(160, 48)
        self.submit.setStyleSheet("QPushButton { background-color: #0e7c6b; color: white; font-size: 17px;"
                                  " font-weight: bold; border-radius: 6px; }"
                                  " QPushButton:hover { background-color: #0b6658; }")
        self.submit.clicked.connect(self.try_submit)
        self.remaining = QtWidgets.QLabel()
        self.remaining.setStyleSheet("font-size: 15px; color: gray;")
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.remaining, 1)
        bottom.addWidget(self.submit)
        outer.addLayout(bottom)

        # Fit the window to the screen: all scales if they fit, otherwise scroll.
        inner.adjustSize()
        want_w = inner.sizeHint().width() + 40
        want_h = inner.sizeHint().height() + 160
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            room = screen.availableGeometry()
            want_w = min(want_w, room.width() - 20)
            want_h = min(want_h, room.height() - 60)
        self.resize(want_w, want_h)
        self._update()

    def missing(self):
        return [key for key, *_ in gs.TLX_SCALES if key not in self.answers]

    def try_submit(self):
        left = self.missing()
        if not left:
            self.accept()
            return
        self._tried = True
        self._update()
        self.scroll.ensureWidgetVisible(self._boxes[left[0]], 0, 20)

    def _answered(self, key, value):
        self.answers[key] = int(value)
        self._update()

    def _update(self):
        left = self.missing()
        for key, box in self._boxes.items():
            box.setStyleSheet(self.MISSING_STYLE if self._tried and key in left else "")
        if not left:
            self.remaining.setText("All questions answered.")
            self.remaining.setStyleSheet("font-size: 15px; color: #0e7c6b;")
        elif self._tried:
            names = ", ".join(name for key, name, *_ in gs.TLX_SCALES if key in left)
            self.remaining.setText(f"Please answer: {names}")
            self.remaining.setStyleSheet("font-size: 15px; color: #a06c00; font-weight: bold;")
        else:
            self.remaining.setText(f"{len(left)} question{'s' if len(left) > 1 else ''} left")
            self.remaining.setStyleSheet("font-size: 15px; color: gray;")

    def seconds(self) -> float:
        return time.monotonic() - self.started

    def reject(self):
        # Escape or the close button: the proctor must confirm, the answers are not saved.
        answer = QtWidgets.QMessageBox.question(self, "Close questionnaire",
                                                "Close without saving this questionnaire?")
        if answer == QtWidgets.QMessageBox.Yes:
            super().reject()


class PreferenceDialog(QtWidgets.QDialog):
    """After both sessions: which looked more realistic, which to train on. All must be answered."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Comparing the two sessions")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.answers = {}
        self._groups = {}
        outer = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel("Comparing the two sessions")
        heading.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(heading)
        intro = QtWidgets.QLabel("Think about the two drilling sessions you just did.")
        intro.setStyleSheet("font-size: 15px;")
        outer.addWidget(intro)
        for key, question in gs.PREFERENCE_QUESTIONS:
            box = QtWidgets.QGroupBox()
            column = QtWidgets.QVBoxLayout(box)
            label = QtWidgets.QLabel(f"<b>{question}</b>")
            label.setStyleSheet("font-size: 16px;")
            column.addWidget(label)
            row = QtWidgets.QHBoxLayout()
            group = QtWidgets.QButtonGroup(self)
            group.setExclusive(True)
            for value, text in gs.PREFERENCE_CHOICES:
                button = QtWidgets.QPushButton(text)
                button.setCheckable(True)
                button.setMinimumSize(170, 44)
                button.setStyleSheet(CHOICE_STYLE)
                button.clicked.connect(lambda _on, key=key, value=value: self._answered(key, value))
                group.addButton(button)
                row.addWidget(button)
                self._groups.setdefault(key, {})[value] = button
            row.addStretch(1)
            column.addLayout(row)
            outer.addWidget(box)
        self.submit = QtWidgets.QPushButton("Submit")
        self.submit.setMinimumHeight(44)
        self.submit.setStyleSheet("QPushButton { background-color: #0e7c6b; color: white; font-size: 17px;"
                                  " font-weight: bold; border-radius: 6px; }"
                                  " QPushButton:disabled { background-color: #9bb; }")
        self.submit.setEnabled(False)
        self.submit.clicked.connect(self.accept)
        outer.addWidget(self.submit)

    def _answered(self, key, value):
        self.answers[key] = value
        self.submit.setEnabled(len(self.answers) == len(gs.PREFERENCE_QUESTIONS))

    def reject(self):
        answer = QtWidgets.QMessageBox.question(self, "Close questions", "Close without saving these answers?")
        if answer == QtWidgets.QMessageBox.Yes:
            super().reject()


# Answer buttons in the launcher's dark theme (study_gui.MIDNIGHT_OCEAN_STYLE draws all
# text light): text colour set explicitly so the labels show before they are picked.
CHOICE_STYLE = ("QPushButton { font-size: 15px; color: #E6EEF5; background: #0F2433; border: 1px solid #2F8FBF;"
                " border-radius: 6px; padding: 0 10px; }"
                " QPushButton:hover { background: #2A4F68; }"
                " QPushButton:checked { background: #0e7c6b; color: white; border: 2px solid #3fd1b5;"
                " font-weight: bold; }")
SUBMIT_STYLE = ("QPushButton { background-color: #0e7c6b; color: white; font-size: 17px;"
                " font-weight: bold; border-radius: 6px; }"
                " QPushButton:disabled { background-color: #9bb; }")


def _choice_row(owner, values, labels, on_pick, width=120):
    """A row of exclusive buttons; returns (layout, {value: button})."""
    row = QtWidgets.QHBoxLayout()
    group = QtWidgets.QButtonGroup(owner)
    group.setExclusive(True)
    buttons = {}
    for value, text in zip(values, labels):
        button = QtWidgets.QPushButton(str(text))
        button.setCheckable(True)
        button.setMinimumSize(width, 40)
        button.setStyleSheet(CHOICE_STYLE)
        button.clicked.connect(lambda _on, value=value: on_pick(value))
        group.addButton(button)
        row.addWidget(button)
        buttons[value] = button
    row.addStretch(1)
    return row, buttons


def _question(layout, text):
    box = QtWidgets.QGroupBox()
    column = QtWidgets.QVBoxLayout(box)
    label = QtWidgets.QLabel(f"<b>{text}</b>")
    label.setWordWrap(True)
    label.setStyleSheet("font-size: 16px;")
    column.addWidget(label)
    layout.addWidget(box)
    return column


class BackgroundDialog(QtWidgets.QDialog):
    """The resident study's background questionnaire, same wording (graphics_study.background_definition)."""

    def __init__(self, parent):
        super().__init__(parent)
        page = gs.background_definition()
        fields = page["fields"]
        self.setWindowTitle(page["title"])
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._started = time.time()
        self.answers = {}
        self._buttons = {}
        self._role_units = fields["training_stage"]["role_units"]
        self._role_pgy = fields["training_stage"].get("role_pgy", {})
        outer = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(page["title"])
        heading.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(heading)
        intro = QtWidgets.QLabel(page.get("instructions", ""))
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 15px;")
        outer.addWidget(intro)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        roles = fields["role"]["options"]
        column = _question(layout, fields["role"]["label"])
        row, self._buttons["role"] = _choice_row(self, roles, [r.capitalize() for r in roles], self._pick_role)
        column.addLayout(row)

        column = _question(layout, fields["training_stage"]["label"])
        hint = QtWidgets.QLabel(fields["training_stage"].get("description", ""))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 14px;")
        column.addWidget(hint)
        pgys = sorted({y for years in self._role_pgy.values() for y in years})
        row, self._buttons["pgy"] = _choice_row(self, pgys, [f"PGY {y}" for y in pgys], self._pick_pgy, width=90)
        column.addLayout(row)
        years = QtWidgets.QHBoxLayout()
        self.years_label = QtWidgets.QLabel("Years in independent practice:")
        self.years = QtWidgets.QSpinBox()
        self.years.setRange(-1, 70)
        self.years.setSpecialValueText("choose")
        self.years.setValue(-1)
        self.years.setMinimumSize(110, 36)
        self.years.valueChanged.connect(lambda _v: self._update())
        years.addWidget(self.years_label)
        years.addWidget(self.years)
        years.addStretch(1)
        column.addLayout(years)

        for key in [k for k in fields if k not in ("role", "training_stage")]:
            field = fields[key]
            column = _question(layout, field["label"])
            labels = field.get("option_labels") or field["options"]
            row, self._buttons[key] = _choice_row(
                self, field["options"], labels, lambda value, key=key: self._answered(key, value))
            column.addLayout(row)
        layout.addStretch(1)

        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #d9534f;")
        outer.addWidget(self.error)
        self.submit = QtWidgets.QPushButton("Submit")
        self.submit.setMinimumHeight(44)
        self.submit.setStyleSheet(SUBMIT_STYLE)
        self.submit.clicked.connect(self.try_submit)
        outer.addWidget(self.submit)
        self._fields = fields
        self._pick_role(None)

    def _pick_role(self, role):
        if role is None:
            self.answers.pop("role", None)
        else:
            self.answers["role"] = role
        allowed = self._role_pgy.get(role, [])
        for year, button in self._buttons["pgy"].items():
            button.setVisible(year in allowed)  # only this role's years (residents 1-5, fellows 6-7)
            if year not in allowed and button.isChecked():
                button.group().setExclusive(False)
                button.setChecked(False)
                button.group().setExclusive(True)
                self.answers.pop("pgy", None)
        typed = role is not None and role not in self._role_pgy
        self.years_label.setVisible(typed)
        self.years.setVisible(typed)
        self._update()

    def _pick_pgy(self, year):
        self.answers["pgy"] = year
        self._update()

    def _answered(self, key, value):
        self.answers[key] = value
        self._update()

    def result_answers(self):
        """The answers in the resident study's form, or None while some are missing."""
        role = self.answers.get("role")
        if role is None:
            return None
        if role in self._role_pgy:
            value = self.answers.get("pgy")
        else:
            value = self.years.value() if self.years.value() >= 0 else None
        if value is None:
            return None
        answers = {"role": role,
                   "training_stage": {"status": "provided", "value": value, "unit": self._role_units[role]}}
        for key in [k for k in self._fields if k not in ("role", "training_stage")]:
            if key not in self.answers:
                return None
            answers[key] = self.answers[key]
        return answers

    def _update(self):
        self.submit.setEnabled(self.result_answers() is not None)
        self.error.setText("")

    def seconds(self) -> float:
        return time.time() - self._started

    def try_submit(self):
        answers = self.result_answers()
        if answers is None:
            return
        try:
            gs.resident_surveys().validate_background(answers)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()

    def reject(self):
        answer = QtWidgets.QMessageBox.question(self, "Close questionnaire", "Close without saving these answers?")
        if answer == QtWidgets.QMessageBox.Yes:
            super().reject()


class ReflectionDialog(QtWidgets.QDialog):
    """The resident study's last page, "Post Sim Reflection" (graphics_study.reflection_definition)."""

    def __init__(self, parent):
        super().__init__(parent)
        page = gs.reflection_definition()
        self._page = page
        self.setWindowTitle(page["title"])
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.practice_interest = None
        outer = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(page["title"])
        heading.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(heading)
        if page.get("instructions"):
            intro = QtWidgets.QLabel(page["instructions"])
            intro.setWordWrap(True)
            outer.addWidget(intro)
        item = page["practice_interest"]
        column = _question(outer, item["text"])
        row, self._buttons = _choice_row(self, item["options"], item.get("option_labels") or item["options"],
                                         self._picked)
        column.addLayout(row)
        item = page["comments"]
        column = _question(outer, item["text"] + ("  (optional)" if item.get("optional") else ""))
        self.comments = QtWidgets.QPlainTextEdit()
        self.comments.setMinimumHeight(140)
        self.comments.textChanged.connect(self._update)
        column.addWidget(self.comments)
        self.count = QtWidgets.QLabel("")
        self.count.setStyleSheet("color: gray;")
        column.addWidget(self.count)
        self.submit = QtWidgets.QPushButton("Submit")
        self.submit.setMinimumHeight(44)
        self.submit.setStyleSheet(SUBMIT_STYLE)
        self.submit.clicked.connect(self.accept)
        outer.addWidget(self.submit)
        self._update()

    def _picked(self, value):
        self.practice_interest = value
        self._update()

    def comment_text(self) -> str:
        return self.comments.toPlainText().strip()

    def _update(self):
        limit = self._page["comments"]["max_length"]
        length = len(self.comment_text())
        self.count.setText(f"{length} / {limit} characters" if length else "")
        self.submit.setEnabled(self.practice_interest is not None and length <= limit)

    def reject(self):
        answer = QtWidgets.QMessageBox.question(self, "Close questions", "Close without saving these answers?")
        if answer == QtWidgets.QMessageBox.Yes:
            super().reject()


class GraphicsStudyPanel(QtWidgets.QWidget):
    PHASE_TEXT = dict(study_mode.StudyPanel.PHASE_TEXT)
    PHASE_TEXT["ready"] = "Waiting for the participant's left pedal to begin this session"

    def __init__(self, ui, rehearsal_only=False):
        super().__init__()
        self.ui = ui
        self.rehearsal_only = rehearsal_only
        self.plan = None          # this participant: id, comparison, order, folders
        self.arm = 0              # 0 or 1
        self.stage = "idle"       # idle, session, tlx, between, background, preferences, reflection, complete
        self.session = None
        self.stand_in = None
        self._io = None
        self._launch_started = None
        self._done_handled = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self.refresh_status)
        self._build()
        self._status_timer.start()
        self.refresh_status()

    # ---- layout ----
    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        about = QtWidgets.QLabel(
            "Tutorial, then the same task twice in random order: cortical mastoidectomy with "
            "identification of the incus short process, once with the FIVRS RTX graphics and once "
            "with AMBF/CHAI3D. NASA-TLX on this monitor after each session. Anatomy A, headset, "
            "haptic device 1, no irrigation.")
        about.setWordWrap(True)
        about.setStyleSheet("color: gray;")
        layout.addWidget(about)

        form = QtWidgets.QGroupBox("1. Participant and comparison")
        rows = QtWidgets.QGridLayout(form)
        rows.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("lastname_firstname  (kept only in the study key)")
        self.name_edit.textChanged.connect(lambda *_: self._set_buttons())
        rows.addWidget(self.name_edit, 0, 1)
        rows.addWidget(QtWidgets.QLabel("Compare"), 1, 0)
        self.comparison = QtWidgets.QComboBox()
        for key, label in gs.COMPARISON_LABELS.items():
            self.comparison.addItem(label, key)
        self.comparison.setToolTip(
            "Upstream look: AMBF before the FIVRS changes (original volume shader, wide spotlight, "
            "no gray room).\nCurrent look: AMBF/CHAI3D as FIVRS renders it today (dark room, narrow "
            "spotlight).\nBoth are shown in the same headset eyepiece as the RTX view.")
        rows.addWidget(self.comparison, 1, 1)
        rows.addWidget(QtWidgets.QLabel("Time limit"), 2, 0)
        self.trial_minutes = QtWidgets.QDoubleSpinBox()
        self.trial_minutes.setRange(0.25, 30.0)
        self.trial_minutes.setSingleStep(0.25)
        self.trial_minutes.setValue(gs.TRIAL_SECONDS / 60.0)
        self.trial_minutes.setSuffix(" min of drilling per session")
        rows.addWidget(self.trial_minutes, 2, 1)
        layout.addWidget(form)

        self.start_button = QtWidgets.QPushButton("Start graphics study")
        self.start_button.setMinimumHeight(48)
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #1fae4b; color: white; font-size: 18px;"
            " font-weight: bold; border-radius: 6px; }"
            " QPushButton:hover { background-color: #23c455; }"
            " QPushButton:disabled { background-color: #3c4a40; color: #8a9a8e; }")
        self.start_button.clicked.connect(self.start)
        layout.addWidget(self.start_button)

        dev = QtWidgets.QHBoxLayout()
        self.rehearse_box = QtWidgets.QCheckBox("Rehearse without the simulator (development)")
        self.rehearse_box.setChecked(self.rehearsal_only)
        self.rehearse_box.setEnabled(not self.rehearsal_only)
        self.rehearse_box.toggled.connect(lambda *_: (self._set_buttons(), self.refresh_status()))
        self.desktop_box = QtWidgets.QCheckBox("Desktop instead of the headset (development)")
        self.desktop_box.setToolTip("No Quest. The CHAI3D arms then show AMBF's desktop window, without study text.")
        self.desktop_box.setVisible(not self.rehearsal_only)
        self.desktop_box.toggled.connect(lambda *_: (self._set_buttons(), self.refresh_status()))
        self.practice_box = QtWidgets.QCheckBox("Practice run in the headset, not a participant (development)")
        self.practice_box.setToolTip(
            "The real graphics study in the headset, but saved under graphics_study_dev with its own "
            "Study IDs and order balance: never a participant. The name is optional.")
        self.practice_box.setVisible(not self.rehearsal_only)
        self.practice_box.toggled.connect(lambda *_: (self._set_buttons(), self.refresh_status()))
        dev.addWidget(self.rehearse_box)
        dev.addWidget(self.desktop_box)
        dev.addWidget(self.practice_box)
        dev.addStretch(1)
        layout.addLayout(dev)

        system = QtWidgets.QGroupBox("System status")
        box = QtWidgets.QVBoxLayout(system)
        self.waiting_label = QtWidgets.QLabel()
        self.waiting_label.setWordWrap(True)
        self.waiting_label.setStyleSheet("font-weight: bold;")
        box.addWidget(self.waiting_label)
        self.checks_label = QtWidgets.QLabel()
        self.checks_label.setTextFormat(Qt.RichText)
        self.checks_label.setWordWrap(True)
        box.addWidget(self.checks_label)
        self.activity = QtWidgets.QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setMaximumBlockCount(200)
        self.activity.setFixedHeight(80)
        box.addWidget(self.activity)
        layout.addWidget(system)

        proctor = QtWidgets.QGroupBox("2. Proctor controls  (desktop only; the participant does not see this)")
        grid = QtWidgets.QGridLayout(proctor)
        grid.setColumnStretch(1, 1)
        self.id_label = QtWidgets.QLabel("Study ID: -")
        self.arm_label = QtWidgets.QLabel("")
        self.arm_label.setStyleSheet("font-size: 15px;")
        self.phase_label = QtWidgets.QLabel("Not started.")
        self.phase_label.setWordWrap(True)
        self.phase_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.step_label = QtWidgets.QLabel("")
        self.step_label.setWordWrap(True)
        self.clock_label = QtWidgets.QLabel("10:00")
        self.clock_label.setStyleSheet("font-size: 40px; font-weight: bold;")
        grid.addWidget(self.id_label, 0, 0, 1, 2)
        grid.addWidget(self.clock_label, 0, 2, 3, 1, Qt.AlignRight | Qt.AlignTop)
        grid.addWidget(self.arm_label, 1, 0, 1, 2)
        grid.addWidget(self.phase_label, 2, 0, 1, 2)
        grid.addWidget(self.step_label, 3, 0, 1, 3)

        self.incus_button = QtWidgets.QPushButton("Incus identified")
        self.incus_button.setMinimumHeight(44)
        self.incus_button.setStyleSheet("QPushButton { background-color: #0e7c6b; color: white; font-size: 17px;"
                                        " font-weight: bold; border-radius: 6px; }"
                                        " QPushButton:disabled { background-color: #7a9a94; }")
        self.incus_button.clicked.connect(self.incus_identified)
        self.skip_button = QtWidgets.QPushButton("Skip tutorial step")
        self.skip_button.clicked.connect(lambda: self.session and self.session.skip_step())
        self.begin_button = QtWidgets.QPushButton("Begin timed trial")
        self.begin_button.clicked.connect(self.begin_trial)
        self.end_button = QtWidgets.QPushButton("End session early")
        self.end_button.clicked.connect(self.end_early)
        self.next_button = QtWidgets.QPushButton("Start session 2")
        self.next_button.clicked.connect(self.start_next_arm)
        self.retry_button = QtWidgets.QPushButton("Restart this session")
        self.retry_button.clicked.connect(self.restart_arm)
        self.tlx_button = QtWidgets.QPushButton("Open questionnaire")
        self.tlx_button.clicked.connect(self.open_questionnaire)
        explain = [
            (self.incus_button, "Press when the participant says they see the short process of the incus. "
                                "Ends this session and records the time."),
            (self.skip_button, "Tutorial (session 1 only): move the participant past a stuck step."),
            (self.begin_button, "Normally the participant does this with the left pedal."),
            (self.end_button, "Stop the session now (hardware fault, or the participant stops)."),
            (self.tlx_button, "Open the current questionnaire (NASA-TLX, background, comparison or "
                              "Post Sim Reflection) again if it was closed without saving."),
            (self.next_button, "After the first questionnaire, once the headset is back on: opens the "
                               "simulator in the other graphics."),
            (self.retry_button, "The simulator closed or failed during a session: open it again for "
                                "the same session (no tutorial)."),
        ]
        for row, (button, text) in enumerate(explain, start=4):
            button.setMinimumWidth(170)
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            grid.addWidget(button, row, 0)
            grid.addWidget(label, row, 1, 1, 2)
        layout.addWidget(proctor)
        self._set_buttons()

    # ---- helpers ----
    @property
    def rehearsing(self) -> bool:
        return self.rehearse_box.isChecked()

    def _desktop(self) -> bool:
        return self.desktop_box.isChecked()

    def _dev_run(self) -> bool:
        """Desktop or headset practice: saved under graphics_study_dev, never a participant."""
        return self._desktop() or self.practice_box.isChecked()

    def graphics(self) -> str | None:
        return None if self.plan is None else self.plan["order"][self.arm]

    def activity_line(self, text: str):
        self.activity.appendPlainText(text.rstrip())

    def _say(self, text: str):
        if hasattr(self.ui, "print_info"):
            self.ui.print_info(text)
        else:
            self.activity_line(text)

    def _process_state(self, name):
        process = getattr(self.ui, name, None)
        if process is None:
            return "none"
        return {QProcess.NotRunning: "stopped", QProcess.Starting: "starting",
                QProcess.Running: "running"}[process.state()]

    def _set_buttons(self):
        phase = self.session.phase if self.session is not None else None
        in_session = self.stage == "session"
        idle = self.stage in ("idle", "complete")
        name_ok = bool(NAME_FORMAT.match(self.name_edit.text().strip()))
        self.start_button.setEnabled(idle and (self.rehearsing or self._dev_run() or name_ok))
        for widget in (self.name_edit, self.comparison, self.trial_minutes):
            widget.setEnabled(idle)
        self.rehearse_box.setEnabled(idle and not self.rehearsal_only)
        self.desktop_box.setEnabled(idle and not self.rehearsing)
        self.practice_box.setEnabled(idle and not self.rehearsing)
        self.incus_button.setEnabled(in_session and phase == "running")
        self.skip_button.setEnabled(in_session and phase == "tutorial")
        self.begin_button.setEnabled(in_session and phase in ("tutorial", "ready", "briefing"))
        self.end_button.setEnabled(in_session and phase in ("armed", "running"))
        self.tlx_button.setEnabled(self.stage in ("tlx", "background", "preferences", "reflection"))
        self.next_button.setEnabled(self.stage == "between")
        self.retry_button.setEnabled(self.stage == "session" and phase != "done" and not self.rehearsing
                                     and self._process_state("_ambf_process") == "stopped")

    # ---- status ----
    def refresh_status(self):
        if self.rehearsing:
            self.waiting_label.setText("Rehearsal: the stand-in window replaces the simulator.")
            self.checks_label.setText("")
            return
        headset = not self._desktop()
        ambf = self._process_state("_ambf_process")
        if self._launch_started is not None and ambf != "stopped":
            rows = study_checks.launch_status(ambf, self._process_state("_rtx_process") == "running",
                                              self._launch_started, headset=headset)
            if self.graphics() != "rtx":
                rows = [r for r in rows if r.name != "RTX view"]
            headline = study_checks.waiting_on(rows)
            if headline == "Everything is ready." and self.session is not None:
                headline = "Simulator running. " + self.PHASE_TEXT.get(self.session.phase, "")
        else:
            pedal = ""
            config = getattr(self.ui, "gui_configuration", None)
            if config is not None:
                pedal = str(config.footpedal_device.get())
            rows = study_checks.preflight(pedal, sim_running=ambf == "running", headset=headset)
            missing = [r for r in rows if r.required and r.state == study_checks.FAIL]
            headline = ("Ready to start." if not missing else
                        "Not ready: " + ", ".join(r.name for r in missing) + ".")
        self.waiting_label.setText(headline)
        cells = "".join(
            f"<tr><td style='color:{study_checks.COLOUR[r.state]}; font-weight:bold; padding-right:6px'>"
            f"{study_checks.ICON[r.state]}</td><td style='padding-right:10px'>{r.name}</td>"
            f"<td style='color:gray'>{r.detail}</td></tr>" for r in rows)
        self.checks_label.setText(f"<table>{cells}</table>")
        self._set_buttons()

    # ---- start ----
    def start(self):
        other = getattr(self.ui, "study_panel", None)
        if other is not None and other.session is not None and other.session.phase != "done":
            QtWidgets.QMessageBox.warning(self, "Graphics study", "A resident study session is running on the Study tab.")
            return
        comparison = self.comparison.currentData()
        if self.rehearsing:
            folder = None
            study_id = "R001"
            order = list(gs.COMPARISONS[comparison])
            import random
            random.shuffle(order)
            how = "random (rehearsal, not saved)"
        else:
            name = self.name_edit.text().strip()
            if not NAME_FORMAT.match(name):
                if not self._dev_run():
                    QtWidgets.QMessageBox.warning(self, "Graphics study",
                                                  "Enter the name as lastname_firstname, for example smith_jane.")
                    return
                name = name or ("dev_desktop" if self._desktop() else "dev_practice")
            if self.ui._ambf_process.state() != QProcess.NotRunning:
                QtWidgets.QMessageBox.warning(self, "Graphics study", "Close the running simulation first.")
                return
            if self._study_volume() is None:
                QtWidgets.QMessageBox.warning(self, "Graphics study", f"{study_mode.STUDY_VOLUME} is not in gui_setup.yaml.")
                return
            folder = gs.study_folder(self.ui.gui_configuration.recording_base_path.get(), self._dev_run())
            study_id = gs.next_study_id(folder)
            gs.add_to_key(folder, study_id, name)
            arm1, arm2, how = gs.choose_order(folder, comparison)
            order = [arm1, arm2]
            gs.record_assignment(folder, study_id, comparison, arm1, arm2, how)
            self.name_edit.clear()
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base = (study_rehearsal.rehearsal_dir() if folder is None else folder / "recordings") / f"{study_id}_{stamp}"
        self.plan = {"study_id": study_id, "comparison": comparison, "order": order, "folder": folder,
                     "base": base, "trial_seconds": self.trial_minutes.value() * 60.0, "how": how}
        self.arm = 0
        self.activity.clear()
        self.id_label.setText(f"Study ID: {study_id}" + (" (rehearsal)" if folder is None else ""))
        self._say(f"Graphics study {study_id}: {gs.COMPARISON_LABELS[comparison]}; order "
                  f"{gs.GRAPHICS[order[0]]} then {gs.GRAPHICS[order[1]]} ({how})")
        self._launch_arm(tutorial=True)

    def _study_volume(self):
        for button in self.ui.anatomy_button_group.buttons():
            if pathlib.Path(button.volume_adf).name == study_mode.STUDY_VOLUME:
                return button
        return None

    def _recording_dir(self, retry: int = 0) -> pathlib.Path:
        name = f"arm{self.arm + 1}_{self.graphics()}" + (f"_retry{retry}" if retry else "")
        return self.plan["base"] / name

    def _launch_arm(self, tutorial: bool, retry: int = 0):
        graphics = self.graphics()
        self.arm_label.setText(f"Session {self.arm + 1} of 2: {gs.GRAPHICS[graphics]}"
                               + ("  (with tutorial)" if tutorial else ""))
        recording_dir = self._recording_dir(retry)
        self.recording_dir = recording_dir
        metadata = {
            "graphics_study": {"comparison": self.plan["comparison"], "arm": self.arm + 1,
                               "graphics": graphics, "graphics_label": gs.GRAPHICS[graphics],
                               "order": self.plan["order"], "order_chosen": self.plan["how"],
                               "retry": retry},
        }
        text = gs.briefing(self.arm + 1, self.plan["trial_seconds"])
        if self.plan["folder"] is None:
            io = self._rehearsal_io()
            metadata["rehearsal"] = True
        else:
            volume = self._study_volume()
            volume.setChecked(True)
            volume_adf = (HERE / volume.volume_adf).resolve()
            desktop = self._desktop()
            for box, on in ((self.ui.check_rtx_view, graphics == "rtx"),
                            (self.ui.check_openxr_eyepiece, not desktop),
                            (self.ui.check_openvr_hmd, False), (self.ui.check_video_record, False)):
                if box is not None:
                    box.setChecked(on)
            self.ui.study_mode = {"study_id": self.plan["study_id"], "graphics": graphics}
            self._launch_started = time.time()
            self._say(f"{self.plan['study_id']} session {self.arm + 1}: starting the simulator ({gs.GRAPHICS[graphics]})")
            self.ui.pressed_start_simulation()
            if self.ui._ambf_process.state() == QProcess.NotRunning:
                self.ui.study_mode = None
                self._launch_started = None
                self.stage = "session"
                self.session = None
                self.phase_label.setText("The simulator did not start. See System status, then Restart this session.")
                self._set_buttons()
                return
            metadata.update({
                "volume_adf": str(volume_adf), "volume_name": volume.volume_name,
                "starting_burr_mm": self.ui.get_starting_burr_mm(),
                "study_mode": {"irrigation": False, "haptic_devices": 1,
                               "display": "desktop (development)" if desktop else "headset eyepiece"},
            })
            io = study_session.SessionIO(recorder_cmd=study_mode.recorder_command(volume_adf))
        self.session = study_session.StudySession(
            io, self.plan["study_id"], recording_dir, metadata, trial_seconds=self.plan["trial_seconds"],
            tutorial=tutorial, rtx_reload=(graphics == "rtx"), text=text)
        self.stage = "session"
        self._done_handled = False
        self._timer.start()
        self._set_buttons()

    def _rehearsal_io(self):
        if self.stand_in is None:
            self._io = study_rehearsal.RehearsalIO()
            self.stand_in = study_rehearsal.SimulatorStandIn(self._io)
            self.stand_in.closed.connect(self._stand_in_closed)
            self.stand_in.resize(1300, 560)
            self.stand_in.show()
        return self._io

    def _stand_in_closed(self):
        self.stand_in = None
        self.simulator_closed()

    # ---- proctor actions ----
    def begin_trial(self):
        if self.session is None:
            return
        if self.session.phase == "tutorial":
            answer = QtWidgets.QMessageBox.question(self, "Begin trial", "The tutorial is not finished. Begin anyway?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.session.error = None
        self.session.begin_trial()

    def incus_identified(self):
        if self.session is not None:
            self.session.incus_identified()
            self._say(f"Session {self.arm + 1}: incus identified at "
                      f"{self.session.incus_drilling_s or 0:.0f} s of drilling")

    def end_early(self):
        if self.session is None:
            return
        if QtWidgets.QMessageBox.question(self, "End session", "Stop this session now?") == QtWidgets.QMessageBox.Yes:
            self.session.end_early()

    def restart_arm(self):
        retry = 1
        while self._recording_dir(retry).exists():
            retry += 1
        self._say(f"Session {self.arm + 1}: restarting the simulator")
        self._launch_arm(tutorial=False, retry=retry)

    def start_next_arm(self):
        if self.stage != "between":
            return
        if self.plan["folder"] is not None and self.ui._ambf_process.state() != QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, "Graphics study", "The simulator is still closing. Try again in a moment.")
            return
        self.arm = 1
        self._launch_arm(tutorial=False)

    # ---- simulator ----
    def simulator_closed(self):
        """Called by the launcher when AMBF exits (or the rehearsal stand-in closes)."""
        self._launch_started = None
        if self.session is not None and self.session.phase in ("armed", "running"):
            self.session.end_early()
            self._say("Simulator closed during the session; recording stopped.")
        if self.stage == "session" and self.session is not None and self.session.phase != "done":
            self._timer.stop()
            self.session.io.clear_hud()
            self.session.io.set_drill_lock(False)
            self.phase_label.setText("The simulator closed before the session finished. Restart this session.")
        if self.stage == "session":
            self.ui.study_mode = None
        self._set_buttons()

    def end_participant(self):
        """Proctor (master Study tab, "End study"): stop this participant midway. What is
        already saved (key, assignment, finished sessions, questionnaires) is kept."""
        if self.stage in ("idle", "complete"):
            return
        stage = self.stage
        self._timer.stop()
        self._launch_started = None
        if self.session is not None and self.session.phase != "done":
            self.session.io.clear_hud()
            self.session.io.set_drill_lock(False)
        if (self.ui.study_mode or {}).get("graphics") is not None:
            self.ui.study_mode = None
        self.stage = "idle"
        if self.stand_in is not None:
            self.stand_in.close()
        study_id = self.plan["study_id"] if self.plan else "-"
        self.phase_label.setText(f"{study_id} ended by the proctor. What was saved is kept.")
        self.arm_label.setText("")
        self._say(f"Graphics study {study_id} ended by the proctor (session {self.arm + 1}, {stage})")
        self._set_buttons()

    # ---- loop ----
    def _tick(self):
        session = self.session
        if session is None:
            return
        phase = session.tick()
        self._set_buttons()
        self.phase_label.setText(session.error or self.PHASE_TEXT.get(phase, phase))
        if phase == "tutorial":
            step = study_session.TUTORIAL[session.step]
            status = ("done; waiting for the left pedal" if session.progress.step_done else "in progress")
            self.step_label.setText(f"Tutorial page {session.step + 1} of {len(study_session.TUTORIAL)}: "
                                    f"{step.title} ({status})")
        else:
            self.step_label.setText("")
        if phase == "running" and session.io.read_state().get("paused"):
            self.phase_label.setText("Session paused (pause menu open)")
        remaining = session.remaining()
        self.clock_label.setText(f"{int(remaining) // 60}:{int(remaining) % 60:02d}")
        if phase == "done" and not self._done_handled:
            self._done_handled = True
            self._arm_finished()

    def _arm_finished(self):
        session = self.session
        self._timer.stop()
        ended = {"incus_identified": "incus identified", "time_up": "time limit reached",
                 "proctor": "ended by the proctor", "resident": "ended by the participant (End trial)"}.get(session.ended_by, session.ended_by or "ended")
        self._say(f"Session {self.arm + 1} complete ({ended}): {self.recording_dir}")
        if self.plan["folder"] is not None:
            gs.record_trial(self.plan["folder"], self.plan["study_id"], self.plan["comparison"], self.arm + 1,
                            self.graphics(), session, self.recording_dir)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"Session {self.arm + 1} complete")
        box.setText(f"Session {self.arm + 1} of 2 is complete ({ended}).\n\nHelp the participant take off the "
                    "headset. The workload questionnaire opens on this monitor.")
        box.addButton("Open questionnaire", QtWidgets.QMessageBox.AcceptRole)
        box.exec_()
        session.io.clear_hud()
        if self.plan["folder"] is not None:
            self.ui.study_mode = None
            self.ui.close_simulation()
        self.stage = "tlx"
        self._set_buttons()
        self.open_tlx()

    def open_questionnaire(self):
        {"tlx": self.open_tlx, "background": self.open_background, "preferences": self.open_preferences,
         "reflection": self.open_reflection}.get(self.stage, lambda: None)()

    def open_tlx(self):
        if self.stage != "tlx":
            return
        dialog = TlxDialog(self, f"Session {self.arm + 1} of 2")
        dialog.resize(900, 760)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self.phase_label.setText("Questionnaire closed without saving. Press Open questionnaire.")
            return
        if self.plan["folder"] is not None:
            score = gs.record_tlx(self.plan["folder"], self.plan["study_id"], self.plan["comparison"],
                                  self.arm + 1, self.graphics(), dialog.answers, dialog.seconds())
        else:
            score = gs.raw_tlx(dialog.answers)
        self._say(f"Session {self.arm + 1} NASA-TLX saved (raw {score:.1f})")
        if self.arm == 0:
            self.stage = "between"
            self.phase_label.setText("Questionnaire saved. Put the headset back on the participant, "
                                     "then press Start session 2.")
            self.arm_label.setText(f"Next: session 2 of 2, {gs.GRAPHICS[self.plan['order'][1]]}")
        else:
            self.stage = "background"
            self._set_buttons()
            self.open_background()
            return
        self._set_buttons()

    def open_background(self):
        if self.stage != "background":
            return
        dialog = BackgroundDialog(self)
        dialog.resize(900, 820)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self.phase_label.setText("Background questionnaire closed without saving. Press Open questionnaire.")
            return
        row = gs.record_background(self.plan["folder"], self.plan["study_id"], self.plan["comparison"],
                                   dialog.result_answers(), dialog.seconds())
        self._say(f"Background saved ({row['role']}, {row['training_stage_unit']} {row['training_stage_value']})")
        self.stage = "preferences"
        self._set_buttons()
        self.open_preferences()

    def open_preferences(self):
        if self.stage != "preferences":
            return
        dialog = PreferenceDialog(self)
        dialog.resize(700, 360)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self.phase_label.setText("Comparison questions closed without saving. Press Open questionnaire.")
            return
        if self.plan["folder"] is not None:
            row = gs.record_preferences(self.plan["folder"], self.plan["study_id"], self.plan["comparison"],
                                        self.plan["order"], dialog.answers)
            self._say(f"Preferences saved: realism {row['realism_graphics']}, training {row['training_graphics']}")
        self.stage = "reflection"
        self._set_buttons()
        self.open_reflection()

    def open_reflection(self):
        if self.stage != "reflection":
            return
        dialog = ReflectionDialog(self)
        dialog.resize(800, 520)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self.phase_label.setText("Post Sim Reflection closed without saving. Press Open questionnaire.")
            return
        gs.record_reflection(self.plan["folder"], self.plan["study_id"], self.plan["comparison"],
                             dialog.practice_interest, dialog.comment_text())
        self._say(f"Post Sim Reflection saved (practice interest: {dialog.practice_interest})")
        self.stage = "complete"
        where = self.plan["folder"] or "(rehearsal: nothing saved)"
        self.phase_label.setText(f"Graphics study complete. Results in {where}")
        self.arm_label.setText("")
        self._say(f"Graphics study {self.plan['study_id']} complete")
        if self.stand_in is not None:
            self.stand_in.close()
        self._set_buttons()


def main():
    """The tab on its own with the rehearsal stand-in (no simulator)."""
    app = QtWidgets.QApplication(sys.argv)

    class _Ui:
        study_mode = None
        study_panel = None
        _ambf_process = None

    panel = GraphicsStudyPanel(_Ui(), rehearsal_only=True)
    panel.setWindowTitle("Graphics study (rehearsal)")
    panel.resize(820, 900)
    panel.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
