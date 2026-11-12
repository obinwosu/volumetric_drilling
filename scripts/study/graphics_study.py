"""Graphics study: the same drilling task with two renderers, NASA-TLX after each.

Protocol (launcher tab: scripts/study_gui/graphics_study_mode.py):
    1. The tutorial from the resident study, in the first arm's graphics.
    2. Arm 1: cortical mastoidectomy, identify the incus short process. The proctor
       presses "Incus identified" when the participant calls it; 10 min drilling cap.
    3. NASA-TLX (raw) on the desktop monitor, headset off.
    4. The simulator restarts in the other graphics; arm 2, same task, no tutorial.
    5. NASA-TLX again.
    6. Two preference questions comparing the sessions (more realistic, rather train on).

The proctor picks the comparison:
    upstream_vs_rtx  AMBF/CHAI3D as it looked before the FIVRS changes  vs  RTX
    chai_vs_rtx      AMBF/CHAI3D as it looks today (dark room, narrow spot)  vs  RTX
Every arm uses Anatomy A, the headset, the same eyepiece framing and the same
controls; only the renderer and its look differ.

Order is random and balanced within each comparison: the order used less often
so far is taken, a tie is broken at random (permuted blocks of two).

Files, under <recording base>/graphics_study/ (graphics_study_dev/ for desktop runs):
    participant_key.csv   study_id, name   (the name is kept only here)
    assignments.csv       study_id, comparison, arm1, arm2, how the order was chosen
    tlx.csv               one row per NASA-TLX
    trials.csv            one row per drilling arm (time to incus, how it ended)
    preferences.csv       one row per participant: the session picked, and its graphics
    recordings/<id>_<stamp>/arm<n>_<graphics>/   recorder output + metadata.json
"""
from __future__ import annotations

import csv
import datetime
import pathlib
import random

GRAPHICS = {
    "rtx": "FIVRS RTX (path traced)",
    "chai": "AMBF/CHAI3D, current look",
    "upstream": "AMBF/CHAI3D, upstream look",
}

COMPARISONS = {
    "upstream_vs_rtx": ("upstream", "rtx"),
    "chai_vs_rtx": ("chai", "rtx"),
}
COMPARISON_LABELS = {
    "upstream_vs_rtx": "Upstream AMBF/CHAI3D look  vs  FIVRS RTX",
    "chai_vs_rtx": "Current AMBF/CHAI3D look  vs  FIVRS RTX",
}

ID_PREFIX = "G"
KEY_FILE = "participant_key.csv"
ASSIGNMENTS_FILE = "assignments.csv"
TLX_FILE = "tlx.csv"
TRIALS_FILE = "trials.csv"
PREFERENCES_FILE = "preferences.csv"
TRIAL_SECONDS = 600.0

# NASA Task Load Index (Hart & Staveland, 1988), raw (unweighted) form.
# 21 marks from 0 to 100 in steps of 5.
TLX_SCALES = [
    ("mental", "Mental Demand", "How mentally demanding was the task?", "Very Low", "Very High"),
    ("physical", "Physical Demand", "How physically demanding was the task?", "Very Low", "Very High"),
    ("temporal", "Temporal Demand", "How hurried or rushed was the pace of the task?", "Very Low", "Very High"),
    ("performance", "Performance",
     "How successful were you in accomplishing what you were asked to do?", "Perfect", "Failure"),
    ("effort", "Effort",
     "How hard did you have to work to accomplish your level of performance?", "Very Low", "Very High"),
    ("frustration", "Frustration",
     "How insecure, discouraged, irritated, stressed, and annoyed were you?", "Very Low", "Very High"),
]
TLX_STEPS = list(range(0, 101, 5))


def raw_tlx(answers: dict) -> float:
    """Mean of the six subscales (0-100)."""
    values = [float(answers[key]) for key, *_ in TLX_SCALES]
    return round(sum(values) / len(values), 2)


# ---- headset wording (participants are not told which renderer is which) ----

def briefing(arm_number: int, trial_seconds: float = TRIAL_SECONDS) -> dict:
    minutes = int(round(trial_seconds / 60.0))
    if arm_number == 1:
        lines = [
            "This is a small study comparing two ways of displaying the drilling simulator. "
            "You will do two drilling sessions, each with a different display.",
            "",
            "In each session, perform a cortical mastoidectomy and identify the short process "
            "of the incus. Tell the proctor as soon as you see the incus; that ends the session. "
            f"Each session is limited to {minutes} minutes of drilling.",
            "",
            "After each session you will remove the headset and answer a short workload "
            "questionnaire.",
            "",
            "Your performance will not be used to evaluate you.",
        ]
        return {"briefing_title": "Session 1 of 2", "briefing_lines": lines,
                "armed_lines": ["Begin drilling when ready.",
                                f"The {minutes}-minute timer starts when the drill first removes bone."]}
    lines = [
        "This session uses a different display of the simulator. The task is the same: "
        "perform a cortical mastoidectomy and identify the short process of the incus.",
        "",
        f"Tell the proctor as soon as you see the incus. The limit is again {minutes} minutes of drilling.",
    ]
    return {"briefing_title": "Session 2 of 2", "briefing_lines": lines,
            "ready_title": "Session 2 of 2",
            "ready_lines": ["You may move the drill and the microscope to get used to this display."],
            "ready_advance": "Press the left foot pedal when you are ready",
            "armed_lines": ["Begin drilling when ready.",
                            f"The {minutes}-minute timer starts when the drill first removes bone."]}


# ---- participant key, assignment and results ----

def study_folder(recording_base, development: bool = False) -> pathlib.Path:
    return pathlib.Path(str(recording_base)).expanduser() / ("graphics_study_dev" if development else "graphics_study")


def _rows(path: pathlib.Path):
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _append(path: pathlib.Path, fields, row: dict) -> None:
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def next_study_id(folder: pathlib.Path) -> str:
    used = {row["study_id"] for row in _rows(folder / KEY_FILE)}
    number = 1
    while f"{ID_PREFIX}{number:03d}" in used:
        number += 1
    return f"{ID_PREFIX}{number:03d}"


def add_to_key(folder: pathlib.Path, study_id: str, name: str) -> None:
    _append(folder / KEY_FILE, ["study_id", "name", "created"],
            {"study_id": study_id, "name": name, "created": _now()})


def choose_order(folder: pathlib.Path, comparison: str, rng: random.Random | None = None):
    """(arm1, arm2, how): the order used less often in this comparison, a tie at random."""
    pair = COMPARISONS[comparison]
    counts = {pair: 0, pair[::-1]: 0}
    for row in _rows(folder / ASSIGNMENTS_FILE):
        if row.get("comparison") == comparison:
            key = (row.get("arm1"), row.get("arm2"))
            if key in counts:
                counts[key] += 1
    a, b = counts[pair], counts[pair[::-1]]
    if a != b:
        order = pair if a < b else pair[::-1]
        how = "balanced"
    else:
        order = (rng or random.SystemRandom()).choice([pair, pair[::-1]])
        how = "random"
    return order[0], order[1], how


def record_assignment(folder: pathlib.Path, study_id: str, comparison: str, arm1: str, arm2: str, how: str):
    _append(folder / ASSIGNMENTS_FILE, ["study_id", "comparison", "arm1", "arm2", "order_chosen", "created"],
            {"study_id": study_id, "comparison": comparison, "arm1": arm1, "arm2": arm2,
             "order_chosen": how, "created": _now()})


TLX_FIELDS = (["study_id", "comparison", "arm", "graphics"] + [key for key, *_ in TLX_SCALES]
              + ["raw_tlx", "seconds_to_complete", "completed"])


def record_tlx(folder: pathlib.Path, study_id: str, comparison: str, arm: int, graphics: str,
               answers: dict, seconds: float | None = None) -> float:
    score = raw_tlx(answers)
    row = {"study_id": study_id, "comparison": comparison, "arm": arm, "graphics": graphics,
           "raw_tlx": score, "seconds_to_complete": None if seconds is None else round(seconds, 1),
           "completed": _now()}
    row.update({key: int(answers[key]) for key, *_ in TLX_SCALES})
    _append(folder / TLX_FILE, TLX_FIELDS, row)
    return score


TRIAL_FIELDS = ["study_id", "comparison", "arm", "graphics", "ended_by", "incus_identified",
                "time_to_incus_s", "drilling_s", "paused_s", "trial_limit_s", "tutorial_in_this_arm",
                "recording_dir", "finished"]


def record_trial(folder: pathlib.Path, study_id: str, comparison: str, arm: int, graphics: str,
                 session, recording_dir) -> dict:
    """session: the finished study_session.StudySession for this arm."""
    incus = session.incus_drilling_s
    row = {
        "study_id": study_id, "comparison": comparison, "arm": arm, "graphics": graphics,
        "ended_by": session.ended_by or "",
        "incus_identified": int(incus is not None),
        "time_to_incus_s": "" if incus is None else round(incus, 1),
        "drilling_s": round(session.drilling_elapsed, 1),
        "paused_s": round(session.paused_s, 1),
        "trial_limit_s": session.trial_seconds,
        "tutorial_in_this_arm": int(session.tutorial),
        "recording_dir": str(recording_dir),
        "finished": _now(),
    }
    _append(folder / TRIALS_FILE, TRIAL_FIELDS, row)
    return row


# ---- end-of-study preference questions ----
# Answers are "1", "2" (the session) or "none" (no difference).
PREFERENCE_QUESTIONS = [
    ("realism", "Which session looked more realistic?"),
    ("training", "Which session would you rather use to train?"),
]
PREFERENCE_CHOICES = [("1", "Session 1"), ("2", "Session 2"), ("none", "No difference")]

PREFERENCE_FIELDS = (["study_id", "comparison", "arm1", "arm2"]
                     + [f"{key}_{part}" for key, _ in PREFERENCE_QUESTIONS for part in ("session", "graphics")]
                     + ["completed"])


def record_preferences(folder: pathlib.Path, study_id: str, comparison: str, order, answers: dict) -> dict:
    """answers: {question key: "1" | "2" | "none"}. Stores the session and the graphics it had."""
    row = {"study_id": study_id, "comparison": comparison, "arm1": order[0], "arm2": order[1],
           "completed": _now()}
    for key, _question in PREFERENCE_QUESTIONS:
        choice = str(answers[key])
        if choice not in ("1", "2", "none"):
            raise ValueError(f"{key}: {choice!r}")
        row[f"{key}_session"] = choice
        row[f"{key}_graphics"] = "none" if choice == "none" else order[int(choice) - 1]
    _append(folder / PREFERENCES_FILE, PREFERENCE_FIELDS, row)
    return row
