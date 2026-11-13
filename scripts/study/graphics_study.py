"""Graphics study: the same drilling task with two renderers, NASA-TLX after each.

Protocol (launcher tab: scripts/study_gui/graphics_study_mode.py):
    1. The tutorial from the resident study, in the first arm's graphics.
    2. Arm 1: cortical mastoidectomy, identify the incus short process. The proctor
       presses "Incus identified" when the participant calls it; 10 min drilling cap.
    3. NASA-TLX (raw) on the desktop monitor, headset off.
    4. The simulator restarts in the other graphics; arm 2, same task, no tutorial.
    5. NASA-TLX again.
    6. The resident study's background questionnaire (role, training year, prior
       mastoidectomies, VR / simulator experience).
    7. Two preference questions comparing the sessions (more realistic, rather train on).
    8. The resident study's last page, "Post Sim Reflection" (would practise on this
       simulator; optional comments).
The participant can end a session early from the pause menu (End trial), as in
the resident study. Steps 6 and 8 use the resident study's wording, read from
feedback_system/src/evidence_system/study_surveys.py, and record its version.

The proctor picks the comparison:
    upstream_vs_rtx  AMBF/CHAI3D as it looked before the FIVRS changes  vs  RTX
    chai_vs_rtx      AMBF/CHAI3D as it looks today (dark room, narrow spot)  vs  RTX
Every arm uses Anatomy A, the headset, the same eyepiece framing and the same
controls; only the renderer and its look differ.

Order is random and balanced within each comparison: the order used less often
so far is taken, a tie is broken at random (permuted blocks of two).

Files, under <recording base>/graphics_study/ (graphics_study_dev/ for desktop and
headset practice runs):
    participant_key.csv   study_id, name   (the name is kept only here)
    assignments.csv       study_id, comparison, arm1, arm2, how the order was chosen
    tlx.csv               one row per NASA-TLX
    trials.csv            one row per drilling arm (time to incus, how it ended)
    preferences.csv       one row per participant: the session picked, and its graphics
    background.csv        one row per participant: the background questionnaire
    reflection.csv        one row per participant: Post Sim Reflection
    recordings/<id>_<stamp>/arm<n>_<graphics>/   recorder output + metadata.json
"""
from __future__ import annotations

import csv
import datetime
import importlib.util
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[2]
SURVEYS_PY = REPO / "feedback_system" / "src" / "evidence_system" / "study_surveys.py"

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
BACKGROUND_FILE = "background.csv"
REFLECTION_FILE = "reflection.csv"
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

def _armed_lines(minutes: int) -> list:
    # As the resident study's "Timed trial" screen.
    return ["Begin drilling when ready.",
            f"The {minutes}-minute timer starts when the drill first removes bone.",
            "If you finish early, hold the left foot pedal and select End trial."]


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
            "It is okay if you do not find the incus in the allotted time. If you want to end "
            "a session early, hold the left foot pedal to open the pause menu and select End trial.",
            "",
            "After each session you will remove the headset and answer a short workload "
            "questionnaire. After the second, a few questions about your background and the "
            "two sessions follow.",
            "",
            "Your performance will not be used to evaluate you.",
        ]
        return {"briefing_title": "Session 1 of 2", "briefing_lines": lines,
                "armed_lines": _armed_lines(minutes)}
    lines = [
        "This session uses a different display of the simulator. The task is the same: "
        "perform a cortical mastoidectomy and identify the short process of the incus.",
        "",
        f"Tell the proctor as soon as you see the incus. The limit is again {minutes} minutes of drilling.",
        "",
        "To end the session early, hold the left foot pedal to open the pause menu and select End trial.",
    ]
    return {"briefing_title": "Session 2 of 2", "briefing_lines": lines,
            "ready_title": "Session 2 of 2",
            "ready_lines": ["You may move the drill and the microscope to get used to this display."],
            "ready_advance": "Tap the left foot pedal when you are ready",
            "armed_lines": _armed_lines(minutes)}


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


# ---- the resident study's background questionnaire and last page ----

_surveys = None


def resident_surveys():
    """The resident study's survey module (study_surveys.py), loaded on its own so the
    feedback_system package and its dependencies are not imported."""
    global _surveys
    if _surveys is None:
        if not SURVEYS_PY.is_file():
            raise FileNotFoundError(f"The resident study's questionnaires are missing: {SURVEYS_PY}")
        spec = importlib.util.spec_from_file_location("fivrs_resident_surveys", SURVEYS_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _surveys = module
    return _surveys


def background_definition() -> dict:
    return resident_surveys().SURVEY_DEFINITION["background"]


def reflection_definition() -> dict:
    return resident_surveys().SURVEY_DEFINITION["closing"]


def _experience_keys():
    return [key for key in background_definition()["fields"]
            if key not in ("role", "training_stage", "cortical_mastoidectomies")]


def background_fields():
    return (["study_id", "comparison", "role", "training_stage_value", "training_stage_unit",
             "cortical_mastoidectomies"] + _experience_keys() + ["survey_version", "seconds", "completed"])


def record_background(folder, study_id: str, comparison: str, answers: dict, seconds: float = 0.0) -> dict:
    """answers: as the resident study stores them, checked by its validate_background:
    {"role", "training_stage": {"status", "value", "unit"}, "cortical_mastoidectomies", <experience>}."""
    surveys = resident_surveys()
    row_in = surveys.validate_background(answers)
    stage = row_in["training_stage"]
    row = {"study_id": study_id, "comparison": comparison, "role": row_in["role"],
           "training_stage_value": stage["value"], "training_stage_unit": stage["unit"],
           "cortical_mastoidectomies": row_in["cortical_mastoidectomies"],
           "survey_version": surveys.SURVEY_VERSION, "seconds": round(seconds, 1), "completed": _now()}
    for key in _experience_keys():
        row[key] = row_in[key]
    if folder is not None:
        _append(pathlib.Path(folder) / BACKGROUND_FILE, background_fields(), row)
    return row


REFLECTION_FIELDS = ["study_id", "comparison", "practice_interest", "comments", "survey_version", "completed"]


def record_reflection(folder, study_id: str, comparison: str, practice_interest: str, comments: str = "") -> dict:
    page = reflection_definition()
    if practice_interest not in page["practice_interest"]["options"]:
        raise ValueError(f"practice_interest: {practice_interest!r}")
    comments = (comments or "").strip()
    if len(comments) > page["comments"]["max_length"]:
        raise ValueError("comments are too long")
    row = {"study_id": study_id, "comparison": comparison, "practice_interest": practice_interest,
           "comments": comments, "survey_version": resident_surveys().SURVEY_VERSION, "completed": _now()}
    if folder is not None:
        _append(pathlib.Path(folder) / REFLECTION_FILE, REFLECTION_FIELDS, row)
    return row
