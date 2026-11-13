"""Which feedback view a resident sees first, balanced over completed sessions.

The two views (Video-Based Feedback, key "event-based"; Metric-Based Feedback, key "summary") are counterbalanced across residents who
finish the feedback, not across Study IDs, so aborted or crashed sessions do not
skew the groups. A session counts as completed when it is in
feedback_order_log.csv (written when the resident presses "Return to the
simulator" on the end page) or when the feedback system has marked it complete in
its participant file (study_state_participants/<Study ID>.json), in case the
browser was closed instead.

The next resident gets the view with fewer completed sessions; on a tie, the
opposite of the most recently completed one. Once a resident has an order, it
never changes (the feedback system also locks it in their participant file).
"""
from __future__ import annotations

import csv
import datetime
import json
import pathlib

VIEWS = ("event-based", "summary")
FIRST_EVER = "event-based"
ORDER_LOG = "feedback_order_log.csv"
PARTICIPANTS_DIR = "study_state_participants"  # the feedback system's --study-state study_state.json
FIELDS = ["study_id", "first_review_view", "completed_at", "source"]


def other(view: str) -> str:
    return VIEWS[1] if view == VIEWS[0] else VIEWS[0]


def _participant_sessions(study_dir: pathlib.Path):
    """(study_id, first_review_view, case_completed_at) for every session the feedback system stored."""
    folder = pathlib.Path(study_dir) / PARTICIPANTS_DIR
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sessions = ((payload.get("expert_study") or {}).get("sessions") or {}).values()
        for session in sessions:
            study_id = session.get("participant_id") or payload.get("participant_id")
            view = session.get("first_review_view")
            if study_id and view in VIEWS:
                yield str(study_id), view, session.get("case_completed_at")


def completed_sessions(study_dir: pathlib.Path) -> dict:
    """Study ID -> {"first_review_view", "completed_at"} for every finished session."""
    done = {}
    log = pathlib.Path(study_dir) / ORDER_LOG
    if log.is_file():
        with open(log, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("study_id") and row.get("first_review_view") in VIEWS:
                    done[row["study_id"]] = {"first_review_view": row["first_review_view"],
                                             "completed_at": row.get("completed_at") or ""}
    for study_id, view, completed_at in _participant_sessions(study_dir):
        if completed_at and study_id not in done:
            done[study_id] = {"first_review_view": view, "completed_at": str(completed_at)}
    return done


def counts(study_dir: pathlib.Path) -> dict:
    tally = {view: 0 for view in VIEWS}
    for row in completed_sessions(study_dir).values():
        tally[row["first_review_view"]] += 1
    return tally


def stored_view(study_dir: pathlib.Path, study_id: str):
    """The order the feedback system already gave this resident, if any (e.g. a retried launch)."""
    for sid, view, _completed in _participant_sessions(study_dir):
        if sid == study_id:
            return view
    return None


def next_first_view(study_dir: pathlib.Path) -> str:
    done = completed_sessions(study_dir)
    tally = {view: 0 for view in VIEWS}
    for row in done.values():
        tally[row["first_review_view"]] += 1
    if tally[VIEWS[0]] != tally[VIEWS[1]]:
        return min(VIEWS, key=lambda view: tally[view])
    if not done:
        return FIRST_EVER
    # Most recent completion; equal times fall back to the order in the log.
    rows = list(done.values())
    last = max(range(len(rows)), key=lambda i: (_time(rows[i]["completed_at"]), i))
    return other(rows[last]["first_review_view"])


def first_view_for(study_dir: pathlib.Path, study_id: str) -> str:
    """The order for this resident: their stored one, else the balanced next one."""
    return stored_view(study_dir, study_id) or next_first_view(study_dir)


def record_completion(study_dir: pathlib.Path, study_id: str, view: str, source: str) -> bool:
    """Add the resident to the log once. Returns False when they were already in it."""
    if view not in VIEWS:
        raise ValueError(f"unknown feedback view: {view}")
    log = pathlib.Path(study_dir) / ORDER_LOG
    if log.is_file():
        with open(log, newline="") as handle:
            if any(row.get("study_id") == study_id for row in csv.DictReader(handle)):
                return False
    new = not log.is_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"study_id": study_id, "first_review_view": view,
                         "completed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                         "source": source})
    return True


def _time(text: str) -> datetime.datetime:
    try:
        value = datetime.datetime.fromisoformat(str(text))
    except ValueError:
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    if value.tzinfo is None:
        value = value.astimezone()
    return value
