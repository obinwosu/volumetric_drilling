import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import feedback_order as fo  # noqa: E402


def participant_file(study_dir, study_id, view, completed_at=None):
    folder = study_dir / fo.PARTICIPANTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    session = {"participant_id": study_id, "run_id": "run", "first_review_view": view,
               "case_completed_at": completed_at}
    payload = {"schema": "local_expert_study_state_v1", "participant_id": study_id,
               "expert_study": {"sessions": {f"{study_id}::run": session}}}
    (folder / f"{study_id}.json").write_text(json.dumps(payload))


def test_first_resident_gets_event_based(tmp_path):
    assert fo.next_first_view(tmp_path) == "event-based"


def test_alternates_after_each_completion(tmp_path):
    order = []
    for number in range(1, 7):
        study_id = f"S{number:03d}"
        view = fo.first_view_for(tmp_path, study_id)
        order.append(view)
        assert fo.record_completion(tmp_path, study_id, view, "test")
    assert order == ["event-based", "summary"] * 3
    assert fo.counts(tmp_path) == {"event-based": 3, "summary": 3}


def test_aborted_session_does_not_count(tmp_path):
    fo.record_completion(tmp_path, "S001", "event-based", "test")
    # S002 is assigned summary but never finishes (no log row, no completion).
    participant_file(tmp_path, "S002", "summary", completed_at=None)
    assert fo.first_view_for(tmp_path, "S003") == "summary"


def test_fewer_completed_view_wins_over_alternation(tmp_path):
    for study_id in ("S001", "S002"):
        fo.record_completion(tmp_path, study_id, "summary", "test")
    assert fo.next_first_view(tmp_path) == "event-based"


def test_completion_in_participant_file_counts_without_log(tmp_path):
    fo.record_completion(tmp_path, "S001", "event-based", "test")
    participant_file(tmp_path, "S002", "summary", completed_at="2026-11-13T15:00:00+00:00")
    assert fo.counts(tmp_path) == {"event-based": 1, "summary": 1}
    # Tie: the most recent completion was summary first.
    assert fo.next_first_view(tmp_path) == "event-based"


def test_stored_order_is_kept_for_a_retry(tmp_path):
    participant_file(tmp_path, "S001", "summary")
    assert fo.first_view_for(tmp_path, "S001") == "summary"


def test_record_completion_is_once_per_resident(tmp_path):
    assert fo.record_completion(tmp_path, "S001", "summary", "test")
    assert not fo.record_completion(tmp_path, "S001", "summary", "test")
    assert fo.counts(tmp_path) == {"event-based": 0, "summary": 1}


def test_unreadable_participant_file_is_ignored(tmp_path):
    folder = tmp_path / fo.PARTICIPANTS_DIR
    folder.mkdir()
    (folder / "S001.json").write_text("{not json")
    assert fo.next_first_view(tmp_path) == "event-based"
