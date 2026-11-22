"""Recorder -> post-processing -> replay, without the simulator.

    feedback_system/.venv/bin/python scripts/expert_drilling/test_expert_recording.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import expert_dataset  # noqa: E402
import expert_recorder  # noqa: E402

BONE, EAR = (255, 249, 219, 255), (91, 123, 91, 255)


def volume():
    rgba = np.zeros((20, 20, 20, 4), np.uint8)
    rgba[2:18, 2:18, 2:18] = BONE
    rgba[9:11, 9:11, 9:11] = EAR
    return rgba


def write_meta(folder):
    (folder / "session.json").write_text(json.dumps({
        "specimen": {"labels": {"bone": "fff9db", "inner_ear": "5b7b5b"}},
        "launch": {"starting_burr_mm": 4},
        "volume": {"R_world_volume": np.eye(3).tolist(), "p_world_volume": [0, 0, 0], "dimensions_sim": [1, 1, 1]},
    }))


def record(folder, lose_one=False):
    write_meta(folder)
    recorder = expert_recorder.SessionRecorder(folder)
    base = volume()
    recorder.add_epoch(base.copy(), 0)
    for k in range(200):
        t = 10 + k * 0.01
        recorder.add("drill", (t, t, -0.5 + 0.05 * 9.5, -0.025, -0.025, 0, 0, 0, 1))
        recorder.add("force", (t, t, 0.5, 0, 0))
    recorder.add("burr", (10.0, 10.0, 4.0))
    recorder.on_voxels(10.2, 0, [(9, 9, 9), (10, 9, 9)])
    recorder.on_voxels(10.25, 1, [(9, 10, 9)])
    recorder.on_voxels(11.5, 2, [(3, 3, 3)])  # later stroke, far from the burr
    final = base.copy()
    for x, y, z in ((9, 9, 9), (10, 9, 9), (9, 10, 9), (3, 3, 3)):
        final[x, y, z] = 0
    if lose_one:
        final[5, 5, 5] = 0
    recorder.add_final(final)
    recorder.close()
    return expert_dataset.finalize(folder)


def test_replay_labels_and_strokes():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        summary = record(folder)
        assert summary["replay_check"]["status"] == "exact", summary
        assert summary["voxels_removed"] == 4 and summary["inner_ear_voxels_removed"] == 3
        assert summary["bone_voxels_removed"] == 1 and summary["strokes"] == 2
        assert summary["detached_voxels_cleared"] == 1, summary
        session = expert_dataset.ExpertSession(folder)
        assert session.occupancy_at(10.21)[9, 9, 9] == False and session.occupancy_at(10.21)[9, 10, 9] == True
        assert session.occupancy_at(9.0)[9, 9, 9] == True
        session.close()
        assert expert_dataset.finalize(folder)["strokes"] == 2  # can run again


def test_lost_voxel_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        summary = record(pathlib.Path(tmp), lose_one=True)
        assert summary["replay_check"]["status"] == "mismatch"
        assert summary["replay_check"]["voxels_ambf_removed_not_recorded"] == 1


def test_stopping_mid_drill_is_not_a_mismatch():
    for late_batch_explained in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            write_meta(folder)
            recorder = expert_recorder.SessionRecorder(folder)
            recorder.add_epoch(volume(), 0)
            recorder.add("drill", (1, 1, 0, 0, 0, 0, 0, 0, 1))
            recorder.on_voxels(1.0, 0, [(5, 5, 5)])
            before = recorder.batch_count() if late_batch_explained else recorder.batch_count() + 1
            recorder.on_voxels(1.1, 1, [(6, 5, 5)])  # after the capture was requested; still in its volume
            final = volume()
            final[5, 5, 5] = 0
            recorder.add_final(final, before)
            recorder.close()
            check = expert_dataset.finalize(folder)["replay_check"]
            assert check["status"] == ("exact" if late_batch_explained else "mismatch"), check


def test_reset_starts_an_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        write_meta(folder)
        recorder = expert_recorder.SessionRecorder(folder)
        recorder.add_epoch(volume(), 0)
        recorder.add("drill", (1, 1, 0, 0, 0, 0, 0, 0, 1))
        recorder.on_voxels(2.0, 0, [(9, 9, 9)])
        recorder.add("drill", (4, 4, 0, 0, 0, 0, 0, 0, 1))
        recorder.add_epoch(volume(), recorder.batch_count())
        recorder.add("drill", (9, 9, 0, 0, 0, 0, 0, 0, 1))
        recorder.on_voxels(5.0, 1, [(4, 4, 4)])
        final = volume()
        final[4, 4, 4] = 0
        recorder.add_final(final)
        recorder.close()
        summary = expert_dataset.finalize(folder)
        assert summary["replay_check"]["status"] == "exact", summary
        assert summary["volume_resets"] == 1
        session = expert_dataset.ExpertSession(folder)
        assert session.occupancy_at(3.0)[9, 9, 9] == False
        assert session.occupancy_at(6.0)[9, 9, 9] == True and session.occupancy_at(6.0)[4, 4, 4] == False
        session.close()


def test_burr_changes_are_tracked():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        write_meta(folder)
        recorder = expert_recorder.SessionRecorder(folder)
        recorder.add_epoch(volume(), 0)
        def state(t, size, style="cutting", limit=10):
            return {"sim_time": t, "size_mm": size, "style": style, "voxel_removal_limit": limit,
                    "burr_diameter_sim": size * 0.01007}
        for t, size, style, limit in ((1.0, 6, "cutting", 10), (2.0, 6, "cutting", 10), (3.0, 4, "cutting", 6),
                                      (4.0, 4, "diamond", 6), (5.0, 4, "diamond", 6)):
            recorder.add("state", expert_recorder.state_row(state(t, size, style, limit), t))
            recorder.add("drill", (t, t, 0, 0, 0, 0, 0, 0, 1))
        recorder.on_voxels(2.5, 0, [(5, 5, 5)])
        recorder.on_voxels(3.05, 1, [(6, 5, 5), (7, 5, 5)])  # no time gap: split by the burr change
        recorder.on_voxels(4.5, 2, [(8, 5, 5)])
        recorder.close()
        summary = expert_dataset.finalize(folder)
        assert summary["starting_burr_mm"] == 6
        assert [(b["size_mm"], b["type"]) for b in summary["burr_sequence"]] == [
            (6, "cutting"), (4, "cutting"), (4, "diamond")]
        assert summary["voxels_by_burr_mm"] == {"4": 3, "6": 1} and summary["strokes"] == 3
        with h5py.File(folder / "ground_truth.h5") as f:
            assert f["voxels/burr_mm"][()].tolist() == [6, 4, 4, 4]
            assert f["voxels/burr_diamond"][()].tolist() == [0, 0, 0, 1]
            columns = f["derived/strokes"].attrs["columns"].split(",")
            assert f["derived/strokes"][:, columns.index("burr_mm")].tolist() == [6, 4, 4]
            assert f["derived/burr_timeline"][:, 4].tolist() == [10, 6, 6]


def test_session_without_a_change_keeps_its_burr():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        write_meta(folder)
        recorder = expert_recorder.SessionRecorder(folder)
        recorder.add_epoch(volume(), 0)
        recorder.add("state", expert_recorder.state_row(
            {"sim_time": 1.0, "size_mm": 2, "style": "cutting", "voxel_removal_limit": 3, "burr_diameter_sim": 0.02}, 1.0))
        recorder.on_voxels(1.5, 0, [(5, 5, 5)])
        recorder.close()
        summary = expert_dataset.finalize(folder)
        assert summary["starting_burr_mm"] == 2 and summary["voxels_by_burr_mm"] == {"2": 1}


def test_video_round_trip():
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        recorder = expert_recorder.SessionRecorder(folder, ("left", "right"))
        image = np.full((48, 64, 3), 120, np.uint8)
        for k in range(5):
            recorder.frame(float(k), float(k), [0, 0, 0, 0, 0, 0, 1], 0.2, 1.5, [image, image // 2])
        recorder.close()
        with h5py.File(folder / "video.h5") as f:
            assert f["frames"].shape[0] == 5 and f["right"].shape[0] == 5
            right = cv2.imdecode(f["right"][3], cv2.IMREAD_COLOR)
            assert right.shape == image.shape and abs(int(right.mean()) - 60) <= 2


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
