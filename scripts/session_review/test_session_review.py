"""Session review: replay at any time, curves, export, and the window's controls.

    QT_QPA_PLATFORM=offscreen python3 scripts/session_review/test_session_review.py
"""
import csv
import os
import pathlib
import sys
import tempfile

os.environ["FIVRS_REVIEW_NO_3D"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import yaml  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

import make_demo_session as demo  # noqa: E402
import review_app  # noqa: E402
import session_data as data  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
SPECIMEN = yaml.safe_load((data.REPO / "scripts/expert_drilling/specimens.yaml").read_text())["specimens"][0]


def test_everything_on_one_session():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp) / "expert_drillings_dev" / "Session_Demo" / "Specimen_1" / "session_001"
        summary = demo.build(folder, SPECIMEN, stereo=True)
        assert summary["replay_check"]["status"] == "exact", summary["replay_check"]
        assert summary["volume_resets"] == 1 and summary["voxels_removed"] > 0
        assert [b["size_mm"] for b in summary["burr_sequence"]] == [6, 4, 2]

        session = data.ReviewSession(folder)
        reference = data.dataset.ExpertSession(folder)
        # The replay equals the plain rebuild, forward, backward, and on both sides of the reset.
        for t in (0.0, 5.0, 12.0, 20.9, 21.5, 25.9, 8.0, 0.5):
            solid = session.voxels.labels_at(t) > 0
            assert np.array_equal(solid, reference.occupancy_at(t + session.t0)), t
        reference.close()
        early, late = (session.voxels.labels_at(t) > 0 for t in (1.0, 20.0))
        assert late.sum() < early.sum() and not (late & ~early).any()  # bone only goes away
        assert (session.voxels.labels_at(21.5) > 0).sum() > late.sum()  # the reset brought it back
        assert session.burr_at(3.0)[0] == 6 and session.burr_at(10.0)[0] == 4 and session.burr_at(20.0)[0] == 2
        for key in ("speed", "acceleration", "force", "voxels", "removal_rate", "burr", "on", "pos_x", "fov"):
            assert key in session.channels, key
        assert session.value_at("burr", 10.0) == 4.0
        assert session.video.eyes == ("left", "right") and session.video.frame_at(3.0, "right").shape == (360, 640, 3)

        mp4 = pathlib.Path(tmp) / "clip.mp4"
        frames = data.export_video(session, mp4, 2.0, 4.0, "both")
        capture = cv2.VideoCapture(str(mp4))
        ok, image = capture.read()
        assert ok and image.shape[:2] == (360, 1280) and abs(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - frames) <= 1
        capture.release()
        table = pathlib.Path(tmp) / "curves.csv"
        data.export_csv(session, table, 9.5, 10.5, ["burr", "speed"])
        rows = list(csv.DictReader(table.open()))
        assert len(rows) == 101 and float(rows[50]["burr_mm"]) == 4.0 and rows[50]["speed_mm_per_s"]
        session.close()

        found = data.discover(tmp)
        assert len(found) == 1 and found[0]["development"] and found[0]["number"] == 1

        window = review_app.ReviewWindow(tmp)
        window.resize(1400, 900)
        window.show()
        window.load(folder)
        assert window.session is not None and window.tree.topLevelItemCount() == 1
        window.seek(10.0)
        assert "4 mm" in window.label_now.text()
        window.set_mark(0)
        window.seek(12.0)
        window.set_mark(1)
        assert window._range() == (10.0, 12.0)
        window.step("next")
        assert window.t > 12.0
        for index in range(window.combo_layout.count()):
            window.combo_layout.setCurrentIndex(index)
            app.processEvents()
        window.combo_layout.setCurrentIndex(0)
        window.combo_window.setCurrentIndex(3)
        window.t = 11.0
        window.combo_speed.setCurrentIndex(3)
        window.toggle_play()
        for _ in range(8):
            window._last_tick -= 0.05
            window._tick()
            app.processEvents()
        assert window.playing and window.t > 11.2
        window.pause()
        window.grab()
        window.close()


def test_a_session_without_video_or_finish_still_opens():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp) / "expert_drillings_dev" / "A_B" / "Specimen_1" / "session_001"
        demo.build(folder, SPECIMEN)
        (folder / "video.h5").unlink()
        meta = data.json.loads((folder / "session.json").read_text())
        meta.pop("summary")
        (folder / "session.json").write_text(data.json.dumps(meta))
        window = review_app.ReviewWindow(tmp)
        window.show()
        window.load(folder)
        window.seek(5.0)
        assert "did not finish" in window.info.text() and not window.button_export_video.isEnabled()
        window.close()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
