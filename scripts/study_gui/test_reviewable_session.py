"""Main tab's reviewable session and the Review tab, with a stand-in recorder (no simulator, no Touch).

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_reviewable_session.py
"""
import json
import os
import pathlib
import sys
import tempfile
import types

os.environ["FIVRS_REVIEW_NO_3D"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtCore import QProcess  # noqa: E402

import expert_drilling_tab as expert  # noqa: E402
import review_tab  # noqa: E402
import reviewable_session as rs  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
ADF = str(expert.REPO / "ADF" / "volume_256_RT143.yaml")


class FakeRecorder:
    pid = 0

    def __init__(self, folder, starts=True):
        self.folder, self.code = folder, None
        if starts:
            (folder / "recording.started").write_text("1")

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def stop(self):
        meta = json.loads((self.folder / "session.json").read_text())
        meta["summary"] = {"duration_s": 5.0, "voxels_removed": 9, "strokes": 1, "replay_check": {"status": "exact"}}
        (self.folder / "session.json").write_text(json.dumps(meta))
        self.code = 0


class FakeUi(QtWidgets.QWidget):
    def __init__(self, tmp):
        super().__init__()
        self.layout_row = QtWidgets.QHBoxLayout(self)
        self.layout_row.setObjectName("recordingLayout")
        self.saved, self.messages, self.closed = {}, [], 0
        self.study_mode, self.expert_launch, self._rtx_loading = None, None, None
        self.state = QProcess.Running
        self._ambf_process = self._rtx_process = types.SimpleNamespace(state=lambda: self.state)
        self.button_start_simulation = QtWidgets.QPushButton()
        self.gui_configuration = types.SimpleNamespace(
            yaml_data={}, save_value=self.saved.__setitem__,
            recording_base_path=types.SimpleNamespace(get=lambda: tmp))
        self.print_info = self.messages.append

    def close_simulation(self):
        self.closed += 1
        self.state = QProcess.NotRunning


def controller(tmp):
    expert.DRILL_LOCK = pathlib.Path(tmp) / "runtime" / "study_drill_lock"
    ui = FakeUi(tmp)
    session = rs.ReviewableSession(ui)
    session.spawn = lambda command, folder: FakeRecorder(folder)
    session.signal = lambda number: session.recorder.stop()
    return session, ui


def request():
    return {"adf": ADF, "name": "Anatomy A", "user": "Ada Lovelace", "laterality": "right", "burr": 4,
            "view": "default", "rtx": True, "video": "mono", "options": {}, "checks": {"rtx": True}}


def test_off_by_default_records_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        session, _ = controller(tmp)
        assert not session.check.isChecked()
        session.before_launch()
        session.launched(request())
        assert not session.active() and not expert.DRILL_LOCK.exists()


def test_session_is_saved_and_close_waits_for_it():
    with tempfile.TemporaryDirectory() as tmp:
        session, ui = controller(tmp)
        session.check.setChecked(True)
        assert ui.saved[rs.SETTING] is True
        session.before_launch()
        assert expert.DRILL_LOCK.exists()
        session.launched(request())
        session.tick()  # recorder started
        session.tick()  # recorder ready
        assert not expert.DRILL_LOCK.exists()
        folder = pathlib.Path(tmp, rs.DATA_FOLDER, "Lovelace_Ada", "Anatomy_A", "session_001")
        assert json.loads((folder / "session.json").read_text())["specimen"]["labels"] == rs.LABELS
        session.end(close_after=True)
        assert ui.closed == 0 and expert.DRILL_LOCK.exists()  # the final volume comes first
        session.tick()
        assert ui.closed == 1 and not expert.DRILL_LOCK.exists() and not session.active()
        assert (pathlib.Path(tmp) / rs.DATA_FOLDER / "sessions.csv").is_file()


def test_recorder_failure_never_blocks_the_drill():
    with tempfile.TemporaryDirectory() as tmp:
        session, ui = controller(tmp)
        session.check.setChecked(True)
        session.spawn = lambda command, folder: types.SimpleNamespace(pid=0, poll=lambda: 1, wait=lambda timeout=None: 1)
        expert.discard_session = lambda folder: __import__("shutil").rmtree(folder)
        session.before_launch()
        session.launched(request())
        session.tick()
        session.tick()
        assert not expert.DRILL_LOCK.exists() and not session.active() and ui.state == QProcess.Running
        assert any("did not start" in m for m in ui.messages)


def test_not_used_by_studies_or_ur5():
    with tempfile.TemporaryDirectory() as tmp:
        session, ui = controller(tmp)
        session.check.setChecked(True)
        ui.study_mode = {"study_id": "G001"}
        assert not session.wanted()
        ui.study_mode, ui.expert_launch = None, {"options": {}}
        assert not session.wanted()


def test_several_inner_ear_colors():
    rgba = np.array([[[0xff, 0xf9, 0xdb, 255], [0x5b, 0x7b, 0x5b, 255], [0x41, 0x84, 0xff, 255], [1, 2, 3, 255]]], np.uint8)
    sys.path.insert(0, str(review_tab.REVIEW_DIR))
    import session_data  # also finds h5py for the system Python
    assert np.array_equal(session_data.dataset.label_of(rgba, rs.LABELS), [[1, 2, 2, 0]])


def test_review_tab_builds_when_opened_and_keeps_keys_to_itself():
    with tempfile.TemporaryDirectory() as tmp:
        ui = FakeUi(tmp)
        tabs = QtWidgets.QTabWidget(ui)
        tabs.setObjectName("tabWidget")
        name = QtWidgets.QLineEdit()
        tabs.addTab(name, "Main")
        panel = review_tab.ReviewPanel(ui)
        tabs.addTab(panel, "Review")
        ui.show()
        assert panel.window_widget is None
        panel.open_session()
        app.processEvents()
        assert tabs.currentWidget() is panel and panel.window_widget is not None
        shortcuts = panel.window_widget.findChildren(QtWidgets.QShortcut)
        assert shortcuts and all(s.context() != review_tab.Qt.WindowShortcut for s in shortcuts)
        panel.window_widget.playing = True
        tabs.setCurrentIndex(0)
        app.processEvents()
        assert not panel.window_widget.playing
        panel.shutdown()
        ui.hide()


if __name__ == "__main__":
    for test_name, test in list(globals().items()):
        if test_name.startswith("test_"):
            test()
            print("ok", test_name)
