"""Expert drilling tab: defaults, names, session numbering.

    QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_expert_drilling_tab.py
"""
import json
import pathlib
import shutil
import sys
import tempfile
import types

from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import expert_drilling_tab as tab  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def fake_ui(base):
    process = types.SimpleNamespace(state=lambda: QProcess.NotRunning)
    config = types.SimpleNamespace(yaml_data={}, recording_base_path=types.SimpleNamespace(get=lambda: base))
    return types.SimpleNamespace(gui_configuration=config, _ambf_process=process, study_mode=None,
                                 expert_launch=None, print_info=lambda *_: None)


def test_defaults_and_specimens():
    with tempfile.TemporaryDirectory() as tmp:
        panel = tab.ExpertDrillingPanel(fake_ui(tmp))
        assert [s["name"] for s in panel.specimens][:3] == ["Specimen 1", "Specimen 2", "Specimen 3"]
        assert panel.combo_preset.currentData() == "fastest"
        assert not panel.check_suction.isChecked() and not panel.check_irrigation.isChecked()
        assert panel.video_mode() == "mono"
        panel.check_openxr.setChecked(True)
        assert panel.video_mode() == "stereo"
        panel.check_video.setChecked(False)
        assert panel.video_mode() == "none"
        panel.check_video.setChecked(True)
        panel.first_name.setText("Ada")
        panel.last_name.setText("Lovelace")
        request = panel.launch_request()
        assert request["options"]["suction"] is False and request["options"]["preset"] == "fastest"
        assert request["user"] == "Ada Lovelace" and request["name"] == "Specimen 1"


def test_development_sessions_are_separate():
    with tempfile.TemporaryDirectory() as tmp:
        panel = tab.ExpertDrillingPanel(fake_ui(tmp))
        assert panel.check_development.isChecked()
        assert panel.root() == pathlib.Path(tmp) / "expert_drillings_dev"
        panel.check_development.setChecked(False)
        assert panel.root() == pathlib.Path(tmp) / "expert_drillings"
        specimen = panel.specimens[0]
        _, real = tab.create_session_folder(panel.root(), "Ada", "Lovelace", specimen)
        panel.check_development.setChecked(True)
        _, dev = tab.create_session_folder(panel.root(), "Ada", "Lovelace", specimen)
        assert (real, dev) == (1, 1)  # each has its own numbering


class FakeRecorder:
    """Stands in for the recorder process: ready at once, writes a summary when signalled."""

    pid = 0

    def __init__(self, folder):
        self.folder = folder
        self.code = None
        (folder / "recording.started").write_text("1")

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def stop(self):
        meta = json.loads((self.folder / "session.json").read_text())
        meta["summary"] = {"duration_s": 12.0, "voxels_removed": 40, "strokes": 2,
                           "replay_check": {"status": "exact"}}
        (self.folder / "session.json").write_text(json.dumps(meta))
        self.code = 0


def session_panel(tmp):
    tab.DRILL_LOCK = pathlib.Path(tmp) / "runtime" / "study_drill_lock"
    ui = fake_ui(tmp)
    ui.state = QProcess.NotRunning
    ui._ambf_process = types.SimpleNamespace(state=lambda: ui.state)
    ui._rtx_process = types.SimpleNamespace(state=lambda: ui.state)
    ui._rtx_loading = None
    ui._pupil_process = types.SimpleNamespace(state=lambda: QProcess.NotRunning)
    panel = tab.ExpertDrillingPanel(ui)
    panel._poll.stop()

    def start(request):
        ui.expert_launch = request
        ui.state = QProcess.Running

    def close():
        ui.state = QProcess.NotRunning
        ui.expert_launch = None
        panel.simulator_closed()

    ui.start_expert_simulation, ui.close_simulation = start, close
    panel._spawn_recorder = lambda command, folder: FakeRecorder(folder)
    panel._signal_recorder = lambda number: panel.recorder.stop()
    panel.first_name.setText("Ada")
    panel.last_name.setText("Lovelace")
    return panel, ui


def test_no_record_button_and_names_required():
    with tempfile.TemporaryDirectory() as tmp:
        panel, ui = session_panel(tmp)
        assert not hasattr(panel, "button_record")
        panel.first_name.setText("")
        panel.pressed_sim()
        assert ui.state == QProcess.NotRunning and not tab.DRILL_LOCK.exists()


def test_session_records_itself_then_save():
    with tempfile.TemporaryDirectory() as tmp:
        panel, ui = session_panel(tmp)
        ui._rtx_loading = {"label": "loading"}
        panel.pressed_sim()
        assert ui.state == QProcess.Running and tab.DRILL_LOCK.exists()  # locked while loading
        panel._tick()
        assert panel.recorder is None and tab.DRILL_LOCK.exists()  # not ready yet
        ui._rtx_loading = None
        panel._tick()   # recorder starts and is ready
        assert panel.recording() and not tab.DRILL_LOCK.exists()
        folder = panel.session["folder"]
        panel.rtx_view_closed()
        assert panel._ask_box is not None and tab.DRILL_LOCK.exists() and ui.state == QProcess.Running
        panel._tick()   # the recorder finishes while the question is open
        assert not (panel.root() / "sessions.csv").exists()
        panel._answer(True)
        assert ui.state == QProcess.NotRunning and not tab.DRILL_LOCK.exists()
        assert folder.exists() and "Lovelace" in (panel.root() / "sessions.csv").read_text()
        assert panel.last_session == folder


def test_discard_goes_to_trash_and_number_is_free():
    with tempfile.TemporaryDirectory() as tmp:
        panel, ui = session_panel(tmp)
        trashed = []
        original = tab.move_to_trash
        tab.move_to_trash = lambda path: (trashed.append(str(path)), shutil.rmtree(path), True)[-1]
        try:
            panel.pressed_sim()
            panel._tick()
            folder = panel.session["folder"]
            assert panel.session["number"] == 1
            panel.rtx_view_closed()
            panel._answer(False)   # answered before the recorder finished
            assert folder.exists()
            panel._tick()
        finally:
            tab.move_to_trash = original
        assert trashed == [str(folder)] and not folder.exists()
        assert ui.state == QProcess.NotRunning and not (panel.root() / "sessions.csv").exists()
        panel.pressed_sim()
        panel._tick()
        assert panel.session["number"] == 1
        panel.recorder.wait = lambda timeout=None: panel.recorder.code
        panel.shutdown()  # the launcher closing saves what is open
        assert "Lovelace" in (panel.root() / "sessions.csv").read_text()


def test_save_value_replaces_a_whole_block():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "setup.yaml"
        path.write_text("a: 1\nblock:\n  one: 1\n  two: 2\nz: 9\n")
        config = types.SimpleNamespace(yaml_data={}, yaml_file_path=path)
        tab.gui_setup.GUIConfiguration.save_value(config, "block", {"one": 5})
        assert path.read_text() == "a: 1\nblock:\n  one: 5\nz: 9\n"


def test_session_numbers():
    specimen = {"name": "Specimen 2"}
    with tempfile.TemporaryDirectory() as tmp:
        one, n1 = tab.create_session_folder(tmp, "Ada", "Lovelace", specimen)
        two, n2 = tab.create_session_folder(tmp, "ada", "LOVELACE", specimen)  # same person
        other, n3 = tab.create_session_folder(tmp, "Ada", "Lovelace", {"name": "Specimen 3"})
        assert (n1, n2, n3) == (1, 2, 1)
        assert one.parent == two.parent and one.name == "session_001" and two.name == "session_002"
        assert other.parent.name == "Specimen_3"


def test_names_cannot_leave_the_folder():
    assert "/" not in tab.person_folder("../../etc", "a/b")
    assert tab.name_part("  José  María ") == "José-María"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
