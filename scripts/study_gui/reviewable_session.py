"""Main tab's "Generate reviewable session": record a normal session for the Review tab.

Uses the UR5 recorder, so a session has the same files and opens the same way:
    <recording base>/reviewable_sessions/<person>/<anatomy>/session_NNN/
plus one row in <recording base>/reviewable_sessions/sessions.csv.

The drill waits until the recorder holds the starting volume, so no cut is missed. If the recorder
cannot start or stops, the drill is released and the simulator carries on: on the Main tab a
recording problem never blocks drilling. The box was ticked on purpose, so sessions are kept
without asking; Close Simulation waits for the recording so its replay can be checked.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import shlex
import signal
import subprocess
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, QTimer

import expert_drilling_tab as expert

SETTING = "reviewable_session"
DATA_FOLDER = "reviewable_sessions"
LOCK_TEXT = "review\n"
VIDEO_HZ = expert.VIDEO_HZ
START_TIMEOUT_S = 45.0
STOP_TIMEOUT_S = 120.0
# Bone, and the inner ear colors of the Main anatomies: bony labyrinth (tbone model), scalae (Delta).
LABELS = {"bone": "fff9db", "inner_ear": ["5b7b5b", "4184ff", "ffdebe"]}


def split_name(name: str) -> tuple[str, str]:
    parts = (name or "").split()
    if not parts:
        return "Unnamed", ""
    return (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else (parts[0], "")


class ReviewableSession:
    def __init__(self, ui):
        self.ui = ui
        self.recorder = None
        self.session = None
        self.request = None
        self._log = None
        self._ready_since = None
        self._stop_deadline = None
        self._close_after = False
        self.check = QtWidgets.QCheckBox("Generate reviewable session")
        self.check.setToolTip("Record this session (pose, force, voxels, burr, video) to open in the Review tab.")
        self.check.setChecked(bool(ui.gui_configuration.yaml_data.get(SETTING, False)))
        self.check.toggled.connect(lambda on: ui.gui_configuration.save_value(SETTING, bool(on)))
        layout = ui.findChild(QtWidgets.QHBoxLayout, "recordingLayout")
        if layout is not None:
            layout.addWidget(self.check)
        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.tick)

    # ---------------------------------------------------------------- state
    def wanted(self) -> bool:
        return self.check.isChecked() and not self.ui.study_mode and not self.ui.expert_launch

    def active(self) -> bool:
        return self.request is not None or self.recorder is not None

    def root(self) -> pathlib.Path:
        return pathlib.Path(str(self.ui.gui_configuration.recording_base_path.get())) / DATA_FOLDER

    @staticmethod
    def _lock():
        expert.DRILL_LOCK.parent.mkdir(parents=True, exist_ok=True)
        expert.DRILL_LOCK.write_text(LOCK_TEXT)

    @staticmethod
    def _unlock():
        if expert.DRILL_LOCK.is_file() and expert.DRILL_LOCK.read_text() == LOCK_TEXT:
            expert.DRILL_LOCK.unlink(missing_ok=True)

    # ---------------------------------------------------------------- launcher hooks
    def before_launch(self):
        if self.wanted():
            self._lock()

    def launch_failed(self):
        self.request = None
        self._unlock()

    def launched(self, request: dict):
        """AMBF is up. request: adf, name, laterality, user, burr, view, rtx, video, checks, options."""
        if not self.wanted():
            return
        self.request = dict(request, launched_at=time.monotonic())
        self.check.setEnabled(False)
        self.timer.start()
        self.ui.print_info("Reviewable session: recording starts when the simulator is up.")

    def _simulator_ready(self) -> bool:
        if self.request["rtx"]:
            return self.ui._rtx_process.state() == QProcess.Running and getattr(self.ui, "_rtx_loading", True) is None
        return time.monotonic() - self.request["launched_at"] >= expert.NO_RTX_SETTLE_S

    def spawn(self, command, folder):
        self._log = open(folder / "recorder.log", "w")
        return subprocess.Popen(["bash", "-lc", command], stdout=self._log, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def signal(self, number):
        os.killpg(self.recorder.pid, number)

    def _start_recording(self):
        request = self.request
        first, last = split_name(request["user"])
        adf = pathlib.Path(request["adf"]).resolve()
        specimen = {"name": request["name"] or adf.stem, "laterality": request.get("laterality") or "",
                    "labels": LABELS, "adf_path": str(adf)}
        folder, number = expert.create_session_folder(self.root(), first, last, specimen)
        started = datetime.datetime.now().isoformat(timespec="seconds")
        mode = request["video"]
        metadata = {
            "format": "fivrs_expert_drilling_v1", "source": "main_tab", "development": False,
            "participant": {"first_name": first, "last_name": last},
            "specimen": specimen, "session_number": number, "started": started,
            "video": {"mode": mode, "rate_hz": VIDEO_HZ if mode != "none" else 0,
                      "stereo_base_mm": 23.0 if mode == "stereo" else None,
                      "stereo_base_sim": 23.0 * 0.01007 if mode == "stereo" else None},
            "launch": {"checks": request["checks"], "render": request["options"],
                       "starting_burr_mm": request["burr"], "startup_view": request["view"]},
            "volume": expert.volume_pose(adf),
            "software": {"commit": expert.git_commit()},
        }
        (folder / "session.json").write_text(json.dumps(metadata, indent=2))
        command = ("set +u && source /opt/ros/jazzy/setup.bash && "
                   'source "$HOME/ros_ambf_ws/install/setup.bash" && '
                   f"exec {shlex.quote(str(expert.RECORDER_PYTHON))} -u {shlex.quote(str(expert.RECORDER))} "
                   f"--output_dir {shlex.quote(str(folder))} --volume_adf {shlex.quote(str(adf))} "
                   f"--video {mode} --video_hz {VIDEO_HZ}")
        self.recorder = self.spawn(command, folder)
        self.session = {"folder": folder, "number": number, "first": first, "last": last, "specimen": specimen,
                        "started": started, "video": mode, "started_at": time.monotonic(), "ready": False}
        self.request = None

    def tick(self):
        if self.request is not None:
            if self.ui._ambf_process.state() == QProcess.NotRunning:
                self._finish_nothing()
            elif self._simulator_ready():
                self._start_recording()
            elif time.monotonic() - self.request["launched_at"] > 300.0:
                self.ui.print_info("Reviewable session: the simulator never became ready; not recording.")
                self._finish_nothing()
            return
        if self.recorder is None:
            return
        session = self.session
        if not session["ready"] and (session["folder"] / "recording.started").exists():
            session["ready"] = True
            self._unlock()
            self.ui.print_info(f"Reviewable session {session['number']} is recording.")
        code = self.recorder.poll()
        now = time.monotonic()
        late_start = not session["ready"] and self._stop_deadline is None and now - session["started_at"] > START_TIMEOUT_S
        late_stop = self._stop_deadline is not None and now > self._stop_deadline
        if code is None and (late_start or late_stop):
            self.signal(signal.SIGKILL)
            code = self.recorder.wait()
        if code is not None:
            self._recording_ended(code)

    def end(self, close_after=False):
        """The session is over. close_after: then close the simulator."""
        self._close_after = self._close_after or close_after
        if self.request is not None:
            self._finish_nothing()
            return
        if self.recorder is None or self._stop_deadline is not None:
            return
        self._lock()  # the final volume and the last removals describe the same moment
        if self.recorder.poll() is None:
            self.signal(signal.SIGINT)
        self._stop_deadline = time.monotonic() + STOP_TIMEOUT_S
        if close_after:
            self.ui.print_info("Finishing the reviewable session, then closing the simulator...")
            self.ui.button_start_simulation.setEnabled(False)

    def _finish_nothing(self):
        self.request = None
        self._done()

    def _recording_ended(self, code):
        session, expected = self.session, self._stop_deadline is not None
        self.recorder, self.session, self._stop_deadline = None, None, None
        if self._log:
            self._log.close()
            self._log = None
        if not session["ready"]:
            expert.discard_session(session["folder"])
            self.ui.print_info(f"WARNING! The reviewable recording did not start (code {code}); drilling is not recorded.")
        else:
            self._report(session, code, expected)
        self._done()

    def _report(self, session, code, expected):
        expert.append_session_row(self.root(), {
            "session_id": f"{expert.person_folder(session['first'], session['last'])}__"
                          f"{expert.specimen_folder(session['specimen'])}__{session['number']:03d}",
            "last_name": session["last"], "first_name": session["first"], "specimen": session["specimen"]["name"],
            "session_number": session["number"], "started": session["started"], "video": session["video"],
            "folder": str(session["folder"])})
        try:
            summary = json.loads((session["folder"] / "session.json").read_text()).get("summary")
        except (OSError, ValueError):
            summary = None
        if not expected or summary is None:
            self.ui.print_info(f"WARNING! Reviewable session {session['number']} stopped early (code {code}). "
                               f"Log: {session['folder']}/recorder.log")
        else:
            self.ui.print_info(f"Saved reviewable session {session['number']}: {summary['duration_s']:.0f} s, "
                               f"{summary['voxels_removed']:,} voxels. Open it in the Review tab.")

    def _done(self):
        self.timer.stop()
        self._unlock()
        self.check.setEnabled(True)
        self.ui.button_start_simulation.setEnabled(True)
        close, self._close_after = self._close_after, False
        if close and self.ui._ambf_process.state() != QProcess.NotRunning:
            self.ui.close_simulation()

    def shutdown(self):
        """The launcher is closing: finish the recording so it is not lost."""
        self._close_after = False
        self.request = None
        if self.recorder is not None:
            self.end()
            try:
                code = self.recorder.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.signal(signal.SIGKILL)
                code = self.recorder.wait()
            self._recording_ended(code)
        self.timer.stop()
        self._unlock()
