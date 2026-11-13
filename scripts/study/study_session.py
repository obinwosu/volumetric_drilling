"""Study session controller: tutorial, bone reset, timed trial, end.

Plain Python, driven by ``tick()`` about ten times a second from the launcher.
It reads the simulator's /tmp/fivrs-rtx/drill_state.json, writes the headset
text (study_hud.json, drawn by scripts/rtx_view/headset_hud.py), locks the
drill (study_drill_lock), resets the bone, and starts and stops the recorder.
Everything that touches files or processes goes through ``SessionIO`` so tests
can replace it.

Flow:
    tutorial  -> each step ticks off from the simulator's live state; left pedal advances
    ready     -> "Tap the left foot pedal to begin your study session" (or the proctor's Begin)
    briefing  -> the study introduction covers the view while, behind it, the drill is
                 locked, the burr goes back to 6 mm cutting, the bone is restored in AMBF and
                 reloaded in RTX and the recorder starts; the left pedal (once all that is
                 done) begins the trial
    armed     -> recorder running, drill unlocked; timer waits for first bone removed
    running   -> 10:00 of drilling time; pauses (pause menu) do not count
    done      -> drill locked, recorder stopped, metadata written
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "rtx_view"))
import headset_hud  # noqa: E402  (camera projection shared with the headset drawing)

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
TRIAL_SECONDS = 600.0
WARNING_SECONDS = 60.0
RESET_TIMEOUT_S = 45.0
RECORDER_START_TIMEOUT_S = 30.0
PAUSE_FOOTER = "Hold the left pedal any time for the pause menu."

BRIEFING_TITLE = "The study session"
BRIEFING_LINES = [
    "This is a small user study examining two different methods of automated feedback in "
    "virtual reality mastoidectomy. You will be given 10 minutes to perform a cortical "
    "mastoidectomy with the goal of identifying the incus short process.",
    "",
    "Your performance is not a primary endpoint in the study and will not be used to evaluate "
    "you whatsoever.",
    "",
    "It is okay if you do not complete the task in the allotted time.",
    "",
    "After 10 minutes, you will be asked to complete a handful of brief questionnaires and "
    "review two forms of feedback.",
]
BRIEFING_FOOTER = "Tap the left foot pedal once you are ready to begin."

@dataclass
class Step:
    key: str
    title: str
    lines: list          # how to do it, shown in the headset
    pictures: list = field(default_factory=list)  # headset_hud device pictures with marked buttons


ADVANCE_TEXT = "Tap the left foot pedal to advance"

# Wording shown in the headset. Each step is checked from drill_state.json in _tutorial();
# once done it turns green, the resident may keep practising, and the left pedal advances.
TUTORIAL = [
    Step("welcome", "Welcome",
         ["This tutorial includes steps intended to familiarize you with the drilling simulator. "
          "After completing a tutorial step, tap the left foot pedal to advance. Once you complete "
          "all tutorial steps, you may begin the study session."],
         [{"device": "pedals", "marks": {"left": "Advance"}}]),
    Step("drill_move", "Moving the drill",
         ["Move the stylus to move the drill around the surgical field."],
         [{"device": "stylus", "marks": {}}]),
    Step("reposition", "Repositioning the haptic device",
         ["The haptic device has a set working area. Holding the stylus action button allows "
          "you to reposition the device without moving the drill.",
          "Hold the action button, move the stylus, then release."],
         [{"device": "stylus", "marks": {"action": "Hold"}}]),
    Step("recenter", "Recentering the drill in view",
         ["To quickly recenter the drill and keep the haptic device in the working area, "
          "double-tap the stylus action button."],
         [{"device": "stylus", "marks": {"action": "Double-tap"}}]),
    Step("camera", "Moving the microscope",
         ["The view in the simulator is controlled by the position of the Quest headset. To move "
          "the camera, press and hold the button on the microscope handle and then reposition "
          "the Quest headset.",
          "Move the view until the yellow target is inside the centre circle."],
         [{"device": "scope", "marks": {"button": "Hold"}}, {"device": "headset", "marks": {}}]),
    Step("zoom", "Zooming in and out",
         ["To zoom in, press the button on the microscope AND press the left foot pedal. To zoom "
          "out, press the button on the microscope AND press the middle foot pedal.",
          "Zoom in, then zoom out.",
          "Keep the microscope button held while zooming: tapping the left pedal on its own "
          "moves on to the next step."],
         [{"device": "scope", "marks": {"button": "Hold"}},
          {"device": "pedals", "marks": {"left": "Zoom in", "middle": "Zoom out"}}]),
    Step("drill", "Activating the drill",
         ["To activate the drill, press and hold the right pedal. Practice making contact and "
          "removing bone.",
          "This practice bone is restored before the study session."],
         [{"device": "pedals", "marks": {"right": "Drill"}}]),
    Step("pause", "The pause menu",
         ["The pause menu pauses the simulator. The burr size and type are changed there.",
          "To open the pause menu, press and hold the left foot pedal. Open it now."],
         [{"device": "pedals", "marks": {"left": "Open"}}]),
]

# Shown above the pause menu while it is open during the tutorial (headset_hud draws it
# only while the simulator reports paused). The pause step's page covers opening only.
MENU_HELP = {
    "title": "Using the pause menu",
    "lines": ["Tap the left pedal to move down, and press the middle pedal to select. "
              "Each list ends with Back.",
              "To close the menu, select Resume, or hold the left pedal and choose Yes."],
    "pictures": [{"device": "pedals", "marks": {"left": "Down / Close", "middle": "Select"}}],
}

# Thresholds for each tutorial check.
DRILL_TRAVEL_UNITS = 0.5   # simulator units of tip travel (about 25 mm)
DEVICE_TRAVEL_M = 0.015    # stylus travel while clutched, metres
HOLD_SECONDS = 1.0         # clutch held this long, when the plugin reports no stylus position
ZOOM_EACH_WAY = 0.5        # doublings of magnification, in and then out
DRILL_VOXELS = 200         # voxels removed in practice
LOOK_RADIUS = 0.15         # centre circle, in half-heights of the view
LOOK_OFFSET = (0.55, 0.30) # target start: right and up of centre, in half-heights
LOOK_HOLD_S = 0.3          # target held inside the circle


def look_target(state: dict):
    """A point on the drill tip's depth plane, right of and above the centre of the view."""
    camera = headset_hud.camera_from_state(state)
    if camera is None:
        return None
    c, look, up, fov = camera
    tip = np.array([state.get("px", 0.0), state.get("py", 0.0), state.get("pz", 0.0)], float)
    depth = float((tip - c) @ look)
    if depth <= 1e-6:
        depth = 1.0
    half = depth * np.tan(fov / 2.0)
    right = np.cross(look, up)
    point = c + look * depth + right * (LOOK_OFFSET[0] * half) + up * (LOOK_OFFSET[1] * half)
    return [float(v) for v in point]


class SessionIO:
    """Real files and processes. Tests pass a fake with the same methods."""

    def __init__(self, recorder_cmd=None, runtime=RUNTIME):
        self.runtime = pathlib.Path(runtime)
        self.recorder_cmd = recorder_cmd
        self.recorder = None
        self._recorder_log = None
        # drill_state.json outlives the simulator. Until the new simulator writes
        # it, the file still holds the last run's counters (advance_count etc.).
        # File times run on a coarser clock than time.time(), hence the margin.
        self._since = time.time() - 2.0

    def now(self) -> float:
        return time.time()

    def read_state(self):
        path = self.runtime / "drill_state.json"
        try:
            if path.stat().st_mtime < self._since:
                return None  # left over from an earlier simulator run
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def write_hud(self, hud: dict) -> None:
        self.runtime.mkdir(parents=True, exist_ok=True)
        partial = self.runtime / "study_hud.json.part"
        partial.write_text(json.dumps(hud))
        partial.replace(self.runtime / "study_hud.json")

    def clear_hud(self) -> None:
        (self.runtime / "study_hud.json").unlink(missing_ok=True)

    def set_drill_lock(self, locked: bool) -> None:
        path = self.runtime / "study_drill_lock"
        if locked:
            self.runtime.mkdir(parents=True, exist_ok=True)
            path.write_text("study\n")
        else:
            path.unlink(missing_ok=True)

    def request_restore(self) -> None:
        (self.runtime / "restore.done").unlink(missing_ok=True)
        (self.runtime / "restore.request").write_text("restore\n")

    def restore_done(self) -> bool:
        return (self.runtime / "restore.done").is_file()

    def request_reload(self) -> None:
        (self.runtime / "reload.request").write_text("reload\n")

    def request_burr_reset(self) -> None:
        """The plugin sets the burr to 6 mm cutting (volumetric_drilling.cpp)."""
        self.runtime.mkdir(parents=True, exist_ok=True)
        (self.runtime / "burr_reset.request").write_text("6mm cutting\n")

    def restore_reload_wall(self) -> float:
        """Wall time of the RTX window's last bone reload after a restore, or 0.

        The window's generation counter restarts at each launch, so a file left by
        an earlier run can carry the same number; the wall time cannot repeat.
        """
        try:
            ready = json.loads((self.runtime / "mesh_ready.json").read_text())
        except (OSError, json.JSONDecodeError):
            return 0.0
        if ready.get("reason") != "restore":
            return 0.0
        try:
            return float(ready.get("wall") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def start_recorder(self, output_dir: pathlib.Path) -> None:
        # A request file left by a crashed recorder would look like a ready one.
        (pathlib.Path("/dev/shm") / "fivrs_record_on").unlink(missing_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._recorder_log = open(output_dir.parent / (output_dir.name + "_recorder.log"), "w")
        self.recorder = subprocess.Popen(
            ["bash", "-lc", self.recorder_cmd(output_dir)],
            stdout=self._recorder_log, stderr=subprocess.STDOUT, start_new_session=True,
        )

    def recorder_running(self) -> bool:
        return self.recorder is not None and self.recorder.poll() is None

    def recorder_ready(self) -> bool:
        # The recorder creates the frame request file once its ROS node is up.
        return self.recorder_running() and (pathlib.Path("/dev/shm") / "fivrs_record_on").exists()

    def stop_recorder(self, timeout_s: float = 60.0) -> int | None:
        if self.recorder is None:
            return None
        if self.recorder.poll() is None:
            os.killpg(self.recorder.pid, signal.SIGINT)
            try:
                self.recorder.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.recorder.pid, signal.SIGKILL)
                self.recorder.wait()
        code = self.recorder.returncode
        if self._recorder_log:
            self._recorder_log.close()
        return code


@dataclass
class Progress:
    travel: float = 0.0
    last_tip: tuple | None = None
    device_travel: float = 0.0
    last_device: tuple | None = None
    clutch_s: float = 0.0
    zoom_start: tuple | None = None
    recenter_start: int | None = None
    removed_start: int | None = None
    saw_pause: bool = False
    look_target: list | None = None
    look_s: float = 0.0
    step_done: bool = False          # current step done; waiting for the left pedal
    advance_base: int | None = None  # advance_count when it was done
    done: dict = field(default_factory=dict)


class StudySession:
    """One timed trial, optionally after the tutorial.

    The graphics study (scripts/study/graphics_study.py) runs one of these per
    graphics arm: the first with the tutorial, the second starting at "ready".
    tutorial      False: start at "ready" (the tutorial was done in an earlier arm)
    rtx_reload    False: no RTX window, so the reset is done once AMBF restores the bone
    text          optional headset wording: briefing_title, briefing_lines, ready_title,
                  ready_lines, ready_advance, armed_lines
    """

    def __init__(self, io, study_id: str, recording_dir: pathlib.Path, metadata: dict,
                 trial_seconds: float = TRIAL_SECONDS, tutorial: bool = True,
                 rtx_reload: bool = True, text: dict | None = None):
        self.io = io
        self.tutorial = bool(tutorial)
        self.rtx_reload = bool(rtx_reload)
        self.text = dict(text or {})
        self.ended_by = None
        self.incus_drilling_s = None
        self.study_id = study_id
        self.recording_dir = pathlib.Path(recording_dir)
        self.metadata = dict(metadata)
        self.trial_seconds = float(trial_seconds)
        self.phase = "tutorial"
        self.step = 0
        self.progress = Progress()
        self.events = []
        self._last_tick = None
        self._reset_started = None
        self._recorder_started = None
        self.first_contact_wall = None
        self.drilling_elapsed = 0.0
        self.paused_s = 0.0
        self.finished_wall = None
        self.status = ""
        self.error = None
        self.io.set_drill_lock(False)
        if self.tutorial:
            self._log("tutorial_started")
        else:
            self.phase = "ready"
            self.step = len(TUTORIAL) - 1
            self._log("ready_without_tutorial")

    # ---- proctor actions ----
    def skip_step(self):
        if self.phase == "tutorial":
            self._log("tutorial_step_skipped", step=TUTORIAL[self.step].key)
            self._advance_step()

    def begin_trial(self, by: str = "proctor"):
        """Show the study introduction and prepare the trial behind it.

        From the tutorial (resident's left pedal or the proctor's Begin): lock the
        drill, put the burr back to 6 mm cutting, restore the bone and start the
        recorder while the resident reads. Pressed again during the introduction
        by the proctor, it skips waiting for the resident's pedal.
        """
        if self.phase == "briefing" and by == "proctor":
            self._briefing_ack = True
            self._log("briefing_skipped_by_proctor")
            return
        if self.phase not in ("tutorial", "ready"):
            return
        self._log("trial_requested", by=by, tutorial_complete=self.phase == "ready")
        self._ready_base = None  # a failed reset returns to ready and waits for a new press
        self.phase = "briefing"
        self.error = None
        self.io.set_drill_lock(True)
        self.io.request_burr_reset()
        self._reset_started = self.io.now()
        self._reload_sent = False
        self._reset_ready = False
        self._briefing_ack = False
        self._briefing_base = int((self.io.read_state() or {}).get("advance_count") or 0)
        self._recorder_started = None
        self.io.request_restore()

    def end_early(self):
        """Proctor: stop the trial now (for example, a hardware fault)."""
        if self.phase in ("armed", "running"):
            self.ended_by = "proctor"
            self._log("trial_ended_by_proctor")
            self._finish()

    def incus_identified(self):
        """Proctor: the participant has identified the incus short process. Ends the trial."""
        if self.phase == "running":
            self.ended_by = "incus_identified"
            self.incus_drilling_s = self.drilling_elapsed
            self._log("incus_identified", drilling_s=round(self.drilling_elapsed, 3))
            self._finish()

    # ---- main loop ----
    def tick(self):
        now = self.io.now()
        dt = 0.0 if self._last_tick is None else max(0.0, min(1.0, now - self._last_tick))
        self._last_tick = now
        state = self.io.read_state() or {}
        if self._right_arrow(state):
            self._log("right_arrow", phase=self.phase,
                      step=TUTORIAL[self.step].key if self.phase == "tutorial" else None)
            if self.phase == "tutorial":
                self._advance_step()  # move on whether or not the step is done
            elif self.phase == "ready":
                self.begin_trial(by="resident")
            elif self.phase == "briefing":
                self._briefing_ack = True
        if self.phase == "tutorial":
            self._tutorial(state, dt)
        elif self.phase == "ready":
            self._ready(state)
        elif self.phase == "briefing":
            self._briefing(state, now)
        elif self.phase == "armed":
            self._armed(state, now)
        elif self.phase == "running":
            self._running(state, dt, now)
        self.io.write_hud(self.hud(now))
        return self.phase

    # ---- tutorial ----
    def _tutorial(self, state, dt):
        p = self.progress
        key = TUTORIAL[self.step].key
        tip = (state.get("px"), state.get("py"), state.get("pz"))
        if None not in tip:
            if p.last_tip is not None and key == "drill_move":
                p.travel += math.dist(tip, p.last_tip)
            p.last_tip = tip
        device = (state.get("hx"), state.get("hy"), state.get("hz"))
        clutched = bool(state.get("device_clutch"))
        if None not in device:
            if p.last_device is not None and clutched and key == "reposition":
                p.device_travel += math.dist(device, p.last_device)
            p.last_device = device
        p.clutch_s += dt if clutched and key == "reposition" else 0.0
        zoom = (float(state.get("zoom_in") or 0.0), float(state.get("zoom_out") or 0.0))
        recenter = int(state.get("recenter_count") or 0)
        removed = int(state.get("removed_voxels") or 0)
        if p.zoom_start is None or key != "zoom":
            p.zoom_start = zoom
        if p.recenter_start is None or key != "recenter":
            p.recenter_start = recenter
        if p.removed_start is None or key != "drill":
            p.removed_start = removed
        if key != "pause":
            p.saw_pause = False
        elif state.get("paused"):
            p.saw_pause = True
        if key == "camera":
            if p.look_target is None:
                p.look_target = look_target(state)
            camera = headset_hud.camera_from_state(state)
            spot = headset_hud.project(p.look_target, camera) if (camera and p.look_target) else None
            inside = spot is not None and math.hypot(*spot) <= LOOK_RADIUS
            p.look_s = p.look_s + dt if inside else 0.0
        if not p.step_done:
            has_device = None not in device
            checks = {
                "welcome": True,
                "drill_move": p.travel >= DRILL_TRAVEL_UNITS,
                "reposition": (p.device_travel >= DEVICE_TRAVEL_M if has_device
                               else p.clutch_s >= HOLD_SECONDS),
                "recenter": recenter > p.recenter_start,
                "camera": p.look_s >= LOOK_HOLD_S,
                "zoom": (zoom[0] - p.zoom_start[0] >= ZOOM_EACH_WAY - 1e-6
                         and zoom[1] - p.zoom_start[1] >= ZOOM_EACH_WAY - 1e-6),
                "drill": removed - p.removed_start >= DRILL_VOXELS,
                "pause": p.saw_pause and not state.get("paused"),
            }
            if checks[key]:
                p.step_done = True
                p.advance_base = int(state.get("advance_count") or 0)
                self.progress.done[key] = True
                self._log("tutorial_step_done", step=key)
        elif int(state.get("advance_count") or 0) < p.advance_base:
            p.advance_base = int(state.get("advance_count") or 0)  # the simulator restarted
        elif not state.get("paused") and int(state.get("advance_count") or 0) > p.advance_base:
            self._log("tutorial_step_advanced", step=key)
            self._advance_step()

    def _advance_step(self):
        self.progress.done.setdefault(TUTORIAL[self.step].key, False)
        self.progress.step_done = False
        self.progress.advance_base = None
        self.progress.look_target = None
        self.progress.look_s = 0.0
        self.step += 1
        if self.step >= len(TUTORIAL):
            self.step = len(TUTORIAL) - 1
            self.phase = "ready"
            self._log("tutorial_complete")

    def _right_arrow(self, state) -> bool:
        """A new Right-arrow press (the plugin's skip_count), outside the pause menu."""
        count = int(state.get("skip_count") or 0)
        seen = getattr(self, "_skip_seen", None)
        self._skip_seen = count
        return seen is not None and count > seen and not state.get("paused")

    # ---- end of the tutorial ----
    def _ready(self, state):
        count = int(state.get("advance_count") or 0)
        if getattr(self, "_ready_base", None) is None:
            self._ready_base = count
        if count > self._ready_base and not state.get("paused"):
            self._ready_base = None
            self.begin_trial(by="resident")

    def _briefing(self, state, now):
        if not state.get("paused") and int(state.get("advance_count") or 0) > self._briefing_base:
            if not self._briefing_ack:
                self._log("briefing_read")
            self._briefing_ack = True
        if not self._reset_ready:
            self._resetting(now)
        if self.phase == "briefing" and self._reset_ready and self._briefing_ack:
            state = self.io.read_state() or {}
            self._armed_removed = int(state.get("removed_voxels") or 0)
            self.io.set_drill_lock(False)
            self.phase = "armed"
            self._log("trial_armed")

    # ---- reset and start ----
    def _resetting(self, now):
        if now - self._reset_started > RESET_TIMEOUT_S:
            self.error = "The bone did not reset. Check the simulator and the RTX window."
            self.status = self.error
            self.phase = "ready"
            self.io.set_drill_lock(False)
            self._log("reset_failed")
            return
        if not self._reload_sent:
            if self.io.restore_done():
                if self.rtx_reload:
                    self.io.request_reload()
                self._reload_sent = True
            return
        reloaded = (not self.rtx_reload) or self.io.restore_reload_wall() >= self._reset_started
        if self._recorder_started is None and reloaded:
            self._log("bone_reset")
            self.io.start_recorder(self.recording_dir)
            self._recorder_started = now
            return
        if self._recorder_started is not None:
            if self.io.recorder_ready():
                self._log("recording_started")
                self._reset_ready = True
            elif not self.io.recorder_running() or now - self._recorder_started > RECORDER_START_TIMEOUT_S:
                self.error = "The recorder did not start. See the recorder log next to the recording."
                self.status = self.error
                self.io.stop_recorder(5.0)
                self._recorder_started = None
                self.phase = "ready"
                self.io.set_drill_lock(False)
                self._log("recorder_failed")

    def _armed(self, state, now):
        # Measured from the count at arming: the plugin zeroes it on restore, but
        # the timer must not start from practice drilling if that ever changes.
        if int(state.get("removed_voxels") or 0) > getattr(self, "_armed_removed", 0):
            self.first_contact_wall = now
            self.phase = "running"
            self._log("first_bone_contact")

    def _running(self, state, dt, now):
        if state.get("paused"):
            self.paused_s += dt
        else:
            self.drilling_elapsed += dt
        if self.drilling_elapsed >= self.trial_seconds:
            self.ended_by = "time_up"
            self._log("time_up")
            self._finish()

    def _finish(self):
        self.io.set_drill_lock(True)
        self.phase = "done"
        self.finished_wall = self.io.now()
        code = self.io.stop_recorder()
        self._log("recording_stopped", recorder_exit=code)
        self.write_metadata()

    def remaining(self) -> float:
        return max(0.0, self.trial_seconds - self.drilling_elapsed)

    # ---- outputs ----
    def hud(self, now) -> dict:
        if self.phase == "tutorial":
            step = TUTORIAL[self.step]
            p = self.progress
            numbered = [s for s in TUTORIAL if s.key != "welcome"]
            title = step.title if step.key == "welcome" else \
                f"Step {numbered.index(step) + 1} of {len(numbered)}: {step.title}"
            panel = {
                "title": title,
                "lines": step.lines,
                "pictures": step.pictures,
                "done": p.step_done and step.key != "welcome",
                "advance": ADVANCE_TEXT if p.step_done else None,
                # One row of dots, not a checklist: the panel stays short so it
                # sits above the field instead of over the bone.
                "progress": [bool(p.done.get(s.key)) for s in numbered],
            }
            if step.key == "welcome":
                panel["progress"] = []
            if step.key == "camera" and p.look_target is not None and not p.step_done:
                panel["look"] = {"target": p.look_target, "radius": LOOK_RADIUS}
            return {"panel": panel, "timer": None, "banner": None, "menu_help": MENU_HELP}
        if self.phase == "ready":
            lines = list(self.text.get("ready_lines") or
                         ["You have completed the tutorial. You may keep practicing."])
            if self.error:
                lines = [self.error, "Please tell the proctor."]
            advance = self.text.get("ready_advance") or "Tap the left foot pedal to begin your study session"
            return {"panel": {"title": self.text.get("ready_title") or "Tutorial complete", "lines": lines,
                              "done": True, "advance": None if self.error else advance},
                    "timer": None, "banner": None, "menu_help": MENU_HELP}
        if self.phase == "briefing":
            footer = BRIEFING_FOOTER
            if self._briefing_ack and not self._reset_ready:
                footer = "Thank you. Preparing the bone..."
            return {"panel": {"title": self.text.get("briefing_title") or BRIEFING_TITLE,
                              "lines": list(self.text.get("briefing_lines") or BRIEFING_LINES),
                              "footer": footer, "cover": True},
                    "timer": None, "banner": None}
        timer = {"remaining_s": self.remaining(), "running": False, "written_wall": now}
        if self.phase == "armed":
            return {
                "panel": {"title": "Timed trial",
                          "lines": list(self.text.get("armed_lines") or
                                        ["Begin drilling when ready.",
                                         "The 10-minute timer starts when the drill first removes bone."])},
                "timer": timer, "banner": None,
            }
        if self.phase == "running":
            warning = self.remaining() <= WARNING_SECONDS
            return {
                "panel": {"title": "1 minute remaining", "lines": []} if warning and self.remaining() > WARNING_SECONDS - 8 else None,
                "timer": timer, "banner": None,
            }
        return {"panel": None, "timer": None,
                "banner": "Session complete. Please remove the headset."}

    def write_metadata(self):
        meta = dict(self.metadata)
        meta.update({
            "study_id": self.study_id,
            "trial": {
                "seconds_planned": self.trial_seconds,
                "drilling_seconds": round(self.drilling_elapsed, 3),
                "paused_seconds": round(self.paused_s, 3),
                "first_bone_contact_wall": self.first_contact_wall,
                "finished_wall": self.finished_wall,
                "ended_early": self.drilling_elapsed < self.trial_seconds - 1e-6,
                "ended_by": self.ended_by,
                "incus_identified_drilling_s": (None if self.incus_drilling_s is None
                                                else round(self.incus_drilling_s, 3)),
            },
            "tutorial_run": self.tutorial,
            "tutorial_steps_done": sorted(k for k, v in self.progress.done.items() if v),
            "events": self.events,
        })
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        (self.recording_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def _log(self, name, **detail):
        self.events.append({"event": name, "wall": self.io.now(), **detail})
