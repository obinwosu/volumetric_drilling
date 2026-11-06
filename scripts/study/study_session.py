"""Study session controller: tutorial, bone reset, timed trial, end.

Plain Python, driven by ``tick()`` about ten times a second from the launcher.
It reads the simulator's /tmp/fivrs-rtx/drill_state.json, writes the headset
text (study_hud.json, drawn by scripts/rtx_view/headset_hud.py), locks the
drill (study_drill_lock), resets the bone, and starts and stops the recorder.
Everything that touches files or processes goes through ``SessionIO`` so tests
can replace it.

Flow:
    tutorial  -> each step ticks off from the simulator's live state
    ready     -> "Tell the proctor"; the proctor presses Begin trial
    resetting -> drill locked, bone restored in AMBF and reloaded in RTX
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
import time
from dataclasses import dataclass, field

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
TRIAL_SECONDS = 600.0
WARNING_SECONDS = 60.0
RESET_TIMEOUT_S = 45.0
RECORDER_START_TIMEOUT_S = 30.0
PAUSE_FOOTER = "Double-tap Stylus 1 any time for the pause menu."


@dataclass
class Step:
    key: str
    title: str
    lines: list
    checklist: str


# Wording shown in the headset. Each step is checked from drill_state.json in _progress().
TUTORIAL = [
    Step("drill_move", "Move the drill",
         ["Move the stylus to move the drill around the field."], "Move the drill"),
    Step("camera", "Move the microscope",
         ["Hold the microscope button and move your head to move the view.",
          "Release the button to fix the view."], "Move the microscope"),
    Step("zoom", "Zoom",
         ["Hold the left pedal and press Stylus 2 to zoom in, Stylus 1 to zoom out."], "Zoom in and out"),
    Step("device_clutch", "Drill clutch",
         ["Hold Stylus 2 and move the stylus to a comfortable spot without moving the drill.",
          "Release Stylus 2 to continue."], "Drill clutch"),
    Step("recenter", "Recenter the drill",
         ["Double-tap Stylus 2 to bring the drill back into view."], "Recenter the drill"),
    Step("orbit", "Orbit around the burr",
         ["Hold the right pedal and Stylus 2, then move the stylus to orbit the view around the burr."],
         "Orbit around the burr"),
    Step("drill", "Drill",
         ["Hold the middle pedal to run the drill and remove some bone.",
          "This practice bone is restored before the timed trial."], "Drill some bone"),
    Step("pause", "Pause menu and burrs",
         ["Double-tap Stylus 1 to open the pause menu.",
          "Right pedal moves down to Burr size, left pedal opens it and picks a size,",
          "middle pedal goes back. Double-tap Stylus 1 to resume."],
         "Change the burr in the pause menu"),
]

# Thresholds for each tutorial check.
DRILL_TRAVEL_UNITS = 0.5   # simulator units of tip travel (about 25 mm)
HOLD_SECONDS = 1.0         # clutch / orbit held long enough to have been tried
ZOOM_DOUBLINGS = 1.0       # total zoom in plus out, in doublings of magnification
DRILL_VOXELS = 200         # voxels removed in practice


class SessionIO:
    """Real files and processes. Tests pass a fake with the same methods."""

    def __init__(self, recorder_cmd=None, runtime=RUNTIME):
        self.runtime = pathlib.Path(runtime)
        self.recorder_cmd = recorder_cmd
        self.recorder = None
        self._recorder_log = None

    def now(self) -> float:
        return time.time()

    def read_state(self):
        try:
            return json.loads((self.runtime / "drill_state.json").read_text())
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
    clutch_s: float = 0.0
    device_clutch_s: float = 0.0
    orbit_s: float = 0.0
    zoom_start: float | None = None
    recenter_start: int | None = None
    removed_start: int | None = None
    saw_pause: bool = False
    burr_start: tuple | None = None
    done: dict = field(default_factory=dict)


class StudySession:
    def __init__(self, io, study_id: str, recording_dir: pathlib.Path, metadata: dict,
                 trial_seconds: float = TRIAL_SECONDS):
        self.io = io
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
        self._log("tutorial_started")

    # ---- proctor actions ----
    def skip_step(self):
        if self.phase == "tutorial":
            self._log("tutorial_step_skipped", step=TUTORIAL[self.step].key)
            self._advance_step()

    def begin_trial(self):
        """Proctor: the resident is ready. Reset the bone, then arm the trial."""
        if self.phase not in ("tutorial", "ready"):
            return
        self._log("trial_requested", tutorial_complete=self.phase == "ready")
        self.phase = "resetting"
        self.io.set_drill_lock(True)
        self._reset_started = self.io.now()
        self._reload_sent = False
        self.io.request_restore()

    def end_early(self):
        """Proctor: stop the trial now (for example, a hardware fault)."""
        if self.phase in ("armed", "running"):
            self._log("trial_ended_by_proctor")
            self._finish()

    # ---- main loop ----
    def tick(self):
        now = self.io.now()
        dt = 0.0 if self._last_tick is None else max(0.0, min(1.0, now - self._last_tick))
        self._last_tick = now
        state = self.io.read_state() or {}
        if self.phase == "tutorial":
            self._tutorial(state, dt)
        elif self.phase == "resetting":
            self._resetting(now)
        elif self.phase == "armed":
            self._armed(state, now)
        elif self.phase == "running":
            self._running(state, dt, now)
        self.io.write_hud(self.hud(now))
        return self.phase

    # ---- tutorial ----
    def _tutorial(self, state, dt):
        p = self.progress
        tip = (state.get("px"), state.get("py"), state.get("pz"))
        if None not in tip:
            if p.last_tip is not None:
                p.travel += math.dist(tip, p.last_tip)
            p.last_tip = tip
        p.clutch_s += dt if state.get("cam_clutch") else 0.0
        p.device_clutch_s += dt if state.get("device_clutch") else 0.0
        p.orbit_s += dt if state.get("orbit") else 0.0
        key = TUTORIAL[self.step].key
        zoom = float(state.get("zoom_travel") or 0.0)
        recenter = int(state.get("recenter_count") or 0)
        removed = int(state.get("removed_voxels") or 0)
        if p.zoom_start is None or key != "zoom":
            p.zoom_start = zoom
        if p.recenter_start is None or key != "recenter":
            p.recenter_start = recenter
        if p.removed_start is None or key != "drill":
            p.removed_start = removed
        burr = (state.get("drill"), state.get("style"))
        if key != "pause" or p.burr_start is None:
            p.burr_start = burr
        if key != "pause":
            p.saw_pause = False
        done = {
            "drill_move": p.travel >= DRILL_TRAVEL_UNITS,
            "camera": p.clutch_s >= HOLD_SECONDS,
            "zoom": zoom - p.zoom_start >= ZOOM_DOUBLINGS,
            "device_clutch": p.device_clutch_s >= HOLD_SECONDS,
            "recenter": recenter > p.recenter_start,
            "orbit": p.orbit_s >= HOLD_SECONDS,
            "drill": removed - p.removed_start >= DRILL_VOXELS,
            "pause": p.saw_pause and burr != p.burr_start and not state.get("paused"),
        }
        if key == "pause" and state.get("paused"):
            p.saw_pause = True
        if done[key]:
            self._log("tutorial_step_done", step=key)
            self._advance_step()

    def _advance_step(self):
        self.progress.done[TUTORIAL[self.step].key] = True
        self.step += 1
        if self.step >= len(TUTORIAL):
            self.step = len(TUTORIAL) - 1
            self.phase = "ready"
            self._log("tutorial_complete")

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
                self.io.request_reload()
                self._reload_sent = True
            return
        if self._recorder_started is None and self.io.restore_reload_wall() >= self._reset_started:
            self._log("bone_reset")
            self.io.start_recorder(self.recording_dir)
            self._recorder_started = now
            return
        if self._recorder_started is not None:
            if self.io.recorder_ready():
                self._log("recording_started")
                state = self.io.read_state() or {}
                self._armed_removed = int(state.get("removed_voxels") or 0)
                self.io.set_drill_lock(False)
                self.phase = "armed"
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
            return {
                "panel": {
                    "title": f"Tutorial {self.step + 1} of {len(TUTORIAL)}: {step.title}",
                    "lines": step.lines,
                    # One row of dots, not a checklist: the panel stays short so it
                    # sits above the field instead of over the bone.
                    "progress": [bool(self.progress.done.get(s.key)) for s in TUTORIAL],
                },
                "timer": None, "banner": None,
            }
        if self.phase == "ready":
            lines = ["You have practiced every control.", "Tell the proctor when you are ready to begin."]
            if self.error:
                lines = [self.error]
            return {"panel": {"title": "Tutorial complete", "lines": lines, "footer": PAUSE_FOOTER},
                    "timer": None, "banner": None}
        if self.phase == "resetting":
            return {"panel": None, "timer": None, "banner": "Restoring the bone for the timed trial..."}
        timer = {"remaining_s": self.remaining(), "running": False, "written_wall": now}
        if self.phase == "armed":
            return {
                "panel": {"title": "Timed trial",
                          "lines": ["Begin drilling when ready.",
                                    "The 10-minute timer starts when the drill first removes bone."]},
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
            },
            "tutorial_steps_done": sorted(k for k, v in self.progress.done.items() if v),
            "events": self.events,
        })
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        (self.recording_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def _log(self, name, **detail):
        self.events.append({"event": name, "wall": self.io.now(), **detail})
