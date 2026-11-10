"""What the study needs before Start, and what the launch is doing after it.

Shown on the Study tab (study_mode.py) so the proctor can see why nothing has
happened yet. Everything here only reads files and /sys; nothing is changed.
"""
from __future__ import annotations

import os
import pathlib
import re
import time
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
EYES = pathlib.Path("/dev/shm/fivrs_rtx_eyes")

OK, WAIT, FAIL, INFO = "ok", "wait", "fail", "info"
ICON = {OK: "✓", WAIT: "…", FAIL: "✗", INFO: "•"}
COLOUR = {OK: "#2e9d4f", WAIT: "#c98a00", FAIL: "#d9534f", INFO: "#7a8a99"}


@dataclass
class Check:
    name: str
    state: str          # OK / WAIT / FAIL / INFO
    detail: str
    required: bool = True


def usb_devices():
    """(vendor id, product id, product name) for each USB device."""
    out = []
    for dev in pathlib.Path("/sys/bus/usb/devices").glob("*"):
        try:
            vendor = (dev / "idVendor").read_text().strip()
            product = (dev / "idProduct").read_text().strip()
        except OSError:
            continue
        try:
            name = (dev / "product").read_text().strip()
        except OSError:
            name = ""
        out.append((vendor, product, name))
    return out


def _nvidia_flatpak_problem():
    version_file = pathlib.Path("/proc/driver/nvidia/version")
    if not version_file.is_file():
        return None
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", version_file.read_text(errors="replace"))
    if not match:
        return None
    ext_id = "org.freedesktop.Platform.GL.nvidia-" + match.group(1).replace(".", "-")
    roots = (pathlib.Path("/var/lib/flatpak/runtime") / ext_id,
             pathlib.Path.home() / ".local/share/flatpak/runtime" / ext_id)
    if any(path.exists() for path in roots):
        return None
    return f"Run: flatpak install -y flathub {ext_id}, then restart the WiVRn dashboard"


def preflight(footpedal_path: str = "", sim_running: bool = False, headset: bool = True):
    """Checks before Start. Required ones must pass for the study launch to work."""
    checks = []
    usb = usb_devices()
    if not headset:
        checks.append(Check("Quest headset", INFO, "not used: desktop view (development)", required=False))

    checks.extend(_headset_checks() if headset else [])
    return _device_checks(checks, usb, footpedal_path, sim_running)


WIVRN_LOG_DIRS = (
    pathlib.Path.home() / ".var/app/io.github.wivrn.wivrn/.local/state/wivrn/wivrn-dashboard",  # flatpak
    pathlib.Path.home() / ".local/state/wivrn/wivrn-dashboard",
)
# Lines the WiVRn server writes when the headset joins or leaves. "Client connected"
# alone is the headset; apps show up as "[client_connected] Client N connected".
_HEADSET_JOINED = re.compile(r"\] Client connected\s*$")
_HEADSET_LEFT = re.compile(r"Network exception|Server exited|Waiting for new connection")


def wivrn_log():
    """The WiVRn dashboard's newest server log, or None."""
    logs = [log for folder in WIVRN_LOG_DIRS for log in folder.glob("server_logs_*.txt")]
    return max(logs, key=lambda log: log.stat().st_mtime, default=None)


def _tail(path: pathlib.Path, size: int = 65536) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - size))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def headset_connected(log=None):
    """True or False from the WiVRn server log, or None when there is no log to read."""
    log = wivrn_log() if log is None else log
    if log is None:
        return None
    state = None
    for line in _tail(log).splitlines():
        if _HEADSET_JOINED.search(line):
            state = True
        elif _HEADSET_LEFT.search(line):
            state = False
    return state


def _headset_checks():
    checks = []
    ipc = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "wivrn" / "comp_ipc"
    checks.append(Check("WiVRn server", OK if ipc.exists() else FAIL,
                        "running" if ipc.exists() else "not running: open the WiVRn dashboard on this computer"))
    runtime = pathlib.Path.home() / ".config/openxr/1/active_runtime.json"
    connected = headset_connected() if runtime.is_file() else False
    if connected is None:
        checks.append(Check("Quest headset", INFO, "WiVRn log not found; check the Quest shows it is connected",
                            required=False))
    else:
        checks.append(Check("Quest headset", OK if connected else FAIL,
                            "connected to WiVRn" if connected else
                            "not connected: put on the Quest, open WiVRn and connect to this computer"))
    gpu = _nvidia_flatpak_problem()
    checks.append(Check("WiVRn on the NVIDIA GPU", FAIL if gpu else OK, gpu or "ready"))
    return checks


FEEDBACK = REPO / "feedback_system"
STUDY_SPECIMEN = "RT143"  # Anatomy A


def edt_folder(specimen: str = STUDY_SPECIMEN) -> pathlib.Path:
    """Where the feedback analysis looks for this anatomy's distance grids."""
    import json

    root = "data/edt"
    try:
        root = json.loads((FEEDBACK / "config.local.json").read_text()).get("edt_root") or root
    except (OSError, ValueError):
        pass
    root = pathlib.Path(root).expanduser()
    if not root.is_absolute():
        root = FEEDBACK / root
    return root / f"edt_grids_{specimen}_256"


def edt_problem(specimen: str = STUDY_SPECIMEN):
    """None when the grids are in place, else what to do."""
    folder = edt_folder(specimen)
    if folder.is_dir() and any(folder.glob("*.edt")):
        return None
    return (f"missing: download the {folder.name} folder from the collaborator's SharePoint "
            f"and put it in {folder.parent}")


def _device_checks(checks, usb, footpedal_path, sim_running):

    touches = [d for d in usb if d[0] == "2988"]
    checks.append(Check("Haptic device (Touch)", OK if touches else FAIL,
                        "found on USB" if touches else "not found: check the USB cable and power"))
    pedal = footpedal_path and pathlib.Path(footpedal_path).exists()
    pedal = pedal or any(d[0] == "3553" for d in usb)
    checks.append(Check("Foot pedal", OK if pedal else FAIL,
                        "found on USB" if pedal else "not found: plug in the foot pedal"))
    scope = any("sayo" in d[2].lower() or d[0] == "8089" for d in usb)
    checks.append(Check("Microscope handle button", OK if scope else INFO,
                        "found on USB" if scope else
                        "not detected by name; check that pressing it moves the view", required=False))

    venv = REPO / "middle_ear_newton" / ".venv" / "bin" / "python"
    checks.append(Check("RTX renderer", OK if venv.is_file() else FAIL,
                        "installed" if venv.is_file() else
                        "missing; run: bash scripts/setup_rtx_env.sh"))
    edt = edt_problem()
    checks.append(Check("Feedback distance maps (EDT)", FAIL if edt else OK, edt or "in place",
                        required=False))
    checks.append(Check("Simulator closed", FAIL if sim_running else OK,
                        "close the running simulation first" if sim_running else "ready"))
    return checks


def _recent(path: pathlib.Path, seconds: float) -> bool:
    try:
        return time.time() - path.stat().st_mtime < seconds
    except OSError:
        return False


def _view_progress(log: pathlib.Path):
    """(state, detail) for the RTX window from the end of its log."""
    try:
        with open(log, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 20000))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return WAIT, "starting"
    # Only this launch: the window prints this line first.
    start = text.rfind("Initializing RTX View")
    if start >= 0:
        text = text[start:]
    if "Traceback" in text or "RTX view failed" in text:
        return FAIL, f"stopped with an error: {RESTART_SIM} (details in {log})"
    if "RTX window closed" in text:
        return FAIL, f"its window was closed: {RESTART_SIM}"
    if "RTX bone ready" in text:
        return OK, "bone loaded, window shown"
    percents = re.findall(r"Loading bone (\d+)%", text)
    if percents:
        return WAIT, f"loading the bone {percents[-1]}%"
    if "RTX view ready" in text:
        return WAIT, "loading the bone"
    return WAIT, "starting the renderer (about 20-40 s)"


# How long a heartbeat file may go unwritten before the launch counts as stuck.
# AMBF writes status.json every 0.5 s from its graphics loop; the RTX window
# writes summary.json every second.
AMBF_STALL_S = 5.0
RTX_STALL_S = 6.0
# AMBF's headset plugin connects to WiVRn from a background thread (AMBF keeps
# drawing meanwhile); WiVRn usually lets it in within 2-6 s.
HEADSET_SLOW_S = 15.0
HEADSET_LINK_S = 45.0
RESTART_WIVRN = ("on the Quest, quit WiVRn and open it again, then connect to this computer; "
                 "the simulator carries on once it connects")
RESTART_SIM = "Close Simulation and Start again"
RESTORE_AMBF = ("if the AMBF window is minimized, click it in the dock to show it; "
                "otherwise Close Simulation and Start again")


def _age(path: pathlib.Path, since_wall: float):
    """Seconds since the file was last written, or None if not written since since_wall."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return None if mtime < since_wall else time.time() - mtime


def _headset_link(ambf_text: str, elapsed: float):
    """(state, detail) for AMBF's OpenXR session, from AMBF's log."""
    if "OpenXR eyepiece failed to start" in ambf_text:
        return FAIL, f"the simulator could not open the headset: {RESTART_WIVRN}, then {RESTART_SIM}"
    if "OpenXR eyepiece session READY" in ambf_text:
        return OK, "connected"
    if "OpenXR eyepiece session created" in ambf_text:
        return WAIT, "connected; the image starts when the Quest is worn"
    if elapsed < HEADSET_SLOW_S:
        return WAIT, "connecting to WiVRn"
    if elapsed < HEADSET_LINK_S:
        return WAIT, (f"WiVRn is slow to let the simulator in ({elapsed:.0f} s); "
                      f"if it is not in by {HEADSET_LINK_S:.0f} s, {RESTART_WIVRN.split(';')[0]}")
    return FAIL, f"WiVRn has not let the simulator in after {elapsed:.0f} s: {RESTART_WIVRN}"


def launch_status(ambf_state: str, rtx_running: bool, started_wall: float, headset: bool = True):
    """Rows for the launch in progress: simulator, renderer, headset image, live input."""
    rows = []
    elapsed = time.time() - started_wall
    ambf_text = _tail(RUNTIME / "ambf.log", 200000)
    gfx_age = _age(RUNTIME / "status.json", started_wall)
    if not headset:
        link = (OK, "")
    elif gfx_age is None:
        # The headset plugin connects from AMBF's graphics loop.
        link = (WAIT, "waiting for the simulator's first frame")
    else:
        link = _headset_link(ambf_text, elapsed)
    if ambf_state == "running":
        if gfx_age is not None and gfx_age <= AMBF_STALL_S:
            rows.append(Check("Simulator (AMBF)", OK, "running"))
        elif gfx_age is None and elapsed < HEADSET_SLOW_S:
            rows.append(Check("Simulator (AMBF)", WAIT, "starting"))
        elif gfx_age is None:
            rows.append(Check("Simulator (AMBF)", FAIL,
                              f"has not drawn a frame in {elapsed:.0f} s: {RESTORE_AMBF}"))
        else:
            stalled = elapsed if gfx_age is None else gfx_age
            rows.append(Check("Simulator (AMBF)", FAIL,
                              f"stopped drawing {stalled:.0f} s ago: {RESTORE_AMBF}"))
    elif ambf_state == "starting":
        rows.append(Check("Simulator (AMBF)", WAIT, "starting"))
    else:
        rows.append(Check("Simulator (AMBF)", FAIL, f"not running: see {RUNTIME / 'ambf.log'}"))
    if headset:
        rows.append(Check("Headset link (WiVRn)", *link))
    view_log = RUNTIME / "view.log"
    if not rtx_running:
        rows.append(Check("RTX view", FAIL, f"not running: {RESTART_SIM} (details in {view_log})"))
    elif not _recent(view_log, elapsed + 1):
        rows.append(Check("RTX view", WAIT, "starting"))
    else:
        state, detail = _view_progress(view_log)
        rtx_age = _age(RUNTIME / "summary.json", started_wall)
        if state == OK and (rtx_age is None or rtx_age > RTX_STALL_S):
            stalled = elapsed if rtx_age is None else rtx_age
            state, detail = FAIL, f"stopped responding {stalled:.0f} s ago: {RESTART_SIM} (details in {view_log})"
        rows.append(Check("RTX view", state, detail))
    state_live = _recent(RUNTIME / "drill_state.json", 3.0)
    rows.append(Check("Drill and stylus input", OK if state_live else WAIT,
                      "live" if state_live else "waiting for the simulator's first frames"))
    if headset:
        eyes = _recent(EYES, 3.0)
        rows.append(Check("Image in the headset", OK if eyes else WAIT,
                          "streaming" if eyes else "waiting for the RTX view"))
    return rows


PROBLEM_LOG = RUNTIME / "launch_problems.log"


def thread_snapshot(pid: int) -> str:
    """Where each thread of a process is waiting (state and kernel wait channel).

    Only reads /proc files the owner may read, so it works without ptrace.
    """
    lines = []
    for task in sorted(pathlib.Path(f"/proc/{pid}/task").glob("*"), key=lambda p: int(p.name)):
        try:
            name = (task / "comm").read_text().strip()
            wchan = (task / "wchan").read_text().strip() or "-"
            state = (task / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        lines.append(f"  {task.name:>8} {state} {wchan:<28} {name}")
    return "\n".join(lines)


def log_problem(text: str, pids=()) -> None:
    """Append a launch problem to PROBLEM_LOG, with where each process is waiting."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"[{stamp}] {text}"]
    for label, pid in pids:
        if pid:
            parts.append(f" {label} (pid {pid}) threads:\n{thread_snapshot(pid)}")
    try:
        RUNTIME.mkdir(parents=True, exist_ok=True)
        with open(PROBLEM_LOG, "a", encoding="utf-8") as handle:
            handle.write("\n".join(parts) + "\n")
    except OSError:
        pass


def waiting_on(rows) -> str:
    """One line: the first thing not ready yet, or that everything is ready."""
    for row in rows:
        if row.state == FAIL and row.required:
            return f"Problem: {row.name}: {row.detail}"
    for row in rows:
        if row.state == WAIT:
            return f"Waiting on: {row.name} ({row.detail})"
    return "Everything is ready."
