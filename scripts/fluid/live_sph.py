#!/usr/bin/env python3
"""Pedal-gated Maelstrom DFSPH against the frozen bone, dumped for the RTX view.

This process does not draw bone and does not run in the haptic loop. It
rewrites inlet seeds from drill_state.json, steps consoleParticles, and
leaves positions in /tmp/fivrs-rtx/sph_positions.bin.

Gravity is the right transtemporal screen-down, not world -Z. Hold Space
(or the drill pedal) to emit. The bone collider is the frozen VDB.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluid.frozen_bone import write_domain_obj
from fluid.irrigation_bench import IRRIGATION_PATH, load_irrigation
from fluid.moving_inlet import (
    DEFAULT_STATE,
    attach_inlet_config,
    inlet_from_drill,
    inlet_from_pose,
    write_inlet_vdb,
    write_state,
)
from fluid.occupancy import drill_pose, read_drill_state
from fluid.outlet import MM_TO_SIM
from fluid.sph_dump import DEFAULT_PATH as DUMP_PATH

RUNTIME = Path("/tmp/fivrs-rtx")
BONE_DIR = RUNTIME / "frozen_bone"
DRILL_HOLD = RUNTIME / "drill_hold"
STOP_PATH = RUNTIME / "sph.stop"
PID_PATH = RUNTIME / "sph.pid"
CONSOLE = Path.home() / "opt" / "openMaelstrom" / "build" / "bin" / "consoleParticles"
CUDA_BIN = Path.home() / "opt" / "cuda" / "bin"
CUDA_LIB = Path.home() / "opt" / "cuda" / "lib64"
VFX_LIB = Path.home() / "opt" / "vfx" / "lib"
VFX_LIB64 = Path.home() / "opt" / "vfx" / "lib" / "x86_64-linux-gnu"
REPO = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO / "middle_ear_newton" / ".venv" / "bin" / "python"
RTX_SCRIPT = REPO / "scripts" / "rtx_view" / "fivrs_rtx_view.py"
DEFAULT_ADF = REPO / "ADF" / "volume_256_RT143.yaml"
G_SIM = 9.81 * 1000.0 * MM_TO_SIM
# 5 mm to the right of the shaft, beside the tip. LIVE_JET_MM is the nozzle
# diameter. These rates apply whenever the drill is emitting. Aim does not
# change them. 80 cc/s through the 1 mm disk is about 100 m/s (Q = A U).
LIVE_PARTICLE_MM = 0.5
LIVE_JET_MM = 1.0
LIVE_OFFSET_MM = 5.0
LIVE_PROXIMAL_MM = 0.0
LIVE_CC_PER_S = 80.0
LIVE_NUMPTCLS = 96000
DOMAIN_PAD_MM = 80.0
# Right transtemporal camera, ADF/world/canonical_views.yaml. Column 2 of
# this quaternion's matrix is screen-up there.
RIGHT_TRANSTEMPORAL_QUAT = (-0.380217, 0.0698274, 0.131694, 0.912807)


def _quat_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def screen_down(quat=RIGHT_TRANSTEMPORAL_QUAT) -> np.ndarray:
    """Unit vector toward the bottom of the right transtemporal view."""
    up = _quat_matrix(*quat)[:, 2]
    norm = float(np.linalg.norm(up))
    if norm < 1e-12:
        return np.array((0.0, 0.0, -1.0), dtype=np.float64)
    return -up / norm


def gravity_vector():
    down = screen_down()
    return (float(down[0] * G_SIM), float(down[1] * G_SIM), float(down[2] * G_SIM), 0.0)


def live_irrigation() -> dict:
    """Nozzle and flow saved from the RTX irrigation tuner."""
    settings = load_irrigation(IRRIGATION_PATH)
    settings.setdefault("particle_mm", LIVE_PARTICLE_MM)
    settings.setdefault("jet_mm", LIVE_JET_MM)
    settings.setdefault("cc_per_s", LIVE_CC_PER_S)
    settings.setdefault("offset_mm", LIVE_OFFSET_MM)
    settings.setdefault("proximal_mm", LIVE_PROXIMAL_MM)
    return settings


def domain_bounds(origin, world_max, extras, pad_mm: float = DOMAIN_PAD_MM):
    pad = float(pad_mm) * MM_TO_SIM
    pts = [np.asarray(origin, dtype=np.float64), np.asarray(world_max, dtype=np.float64)]
    pts.extend(np.asarray(p, dtype=np.float64).reshape(3) for p in extras)
    stack = np.stack(pts, axis=0)
    return stack.min(axis=0) - pad, stack.max(axis=0) + pad


def prepare_live_config(
    bone_dir: Path = BONE_DIR,
    outlet=None,
    particle_mm: float = LIVE_PARTICLE_MM,
    jet_mm: float = LIVE_JET_MM,
    numptcls: int = LIVE_NUMPTCLS,
    pad_mm: float = DOMAIN_PAD_MM,
) -> dict:
    bone_dir = Path(bone_dir)
    report = json.loads((bone_dir / "report.json").read_text())
    extras = []
    if outlet is not None:
        extras.append(outlet)
    lo, hi = domain_bounds(report["origin"], report["world_max"], extras, pad_mm=pad_mm)
    write_domain_obj(bone_dir / "domain.obj", lo, hi, pad=0.0)
    radius = 0.5 * float(particle_mm) * MM_TO_SIM
    gx, gy, gz, gw = gravity_vector()
    config_path = bone_dir / "config.json"
    payload = json.loads(config_path.read_text()) if config_path.is_file() else {}
    modules = payload.setdefault("modules", {})
    # Packed disk so DFSPH has neighbors. Empty-neighborhood divisions are
    # guarded in SPH/DFSPH/dfsph.cu.
    modules["pressure"] = "DFSPH"
    modules["volumeBoundary"] = True
    modules["volumeInlets"] = True
    # This Maelstrom build only compiles the compactMLM cell launch.
    # hashed_cell (the default) makes Inlet::emit a no-op.
    modules["resorting"] = "compactMLM"
    modules["neighborhood"] = "constrained"
    modules["density"] = "standard"
    modules["xsph"] = False
    modules["rayTracing"] = False
    modules["anisotropicSurface"] = False
    modules["surfaceDetection"] = False
    modules["error_checking"] = True
    payload.setdefault("adaptive", {})["delay"] = 0.0
    payload.setdefault("boundary_volumes", {}).setdefault("volume1", {})
    payload["boundary_volumes"]["volume1"]["file"] = "bone_surface.vdb"
    payload["boundary_volumes"]["volume1"]["kind"] = 2
    payload["particle_settings"] = {
        "radius": float(radius),
        "density": 998,
    }
    payload["simulation_settings"] = {
        "boundaryObject": "domain.obj",
        "domainWalls": "x+-y+-z+-",
        "numptcls": int(numptcls),
        "auto_boundary": True,
        # 80 cc/s through the 1 mm disk is about 100 m/s. The march stays under the bone band.
        "timestep_min": 0.000002,
        "timestep_max": 0.00002,
        "neighborlimit": 120,
        "external_force": f"{gx} {gy} {gz} {gw}",
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "origin": [float(v) for v in lo],
        "world_max": [float(v) for v in hi],
        "radius": float(radius),
        "particle_mm": float(particle_mm),
        "jet_mm": float(jet_mm),
        "numptcls": int(numptcls),
        "gravity": [gx, gy, gz, gw],
        "config": str(config_path),
    }


def sync_inlet(
    drill_path: Path,
    bone_dir: Path = BONE_DIR,
    state_path: Path = DEFAULT_STATE,
    force_emit: bool = False,
    particle_mm: float = LIVE_PARTICLE_MM,
    jet_mm: float = LIVE_JET_MM,
    cc_per_s: float = LIVE_CC_PER_S,
):
    settings = live_irrigation()
    inlet = inlet_from_drill(
        drill_path,
        force_emit=force_emit,
        particle_mm=particle_mm,
        jet_mm=jet_mm,
        cc_per_s=cc_per_s,
        offset_mm=float(settings["offset_mm"]),
        proximal_mm=float(settings["proximal_mm"]),
        disk_area=True,
    )
    write_state(state_path, inlet)
    write_inlet_vdb(bone_dir / "inlet_surface.vdb", inlet)
    attach_inlet_config(bone_dir / "config.json", inlet, vdb_name="inlet_surface.vdb")
    return inlet


def console_env() -> dict:
    env = os.environ.copy()
    path_parts = [str(CUDA_BIN), env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(p for p in path_parts if p)
    libs = [str(CUDA_LIB), str(VFX_LIB), str(VFX_LIB64), env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in libs if p)
    env["FIVRS_INLET_STATE"] = str(DEFAULT_STATE)
    env["FIVRS_SPH_DUMP"] = str(DUMP_PATH)
    env["FIVRS_SPH_STOP"] = str(STOP_PATH)
    return env


def start_console(bone_dir: Path = BONE_DIR, log_path: Path | None = None) -> subprocess.Popen:
    if not CONSOLE.is_file():
        raise FileNotFoundError(f"consoleParticles missing: {CONSOLE}")
    STOP_PATH.unlink(missing_ok=True)
    log_path = Path(log_path or RUNTIME / "sph.log")
    handle = log_path.open("w")
    proc = subprocess.Popen(
        [str(CONSOLE), "--config", str((bone_dir / "config.json").resolve())],
        cwd=str(Path.home() / "opt" / "openMaelstrom" / "build"),
        env=console_env(),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    PID_PATH.write_text(f"{proc.pid}\n")
    return proc


def stop_console(proc: subprocess.Popen | None = None) -> None:
    STOP_PATH.write_text("stop\n")
    pid = None
    if proc is not None:
        pid = proc.pid
    elif PID_PATH.is_file():
        try:
            pid = int(PID_PATH.read_text().strip())
        except ValueError:
            pid = None
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.time() + 4.0
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.05)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if proc is not None:
        proc.wait(timeout=2)
    PID_PATH.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _comm_pids(name: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-x", name], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


def _script_pids(script: str) -> list[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        if "cursorsandbox" in args:
            continue
        if script not in args:
            continue
        if pid_s.isdigit():
            pids.append(int(pid_s))
    return pids


def ambf_running() -> bool:
    return bool(_comm_pids("ambf_simulator"))


def rtx_pids() -> list[int]:
    return [pid for pid in _script_pids("fivrs_rtx_view.py") if pid != os.getpid()]


def start_ambf(anatomy_adf: Path = DEFAULT_ADF) -> subprocess.Popen:
    anatomy_adf = Path(anatomy_adf).resolve()
    launch = REPO / "launch.yaml"
    log_path = RUNTIME / "ambf_live.log"
    command = (
        "set +u && "
        "source /opt/ros/jazzy/setup.bash && "
        "source \"$HOME/ros_ambf_ws/install/setup.bash\" && "
        f"source \"{REPO / 'scripts' / 'setup_haptic_env.sh'}\" && "
        "export DISPLAY=\"${DISPLAY:-:1}\" && "
        "export __GLX_VENDOR_LIBRARY_NAME=\"${__GLX_VENDOR_LIBRARY_NAME:-nvidia}\" && "
        f"exec ambf_simulator --launch_file \"{launch}\" -l 0,7 "
        f"-a \"{anatomy_adf}\" --nt 2 --rtx_view 1 --anatomy_adf \"{anatomy_adf}\""
    )
    handle = log_path.open("w")
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(REPO),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    (RUNTIME / "ambf.pid").write_text(f"{proc.pid}\n")
    return proc


def start_rtx(anatomy_adf: Path = DEFAULT_ADF) -> subprocess.Popen:
    anatomy_adf = Path(anatomy_adf).resolve()
    for pid in rtx_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 3.0
    while time.time() < deadline and rtx_pids():
        time.sleep(0.1)
    for pid in rtx_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    log_path = RUNTIME / "rtx_live.log"
    command = (
        "set +u && "
        "source /opt/ros/jazzy/setup.bash && "
        "source \"$HOME/ros_ambf_ws/install/setup.bash\" && "
        "export DISPLAY=\"${DISPLAY:-:1}\" && "
        "export __GLX_VENDOR_LIBRARY_NAME=\"${__GLX_VENDOR_LIBRARY_NAME:-nvidia}\" && "
        "export VK_DRIVER_FILES=\"${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}\" && "
        f"exec \"{VENV_PYTHON}\" \"{RTX_SCRIPT}\" --anatomy-adf \"{anatomy_adf}\""
    )
    handle = log_path.open("w")
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(REPO),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    (RUNTIME / "rtx.pid").write_text(f"{proc.pid}\n")
    return proc


def read_rtx_fps() -> dict | None:
    path = RUNTIME / "summary.json"
    try:
        if time.time() - path.stat().st_mtime > 3.0:
            return None
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload


def wait_for_rtx(timeout: float = 120.0) -> dict:
    deadline = time.time() + float(timeout)
    last = None
    while time.time() < deadline:
        last = read_rtx_fps()
        if last is not None and float(last.get("rtx_fps") or 0.0) > 1.0:
            return last
        time.sleep(0.5)
    raise SystemExit("RTX summary.json never became live")


def wait_for_dump(timeout: float = 180.0) -> None:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if DUMP_PATH.is_file() and DUMP_PATH.stat().st_size >= 24:
            return
        time.sleep(0.2)
    raise SystemExit(f"SPH dump never appeared: {DUMP_PATH}")


def sample_fps(seconds: float, label: str) -> dict:
    samples = []
    sph_counts = []
    deadline = time.time() + float(seconds)
    while time.time() < deadline:
        summary = read_rtx_fps()
        if summary is not None and summary.get("rtx_fps"):
            samples.append(float(summary["rtx_fps"]))
            sph = summary.get("sph") or {}
            if sph.get("count") is not None:
                sph_counts.append(int(sph["count"]))
        time.sleep(0.25)
    if not samples:
        return {"label": label, "rtx_fps": None, "n": 0, "sph_count": 0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "label": label,
        "rtx_fps": float(arr.mean()),
        "rtx_fps_min": float(arr.min()),
        "rtx_fps_max": float(arr.max()),
        "n": int(arr.size),
        "sph_count": int(np.median(sph_counts)) if sph_counts else 0,
    }


class InletPoller(threading.Thread):
    """Rewrite the inlet from the drill pose on every tick. Nothing is held."""

    def __init__(self, drill_path: Path, particle_mm: float, jet_mm: float, hz: float = 120.0):
        super().__init__(name="fivrs-sph-inlet", daemon=True)
        self.drill_path = Path(drill_path)
        self.particle_mm = float(particle_mm)
        self.jet_mm = float(jet_mm)
        self.cc_per_s = LIVE_CC_PER_S
        self.offset_mm = LIVE_OFFSET_MM
        self.proximal_mm = LIVE_PROXIMAL_MM
        self._irrigation_mtime = None
        self.period = 1.0 / max(float(hz), 1.0)
        self.force_emit = False
        self.stop_event = threading.Event()
        self.last_inlet = None
        self._pull_irrigation()

    def _pull_irrigation(self):
        try:
            mtime = IRRIGATION_PATH.stat().st_mtime
        except OSError:
            return
        if self._irrigation_mtime is not None and mtime <= self._irrigation_mtime:
            return
        settings = live_irrigation()
        self.particle_mm = float(settings["particle_mm"])
        self.jet_mm = float(settings["jet_mm"])
        self.cc_per_s = float(settings["cc_per_s"])
        self.offset_mm = float(settings["offset_mm"])
        self.proximal_mm = float(settings["proximal_mm"])
        self._irrigation_mtime = mtime

    def run(self):
        while not self.stop_event.is_set() and not STOP_PATH.is_file():
            self.tick()
            self.stop_event.wait(self.period)

    def tick(self):
        self._pull_irrigation()
        try:
            tip, quat, on = drill_pose(read_drill_state(self.drill_path))
        except Exception:
            return
        if tip is None or quat is None:
            return
        emitting = bool(on or self.force_emit or DRILL_HOLD.is_file())
        inlet = inlet_from_pose(
            tip,
            quat,
            emitting=emitting,
            particle_mm=self.particle_mm,
            jet_mm=self.jet_mm,
            cc_per_s=self.cc_per_s,
            offset_mm=self.offset_mm,
            proximal_mm=self.proximal_mm,
            disk_area=True,
        )
        write_state(DEFAULT_STATE, inlet)
        self.last_inlet = inlet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bone", type=Path, default=BONE_DIR)
    parser.add_argument("--drill-state", type=Path, default=RUNTIME / "drill_state.json")
    parser.add_argument("--anatomy-adf", type=Path, default=DEFAULT_ADF)
    parser.add_argument("--force-emit", action="store_true")
    settings = live_irrigation()
    parser.add_argument("--particle-mm", type=float, default=float(settings["particle_mm"]))
    parser.add_argument("--jet-mm", type=float, default=float(settings["jet_mm"]))
    parser.add_argument("--numptcls", type=int, default=LIVE_NUMPTCLS)
    parser.add_argument("--measure-s", type=float, default=0.0)
    parser.add_argument("--no-rtx", action="store_true")
    parser.add_argument("--no-ambf", action="store_true")
    args = parser.parse_args()
    print(
        f"Live irrigation {IRRIGATION_PATH.name}: {args.jet_mm:g} mm jet, "
        f"{args.particle_mm:g} mm particles, {float(settings['cc_per_s']):g} cc/s, "
        f"offset {float(settings['offset_mm']):g} mm",
        flush=True,
    )

    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not (args.bone / "bone_surface.vdb").is_file():
        raise SystemExit(f"frozen bone VDB missing: {args.bone / 'bone_surface.vdb'}")

    if not args.no_ambf and not ambf_running():
        print("AMBF is down; starting the simulator", flush=True)
        start_ambf(args.anatomy_adf)
        deadline = time.time() + 90.0
        while time.time() < deadline and not ambf_running():
            time.sleep(0.5)
        if not ambf_running():
            raise SystemExit("AMBF did not start")
        time.sleep(8.0)

    outlet = None
    try:
        inlet = sync_inlet(
            args.drill_state,
            bone_dir=args.bone,
            force_emit=False,
            particle_mm=args.particle_mm,
            jet_mm=args.jet_mm,
            cc_per_s=float(settings["cc_per_s"]),
        )
        outlet = inlet.frame["outlet"]
    except RuntimeError as exc:
        print(f"inlet pose not ready yet ({exc})", flush=True)

    prepared = prepare_live_config(
        args.bone,
        outlet=outlet,
        particle_mm=args.particle_mm,
        jet_mm=args.jet_mm,
        numptcls=args.numptcls,
    )
    if outlet is not None:
        attach_inlet_config(
            args.bone / "config.json",
            inlet_from_drill(
                args.drill_state,
                force_emit=False,
                particle_mm=args.particle_mm,
                jet_mm=args.jet_mm,
                cc_per_s=float(settings["cc_per_s"]),
                offset_mm=float(settings["offset_mm"]),
                proximal_mm=float(settings["proximal_mm"]),
                disk_area=True,
            ),
            vdb_name="inlet_surface.vdb",
        )
    print(json.dumps({"prepared": prepared, "outlet": None if outlet is None else [float(v) for v in outlet]}, indent=2), flush=True)

    if not args.no_rtx:
        if not ambf_running():
            raise SystemExit("AMBF is not running; RTX needs volume_info")
        print("Restarting RTX with the SPH overlay", flush=True)
        start_rtx(args.anatomy_adf)

    print("Starting consoleParticles", flush=True)
    proc = start_console(args.bone)

    def _stop_requested(_signum, _frame):
        STOP_PATH.write_text("stop\n")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop_requested)
    poller = InletPoller(args.drill_state, args.particle_mm, args.jet_mm)
    poller.force_emit = bool(args.force_emit)
    poller.start()
    report = {
        "pid": proc.pid,
        "prepared": prepared,
        "idle": None,
        "emitting": None,
    }
    try:
        time.sleep(1.0)
        if proc.poll() is not None:
            raise SystemExit(f"consoleParticles exited {proc.returncode}; see {RUNTIME / 'sph.log'}")
        wait_for_dump()
        if not args.no_rtx:
            print("Waiting for RTX frames", flush=True)
            wait_for_rtx()
        if args.measure_s > 0:
            print(f"Sampling idle RTX FPS for {args.measure_s:g}s", flush=True)
            report["idle"] = sample_fps(args.measure_s, "gated_off")
            print(json.dumps(report["idle"]), flush=True)
            print("Force-emitting so the jet hits the frozen bone", flush=True)
            poller.force_emit = True
            poller.tick()
            time.sleep(1.5)
            report["emitting"] = sample_fps(args.measure_s, "force_emit")
            print(json.dumps(report["emitting"]), flush=True)
            if not args.force_emit:
                print("Returning emit to the drill pedal", flush=True)
                poller.force_emit = False
                poller.tick()
            (RUNTIME / "sph_fps.json").write_text(json.dumps(report, indent=2) + "\n")
        print("DFSPH is live. Hold Space or the drill pedal to emit. Touch sph.stop to end.", flush=True)
        while not STOP_PATH.is_file():
            code = proc.poll()
            if code is not None:
                print(f"consoleParticles exited {code}; see {RUNTIME / 'sph.log'}", flush=True)
                break
            time.sleep(0.5)
    finally:
        poller.stop_event.set()
        stop_console(proc)
        (RUNTIME / "sph_fps.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
