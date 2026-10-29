#!/usr/bin/env python3
"""Moving Maelstrom inlet: seeds follow the constrained tip, pedal gates emit.

Native inletPositions are fixed world points. This module writes a small
"surface" VDB at the current outlet, a reloadable seed buffer, and the
kind=2 config inlet block. It does not draw bone or run in the haptic loop.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluid.frozen_bone import inspect_vdb, run_writer, write_dense
from fluid.occupancy import drill_pose, read_drill_state
from fluid.outlet import (
    CC_PER_S,
    JET_MM,
    MM_TO_SIM,
    OFFSET_MM,
    PARTICLE_MM,
    PROXIMAL_MM,
    jet_speed_mm_s,
    outlet_frame,
)

STATE_MAGIC = b"FIVRINLT"
STATE_VERSION = 2
MAX_SEEDS = 64
DEFAULT_STATE = Path("/tmp/fivrs-rtx/inlet_state.bin")


@dataclass
class MovingInlet:
    frame: dict
    seeds: np.ndarray
    velocity: np.ndarray
    emitting: bool
    particle_radius: float
    support: float
    jet_mm: float
    cc_per_s: float
    flow_sim3_s: float
    particle_volume: float
    disk_area: bool = False

    @property
    def seed_count(self) -> int:
        return int(self.seeds.shape[0])

    @property
    def count(self) -> int:
        return self.seed_count if self.emitting else 0

    def report(self) -> dict:
        return {
            "emitting": bool(self.emitting),
            "count": self.count,
            "outlet": _vec(self.frame["outlet"]),
            "tip": _vec(self.frame["tip"]),
            "emit": _vec(self.frame["emit"]),
            "velocity": _vec(self.velocity),
            "proximal_mm": float(self.frame["proximal_mm"]),
            "radial_mm": float(self.frame["radial_mm"]),
            "emit_dot_shaft": float(self.frame["emit_dot_shaft"]),
            "emit_dot_tangent": float(self.frame["emit_dot_tangent"]),
            "particle_radius": float(self.particle_radius),
            "cc_per_s": float(self.cc_per_s),
            "seed_count": self.seed_count,
            "flow_sim3_s": float(self.flow_sim3_s),
            "particle_volume": float(self.particle_volume),
            "speed_mm_s": float(np.linalg.norm(self.velocity) / MM_TO_SIM),
        }


def disk_offsets(jet_mm: float, particle_mm: float) -> np.ndarray:
    """Packing in the right-tangent plane. Occupancy, not particles/s."""
    half = 0.5 * float(jet_mm) * MM_TO_SIM
    spacing = float(particle_mm) * MM_TO_SIM
    if spacing <= 0.0 or half <= 0.0:
        raise ValueError("jet_mm and particle_mm must be positive")
    n = max(1, int(round(float(jet_mm) / float(particle_mm))))
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)
    extent = 0.5 * (n - 1) * spacing
    axis = np.linspace(-extent, extent, n)
    pts = []
    for y in axis:
        for z in axis:
            if y * y + z * z <= (half + 0.25 * spacing) ** 2:
                pts.append((y, z))
    if not pts:
        return np.zeros((1, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def particle_volume_sim(particle_mm: float) -> float:
    """SPH particle volume, (4/3) pi r^3. This is the discretization, not the drop you see."""
    radius = 0.5 * float(particle_mm) * MM_TO_SIM
    return (4.0 / 3.0) * math.pi * radius ** 3


def flow_sim3_s(cc_per_s: float) -> float:
    return float(cc_per_s) * 1000.0 * (MM_TO_SIM ** 3)


def disk_area_mm2(diameter_mm: float) -> float:
    return math.pi * (0.5 * float(diameter_mm)) ** 2


def jet_speed_disk_mm_s(cc_per_s: float, diameter_mm: float) -> float:
    """U from Q = A U. A is the circular patch, not a square of side diameter_mm."""
    area = disk_area_mm2(diameter_mm)
    if area <= 0.0:
        raise ValueError("diameter_mm must be positive")
    return float(cc_per_s) * 1000.0 / area


def accumulate_births(remainder: float, flow: float, volume: float, dt: float, cap: int = MAX_SEEDS):
    """N = floor(remainder + Q dt / V). The fraction is kept for the next step."""
    if not (volume > 0.0 and dt > 0.0 and flow > 0.0):
        return float(remainder), 0
    remainder = min(float(cap), float(remainder) + float(flow) * float(dt) / float(volume))
    born = int(math.floor(remainder))
    if born > cap:
        born = cap
    return remainder - born, born


def inlet_from_pose(
    tip,
    quat,
    emitting: bool,
    offset_mm: float = OFFSET_MM,
    proximal_mm: float = PROXIMAL_MM,
    jet_mm: float = JET_MM,
    particle_mm: float = PARTICLE_MM,
    cc_per_s: float = CC_PER_S,
    disk_area: bool = False,
) -> MovingInlet:
    frame = outlet_frame(tip, quat, offset_mm=offset_mm, proximal_mm=proximal_mm)
    if disk_area:
        speed = jet_speed_disk_mm_s(cc_per_s, jet_mm) * MM_TO_SIM
    else:
        speed = jet_speed_mm_s(cc_per_s, jet_mm) * MM_TO_SIM
    velocity = frame["emit"] * speed
    offsets = disk_offsets(jet_mm, particle_mm)
    seeds = (
        frame["outlet"][None, :]
        + offsets[:, 0:1] * frame["right"][None, :]
        + offsets[:, 1:2] * frame["tangent"][None, :]
    )
    radius = 0.5 * float(particle_mm) * MM_TO_SIM
    return MovingInlet(
        frame=frame,
        seeds=np.ascontiguousarray(seeds, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        emitting=bool(emitting),
        particle_radius=radius,
        support=2.0 * radius,
        jet_mm=float(jet_mm),
        cc_per_s=float(cc_per_s),
        flow_sim3_s=flow_sim3_s(cc_per_s),
        particle_volume=particle_volume_sim(particle_mm),
        disk_area=bool(disk_area),
    )


def inlet_from_drill(path: Path, force_emit: bool = False, **kwargs) -> MovingInlet:
    tip, quat, on = drill_pose(read_drill_state(path))
    if tip is None or quat is None:
        raise RuntimeError(f"no tip pose in {path}")
    return inlet_from_pose(tip, quat, emitting=bool(on or force_emit), **kwargs)


def write_state(path: Path, inlet: MovingInlet) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The patch is written whether or not the pedal is down, so the next
    # solver step can follow the drill. emitting only gates births.
    seeds = inlet.seeds
    count = int(seeds.shape[0])
    if count > MAX_SEEDS:
        raise ValueError(f"inlet has {count} seeds; max is {MAX_SEEDS}")
    packed = np.zeros((count, 4), dtype=np.float32)
    if count:
        packed[:, :3] = seeds
        packed[:, 3] = inlet.support
    vel = np.asarray(inlet.velocity, dtype=np.float32)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(STATE_MAGIC)
        handle.write(struct.pack("<iii", STATE_VERSION, 1 if inlet.emitting else 0, count))
        handle.write(struct.pack("<ffff", float(vel[0]), float(vel[1]), float(vel[2]), 0.0))
        handle.write(struct.pack("<ff", float(inlet.flow_sim3_s), float(inlet.particle_volume)))
        handle.write(packed.tobytes())
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(inlet.report(), indent=2) + "\n")
    return path


def read_state(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if raw[:8] != STATE_MAGIC:
        raise ValueError("inlet state magic is not FIVRINLT")
    version, emitting, count = struct.unpack_from("<iii", raw, 8)
    vx, vy, vz, vw = struct.unpack_from("<ffff", raw, 20)
    if version >= 2:
        flow, particle_volume = struct.unpack_from("<ff", raw, 36)
        pos_off = 44
    else:
        flow, particle_volume = 0.0, 0.0
        pos_off = 36
    if count:
        packed = np.frombuffer(raw[pos_off:], dtype=np.float32, count=count * 4).reshape(count, 4)
        seeds = packed[:, :3].copy()
        support = packed[:, 3].copy()
    else:
        seeds = np.zeros((0, 3), np.float32)
        support = np.zeros(0, np.float32)
    return {
        "version": version,
        "emitting": bool(emitting),
        "count": count,
        "velocity": [float(vx), float(vy), float(vz), float(vw)],
        "flow_sim3_s": float(flow),
        "particle_volume": float(particle_volume),
        "seeds": seeds,
        "support": support,
    }


def write_inlet_vdb(path: Path, inlet: MovingInlet, samples: int = 24) -> Path:
    """Sphere at the outlet. Negative inside. fileToVDB looks for name=surface."""
    radius = max(0.5 * inlet.jet_mm * MM_TO_SIM, inlet.particle_radius)
    band = 2.0 * radius
    spacing = radius / 3.0
    half = radius + band
    n = int(np.ceil(2.0 * half / spacing)) + 1
    n = max(n, samples)
    origin = inlet.frame["outlet"] - 0.5 * (n - 1) * spacing
    grid = np.stack(
        np.meshgrid(
            origin[0] + np.arange(n, dtype=np.float64) * spacing,
            origin[1] + np.arange(n, dtype=np.float64) * spacing,
            origin[2] + np.arange(n, dtype=np.float64) * spacing,
            indexing="ij",
        ),
        axis=-1,
    )
    dist = np.linalg.norm(grid - inlet.frame["outlet"], axis=-1) - radius
    distance = np.clip(dist, -band, band).astype(np.float32)
    path = Path(path)
    dense = path.with_suffix(".dense.bin")
    write_dense(dense, origin, spacing, distance)
    run_writer(["write", str(dense), str(path)])
    dense.unlink(missing_ok=True)
    return path


def attach_inlet_config(path: Path, inlet: MovingInlet, vdb_name: str = "inlet_surface.vdb") -> Path:
    path = Path(path)
    payload = json.loads(path.read_text()) if path.is_file() else {}
    modules = payload.setdefault("modules", {})
    modules["volumeInlets"] = True
    vel = inlet.velocity
    payload["inlet_volumes"] = {
        "volume1": {
            "file": vdb_name,
            "dur": -1,
            "del": 0.0,
            "r": float(inlet.particle_radius),
            "vel": f"{vel[0]} {vel[1]} {vel[2]} 0",
        }
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_inlet(out: Path, inlet: MovingInlet, state_path: Path | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vdb = write_inlet_vdb(out / "inlet_surface.vdb", inlet)
    state = write_state(state_path or DEFAULT_STATE, inlet)
    write_state(out / "inlet_state.bin", inlet)
    attach_inlet_config(out / "config.json", inlet)
    report = inlet.report()
    report["vdb"] = str(vdb)
    report["vdb_bytes"] = vdb.stat().st_size
    report["state"] = str(state)
    report["inspect"] = inspect_vdb(vdb)
    (out / "inlet_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _vec(value) -> list:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


def main():
    runtime = Path("/tmp/fivrs-rtx")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drill-state", type=Path, default=runtime / "drill_state.json")
    parser.add_argument("--out", type=Path, default=runtime / "frozen_bone")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--force-emit", action="store_true")
    parser.add_argument("--cc-per-s", type=float, default=CC_PER_S)
    args = parser.parse_args()
    inlet = inlet_from_drill(args.drill_state, force_emit=args.force_emit, cc_per_s=args.cc_per_s)
    report = write_inlet(args.out, inlet, state_path=args.state)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
