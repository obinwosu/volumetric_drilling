"""Prescribed irrigation stream for the RTX window.

This is not SPH. The bone renderer is untouched. 60 cc/s is not a target
rate: it only made the jet faster, not denser, and the previous 131072-point
upload ran even while the drill was off.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

from fluid.outlet import (
    AZIMUTH_DEG,
    MM_TO_SIM,
    OFFSET_MM,
    PROXIMAL_MM,
    jet_speed_mm_s,
    outlet_frame,
    particles_per_second,
    quat_axes,
)

# Diagnostic default. 60 cc/s is kept as a comparison row, not the setpoint.
CC_PER_S = 1.0
PARTICLE_MM = 1.0
JET_MM = 1.0
GRAVITY_MM_S2 = 9810.0
MAX_PARTICLES = 8192
LIFETIME_S = 1.5
# Grey, not cyan. These are still opaque points; real irrigation is a film.
COLOR = (0.62, 0.64, 0.67)
OUTLET_COLOR = (0.95, 0.75, 0.15)
RIGHT_COLOR = (0.95, 0.35, 0.2)
SHAFT_COLOR = (0.55, 0.55, 0.55)
EMIT_COLOR = (0.15, 0.95, 0.55)
GIZMO_JET_MM = 16.0
GIZMO_SHAFT_MM = 16.0
COMPARE_RATES = (0.25, 1.0, 15.0, 60.0)


def spacing_mm(particle_mm: float = PARTICLE_MM, jet_mm: float = JET_MM) -> float:
    """Along-jet spacing of 1 mm^3 drops in a 1 mm^2 face is 1 mm at any rate."""
    return float(particle_mm) ** 3 / float(jet_mm) ** 2


def travel_per_frame_mm(cc_per_s: float, fps: float = 60.0, jet_mm: float = JET_MM) -> float:
    return jet_speed_mm_s(cc_per_s, jet_mm) / float(fps)


def rate_rows(particle_mm: float = PARTICLE_MM, jet_mm: float = JET_MM, fps: float = 60.0):
    rows = []
    for cc in COMPARE_RATES:
        rows.append(
            {
                "cc_per_s": cc,
                "particles_per_s": particles_per_second(cc, particle_mm),
                "speed_mm_s": jet_speed_mm_s(cc, jet_mm),
                "spacing_mm": spacing_mm(particle_mm, jet_mm),
                "travel_per_16ms_mm": travel_per_frame_mm(cc, fps, jet_mm),
            }
        )
    return rows


@wp.kernel
def _integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3,
    lifetime: float,
):
    i = wp.tid()
    if ages[i] < 0.0:
        return
    age = ages[i] + dt
    if age > lifetime:
        ages[i] = -1.0
        return
    vel = velocities[i] + gravity * dt
    velocities[i] = vel
    positions[i] = positions[i] + vel * dt
    ages[i] = age


@wp.kernel
def _emit(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    cursor: int,
    count: int,
    outlet: wp.vec3,
    emit: wp.vec3,
    right: wp.vec3,
    up: wp.vec3,
    speed: float,
    half_jet: float,
    seed: int,
    capacity: int,
    dt: float,
):
    i = wp.tid()
    if i >= count:
        return
    slot = (cursor + i) % capacity
    rng = wp.rand_init(seed, i)
    jy = (wp.randf(rng) - 0.5) * 2.0 * half_jet
    jz = (wp.randf(rng) - 0.5) * 2.0 * half_jet
    # Birth time in this frame. i=0 left at the start, so by draw it has
    # already traveled speed*dt. That fills the gap to last frame's newest.
    t_i = (float(i) + 0.5) * dt / float(count)
    travel = speed * (dt - t_i)
    positions[slot] = outlet + emit * travel + right * jy + up * jz
    velocities[slot] = emit * speed
    ages[slot] = dt - t_i


@wp.kernel
def _compact(
    positions: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    packed: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    counter: wp.array(dtype=int),
    radius: float,
):
    i = wp.tid()
    if ages[i] < 0.0:
        return
    slot = wp.atomic_add(counter, 0, 1)
    packed[slot] = positions[i]
    radii[slot] = radius


class PrescribedStream:
    """GPU point jet. Draw only live particles. Gizmos show outlet, right, and jet."""

    def __init__(
        self,
        cc_per_s: float = CC_PER_S,
        particle_mm: float = PARTICLE_MM,
        jet_mm: float = JET_MM,
        offset_mm: float = OFFSET_MM,
        proximal_mm: float = PROXIMAL_MM,
        azimuth_deg: float = AZIMUTH_DEG,
        max_particles: int = MAX_PARTICLES,
        config_path: Path | None = None,
        gravity: bool = True,
    ):
        self.particle_mm = float(particle_mm)
        self.jet_mm = float(jet_mm)
        self.max_particles = int(max_particles)
        self.config_path = Path(config_path) if config_path is not None else None
        self._config_mtime = None
        self.cc_per_s = float(cc_per_s)
        self.offset_mm = float(offset_mm)
        self.proximal_mm = float(proximal_mm)
        self.azimuth_deg = float(azimuth_deg)
        self.gizmos = True
        self.enabled = True
        self.force_emit = False
        self.gravity_on = bool(gravity)
        self._apply_rate()
        self.positions = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.velocities = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.ages = wp.full(self.max_particles, -1.0, dtype=float)
        self.packed = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.packed_radii = wp.zeros(self.max_particles, dtype=float)
        self._counter = wp.zeros(1, dtype=int)
        self._carry = 0.0
        self._cursor = 0
        self._seed = 1
        self.live = 0
        self.emitted = 0
        self.accepted = 0
        self.dropped = 0
        self.requested = 0.0
        self.last_dt = 0.0
        self.last_n = 0
        self.last_frame = None
        self.step_ms = 0.0
        self.compact_ms = 0.0
        self.draw_ms = 0.0
        self._reload_config()

    def _apply_rate(self) -> None:
        self.rate = particles_per_second(self.cc_per_s, self.particle_mm)
        self.speed = jet_speed_mm_s(self.cc_per_s, self.jet_mm) * MM_TO_SIM
        self.radius = 0.5 * self.particle_mm * MM_TO_SIM
        self.half_jet = 0.5 * self.jet_mm * MM_TO_SIM
        g = GRAVITY_MM_S2 * MM_TO_SIM if self.gravity_on else 0.0
        self.gravity = wp.vec3(0.0, 0.0, -g)

    def _reload_config(self) -> None:
        if self.config_path is None or not self.config_path.is_file():
            return
        mtime = self.config_path.stat().st_mtime
        if self._config_mtime is not None and mtime <= self._config_mtime:
            return
        try:
            data = json.loads(self.config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self._config_mtime = mtime
        if "cc_per_s" in data:
            self.cc_per_s = float(data["cc_per_s"])
        if "jet_mm" in data:
            self.jet_mm = float(data["jet_mm"])
        if "particle_mm" in data:
            self.particle_mm = float(data["particle_mm"])
        if "offset_mm" in data:
            self.offset_mm = float(data["offset_mm"])
        if "proximal_mm" in data:
            self.proximal_mm = float(data["proximal_mm"])
        if "azimuth_deg" in data:
            self.azimuth_deg = float(data["azimuth_deg"])
        if "gizmos" in data:
            self.gizmos = bool(data["gizmos"])
        if "enabled" in data:
            self.enabled = bool(data["enabled"])
        if "force_emit" in data:
            self.force_emit = bool(data["force_emit"])
        self._apply_rate()

    def step(self, dt: float, tip, quat, emitting: bool) -> dict:
        self._reload_config()
        dt = float(max(dt, 0.0))
        self.last_dt = dt
        self.last_n = 0
        started = time.perf_counter()
        frame = outlet_frame(tip, quat, self.offset_mm, self.azimuth_deg, self.proximal_mm)
        self.last_frame = frame
        if dt > 0.0 and self.live > 0:
            wp.launch(
                _integrate,
                dim=self.max_particles,
                inputs=[self.positions, self.velocities, self.ages, dt, self.gravity, LIFETIME_S],
            )
        requested = 0.0
        accepted = 0
        if emitting and dt > 0.0:
            self._carry += self.rate * dt
            requested = self._carry
            n = int(self._carry)
            self._carry -= float(n)
            accepted = min(n, self.max_particles)
            dropped = n - accepted
            self.requested += requested
            self.accepted += accepted
            self.dropped += dropped
            self.last_n = accepted
            if accepted > 0:
                face_u, face_v = _face_axes(frame)
                wp.launch(
                    _emit,
                    dim=accepted,
                    inputs=[
                        self.positions,
                        self.velocities,
                        self.ages,
                        self._cursor,
                        accepted,
                        _vec(frame["outlet"]),
                        _vec(frame["emit"]),
                        _vec(face_u),
                        _vec(face_v),
                        self.speed,
                        self.half_jet,
                        self._seed,
                        self.max_particles,
                        dt,
                    ],
                )
                self._cursor = (self._cursor + accepted) % self.max_particles
                self._seed += accepted
                self.emitted += accepted
        self.step_ms = (time.perf_counter() - started) * 1e3
        return frame

    def compact(self) -> int:
        started = time.perf_counter()
        self._counter.zero_()
        wp.launch(
            _compact,
            dim=self.max_particles,
            inputs=[self.positions, self.ages, self.packed, self.packed_radii, self._counter, self.radius],
        )
        self.live = int(self._counter.numpy()[0])
        self.compact_ms = (time.perf_counter() - started) * 1e3
        return self.live

    def hide(self, viewer) -> None:
        viewer.log_points("fluid/stream", None, hidden=True)
        viewer.log_points("fluid/outlet", None, hidden=True)
        viewer.log_lines("fluid/emit_dir", None, None, EMIT_COLOR, hidden=True)
        viewer.log_lines("fluid/right_dir", None, None, RIGHT_COLOR, hidden=True)
        viewer.log_lines("fluid/shaft_dir", None, None, SHAFT_COLOR, hidden=True)

    def draw(self, viewer) -> None:
        started = time.perf_counter()
        if self.live == 0 and self.last_n == 0:
            n = 0
            self.compact_ms = 0.0
        else:
            n = self.compact()
        if n <= 0:
            viewer.log_points("fluid/stream", None, hidden=True)
        else:
            viewer.log_points(
                "fluid/stream",
                wp.clone(self.packed[:n]),
                radii=wp.clone(self.packed_radii[:n]),
                colors=COLOR,
            )
        if self.gizmos and self.last_frame is not None:
            self._draw_gizmos(viewer, self.last_frame)
        else:
            viewer.log_points("fluid/outlet", None, hidden=True)
            viewer.log_lines("fluid/emit_dir", None, None, EMIT_COLOR, hidden=True)
            viewer.log_lines("fluid/right_dir", None, None, RIGHT_COLOR, hidden=True)
            viewer.log_lines("fluid/shaft_dir", None, None, SHAFT_COLOR, hidden=True)
        self.draw_ms = (time.perf_counter() - started) * 1e3

    def _draw_gizmos(self, viewer, frame) -> None:
        jet = GIZMO_JET_MM * MM_TO_SIM
        shaft_len = GIZMO_SHAFT_MM * MM_TO_SIM
        outlet = frame["outlet"]
        tip = frame["tip"]
        viewer.log_points(
            "fluid/outlet",
            wp.array([wp.vec3(*outlet.tolist())], dtype=wp.vec3),
            radii=1.6 * MM_TO_SIM,
            colors=OUTLET_COLOR,
        )
        viewer.log_lines(
            "fluid/right_dir",
            wp.array([wp.vec3(*tip.tolist())], dtype=wp.vec3),
            wp.array([wp.vec3(*outlet.tolist())], dtype=wp.vec3),
            RIGHT_COLOR,
            width=0.45 * MM_TO_SIM,
        )
        viewer.log_lines(
            "fluid/emit_dir",
            wp.array([wp.vec3(*outlet.tolist())], dtype=wp.vec3),
            wp.array([wp.vec3(*(outlet + frame["emit"] * jet).tolist())], dtype=wp.vec3),
            EMIT_COLOR,
            width=0.55 * MM_TO_SIM,
        )
        viewer.log_lines(
            "fluid/shaft_dir",
            wp.array([wp.vec3(*tip.tolist())], dtype=wp.vec3),
            wp.array([wp.vec3(*(tip + frame["shaft"] * shaft_len).tolist())], dtype=wp.vec3),
            SHAFT_COLOR,
            width=0.3 * MM_TO_SIM,
        )

    def snapshot(self) -> dict:
        frame = self.last_frame or {}
        return {
            "cc_per_s": self.cc_per_s,
            "offset_mm": self.offset_mm,
            "proximal_mm": self.proximal_mm,
            "azimuth_deg": self.azimuth_deg,
            "rate_particles_s": self.rate,
            "speed_mm_s": self.speed / MM_TO_SIM,
            "spacing_mm": spacing_mm(self.particle_mm, self.jet_mm),
            "capacity": self.max_particles,
            "live": self.live,
            "emitted": self.emitted,
            "accepted": self.accepted,
            "dropped": self.dropped,
            "last_dt": self.last_dt,
            "last_emit_count": self.last_n,
            "enabled": self.enabled,
            "force_emit": self.force_emit,
            "emit_mode": "rate*dt carry; births staggered along the jet over the frame dt",
            "outlet": None if "outlet" not in frame else frame["outlet"].tolist(),
            "tip": None if "tip" not in frame else frame["tip"].tolist(),
            "emit": None if "emit" not in frame else frame["emit"].tolist(),
            "shaft": None if "shaft" not in frame else frame["shaft"].tolist(),
            "right": None if "right" not in frame else frame["right"].tolist(),
            "offset_from_tip_mm": frame.get("offset_mm"),
            "radial_from_shaft_mm": frame.get("radial_mm"),
            "proximal_from_tip_mm": frame.get("proximal_mm"),
            "emit_dot_shaft": frame.get("emit_dot_shaft"),
            "emit_dot_tangent": frame.get("emit_dot_tangent"),
            "emit_dot_right": frame.get("emit_dot_right"),
            "step_ms": self.step_ms,
            "compact_ms": self.compact_ms,
            "draw_ms": self.draw_ms,
            "rate_rows": rate_rows(self.particle_mm, self.jet_mm),
        }


def write_default_config(path: Path, cc_per_s: float) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cc_per_s": float(cc_per_s),
                "offset_mm": OFFSET_MM,
                "proximal_mm": PROXIMAL_MM,
                "azimuth_deg": AZIMUTH_DEG,
                "gizmos": True,
                "enabled": True,
                "force_emit": False,
            },
            indent=2,
        )
        + "\n"
    )


def _face_axes(frame):
    """1 mm face lies across the jet, not along it."""
    emit = np.asarray(frame["emit"], dtype=np.float64)
    radial = np.asarray(frame["radial"], dtype=np.float64)
    shaft = np.asarray(frame["shaft"], dtype=np.float64)
    u = radial - emit * float(np.dot(radial, emit))
    if float(np.linalg.norm(u)) < 1e-8:
        u = shaft - emit * float(np.dot(shaft, emit))
    u = u / float(np.linalg.norm(u))
    v = np.cross(emit, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def _vec(values) -> wp.vec3:
    return wp.vec3(float(values[0]), float(values[1]), float(values[2]))
