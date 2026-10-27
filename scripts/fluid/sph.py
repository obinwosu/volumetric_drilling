"""Headless WCSPH against a frozen occupancy SDF.

This is the first SPH sidecar, not openMaelstrom and not the prescribed
stream. The bone renderer and haptic loop are not imported.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

from .occupancy import VolumeFrame, drill_pose, read_drill_state
from .rtx_stream import (
    CC_PER_S,
    MM_TO_SIM,
    OFFSET_MM,
    PARTICLE_MM,
    PROXIMAL_MM,
    jet_speed_mm_s,
    outlet_frame,
    particles_per_second,
)
from .sdf_proxy import FluidCollisionProxy

MAX_PARTICLES = 2048
DT = 2.0e-4
GRAVITY = wp.vec3(0.0, 0.0, -9810.0 * MM_TO_SIM)


@wp.func
def _poly6(r2: float, h: float, scale: float) -> float:
    h2 = h * h
    if r2 >= h2:
        return 0.0
    d = h2 - r2
    return scale * d * d * d


@wp.func
def _spiky_grad(delta: wp.vec3, r: float, h: float, scale: float) -> wp.vec3:
    if r <= 1.0e-8 or r >= h:
        return wp.vec3(0.0, 0.0, 0.0)
    q = h - r
    return delta * (scale * q * q / r)


@wp.func
def _trilinear(field: wp.array3d(dtype=float), p: wp.vec3) -> float:
    nx = float(field.shape[0] - 1)
    ny = float(field.shape[1] - 1)
    nz = float(field.shape[2] - 1)
    x = wp.clamp(p[0], 0.0, nx - 1.0e-4)
    y = wp.clamp(p[1], 0.0, ny - 1.0e-4)
    z = wp.clamp(p[2], 0.0, nz - 1.0e-4)
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    z0 = int(wp.floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    fx = x - float(x0)
    fy = y - float(y0)
    fz = z - float(z0)
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.kernel
def _density(
    positions: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    mass: float,
    h: float,
    scale: float,
    count: int,
):
    i = wp.tid()
    if ages[i] < 0.0:
        densities[i] = 0.0
        return
    rho = float(0.0)
    pi = positions[i]
    for j in range(count):
        if ages[j] < 0.0:
            continue
        d = pi - positions[j]
        rho += mass * _poly6(wp.dot(d, d), h, scale)
    densities[i] = rho


@wp.kernel
def _forces(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    accels: wp.array(dtype=wp.vec3),
    mass: float,
    rest: float,
    stiffness: float,
    visc: float,
    h: float,
    spiky: float,
    gravity: wp.vec3,
    count: int,
):
    i = wp.tid()
    if ages[i] < 0.0:
        accels[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    rho_i = densities[i]
    if rho_i <= 1.0e-8:
        accels[i] = gravity
        return
    p_i = stiffness * (rho_i - rest)
    force = wp.vec3(0.0, 0.0, 0.0)
    pi = positions[i]
    vi = velocities[i]
    for j in range(count):
        if i == j or ages[j] < 0.0:
            continue
        delta = pi - positions[j]
        r = wp.length(delta)
        if r >= h or r <= 1.0e-8:
            continue
        rho_j = densities[j]
        if rho_j <= 1.0e-8:
            continue
        p_j = stiffness * (rho_j - rest)
        force = force + _spiky_grad(delta, r, h, spiky) * (-mass * (p_i + p_j) / (2.0 * rho_j))
        force = force + (velocities[j] - vi) * (visc * mass / rho_j * (h - r))
    accels[i] = gravity + force


@wp.kernel
def _integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    accels: wp.array(dtype=wp.vec3),
    sdf: wp.array3d(dtype=float),
    dt: float,
    radius: float,
    voxel_world: float,
    downsample: float,
    origin: wp.vec3,
    rot0: wp.vec3,
    rot1: wp.vec3,
    rot2: wp.vec3,
    dimensions: wp.vec3,
    counts: wp.vec3,
):
    i = wp.tid()
    if ages[i] < 0.0:
        return
    vel = velocities[i] + accels[i] * dt
    pos = positions[i] + vel * dt
    local = wp.vec3(
        wp.dot(pos - origin, rot0),
        wp.dot(pos - origin, rot1),
        wp.dot(pos - origin, rot2),
    )
    index = wp.cw_mul(local + dimensions * 0.5, wp.cw_div(counts, dimensions))
    coarse = index / downsample
    distance = _trilinear(sdf, coarse) * voxel_world
    if distance < radius:
        gx = _trilinear(sdf, coarse + wp.vec3(0.4, 0.0, 0.0)) - _trilinear(sdf, coarse - wp.vec3(0.4, 0.0, 0.0))
        gy = _trilinear(sdf, coarse + wp.vec3(0.0, 0.4, 0.0)) - _trilinear(sdf, coarse - wp.vec3(0.0, 0.4, 0.0))
        gz = _trilinear(sdf, coarse + wp.vec3(0.0, 0.0, 0.4)) - _trilinear(sdf, coarse - wp.vec3(0.0, 0.0, 0.4))
        grad_i = wp.vec3(gx, gy, gz)
        n_local = wp.normalize(grad_i)
        normal = wp.vec3(
            n_local[0] * rot0[0] + n_local[1] * rot1[0] + n_local[2] * rot2[0],
            n_local[0] * rot0[1] + n_local[1] * rot1[1] + n_local[2] * rot2[1],
            n_local[0] * rot0[2] + n_local[1] * rot1[2] + n_local[2] * rot2[2],
        )
        nlen = wp.length(normal)
        if nlen > 1.0e-8:
            normal = normal / nlen
            pos = pos + normal * (radius - distance)
            inward = wp.dot(vel, normal)
            if inward < 0.0:
                vel = vel - normal * inward
    positions[i] = pos
    velocities[i] = vel
    ages[i] = ages[i] + dt


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
    t_i = (float(i) + 0.5) * dt / float(count)
    travel = speed * (dt - t_i)
    positions[slot] = outlet + emit * travel + right * jy + up * jz
    velocities[slot] = emit * speed
    ages[slot] = dt - t_i


class SphSolver:
    """WCSPH on a static occupancy SDF. Inlet seeds follow the drill pose."""

    def __init__(
        self,
        proxy: FluidCollisionProxy,
        frame: VolumeFrame,
        cc_per_s: float = CC_PER_S,
        particle_mm: float = PARTICLE_MM,
        offset_mm: float = OFFSET_MM,
        proximal_mm: float = PROXIMAL_MM,
        max_particles: int = MAX_PARTICLES,
        dt: float = DT,
    ):
        self.proxy = proxy
        self.frame = frame
        self.cc_per_s = float(cc_per_s)
        self.particle_mm = float(particle_mm)
        self.offset_mm = float(offset_mm)
        self.proximal_mm = float(proximal_mm)
        self.max_particles = int(max_particles)
        self.dt = float(dt)
        self.radius = 0.5 * self.particle_mm * MM_TO_SIM
        self.h = 2.2 * (self.particle_mm * MM_TO_SIM)
        self.mass = 1.0
        self.poly6 = 315.0 / (64.0 * math.pi * self.h**9)
        self.rest = self.mass * self.poly6 * (self.h**2) ** 3
        self.stiffness = self.rest * (20.0**2) / 7.0
        self.spiky = -45.0 / (math.pi * self.h**6)
        self.visc = 0.12
        self.speed = jet_speed_mm_s(self.cc_per_s) * MM_TO_SIM
        self.rate = particles_per_second(self.cc_per_s, self.particle_mm)
        self.half_jet = 0.5 * self.particle_mm * MM_TO_SIM
        self.voxel_world = float(np.max(proxy.spacing) * proxy.downsample)
        self.positions = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.velocities = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.ages = wp.full(self.max_particles, -1.0, dtype=float)
        self.densities = wp.zeros(self.max_particles, dtype=float)
        self.accels = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.sdf = wp.array(np.ascontiguousarray(proxy.distance), dtype=float, ndim=3)
        self._carry = 0.0
        self._cursor = 0
        self._seed = 1
        self.live = 0
        self.emitted = 0
        self.step_ms = 0.0
        self.last_n = 0
        self.last_frame = None
        self.bake_generation = proxy.generation

    def seed(self, position, velocity=(0.0, 0.0, 0.0)) -> int:
        """Place one live particle. Used by tests; the sidecar uses the inlet."""
        ages = self.ages.numpy()
        slot = int(np.where(ages < 0.0)[0][0]) if np.any(ages < 0.0) else 0
        pos = self.positions.numpy()
        vel = self.velocities.numpy()
        pos[slot] = np.asarray(position, dtype=np.float32)
        vel[slot] = np.asarray(velocity, dtype=np.float32)
        ages[slot] = 0.0
        self.positions.assign(pos)
        self.velocities.assign(vel)
        self.ages.assign(ages)
        self.live = int(np.count_nonzero(ages >= 0.0))
        return slot

    def step(self, dt: float | None, tip, quat, emitting: bool) -> dict:
        dt = self.dt if dt is None else float(max(dt, 0.0))
        started = time.perf_counter()
        frame = outlet_frame(tip, quat, self.offset_mm, 0.0, self.proximal_mm)
        self.last_frame = frame
        if dt > 0.0:
            wp.launch(
                _density,
                dim=self.max_particles,
                inputs=[
                    self.positions,
                    self.ages,
                    self.densities,
                    self.mass,
                    self.h,
                    self.poly6,
                    self.max_particles,
                ],
            )
            wp.launch(
                _forces,
                dim=self.max_particles,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.ages,
                    self.densities,
                    self.accels,
                    self.mass,
                    self.rest,
                    self.stiffness,
                    self.visc,
                    self.h,
                    self.spiky,
                    GRAVITY,
                    self.max_particles,
                ],
            )
            origin = wp.vec3(*self.frame.translation.tolist())
            rot = self.frame.rotation
            wp.launch(
                _integrate,
                dim=self.max_particles,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.ages,
                    self.accels,
                    self.sdf,
                    dt,
                    self.radius,
                    self.voxel_world,
                    float(self.proxy.downsample),
                    origin,
                    wp.vec3(*rot[:, 0].tolist()),
                    wp.vec3(*rot[:, 1].tolist()),
                    wp.vec3(*rot[:, 2].tolist()),
                    wp.vec3(*self.frame.dimensions.tolist()),
                    wp.vec3(*self.frame.counts.tolist()),
                ],
            )
        self.last_n = 0
        if emitting and dt > 0.0:
            self._carry += self.rate * dt
            n = int(self._carry)
            self._carry -= float(n)
            accepted = min(n, self.max_particles)
            self.last_n = accepted
            if accepted > 0:
                wp.launch(
                    _emit,
                    dim=accepted,
                    inputs=[
                        self.positions,
                        self.velocities,
                        self.ages,
                        self._cursor,
                        accepted,
                        wp.vec3(*frame["outlet"].tolist()),
                        wp.vec3(*frame["emit"].tolist()),
                        wp.vec3(*frame["right"].tolist()),
                        wp.vec3(*frame["tangent"].tolist()),
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
        self.live = int(np.count_nonzero(self.ages.numpy() >= 0.0))
        self.step_ms = (time.perf_counter() - started) * 1e3
        return frame

    def snapshot(self) -> dict:
        frame = self.last_frame or {}
        return {
            "solver": "wcsph",
            "live": self.live,
            "emitted": self.emitted,
            "last_emit_count": self.last_n,
            "step_ms": self.step_ms,
            "dt": self.dt,
            "cc_per_s": self.cc_per_s,
            "bake_generation": self.bake_generation,
            "proxy_shape": list(self.proxy.shape),
            "voxel_world": self.voxel_world,
            "outlet": None if "outlet" not in frame else frame["outlet"].tolist(),
            "emit_dot_shaft": frame.get("emit_dot_shaft"),
            "proximal_mm": frame.get("proximal_mm"),
            "radial_mm": frame.get("radial_mm"),
        }


def bake_solver(
    occupied: np.ndarray,
    frame: VolumeFrame,
    downsample: int = 4,
    cc_per_s: float = CC_PER_S,
    max_particles: int = MAX_PARTICLES,
) -> tuple[SphSolver, float]:
    started = time.perf_counter()
    radius = 0.5 * PARTICLE_MM * MM_TO_SIM
    proxy = FluidCollisionProxy(
        occupied,
        downsample=downsample,
        particle_radius=radius,
        dimensions=frame.dimensions,
    )
    proxy.rebuild()
    solver = SphSolver(proxy, frame, cc_per_s=cc_per_s, max_particles=max_particles)
    return solver, (time.perf_counter() - started) * 1e3


def write_state(path: Path, solver: SphSolver, extra: dict | None = None) -> None:
    data = solver.snapshot()
    if extra:
        data.update(extra)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def step_from_drill(solver: SphSolver, drill_path: Path, force_emit: bool = False) -> dict:
    state = read_drill_state(drill_path)
    tip, quat, on = drill_pose(state)
    if tip is None:
        tip = np.zeros(3)
        quat = (0.0, 0.0, 0.0, 1.0)
        on = False
    return solver.step(None, tip, quat, bool(on or force_emit))
