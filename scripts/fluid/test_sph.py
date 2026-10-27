"""Frozen-bone SPH: gravity, SDF stop, inlet pose, pedal gate."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import warp as wp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.occupancy import identity_frame
from fluid.rtx_stream import MM_TO_SIM, outlet_frame
from fluid.sdf_proxy import FluidCollisionProxy
from fluid.sph import SphSolver, bake_solver


def _floor(shape=(32, 32, 32), top=16):
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, :, :top] = True
    return occupied


def test_world_index_roundtrip():
    frame = identity_frame((32, 32, 32))
    world = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    index = frame.world_to_index(world)
    back = frame.index_to_world(index)
    assert np.allclose(world, back, atol=1e-9)


def test_particle_falls():
    occupied = np.zeros((16, 16, 16), dtype=bool)
    frame = identity_frame(occupied.shape)
    proxy = FluidCollisionProxy(occupied, downsample=2, band=3, dimensions=frame.dimensions)
    proxy.rebuild()
    solver = SphSolver(proxy, frame, max_particles=8, dt=0.001)
    tip = np.array([0.0, 0.0, 0.3], dtype=np.float64)
    solver.step(0.001, tip, (0.0, 0.0, 0.0, 1.0), True)
    z0 = float(solver.positions.numpy()[0][2])
    for _ in range(8):
        solver.step(0.001, tip, (0.0, 0.0, 0.0, 1.0), False)
    z1 = float(solver.positions.numpy()[0][2])
    assert z1 < z0


def test_particle_rests_on_slab():
    occupied = _floor()
    frame = identity_frame(occupied.shape)
    solver, _bake = bake_solver(occupied, frame, downsample=2, max_particles=8)
    solver.dt = 0.001
    # Floor occupies z index [0, 16). Top face is near local z = 16/32 - 0.5 = 0.
    slot = solver.seed((0.0, 0.0, 0.18))
    tip = np.zeros(3)
    for _ in range(80):
        solver.step(0.001, tip, (0.0, 0.0, 0.0, 1.0), False)
    z = float(solver.positions.numpy()[slot][2])
    assert z > -0.02
    assert z < 0.12


def test_inlet_uses_outlet_frame_and_pedal():
    occupied = np.zeros((16, 16, 16), dtype=bool)
    frame = identity_frame(occupied.shape)
    proxy = FluidCollisionProxy(occupied, downsample=2, band=3, dimensions=frame.dimensions)
    proxy.rebuild()
    solver = SphSolver(proxy, frame, max_particles=64, dt=0.016)
    tip = np.array([0.05, -0.02, 0.1], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    expected = outlet_frame(tip, quat, offset_mm=5.0, proximal_mm=10.0)
    solver.step(0.016, tip, quat, False)
    assert solver.emitted == 0
    assert solver.live == 0
    solver.step(0.016, tip, quat, True)
    assert solver.last_n == 16
    assert solver.live == 16
    pos = solver.positions.numpy()[:16]
    along = (pos - expected["outlet"]) @ expected["emit"]
    assert along.min() >= -1e-5
    side = (pos - expected["outlet"]) @ expected["right"]
    assert abs(np.mean(side)) < 0.5 * MM_TO_SIM
    assert abs(expected["emit_dot_shaft"] + 1.0) < 1e-9


def test_bake_and_step_do_not_crash():
    occupied = _floor((24, 24, 24), top=8)
    frame = identity_frame(occupied.shape)
    solver, bake_ms = bake_solver(occupied, frame, downsample=2, max_particles=64)
    assert bake_ms >= 0.0
    tip = np.zeros(3)
    for _ in range(12):
        solver.step(None, tip, (0.0, 0.0, 0.0, 1.0), True)
    assert solver.live > 0
    assert solver.emitted >= solver.live


def main():
    wp.init()
    test_world_index_roundtrip()
    test_particle_falls()
    test_particle_rests_on_slab()
    test_inlet_uses_outlet_frame_and_pedal()
    test_bake_and_step_do_not_crash()
    print("sph tests passed", flush=True)


if __name__ == "__main__":
    main()
