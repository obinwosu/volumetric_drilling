"""Stage 1 debris: volume becomes a capped dusting that falls and then rests."""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import warp as wp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from bone_debris import BoneDebris, load_config, outside_zones, parcels_from_volume, suction_radius_mm
from fluid.occupancy import MM_TO_SIM, VolumeFrame

VOXEL_MM = 0.390625


class _Viewer:
    def __init__(self):
        self.calls = []

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        count = 0 if points is None else int(points.shape[0])
        self.calls.append((name, count, hidden))


def _frame():
    counts = np.array([64, 64, 64], dtype=np.float64)
    dimensions = counts * (VOXEL_MM * MM_TO_SIM)
    return VolumeFrame(np.zeros(3), np.eye(3), dimensions, counts)


def _config(**overrides):
    config = load_config()
    config.update(overrides)
    return config


def _bite(debris, n, serial=1):
    base = np.array([30, 32, 32], dtype=np.int64)
    grids = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    return [(0.0, 0.0, serial, (base + grids).tolist())]


def _step_draw(debris, dt, quat=(0.0, 0.0, 0.0, 1.0), spin=0.0, viewer=None):
    debris.step(dt, np.zeros(3), quat, spin)
    debris.draw(viewer or _Viewer())


def test_volume_becomes_fewer_parcels_than_voxels():
    mm3 = 64 * (VOXEL_MM ** 3)
    config = _config()
    earned, carry = parcels_from_volume(mm3, config, 0.0)
    assert earned == int(mm3 * config["parcels_per_mm3"] * config["emission_scale"])
    assert config["clump_mm"] <= 0.5
    assert config["lifetime_s"] <= 12.0
    assert 0.0 <= carry < 1.0
    # A leftover fraction does not become a parcel, and it does not grow into a backlog.
    more, carry = parcels_from_volume(0.0, _config(), carry)
    assert more == 0
    huge, carry = parcels_from_volume(1.0e6, _config(), 0.0)
    assert huge > 0
    assert carry < 1.0


def test_dust_falls_and_then_rests():
    down = np.array((0.0, -1.0, 0.0), dtype=np.float64)
    debris = BoneDebris(
        _frame(),
        _config(
            scatter=0.0,
            spread_voxels=0.0,
            speed_mm_s=0.0,
            shaft_bias=0.0,
            flute_bias=0.0,
            drag_per_s=0.0,
            gravity_mm_s2=500.0,
            settle_s=0.05,
            lifetime_s=2.0,
            max_spawn_per_frame=16,
        ),
    )
    debris.note_removals(_bite(debris, 3, serial=3))
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, down)
    start = debris.sample_positions()
    assert len(start) > 0
    assert len(start) < 64
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, down)
    delta = (debris.sample_positions() - start).mean(axis=0)
    assert delta[1] < 0.0
    assert abs(delta[0]) < abs(delta[1]) * 0.2
    assert abs(delta[2]) < abs(delta[1]) * 0.2
    for _ in range(4):
        debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, down)
    rested = debris.sample_positions()
    debris.step(0.02, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, down)
    shift = np.linalg.norm(debris.sample_positions() - rested, axis=1)
    assert float(np.max(shift)) < 1e-5


def test_irrigation_and_suction_clear_paste_and_block_new_paste():
    assert abs(suction_radius_mm(12) - 3.0) < 1e-6
    assert suction_radius_mm(3) < suction_radius_mm(12)
    debris = BoneDebris(
        _frame(),
        _config(scatter=0.0, spread_voxels=0.0, speed_mm_s=0.0, gravity_mm_s2=0.0, lifetime_s=5.0),
    )
    debris.note_removals(_bite(debris, 3, serial=2))
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0)
    born = debris.sample_positions()
    assert len(born) > 0
    zones = {"spheres": [(born.mean(axis=0), 5.0 * MM_TO_SIM)], "segment": None}
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, zones=zones)
    assert debris.count == 0
    debris.note_removals(_bite(debris, 3, serial=8))
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, zones=zones)
    assert debris.count == 0
    assert debris.washed > 0
    far = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
    assert bool(outside_zones(far, zones)[0])


def test_wet_bone_clears_paste_after_the_water_is_gone():
    class _Film:
        any_wet = True

        def covers(self, points):
            return np.linalg.norm(points, axis=1) < 1.0

    debris = BoneDebris(
        _frame(),
        _config(scatter=0.0, spread_voxels=0.0, speed_mm_s=0.0, gravity_mm_s2=0.0, lifetime_s=5.0),
    )
    debris.note_removals(_bite(debris, 2, serial=4))
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0)
    assert debris.count > 0
    debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0, zones={"film": _Film()})
    assert debris.count == 0
    assert debris.washed > 0


def test_count_stays_capped_and_then_clears():
    debris = BoneDebris(_frame(), _config(max_active=8, max_spawn_per_frame=8, lifetime_s=0.22))
    for serial in range(6):
        debris.note_removals(_bite(debris, 4, serial=serial + 1))
        _step_draw(debris, 0.016)
        assert debris.count <= 8
    assert debris.count == 8
    for _ in range(20):
        _step_draw(debris, 0.05)
    assert debris.count == 0
    assert debris.snapshot()["retired"] >= 8


def test_idle_draw_does_not_upload_and_live_path_does_not_read_back():
    debris = BoneDebris(_frame(), _config())
    viewer = _Viewer()
    _step_draw(debris, 0.016, viewer=viewer)
    assert viewer.calls == []
    assert debris.snapshot()["readbacks"] == 0
    debris.note_removals(_bite(debris, 4, serial=2))
    _step_draw(debris, 0.016, viewer=viewer)
    assert viewer.calls[-1][1] == debris.count
    assert viewer.calls[-1][1] < 64
    assert debris.snapshot()["readbacks"] == 0


def test_frame_cost():
    idle = BoneDebris(_frame(), _config())
    busy = BoneDebris(
        _frame(),
        _config(max_active=192, max_spawn_per_frame=192, parcels_per_mm3=80.0, lifetime_s=10.0),
    )
    busy.note_removals(_bite(busy, 6, serial=4))
    _step_draw(busy, 0.016)
    assert busy.count > 32
    rows = {
        "idle": _timed(idle, 40),
        "live": _timed(busy, 40),
    }
    print(
        f"debris cost  idle {rows['idle']:.3f} ms   "
        f"live {busy.count} parcels {rows['live']:.3f} ms   "
        f"voxel {idle.voxel_mm3:.4f} mm3   "
        f"realized {busy.snapshot()['parcels_per_mm3']:.2f} parcels/mm3",
        flush=True,
    )
    assert rows["idle"] < 1.0
    assert rows["live"] < 8.0


def _timed(debris, steps):
    viewer = _Viewer()
    wp.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        debris.step(0.016, np.zeros(3), (0.0, 0.0, 0.0, 1.0), 0.0)
        debris.draw(viewer)
    wp.synchronize()
    return (time.perf_counter() - started) * 1e3 / steps


if __name__ == "__main__":
    test_volume_becomes_fewer_parcels_than_voxels()
    test_dust_falls_and_then_rests()
    test_irrigation_and_suction_clear_paste_and_block_new_paste()
    test_wet_bone_clears_paste_after_the_water_is_gone()
    test_count_stays_capped_and_then_clears()
    test_idle_draw_does_not_upload_and_live_path_does_not_read_back()
    test_frame_cost()
    print("bone debris stage 1 ok", flush=True)
