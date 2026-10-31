"""Packed 1 mm jet, 0.2 mm spheres, and a film that dies in tens of milliseconds."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.irrigation_bench import (
    CC_PER_S,
    FILM_MS,
    JET_MM,
    SPHERE_MM,
    IrrigationBench,
    cylinder_mesh,
    first_hit,
    splash_mesh,
    splash_volume_mm3,
    STREAM_SLICES,
    stream_mesh,
    load_irrigation,
    packed_column,
    quat_aiming_at_bone,
    save_irrigation,
    visual_scroll_mm_s,
)
from fluid.occupancy import VolumeFrame
from fluid.outlet import MM_TO_SIM, outlet_frame


def _box_occupancy():
    occupied = np.zeros((32, 32, 32), dtype=bool)
    occupied[16:24, 8:24, 8:24] = True
    frame = VolumeFrame(
        translation=np.zeros(3, dtype=np.float64),
        rotation=np.eye(3),
        dimensions=np.ones(3, dtype=np.float64),
        counts=np.array([32, 32, 32], dtype=np.float64),
    )
    return occupied, frame


def test_packed_column_is_one_mm_across():
    outlet = np.zeros(3, dtype=np.float64)
    emit = np.array((1.0, 0.0, 0.0))
    hit = outlet + emit * (20.0 * MM_TO_SIM)
    right = np.array((0.0, 1.0, 0.0))
    tangent = np.array((0.0, 0.0, 1.0))
    points = packed_column(outlet, hit, emit, right, tangent, JET_MM, SPHERE_MM)
    radial = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
    assert radial.max() < 0.65 * MM_TO_SIM
    assert radial.max() > 0.25 * MM_TO_SIM
    span = float(points[:, 0].max() - points[:, 0].min())
    assert span > 15.0 * MM_TO_SIM


def test_tube_radius_is_half_a_millimetre():
    start = np.zeros(3)
    end = np.array((10.0 * MM_TO_SIM, 0.0, 0.0))
    mesh = cylinder_mesh(start, end, 0.5 * JET_MM * MM_TO_SIM)
    assert mesh is not None
    radial = np.sqrt(mesh["verts"][:, 1] ** 2 + mesh["verts"][:, 2] ** 2)
    assert abs(float(radial.mean()) - 0.5 * MM_TO_SIM) < 0.05 * MM_TO_SIM
    assert len(mesh["tris"]) > 8


def test_ray_hits_occupancy_box():
    occupied, frame = _box_occupancy()
    origin = np.array((-0.2, 0.0, 0.0))
    hit = first_hit(origin, np.array((1.0, 0.0, 0.0)), frame, occupied)
    assert hit is not None
    ijk = np.floor(frame.world_to_index(hit)).astype(int)
    assert occupied[tuple(ijk)]


def test_short_standoff_still_hits():
    occupied, frame = _box_occupancy()
    # Surface of the box is at world x = 0. A 25 mm standoff is the tuner pose.
    origin = np.array((-25.0 * MM_TO_SIM, 0.0, 0.0))
    hit = first_hit(origin, np.array((1.0, 0.0, 0.0)), frame, occupied, max_mm=80.0)
    assert hit is not None
    assert float(hit[0] - origin[0]) < 30.0 * MM_TO_SIM


def test_long_gap_still_hits():
    occupied, frame = _box_occupancy()
    origin = np.array((-200.0 * MM_TO_SIM, 0.0, 0.0))
    hit = first_hit(origin, np.array((1.0, 0.0, 0.0)), frame, occupied)
    assert hit is not None
    assert float(hit[0] - origin[0]) < 210.0 * MM_TO_SIM


def test_film_is_gone_after_tens_of_milliseconds():
    occupied, frame = _box_occupancy()
    down = np.array((0.0, 0.0, -1.0))
    bench = IrrigationBench(occupied, frame, down=down)
    bench.settings["film_ms"] = FILM_MS
    hit = np.array((0.0, 0.0, 0.0))
    bench._birth_film(hit, dt=0.016, film_s=FILM_MS * 1e-3)
    assert int(np.count_nonzero(bench.film_age >= 0.0)) > 0
    bench._age_film(0.02, FILM_MS * 1e-3)
    assert int(np.count_nonzero(bench.film_age >= 0.0)) > 0
    bench._age_film(0.08, FILM_MS * 1e-3)
    assert int(np.count_nonzero(bench.film_age >= 0.0)) == 0


def test_zero_film_does_not_pool():
    occupied, frame = _box_occupancy()
    bench = IrrigationBench(occupied, frame, down=np.array((0.0, 0.0, -1.0)))
    bench.settings["film_ms"] = 0.0
    origin = np.array((-25.0 * MM_TO_SIM, 0.0, 0.0))
    quat = quat_aiming_at_bone(np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
    bench.step(0.016, origin, quat)
    assert bench.last_hit is not None
    assert int(np.count_nonzero(bench.film_age >= 0.0)) == 0


def test_stream_stays_within_twenty_percent_of_the_set_width():
    start = np.zeros(3)
    end = np.array((20.0 * MM_TO_SIM, 0.0, 0.0))
    radius = 0.5 * 0.58 * MM_TO_SIM
    mesh = stream_mesh(start, end, radius, time=0.37, cc_per_s=81.0)
    assert mesh is not None
    slices = STREAM_SLICES
    radial = np.sqrt(mesh["verts"][:, 1] ** 2 + mesh["verts"][:, 2] ** 2)
    assert float(radial.max()) <= radius * 1.20 + 1e-5
    assert float(radial.min()) >= radius * 0.80 - 1e-5
    assert float(radial.max() - radial.min()) > 0.05 * radius
    nose = mesh["verts"][:slices].mean(axis=0)
    tail = mesh["verts"][-slices:].mean(axis=0)
    assert float(np.linalg.norm(nose - start)) < 1e-4
    assert float(np.linalg.norm(tail - end)) < 1e-4
    assert len(mesh["verts"]) <= 18 * slices


def test_splash_is_one_to_two_cubic_millimetres():
    volume = splash_volume_mm3()
    assert 1.0 <= volume <= 2.0
    mesh = splash_mesh(np.zeros(3), np.array((0.0, 0.0, 1.0)))
    assert mesh is not None
    assert len(mesh["verts"]) < 200
    assert int(mesh["tris"].max()) < len(mesh["verts"])


def test_flow_changes_how_fast_the_beads_move():
    assert visual_scroll_mm_s(CC_PER_S) == 40.0
    assert visual_scroll_mm_s(2.0 * CC_PER_S) == 80.0
    assert visual_scroll_mm_s(0.5 * CC_PER_S) == 20.0


def test_save_irrigation_round_trips(tmp_path=None):
    import tempfile

    settings = {
        "draw": "spheres",
        "jet_mm": 0.6100000143051147,
        "particle_mm": 0.15,
        "cc_per_s": 58.0,
        "film_ms": 51.0,
        "offset_mm": 2.0,
        "proximal_mm": 0.0,
        "opacity": 0.057999998331069946,
        "enabled": True,
        "water": {"geometry_opacity": 0.057999998331069946},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "tuner_irrigation.json"
        save_irrigation(path, settings)
        loaded = load_irrigation(path)
        assert loaded["draw"] == "spheres"
        assert loaded["jet_mm"] == 0.61
        assert loaded["cc_per_s"] == 58.0
        assert loaded["opacity"] == 0.058


def test_jet_aims_at_the_bone():
    forward = np.array((1.0, 0.0, 0.0))
    right = np.array((0.0, 1.0, 0.0))
    quat = quat_aiming_at_bone(forward, right)
    frame = outlet_frame(np.zeros(3), quat, offset_mm=5.0, proximal_mm=0.0)
    assert float(np.dot(frame["emit"], forward)) > 0.9


if __name__ == "__main__":
    test_packed_column_is_one_mm_across()
    test_tube_radius_is_half_a_millimetre()
    test_ray_hits_occupancy_box()
    test_short_standoff_still_hits()
    test_long_gap_still_hits()
    test_film_is_gone_after_tens_of_milliseconds()
    test_zero_film_does_not_pool()
    test_splash_is_one_to_two_cubic_millimetres()
    test_stream_stays_within_twenty_percent_of_the_set_width()
    test_flow_changes_how_fast_the_beads_move()
    test_save_irrigation_round_trips()
    test_jet_aims_at_the_bone()
    print("irrigation bench ok")
