"""Geometry, rate, and displacement checks for the prescribed stream."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.rtx_stream import (
    CC_PER_S,
    MM_TO_SIM,
    OFFSET_MM,
    PROXIMAL_MM,
    PrescribedStream,
    jet_speed_mm_s,
    outlet_frame,
    particles_per_second,
    rate_rows,
    spacing_mm,
)


def test_rate_table_does_not_treat_60_as_denser():
    rows = {row["cc_per_s"]: row for row in rate_rows()}
    assert abs(rows[1.0]["particles_per_s"] - 1000.0) < 1e-9
    assert abs(rows[60.0]["particles_per_s"] - 60000.0) < 1e-9
    assert abs(rows[1.0]["spacing_mm"] - rows[60.0]["spacing_mm"]) < 1e-9
    assert rows[60.0]["travel_per_16ms_mm"] > 900.0


def test_outlet_is_5mm_drill_right_and_10mm_proximal():
    tip = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    frame = outlet_frame(tip, quat, offset_mm=5.0, proximal_mm=10.0, azimuth_deg=0.0)
    offset = frame["outlet"] - tip
    # Identity: shaft +X (tip→handle), right +Y, jet toward the tip.
    assert np.allclose(frame["shaft"], (1.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(frame["right"], (0.0, 1.0, 0.0), atol=1e-6)
    assert np.allclose(frame["emit"], (-1.0, 0.0, 0.0), atol=1e-6)
    assert abs(frame["proximal_mm"] - 10.0) < 1e-6
    assert abs(frame["radial_mm"] - 5.0) < 1e-6
    assert abs(np.dot(offset, frame["shaft"]) / MM_TO_SIM - 10.0) < 1e-6
    assert abs(np.dot(offset, frame["right"]) / MM_TO_SIM - 5.0) < 1e-6
    assert abs(np.dot(offset, frame["tangent"])) < 1e-9
    assert abs(frame["emit_dot_shaft"] + 1.0) < 1e-9
    assert abs(frame["emit_dot_tangent"]) < 1e-9
    assert abs(frame["emit_dot_right"]) < 1e-9
    assert float(np.linalg.norm(np.cross(frame["emit"], frame["shaft"]))) < 1e-9
    expected = frame["shaft"] * (10.0 * MM_TO_SIM) + frame["right"] * (5.0 * MM_TO_SIM)
    assert np.allclose(offset, expected, atol=1e-9)


def test_right_and_jet_follow_drill_pose_not_camera():
    tip = np.array([0.3, 0.1, -0.2], dtype=np.float64)
    # 90° about world Z: local +X → +Y, local +Y → -X.
    half = math.sqrt(0.5)
    quat = (0.0, 0.0, half, half)
    frame = outlet_frame(tip, quat, offset_mm=5.0, proximal_mm=10.0, azimuth_deg=0.0)
    assert np.allclose(frame["shaft"], (0.0, 1.0, 0.0), atol=1e-6)
    assert np.allclose(frame["right"], (-1.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(frame["emit"], (0.0, -1.0, 0.0), atol=1e-6)
    assert abs(frame["emit_dot_shaft"] + 1.0) < 1e-9
    assert abs(frame["emit_dot_tangent"]) < 1e-9
    assert float(np.linalg.norm(np.cross(frame["emit"], frame["shaft"]))) < 1e-9
    offset = frame["outlet"] - tip
    assert abs(np.dot(offset, frame["shaft"]) / MM_TO_SIM - 10.0) < 1e-6
    assert abs(np.dot(offset, frame["right"]) / MM_TO_SIM - 5.0) < 1e-6
    # Camera screen-right would stay world +X; the 5 mm side offset is drill -X.
    camera_right = np.array((1.0, 0.0, 0.0))
    side = offset - frame["shaft"] * float(np.dot(offset, frame["shaft"]))
    assert abs(np.dot(side / np.linalg.norm(side), camera_right) + 1.0) < 1e-6


def test_displacement_matches_speed():
    tip = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    stream = PrescribedStream(cc_per_s=1.0, gravity=False, max_particles=64)
    stream.step(0.001, tip, quat, True)
    assert stream.compact() == 1
    start = stream.packed.numpy()[0].copy()
    dt = 0.05
    stream.step(dt, tip, quat, False)
    assert stream.compact() == 1
    moved = stream.packed.numpy()[0] - start
    expected = stream.last_frame["emit"] * stream.speed * dt
    assert np.allclose(moved, expected, atol=1e-5)
    assert abs(np.linalg.norm(moved) / MM_TO_SIM - jet_speed_mm_s(1.0) * dt) < 1e-3


def test_stagger_fills_the_frame_gap():
    tip = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    stream = PrescribedStream(cc_per_s=1.0, gravity=False, max_particles=64)
    dt = 0.016
    stream.step(dt, tip, quat, True)
    n = stream.compact()
    assert n == 16
    frame = stream.last_frame
    along = (stream.packed.numpy()[:n] - frame["outlet"]) @ frame["emit"]
    along.sort()
    span = float(along[-1] - along[0])
    expected_span = stream.speed * dt * (n - 1) / n
    assert abs(span - expected_span) / max(stream.speed, 1e-9) < 2e-3
    gaps = np.diff(along)
    assert np.allclose(gaps, gaps.mean(), rtol=0.05)
    assert abs(gaps.mean() / MM_TO_SIM - spacing_mm()) < 0.05


def test_idle_does_not_integrate_empty_buffer():
    stream = PrescribedStream(cc_per_s=1.0, max_particles=64)
    tip = np.zeros(3)
    stream.step(0.016, tip, (0, 0, 0, 1), False)
    assert stream.live == 0
    assert stream.last_n == 0


def main():
    test_rate_table_does_not_treat_60_as_denser()
    test_outlet_is_5mm_drill_right_and_10mm_proximal()
    test_right_and_jet_follow_drill_pose_not_camera()
    test_displacement_matches_speed()
    test_stagger_fills_the_frame_gap()
    test_idle_does_not_integrate_empty_buffer()
    print(
        f"default {CC_PER_S:g} cc/s, {particles_per_second():g} /s, "
        f"spacing {spacing_mm():g} mm, right {OFFSET_MM:g} mm, "
        f"proximal {PROXIMAL_MM:g} mm",
        flush=True,
    )
    for row in rate_rows():
        print(
            f"  {row['cc_per_s']:g} cc/s  {row['particles_per_s']:.0f}/s  "
            f"{row['speed_mm_s']:.0f} mm/s  travel/frame {row['travel_per_16ms_mm']:.1f} mm",
            flush=True,
        )


if __name__ == "__main__":
    main()
