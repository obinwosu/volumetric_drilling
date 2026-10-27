"""Seeds follow the constrained tip. Pedal off writes an empty inlet."""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.moving_inlet import (
    attach_inlet_config,
    inlet_from_pose,
    read_state,
    write_inlet_vdb,
    write_state,
)
from fluid.outlet import MM_TO_SIM, outlet_frame


def test_seeds_sit_on_outlet_disk_and_jet_is_minus_shaft():
    tip = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    inlet = inlet_from_pose(tip, quat, emitting=True, particle_mm=0.5, jet_mm=1.0)
    expected = outlet_frame(tip, quat)
    assert inlet.emitting
    assert inlet.count > 1
    assert abs(inlet.frame["proximal_mm"] - 10.0) < 1e-6
    assert abs(inlet.frame["radial_mm"] - 5.0) < 1e-6
    assert abs(inlet.frame["emit_dot_shaft"] + 1.0) < 1e-9
    assert abs(inlet.frame["emit_dot_tangent"]) < 1e-9
    assert np.allclose(inlet.velocity / np.linalg.norm(inlet.velocity), expected["emit"])
    for seed in inlet.seeds:
        delta = seed - expected["outlet"]
        assert abs(np.dot(delta, expected["emit"])) < 1e-9
        radial = math.sqrt(float(np.dot(delta, expected["right"])) ** 2 + float(np.dot(delta, expected["tangent"])) ** 2)
        assert radial / MM_TO_SIM <= 0.6


def test_pedal_off_writes_zero_seeds():
    tip = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    inlet = inlet_from_pose(tip, quat, emitting=False)
    assert inlet.count == 0
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "inlet_state.bin"
        write_state(path, inlet)
        state = read_state(path)
        assert state["emitting"] is False
        assert state["count"] == 0
        assert state["seeds"].shape == (0, 3)


def test_seeds_track_tip_pose():
    tip_a = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    tip_b = tip_a + np.array([0.03, -0.02, 0.01], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    a = inlet_from_pose(tip_a, quat, emitting=True)
    b = inlet_from_pose(tip_b, quat, emitting=True)
    assert np.allclose(b.seeds, a.seeds + (tip_b - tip_a))
    half = math.sqrt(0.5)
    rotated = inlet_from_pose(tip_a, (0.0, 0.0, half, half), emitting=True)
    assert not np.allclose(rotated.seeds, a.seeds)
    assert abs(rotated.frame["proximal_mm"] - 10.0) < 1e-6
    assert abs(rotated.frame["radial_mm"] - 5.0) < 1e-6
    assert np.allclose(rotated.frame["emit"], (0.0, -1.0, 0.0), atol=1e-6)


def test_inlet_vdb_is_negative_at_outlet():
    tip = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    inlet = inlet_from_pose(tip, quat, emitting=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        vdb = write_inlet_vdb(root / "inlet_surface.vdb", inlet)
        write_state(root / "inlet_state.bin", inlet)
        attach_inlet_config(root / "config.json", inlet)
        from fluid.frozen_bone import ensure_writer, inspect_vdb
        import subprocess

        writer = ensure_writer()
        outlet = inlet.frame["outlet"]
        far = outlet + np.array([0.05, 0.0, 0.0])
        inside = float(
            subprocess.check_output(
                [str(writer), "sample", str(vdb), *(f"{c:.8f}" for c in outlet)],
                text=True,
            ).strip()
        )
        outside = float(
            subprocess.check_output(
                [str(writer), "sample", str(vdb), *(f"{c:.8f}" for c in far)],
                text=True,
            ).strip()
        )
        info = inspect_vdb(vdb)
        payload = (root / "config.json").read_text()
        assert inside < 0.0
        assert outside > 0.0
        assert info["name"] == "surface"
        assert "volumeInlets" in payload


def main():
    test_seeds_sit_on_outlet_disk_and_jet_is_minus_shaft()
    test_pedal_off_writes_zero_seeds()
    test_seeds_track_tip_pose()
    test_inlet_vdb_is_negative_at_outlet()
    print("moving inlet tests passed", flush=True)


if __name__ == "__main__":
    main()
