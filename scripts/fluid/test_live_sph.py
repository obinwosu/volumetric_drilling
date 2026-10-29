"""CPU checks for the live SPH dump and the domain that must cover the outlet."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.frozen_bone import write_domain_obj
from fluid.irrigation_bench import IRRIGATION_PATH, load_irrigation
from fluid.live_sph import (
    domain_bounds,
    gravity_vector,
    live_irrigation,
    prepare_live_config,
    screen_down,
)
from fluid.moving_inlet import jet_speed_disk_mm_s
from fluid.outlet import MM_TO_SIM
from fluid.sph_dump import read_positions, write_positions


def test_dump_roundtrip():
    xyz = np.array([[-4.96, 0.09, 0.01], [-5.01, 0.08, 0.00]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sph_positions.bin"
        write_positions(path, xyz, radius=0.0025, generation=7, flags=1)
        got = read_positions(path)
        assert got is not None
        assert got["generation"] == 7
        assert got["count"] == 2
        assert got["flags"] == 1
        assert abs(got["radius"] - 0.0025) < 1e-6
        assert np.allclose(got["xyz"], xyz)


def test_empty_dump_is_valid():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sph_positions.bin"
        write_positions(path, np.zeros((0, 3), np.float32), radius=0.005, generation=1)
        got = read_positions(path)
        assert got["count"] == 0
        assert got["xyz"].shape == (0, 3)


def test_domain_covers_outlet_outside_bone():
    origin = np.array([-6.65, -0.53, -0.60], dtype=np.float64)
    world_max = np.array([-5.43, 0.53, 0.60], dtype=np.float64)
    outlet = np.array([-4.9586, 0.0895, 0.0085], dtype=np.float64)
    lo, hi = domain_bounds(origin, world_max, [outlet], pad_mm=80.0)
    pad = 80.0 * MM_TO_SIM
    assert lo[0] < outlet[0] < hi[0]
    assert lo[1] < outlet[1] < hi[1]
    assert lo[2] < outlet[2] < hi[2]
    assert hi[0] >= outlet[0] + pad * 0.5
    assert lo[0] <= origin[0]


def test_live_nozzle_follows_the_tuner_file():
    settings = live_irrigation()
    saved = load_irrigation(IRRIGATION_PATH)
    assert settings["jet_mm"] == saved["jet_mm"]
    assert settings["cc_per_s"] == saved["cc_per_s"]
    assert settings["particle_mm"] == saved["particle_mm"]
    assert settings["jet_mm"] > 0.0
    speed = jet_speed_disk_mm_s(settings["cc_per_s"], settings["jet_mm"])
    assert speed > 0.0


def test_prepare_live_config_writes_gravity_and_domain(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        origin = [-6.65, -0.53, -0.60]
        world_max = [-5.43, 0.53, 0.60]
        (root / "report.json").write_text(
            json.dumps({"origin": origin, "world_max": world_max}) + "\n"
        )
        (root / "config.json").write_text("{}\n")
        (root / "bone_surface.vdb").write_bytes(b"")
        write_domain_obj(root / "domain.obj", origin, world_max, pad=0.0)
        outlet = [-4.9586, 0.0895, 0.0085]
        prepared = prepare_live_config(root, outlet=outlet, particle_mm=0.5, jet_mm=2.0, numptcls=40000)
        payload = json.loads((root / "config.json").read_text())
        assert prepared["origin"][0] < outlet[0] < prepared["world_max"][0]
        assert payload["simulation_settings"]["numptcls"] == 40000
        force = np.array(
            [float(v) for v in payload["simulation_settings"]["external_force"].split()[:3]],
            dtype=np.float64,
        )
        down = screen_down()
        assert np.dot(force, down) > 90.0
        assert abs(float(np.linalg.norm(force)) - float(np.linalg.norm(gravity_vector()[:3]))) < 1e-3
        assert payload["simulation_settings"]["timestep_max"] <= 0.00002
        assert payload["modules"]["volumeInlets"] is True
        assert payload["modules"]["pressure"] == "DFSPH"
        assert payload["modules"]["resorting"] == "compactMLM"
        assert payload["modules"]["neighborhood"] == "constrained"
        assert payload["modules"]["density"] == "standard"
        assert payload["modules"]["xsph"] is False
        assert payload["modules"]["rayTracing"] is False
        assert payload["modules"]["error_checking"] is True
        assert payload["modules"]["surfaceDetection"] is False
        assert payload["adaptive"]["delay"] == 0.0
        assert payload["boundary_volumes"]["volume1"]["kind"] == 2


def main():
    test_dump_roundtrip()
    test_empty_dump_is_valid()
    test_domain_covers_outlet_outside_bone()
    test_live_nozzle_follows_the_tuner_file()
    test_prepare_live_config_writes_gravity_and_domain()
    print("live sph tests passed", flush=True)


if __name__ == "__main__":
    main()
