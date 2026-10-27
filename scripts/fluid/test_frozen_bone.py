"""CPU checks for occupancy → world SDF → Maelstrom kind=2 VDB."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.frozen_bone import (
    KIND_STATIC,
    bake_frozen_bone,
    inspect_vdb,
    volume_world_aabb,
)
from fluid.occupancy import VolumeFrame, identity_frame, rpy_matrix


def _block(shape, lo, hi):
    occupied = np.zeros(shape, dtype=bool)
    occupied[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return occupied


def test_sign_in_identity_frame():
    occupied = _block((32, 32, 32), (8, 8, 8), (24, 24, 24))
    frame = identity_frame(occupied.shape)
    bone = bake_frozen_bone(occupied, frame, target_res=48, particle_radius=0.02)
    center = frame.index_to_world((16.0, 16.0, 16.0))
    outside = frame.index_to_world((1.0, 1.0, 1.0))
    assert bone.sample_world(center) < 0.0
    assert bone.sample_world(outside) > 0.0


def test_rotated_volume_stays_inside():
    occupied = _block((24, 24, 24), (6, 6, 6), (18, 18, 18))
    frame = VolumeFrame(
        np.array([0.2, -0.1, 0.3], dtype=np.float64),
        rpy_matrix(1.57079, 1.4, 0.0),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array(occupied.shape, dtype=np.float64),
    )
    bone = bake_frozen_bone(occupied, frame, target_res=40, particle_radius=0.02)
    inside = frame.index_to_world((12.0, 12.0, 12.0))
    far = frame.index_to_world((-4.0, 12.0, 12.0))
    assert bone.sample_world(inside) < 0.0
    assert bone.sample_world(far) > 0.0
    lo, hi = volume_world_aabb(frame)
    assert np.all(bone.origin < lo)
    assert np.all(bone.world_max > hi)


def test_vdb_is_named_surface_level_set():
    occupied = _block((20, 20, 20), (5, 5, 5), (15, 15, 15))
    bone = bake_frozen_bone(occupied, identity_frame(occupied.shape), target_res=32, particle_radius=0.03)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        vdb = bone.write_vdb(root / "bone_surface.vdb")
        cfg = bone.write_config(root / "config.json")
        info = inspect_vdb(vdb)
        payload = json.loads(cfg.read_text())
        assert info["name"] == "surface"
        assert info["level_set"] == 1
        assert info["active"] > 0
        assert payload["boundary_volumes"]["volume1"]["kind"] == KIND_STATIC
        assert payload["boundary_volumes"]["volume1"]["file"] == "bone_surface.vdb"
        assert (root / "domain.obj").is_file()
        mid = bone.sample_world(0.5 * (bone.origin + bone.world_max))
        assert np.sign(info["mid"]) == np.sign(mid) or abs(info["mid"]) < 1e-6


def main():
    test_sign_in_identity_frame()
    test_rotated_volume_stays_inside()
    test_vdb_is_named_surface_level_set()
    print("frozen bone tests passed", flush=True)


if __name__ == "__main__":
    main()
