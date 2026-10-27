"""CPU checks for the fluid collision proxy. The bone renderer is not involved."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fluid.maelstrom_boundary import (
    FILTER_HALO,
    FACTS,
    TEXTURE_RES,
    dirty_write_box,
    influence_radius,
    recommended_band,
)
from fluid.sdf_proxy import FluidCollisionProxy


def _block(shape, lo, hi):
    occupied = np.zeros(shape, dtype=bool)
    occupied[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return occupied


def test_facts_do_not_claim_runtime_patches():
    assert "empty stub" in FACTS.runtime_update
    assert "not stored" in FACTS.array_pointer.lower() or "discards" in FACTS.array_pointer
    assert TEXTURE_RES == 256


def test_band_covers_kernel_and_filter():
    spacing = 0.01
    radius = 0.002
    band = recommended_band(spacing, radius)
    assert band >= FILTER_HALO + 1
    assert band * spacing >= influence_radius(radius)


def test_sign_inside_and_outside():
    occupied = _block((32, 32, 32), (8, 8, 8), (24, 24, 24))
    proxy = FluidCollisionProxy(occupied, downsample=2, band=4)
    assert proxy.sample_index(np.array([16.0, 16.0, 16.0])) < 0.0
    assert proxy.sample_index(np.array([1.0, 1.0, 1.0])) > 0.0


def test_small_cavity_stays_solid():
    occupied = _block((32, 32, 32), (4, 4, 4), (28, 28, 28))
    proxy = FluidCollisionProxy(occupied, downsample=4, band=3)
    before = proxy.sample_index(np.array([16.0, 16.0, 16.0]))
    patch = proxy.clear_voxels([16], [16], [16])
    assert patch is not None
    assert patch.coarse_changed == 0
    assert patch.small_cavity_ignored == 1
    assert proxy.sample_index(np.array([16.0, 16.0, 16.0])) == before


def test_cleared_coarse_cell_opens():
    occupied = _block((32, 32, 32), (4, 4, 4), (28, 28, 28))
    proxy = FluidCollisionProxy(occupied, downsample=4, band=3)
    xs, ys, zs = np.mgrid[16:20, 16:20, 16:20]
    patch = proxy.clear_voxels(xs, ys, zs)
    assert patch is not None
    assert patch.coarse_changed > 0
    assert proxy.coarse[4, 4, 4] == False
    assert proxy.distance[4, 4, 4] > 0.0
    # Linear filtering still blends a one-cell hole with solid neighbors.
    # SPH tex3D is the same; a cavity must span several proxy cells to sample open.


def test_wide_cavity_samples_open():
    occupied = _block((48, 48, 48), (4, 4, 4), (44, 44, 44))
    proxy = FluidCollisionProxy(occupied, downsample=2, band=3)
    xs, ys, zs = np.mgrid[18:30, 18:30, 18:30]
    patch = proxy.clear_voxels(xs, ys, zs)
    assert patch is not None and patch.coarse_changed > 0
    assert proxy.sample_index(np.array([24.0, 24.0, 24.0])) > 0.0
    assert patch.bytes < proxy.rebuild().bytes


def test_region_matches_full_rebuild():
    occupied = _block((48, 40, 36), (4, 4, 4), (44, 36, 32))
    proxy = FluidCollisionProxy(occupied, downsample=2, band=3)
    xs, ys, zs = np.mgrid[20:26, 18:24, 16:22]
    patch = proxy.clear_voxels(xs, ys, zs)
    assert patch is not None and patch.coarse_changed > 0
    regional = proxy.distance.copy()
    full = FluidCollisionProxy(proxy.fine, downsample=2, band=3)
    write, _read = dirty_write_box(patch.write_lo, patch.write_hi, 0, proxy.shape)
    sl = (
        slice(write[0][0], write[1][0] + 1),
        slice(write[0][1], write[1][1] + 1),
        slice(write[0][2], write[1][2] + 1),
    )
    delta = np.max(np.abs(regional[sl] - full.distance[sl]))
    if delta > 1e-5:
        raise SystemExit(f"regional SDF differs from full bake by {delta}")
    args = patch.cuda_args()
    assert args["srcBytes"] == patch.bytes
    assert args["requires"].startswith("retained")


def main():
    test_facts_do_not_claim_runtime_patches()
    test_band_covers_kernel_and_filter()
    test_sign_inside_and_outside()
    test_small_cavity_stays_solid()
    test_cleared_coarse_cell_opens()
    test_wide_cavity_samples_open()
    test_region_matches_full_rebuild()
    print("fluid proxy tests passed", flush=True)


if __name__ == "__main__":
    main()
