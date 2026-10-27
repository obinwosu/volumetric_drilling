#!/usr/bin/env python3
"""Measure coarse-SDF bake vs regional patch cost. Does not touch the bone renderer."""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fluid.maelstrom_boundary import FULL_TEXTURE_BYTES, TEXTURE_RES, recommended_band
from fluid.sdf_proxy import FluidCollisionProxy


def _specimen(shape=(256, 256, 128)):
    occupied = np.zeros(shape, dtype=bool)
    occupied[16:-16, 16:-16, 8:-8] = True
    return occupied


def _burr(center, radius):
    rng = [np.arange(c - radius, c + radius + 1) for c in center]
    grid = np.meshgrid(*rng, indexing="ij")
    dist = sum((axis - c) ** 2 for axis, c in zip(grid, center))
    keep = dist <= radius * radius
    return grid[0][keep], grid[1][keep], grid[2][keep]


def _ms(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def main():
    occupied = _specimen()
    downsample = 4
    proxy = FluidCollisionProxy(occupied, downsample=downsample, band=4)
    full, bake_ms = _ms(proxy.rebuild)
    xs, ys, zs = _burr((occupied.shape[0] // 2, occupied.shape[1] // 2, occupied.shape[2] // 2), 6)
    # Fresh proxy so the first cut is the measured one.
    live = FluidCollisionProxy(occupied, downsample=downsample, band=proxy.band)
    patch, patch_ms = _ms(lambda: live.clear_voxels(xs, ys, zs))
    report = {
        "fine_shape": list(occupied.shape),
        "downsample": downsample,
        "proxy_shape": list(proxy.shape),
        "band_voxels": proxy.band,
        "recommended_band_r2mm": recommended_band(proxy.proxy_spacing, 0.002),
        "full_bake_ms": round(bake_ms, 3),
        "full_bytes": full.bytes,
        "maelstrom_full_texture_bytes": FULL_TEXTURE_BYTES,
        "maelstrom_texture_res": TEXTURE_RES,
        "removed_fine_voxels": int(xs.size),
        "patch_ms": None if patch is None else round(patch_ms, 3),
        "patch_bytes": None if patch is None else patch.bytes,
        "patch_extent": None if patch is None else list(patch.extent),
        "coarse_changed": None if patch is None else patch.coarse_changed,
        "write_voxels": None if patch is None else patch.write_voxels,
        "small_cavity_ignored": None if patch is None else patch.small_cavity_ignored,
        "gpu_region_update": "not implemented in openMaelstrom; cudaArray* is discarded",
        "bone_renderer": "unchanged",
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
