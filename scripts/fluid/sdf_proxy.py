"""Coarse occupancy SDF used only as an SPH collision proxy.

The AMBF voxel grid stays authoritative for drilling. This object may drop
cavities smaller than one proxy cell. It never writes the rendered bone mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .maelstrom_boundary import (
    FILTER_HALO,
    dirty_write_box,
    memcpy3d_args,
    pack_texels,
    recommended_band,
)


@dataclass(frozen=True)
class TexturePatch:
    """Host slab that could replace a GPU texture region.

    origin / extent are in proxy-texel index space. host_zyx matches
    generation::cudaVolume (Z, Y, X, 4) float32.
    """

    origin: tuple[int, int, int]
    extent: tuple[int, int, int]
    host_zyx: np.ndarray
    write_lo: tuple[int, int, int]
    write_hi: tuple[int, int, int]
    coarse_changed: int
    write_voxels: int
    small_cavity_ignored: int

    @property
    def bytes(self) -> int:
        return int(self.host_zyx.nbytes)

    def cuda_args(self) -> dict:
        return memcpy3d_args(self.origin, self.extent, self.host_zyx)


class FluidCollisionProxy:
    """Downsampled occupancy plus a truncated signed distance.

    A coarse cell stays solid while any fine AMBF voxel inside it is occupied.
    Fluid therefore cannot enter a hole until that whole proxy cell is empty.
    """

    def __init__(
        self,
        occupied: np.ndarray,
        downsample: int = 4,
        band: int | None = None,
        particle_radius: float | None = None,
        dimensions=(1.0, 1.0, 1.0),
    ):
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
        occupied = np.asarray(occupied, dtype=bool)
        if occupied.ndim != 3:
            raise ValueError("occupied must be (x, y, z)")
        self.fine = np.ascontiguousarray(occupied)
        self.downsample = int(downsample)
        self.dimensions = np.asarray(dimensions, dtype=np.float64)
        self.fine_shape = tuple(int(v) for v in self.fine.shape)
        self.spacing = self.dimensions / np.asarray(self.fine_shape, dtype=np.float64)
        self.proxy_spacing = float(np.max(self.spacing) * self.downsample)
        if band is None:
            radius = 0.002 if particle_radius is None else float(particle_radius)
            band = recommended_band(self.proxy_spacing, radius)
        self.band = int(band)
        self.coarse, self.counts = _pool(self.fine, self.downsample)
        self.distance, self.normals = _signed_distance(self.coarse, self.band)
        self.generation = 0

    @property
    def shape(self):
        return tuple(int(v) for v in self.coarse.shape)

    def clear_voxels(self, xs, ys, zs) -> TexturePatch | None:
        """Apply published AMBF removals. Returns a GPU patch if the proxy changed."""
        xs = np.asarray(xs, dtype=np.int32).reshape(-1)
        ys = np.asarray(ys, dtype=np.int32).reshape(-1)
        zs = np.asarray(zs, dtype=np.int32).reshape(-1)
        nx, ny, nz = self.fine_shape
        keep = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny) & (zs >= 0) & (zs < nz)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        if xs.size == 0:
            return None
        was = self.fine[xs, ys, zs]
        if not np.any(was):
            return None
        xs, ys, zs = xs[was], ys[was], zs[was]
        self.fine[xs, ys, zs] = False
        cx = xs // self.downsample
        cy = ys // self.downsample
        cz = zs // self.downsample
        cells, inverse = np.unique(np.stack((cx, cy, cz), axis=1), axis=0, return_inverse=True)
        ignored = 0
        changed = []
        for index, (i, j, k) in enumerate(cells):
            dropped = int(np.count_nonzero(inverse == index))
            self.counts[i, j, k] -= dropped
            if self.counts[i, j, k] < 0:
                self.counts[i, j, k] = 0
            now_solid = self.counts[i, j, k] > 0
            if now_solid:
                ignored += dropped
            if self.coarse[i, j, k] != now_solid:
                self.coarse[i, j, k] = now_solid
                changed.append((int(i), int(j), int(k)))
        if not changed:
            return TexturePatch(
                origin=(0, 0, 0),
                extent=(0, 0, 0),
                host_zyx=np.zeros((0, 0, 0, 4), np.float32),
                write_lo=(0, 0, 0),
                write_hi=(0, 0, 0),
                coarse_changed=0,
                write_voxels=0,
                small_cavity_ignored=ignored,
            )
        changed = np.asarray(changed, dtype=np.int32)
        lo = tuple(int(v) for v in changed.min(axis=0))
        hi = tuple(int(v) for v in changed.max(axis=0))
        return self._rebuild_box(lo, hi, ignored, flipped=len(changed))

    def rebuild(self) -> TexturePatch:
        """Full proxy bake. Used for the first upload and for tests."""
        self.distance, self.normals = _signed_distance(self.coarse, self.band)
        self.generation += 1
        packed = pack_texels(self.normals, self.distance)
        shape = self.shape
        voxels = int(self.coarse.size)
        return TexturePatch(
            origin=(0, 0, 0),
            extent=shape,
            host_zyx=packed,
            write_lo=(0, 0, 0),
            write_hi=(shape[0] - 1, shape[1] - 1, shape[2] - 1),
            coarse_changed=voxels,
            write_voxels=voxels,
            small_cavity_ignored=0,
        )

    def sample_index(self, xyz: np.ndarray) -> np.ndarray:
        """Trilinear SDF at continuous fine-index coordinates."""
        scale = float(self.downsample)
        q = np.asarray(xyz, dtype=np.float64) / scale
        return _sample_field(self.distance, q)

    def _rebuild_box(self, lo, hi, ignored: int, flipped: int) -> TexturePatch:
        write, read = dirty_write_box(lo, hi, self.band, self.shape)
        slab = self.coarse[
            read[0][0] : read[1][0] + 1,
            read[0][1] : read[1][1] + 1,
            read[0][2] : read[1][2] + 1,
        ]
        distance, normals = _signed_distance(slab, self.band)
        wlo, whi = write
        rlo = read[0]
        sl = (
            slice(wlo[0] - rlo[0], whi[0] - rlo[0] + 1),
            slice(wlo[1] - rlo[1], whi[1] - rlo[1] + 1),
            slice(wlo[2] - rlo[2], whi[2] - rlo[2] + 1),
        )
        self.distance[wlo[0] : whi[0] + 1, wlo[1] : whi[1] + 1, wlo[2] : whi[2] + 1] = distance[sl]
        self.normals[wlo[0] : whi[0] + 1, wlo[1] : whi[1] + 1, wlo[2] : whi[2] + 1] = normals[sl]
        self.generation += 1
        packed = pack_texels(normals[sl], distance[sl])
        extent = (
            whi[0] - wlo[0] + 1,
            whi[1] - wlo[1] + 1,
            whi[2] - wlo[2] + 1,
        )
        return TexturePatch(
            origin=wlo,
            extent=extent,
            host_zyx=packed,
            write_lo=wlo,
            write_hi=whi,
            coarse_changed=int(flipped),
            write_voxels=int(np.prod(extent)),
            small_cavity_ignored=ignored,
        )


def _pool(fine: np.ndarray, factor: int):
    nx, ny, nz = fine.shape
    cx = (nx + factor - 1) // factor
    cy = (ny + factor - 1) // factor
    cz = (nz + factor - 1) // factor
    padded = np.zeros((cx * factor, cy * factor, cz * factor), dtype=bool)
    padded[:nx, :ny, :nz] = fine
    grouped = padded.reshape(cx, factor, cy, factor, cz, factor)
    counts = grouped.sum(axis=(1, 3, 5)).astype(np.int32)
    return counts > 0, counts


def _signed_distance(solid: np.ndarray, band: int):
    """Positive outside, negative inside, clamped to ±band proxy voxels."""
    if solid.size == 0:
        empty = np.zeros(solid.shape, np.float32)
        return empty, np.zeros(solid.shape + (3,), np.float32)
    if np.all(solid):
        distance = np.full(solid.shape, -float(band), np.float32)
    elif not np.any(solid):
        distance = np.full(solid.shape, float(band), np.float32)
    else:
        outside = ndimage.distance_transform_edt(~solid)
        inside = ndimage.distance_transform_edt(solid)
        distance = np.clip(outside - inside, -band, band).astype(np.float32)
    gx, gy, gz = np.gradient(distance)
    normals = np.stack((gx, gy, gz), axis=-1)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.maximum(length, 1e-6), dtype=np.float32)
    return distance, normals.astype(np.float32)


def _sample_field(field: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    nx, ny, nz = field.shape
    x = np.clip(points[:, 0], 0.0, nx - 1.0001)
    y = np.clip(points[:, 1], 0.0, ny - 1.0001)
    z = np.clip(points[:, 2], 0.0, nz - 1.0001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    z1 = np.minimum(z0 + 1, nz - 1)
    fx, fy, fz = x - x0, y - y0, z - z0
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
    values = c0 * (1.0 - fz) + c1 * fz
    if np.asarray(xyz).ndim == 1:
        return float(values[0])
    return values
