"""openMaelstrom rigid-volume store, query, and what a region patch would need.

Verified against /tmp/openMaelstrom (wi-re/openMaelstrom). None of the
runtime update options below are implemented there today. This module only
describes the layout and the margins a FIVRS proxy would have to supply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Spline4 Kernel<>::kernel_size() in utility/mathv2/kernels.h.
KERNEL_SIZE = 1.825742
# Neighbor count used by support_from_volume.
KERNEL_NEIGHBORS = 50
# volumeBoundary::distance_fn ignores a sample unless d < 0.995 * h * kernelSize().
INFLUENCE_SCALE = 0.995
# cudaTextureDesc.filterMode = cudaFilterModeLinear. A write must refresh
# one texel on each side or a neighboring sample still sees the old value.
FILTER_HALO = 1
# generation::cudaVolume hardcodes dimension = int4{256,256,256,0}.
TEXTURE_RES = 256
# 256^3 float4. A full replace is this many bytes on the host-to-array copy.
FULL_TEXTURE_BYTES = TEXTURE_RES ** 3 * 16


@dataclass(frozen=True)
class BoundaryFacts:
    """What the solver actually does, not what a mesh pipeline would need."""

    stores: str
    queries: str
    runtime_update: str
    voxel_input: str
    array_pointer: str
    region_patch: str


FACTS = BoundaryFacts(
    stores=(
        "One CUDA 3D array per rigid, bound to a cudaTextureObject_t. "
        "Texel is float4 (nx, ny, nz, signedDistance). Negative w is inside."
    ),
    queries=(
        "volumeDistanceFni: inverse rigid transform, AABB test, then tex3D "
        "at normalized (p-min)/(max-min). Linear filter, clamp, normalized coords."
    ),
    runtime_update=(
        "updateVolumeBoundary copies pose and force only, then steps Bullet. "
        "SPH::volume::update() is an empty stub. animationPath is unused."
    ),
    voxel_input=(
        "fileToVDB accepts .vdb / .obj / .ply. Occupancy, NRRD, and PNG stacks "
        "are not inputs. A mesh is converted to an SDF, never used as a mesh."
    ),
    array_pointer=(
        "cudaVolume discards the cudaArray* after cudaCreateTextureObject. "
        "volumeBoundaryVolumes stores only the texture handle. A patch cannot "
        "be issued until that array pointer is retained."
    ),
    region_patch=(
        "cudaMemcpy3D can write a dstPos/extent into an existing cudaArray. "
        "That path is not wired. Treat regional GPU updates as an extension "
        "to test, not existing support."
    ),
)


def support_from_radius(particle_radius: float) -> float:
    """Maelstrom support_from_volume for a sphere of the given radius."""
    volume = (4.0 / 3.0) * math.pi * particle_radius ** 3
    kernel_epsilon = (KERNEL_NEIGHBORS * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) / KERNEL_SIZE
    return kernel_epsilon * volume ** (1.0 / 3.0)


def influence_radius(particle_radius: float) -> float:
    """World distance at which volumeDistanceFni still uses a texel."""
    return INFLUENCE_SCALE * support_from_radius(particle_radius) * KERNEL_SIZE


def recommended_band(proxy_spacing: float, particle_radius: float) -> int:
    """Truncation half-width in proxy voxels.

    SPH only reads the SDF inside influence_radius. A truncated field that
    stays correct in that band is enough. Add the linear-filter halo so a
    sample whose footprint crosses the dirty edge still matches a full bake.
    """
    if proxy_spacing <= 0.0:
        raise ValueError("proxy_spacing must be positive")
    needed = influence_radius(particle_radius) / proxy_spacing
    return max(1, int(math.ceil(needed)) + FILTER_HALO)


def dirty_write_box(changed_lo, changed_hi, band: int, shape):
    """Inclusive box that must be rewritten after occupancy in changed_* changes.

    Write = changed dilated by (band + FILTER_HALO). Occupancy used to
    recompute the EDT must be that write box dilated by band, so every
    written voxel still sees its nearest surface inside the truncation.
    """
    write = _dilate(changed_lo, changed_hi, band + FILTER_HALO, shape)
    read = _dilate(write[0], write[1], band, shape)
    return write, read


def _dilate(lo, hi, radius: int, shape):
    lo = [max(0, int(v) - radius) for v in lo]
    hi = [min(int(n) - 1, int(v) + radius) for v, n in zip(hi, shape)]
    return tuple(lo), tuple(hi)


def pack_texels(normals: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Maelstrom host layout: index = x + X * (y + Y * z)."""
    if normals.shape[:3] != distance.shape:
        raise ValueError("normals and distance shapes differ")
    packed = np.empty(distance.shape + (4,), np.float32)
    packed[..., :3] = normals
    packed[..., 3] = distance
    return np.ascontiguousarray(np.transpose(packed, (2, 1, 0, 3)))


def unpack_texels(packed: np.ndarray, shape):
    """Inverse of pack_texels. packed is (Z, Y, X, 4) as written for cudaMemcpy3D."""
    xyz = np.transpose(packed, (2, 1, 0, 3))
    if tuple(xyz.shape[:3]) != tuple(shape):
        raise ValueError(f"packed shape {xyz.shape[:3]} != {shape}")
    return xyz[..., :3], xyz[..., 3]


def memcpy3d_args(origin, extent, host_zyx: np.ndarray) -> dict:
    """Arguments a future cudaMemcpy3D into a retained cudaArray would use.

    origin and extent are in texel index space. host_zyx is (dz, dy, dx, 4)
    float32, the same axis order generation::cudaVolume uploads.
    """
    ox, oy, oz = (int(v) for v in origin)
    dx, dy, dz = (int(v) for v in extent)
    if host_zyx.shape != (dz, dy, dx, 4):
        raise ValueError(f"host slab {host_zyx.shape} != {(dz, dy, dx, 4)}")
    return {
        "dstPos": (ox, oy, oz),
        "extent": (dx, dy, dz),
        "srcBytes": int(host_zyx.nbytes),
        "kind": "cudaMemcpyHostToDevice",
        "requires": "retained cudaArray*; not stored by openMaelstrom today",
    }


def sample_normalized(packed_zyx: np.ndarray, u: float, v: float, w: float) -> np.ndarray:
    """CPU stand-in for tex3D linear + clamp + normalized coords."""
    dz, dy, dx = packed_zyx.shape[:3]
    x = np.clip(u * dx - 0.5, 0.0, dx - 1.0001)
    y = np.clip(v * dy - 0.5, 0.0, dy - 1.0001)
    z = np.clip(w * dz - 0.5, 0.0, dz - 1.0001)
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = min(x0 + 1, dx - 1), min(y0 + 1, dy - 1), min(z0 + 1, dz - 1)
    fx, fy, fz = x - x0, y - y0, z - z0

    def at(xi, yi, zi):
        return packed_zyx[zi, yi, xi]

    c00 = at(x0, y0, z0) * (1.0 - fx) + at(x1, y0, z0) * fx
    c01 = at(x0, y0, z1) * (1.0 - fx) + at(x1, y0, z1) * fx
    c10 = at(x0, y1, z0) * (1.0 - fx) + at(x1, y1, z0) * fx
    c11 = at(x0, y1, z1) * (1.0 - fx) + at(x1, y1, z1) * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz
