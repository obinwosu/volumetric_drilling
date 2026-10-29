"""One water surface from SPH positions.

Particles closer than a cell fuse into a sheet. A lone particle is below the
threshold and contributes no triangles. The grid is coarsened when the cloud
spreads, so a field of specks cannot allocate a huge volume.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from fluid.outlet import MM_TO_SIM

# Fine enough that a 1 mm jet is a few cells across, not a facet the size of the burr.
CELL_MM = 0.35
# Kept so older callers can still pass a cap. The grid grows the cell from
# its volume, so a long thin jet is not coarsened just because it is long.
MAX_AXIS = 96
# After a 3-wide box filter, one particle peaks near 1/27. A line of neighbors
# sits above this and becomes a column.
ISO = 0.055
# Empty cells around the cloud. The filter reaches one cell out, so two cells
# of padding are what leave a real outside for the surface to close against.
PAD = 2
# A sheet or a pool fits. A cloud that fills the whole cap is downsampled so
# one frame cannot spend tens of milliseconds building a noisy shell.
MAX_CELLS = 40 * 40 * 16

_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)
# Cube edges as pairs of corner indices.
_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _smooth_verts(verts: np.ndarray, tris: np.ndarray, rounds: int = 2, amount: float = 0.45) -> np.ndarray:
    """Pull each vertex toward its neighbors so the sheet reads as a stream, not facets."""
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]), axis=0)
    out = verts.astype(np.float64, copy=True)
    acc = np.empty_like(out)
    count = np.empty(len(out), dtype=np.float64)
    for _ in range(rounds):
        acc.fill(0.0)
        count.fill(0.0)
        np.add.at(acc, edges[:, 0], out[edges[:, 1]])
        np.add.at(count, edges[:, 0], 1.0)
        np.add.at(acc, edges[:, 1], out[edges[:, 0]])
        np.add.at(count, edges[:, 1], 1.0)
        lonely = count < 1.0
        count[lonely] = 1.0
        acc[lonely] = out[lonely]
        out = out + amount * (acc / count[:, None] - out)
    return out.astype(np.float32)


def _normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts)
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    face = np.cross(b - a, c - a)
    for corner in range(3):
        np.add.at(normals, tris[:, corner], face)
    length = np.linalg.norm(normals, axis=1)
    length[length < 1e-12] = 1.0
    return normals / length[:, None]


def _edge_quads(solid, ids, origin, cell):
    """Quads around every grid edge whose density crosses the surface."""
    nx, ny, nz = solid.shape
    batches = []
    outsides = []

    def take(cross, a, b, c, d, outside):
        if not np.any(cross):
            return
        quads = np.stack((a[cross], b[cross], c[cross], d[cross]), axis=1)
        keep = np.all(quads >= 0, axis=1)
        if not np.any(keep):
            return
        batches.append(quads[keep])
        outsides.append(outside[cross][keep])

    if nx >= 2 and ny >= 3 and nz >= 3:
        s0 = solid[:-1, 1:-1, 1:-1]
        s1 = solid[1:, 1:-1, 1:-1]
        cross = s0 != s1
        p = np.arange(nx - 1)[:, None, None]
        q = np.arange(ny - 2)[None, :, None]
        r = np.arange(nz - 2)[None, None, :]
        outside = np.stack(
            (
                origin[0] + (p + s0) * cell,
                np.broadcast_to(origin[1] + (q + 1) * cell, s0.shape),
                np.broadcast_to(origin[2] + (r + 1) * cell, s0.shape),
            ),
            axis=-1,
        )
        take(
            cross,
            ids[:, :-1, :-1],
            ids[:, 1:, :-1],
            ids[:, 1:, 1:],
            ids[:, :-1, 1:],
            outside,
        )
    if ny >= 2 and nx >= 3 and nz >= 3:
        s0 = solid[1:-1, :-1, 1:-1]
        s1 = solid[1:-1, 1:, 1:-1]
        cross = s0 != s1
        p = np.arange(nx - 2)[:, None, None]
        q = np.arange(ny - 1)[None, :, None]
        r = np.arange(nz - 2)[None, None, :]
        outside = np.stack(
            (
                np.broadcast_to(origin[0] + (p + 1) * cell, s0.shape),
                origin[1] + (q + s0) * cell,
                np.broadcast_to(origin[2] + (r + 1) * cell, s0.shape),
            ),
            axis=-1,
        )
        take(
            cross,
            ids[:-1, :, :-1],
            ids[1:, :, :-1],
            ids[1:, :, 1:],
            ids[:-1, :, 1:],
            outside,
        )
    if nz >= 2 and nx >= 3 and ny >= 3:
        s0 = solid[1:-1, 1:-1, :-1]
        s1 = solid[1:-1, 1:-1, 1:]
        cross = s0 != s1
        p = np.arange(nx - 2)[:, None, None]
        q = np.arange(ny - 2)[None, :, None]
        r = np.arange(nz - 1)[None, None, :]
        outside = np.stack(
            (
                np.broadcast_to(origin[0] + (p + 1) * cell, s0.shape),
                np.broadcast_to(origin[1] + (q + 1) * cell, s0.shape),
                origin[2] + (r + s0) * cell,
            ),
            axis=-1,
        )
        take(
            cross,
            ids[:-1, :-1, :],
            ids[1:, :-1, :],
            ids[1:, 1:, :],
            ids[:-1, 1:, :],
            outside,
        )
    if not batches:
        return None
    return np.concatenate(batches, axis=0), np.concatenate(outsides, axis=0)


def _surface(smooth, origin, cell, iso):
    solid = smooth >= iso
    nx, ny, nz = solid.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    if min(cx, cy, cz) < 1:
        return None
    corners = [solid[ox : ox + cx, oy : oy + cy, oz : oz + cz] for ox, oy, oz in _CORNERS]
    values = [smooth[ox : ox + cx, oy : oy + cy, oz : oz + cz] for ox, oy, oz in _CORNERS]
    boundary = np.zeros((cx, cy, cz), dtype=bool)
    for corner in corners[1:]:
        boundary |= corner != corners[0]
    if not np.any(boundary):
        return None

    acc = np.zeros((cx, cy, cz, 3), dtype=np.float64)
    count = np.zeros((cx, cy, cz), dtype=np.float64)
    ii = np.arange(cx, dtype=np.float64)[:, None, None]
    jj = np.arange(cy, dtype=np.float64)[None, :, None]
    kk = np.arange(cz, dtype=np.float64)[None, None, :]
    for a, b in _EDGES:
        va = values[a]
        vb = values[b]
        cross = corners[a] != corners[b]
        if not np.any(cross):
            continue
        alpha = np.clip((iso - va) / np.where(cross, vb - va, 1.0), 0.0, 1.0)
        ca = _CORNERS[a]
        cb = _CORNERS[b]
        acc[..., 0] += np.where(cross, origin[0] + (ii + ca[0] + alpha * (cb[0] - ca[0])) * cell, 0.0)
        acc[..., 1] += np.where(cross, origin[1] + (jj + ca[1] + alpha * (cb[1] - ca[1])) * cell, 0.0)
        acc[..., 2] += np.where(cross, origin[2] + (kk + ca[2] + alpha * (cb[2] - ca[2])) * cell, 0.0)
        count += cross
    count[count == 0.0] = 1.0
    grid_pos = acc / count[..., None]
    ids = np.full((cx, cy, cz), -1, dtype=np.int32)
    ids[boundary] = np.arange(int(boundary.sum()), dtype=np.int32)
    gathered = _edge_quads(solid, ids, origin, cell)
    if gathered is None:
        return None
    quads, outside = gathered
    verts = grid_pos[boundary].astype(np.float32)
    tris = np.concatenate(
        (
            np.stack((quads[:, 0], quads[:, 1], quads[:, 2]), axis=1),
            np.stack((quads[:, 0], quads[:, 2], quads[:, 3]), axis=1),
        ),
        axis=0,
    )
    toward = np.concatenate((outside, outside), axis=0)
    corners = verts[tris]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.einsum("ij,ij->i", normal, toward - corners.mean(axis=1)) < 0.0
    if np.any(flip):
        tris = tris.copy()
        tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()
    area = np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)
    tris = tris[area > 1e-16]
    if len(tris) == 0:
        return None
    used = np.unique(tris)
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    verts = verts[used]
    tris = remap[tris]
    return verts, tris.astype(np.int32)


# Half-width of the drawn stream. The nozzle is 1 mm, so particles farther
# sideways than this are splash and are not part of the column or the ribbon.
RIBBON_HALF_MM = 0.55


def ribbon_particles(xyz, origin, emit, down, half_mm: float = RIBBON_HALF_MM):
    """Keep the jet and the downhill ribbon. Drop the scatter off to the side."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    emit = np.asarray(emit, dtype=np.float64).reshape(3)
    down = np.asarray(down, dtype=np.float64).reshape(3)
    emit_n = float(np.linalg.norm(emit))
    if emit_n < 1e-8:
        return points
    emit = emit / emit_n
    side = np.cross(emit, down)
    side_n = float(np.linalg.norm(side))
    if side_n < 1e-8:
        return points
    side = side / side_n
    lateral = (points - origin) @ side
    return points[np.abs(lateral) <= float(half_mm) * MM_TO_SIM]


def water_surface(xyz, cell_mm: float = CELL_MM, max_axis: int = MAX_AXIS, iso: float = ISO):
    """Return verts, tris, normals. None when nothing fuses."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return None
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = np.maximum(hi - lo, 1e-8)
    cell = float(cell_mm) * MM_TO_SIM
    # A long thin jet keeps the fine cell. The cell grows only when the whole
    # grid would be heavy, which is a cloud spread across the field.
    origin = lo
    dims = np.array([3, 3, 3], dtype=np.int32)
    for _ in range(6):
        origin = lo - PAD * cell
        last = np.floor((hi - origin) / cell).astype(np.int32)
        dims = np.maximum(last + 1 + PAD, PAD * 2 + 1)
        if int(np.prod(dims.astype(np.int64))) <= MAX_CELLS and int(dims.max()) <= max_axis:
            break
        cell *= 1.5
    ijk = np.floor((points - origin) / cell).astype(np.int32)
    ijk = np.clip(ijk, 0, dims - 1)
    field = np.zeros(tuple(int(v) for v in dims), dtype=np.float32)
    np.add.at(field, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
    smooth = uniform_filter(field, size=3, mode="constant", cval=0.0)
    while int(np.prod(smooth.shape)) > MAX_CELLS and min(smooth.shape) >= 6:
        smooth = smooth[::2, ::2, ::2]
        cell *= 2.0
    if float(smooth.max()) < iso:
        return None
    extracted = _surface(smooth, origin, cell, iso)
    if extracted is None:
        return None
    verts, tris = extracted
    verts = _smooth_verts(verts, tris)
    return {
        "verts": verts,
        "tris": tris,
        "normals": _normals(verts, tris).astype(np.float32),
        "cells": tuple(int(v) for v in smooth.shape),
        "cell": float(cell),
    }
