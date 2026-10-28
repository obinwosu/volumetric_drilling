"""Dirty-block marching cubes, smoothing, and normals on the GPU.

The occupancy mirror stays on the device. A cut uploads only the cleared
voxels. Extraction matches the CPU midpoint tables, including the empty
crop border. Smoothing reads a halo of the neighboring block and does not
pin the shared face, so both sides meet. Timing calls synchronize, so the
recorded milliseconds are finished GPU work.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from mc_tables import EDGE_TABLE, TRI_TABLE
from surface import BLOCK, SMOOTH_ITERS, _owns_cell, block_halo_bounds, smooth_pinned

# 1: GPU extraction, CPU smooth and prepare. Measured, and slower once per-block
# synchronizations were included.
# 3: extraction, smoothing, world positions, and normals stay on the device
# until one synchronization for the whole dirty set.
STAGE = 3
SLOT_COUNT = 8
MAX_AXIS = 40
MAX_VERTS = 3 * MAX_AXIS * MAX_AXIS * MAX_AXIS
MAX_CELLS = MAX_AXIS * MAX_AXIS * MAX_AXIS
MAX_DEGREE = 32

_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int32,
)
_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


@wp.func
def sample_occ(occ: wp.array(dtype=wp.uint8), nx: int, ny: int, nz: int, x: int, y: int, z: int) -> int:
    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
        return 0
    return int(occ[((x * ny) + y) * nz + z])


@wp.func
def sample_color(colors: wp.array(dtype=wp.uint32), nx: int, ny: int, nz: int, x: int, y: int, z: int) -> wp.uint32:
    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
        return wp.uint32(0)
    return colors[((x * ny) + y) * nz + z]


@wp.kernel
def clear_voxels(
    occ: wp.array(dtype=wp.uint8),
    colors: wp.array(dtype=wp.uint32),
    xs: wp.array(dtype=wp.int32),
    ys: wp.array(dtype=wp.int32),
    zs: wp.array(dtype=wp.int32),
    ny: int,
    nz: int,
):
    i = wp.tid()
    index = ((int(xs[i]) * ny) + int(ys[i])) * nz + int(zs[i])
    occ[index] = wp.uint8(0)
    colors[index] = wp.uint32(0)


@wp.kernel
def mark_edges(
    occ: wp.array(dtype=wp.uint8),
    corners: wp.array(dtype=wp.int32),
    edges: wp.array(dtype=wp.int32),
    edge_table: wp.array(dtype=wp.int32),
    flags: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: int,
    origin_y: int,
    origin_z: int,
    sx: int,
    sy: int,
    sz: int,
    own_hi_x: int,
    own_hi_y: int,
    own_hi_z: int,
):
    ix, iy, iz = wp.tid()
    cx = origin_x + ix
    cy = origin_y + iy
    cz = origin_z + iz
    if cx >= own_hi_x or cy >= own_hi_y or cz >= own_hi_z:
        return
    case = int(0)
    for corner in range(8):
        vx = cx + int(corners[corner * 3 + 0])
        vy = cy + int(corners[corner * 3 + 1])
        vz = cz + int(corners[corner * 3 + 2])
        if sample_occ(occ, nx, ny, nz, vx, vy, vz) != 0:
            case = case | (int(1) << corner)
    if case == 0 or case == 255:
        return
    mask = int(edge_table[case])
    plane = sx * sy * sz
    row = sy * sz
    for edge in range(12):
        if (mask & (int(1) << edge)) == 0:
            continue
        a = int(edges[edge * 2 + 0])
        b = int(edges[edge * 2 + 1])
        ax = int(corners[a * 3 + 0])
        ay = int(corners[a * 3 + 1])
        az = int(corners[a * 3 + 2])
        bx = int(corners[b * 3 + 0])
        by = int(corners[b * 3 + 1])
        bz = int(corners[b * 3 + 2])
        ox = ax
        oy = ay
        oz = az
        axis = int(0)
        if bx != ax:
            axis = int(0)
            if bx < ax:
                ox = bx
        elif by != ay:
            axis = int(1)
            if by < ay:
                oy = by
        else:
            axis = int(2)
            if bz < az:
                oz = bz
        slot = axis * plane + (ix + ox) * row + (iy + oy) * sz + (iz + oz)
        flags[slot] = 1


@wp.kernel
def emit_vertices(
    occ: wp.array(dtype=wp.uint8),
    colors: wp.array(dtype=wp.uint32),
    flags: wp.array(dtype=wp.int32),
    scan: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    point_colors: wp.array(dtype=wp.uint32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: int,
    origin_y: int,
    origin_z: int,
    sx: int,
    sy: int,
    sz: int,
):
    slot = wp.tid()
    if int(flags[slot]) == 0:
        return
    vid = int(scan[slot]) - 1
    plane = sx * sy * sz
    axis = slot // plane
    rem = slot - axis * plane
    row = sy * sz
    x = rem // row
    rem = rem - x * row
    y = rem // sz
    z = rem - y * sz
    wx = origin_x + x
    wy = origin_y + y
    wz = origin_z + z
    pos = wp.vec3(float(wx), float(wy), float(wz))
    if axis == 0:
        pos[0] = pos[0] + 0.5
    elif axis == 1:
        pos[1] = pos[1] + 0.5
    else:
        pos[2] = pos[2] + 0.5
    points[vid] = pos
    if sample_occ(occ, nx, ny, nz, wx, wy, wz) != 0:
        point_colors[vid] = sample_color(colors, nx, ny, nz, wx, wy, wz)
    elif axis == 0:
        point_colors[vid] = sample_color(colors, nx, ny, nz, wx + 1, wy, wz)
    elif axis == 1:
        point_colors[vid] = sample_color(colors, nx, ny, nz, wx, wy + 1, wz)
    else:
        point_colors[vid] = sample_color(colors, nx, ny, nz, wx, wy, wz + 1)


@wp.kernel
def count_triangles(
    occ: wp.array(dtype=wp.uint8),
    corners: wp.array(dtype=wp.int32),
    tri_table: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: int,
    origin_y: int,
    origin_z: int,
    sx: int,
    sy: int,
    own_hi_x: int,
    own_hi_y: int,
    own_hi_z: int,
):
    ix, iy, iz = wp.tid()
    cell = (iz * sy + iy) * sx + ix
    counts[cell] = 0
    cx = origin_x + ix
    cy = origin_y + iy
    cz = origin_z + iz
    if cx >= own_hi_x or cy >= own_hi_y or cz >= own_hi_z:
        return
    case = int(0)
    for corner in range(8):
        vx = cx + int(corners[corner * 3 + 0])
        vy = cy + int(corners[corner * 3 + 1])
        vz = cz + int(corners[corner * 3 + 2])
        if sample_occ(occ, nx, ny, nz, vx, vy, vz) != 0:
            case = case | (int(1) << corner)
    if case == 0 or case == 255:
        return
    found = int(0)
    for slot in range(16):
        if int(tri_table[case, slot]) >= 0:
            found = found + 1
    counts[cell] = found // 3


@wp.kernel
def emit_triangles(
    occ: wp.array(dtype=wp.uint8),
    corners: wp.array(dtype=wp.int32),
    edges: wp.array(dtype=wp.int32),
    tri_table: wp.array2d(dtype=wp.int32),
    edge_scan: wp.array(dtype=wp.int32),
    tri_counts: wp.array(dtype=wp.int32),
    tri_scan: wp.array(dtype=wp.int32),
    tris: wp.array(dtype=wp.int32),
    tri_cell: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    origin_x: int,
    origin_y: int,
    origin_z: int,
    sx: int,
    sy: int,
    sz: int,
    cells_x: int,
    cells_y: int,
    own_hi_x: int,
    own_hi_y: int,
    own_hi_z: int,
):
    ix, iy, iz = wp.tid()
    cell = (iz * cells_y + iy) * cells_x + ix
    count = int(tri_counts[cell])
    if count == 0:
        return
    cx = origin_x + ix
    cy = origin_y + iy
    cz = origin_z + iz
    if cx >= own_hi_x or cy >= own_hi_y or cz >= own_hi_z:
        return
    case = int(0)
    for corner in range(8):
        vx = cx + int(corners[corner * 3 + 0])
        vy = cy + int(corners[corner * 3 + 1])
        vz = cz + int(corners[corner * 3 + 2])
        if sample_occ(occ, nx, ny, nz, vx, vy, vz) != 0:
            case = case | (int(1) << corner)
    plane = sx * sy * sz
    row = sy * sz
    base = int(tri_scan[cell]) - count
    written = int(0)
    for tri in range(5):
        e0 = int(tri_table[case, tri * 3 + 0])
        if e0 < 0:
            return
        e1 = int(tri_table[case, tri * 3 + 1])
        e2 = int(tri_table[case, tri * 3 + 2])
        # The published table points into the bone. Swap so the normal faces out.
        ids = wp.vec3i(e0, e2, e1)
        for corner in range(3):
            edge = int(ids[corner])
            a = int(edges[edge * 2 + 0])
            b = int(edges[edge * 2 + 1])
            ax = int(corners[a * 3 + 0])
            ay = int(corners[a * 3 + 1])
            az = int(corners[a * 3 + 2])
            bx = int(corners[b * 3 + 0])
            by = int(corners[b * 3 + 1])
            bz = int(corners[b * 3 + 2])
            ox = ax
            oy = ay
            oz = az
            axis = int(0)
            if bx != ax:
                axis = int(0)
                if bx < ax:
                    ox = bx
            elif by != ay:
                axis = int(1)
                if by < ay:
                    oy = by
            else:
                axis = int(2)
                if bz < az:
                    oz = bz
            slot = axis * plane + (ix + ox) * row + (iy + oy) * sz + (iz + oz)
            tris[(base + written) * 3 + corner] = int(edge_scan[slot]) - 1
        # Bias 8 keeps the crop-border cell -1 inside an unsigned 10-bit field.
        tri_cell[base + written] = (cx + 8) | ((cy + 8) << 10) | ((cz + 8) << 20)
        written = written + 1


@wp.func
def on_shared_face(value: float, count: int, step: int) -> bool:
    nearest = wp.round(value / float(step))
    face = nearest * float(step)
    return wp.abs(value - face) < 1.0e-5 and face > 0.0 and face < float(count)


@wp.kernel
def mark_pins(points: wp.array(dtype=wp.vec3), pinned: wp.array(dtype=wp.int32), nx: int, ny: int, nz: int, step: int):
    i = wp.tid()
    p = points[i]
    pinned[i] = int(0)
    if on_shared_face(p[0], nx, step) or on_shared_face(p[1], ny, step) or on_shared_face(p[2], nz, step):
        pinned[i] = int(1)


@wp.kernel
def scatter_neighbors(
    tris: wp.array(dtype=wp.int32),
    degree: wp.array(dtype=wp.int32),
    neighbors: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    max_degree: int,
):
    tri = wp.tid()
    for corner in range(3):
        src = int(tris[tri * 3 + corner])
        dst = int(tris[tri * 3 + ((corner + 1) % 3)])
        slot = wp.atomic_add(degree, src, 1)
        if slot < max_degree:
            neighbors[src * max_degree + slot] = dst
        else:
            overflow[0] = 1


@wp.kernel
def smooth_step(
    current: wp.array(dtype=wp.vec3),
    original: wp.array(dtype=wp.vec3),
    degree: wp.array(dtype=wp.int32),
    neighbors: wp.array(dtype=wp.int32),
    pinned: wp.array(dtype=wp.int32),
    nxt: wp.array(dtype=wp.vec3),
    max_degree: int,
):
    i = wp.tid()
    if int(pinned[i]) != 0:
        nxt[i] = original[i]
        return
    deg = int(degree[i])
    if deg <= 0:
        nxt[i] = current[i]
        return
    acc = wp.vec3(0.0, 0.0, 0.0)
    used = deg
    if used > max_degree:
        used = max_degree
    for slot in range(32):
        if slot < used:
            acc = acc + current[int(neighbors[i * max_degree + slot])]
    mean = acc / float(used)
    delta = (mean - current[i]) * 0.5
    length = wp.length(delta)
    if length > 0.4:
        delta = delta * (0.4 / length)
    nxt[i] = current[i] + delta


@wp.kernel
def world_positions(
    index_points: wp.array(dtype=wp.vec3),
    world: wp.array(dtype=wp.vec3),
    dim_x: float,
    dim_y: float,
    dim_z: float,
    count_x: float,
    count_y: float,
    count_z: float,
    rotation: wp.array(dtype=wp.float32),
    translation: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p = index_points[i]
    local = wp.vec3(
        p[0] * (dim_x / count_x) - dim_x * 0.5,
        p[1] * (dim_y / count_y) - dim_y * 0.5,
        p[2] * (dim_z / count_z) - dim_z * 0.5,
    )
    # translation + rotation * local. rotation is row-major.
    world[i] = wp.vec3(
        local[0] * rotation[0] + local[1] * rotation[1] + local[2] * rotation[2] + translation[0],
        local[0] * rotation[3] + local[1] * rotation[4] + local[2] * rotation[5] + translation[1],
        local[0] * rotation[6] + local[1] * rotation[7] + local[2] * rotation[8] + translation[2],
    )


@wp.kernel
def save_count(values: wp.array(dtype=wp.int32), index: int, dest: wp.array(dtype=wp.int32), slot: int):
    dest[slot] = values[index]


@wp.kernel
def smooth_step_limited(
    current: wp.array(dtype=wp.vec3),
    original: wp.array(dtype=wp.vec3),
    degree: wp.array(dtype=wp.int32),
    neighbors: wp.array(dtype=wp.int32),
    pinned: wp.array(dtype=wp.int32),
    nxt: wp.array(dtype=wp.vec3),
    max_degree: int,
    limit: wp.array(dtype=wp.int32),
    slot: int,
):
    i = wp.tid()
    if i >= int(limit[slot]):
        return
    if int(pinned[i]) != 0:
        nxt[i] = original[i]
        return
    deg = int(degree[i])
    if deg <= 0:
        nxt[i] = current[i]
        return
    acc = wp.vec3(0.0, 0.0, 0.0)
    used = deg
    if used > max_degree:
        used = max_degree
    for neighbor in range(32):
        if neighbor < used:
            acc = acc + current[int(neighbors[i * max_degree + neighbor])]
    mean = acc / float(used)
    delta = (mean - current[i]) * 0.5
    length = wp.length(delta)
    if length > 0.4:
        delta = delta * (0.4 / length)
    nxt[i] = current[i] + delta


@wp.kernel
def mark_pins_limited(points: wp.array(dtype=wp.vec3), pinned: wp.array(dtype=wp.int32), nx: int, ny: int, nz: int, step: int, limit: wp.array(dtype=wp.int32), slot: int):
    i = wp.tid()
    if i >= int(limit[slot]):
        return
    p = points[i]
    pinned[i] = int(0)
    if on_shared_face(p[0], nx, step) or on_shared_face(p[1], ny, step) or on_shared_face(p[2], nz, step):
        pinned[i] = int(1)


@wp.kernel
def scatter_neighbors_limited(
    tris: wp.array(dtype=wp.int32),
    degree: wp.array(dtype=wp.int32),
    neighbors: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    max_degree: int,
    limit: wp.array(dtype=wp.int32),
    slot: int,
):
    tri = wp.tid()
    if tri >= int(limit[slot]):
        return
    for corner in range(3):
        src = int(tris[tri * 3 + corner])
        dst = int(tris[tri * 3 + ((corner + 1) % 3)])
        place = wp.atomic_add(degree, src, 1)
        if place < max_degree:
            neighbors[src * max_degree + place] = dst
        else:
            overflow[0] = 1


@wp.kernel
def zero_range(values: wp.array(dtype=wp.int32)):
    values[wp.tid()] = 0


@wp.kernel
def zero_vec3(values: wp.array(dtype=wp.vec3)):
    values[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def accumulate_normals(
    world: wp.array(dtype=wp.vec3),
    tris: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
):
    tri = wp.tid()
    i0 = int(tris[tri * 3 + 0])
    i1 = int(tris[tri * 3 + 1])
    i2 = int(tris[tri * 3 + 2])
    face = wp.cross(world[i1] - world[i0], world[i2] - world[i0])
    wp.atomic_add(normals, i0, face)
    wp.atomic_add(normals, i1, face)
    wp.atomic_add(normals, i2, face)


@wp.kernel
def normalize_vectors(normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    length = wp.length(normals[i])
    if length < 1.0e-8:
        normals[i] = wp.vec3(0.0, 0.0, 1.0)
    else:
        normals[i] = normals[i] / length


def _keep_owned(verts, tris, rgb, codes, block, shape, world=None, normals=None):
    """Drop halo triangles, then compact every per-vertex array with them."""
    codes = np.asarray(codes, dtype=np.int32)
    ci = (codes & 1023).astype(np.int64) - 8
    cj = ((codes >> 10) & 1023).astype(np.int64) - 8
    ck = ((codes >> 20) & 1023).astype(np.int64) - 8
    bx, by, bz = (int(v) for v in block)
    keep = _owns_cell(ci, bx, shape[0]) & _owns_cell(cj, by, shape[1]) & _owns_cell(ck, bz, shape[2])
    tris = np.asarray(tris)[keep]
    empty = (
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.int32),
        np.zeros((0, 3), np.uint8),
        None,
        None,
    )
    if len(tris) == 0:
        return empty
    used, inverse = np.unique(tris.reshape(-1), return_inverse=True)
    verts = verts[used]
    rgb = rgb[used]
    tris = inverse.reshape(-1, 3).astype(np.int32)
    if world is not None:
        world = np.ascontiguousarray(world[used], dtype=np.float32)
        normals = np.ascontiguousarray(normals[used], dtype=np.float32)
    return verts, tris, rgb, world, normals


class WarpMesher:
    """One device-side occupancy mirror and reusable scratch buffers."""

    def __init__(self):
        wp.init()
        self.device = "cuda:0"
        self.occ = None
        self.colors = None
        self.shape = None
        self.upload_bytes = 0
        self.download_bytes = 0
        self.transfer_ms = 0.0
        self.smooth_overflows = 0
        self._corners = wp.array(_CORNERS.reshape(-1), dtype=wp.int32, device=self.device)
        self._edges = wp.array(_EDGES.reshape(-1), dtype=wp.int32, device=self.device)
        self._edge_table = wp.array(np.asarray(EDGE_TABLE, dtype=np.int32), dtype=wp.int32, device=self.device)
        tri = np.asarray(TRI_TABLE, dtype=np.int32)
        self._tri_table = wp.array(tri, dtype=wp.int32, device=self.device)
        self._flags = wp.zeros(MAX_VERTS, dtype=wp.int32, device=self.device)
        self._scan = wp.zeros(MAX_VERTS, dtype=wp.int32, device=self.device)
        self._points = wp.zeros(MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._point_colors = wp.zeros(MAX_VERTS, dtype=wp.uint32, device=self.device)
        self._tri_counts = wp.zeros(MAX_CELLS, dtype=wp.int32, device=self.device)
        self._tri_scan = wp.zeros(MAX_CELLS, dtype=wp.int32, device=self.device)
        self._tris = wp.zeros(MAX_CELLS * 15, dtype=wp.int32, device=self.device)
        self._tri_cell = wp.zeros(MAX_CELLS * 5, dtype=wp.int32, device=self.device)
        self._degree = wp.zeros(MAX_VERTS, dtype=wp.int32, device=self.device)
        self._neighbors = wp.zeros(MAX_VERTS * MAX_DEGREE, dtype=wp.int32, device=self.device)
        self._overflow = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._pinned = wp.zeros(MAX_VERTS, dtype=wp.int32, device=self.device)
        self._original = wp.zeros(MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._swap = wp.zeros(MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._world = wp.zeros(MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._normals = wp.zeros(MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._rotation = wp.zeros(9, dtype=wp.float32, device=self.device)
        self._translation = wp.zeros(3, dtype=wp.float32, device=self.device)
        self._count_v = wp.zeros(SLOT_COUNT, dtype=wp.int32, device=self.device)
        self._count_t = wp.zeros(SLOT_COUNT, dtype=wp.int32, device=self.device)
        self._slot_points = wp.zeros(SLOT_COUNT * MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._slot_colors = wp.zeros(SLOT_COUNT * MAX_VERTS, dtype=wp.uint32, device=self.device)
        self._slot_tris = wp.zeros(SLOT_COUNT * MAX_CELLS * 15, dtype=wp.int32, device=self.device)
        self._slot_tri_cell = wp.zeros(SLOT_COUNT * MAX_CELLS * 5, dtype=wp.int32, device=self.device)
        self._slot_world = wp.zeros(SLOT_COUNT * MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._slot_normals = wp.zeros(SLOT_COUNT * MAX_VERTS, dtype=wp.vec3, device=self.device)
        self._events = [wp.Event(self.device, enable_timing=True) for _ in range(4)]
        self._warmed = False

    def upload_volume(self, occupied, colors):
        started = time.perf_counter()
        occ = np.ascontiguousarray(occupied, dtype=np.uint8).reshape(-1)
        packed = (
            colors[:, :, :, 0].astype(np.uint32)
            | (colors[:, :, :, 1].astype(np.uint32) << 8)
            | (colors[:, :, :, 2].astype(np.uint32) << 16)
        )
        packed = np.ascontiguousarray(packed).reshape(-1)
        self.occ = wp.array(occ, dtype=wp.uint8, device=self.device)
        self.colors = wp.array(packed, dtype=wp.uint32, device=self.device)
        self.shape = tuple(int(v) for v in occupied.shape)
        wp.synchronize()
        self.upload_bytes += int(occ.nbytes + packed.nbytes)
        self.transfer_ms += (time.perf_counter() - started) * 1e3

    def clear(self, xs, ys, zs):
        count = len(xs)
        if count == 0 or self.occ is None:
            return
        started = time.perf_counter()
        xs_d = wp.array(np.ascontiguousarray(xs, dtype=np.int32), dtype=wp.int32, device=self.device)
        ys_d = wp.array(np.ascontiguousarray(ys, dtype=np.int32), dtype=wp.int32, device=self.device)
        zs_d = wp.array(np.ascontiguousarray(zs, dtype=np.int32), dtype=wp.int32, device=self.device)
        _ny, nz = self.shape[1], self.shape[2]
        wp.launch(clear_voxels, dim=count, inputs=[self.occ, self.colors, xs_d, ys_d, zs_d, self.shape[1], nz], device=self.device)
        wp.synchronize()
        self.upload_bytes += int(count * 12)
        self.transfer_ms += (time.perf_counter() - started) * 1e3

    def warmup(self, occupied, colors):
        """Compile and run the kernels once. Later timings exclude this launch."""
        if self.occ is None:
            self.upload_volume(occupied, colors)
        identity = np.eye(3, dtype=np.float64)
        self.mesh_blocks([(0, 0, 0)], smooth=True, prepare_pose=(np.ones(3), np.full(3, 256.0), identity, np.zeros(3)))
        self._warmed = True
        self.transfer_ms = 0.0
        self.upload_bytes = 0
        self.download_bytes = 0

    def _scalar(self, array, index):
        return int(array.numpy()[index])

    def extract_block(self, block, smooth, prepare_pose):
        """Return verts, tris, colors, and finished-GPU phase times.

        ``prepare_pose`` set means also write world positions and normals.
        """
        phases = {"extract_ms": 0.0, "edge_ms": 0.0, "smooth_ms": 0.0, "prepare_gpu_ms": 0.0, "transfer_ms": 0.0, "download_bytes": 0}
        shape = self.shape
        geo = self._block_geometry(block)
        i0, j0, k0 = geo["origin"]
        sx, sy, sz = geo["size"]
        empty = (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.int32),
            np.zeros((0, 3), np.uint8),
            None,
            None,
        )
        if min(sx, sy, sz) < 2 or sx > MAX_AXIS or sy > MAX_AXIS or sz > MAX_AXIS:
            return empty, phases
        cx, cy, cz = geo["cells"]
        edge_n = 3 * sx * sy * sz
        cell_n = cx * cy * cz
        own_hi = geo["own_hi"]
        started = time.perf_counter()
        self._flags.zero_()
        self._tri_counts.zero_()
        wp.launch(
            mark_edges,
            dim=(cx, cy, cz),
            inputs=[
                self.occ, self._corners, self._edges, self._edge_table, self._flags,
                shape[0], shape[1], shape[2], i0, j0, k0, sx, sy, sz, own_hi[0], own_hi[1], own_hi[2],
            ],
            device=self.device,
        )
        wp.utils.array_scan(self._flags[:edge_n], self._scan[:edge_n], inclusive=True)
        wp.launch(
            emit_vertices,
            dim=edge_n,
            inputs=[
                self.occ, self.colors, self._flags, self._scan, self._points, self._point_colors,
                shape[0], shape[1], shape[2], i0, j0, k0, sx, sy, sz,
            ],
            device=self.device,
        )
        wp.launch(
            count_triangles,
            dim=(cx, cy, cz),
            inputs=[
                self.occ, self._corners, self._tri_table, self._tri_counts,
                shape[0], shape[1], shape[2], i0, j0, k0, cx, cy, own_hi[0], own_hi[1], own_hi[2],
            ],
            device=self.device,
        )
        wp.utils.array_scan(self._tri_counts[:cell_n], self._tri_scan[:cell_n], inclusive=True)
        n_verts = self._scalar(self._scan, edge_n - 1) if edge_n else 0
        n_tris = self._scalar(self._tri_scan, cell_n - 1) if cell_n else 0
        if n_verts == 0 or n_tris == 0:
            wp.synchronize()
            phases["extract_ms"] = (time.perf_counter() - started) * 1e3
            return empty, phases
        wp.launch(
            emit_triangles,
            dim=(cx, cy, cz),
            inputs=[
                self.occ, self._corners, self._edges, self._tri_table, self._scan, self._tri_counts, self._tri_scan, self._tris, self._tri_cell,
                shape[0], shape[1], shape[2], i0, j0, k0, sx, sy, sz, cx, cy, own_hi[0], own_hi[1], own_hi[2],
            ],
            device=self.device,
        )
        wp.synchronize()
        phases["extract_ms"] = (time.perf_counter() - started) * 1e3

        smoothed_on_gpu = False
        if smooth and n_tris:
            smoothing = time.perf_counter()
            self._overflow.zero_()
            self._degree.zero_()
            wp.launch(zero_range, dim=n_verts, inputs=[self._pinned], device=self.device)
            wp.launch(
                scatter_neighbors,
                dim=n_tris,
                inputs=[self._tris, self._degree, self._neighbors, self._overflow, MAX_DEGREE],
                device=self.device,
            )
            wp.copy(self._original, self._points, count=n_verts)
            current = self._points
            nxt = self._swap
            for _ in range(SMOOTH_ITERS):
                wp.launch(
                    smooth_step,
                    dim=n_verts,
                    inputs=[current, self._original, self._degree, self._neighbors, self._pinned, nxt, MAX_DEGREE],
                    device=self.device,
                )
                current, nxt = nxt, current
            if current is not self._points:
                wp.copy(self._points, current, count=n_verts)
            wp.synchronize()
            if int(self._overflow.numpy()[0]):
                self.smooth_overflows += 1
            else:
                smoothed_on_gpu = True
            phases["smooth_ms"] = (time.perf_counter() - smoothing) * 1e3

        downloading = time.perf_counter()
        verts = self._points.numpy()[:n_verts].astype(np.float64, copy=False)
        tris = self._tris.numpy()[: n_tris * 3].reshape(-1, 3).astype(np.int32, copy=False)
        codes = self._tri_cell.numpy()[:n_tris]
        packed = self._point_colors.numpy()[:n_verts]
        rgb = np.stack([(packed) & 255, (packed >> 8) & 255, (packed >> 16) & 255], axis=1).astype(np.uint8)
        phases["transfer_ms"] += (time.perf_counter() - downloading) * 1e3
        phases["download_bytes"] = int(n_verts * 16 + n_tris * 16 + n_verts * 4)
        self.download_bytes += phases["download_bytes"]
        if smooth and not smoothed_on_gpu and len(tris):
            fallback = time.perf_counter()
            verts = smooth_pinned(verts, tris, np.zeros(len(verts), dtype=bool), iterations=SMOOTH_ITERS)
            phases["smooth_ms"] += (time.perf_counter() - fallback) * 1e3
        world = None
        normals = None
        if prepare_pose is not None and len(tris):
            preparing = time.perf_counter()
            dims, counts, rotation, translation = prepare_pose
            rot = np.ascontiguousarray(rotation, dtype=np.float32).reshape(-1)
            self._rotation.assign(rot)
            self._translation.assign(np.ascontiguousarray(translation, dtype=np.float32))
            # Normals see the halo triangles, then the halo is dropped below.
            full_points = wp.array(np.ascontiguousarray(verts, dtype=np.float32), dtype=wp.vec3, device=self.device)
            full_tris = wp.array(np.ascontiguousarray(tris.reshape(-1), dtype=np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                world_positions,
                dim=len(verts),
                inputs=[
                    full_points, self._world,
                    float(dims[0]), float(dims[1]), float(dims[2]),
                    float(counts[0]), float(counts[1]), float(counts[2]),
                    self._rotation, self._translation,
                ],
                device=self.device,
            )
            wp.launch(zero_vec3, dim=len(verts), inputs=[self._normals], device=self.device)
            wp.launch(accumulate_normals, dim=len(tris), inputs=[self._world, full_tris, self._normals], device=self.device)
            wp.launch(normalize_vectors, dim=len(verts), inputs=[self._normals], device=self.device)
            wp.synchronize()
            world = self._world.numpy()[: len(verts)].astype(np.float32, copy=True)
            normals = self._normals.numpy()[: len(verts)].astype(np.float32, copy=True)
            phases["prepare_gpu_ms"] = (time.perf_counter() - preparing) * 1e3
            phases["download_bytes"] += int(len(verts) * 12 * 2)
            self.download_bytes += int(len(verts) * 12 * 2)
            self.upload_bytes += int(verts.nbytes + tris.nbytes)
        verts, tris, rgb, world, normals = _keep_owned(verts, tris, rgb, codes, block, shape, world, normals)
        return (verts, tris, rgb, world, normals), phases


    def _block_geometry(self, block):
        shape = self.shape
        bx, by, bz = (int(v) for v in block)
        i0, i1 = block_halo_bounds(bx, shape[0], BLOCK)
        j0, j1 = block_halo_bounds(by, shape[1], BLOCK)
        k0, k1 = block_halo_bounds(bz, shape[2], BLOCK)
        sx, sy, sz = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
        return {
            "origin": (i0, j0, k0),
            "size": (sx, sy, sz),
            "cells": (sx - 1, sy - 1, sz - 1),
            # Exclusive cell end of the halo read. Ownership is applied after smoothing.
            "own_hi": (i1, j1, k1),
        }

    def _enqueue_extract(self, block, slot):
        shape = self.shape
        geo = self._block_geometry(block)
        i0, j0, k0 = geo["origin"]
        sx, sy, sz = geo["size"]
        cx, cy, cz = geo["cells"]
        own_hi = geo["own_hi"]
        edge_n = 3 * sx * sy * sz
        cell_n = cx * cy * cz
        if min(sx, sy, sz) < 2 or max(sx, sy, sz) > MAX_AXIS:
            self._count_v_host = getattr(self, "_count_v_host", None)
            return {"skip": True, "edge_n": 0, "cell_n": 0, "block": block}
        wp.launch(zero_range, dim=edge_n, inputs=[self._flags], device=self.device)
        wp.launch(zero_range, dim=cell_n, inputs=[self._tri_counts], device=self.device)
        wp.launch(
            mark_edges,
            dim=(cx, cy, cz),
            inputs=[
                self.occ, self._corners, self._edges, self._edge_table, self._flags,
                shape[0], shape[1], shape[2], i0, j0, k0, sx, sy, sz, own_hi[0], own_hi[1], own_hi[2],
            ],
            device=self.device,
        )
        wp.utils.array_scan(self._flags[:edge_n], self._scan[:edge_n], inclusive=True)
        wp.launch(
            emit_vertices,
            dim=edge_n,
            inputs=[
                self.occ, self.colors, self._flags, self._scan, self._points, self._point_colors,
                shape[0], shape[1], shape[2], i0, j0, k0, sx, sy, sz,
            ],
            device=self.device,
        )
        wp.launch(
            count_triangles,
            dim=(cx, cy, cz),
            inputs=[
                self.occ, self._corners, self._tri_table, self._tri_counts,
                shape[0], shape[1], shape[2], i0, j0, k0, cx, cy, own_hi[0], own_hi[1], own_hi[2],
            ],
            device=self.device,
        )
        wp.utils.array_scan(self._tri_counts[:cell_n], self._tri_scan[:cell_n], inclusive=True)
        wp.launch(
            emit_triangles,
            dim=(cx, cy, cz),
            inputs=[
                self.occ, self._corners, self._edges, self._tri_table, self._scan, self._tri_counts, self._tri_scan, self._tris, self._tri_cell,
                shape[0], shape[1], shape[2], i0, j0, k0, sx, sy, sz, cx, cy, own_hi[0], own_hi[1], own_hi[2],
            ],
            device=self.device,
        )
        wp.launch(save_count, dim=1, inputs=[self._scan, edge_n - 1, self._count_v, slot], device=self.device)
        wp.launch(save_count, dim=1, inputs=[self._tri_scan, cell_n - 1, self._count_t, slot], device=self.device)
        wp.copy(self._slot_points, self._points, dest_offset=slot * MAX_VERTS, count=edge_n)
        wp.copy(self._slot_colors, self._point_colors, dest_offset=slot * MAX_VERTS, count=edge_n)
        wp.copy(self._slot_tris, self._tris, dest_offset=slot * MAX_CELLS * 15, count=cell_n * 15)
        wp.copy(self._slot_tri_cell, self._tri_cell, dest_offset=slot * MAX_CELLS * 5, count=cell_n * 5)
        return {"skip": False, "edge_n": edge_n, "cell_n": cell_n, "block": block}

    def mesh_blocks(self, blocks, smooth, prepare_pose):
        """Mesh every dirty block, then synchronize once per chunk."""
        phases = {"extract_ms": 0.0, "edge_ms": 0.0, "smooth_ms": 0.0, "prepare_gpu_ms": 0.0, "transfer_ms": 0.0, "download_bytes": 0}
        outputs = []
        blocks = list(blocks)
        for start in range(0, len(blocks), SLOT_COUNT):
            chunk = blocks[start : start + SLOT_COUNT]
            outputs.extend(self._mesh_chunk(chunk, smooth, prepare_pose, phases))
        return outputs, phases

    def _mesh_chunk(self, chunk, smooth, prepare_pose, phases):
        wp.record_event(self._events[0])
        metas = [self._enqueue_extract(block, slot) for slot, block in enumerate(chunk)]
        wp.record_event(self._events[1])
        wp.synchronize()
        phases["extract_ms"] += float(wp.get_event_elapsed_time(self._events[0], self._events[1]))
        counts_v = self._count_v.numpy()[: len(chunk)]
        counts_t = self._count_t.numpy()[: len(chunk)]
        shape = self.shape
        wp.record_event(self._events[1])
        for slot, meta in enumerate(metas):
            nv = 0 if meta["skip"] else int(counts_v[slot])
            nt = 0 if meta["skip"] else int(counts_t[slot])
            meta["nv"] = nv
            meta["nt"] = nt
            if nv == 0 or nt == 0:
                continue
            wp.copy(self._points, self._slot_points, src_offset=slot * MAX_VERTS, count=nv)
            wp.copy(self._tris, self._slot_tris, src_offset=slot * MAX_CELLS * 15, count=nt * 3)
            if smooth:
                wp.launch(zero_range, dim=nv, inputs=[self._degree], device=self.device)
                self._overflow.zero_()
                wp.launch(zero_range, dim=nv, inputs=[self._pinned], device=self.device)
                wp.launch(scatter_neighbors, dim=nt, inputs=[self._tris, self._degree, self._neighbors, self._overflow, MAX_DEGREE], device=self.device)
                wp.copy(self._original, self._points, count=nv)
                current = self._points
                nxt = self._swap
                for _ in range(SMOOTH_ITERS):
                    wp.launch(
                        smooth_step,
                        dim=nv,
                        inputs=[current, self._original, self._degree, self._neighbors, self._pinned, nxt, MAX_DEGREE],
                        device=self.device,
                    )
                    current, nxt = nxt, current
                if current is not self._points:
                    wp.copy(self._points, current, count=nv)
                wp.copy(self._slot_points, self._points, dest_offset=slot * MAX_VERTS, count=nv)
        wp.record_event(self._events[2])
        if prepare_pose is not None:
            dims, counts, rotation, translation = prepare_pose
            self._rotation.assign(np.ascontiguousarray(rotation, dtype=np.float32).reshape(-1))
            self._translation.assign(np.ascontiguousarray(translation, dtype=np.float32))
            for slot, meta in enumerate(metas):
                nv = meta["nv"]
                nt = meta["nt"]
                if nv == 0 or nt == 0:
                    continue
                wp.copy(self._points, self._slot_points, src_offset=slot * MAX_VERTS, count=nv)
                wp.copy(self._tris, self._slot_tris, src_offset=slot * MAX_CELLS * 15, count=nt * 3)
                wp.launch(
                    world_positions,
                    dim=nv,
                    inputs=[
                        self._points, self._world,
                        float(dims[0]), float(dims[1]), float(dims[2]),
                        float(counts[0]), float(counts[1]), float(counts[2]),
                        self._rotation, self._translation,
                    ],
                    device=self.device,
                )
                wp.launch(zero_vec3, dim=nv, inputs=[self._normals], device=self.device)
                wp.launch(accumulate_normals, dim=nt, inputs=[self._world, self._tris, self._normals], device=self.device)
                wp.launch(normalize_vectors, dim=nv, inputs=[self._normals], device=self.device)
                wp.copy(self._slot_world, self._world, dest_offset=slot * MAX_VERTS, count=nv)
                wp.copy(self._slot_normals, self._normals, dest_offset=slot * MAX_VERTS, count=nv)
        wp.record_event(self._events[3])
        wp.synchronize()
        if smooth:
            phases["smooth_ms"] += float(wp.get_event_elapsed_time(self._events[1], self._events[2]))
        if prepare_pose is not None:
            phases["prepare_gpu_ms"] += float(wp.get_event_elapsed_time(self._events[2], self._events[3]))
        downloading = time.perf_counter()
        outputs = []
        empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8), None, None)
        for slot, meta in enumerate(metas):
            nv = meta["nv"]
            nt = meta["nt"]
            if meta["skip"] or nv == 0 or nt == 0:
                outputs.append((meta.get("block"), empty))
                continue
            base_v = slot * MAX_VERTS
            base_t = slot * MAX_CELLS * 15
            base_c = slot * MAX_CELLS * 5
            verts = self._slot_points[base_v : base_v + nv].numpy().astype(np.float64, copy=False)
            tris = self._slot_tris[base_t : base_t + nt * 3].numpy().reshape(-1, 3).astype(np.int32, copy=False)
            codes = self._slot_tri_cell[base_c : base_c + nt].numpy()
            packed = self._slot_colors[base_v : base_v + nv].numpy()
            rgb = np.stack((packed & 255, (packed >> 8) & 255, (packed >> 16) & 255), axis=1).astype(np.uint8)
            world = None
            normals = None
            if prepare_pose is not None:
                world = self._slot_world[base_v : base_v + nv].numpy().astype(np.float32, copy=False)
                normals = self._slot_normals[base_v : base_v + nv].numpy().astype(np.float32, copy=False)
            verts, tris, rgb, world, normals = _keep_owned(
                verts, tris, rgb, codes, meta["block"], shape, world, normals
            )
            phases["download_bytes"] += int(nv * 20 + nt * 16 + (0 if world is None else len(verts) * 24))
            outputs.append((meta["block"], (verts, tris, rgb, world, normals)))
        phases["transfer_ms"] += (time.perf_counter() - downloading) * 1e3
        return outputs


def _pack_phases(parts):
    total = {"extract_ms": 0.0, "edge_ms": 0.0, "smooth_ms": 0.0, "transfer_ms": 0.0, "download_bytes": 0, "prepare_gpu_ms": 0.0}
    for part in parts:
        for key in total:
            total[key] += part[key]
    return total
