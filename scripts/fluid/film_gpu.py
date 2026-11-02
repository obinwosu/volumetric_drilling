"""ImpactFilm with its water, bone surface, and sheet mesh kept on the GPU.

The flow is the same as ImpactFilm: inlet rings, two downhill passes, a blur,
decay, the volume cap, and suction. Each frame is queued on the film's own
stream: the inputs are copied up, the flow and the sheet build run, and the
mesh is copied into pinned host memory. Nothing on the host waits for it. With ``pipelined`` set,
mesh() hands back the newest finished frame, at most one frame old, so the
render loop never stalls behind the path tracer.

The anchoring logic (when to slide, re-read, or turn the grid) stays on the
host in ImpactFilm. Bone reads march the occupancy on the GPU, ideally the
mirror the bone mesher already keeps.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import warp as wp

from fluid import impact_film as F
from fluid.film_mesh_gpu import _display_pass, _edges, _in, _quads
from fluid.outlet import MM_TO_SIM

# Frames in flight. The one drawn, the one finishing, and the one being written.
SETS = 3
# The sheet's triangles cover square blocks of the drawn grid around the water.
# They only change when water reaches a new block or the bone is re-read, so
# most frames move vertices and the renderer refits instead of rebuilding.
BLOCK = 16
# A block stays in the sheet this long after its neighbourhood dries.
BLOCK_HOLD_S = 1.5
# Topologies kept for frames still in flight.
KEEP_TOPOLOGIES = 6
# A quad whose corners differ in height by more than this bridges a drop, such
# as a drilled wall, and stays out of the sheet.
CLIFF_MM = 1.0

CELL = wp.constant(float(F.CELL_MM))
MM = wp.constant(float(MM_TO_SIM))
MIN_DEPTH = wp.constant(float(F.MIN_DEPTH_MM))
CLIFF = wp.constant(CLIFF_MM)
NOISE_CELL = wp.constant(float(F.RIVULET_CELL_MM))

# Per-frame inputs, one float each.
P_DT = wp.constant(0)
P_JET = wp.constant(1)
P_HX = wp.constant(2)
P_HY = wp.constant(3)
P_SPREAD = wp.constant(4)
P_DEPTH = wp.constant(5)
P_STAMP = wp.constant(6)
P_VOLUME = wp.constant(7)
P_LIFT = wp.constant(8)
P_BUDGET = wp.constant(9)
P_LIP = wp.constant(10)  # four, one per flow direction
P_PIN = wp.constant(14)
P_BUMP = wp.constant(15)
P_RIVULETS = wp.constant(16)
P_DECAY = wp.constant(17)
P_CUT = wp.constant(18)
P_LIMIT = wp.constant(19)
P_SUCK = wp.constant(20)
P_TIP = wp.constant(21)  # vec3
P_SUCK_R = wp.constant(24)
P_DOWN = wp.constant(25)  # vec3
P_NORMAL = wp.constant(28)  # vec3
P_TANGENT = wp.constant(31)  # vec3
P_SIDE = wp.constant(34)  # vec3
P_CLOCK = wp.constant(37)
P_RIPPLE = wp.constant(38)
P_KEEP = wp.constant(39)
P_FLOW = wp.constant(40)
P_SKIN = wp.constant(41)
P_FLOOR = wp.constant(42)
P_RIPPLE_FULL = wp.constant(43)
P_GATHER = wp.constant(44)
P_ANCHOR = wp.constant(45)
P_COUNT = 48

# ring[0:8] is the room in each inlet ring, ring[8:17] the volume left before each.
RING_VOL = wp.constant(8)


@wp.func
def _v(P: wp.array(dtype=float), k: int):
    return wp.vec3(P[k], P[k + 1], P[k + 2])


@wp.func
def _dir(k: int):
    if k == 0:
        return wp.vec2i(1, 0)
    if k == 1:
        return wp.vec2i(-1, 0)
    if k == 2:
        return wp.vec2i(0, 1)
    return wp.vec2i(0, -1)


@wp.func
def _radius(P: wp.array(dtype=float), i: int, j: int, n: int):
    c = n // 2
    x = float(i - c) * CELL - P[P_HX]
    y = float(j - c) * CELL - P[P_HY]
    return wp.sqrt(x * x + y * y)


@wp.func
def _home(on_bone: wp.array2d(dtype=wp.int32), reach: wp.array2d(dtype=wp.int32), i: int, j: int):
    return on_bone[i, j] != 0 and reach[i, j] != 0


@wp.kernel
def _begin(P: wp.array(dtype=float), ring: wp.array(dtype=float), sums: wp.array(dtype=float)):
    for k in range(8):
        ring[k] = 0.0
    ring[RING_VOL] = P[P_VOLUME]
    sums[0] = 0.0
    sums[1] = 0.0
    sums[2] = 0.0


@wp.kernel
def _inlet_room(
    P: wp.array(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    reach: wp.array2d(dtype=wp.int32),
    depth: wp.array2d(dtype=float),
    ring: wp.array(dtype=float),
    r: int,
):
    i, j = wp.tid()
    if P[P_JET] == 0.0:
        return
    n = depth.shape[0]
    radius = _radius(P, i, j, n)
    if r == 0 and on_bone[i, j] != 0 and radius <= P[P_SPREAD]:
        reach[i, j] = 1
    stamp = P[P_STAMP] + float(r) * CELL
    if stamp > P[P_SPREAD]:
        return
    if on_bone[i, j] != 0 and radius <= stamp:
        room = wp.max(P[P_DEPTH] - depth[i, j], 0.0)
        if room > 0.0:
            wp.atomic_add(ring, r, room)


@wp.kernel
def _inlet_apply(
    P: wp.array(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    depth: wp.array2d(dtype=float),
    ring: wp.array(dtype=float),
    r: int,
):
    i, j = wp.tid()
    n = depth.shape[0]
    volume = ring[RING_VOL + r]
    space = ring[r]
    stamp = P[P_STAMP] + float(r) * CELL
    share = float(0.0)
    if P[P_JET] != 0.0 and stamp <= P[P_SPREAD] and volume > 1.0e-9 and space > 0.0:
        share = wp.min(1.0, volume / space)
    if i == 0 and j == 0:
        ring[RING_VOL + r + 1] = volume - space * share
    if share > 0.0 and on_bone[i, j] != 0 and _radius(P, i, j, n) <= stamp:
        depth[i, j] = depth[i, j] + wp.max(P[P_DEPTH] - depth[i, j], 0.0) * share


@wp.func
def _level(
    P: wp.array(dtype=float),
    bed: wp.array2d(dtype=wp.vec3),
    rough: wp.array2d(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    depth: wp.array2d(dtype=float),
    i: int,
    j: int,
):
    # Millimetres against the tuned downhill, plus the rivulet bumps.
    base = float(0.0)
    if on_bone[i, j] != 0:
        base = -wp.dot(bed[i, j], _v(P, P_DOWN)) / MM + rough[i, j] * P[P_BUMP]
    return base + depth[i, j] * P[P_LIFT]


@wp.kernel
def _flow_moves(
    P: wp.array(dtype=float),
    depth: wp.array2d(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    reach: wp.array2d(dtype=wp.int32),
    bed: wp.array2d(dtype=wp.vec3),
    rough: wp.array2d(dtype=float),
    moved: wp.array3d(dtype=float),
    give: wp.array3d(dtype=float),
):
    i, j = wp.tid()
    n = depth.shape[0]
    d = depth[i, j]
    home = _home(on_bone, reach, i, j)
    level = _level(P, bed, rough, on_bone, depth, i, j)
    budget = P[P_BUDGET]
    pin = P[P_PIN]
    move = wp.vec4(0.0, 0.0, 0.0, 0.0)
    lower = wp.vec4(0.0, 0.0, 0.0, 0.0)
    total = float(0.0)
    for k in range(4):
        s = _dir(k)
        a = i + s[0]
        b = j + s[1]
        beside = bool(False)
        beside_level = float(0.0)
        beside_wet = bool(False)
        if _in(a, b, n):
            beside = _home(on_bone, reach, a, b)
            beside_level = _level(P, bed, rough, on_bone, depth, a, b)
            beside_wet = depth[a, b] >= MIN_DEPTH
        down = beside and beside_level + 1.0e-4 < level
        if pin > 0.0:
            # A thin front stays put on dry bone, so the flow breaks into streams.
            down = down and (beside_wet or d > pin)
        gap = wp.max(level - beside_level, 0.0)
        amount = float(0.0)
        if down:
            amount = wp.min(d * 0.25, wp.min(budget, 0.5 * gap))
            lower[k] = 1.0
        if home and not beside:
            amount = wp.min(d * P[P_LIP + k], budget)
        if not (home and d > 0.0):
            amount = 0.0
        move[k] = amount
        total += amount
    scale = wp.min(1.0, d / wp.max(total, 1.0e-8))
    for k in range(4):
        moved[i, j, k] = move[k] * scale
        give[i, j, k] = move[k] * scale * lower[k]


@wp.kernel
def _flow_apply(
    P: wp.array(dtype=float),
    src: wp.array2d(dtype=float),
    moved: wp.array3d(dtype=float),
    give: wp.array3d(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    reach: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if P[P_FLOW] == 0.0:
        dst[i, j] = src[i, j]
        return
    n = src.shape[0]
    d = src[i, j]
    for k in range(4):
        d -= moved[i, j, k]
        s = _dir(k)
        a = i - s[0]
        b = j - s[1]
        if _in(a, b, n):
            d += give[a, b, k]
    d = wp.clamp(d, 0.0, P[P_DEPTH])
    if not _home(on_bone, reach, i, j):
        d = 0.0
    dst[i, j] = d


@wp.kernel
def _blur(
    P: wp.array(dtype=float),
    src: wp.array2d(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    reach: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if P[P_FLOW] == 0.0:
        dst[i, j] = src[i, j]
        return
    n = src.shape[0]
    d = src[i, j]
    total = d
    for k in range(4):
        s = _dir(k)
        if _in(i + s[0], j + s[1], n):
            total += src[i + s[0], j + s[1]]
    mix = float(0.35)
    if P[P_RIVULETS] > 0.0 and d < MIN_DEPTH:
        # The blur would otherwise creep onto dry bone and undo the pinning.
        mix = mix * (1.0 - P[P_RIVULETS])
    out = float(0.0)
    if _home(on_bone, reach, i, j):
        out = wp.clamp((1.0 - mix) * d + mix * (0.2 * total), 0.0, P[P_DEPTH])
    dst[i, j] = out


@wp.kernel
def _finish(
    P: wp.array(dtype=float),
    src: wp.array2d(dtype=float),
    dst: wp.array2d(dtype=float),
    sums: wp.array(dtype=float),
):
    i, j = wp.tid()
    d = src[i, j] * P[P_DECAY]
    if d < P[P_CUT]:
        d = 0.0
    d = wp.clamp(d, 0.0, P[P_DEPTH])
    dst[i, j] = d
    if d > 0.0:
        wp.atomic_add(sums, 0, d)


@wp.kernel
def _cap_and_suck(
    P: wp.array(dtype=float),
    depth: wp.array2d(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    bed: wp.array2d(dtype=wp.vec3),
    sums: wp.array(dtype=float),
):
    i, j = wp.tid()
    volume = sums[0] * CELL * CELL
    limit = P[P_LIMIT]
    d = depth[i, j]
    if volume > limit and limit > 0.0:
        d = d * (limit / volume)
    if P[P_SUCK] != 0.0 and on_bone[i, j] != 0:
        if wp.length(bed[i, j] - _v(P, P_TIP)) <= P[P_SUCK_R]:
            d = 0.0
    depth[i, j] = d
    if d > 0.0:
        wp.atomic_add(sums, 1, d)
        wp.atomic_max(sums, 2, d)


@wp.kernel
def _refine_keep(
    shown: wp.array2d(dtype=float),
    on_bone: wp.array2d(dtype=wp.int32),
    bed: wp.array2d(dtype=wp.vec3),
    min_depth: float,
    bed_f: wp.array2d(dtype=wp.vec3),
    depth_f: wp.array2d(dtype=float),
    mask_f: wp.array2d(dtype=wp.int32),
    valid_f: wp.array2d(dtype=wp.int32),
):
    I, J = wp.tid()
    ci = I // 2
    cj = J // 2
    p = I - 2 * ci
    q = J - 2 * cj
    acc = wp.vec3(0.0, 0.0, 0.0)
    depth = float(0.0)
    count = float(0.0)
    wet = float(0.0)
    for a in range(p + 1):
        for b in range(q + 1):
            if on_bone[ci + a, cj + b] != 0:
                acc += bed[ci + a, cj + b]
                depth += shown[ci + a, cj + b]
                count += 1.0
                if shown[ci + a, cj + b] >= min_depth:
                    wet += 1.0
    need = float((p + 1) * (q + 1))
    inv = 1.0 / wp.max(count, 1.0)
    # A vertex whose bone was just drilled away keeps its last place until the
    # sheet's triangles catch up with the new read, instead of jumping to zero.
    if count > 0.0:
        bed_f[I, J] = acc * inv
    depth_f[I, J] = depth * inv
    valid_f[I, J] = 0
    if count >= need:
        valid_f[I, J] = 1
    if count >= need and wet / need >= 0.5 and depth * inv >= min_depth:
        mask_f[I, J] = 1
    else:
        mask_f[I, J] = 0


@wp.kernel
def _lift(
    P: wp.array(dtype=float),
    bed_f: wp.array2d(dtype=wp.vec3),
    depth_f: wp.array2d(dtype=float),
    mask: wp.array2d(dtype=wp.int32),
    pos: wp.array2d(dtype=wp.vec3),
    verts: wp.array(dtype=wp.vec3),
):
    I, J = wp.tid()
    m = mask.shape[0]
    normal = _v(P, P_NORMAL)
    floor = bed_f[I, J] - normal * (P[P_FLOOR] * MM)
    # Dry vertices sit on the floor, so the top meets the floor at the edge of
    # the water and the body is closed without walls.
    if mask[I, J] != 0:
        pos[I, J] = bed_f[I, J] + normal * ((depth_f[I, J] + P[P_SKIN]) * MM)
    else:
        pos[I, J] = floor
    verts[m * m + I * m + J] = floor


@wp.kernel
def _outline_on_shelf(
    P: wp.array(dtype=float),
    src: wp.array2d(dtype=wp.vec3),
    edge: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.vec3),
):
    I, J = wp.tid()
    m = src.shape[0]
    here = src[I, J]
    dst[I, J] = here
    if edge[I, J] == 0:
        return
    normal = _v(P, P_NORMAL)
    acc = wp.vec3(0.0, 0.0, 0.0)
    count = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            if _in(I + di, J + dj, m):
                if edge[I + di, J + dj] != 0:
                    there = src[I + di, J + dj]
                    # Rounding stays on one shelf; an edge across a drop is not a neighbour.
                    if wp.abs(wp.dot(there - here, normal)) <= CLIFF * MM:
                        acc += there
                        count += 1.0
    if count >= 2.0:
        dst[I, J] = 0.5 * here + 0.5 * (acc / count)


@wp.kernel
def _snap_boneless(
    P: wp.array(dtype=float),
    valid: wp.array2d(dtype=wp.int32),
    verts: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
):
    I, J = wp.tid()
    if valid[I, J] != 0:
        return
    m = valid.shape[0]
    top = m * m
    v = I * m + J
    # Until the sheet's triangles catch up with a move or a new bone read, a
    # vertex with no bone under it takes a neighbour's place, so its triangles
    # collapse instead of stretching to where the bone used to be.
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if _in(I + di, J + dj, m):
                if valid[I + di, J + dj] != 0:
                    u = (I + di) * m + (J + dj)
                    verts[v] = verts[u]
                    verts[top + v] = verts[top + u]
                    normals[v] = normals[u]
                    normals[top + v] = normals[top + u]
                    return
    anchor = _v(P, P_ANCHOR)
    verts[v] = anchor
    verts[top + v] = anchor
    normals[v] = _v(P, P_NORMAL)
    normals[top + v] = -_v(P, P_NORMAL)


@wp.kernel
def _block_wet(mask: wp.array2d(dtype=wp.int32), size: int, blocks: wp.array2d(dtype=wp.int32)):
    I, J = wp.tid()
    if mask[I, J] == 0:
        return
    nb = blocks.shape[0]
    bi = wp.min(I // size, nb - 1)
    bj = wp.min(J // size, nb - 1)
    blocks[bi, bj] = 1
    # A vertex on a block edge belongs to the blocks on both sides.
    if I % size == 0 and I > 0:
        blocks[bi - 1, bj] = 1
        if J % size == 0 and J > 0:
            blocks[bi - 1, bj - 1] = 1
    if J % size == 0 and J > 0:
        blocks[bi, bj - 1] = 1


@wp.kernel
def _gather(
    P: wp.array(dtype=float),
    verts: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    ids: wp.array(dtype=wp.int32),
    out_verts: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    k = wp.tid()
    count = int(P[P_GATHER])
    if k >= count:
        return
    top = verts.shape[0] // 2
    v = ids[k]
    out_verts[k] = verts[v]
    out_verts[count + k] = verts[top + v]
    out_normals[k] = normals[v]
    out_normals[count + k] = normals[top + v]


@wp.kernel
def _smooth(
    P: wp.array(dtype=float),
    src: wp.array2d(dtype=wp.vec3),
    mask: wp.array2d(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.vec3),
):
    I, J = wp.tid()
    m = src.shape[0]
    here = src[I, J]
    dst[I, J] = here
    if mask[I, J] == 0:
        return
    total = here
    count = float(1.0)
    for k in range(4):
        s = _dir(k)
        if _in(I + s[0], J + s[1], m):
            if mask[I + s[0], J + s[1]] != 0:
                total += src[I + s[0], J + s[1]]
                count += 1.0
    keep = P[P_KEEP]
    dst[I, J] = keep * here + (1.0 - keep) * (total / count)


@wp.kernel
def _place(pos: wp.array2d(dtype=wp.vec3), verts: wp.array(dtype=wp.vec3)):
    I, J = wp.tid()
    verts[I * pos.shape[0] + J] = pos[I, J]


@wp.func
def _face(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    return wp.cross(p1 - p0, p2 - p0)


@wp.func
def _quad_at(quad: wp.array2d(dtype=wp.int32), i: int, j: int):
    if _in(i, j, quad.shape[0]):
        return quad[i, j] != 0
    return False


@wp.kernel
def _shade(
    P: wp.array(dtype=float),
    pos: wp.array2d(dtype=wp.vec3),
    quad: wp.array2d(dtype=wp.int32),
    mask: wp.array2d(dtype=wp.int32),
    depth_f: wp.array2d(dtype=float),
    waves: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
):
    I, J = wp.tid()
    m = pos.shape[0]
    normal = _v(P, P_NORMAL)
    acc = wp.vec3(0.0, 0.0, 0.0)
    if _quad_at(quad, I, J):
        acc += _face(pos[I, J], pos[I + 1, J], pos[I, J + 1])
    if _quad_at(quad, I - 1, J):
        acc += _face(pos[I - 1, J], pos[I, J], pos[I - 1, J + 1])
        acc += _face(pos[I, J], pos[I, J + 1], pos[I - 1, J + 1])
    if _quad_at(quad, I, J - 1):
        acc += _face(pos[I, J - 1], pos[I + 1, J - 1], pos[I, J])
        acc += _face(pos[I + 1, J - 1], pos[I + 1, J], pos[I, J])
    if _quad_at(quad, I - 1, J - 1):
        acc += _face(pos[I, J - 1], pos[I, J], pos[I - 1, J])
    length = wp.length(acc)
    out = normal
    if length >= 1.0e-10 and mask[I, J] != 0:
        out = acc / length
    amount = P[P_RIPPLE]
    if amount > 0.0 and mask[I, J] != 0:
        tangent = _v(P, P_TANGENT)
        side = _v(P, P_SIDE)
        world_mm = pos[I, J] / MM
        tilt = wp.vec3(0.0, 0.0, 0.0)
        for w in range(waves.shape[0]):
            wave = waves[w]
            direction = wp.cos(wave[2]) * tangent + wp.sin(wave[2]) * side
            phase = (2.0 * wp.pi / wave[0]) * (wp.dot(world_mm, direction) - wave[1] * P[P_CLOCK])
            tilt += wp.cos(phase) * direction
        weight = wp.clamp(depth_f[I, J] / P[P_RIPPLE_FULL], 0.0, 1.0)
        out = out - (amount * weight) * tilt
        out = out / wp.max(wp.length(out), 1.0e-8)
    normals[I * m + J] = out
    normals[m * m + I * m + J] = -normal


@wp.func
def _lattice(ix: wp.int64, iy: wp.int64, iz: wp.int64):
    h = (ix * wp.int64(73856093)) ^ (iy * wp.int64(19349663)) ^ (iz * wp.int64(83492791))
    h = (h ^ (h >> wp.int64(13))) * wp.int64(1274126177)
    h = h ^ (h >> wp.int64(16))
    return float(h & wp.int64(65535)) / 32767.5 - 1.0


@wp.func
def _noise(p_mm: wp.vec3):
    p = p_mm / NOISE_CELL
    base = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - base
    f = wp.cw_mul(wp.cw_mul(f, f), wp.vec3(3.0, 3.0, 3.0) - 2.0 * f)
    bx = wp.int64(base[0])
    by = wp.int64(base[1])
    bz = wp.int64(base[2])
    out = float(0.0)
    for dx in range(2):
        wx = 1.0 - f[0]
        if dx == 1:
            wx = f[0]
        for dy in range(2):
            wy = 1.0 - f[1]
            if dy == 1:
                wy = f[1]
            for dz in range(2):
                wz = 1.0 - f[2]
                if dz == 1:
                    wz = f[2]
                out += wx * wy * wz * _lattice(bx + wp.int64(dx), by + wp.int64(dy), bz + wp.int64(dz))
    return out


@wp.func
def _solid(occ: wp.array(dtype=wp.uint8), p: wp.vec3, nx: int, ny: int, nz: int):
    x = int(wp.floor(p[0]))
    y = int(wp.floor(p[1]))
    z = int(wp.floor(p[2]))
    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
        return False
    return occ[(x * ny + y) * nz + z] != wp.uint8(0)


@wp.kernel
def _sample(
    occ: wp.array(dtype=wp.uint8),
    nx: int,
    ny: int,
    nz: int,
    to_index: wp.mat33,
    index_offset: wp.vec3,
    origin: wp.vec3,
    right: wp.vec3,
    across: wp.vec3,
    normal: wp.vec3,
    above_mm: float,
    steps: int,
    step_mm: float,
    bed: wp.array2d(dtype=wp.vec3),
    on_bone: wp.array2d(dtype=wp.int32),
    rough: wp.array2d(dtype=float),
    depth: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    c = bed.shape[0] // 2
    start = origin + right * (float(i - c) * CELL * MM) + across * (float(j - c) * CELL * MM) + normal * (above_mm * MM)
    first_index = to_index * start + index_offset
    step = to_index * (normal * (-step_mm * MM))
    found = bool(False)
    first = int(0)
    for k in range(steps + 1):
        if _solid(occ, first_index + float(k) * step, nx, ny, nz):
            found = True
            first = k
            break
    if not found:
        bed[i, j] = wp.vec3(0.0, 0.0, 0.0)
        on_bone[i, j] = 0
        rough[i, j] = 0.0
        depth[i, j] = 0.0
        return
    fine = float(first)
    # Half a step back catches the surface the coarse step jumped past.
    if first > 0 and _solid(occ, first_index + (fine - 0.5) * step, nx, ny, nz):
        fine -= 0.5
    point = start - normal * (fine * step_mm * MM)
    bed[i, j] = point
    on_bone[i, j] = 1
    rough[i, j] = _noise(point / MM)


@wp.kernel
def _shift_float(src: wp.array2d(dtype=float), di: int, dj: int, dst: wp.array2d(dtype=float)):
    i, j = wp.tid()
    n = src.shape[0]
    value = float(0.0)
    if _in(i + di, j + dj, n):
        value = src[i + di, j + dj]
    dst[i, j] = value


@wp.kernel
def _shift_int(src: wp.array2d(dtype=wp.int32), di: int, dj: int, dst: wp.array2d(dtype=wp.int32)):
    i, j = wp.tid()
    n = src.shape[0]
    value = int(0)
    if _in(i + di, j + dj, n):
        value = src[i + di, j + dj]
    dst[i, j] = value


def cuda_ready() -> bool:
    try:
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _dilate(blocks: np.ndarray) -> np.ndarray:
    out = blocks.copy()
    out[1:] |= blocks[:-1]
    out[:-1] |= blocks[1:]
    grown = out.copy()
    out[:, 1:] |= grown[:, :-1]
    out[:, :-1] |= grown[:, 1:]
    return out


def _shift_blocks(seen: np.ndarray, si: float, sj: float) -> np.ndarray:
    """Block times after the grid slides by (si, sj) blocks, rounded both ways."""
    nb = seen.shape[0]
    out = np.full_like(seen, -1.0e9)
    for a in {math.floor(si), math.ceil(si)}:
        for b in {math.floor(sj), math.ceil(sj)}:
            moved = np.full_like(seen, -1.0e9)
            src_i = slice(max(a, 0), nb + min(a, 0))
            dst_i = slice(max(-a, 0), nb + min(-a, 0))
            src_j = slice(max(b, 0), nb + min(b, 0))
            dst_j = slice(max(-b, 0), nb + min(-b, 0))
            moved[dst_i, dst_j] = seen[src_i, src_j]
            out = np.maximum(out, moved)
    return out


def _fine(coarse: np.ndarray, combine) -> np.ndarray:
    """Coarse cells to drawn vertices: corners, edge midpoints, and quad centres."""
    n = coarse.shape[0]
    fine = np.empty((2 * n - 1, 2 * n - 1), dtype=coarse.dtype)
    fine[0::2, 0::2] = coarse
    fine[1::2, 0::2] = combine(coarse[:-1], coarse[1:])
    fine[0::2, 1::2] = combine(coarse[:, :-1], coarse[:, 1:])
    fine[1::2, 1::2] = combine(combine(coarse[:-1, :-1], coarse[1:, :-1]), combine(coarse[:-1, 1:], coarse[1:, 1:]))
    return fine


def _sheet_topology(on_bone: np.ndarray, active: np.ndarray, m: int, height_mm=None):
    """Triangles over the active blocks: top, reversed floor, walls on open edges.

    Returns (tris, vertex ids, top triangle count). Vertices are numbered by
    their place in ``ids``; floor vertices follow the top ones.
    """
    whole = _fine(on_bone, np.logical_and)
    quad = whole[:-1, :-1] & whole[1:, :-1] & whole[:-1, 1:] & whole[1:, 1:]
    if height_mm is not None:
        h = _fine(np.where(on_bone, height_mm, 0.0), lambda a, b: 0.5 * (a + b))
        corners = np.stack([h[:-1, :-1], h[1:, :-1], h[:-1, 1:], h[1:, 1:]])
        quad &= corners.max(axis=0) - corners.min(axis=0) <= CLIFF_MM
    rows = np.minimum(np.arange(m - 1) // BLOCK, active.shape[0] - 1)
    quad &= active[rows[:, None], rows[None, :]]
    qi, qj = np.nonzero(quad)
    if len(qi) == 0:
        return np.zeros((0, 3), np.int32), np.zeros(0, np.int32), 0
    used = np.zeros((m, m), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            used[qi + di, qj + dj] = True
    ids = np.flatnonzero(used).astype(np.int32)
    local = np.full(m * m, -1, dtype=np.int64)
    local[ids] = np.arange(len(ids))
    k = len(ids)
    a = local[qi * m + qj]
    b = local[(qi + 1) * m + qj]
    c = local[qi * m + qj + 1]
    d = local[(qi + 1) * m + qj + 1]
    # cross(right, across) is the outward normal, so this winding faces out of the bone.
    top = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    floor = np.concatenate([np.stack([c, b, a], 1), np.stack([c, d, b], 1)]) + k
    padded = np.zeros((m + 1, m + 1), dtype=bool)
    padded[1:m, 1:m] = quad
    walls = []
    # Each top quad runs a -> b -> d -> c; an edge with no quad across it gets a
    # wall using the edge in the opposite direction to the top and floor.
    for u, v, ni, nj in ((a, b, 0, -1), (b, d, 1, 0), (d, c, 0, 1), (c, a, -1, 0)):
        open_edge = ~padded[qi + 1 + ni, qj + 1 + nj]
        u, v = u[open_edge], v[open_edge]
        walls.append(np.stack([v, u, u + k], 1))
        walls.append(np.stack([v, u + k, v + k], 1))
    tris = np.concatenate([top, floor, *walls]).astype(np.int32)
    return tris, ids, len(top)


class GpuFilm(F.ImpactFilm):
    """ImpactFilm whose state lives on the GPU. Host arrays are read-back copies."""

    def __init__(self, device=None, pipelined: bool = False):
        self._dev_ready = False
        super().__init__()
        self.device = wp.get_device(device)
        self.pipelined = bool(pipelined)
        # Returns the bone mesher (with .occ and .shape) when it mirrors the volume.
        self.occupancy_source = None
        d = self.device
        n, m = self.n, self.m
        self._cuda = self.device.is_cuda
        self.stream = None
        if self._cuda:
            try:
                self.stream = wp.Stream(d, priority=-1)
            except Exception:
                self.stream = wp.Stream(d)
        pinned = self._cuda
        self._P = wp.zeros(P_COUNT, dtype=float, device=d)
        self._P_host = [wp.zeros(P_COUNT, dtype=float, device="cpu", pinned=pinned) for _ in range(SETS)]
        self._ring = wp.zeros(17, dtype=float, device=d)
        self._sums = wp.zeros(3, dtype=float, device=d)
        self._depth = wp.zeros((n, n), dtype=float, device=d)
        self._depth_b = wp.zeros((n, n), dtype=float, device=d)
        self._shown = wp.zeros((n, n), dtype=float, device=d)
        self._spare = wp.zeros((n, n), dtype=float, device=d)
        self._reach = wp.zeros((n, n), dtype=wp.int32, device=d)
        self._reach_b = wp.zeros((n, n), dtype=wp.int32, device=d)
        self._on_bone = wp.zeros((n, n), dtype=wp.int32, device=d)
        self._bed = wp.zeros((n, n), dtype=wp.vec3, device=d)
        self._rough = wp.zeros((n, n), dtype=float, device=d)
        self._moved = wp.zeros((n, n, 4), dtype=float, device=d)
        self._give = wp.zeros((n, n, 4), dtype=float, device=d)
        self._bed_f = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._round = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._depth_f = wp.zeros((m, m), dtype=float, device=d)
        self._mask_d = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._edge = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._quad = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._pos = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._pos_b = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._verts_d = wp.zeros(2 * m * m, dtype=wp.vec3, device=d)
        self._normals_d = wp.zeros(2 * m * m, dtype=wp.vec3, device=d)
        self.n_blocks = -(-(m - 1) // BLOCK)
        nb = self.n_blocks
        self._blocks_d = wp.zeros((nb, nb), dtype=wp.int32, device=d)
        self._valid_f = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._ids_d = wp.zeros(m * m, dtype=wp.int32, device=d)
        self._cverts_d = wp.zeros(2 * m * m, dtype=wp.vec3, device=d)
        self._cnormals_d = wp.zeros(2 * m * m, dtype=wp.vec3, device=d)
        self._waves = wp.array(np.asarray(F.RIPPLE_WAVES, dtype=np.float32), dtype=wp.vec3, device=d)
        self._out = [
            {
                "verts": wp.zeros(2 * m * m, dtype=wp.vec3, device="cpu", pinned=pinned),
                "normals": wp.zeros(2 * m * m, dtype=wp.vec3, device="cpu", pinned=pinned),
                "blocks": wp.zeros((nb, nb), dtype=wp.int32, device="cpu", pinned=pinned),
                "sums": wp.zeros(3, dtype=float, device="cpu", pinned=pinned),
                "depth": wp.zeros((n, n), dtype=float, device="cpu", pinned=pinned),
            }
            for _ in range(SETS)
        ]
        self._set_topology = [0] * SETS
        # Version 0 is the empty sheet. Each entry: tris, vertex ids, top count.
        self._topologies = {0: (np.zeros((0, 3), np.int32), np.zeros(0, np.int32), 0)}
        self._topology = 0
        self._topology_key = None
        self._ids_staging = deque(maxlen=KEEP_TOPOLOGIES)
        self._block_seen = np.full((nb, nb), -1.0e9)
        self._commands = {}
        self._bed_host = wp.zeros((n, n), dtype=wp.vec3, device="cpu", pinned=pinned)
        self._bone_host = wp.zeros((n, n), dtype=wp.int32, device="cpu", pinned=pinned)
        self._bed_event = None
        self._bed_cache = None
        self._events = [wp.Event(d) if self._cuda else None for _ in range(SETS)]
        self._inflight = deque()
        self._next = 0
        self._ready = None
        self._max_depth = 0.0
        self._occ_upload = None
        self._occ_key = None
        self._last_read_gen = None
        self._trail = False
        self._dev_ready = True

    # Host views. Reading these waits for the GPU; the render loop does not use them.

    @property
    def depth(self) -> np.ndarray:
        return self._depth_host

    @depth.setter
    def depth(self, value) -> None:
        self._depth_host = np.asarray(value, dtype=np.float64)
        if self._dev_ready:
            self._depth.assign(self._depth_host.astype(np.float32))

    @property
    def bed(self) -> np.ndarray:
        if not self._dev_ready:
            return self._bed_init
        self._bed_synced()
        return self._bed_cache[0]

    @bed.setter
    def bed(self, value) -> None:
        self._bed_init = np.asarray(value, dtype=np.float64)

    @property
    def on_bone(self) -> np.ndarray:
        if not self._dev_ready:
            return self._bone_init
        self._bed_synced()
        return self._bed_cache[1]

    @on_bone.setter
    def on_bone(self, value) -> None:
        self._bone_init = np.asarray(value, dtype=bool)

    @property
    def reach(self) -> np.ndarray:
        if not self._dev_ready:
            return self._reach_init
        self._settle()
        return self._reach.numpy().astype(bool)

    @reach.setter
    def reach(self, value) -> None:
        self._reach_init = np.asarray(value, dtype=bool)

    @property
    def _mask(self) -> np.ndarray:
        if not self._dev_ready:
            return self._mask_init
        self._settle()
        return self._mask_d.numpy().astype(bool)

    @_mask.setter
    def _mask(self, value) -> None:
        self._mask_init = value

    @property
    def draw_bed(self) -> np.ndarray:
        self._settle()
        m = self.m
        bed = self._rounded.numpy().astype(np.float64).reshape(m, m, 3)
        return np.where(self._mask[..., None], bed, np.nan)

    @property
    def any_wet(self) -> bool:
        return self._max_depth >= F.MIN_DEPTH_MM

    def mesh_cpu(self):
        raise NotImplementedError("GpuFilm builds its mesh on the device")

    # Stepping.

    def step(
        self,
        dt: float,
        hit,
        jet_mm: float,
        spread_mm: float,
        depth_mm: float,
        cc_per_s: float,
        frame,
        occupied,
        occ_gen: int,
        incoming=None,
        suction_tip=None,
        suction_radius_mm: float = 0.0,
        down=None,
        life_s: float | None = None,
        smooth: float | None = None,
        ripple: float | None = None,
        rivulets: float | None = None,
    ) -> None:
        self._poll()
        dt = float(max(dt, 0.0))
        self._clock += dt
        self._ripple = float(min(max(ripple or 0.0, 0.0), 1.0))
        self._rivulets = float(min(max(rivulets or 0.0, 0.0), 1.0))
        if down is not None:
            vec = np.asarray(down, dtype=np.float64).reshape(3)
            length = float(np.linalg.norm(vec))
            if length > 1e-6:
                self._down = vec / length
        spread_mm = float(min(max(spread_mm, F.CELL_MM), 0.5 * F.MAX_SPAN_MM))
        depth_mm = float(max(depth_mm, F.MIN_DEPTH_MM))
        occ_gen = int(occ_gen)
        stale = self._clock - self._bed_time >= F.BED_REFRESH_S
        if self._trail and stale and occ_gen == self._occ_gen:
            # The mesher's occupancy can trail the host by a cut. Read once more
            # after the bone stops changing so the bed has the last cut in it.
            self._occ_gen = None
        if hit is not None:
            hit = np.asarray(hit, dtype=np.float64).reshape(3)
            incoming = np.asarray(incoming if incoming is not None else -F.WORLD_DOWN, dtype=np.float64)
            normal = F.surface_normal(frame, occupied, hit, incoming)
            self._move_to(hit, normal, frame, occupied, occ_gen)
        self._set_smooth(smooth)
        if self._origin is None:
            self.volume_mm3 = 0.0
            return
        self._update_topology(hit)
        self._launch(self._inputs(dt, hit, jet_mm, spread_mm, depth_mm, cc_per_s, suction_tip,
                                  suction_radius_mm, life_s))
        if not self.pipelined:
            self._settle()

    def _inputs(self, dt, hit, jet_mm, spread_mm, depth_mm, cc_per_s, tip, tip_mm, life_s) -> np.ndarray:
        p = np.zeros(P_COUNT, dtype=np.float32)
        p[P_DT] = dt
        if hit is not None:
            rel = (hit - self._origin) / MM_TO_SIM
            p[P_JET] = 1.0
            p[P_HX] = float(rel @ self._right)
            p[P_HY] = float(rel @ self._across)
            if dt > 0.0:
                p[P_VOLUME] = F.FLOW_MM3_PER_UNIT * max(0.0, float(cc_per_s)) * dt / (F.CELL_MM * F.CELL_MM)
        p[P_SPREAD] = spread_mm
        p[P_DEPTH] = depth_mm
        p[P_STAMP] = max(0.5 * float(jet_mm), F.IMPACT_MM)
        lip_frac = 0.5
        if life_s is not None:
            lip_frac = min(0.5, 0.5 * F.REF_LIFE_S / max(float(life_s), 0.05))
        normal = self._normal
        p[P_LIFT] = max(0.25, -float(np.dot(normal, self._down)))
        slope = self._down - float(np.dot(self._down, normal)) * normal
        slope_len = float(np.linalg.norm(slope))
        slope = slope / slope_len if slope_len > 1e-3 else np.zeros(3)
        p[P_BUDGET] = F.SLIDE_MM_S * (dt / 2.0)
        for k, (di, dj) in enumerate(F._DIRS):
            downhill = max(0.0, float(np.dot(di * self._right + dj * self._across, slope)))
            p[P_LIP + k] = max(lip_frac, F.LIP_RUN * downhill)
        p[P_PIN] = F.RIVULET_PIN_MM * self._rivulets
        p[P_BUMP] = F.RIVULET_BUMP_MM * self._rivulets
        p[P_RIVULETS] = self._rivulets
        if life_s is not None:
            p[P_DECAY] = math.exp(-dt / max(float(life_s), 0.05))
            p[P_CUT] = F.MIN_DEPTH_MM * 0.25
        else:
            p[P_DECAY] = 1.0
        p[P_LIMIT] = math.pi * spread_mm * spread_mm * depth_mm
        if tip is not None and tip_mm > 0.0:
            p[P_SUCK] = 1.0
            p[P_TIP : P_TIP + 3] = np.asarray(tip, dtype=np.float64).reshape(3)
            p[P_SUCK_R] = float(tip_mm) * MM_TO_SIM
        p[P_DOWN : P_DOWN + 3] = self._down
        p[P_NORMAL : P_NORMAL + 3] = normal
        tangent = self._down - float(np.dot(self._down, normal)) * normal
        length = float(np.linalg.norm(tangent))
        tangent = tangent / length if length > 1e-3 else self._right
        p[P_TANGENT : P_TANGENT + 3] = tangent
        p[P_SIDE : P_SIDE + 3] = np.cross(normal, tangent)
        p[P_CLOCK] = self._clock
        p[P_RIPPLE] = F.RIPPLE_TILT * self._ripple / max(len(F.RIPPLE_WAVES), 1)
        p[P_KEEP] = self._pos_keep
        p[P_FLOW] = 1.0 if dt > 0.0 else 0.0
        p[P_SKIN] = F.SKIN_MM
        p[P_FLOOR] = F.FLOOR_MM
        p[P_RIPPLE_FULL] = F.RIPPLE_FULL_MM
        p[P_ANCHOR : P_ANCHOR + 3] = self._origin - normal * (F.FLOOR_MM * MM_TO_SIM)
        return p

    def _record(self) -> list:
        """The frame's kernel launches, recorded once per smoothing setting.

        Every per-frame value comes from the input array, so the same recorded
        launches replay each frame with almost no Python cost.
        """
        key = (self._height_passes, self._pos_passes)
        cmds = self._commands.get(key)
        if cmds is not None:
            return cmds
        d = self.device
        n, m = self.n, self.m
        P = self._P
        cmds = []

        def add(kernel, dim, inputs):
            cmds.append(wp.launch(kernel, dim, inputs=inputs, device=d, record_cmd=True))

        add(_begin, 1, [P, self._ring, self._sums])
        for r in range(F.SPLASH_RINGS):
            add(_inlet_room, (n, n), [P, self._on_bone, self._reach, self._depth, self._ring, r])
            add(_inlet_apply, (n, n), [P, self._on_bone, self._depth, self._ring, r])
        src, dst = self._depth, self._depth_b
        for _ in range(2):
            add(_flow_moves, (n, n), [P, src, self._on_bone, self._reach, self._bed, self._rough, self._moved, self._give])
            add(_flow_apply, (n, n), [P, src, self._moved, self._give, self._on_bone, self._reach, dst])
            src, dst = dst, src
        add(_blur, (n, n), [P, self._depth, self._on_bone, self._reach, self._depth_b])
        add(_finish, (n, n), [P, self._depth_b, self._depth, self._sums])
        add(_cap_and_suck, (n, n), [P, self._depth, self._on_bone, self._bed, self._sums])

        src, dst = self._depth, self._shown
        for _ in range(self._height_passes):
            add(_display_pass, (n, n), [src, self._on_bone, F.MIN_DEPTH_MM, dst])
            src, dst = dst, (self._spare if dst is self._shown else self._shown)
        add(_refine_keep, (m, m), [src, self._on_bone, self._bed, F.MIN_DEPTH_MM, self._bed_f, self._depth_f, self._mask_d, self._valid_f])
        add(_edges, (m, m), [self._mask_d, self._edge])
        src, dst = self._bed_f, self._round
        for _ in range(F.OUTLINE_PASSES):
            add(_outline_on_shelf, (m, m), [P, src, self._edge, dst])
            src, dst = dst, src
        self._rounded = src
        add(_lift, (m, m), [P, src, self._depth_f, self._mask_d, self._pos, self._verts_d])
        src, dst = self._pos, self._pos_b
        for _ in range(self._pos_passes):
            add(_smooth, (m, m), [P, src, self._mask_d, dst])
            src, dst = dst, src
        add(_place, (m, m), [src, self._verts_d])
        add(_quads, (m, m), [self._mask_d, self._quad])
        add(_shade, (m, m), [P, src, self._quad, self._mask_d, self._depth_f, self._waves, self._normals_d])
        add(_snap_boneless, (m, m), [P, self._valid_f, self._verts_d, self._normals_d])
        add(_block_wet, (m, m), [self._mask_d, BLOCK, self._blocks_d])
        add(_gather, m * m, [P, self._verts_d, self._normals_d, self._ids_d, self._cverts_d, self._cnormals_d])
        self._commands[key] = cmds
        return cmds

    def _frame(self, s: int) -> None:
        """One frame of flow and sheet, queued on the film's stream."""
        cmds = self._record()
        wp.copy(self._P, self._P_host[s])
        self._blocks_d.zero_()
        for cmd in cmds:
            cmd.launch(self.stream)
        out = self._out[s]
        count = 2 * len(self._topologies[self._topology][1])
        if count:
            wp.copy(out["verts"], self._cverts_d, count=count)
            wp.copy(out["normals"], self._cnormals_d, count=count)
        wp.copy(out["blocks"], self._blocks_d)
        wp.copy(out["sums"], self._sums)
        wp.copy(out["depth"], self._depth)

    def _launch(self, inputs: np.ndarray) -> None:
        s = self._next
        self._next = (s + 1) % SETS
        while s in self._inflight:
            self._harvest_oldest(wait=True)
        inputs[P_GATHER] = len(self._topologies[self._topology][1])
        self._P_host[s].numpy()[:] = inputs
        self._set_topology[s] = self._topology
        if not self._cuda:
            self._frame(s)
        else:
            # Plain launches, not a captured graph: the bone mesher's thread reads
            # back through the legacy stream, which CUDA forbids during a capture.
            with wp.ScopedStream(self.stream):
                self._frame(s)
                self.stream.record_event(self._events[s])
        self._inflight.append(s)

    def _done(self, s: int) -> bool:
        return not self._cuda or self._events[s].is_complete

    def _harvest_oldest(self, wait: bool) -> bool:
        s = self._inflight[0]
        if not self._done(s):
            if not wait:
                return False
            wp.synchronize_event(self._events[s])
        self._inflight.popleft()
        out = self._out[s]
        sums = out["sums"].numpy()
        self.volume_mm3 = float(sums[1]) * F.CELL_MM * F.CELL_MM
        self._max_depth = float(sums[2])
        self._depth_host = out["depth"].numpy().astype(np.float64)
        wet = out["blocks"].numpy() != 0
        self._block_seen[_dilate(wet)] = self._clock
        self._ready = s
        return True

    def _poll(self) -> None:
        while self._inflight and self._harvest_oldest(wait=False):
            pass

    def _settle(self) -> None:
        while self._inflight:
            self._harvest_oldest(wait=True)

    def _update_topology(self, hit) -> None:
        """Rebuild the sheet's triangles when its blocks or the bone change."""
        if hit is not None:
            # Blocks around the impact join before the water reaches them.
            rel = (hit - self._origin) / MM_TO_SIM
            center = (self.m - 1) // 2
            fi = center + float(rel @ self._right) / (0.5 * F.CELL_MM)
            fj = center + float(rel @ self._across) / (0.5 * F.CELL_MM)
            bi, bj = int(fi // BLOCK), int(fj // BLOCK)
            nb = self.n_blocks
            self._block_seen[max(bi - 1, 0) : min(bi + 2, nb), max(bj - 1, 0) : min(bj + 2, nb)] = self._clock
        active = self._clock - self._block_seen <= BLOCK_HOLD_S
        key = (active.tobytes(), self._bed_samples)
        if key == self._topology_key:
            return
        self._topology_key = key
        # A new bone read waits here, so no frame draws the new bone with the old
        # triangles; those would bridge the walls of a fresh cut.
        self._bed_synced()
        bed, bone = self._bed_cache
        height = np.nan_to_num((bed - self._origin) @ self._normal) / MM_TO_SIM
        tris, ids, top = _sheet_topology(bone, active, self.m, height)
        version = self._topology + 1
        self._topologies[version] = (tris, ids, top)
        for old in [v for v in self._topologies if 0 < v <= version - KEEP_TOPOLOGIES]:
            del self._topologies[old]
        self._topology = version
        if len(ids):
            # Kept alive until the copy has surely run.
            staging = wp.array(ids, dtype=wp.int32, device="cpu", pinned=self._cuda)
            self._ids_staging.append(staging)

            def upload():
                wp.copy(self._ids_d, staging, count=len(ids))

            self._run(upload)

    def mesh(self):
        """Newest finished sheet. Pipelined, it may be one frame old and never waits longer.

        ``topology`` changes only when the triangles do; between changes only
        the vertices move.
        """
        if self.pipelined:
            self._poll()
            while len(self._inflight) > 1:
                self._harvest_oldest(wait=True)
        else:
            self._settle()
        if self._origin is None or self._ready is None:
            return self._empty()
        version = self._set_topology[self._ready]
        tris, ids, top = self._topologies[version]
        if top == 0 or self._max_depth < F.MIN_DEPTH_MM:
            return self._empty()
        out = self._out[self._ready]
        count = 2 * len(ids)
        self.n_top_tris = top
        self.n_tris = len(tris)
        return {
            "verts": out["verts"].numpy()[:count],
            "tris": tris,
            "normals": out["normals"].numpy()[:count],
            "vert_count": count,
            "vert_ids": ids,
            "topology": version,
        }

    def _empty(self):
        self.n_tris = 0
        self.n_top_tris = 0
        return None

    # Grid moves and bone reads, issued in order on the film's stream.

    def _run(self, fn) -> None:
        if self.stream is None:
            fn()
        else:
            with wp.ScopedStream(self.stream):
                fn()

    def _clear_water(self) -> None:
        self._settle()
        self._max_depth = 0.0
        self.volume_mm3 = 0.0
        self._depth_host = np.zeros((self.n, self.n), dtype=np.float64)

        def clear():
            self._depth.zero_()
            self._reach.zero_()

        self._run(clear)

    def _slide(self, di: int, dj: int) -> None:
        n, d = self.n, self.device

        def slide():
            wp.launch(_shift_float, (n, n), inputs=[self._depth, di, dj, self._depth_b], device=d)
            wp.launch(_shift_int, (n, n), inputs=[self._reach, di, dj, self._reach_b], device=d)
            wp.copy(self._depth, self._depth_b)
            wp.copy(self._reach, self._reach_b)

        self._run(slide)
        self._block_seen = _shift_blocks(self._block_seen, 2 * di / BLOCK, 2 * dj / BLOCK)

    def _device_occupancy(self, frame, occupied, from_mesher_out: list):
        source = self.occupancy_source() if self.occupancy_source is not None else None
        counts = tuple(int(v) for v in frame.counts)
        if source is not None and getattr(source, "occ", None) is not None:
            shape = tuple(int(v) for v in source.shape)
            if shape == counts and source.occ.device == self.device:
                from_mesher_out.append(True)
                return source.occ, shape
        key = (id(occupied), self._occ_gen)
        if self._occ_key != key:
            flat = np.ascontiguousarray(occupied, dtype=np.uint8).reshape(-1)
            self._occ_upload = wp.array(flat, dtype=wp.uint8, device=self.device)
            self._occ_key = key
        return self._occ_upload, tuple(int(v) for v in occupied.shape)

    def _sample_bed(self, frame, occupied) -> None:
        mesher = []
        occ, shape = self._device_occupancy(frame, occupied, mesher)
        rotation = np.asarray(frame.rotation, dtype=np.float64)
        scale = np.asarray(frame.counts, dtype=np.float64) / np.asarray(frame.dimensions, dtype=np.float64)
        to_index = scale[:, None] * rotation.T
        offset = 0.5 * np.asarray(frame.counts, dtype=np.float64) - to_index @ np.asarray(frame.translation, dtype=np.float64)
        steps = int(round((self._above_mm + self._below_mm) / F.BED_STEP_MM))
        n, d = self.n, self.device

        def sample():
            wp.launch(
                _sample, (n, n),
                inputs=[
                    occ, shape[0], shape[1], shape[2],
                    wp.mat33(*to_index.astype(np.float32).reshape(-1)),
                    wp.vec3(*offset.astype(np.float32)),
                    wp.vec3(*self._origin.astype(np.float32)),
                    wp.vec3(*self._right.astype(np.float32)),
                    wp.vec3(*self._across.astype(np.float32)),
                    wp.vec3(*self._normal.astype(np.float32)),
                    float(self._above_mm), steps, float(F.BED_STEP_MM),
                    self._bed, self._on_bone, self._rough, self._depth,
                ],
                device=d,
            )
            wp.copy(self._bed_host, self._bed)
            wp.copy(self._bone_host, self._on_bone)
            if self.stream is not None:
                self._bed_event = self.stream.record_event()

        self._run(sample)
        self._bed_cache = None
        self._trail = bool(mesher) and self._occ_gen != self._last_read_gen
        self._last_read_gen = self._occ_gen
        self._bed_samples += 1
        self._bed_time = self._clock

    def _bed_ready(self) -> bool:
        return self._bed_event is None or self._bed_event.is_complete

    def _bed_synced(self) -> None:
        if self._bed_cache is not None:
            return
        if self._bed_event is not None:
            wp.synchronize_event(self._bed_event)
        bone = self._bone_host.numpy().astype(bool)
        bed = self._bed_host.numpy().astype(np.float64)
        bed[~bone] = np.nan
        self._bed_cache = (bed, bone)

    def _lands(self, hit, frame=None, occupied=None) -> bool:
        rel_mm = (np.asarray(hit, dtype=np.float64) - self._origin) / MM_TO_SIM
        center = self.n // 2
        i = int(np.rint(float(rel_mm @ self._right) / F.CELL_MM + center))
        j = int(np.rint(float(rel_mm @ self._across) / F.CELL_MM + center))
        if not (0 <= i < self.n and 0 <= j < self.n):
            return False
        if frame is None and (self._bed_cache is not None or self._bed_ready()):
            self._bed_synced()
            if not self._bed_cache[1][i, j]:
                return False
            bed = self._bed_cache[0][i, j]
        else:
            # The read just issued is still on the GPU. Its column under the hit
            # is cheap to march here.
            frame = frame if frame is not None else self._frame_ref
            occupied = occupied if occupied is not None else self._occupied_ref
            bed = self._column_bed(i, j, frame, occupied)
            if bed is None:
                return False
        gap_mm = float((hit - bed) @ self._normal) / MM_TO_SIM
        return abs(gap_mm) <= F.LAND_MM

    def _move_to(self, hit, normal, frame, occupied, occ_gen: int) -> None:
        self._frame_ref = frame
        self._occupied_ref = occupied
        super()._move_to(hit, normal, frame, occupied, occ_gen)

    def _column_bed(self, i: int, j: int, frame, occupied):
        center = self.n // 2
        start = (
            self._origin
            + (i - center) * F.CELL_MM * self._right * MM_TO_SIM
            + (j - center) * F.CELL_MM * self._across * MM_TO_SIM
            + self._normal * (self._above_mm * MM_TO_SIM)
        )
        steps = int(round((self._above_mm + self._below_mm) / F.BED_STEP_MM))
        first_index = frame.world_to_index(start)
        step = frame.world_to_index(-self._normal * (F.BED_STEP_MM * MM_TO_SIM)) - frame.world_to_index(np.zeros(3))
        counts = frame.counts.astype(np.int64)
        ks = np.arange(steps + 1, dtype=np.float64)
        ijk = np.floor(first_index + ks[:, None] * step).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < counts), axis=1)
        solid = np.zeros(len(ks), dtype=bool)
        sel = ijk[inside]
        solid[inside] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
        hits = np.flatnonzero(solid)
        if len(hits) == 0:
            return None
        fine = float(hits[0])
        if hits[0] > 0:
            back = np.floor(first_index + (fine - 0.5) * step).astype(np.int64)
            if np.all((back >= 0) & (back < counts)) and occupied[back[0], back[1], back[2]]:
                fine -= 0.5
        return start - self._normal * (fine * F.BED_STEP_MM * MM_TO_SIM)
