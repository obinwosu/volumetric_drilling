"""Warp build of the ImpactFilm sheet.

Same steps as ImpactFilm.mesh_cpu: smoothed depth on the flow grid, a vertex
between every pair of cells, a rounded outline, smoothed positions, area
weighted normals with moving ripples, then a top, a floor, and walls on open
edges. Every vertex of the drawn grid is computed each call; only wet quads
emit triangles. The renderer copies meshes through host memory, so the result
is read back once as NumPy arrays.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from fluid.outlet import MM_TO_SIM


@wp.func
def _in(i: int, j: int, n: int):
    return i >= 0 and i < n and j >= 0 and j < n


@wp.kernel
def _display_pass(
    src: wp.array2d(dtype=wp.float32),
    on_bone: wp.array2d(dtype=wp.int32),
    min_depth: float,
    dst: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n = src.shape[0]
    total = src[i, j]
    near = src[i, j] >= min_depth
    for k in range(4):
        di = 0
        dj = 0
        if k == 0:
            di = 1
        elif k == 1:
            di = -1
        elif k == 2:
            dj = 1
        else:
            dj = -1
        if _in(i + di, j + dj, n):
            v = src[i + di, j + dj]
            total += v
            if v >= min_depth:
                near = True
    # Cells past the edge count as dry neighbours, so the mean is always over five.
    if on_bone[i, j] != 0 and near:
        dst[i, j] = total / 5.0
    else:
        dst[i, j] = 0.0


@wp.kernel
def _refine(
    shown: wp.array2d(dtype=wp.float32),
    on_bone: wp.array2d(dtype=wp.int32),
    bed: wp.array2d(dtype=wp.vec3),
    min_depth: float,
    bed_f: wp.array2d(dtype=wp.vec3),
    depth_f: wp.array2d(dtype=wp.float32),
    mask_f: wp.array2d(dtype=wp.int32),
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
    bed_f[I, J] = acc * inv
    depth_f[I, J] = depth * inv
    # Half-way vertices join an edge; a corner touching one wet cell does not.
    if count >= need and wet / need >= 0.5 and depth * inv >= min_depth:
        mask_f[I, J] = 1
    else:
        mask_f[I, J] = 0


@wp.kernel
def _edges(mask: wp.array2d(dtype=wp.int32), edge: wp.array2d(dtype=wp.int32)):
    I, J = wp.tid()
    m = mask.shape[0]
    out = 0
    if mask[I, J] != 0:
        if not _in(I + 1, J, m) or mask[I + 1, J] == 0:
            out = 1
        if not _in(I - 1, J, m) or mask[I - 1, J] == 0:
            out = 1
        if not _in(I, J + 1, m) or mask[I, J + 1] == 0:
            out = 1
        if not _in(I, J - 1, m) or mask[I, J - 1] == 0:
            out = 1
    edge[I, J] = out


@wp.kernel
def _outline_pass(
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
    acc = wp.vec3(0.0, 0.0, 0.0)
    count = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            if _in(I + di, J + dj, m):
                if edge[I + di, J + dj] != 0:
                    acc += src[I + di, J + dj]
                    count += 1.0
    if count >= 2.0:
        dst[I, J] = 0.5 * here + 0.5 * (acc / count)


@wp.kernel
def _lift(
    bed_f: wp.array2d(dtype=wp.vec3),
    depth_f: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.int32),
    normal: wp.vec3,
    skin_mm: float,
    floor_mm: float,
    mm: float,
    pos: wp.array2d(dtype=wp.vec3),
    floor: wp.array2d(dtype=wp.vec3),
):
    I, J = wp.tid()
    height = float(0.0)
    if mask[I, J] != 0:
        height = depth_f[I, J] + skin_mm
    pos[I, J] = bed_f[I, J] + normal * (height * mm)
    floor[I, J] = bed_f[I, J] - normal * (floor_mm * mm)


@wp.kernel
def _smooth_pass(
    src: wp.array2d(dtype=wp.vec3),
    mask: wp.array2d(dtype=wp.int32),
    keep: float,
    dst: wp.array2d(dtype=wp.vec3),
):
    I, J = wp.tid()
    m = src.shape[0]
    here = src[I, J]
    dst[I, J] = here
    if mask[I, J] == 0:
        return
    acc = here
    count = float(1.0)
    for k in range(4):
        di = 0
        dj = 0
        if k == 0:
            di = 1
        elif k == 1:
            di = -1
        elif k == 2:
            dj = 1
        else:
            dj = -1
        if _in(I + di, J + dj, m):
            if mask[I + di, J + dj] != 0:
                acc += src[I + di, J + dj]
                count += 1.0
    dst[I, J] = keep * here + (1.0 - keep) * (acc / count)


@wp.kernel
def _quads(mask: wp.array2d(dtype=wp.int32), quad: wp.array2d(dtype=wp.int32)):
    i, j = wp.tid()
    m = mask.shape[0]
    out = 0
    if i < m - 1 and j < m - 1:
        if mask[i, j] != 0 and mask[i + 1, j] != 0 and mask[i, j + 1] != 0 and mask[i + 1, j + 1] != 0:
            out = 1
    quad[i, j] = out


@wp.func
def _face(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    return wp.cross(p1 - p0, p2 - p0)


@wp.func
def _quad_at(quad: wp.array2d(dtype=wp.int32), i: int, j: int):
    m = quad.shape[0]
    if _in(i, j, m):
        return quad[i, j] != 0
    return False


@wp.kernel
def _shade(
    pos: wp.array2d(dtype=wp.vec3),
    quad: wp.array2d(dtype=wp.int32),
    mask: wp.array2d(dtype=wp.int32),
    depth_f: wp.array2d(dtype=wp.float32),
    normal: wp.vec3,
    tangent: wp.vec3,
    side: wp.vec3,
    waves: wp.array(dtype=wp.vec3),
    clock: float,
    ripple_amount: float,
    ripple_full_mm: float,
    mm: float,
    out: wp.array2d(dtype=wp.vec3),
):
    I, J = wp.tid()
    # Quad (i, j) is triangles (i,j)(i+1,j)(i,j+1) and (i+1,j)(i+1,j+1)(i,j+1).
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
    n = normal
    if length >= 1.0e-10 and mask[I, J] != 0:
        n = acc / length
    if ripple_amount > 0.0 and mask[I, J] != 0:
        world_mm = pos[I, J] / mm
        tilt = wp.vec3(0.0, 0.0, 0.0)
        for w in range(waves.shape[0]):
            wave = waves[w]
            direction = wp.cos(wave[2]) * tangent + wp.sin(wave[2]) * side
            k = 2.0 * wp.pi / wave[0]
            phase = k * (wp.dot(world_mm, direction) - wave[1] * clock)
            tilt += wp.cos(phase) * direction
        weight = wp.clamp(depth_f[I, J] / ripple_full_mm, 0.0, 1.0)
        n = n - (ripple_amount * weight) * tilt
        n = n / wp.max(wp.length(n), 1.0e-8)
    out[I, J] = n


@wp.kernel
def _emit(
    quad: wp.array2d(dtype=wp.int32),
    top_count: int,
    counts: wp.array(dtype=wp.int32),
    top_tris: wp.array(dtype=wp.vec3i),
    floor_tris: wp.array(dtype=wp.vec3i),
    wall_tris: wp.array(dtype=wp.vec3i),
):
    i, j = wp.tid()
    m = quad.shape[0]
    if quad[i, j] == 0:
        return
    a = i * m + j
    b = (i + 1) * m + j
    c = i * m + (j + 1)
    d = (i + 1) * m + (j + 1)
    # cross(right, across) is the outward normal, so this winding faces out of the bone.
    slot = wp.atomic_add(counts, 0, 2)
    top_tris[slot] = wp.vec3i(a, b, c)
    top_tris[slot + 1] = wp.vec3i(b, d, c)
    slot = wp.atomic_add(counts, 1, 2)
    floor_tris[slot] = wp.vec3i(c + top_count, b + top_count, a + top_count)
    floor_tris[slot + 1] = wp.vec3i(c + top_count, d + top_count, b + top_count)
    # Each top quad runs a -> b -> d -> c. An edge with no quad across it is open,
    # and its wall uses the edge in the opposite direction to the top and floor.
    for e in range(4):
        u = a
        v = b
        open_edge = False
        if e == 0:
            open_edge = not _quad_at(quad, i, j - 1)
        elif e == 1:
            u = b
            v = d
            open_edge = not _quad_at(quad, i + 1, j)
        elif e == 2:
            u = d
            v = c
            open_edge = not _quad_at(quad, i, j + 1)
        else:
            u = c
            v = a
            open_edge = not _quad_at(quad, i - 1, j)
        if open_edge:
            slot = wp.atomic_add(counts, 2, 2)
            wall_tris[slot] = wp.vec3i(v, u, u + top_count)
            wall_tris[slot + 1] = wp.vec3i(v, u + top_count, v + top_count)


class FilmMeshGPU:
    """Device buffers for one ImpactFilm. Bed and bone mask upload on re-read only."""

    def __init__(self, n: int, m: int, device=None):
        self.device = wp.get_device(device)
        self.n = n
        self.m = m
        d = self.device
        self._depth = wp.zeros((n, n), dtype=wp.float32, device=d)
        self._shown = wp.zeros((n, n), dtype=wp.float32, device=d)
        self._spare = wp.zeros((n, n), dtype=wp.float32, device=d)
        self._on_bone = wp.zeros((n, n), dtype=wp.int32, device=d)
        self._bed = wp.zeros((n, n), dtype=wp.vec3, device=d)
        self._bed_version = None
        self._bed_f = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._depth_f = wp.zeros((m, m), dtype=wp.float32, device=d)
        self._mask = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._edge = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._quad = wp.zeros((m, m), dtype=wp.int32, device=d)
        self._round = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._pos = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._pos_b = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._floor = wp.zeros((m, m), dtype=wp.vec3, device=d)
        self._normals = wp.zeros((m, m), dtype=wp.vec3, device=d)
        quads = (m - 1) * (m - 1)
        self._counts = wp.zeros(3, dtype=wp.int32, device=d)
        self._top = wp.zeros(2 * quads, dtype=wp.vec3i, device=d)
        self._floor_tris = wp.zeros(2 * quads, dtype=wp.vec3i, device=d)
        self._walls = wp.zeros(8 * quads, dtype=wp.vec3i, device=d)
        self._waves = None

    def build(self, film, waves, outline_passes: int, consts: dict):
        """Run the sheet build and read back what the renderer needs."""
        n, m, d = self.n, self.m, self.device
        if self._bed_version != film._bed_samples:
            self._bed.assign(np.nan_to_num(film.bed, nan=0.0).astype(np.float32))
            self._on_bone.assign(film.on_bone.astype(np.int32))
            self._bed_version = film._bed_samples
        if self._waves is None:
            self._waves = wp.array(np.asarray(waves, dtype=np.float32), dtype=wp.vec3, device=d)
        self._depth.assign(film.depth.astype(np.float32))
        min_depth = float(consts["min_depth"])
        src, dst = self._depth, self._shown
        for _ in range(film._height_passes):
            wp.launch(_display_pass, (n, n), inputs=[src, self._on_bone, min_depth, dst], device=d)
            src, dst = dst, (self._spare if dst is self._shown else self._shown)
        shown = src
        wp.launch(
            _refine, (m, m),
            inputs=[shown, self._on_bone, self._bed, min_depth, self._bed_f, self._depth_f, self._mask],
            device=d,
        )
        wp.launch(_edges, (m, m), inputs=[self._mask, self._edge], device=d)
        src, dst = self._bed_f, self._round
        for _ in range(outline_passes):
            wp.launch(_outline_pass, (m, m), inputs=[src, self._edge, dst], device=d)
            src, dst = dst, src
        self._rounded_bed = src
        normal = wp.vec3(*film._normal.astype(np.float32))
        mm = float(MM_TO_SIM)
        wp.launch(
            _lift, (m, m),
            inputs=[src, self._depth_f, self._mask, normal, float(consts["skin_mm"]),
                    float(consts["floor_mm"]), mm, self._pos, self._floor],
            device=d,
        )
        src, dst = self._pos, self._pos_b
        for _ in range(film._pos_passes):
            wp.launch(_smooth_pass, (m, m), inputs=[src, self._mask, float(film._pos_keep), dst], device=d)
            src, dst = dst, src
        pos = src
        wp.launch(_quads, (m, m), inputs=[self._mask, self._quad], device=d)
        tangent = film._down - float(np.dot(film._down, film._normal)) * film._normal
        length = float(np.linalg.norm(tangent))
        tangent = tangent / length if length > 1e-3 else film._right
        side = np.cross(film._normal, tangent)
        amount = consts["ripple_tilt"] * film._ripple / max(len(waves), 1)
        wp.launch(
            _shade, (m, m),
            inputs=[pos, self._quad, self._mask, self._depth_f, normal,
                    wp.vec3(*tangent.astype(np.float32)), wp.vec3(*side.astype(np.float32)),
                    self._waves, float(film._clock), float(amount), float(consts["ripple_full_mm"]),
                    mm, self._normals],
            device=d,
        )
        self._counts.zero_()
        wp.launch(
            _emit, (m, m),
            inputs=[self._quad, m * m, self._counts, self._top, self._floor_tris, self._walls],
            device=d,
        )
        counts = self._counts.numpy()
        top_n, floor_n, wall_n = int(counts[0]), int(counts[1]), int(counts[2])
        if top_n == 0:
            return None
        tris = np.concatenate(
            [self._top[:top_n].numpy(), self._floor_tris[:floor_n].numpy(), self._walls[:wall_n].numpy()]
        )
        return {
            "pos": pos.numpy().reshape(-1, 3),
            "floor": self._floor.numpy().reshape(-1, 3),
            "normals": self._normals.numpy().reshape(-1, 3),
            "tris": tris,
            "top_tris": top_n,
            "mask": self._mask.numpy().astype(bool),
        }

    def draw_bed(self) -> np.ndarray:
        """Rounded bed under the last build. Tests compare the floor against it."""
        return self._rounded_bed.numpy().astype(np.float64)
