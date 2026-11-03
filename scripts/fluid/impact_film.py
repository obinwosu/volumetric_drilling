"""Post-impact water sheet on the bone the jet is touching.

The jet width is not this sheet's width. Water is stamped on a patch the
size of the jet, then it runs downhill in world −Z, spreads up to its own
radius, and is deleted at the cavity lip or at that radius. The mesh is one
height sheet on the occupancy surface. It does not remove voxels.
"""

from __future__ import annotations

import math

import numpy as np

from fluid.outlet import MM_TO_SIM

# ADF/world/world.yaml writes gravity on z with a negative sign, and the
# gaze marker builds its camera up as (0, 0, 1). Down is world −Z.
WORLD_DOWN = np.array((0.0, 0.0, -1.0), dtype=np.float64)


def film_gravity(settings) -> np.ndarray:
    """Unit downhill direction. Missing or zero input stays world −Z."""
    raw = None if settings is None else settings.get("gravity")
    vec = np.asarray(WORLD_DOWN if raw is None else raw, dtype=np.float64).reshape(-1)
    if vec.shape != (3,):
        return WORLD_DOWN.copy()
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return WORLD_DOWN.copy()
    return vec / length

CELL_MM = 0.5
# Full width of the sheet. It stays put on the bone while the jet moves, so
# films from earlier aims keep draining where they are inside this square.
MAX_SPAN_MM = 64.0
SPREAD_MM = 20.0
DEPTH_MM = 3.0
MIN_DEPTH_MM = 0.012
SKIN_MM = 0.15
# How fast a thicker cell pushes water toward a lower neighbour.
SLIDE_MM_S = 35.0
# Life is compared with this. It only sets how fast water leaves over the lip;
# it must not slow the spreading, or a long life makes the sheet crawl.
REF_LIFE_S = 0.1
# Suction. Water within the capture radius of the tip goes at once, measured to
# the water surface, so a tip held in a pool takes it. Out to SUCTION_REACH times
# that radius it is drawn out each frame, fastest at the tip; the film's flow
# brings the rest of the pool downhill into the gap, so a pool drains from its
# lowest point and its triangles disappear as the cells empty.
SUCTION_REACH = 2.5
SUCTION_DRAW_PER_S = 12.0
# Inlet volume per unit of the Flow slider, mm^3/s. 80 covers a 15 mm
# radius half a millimetre deep in about four seconds.
FLOW_MM3_PER_UNIT = 1.2
# The jet splashes out to at least this radius where it lands.
IMPACT_MM = 1.0
SPLASH_RINGS = 8
# Bed search along the sheet normal. When the jet lands in a drilled hole,
# the rim is well above the hit, so the search starts this far up or the
# sheet around the hole sits inside the bone. Below reaches bone that curves
# away from the plane of the grid toward its edges.
BED_ABOVE_MM = 12.0
BED_BELOW_MM = 10.0
BED_STEP_MM = 0.5
# The grid only slides once the jet is this far from its centre, so water
# laid down earlier is not moved or cut off by small aims.
REANCHOR_CELLS = 32
# The bed under the hit must be within this of the hit, or the bone is re-read.
LAND_MM = 1.0
# A dry sheet takes the new face's axis past this turn. A wet one keeps its
# axis, so water already down is never dropped for a change of aim.
KEEP_AXIS_DOT = 0.3
# Share of a cell that runs over a lip per pass when the lip is straight downhill.
LIP_RUN = 0.25
# A bone re-read costs ~14 ms on the 64 mm grid. While drilling, the sheet
# follows the new cavity at most this often.
BED_REFRESH_S = 0.25
# The sheet is a closed body of water: a top, a floor under the bone
# surface, and walls at the edge. Refraction needs an inside and an outside.
FLOOR_MM = 0.3
# Moving ripples in the normals. Wavelength mm, speed mm/s, turn from downhill.
RIPPLE_WAVES = ((2.2, 12.0, 0.0), (1.3, 15.0, 0.44), (3.1, 9.0, -0.61))
RIPPLE_TILT = 0.14
RIPPLE_FULL_MM = 0.4
# Rivulets: world-fixed roughness the water runs around, and a wet front
# that must be this thick before it advances onto dry bone.
RIVULET_CELL_MM = 1.6
RIVULET_BUMP_MM = 0.45
RIVULET_PIN_MM = 0.15
# Passes that round the stair-step outline of the drawn sheet.
OUTLINE_PASSES = 3
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_RING = _DIRS + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _lattice(ix, iy, iz) -> np.ndarray:
    """Repeatable value in [-1, 1] per integer lattice point."""
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float64) / 32767.5 - 1.0


def _value_noise(points_mm) -> np.ndarray:
    """Smooth noise fixed to the world, so it does not move with the grid."""
    p = np.asarray(points_mm, dtype=np.float64) / RIVULET_CELL_MM
    base = np.floor(p)
    f = p - base
    f = f * f * (3.0 - 2.0 * f)
    b = base.astype(np.int64)
    out = np.zeros(p.shape[:-1], dtype=np.float64)
    for dx in (0, 1):
        wx = f[..., 0] if dx else 1.0 - f[..., 0]
        for dy in (0, 1):
            wy = f[..., 1] if dy else 1.0 - f[..., 1]
            for dz in (0, 1):
                wz = f[..., 2] if dz else 1.0 - f[..., 2]
                out += wx * wy * wz * _lattice(b[..., 0] + dx, b[..., 1] + dy, b[..., 2] + dz)
    return out


def _shift(field, di: int, dj: int, fill):
    """Value that lives at (i+di, j+dj), with `fill` past the edge."""
    out = np.full(field.shape, fill, dtype=field.dtype)
    n, m = field.shape[:2]
    if abs(di) >= n or abs(dj) >= m:
        return out
    if di >= 0:
        dst_i, src_i = slice(0, n - di), slice(di, n)
    else:
        dst_i, src_i = slice(-di, n), slice(0, n + di)
    if dj >= 0:
        dst_j, src_j = slice(0, m - dj), slice(dj, m)
    else:
        dst_j, src_j = slice(-dj, m), slice(0, m + dj)
    out[dst_i, dst_j] = field[src_i, src_j]
    return out


def _refine(field, valid):
    """Half-cell grid: coarse values at even indices, neighbour means between.

    Only valid coarse cells contribute. `whole` is True where every coarse
    cell under a fine vertex is valid.
    """
    a, b = valid.shape
    flat = field.ndim == 2
    values = field[..., None] if flat else field
    weight = valid.astype(np.float64)
    values = np.where(valid[..., None], values, 0.0)
    out = np.zeros((2 * a - 1, 2 * b - 1, values.shape[-1]), dtype=np.float64)
    count = np.zeros((2 * a - 1, 2 * b - 1), dtype=np.float64)
    need = np.ones_like(count)
    out[0::2, 0::2] = values
    count[0::2, 0::2] = weight
    out[1::2, 0::2] = values[:-1] + values[1:]
    count[1::2, 0::2] = weight[:-1] + weight[1:]
    need[1::2, 0::2] = 2.0
    out[0::2, 1::2] = values[:, :-1] + values[:, 1:]
    count[0::2, 1::2] = weight[:, :-1] + weight[:, 1:]
    need[0::2, 1::2] = 2.0
    out[1::2, 1::2] = values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]
    count[1::2, 1::2] = weight[:-1, :-1] + weight[1:, :-1] + weight[:-1, 1:] + weight[1:, 1:]
    need[1::2, 1::2] = 4.0
    out /= np.maximum(count, 1.0)[..., None]
    if flat:
        out = out[..., 0]
    return out, count >= need


def _basis(normal: np.ndarray):
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    right = np.cross(WORLD_DOWN, normal)
    if float(np.linalg.norm(right)) < 0.25:
        right = np.cross(np.array((1.0, 0.0, 0.0)), normal)
    if float(np.linalg.norm(right)) < 0.25:
        right = np.cross(np.array((0.0, 1.0, 0.0)), normal)
    right = right / float(np.linalg.norm(right))
    across = np.cross(normal, right)
    across = across / float(np.linalg.norm(across))
    return right, across


def surface_normal(frame, occupied, hit, incoming) -> np.ndarray:
    """Outward normal at a hit. Faces back along the incoming ray."""
    ijk = np.floor(frame.world_to_index(np.asarray(hit, dtype=np.float64))).astype(np.int32)
    counts = frame.counts.astype(np.int32)
    gradient = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        for step, weight in ((-1, -1.0), (1, 1.0)):
            sample = ijk.copy()
            sample[axis] += step
            if np.any(sample < 0) or np.any(sample >= counts):
                continue
            if occupied[int(sample[0]), int(sample[1]), int(sample[2])]:
                gradient[axis] += weight
    incoming = np.asarray(incoming, dtype=np.float64)
    incoming = incoming / max(float(np.linalg.norm(incoming)), 1e-8)
    facing = -incoming
    if float(np.linalg.norm(gradient)) < 1e-6:
        normal = facing
    else:
        # Occupancy increases into the bone, so the air side is −gradient.
        # A stair face can be almost edge-on to the jet. That sheet is a line
        # on screen, so a sideways face gives way to the direction of the jet.
        normal = -gradient
        normal = normal / float(np.linalg.norm(normal))
        if float(np.dot(normal, facing)) < 0.5:
            normal = facing
    if float(np.dot(normal, incoming)) > 0.0:
        normal = -normal
    return normal


class ImpactFilm:
    """Fixed-size sheet around the latest impact. Topology stays n×n."""

    def __init__(self):
        span = int(math.floor(MAX_SPAN_MM / CELL_MM))
        if span % 2 == 0:
            span += 1
        self.n = span
        self.depth = np.zeros((span, span), dtype=np.float64)
        self.bed = np.full((span, span, 3), np.nan, dtype=np.float64)
        self.on_bone = np.zeros((span, span), dtype=bool)
        # Cells within the spread of any impact since the grid was placed.
        # Water already down keeps running there after the jet moves on.
        self.reach = np.zeros((span, span), dtype=bool)
        self._above_mm = BED_ABOVE_MM
        self._below_mm = BED_BELOW_MM
        self._origin = None
        self._hit = None
        self._normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        self._down = WORLD_DOWN.copy()
        self._right = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        self._across = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        self._occ_gen = None
        self._bed_samples = 0
        self._clock = 0.0
        self._bed_time = -1.0e9
        self.volume_mm3 = 0.0
        # The drawn grid has a vertex between every pair of water cells, so the
        # outline and ripples are half a cell fine while the flow stays coarse.
        fine = 2 * (span - 1) + 1
        self.m = fine
        # Top grid then floor grid. Same count every frame.
        self.verts = np.zeros((2 * fine * fine, 3), dtype=np.float32)
        self.normals = np.tile(np.array((0.0, 0.0, 1.0), dtype=np.float32), (2 * fine * fine, 1))
        quads = (fine - 1) * (fine - 1)
        # Top, floor, and up to four wall edges per quad.
        self.tris = np.zeros((quads * 12, 3), dtype=np.int32)
        self.n_tris = 0
        self.n_top_tris = 0
        self._mask = np.zeros((fine, fine), dtype=bool)
        self._draw_bed = np.full((fine, fine, 3), np.nan, dtype=np.float64)
        self._gpu = None
        self._built_on_gpu = False
        self._height_passes = 3
        self._pos_passes = 2
        self._pos_keep = 0.45
        self._rough = np.zeros((span, span), dtype=np.float64)
        self._ripple = 0.0
        self._rivulets = 0.0

    def placeholder(self) -> dict:
        """Full buffer, one triangle. The renderer keeps this vertex count."""
        self.tris[0, 0] = 0
        self.tris[0, 1] = 1
        self.tris[0, 2] = self.m
        return {
            "verts": self.verts,
            "tris": self.tris[:1],
            "normals": self.normals,
            "vert_count": 2 * self.m * self.m,
        }

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
        dt = float(max(dt, 0.0))
        self._clock += dt
        self._ripple = float(min(max(ripple or 0.0, 0.0), 1.0))
        self._rivulets = float(min(max(rivulets or 0.0, 0.0), 1.0))
        if down is not None:
            vec = np.asarray(down, dtype=np.float64).reshape(3)
            length = float(np.linalg.norm(vec))
            if length > 1e-6:
                self._down = vec / length
        spread_mm = float(min(max(spread_mm, CELL_MM), 0.5 * MAX_SPAN_MM))
        depth_mm = float(max(depth_mm, MIN_DEPTH_MM))
        if hit is not None:
            hit = np.asarray(hit, dtype=np.float64).reshape(3)
            incoming = np.asarray(
                incoming if incoming is not None else -WORLD_DOWN, dtype=np.float64
            )
            normal = surface_normal(frame, occupied, hit, incoming)
            self._move_to(hit, normal, frame, occupied, int(occ_gen))
            self.reach |= self.on_bone & (self._offsets(hit) <= spread_mm)
            self._inlet(hit, float(jet_mm), spread_mm, depth_mm, float(cc_per_s), dt)
        self._set_smooth(smooth)
        if self._origin is None:
            self.volume_mm3 = 0.0
            return
        self._flow(dt, spread_mm, depth_mm, life_s)
        if life_s is not None:
            # Water runs off and soaks in over the life setting, jet on or off.
            self.depth *= math.exp(-dt / max(float(life_s), 0.05))
            self.depth[self.depth < MIN_DEPTH_MM * 0.25] = 0.0
        self._cap(spread_mm, depth_mm)
        self._suction(suction_tip, suction_radius_mm, dt)
        self.volume_mm3 = float(self.depth.sum() * CELL_MM * CELL_MM)

    def _set_smooth(self, smooth) -> None:
        if smooth is None:
            self._height_passes = 3
            self._pos_passes = 2
            self._pos_keep = 0.45
            return
        amount = float(min(max(float(smooth), 0.0), 1.0))
        self._height_passes = int(round(1.0 + 8.0 * amount))
        self._pos_passes = int(round(6.0 * amount))
        self._pos_keep = 0.75 - 0.6 * amount

    @property
    def any_wet(self) -> bool:
        return float(self.depth.max()) >= MIN_DEPTH_MM

    def covers(self, points) -> np.ndarray:
        """True on bone the sheet is wetting. Dust uses this and does not resettle there."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        covered = np.zeros(len(points), dtype=bool)
        if self._origin is None or len(points) == 0 or not self.any_wet:
            return covered
        rel_mm = (points - self._origin) / MM_TO_SIM
        x = rel_mm @ self._right
        y = rel_mm @ self._across
        center = self.n // 2
        ii = np.rint(x / CELL_MM + center).astype(np.int32)
        jj = np.rint(y / CELL_MM + center).astype(np.int32)
        inside = (ii >= 0) & (ii < self.n) & (jj >= 0) & (jj < self.n)
        if np.any(inside):
            ci, cj = ii[inside], jj[inside]
            # Height above the bone under that cell, not above the grid plane.
            gap = ((points[inside] - np.nan_to_num(self.bed[ci, cj], nan=np.inf)) @ self._normal) / MM_TO_SIM
            near = np.abs(gap) < 1.5
            covered[inside] = near & (self.depth[ci, cj] >= MIN_DEPTH_MM)
        return covered

    def mesh(self):
        """Closed body of water, built on the GPU. Same vertex count every call."""
        if self._origin is None or float(self.depth.sum()) <= 0.0:
            return self._empty()
        if self._gpu is None:
            from fluid.film_mesh_gpu import FilmMeshGPU

            self._gpu = FilmMeshGPU(self.n, self.m)
        built = self._gpu.build(
            self,
            RIPPLE_WAVES,
            OUTLINE_PASSES,
            {
                "min_depth": MIN_DEPTH_MM,
                "skin_mm": SKIN_MM,
                "floor_mm": FLOOR_MM,
                "ripple_tilt": RIPPLE_TILT,
                "ripple_full_mm": RIPPLE_FULL_MM,
            },
        )
        if built is None:
            return self._empty()
        top = self.m * self.m
        self.verts[:top] = built["pos"]
        self.verts[top:] = built["floor"]
        self.normals[:top] = built["normals"]
        self.normals[top:] = -self._normal.astype(np.float32)
        total = len(built["tris"])
        self.tris[:total] = built["tris"]
        self.n_top_tris = int(built["top_tris"])
        self.n_tris = total
        self._mask = built["mask"]
        self._built_on_gpu = True
        return {
            "verts": self.verts,
            "tris": self.tris[: self.n_tris],
            "normals": self.normals,
            "vert_count": 2 * top,
        }

    @property
    def draw_bed(self) -> np.ndarray:
        """Bed under the drawn top after the outline is rounded. NaN off the water."""
        if not self._built_on_gpu:
            return self._draw_bed
        bed = self._gpu.draw_bed().reshape(self.m, self.m, 3)
        return np.where(self._mask[..., None], bed, np.nan)

    def mesh_cpu(self):
        """NumPy reference for mesh(). Tests check the GPU build against it."""
        self._built_on_gpu = False
        if self._origin is None or float(self.depth.sum()) <= 0.0:
            return self._empty()
        shown = self._display_depth()
        wet = self.on_bone & (shown >= MIN_DEPTH_MM)
        if int(wet.sum()) < 3:
            return self._empty()
        # Build only around the water. Most of the grid is dry bone.
        rows = np.flatnonzero(wet.any(axis=1))
        cols = np.flatnonzero(wet.any(axis=0))
        i0, i1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, self.n)
        j0, j1 = max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, self.n)
        box = (slice(i0, i1), slice(j0, j1))
        on_bone = self.on_bone[box]
        bed, whole = _refine(np.nan_to_num(self.bed[box], nan=0.0), on_bone)
        depth, _ = _refine(shown[box], on_bone)
        share, _ = _refine(wet[box].astype(np.float64), np.ones_like(on_bone))
        # Half-way vertices join an edge; a corner touching one wet cell does not,
        # so a stair step becomes a diagonal before the outline is rounded.
        mask = whole & (share >= 0.5) & (depth >= MIN_DEPTH_MM)
        bed = self._round_outline(bed, mask)
        height = np.where(mask, depth + SKIN_MM, 0.0)
        pos = bed + self._normal * (height[..., None] * MM_TO_SIM)
        pos = self._smooth_positions(pos, mask)
        count = self._fill_tris(mask)
        if count == 0:
            return self._empty()
        normals = self._shade(pos, mask, count)
        normals = self._ripples(pos, mask, depth, normals)
        total = self._close(count, mask.shape)
        floor = bed - self._normal * (FLOOR_MM * MM_TO_SIM)

        m = self.m
        top = m * m
        a, b = mask.shape
        fine = (slice(2 * i0, 2 * i0 + a), slice(2 * j0, 2 * j0 + b))
        self.verts[:top].reshape(m, m, 3)[fine] = pos
        self.verts[top:].reshape(m, m, 3)[fine] = floor
        self.normals[:top].reshape(m, m, 3)[fine] = normals
        self.normals[top:] = -self._normal.astype(np.float32)
        self._mask[:] = False
        self._mask[fine] = mask
        self._draw_bed[:] = np.nan
        self._draw_bed[fine] = np.where(mask[..., None], bed, np.nan)
        # Box-local vertex ids to ids in the full drawn grid.
        tris = self.tris[:total]
        below = tris >= a * b
        local = np.where(below, tris - a * b, tris)
        tris[:] = (local // b + 2 * i0) * m + (local % b + 2 * j0) + np.where(below, top, 0)
        self.n_top_tris = count
        self.n_tris = total
        return {
            "verts": self.verts,
            "tris": self.tris[: self.n_tris],
            "normals": self.normals,
            "vert_count": 2 * top,
        }

    def _empty(self):
        self.n_tris = 0
        self.n_top_tris = 0
        self._mask[:] = False
        return None

    def _round_outline(self, bed, mask):
        """Pull each edge vertex toward its edge neighbours so the outline curves."""
        inner = mask.copy()
        for di, dj in _DIRS:
            inner &= _shift(mask, di, dj, False)
        edge = mask & ~inner
        pos = bed
        for _ in range(OUTLINE_PASSES):
            total = np.zeros_like(pos)
            count = np.zeros(mask.shape, dtype=np.float64)
            for di, dj in _RING:
                beside = _shift(edge, di, dj, False)
                total += np.where(beside[..., None], _shift(pos, di, dj, 0.0), 0.0)
                count += beside
            move = edge & (count >= 2.0)
            mean = total / np.maximum(count, 1.0)[..., None]
            pos = np.where(move[..., None], 0.5 * pos + 0.5 * mean, pos)
        return pos

    def _close(self, count: int, shape) -> int:
        """Floor under the top, then walls on every open top edge."""
        rows, n = shape
        top = rows * n
        tris = self.tris
        tris[count : 2 * count] = tris[:count, ::-1] + top
        end = 2 * count
        quad = self._quad
        padded = np.zeros((rows + 1, n + 1), dtype=bool)
        padded[1:rows, 1:n] = quad
        ii, jj = np.nonzero(quad)
        pi, pj = ii + 1, jj + 1
        a = ii * n + jj
        b = (ii + 1) * n + jj
        c = ii * n + (jj + 1)
        d = (ii + 1) * n + (jj + 1)
        # Each top quad runs a -> b -> d -> c. An edge with no quad across it is open.
        for open_edge, u, v in (
            (~padded[pi, pj - 1], a, b),
            (~padded[pi + 1, pj], b, d),
            (~padded[pi, pj + 1], d, c),
            (~padded[pi - 1, pj], c, a),
        ):
            u = u[open_edge]
            v = v[open_edge]
            k = len(u)
            if k == 0:
                continue
            # The wall uses each edge in the opposite direction to the top and floor.
            tris[end : end + k, 0] = v
            tris[end : end + k, 1] = u
            tris[end : end + k, 2] = u + top
            tris[end + k : end + 2 * k, 0] = v
            tris[end + k : end + 2 * k, 1] = u + top
            tris[end + k : end + 2 * k, 2] = v + top
            end += 2 * k
        return end

    def _ripples(self, pos, mask, shown, base):
        """Tilt the top normals with small waves that travel downhill."""
        if self._ripple <= 0.0:
            return base
        normal = self._normal
        tangent = self._down - float(np.dot(self._down, normal)) * normal
        length = float(np.linalg.norm(tangent))
        tangent = tangent / length if length > 1e-3 else self._right
        side = np.cross(normal, tangent)
        world_mm = np.nan_to_num(pos, nan=0.0) / MM_TO_SIM
        tilt = np.zeros(pos.shape, dtype=np.float64)
        for wave_mm, speed, turn in RIPPLE_WAVES:
            direction = math.cos(turn) * tangent + math.sin(turn) * side
            k = 2.0 * math.pi / wave_mm
            phase = k * (world_mm @ direction - speed * self._clock)
            tilt += np.cos(phase)[..., None] * direction
        weight = np.clip(shown / RIPPLE_FULL_MM, 0.0, 1.0) * mask
        amount = (RIPPLE_TILT * self._ripple / len(RIPPLE_WAVES)) * weight
        bent = base - amount[..., None] * tilt
        bent /= np.maximum(np.linalg.norm(bent, axis=2, keepdims=True), 1e-8)
        return bent

    def _move_to(self, hit, normal, frame, occupied, occ_gen: int) -> None:
        keep = self._origin is not None and (
            float(np.dot(normal, self._normal)) >= KEEP_AXIS_DOT or self.any_wet
        )
        if keep:
            # The grid stays on the bone. Water already down stays where it is
            # and the jet adds to the same sheet, so nearby films join.
            self._hit = hit
            # Search deep or high enough to reach this hit. Reaching deeper never
            # changes a column already read, since the first bone from above wins.
            gap_mm = float((hit - self._origin) @ self._normal) / MM_TO_SIM
            self._below_mm = max(self._below_mm, -gap_mm + LAND_MM)
            self._above_mm = max(self._above_mm, gap_mm + LAND_MM)
            stale = self._clock - self._bed_time >= BED_REFRESH_S
            delta_mm = (hit - self._origin) / MM_TO_SIM
            di = int(np.round(float(np.dot(delta_mm, self._right)) / CELL_MM))
            dj = int(np.round(float(np.dot(delta_mm, self._across)) / CELL_MM))
            if abs(di) > REANCHOR_CELLS or abs(dj) > REANCHOR_CELLS:
                # Slide by whole cells so every wet cell keeps its place on the bone.
                if abs(di) >= self.n or abs(dj) >= self.n:
                    self._clear_water()
                else:
                    self._slide(di, dj)
                step = (di * self._right + dj * self._across) * (CELL_MM * MM_TO_SIM)
                self._origin = self._origin + step
                self._occ_gen = occ_gen
                self._sample_bed(frame, occupied)
            elif stale and (occ_gen != self._occ_gen or not self._lands(hit)):
                # New cavity, or the jet reached bone the last read missed.
                self._occ_gen = occ_gen
                self._sample_bed(frame, occupied)
            if self._lands(hit, frame, occupied) or self.any_wet:
                return
        right, across = _basis(normal)
        self._clear_water()
        self._above_mm = BED_ABOVE_MM
        self._below_mm = BED_BELOW_MM
        self._origin = hit.copy()
        self._hit = hit
        self._normal = normal
        self._right = right
        self._across = across
        self._occ_gen = occ_gen
        self._sample_bed(frame, occupied)

    def _clear_water(self) -> None:
        self.depth[:] = 0.0
        self.reach[:] = False

    def _slide(self, di: int, dj: int) -> None:
        self.depth = _shift(self.depth, di, dj, 0.0)
        self.reach = _shift(self.reach, di, dj, False)

    def _lands(self, hit, frame=None, occupied=None) -> bool:
        """True when the bed under the hit is the bone the jet is touching.

        With the volume given, the bed is the one the latest read just wrote.
        """
        rel_mm = (np.asarray(hit, dtype=np.float64) - self._origin) / MM_TO_SIM
        center = self.n // 2
        i = int(np.rint(float(rel_mm @ self._right) / CELL_MM + center))
        j = int(np.rint(float(rel_mm @ self._across) / CELL_MM + center))
        if not (0 <= i < self.n and 0 <= j < self.n) or not self.on_bone[i, j]:
            return False
        gap_mm = float((hit - self.bed[i, j]) @ self._normal) / MM_TO_SIM
        return abs(gap_mm) <= LAND_MM

    def _sample_bed(self, frame, occupied) -> None:
        n = self.n
        center = n // 2
        coords = (np.arange(n) - center) * CELL_MM
        x_mm, y_mm = np.meshgrid(coords, coords, indexing="ij")
        seeds = (
            self._origin
            + x_mm[..., None] * self._right * MM_TO_SIM
            + y_mm[..., None] * self._across * MM_TO_SIM
        )
        start = seeds + self._normal * (self._above_mm * MM_TO_SIM)
        span = self._above_mm + self._below_mm
        # The volume pose is affine, so march in index space and only gather
        # columns that have not reached bone yet.
        origin_idx = frame.world_to_index(start.reshape(-1, 3))
        step_idx = frame.world_to_index(-self._normal * (BED_STEP_MM * MM_TO_SIM)) - frame.world_to_index(
            np.zeros(3)
        )
        counts = frame.counts.astype(np.int64)
        open_cols = np.arange(n * n)
        first = np.zeros(n * n, dtype=np.int64)
        found = np.zeros(n * n, dtype=bool)
        for k in range(int(round(span / BED_STEP_MM)) + 1):
            ijk = np.floor(origin_idx[open_cols] + k * step_idx).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < counts), axis=1)
            hit = np.zeros(len(open_cols), dtype=bool)
            sel = ijk[inside]
            hit[inside] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
            if np.any(hit):
                cols = open_cols[hit]
                found[cols] = True
                first[cols] = k
                open_cols = open_cols[~hit]
                if len(open_cols) == 0:
                    break
        # Half a step back catches the surface the coarse step jumped past.
        back = np.flatnonzero(found & (first > 0))
        if len(back):
            ijk = np.floor(origin_idx[back] + (first[back] - 0.5)[:, None] * step_idx).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < counts), axis=1)
            solid = np.zeros(len(back), dtype=bool)
            sel = ijk[inside]
            solid[inside] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
            fine = first.astype(np.float64)
            fine[back[solid]] -= 0.5
        else:
            fine = first.astype(np.float64)
        found = found.reshape(n, n)
        step_mm = (fine * BED_STEP_MM).reshape(n, n)
        bed = start - step_mm[..., None] * self._normal * MM_TO_SIM
        self._rough = np.where(found, _value_noise(bed / MM_TO_SIM), 0.0)
        bed[~found] = np.nan
        self.bed = bed
        self.on_bone = found
        self.depth[~found] = 0.0
        self._bed_samples += 1
        self._bed_time = self._clock

    def _offsets(self, hit):
        n = self.n
        center = n // 2
        coords = (np.arange(n) - center) * CELL_MM
        x_mm, y_mm = np.meshgrid(coords, coords, indexing="ij")
        hx = float(np.dot((hit - self._origin) / MM_TO_SIM, self._right))
        hy = float(np.dot((hit - self._origin) / MM_TO_SIM, self._across))
        return np.hypot(x_mm - hx, y_mm - hy)

    def _inlet(
        self, hit, jet_mm: float, spread_mm: float, depth_mm: float, cc_per_s: float, dt: float
    ) -> None:
        if dt <= 0.0:
            return
        radius = self._offsets(hit)
        stamp = max(0.5 * jet_mm, IMPACT_MM)
        # Volume in, so the sheet grows at the same rate whatever the jet width.
        volume = FLOW_MM3_PER_UNIT * max(0.0, cc_per_s) * dt / (CELL_MM * CELL_MM)
        # A full impact spot splashes outward instead of losing water to the depth cap.
        # Past a few rings the pool is full and the rest runs off.
        for _ in range(SPLASH_RINGS):
            if volume <= 1e-9 or stamp > spread_mm:
                break
            inlet = self.on_bone & (radius <= stamp)
            room = np.where(inlet, np.clip(depth_mm - self.depth, 0.0, None), 0.0)
            space = float(room.sum())
            if space > 0.0:
                share = min(1.0, volume / space)
                self.depth += room * share
                volume -= space * share
            stamp += CELL_MM

    def _suction(self, tip, radius_mm: float, dt: float = 0.0) -> None:
        if tip is None or radius_mm <= 0.0 or self._origin is None:
            return
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        delta = np.nan_to_num(self.bed - tip, nan=1.0)
        # Distance from the tip to the water, not to the bone under it.
        gap = np.maximum(np.linalg.norm(delta, axis=2) - self.depth * MM_TO_SIM, 0.0)
        capture = float(radius_mm) * MM_TO_SIM
        reach = capture * SUCTION_REACH
        self.depth[self.on_bone & (gap <= capture)] = 0.0
        if dt > 0.0:
            near = self.on_bone & (gap > capture) & (gap < reach)
            pull = 1.0 - (gap[near] - capture) / (reach - capture)
            self.depth[near] *= np.exp(-SUCTION_DRAW_PER_S * float(dt) * pull)

    def _flow(self, dt: float, spread_mm: float, depth_mm: float, life_s: float | None = None) -> None:
        if dt <= 0.0:
            return
        pace = 1.0
        lip_frac = 0.5
        if life_s is not None:
            # Longer life loses less over the lip, so the sheet fills toward its depth cap.
            lip_frac = min(0.5, 0.5 * REF_LIFE_S / max(float(life_s), 0.05))
        home = self.on_bone & self.reach
        lift = max(0.25, -float(np.dot(self._normal, self._down)))
        slope = self._down - float(np.dot(self._down, self._normal)) * self._normal
        slope_len = float(np.linalg.norm(slope))
        slope = slope / slope_len if slope_len > 1e-3 else np.zeros(3)
        # Height is against the tuned downhill, so a slider can turn the runoff.
        bed_mm = np.where(
            self.on_bone,
            -np.sum(self.bed * self._down, axis=2) / MM_TO_SIM
            + self._rough * (RIVULET_BUMP_MM * self._rivulets),
            0.0,
        )
        pin = RIVULET_PIN_MM * self._rivulets
        passes = 2
        step = dt / passes
        budget = SLIDE_MM_S * pace * step
        for _ in range(passes):
            # Millimetres. Mixing sim-metres into this level made the slope invisible.
            level = bed_mm + self.depth * lift
            wet = self.depth >= MIN_DEPTH_MM
            moves = []
            for di, dj in _DIRS:
                neighbour = _shift(home, di, dj, False)
                neighbour_level = _shift(level, di, dj, 0.0)
                lower = neighbour & (neighbour_level + 1e-4 < level)
                if pin > 0.0:
                    # A thin front stays put on dry bone, so the flow breaks into streams.
                    lower &= _shift(wet, di, dj, False) | (self.depth > pin)
                lip = home & ~neighbour
                gap = np.clip(level - neighbour_level, 0.0, None)
                # Half the gap so one pass cannot invert the cells and stripe the sheet.
                share = 0.25 * pace
                slide = np.minimum(self.depth * share, np.minimum(budget, 0.5 * gap))
                move = np.where(lower, slide, 0.0)
                # Water runs off a downhill edge and only seeps over an uphill one,
                # so the sheet does not pool against the edge of its patch.
                downhill = max(0.0, float(np.dot(di * self._right + dj * self._across, slope)))
                lip_share = max(lip_frac, LIP_RUN * downhill)
                dump = np.where(lip, np.minimum(self.depth * lip_share, budget), 0.0)
                move = np.where(lip, dump, move)
                move = np.where(home & (self.depth > 0.0), move, 0.0)
                moves.append((di, dj, move, lower))
            total = np.zeros_like(self.depth)
            for _, _, move, _ in moves:
                total += move
            scale = np.minimum(1.0, self.depth / np.maximum(total, 1e-8))
            for di, dj, move, lower in moves:
                moved = move * scale
                self.depth -= moved
                self.depth += _shift(np.where(lower, moved, 0.0), -di, -dj, 0.0)
            np.clip(self.depth, 0.0, depth_mm, out=self.depth)
            self.depth[~home] = 0.0
        # Share with orthogonal neighbours so the sheet does not stripe.
        blurred = self.depth.copy()
        for di, dj in _DIRS:
            blurred = blurred + _shift(self.depth, di, dj, 0.0)
        blurred *= 0.2
        mix = 0.35 * pace
        if self._rivulets > 0.0:
            # The blur would otherwise creep onto dry bone and undo the pinning.
            dry = self.depth < MIN_DEPTH_MM
            mix = np.where(dry, mix * (1.0 - self._rivulets), mix)
        self.depth = np.where(home, (1.0 - mix) * self.depth + mix * blurred, 0.0)
        np.clip(self.depth, 0.0, depth_mm, out=self.depth)
        self.depth[~home] = 0.0

    def _cap(self, spread_mm: float, depth_mm: float) -> None:
        np.clip(self.depth, 0.0, depth_mm, out=self.depth)
        limit = math.pi * spread_mm * spread_mm * depth_mm
        volume = float(self.depth.sum() * CELL_MM * CELL_MM)
        if volume > limit > 0.0:
            self.depth *= limit / volume

    def _display_depth(self) -> np.ndarray:
        height = self.depth.copy()
        for _ in range(self._height_passes):
            total = height.copy()
            count = np.ones_like(height)
            wet = height >= MIN_DEPTH_MM
            near = wet.copy()
            for di, dj in _DIRS:
                total += _shift(height, di, dj, 0.0)
                count += 1.0
                near |= _shift(wet, di, dj, False)
            height = np.where(self.on_bone & near, total / count, 0.0)
        return height

    def _smooth_positions(self, pos, mask):
        smoothed = pos
        keep = self._pos_keep
        for _ in range(self._pos_passes):
            total = np.where(mask[..., None], smoothed, 0.0)
            count = mask[..., None].astype(np.float64)
            for di, dj in _DIRS:
                neighbour = _shift(mask, di, dj, False)
                total += np.where(neighbour[..., None], _shift(smoothed, di, dj, 0.0), 0.0)
                count += neighbour[..., None]
            blended = total / np.maximum(count, 1.0)
            smoothed = np.where(mask[..., None], keep * smoothed + (1.0 - keep) * blended, smoothed)
        return smoothed

    def _fill_tris(self, mask) -> int:
        n = mask.shape[1]
        quad = (
            mask[:-1, :-1]
            & mask[1:, :-1]
            & mask[:-1, 1:]
            & mask[1:, 1:]
        )
        self._quad = quad
        ii, jj = np.nonzero(quad)
        if len(ii) == 0:
            return 0
        a = ii * n + jj
        b = (ii + 1) * n + jj
        c = ii * n + (jj + 1)
        d = (ii + 1) * n + (jj + 1)
        # cross(right, across) == outward normal, so this winding faces out of the bone.
        count = len(ii) * 2
        tris = self.tris
        tris[: len(ii), 0] = a
        tris[: len(ii), 1] = b
        tris[: len(ii), 2] = c
        tris[len(ii) : count, 0] = b
        tris[len(ii) : count, 1] = d
        tris[len(ii) : count, 2] = c
        return count

    def _shade(self, pos, mask, count: int):
        normals = np.zeros_like(pos)
        ii = self.tris[:count, 0]
        jj = self.tris[:count, 1]
        kk = self.tris[:count, 2]
        p0 = pos.reshape(-1, 3)[ii]
        p1 = pos.reshape(-1, 3)[jj]
        p2 = pos.reshape(-1, 3)[kk]
        face = np.cross(p1 - p0, p2 - p0)
        for corner, index in ((face, ii), (face, jj), (face, kk)):
            np.add.at(normals.reshape(-1, 3), index, corner)
        length = np.linalg.norm(normals, axis=2, keepdims=True)
        bad = (length[..., 0] < 1e-10) | ~mask
        normals = np.where(bad[..., None], self._normal, normals / np.maximum(length, 1e-10))
        if float(np.dot(normals[mask].mean(axis=0), self._normal)) < 0.0:
            normals = -normals
        return normals
