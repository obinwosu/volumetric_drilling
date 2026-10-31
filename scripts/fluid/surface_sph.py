"""A thin sheet of particles on the bone the drill is cutting.

Particles stick to the occupancy surface. On intact bone they stay about half
a millimetre thick and slide toward the bottom of the microscope image. In a
cavity they pile up only until the downhill lip, then cross that lip and run
off. A particle that leaves the bone is deleted and its slot is reused. The
list never grows, and a live particle is never overwritten.

World positions are simulator metres.
"""

from __future__ import annotations

import numpy as np

from fluid.occupancy import MM_TO_SIM

PARTICLE_MM = 0.5
MAX_PARTICLES = 2048
# Center of a particle on open bone. The drawn sheet is about this thick.
SKIN_MM = 0.35
# Open bone cannot hold a deeper coat than this.
SHEET_MM = 0.55
# A neighbor this much higher, along the surface normal, is a wall.
BLOCK_MM = 0.6
SLIDE_MM_S = 30.0
SPAWN_PER_S = 70.0
SUBSTEPS = 4
PROBE_MM = 0.75
MARCH_MM = 8.0
MARCH_STEP_MM = 0.25
# Drop a particle that has been off the bone this many attaches in a row.
DETACH_LIMIT = 1


def _unit(vector):
    axis = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    return axis / length


def _basis(axis):
    normal = _unit(axis)
    helper = np.array((1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
    right = np.cross(helper, normal)
    right = right / float(np.linalg.norm(right))
    across = np.cross(normal, right)
    return right, across, normal


def _indices(frame, points):
    ijk = np.floor(frame.world_to_index(points)).astype(np.int32)
    counts = np.asarray(frame.counts, dtype=np.int32)
    valid = np.all((ijk >= 0) & (ijk < counts), axis=1)
    return ijk, valid


def _occupied_at(occupied, ijk, valid):
    out = np.zeros(len(ijk), dtype=bool)
    if np.any(valid):
        sel = ijk[valid]
        out[valid] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def _outward(occupied, ijk, found, fallback):
    """Normal pointing out of the bone. ``fallback`` is toward the scope."""
    n = len(ijk)
    normal = np.repeat(np.asarray(fallback, dtype=np.float64).reshape(1, 3), n, axis=0)
    if not np.any(found):
        return normal
    shape = np.asarray(occupied.shape, dtype=np.int32)
    grad = np.zeros((n, 3), dtype=np.float64)
    base = ijk.copy()
    for axis in range(3):
        plus = base.copy()
        minus = base.copy()
        plus[:, axis] += 1
        minus[:, axis] -= 1
        plus_ok = found & np.all((plus >= 0) & (plus < shape), axis=1)
        minus_ok = found & np.all((minus >= 0) & (minus < shape), axis=1)
        plus_occ = np.zeros(n, dtype=np.float64)
        minus_occ = np.zeros(n, dtype=np.float64)
        if np.any(plus_ok):
            sel = plus[plus_ok]
            plus_occ[plus_ok] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
        if np.any(minus_ok):
            sel = minus[minus_ok]
            minus_occ[minus_ok] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
        # Out of range counts as air, so the specimen edge points outward.
        grad[:, axis] = plus_occ - minus_occ
    outward = -grad
    length = np.linalg.norm(outward, axis=1, keepdims=True)
    good = found & (length[:, 0] > 0.5)
    normal[good] = outward[good] / length[good]
    return normal


def attach(points, view, occupied, frame, march_mm: float = MARCH_MM):
    """Bone surface just outside the first occupied voxel along the view.

    ``view`` points into the scene. Air marches into the bone. A point already
    inside marches back toward the scope. Missing bone leaves ``found`` false.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    view = _unit(view)
    n = len(points)
    surface = np.full((n, 3), np.nan, dtype=np.float64)
    found = np.zeros(n, dtype=bool)
    bone_ijk = np.zeros((n, 3), dtype=np.int32)
    if n == 0 or occupied is None or frame is None:
        return surface, np.repeat(-view.reshape(1, 3), n, axis=0), found
    step = MARCH_STEP_MM * MM_TO_SIM
    steps = max(1, int(march_mm / MARCH_STEP_MM))
    ijk, valid = _indices(frame, points)
    inside = _occupied_at(occupied, ijk, valid)

    pos = points.copy()
    prev = points.copy()
    active = valid & ~inside
    for _ in range(steps):
        if not np.any(active):
            break
        prev[active] = pos[active]
        pos[active] = pos[active] + view * step
        ijk, ok = _indices(frame, pos)
        occ = _occupied_at(occupied, ijk, ok)
        hit = active & occ
        surface[hit] = prev[hit]
        bone_ijk[hit] = ijk[hit]
        found[hit] = True
        active = active & ~hit & ok

    pos = points.copy()
    active = valid & inside & ~found
    for _ in range(steps):
        if not np.any(active):
            break
        pos[active] = pos[active] - view * step
        ijk, ok = _indices(frame, pos)
        occ = _occupied_at(occupied, ijk, ok)
        left = active & (~occ | ~ok)
        surface[left] = pos[left]
        back = pos[left] + view * step
        back_ijk, _back_ok = _indices(frame, back)
        bone_ijk[left] = back_ijk
        found[left] = True
        active = active & ~left & ok
    normal = _outward(occupied, bone_ijk, found, -view)
    return surface, normal, found


def _layers(gap, tangent, side, downhill, lip, held, skin):
    """Spread across the floor, then stack up to the lip. The rest spills."""
    over = np.zeros(len(gap), dtype=bool)
    tangent = np.array(tangent, copy=True)
    if not np.any(held):
        return gap, over, tangent
    cell = PARTICLE_MM * MM_TO_SIM
    step = cell
    tx = np.einsum("ij,ij->i", tangent, side)
    ty = np.einsum("ij,ij->i", tangent, downhill)
    kx = np.floor(tx / cell).astype(np.int32)
    ky = np.floor(ty / cell).astype(np.int32)
    bins = {}
    for i in np.flatnonzero(held):
        bins.setdefault((int(kx[i]), int(ky[i])), []).append(int(i))
    gap = np.array(gap, copy=True)
    slack = 0.05 * MM_TO_SIM
    for key, group in bins.items():
        group.sort(key=lambda i: float(gap[i]))
        empty = []
        for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (key[0] + ox, key[1] + oy) not in bins:
                empty.append((ox, oy))
        if empty and len(group) > 1:
            ox, oy = empty[0]
            for i in group[1:]:
                tangent[i] = tangent[i] + (side[i] * ox + downhill[i] * oy) * cell
                gap[i] = skin
            gap[group[0]] = skin
            continue
        for rank, i in enumerate(group):
            target = skin + rank * step
            if target > lip[i] - slack:
                over[i] = True
            else:
                gap[i] = target
    return gap, over, tangent


class SurfaceSph:
    """Fixed particle list constrained to the current occupancy."""

    def __init__(self, max_particles: int = MAX_PARTICLES):
        self.max_particles = int(max_particles)
        self.pos = np.zeros((self.max_particles, 3), dtype=np.float64)
        self.vel = np.zeros((self.max_particles, 3), dtype=np.float64)
        self.alive = np.zeros(self.max_particles, dtype=bool)
        self.spill = 0
        self.suction_count = 0
        self.spawned = 0
        self.time = 0.0
        self._owed = 0.0
        self._rng = np.random.default_rng(1)

    @property
    def count(self) -> int:
        return int(self.alive.sum())

    @property
    def any_wet(self) -> bool:
        return self.count > 0

    def positions(self) -> np.ndarray:
        return self.pos[self.alive].copy()

    def covers(self, points) -> np.ndarray:
        """True near a live particle. Used so dust does not sit in the sheet."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        keep = np.zeros(len(points), dtype=bool)
        live = self.positions()
        if len(live) == 0 or len(points) == 0:
            return keep
        radius = 0.8 * MM_TO_SIM
        # Chunk the bone samples so a full particle list stays cheap.
        for start in range(0, len(points), 64):
            batch = points[start : start + 64]
            delta = batch[:, None, :] - live[None, :, :]
            dist = np.linalg.norm(delta, axis=2)
            keep[start : start + 64] = dist.min(axis=1) <= radius
        return keep

    def _free_slots(self, n: int) -> np.ndarray:
        free = np.flatnonzero(~self.alive)
        if len(free) == 0 or n <= 0:
            return free[:0]
        return free[:n]

    def _kill(self, mask, spilled: bool) -> None:
        mask = mask & self.alive
        if not np.any(mask):
            return
        if spilled:
            self.spill += int(mask.sum())
        self.alive[mask] = False
        self.vel[mask] = 0.0

    def _spawn(self, hit, jet_mm, view, occupied, frame, dt: float) -> None:
        if hit is None or dt <= 0.0:
            return
        self._owed += SPAWN_PER_S * dt
        n = int(self._owed)
        if n <= 0:
            return
        slots = self._free_slots(n)
        if len(slots) == 0:
            return
        view = _unit(view)
        right, across, _normal = _basis(view)
        radius = max(0.5 * float(jet_mm), 0.5 * PARTICLE_MM) * MM_TO_SIM
        disc = self._rng.random(len(slots)) ** 0.5 * radius
        theta = self._rng.random(len(slots)) * (2.0 * np.pi)
        offset = right * (np.cos(theta) * disc)[:, None] + across * (np.sin(theta) * disc)[:, None]
        points = np.asarray(hit, dtype=np.float64).reshape(1, 3) + offset
        surface, normal, found = attach(points, view, occupied, frame)
        if not np.any(found):
            return
        use = slots[found]
        skin = SKIN_MM * MM_TO_SIM
        self.pos[use] = surface[found] + normal[found] * skin
        self.vel[use] = 0.0
        self.alive[use] = True
        self.spawned += len(use)
        self._owed -= len(use)

    def _separate(self, index) -> None:
        """Push overlapping neighbors apart. This is the pressure."""
        if len(index) < 2:
            return
        rest = PARTICLE_MM * MM_TO_SIM
        points = self.pos[index]
        cell = rest
        keys = np.floor(points / cell).astype(np.int32)
        bins = {}
        for local, key in enumerate(keys):
            bins.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(local)
        shift = np.zeros_like(points)
        for local, key in enumerate(keys):
            kx, ky, kz = int(key[0]), int(key[1]), int(key[2])
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for oz in (-1, 0, 1):
                        for other in bins.get((kx + ox, ky + oy, kz + oz), ()):
                            if other <= local:
                                continue
                            delta = points[local] - points[other]
                            dist = float(np.linalg.norm(delta))
                            if dist < 1e-8 or dist >= rest:
                                continue
                            direction = delta / dist
                            push = 0.5 * (rest - dist) * direction
                            shift[local] += push
                            shift[other] -= push
        self.pos[index] += shift

    def _substep(self, dt, view, gravity, occupied, frame) -> None:
        index = np.flatnonzero(self.alive)
        if len(index) == 0:
            return
        view = _unit(view)
        gravity = _unit(gravity)
        up_scale = MM_TO_SIM
        skin = SKIN_MM * up_scale
        sheet = SHEET_MM * up_scale
        block = BLOCK_MM * up_scale
        probe = PROBE_MM * up_scale
        slide = SLIDE_MM_S * up_scale * dt
        points = self.pos[index]
        surface, normal, found = attach(points, view, occupied, frame)
        lost = np.zeros(self.max_particles, dtype=bool)
        lost[index[~found]] = True
        self._kill(lost, spilled=True)
        index = index[found]
        if len(index) == 0:
            return
        surface = surface[found]
        normal = normal[found]
        points = self.pos[index]
        gap = np.einsum("ij,ij->i", points - surface, normal)
        tangent_g = gravity - normal * np.einsum("ij,j->i", normal, gravity)[:, None]
        tangent_len = np.linalg.norm(tangent_g, axis=1, keepdims=True)
        tangent_g = tangent_g / np.maximum(tangent_len, 1e-8)
        side = np.cross(normal, tangent_g)
        side_len = np.linalg.norm(side, axis=1, keepdims=True)
        side = side / np.maximum(side_len, 1e-8)
        ahead = points + tangent_g * probe
        ahead_s, ahead_n, ahead_found = attach(ahead, view, occupied, frame)
        lip = np.einsum("ij,ij->i", ahead_s - surface, normal)
        blocked = ahead_found & (lip > block)
        # A particle already as high as the downhill lip crosses onto that lip.
        over = blocked & (gap >= np.maximum(lip - 0.05 * up_scale, skin))
        if np.any(over):
            self.pos[index[over]] = ahead_s[over] + ahead_n[over] * skin
            self.vel[index[over]] = tangent_g[over] * (SLIDE_MM_S * up_scale)
        stay = ~over
        if not np.any(stay):
            return
        stay_index = index[stay]
        self._separate(stay_index)
        moved = self.pos[stay_index]
        surface_s = surface[stay]
        normal_s = normal[stay]
        side_s = side[stay]
        gap_s = np.einsum("ij,ij->i", moved - surface_s, normal_s)
        tangent = (moved - surface_s) - normal_s * gap_s[:, None]
        blocked_s = blocked[stay]
        open_s = ~blocked_s
        tangent[open_s] += tangent_g[stay][open_s] * slide
        gap_s = np.where(open_s, np.minimum(np.maximum(gap_s, skin), sheet), gap_s)
        lip_s = lip[stay]
        held = blocked_s & (gap_s < lip_s + sheet)
        gap_s, piled_over, tangent = _layers(gap_s, tangent, side_s, tangent_g[stay], lip_s, held, skin)
        held = held & ~piled_over
        # Held water does not climb the downhill wall.
        climb = np.einsum("ij,ij->i", tangent, tangent_g[stay])
        tangent[held] -= tangent_g[stay][held] * np.maximum(climb[held], 0.0)[:, None]
        # Side walls of the cavity keep the pool from walking out the sides.
        if np.any(held):
            for sign in (1.0, -1.0):
                direction = side_s * sign
                probe_s, _probe_n, probe_found = attach(
                    self.pos[stay_index] + direction * probe, view, occupied, frame
                )
                side_lip = np.einsum("ij,ij->i", probe_s - surface_s, normal_s)
                wall = probe_found & (side_lip > block) & held
                component = np.einsum("ij,ij->i", tangent, direction)
                toward = wall & (component > 0.0)
                tangent[toward] -= direction[toward] * component[toward, None]
        self.pos[stay_index] = surface_s + normal_s * gap_s[:, None] + tangent
        self.vel[stay_index] = 0.0
        if np.any(piled_over):
            over_i = stay_index[piled_over]
            self.pos[over_i] = ahead_s[stay][piled_over] + ahead_n[stay][piled_over] * skin
            self.vel[over_i] = tangent_g[stay][piled_over] * (SLIDE_MM_S * up_scale)

    def suction(self, tip, radius_mm: float) -> None:
        if tip is None or radius_mm <= 0.0 or self.count == 0:
            return
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        live = self.alive.copy()
        close = live & (np.linalg.norm(self.pos - tip, axis=1) <= float(radius_mm) * MM_TO_SIM)
        self.suction_count += int(close.sum())
        self.alive[close] = False
        self.vel[close] = 0.0

    def step(
        self,
        dt: float,
        hit,
        jet_mm: float,
        view,
        gravity,
        frame=None,
        occupied=None,
        suction_tip=None,
        suction_radius_mm: float = 0.0,
    ) -> None:
        dt = float(max(dt, 0.0))
        if dt <= 0.0 or frame is None or occupied is None:
            return
        self.time += dt
        self._spawn(hit, jet_mm, view, occupied, frame, dt)
        sub = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self._substep(sub, view, gravity, occupied, frame)
        self.suction(suction_tip, suction_radius_mm)

    def gaps_mm(self, view, occupied, frame) -> np.ndarray:
        """Distance from each live particle to the bone, along the surface normal."""
        live = self.positions()
        if len(live) == 0:
            return np.zeros(0, dtype=np.float64)
        surface, normal, found = attach(live, view, occupied, frame)
        gap = np.einsum("ij,ij->i", live - surface, normal) / MM_TO_SIM
        gap[~found] = np.inf
        return gap


# Drawn sheet. Finer than the particles so the facets disappear, and only one
# surface, so the path tracer is not looking at a field of beads.
FILM_CELL_MM = 0.22
# A 0.5 mm particle spacing still bridges this kernel. One splat does not.
FILM_DENSITY = 0.1
FILM_SMOOTH = 5


def _shift2(field, di, dj, fill):
    """Value at (i+di, j+dj). The grid edge is ``fill``, not a wrap."""
    src = np.asarray(field, dtype=np.float64)
    out = np.full(src.shape, fill, dtype=np.float64)
    nx, ny = src.shape
    i0, i1 = max(0, -di), min(nx, nx - di)
    j0, j1 = max(0, -dj), min(ny, ny - dj)
    if i0 >= i1 or j0 >= j1:
        return out
    out[i0:i1, j0:j1] = src[i0 + di : i1 + di, j0 + dj : j1 + dj]
    return out


def _blur(field):
    """Wide box so neighbors half a millimetre apart join, and one splat does not."""
    acc = np.zeros(field.shape, dtype=np.float64)
    count = 0
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if di * di + dj * dj > 5:
                continue
            acc += _shift2(field, di, dj, 0.0)
            count += 1
    return acc / float(count)


def film_surface(points, view, cell_mm: float = FILM_CELL_MM):
    """One smooth sheet through the free surface of the particles.

    A lone particle, and any speck that does not touch its neighbors, draws
    nothing. The mesh is a single surface facing the scope, not a shell around
    each particle.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return None
    _right, _across, view = _basis(view)
    right, across, view = _basis(view)
    origin = points.mean(axis=0)
    rel = points - origin
    x = rel @ right / MM_TO_SIM
    y = rel @ across / MM_TO_SIM
    # Larger elevation is closer to the scope. That is the free surface.
    elev = -(rel @ view) / MM_TO_SIM
    cell = float(cell_mm)
    pad = 1.2
    minx, maxx = float(x.min()) - pad, float(x.max()) + pad
    miny, maxy = float(y.min()) - pad, float(y.max()) + pad
    nx = int(np.ceil((maxx - minx) / cell)) + 1
    ny = int(np.ceil((maxy - miny) / cell)) + 1
    while nx * ny > 180 * 180 and cell < 0.8:
        cell *= 1.35
        nx = int(np.ceil((maxx - minx) / cell)) + 1
        ny = int(np.ceil((maxy - miny) / cell)) + 1
    nx = max(nx, 2)
    ny = max(ny, 2)
    ix = np.clip(np.floor((x - minx) / cell).astype(np.int32), 0, nx - 1)
    iy = np.clip(np.floor((y - miny) / cell).astype(np.int32), 0, ny - 1)
    density = np.zeros((nx, ny), dtype=np.float64)
    height = np.full((nx, ny), -np.inf, dtype=np.float64)
    np.add.at(density, (ix, iy), 1.0)
    # Free surface: the particle nearest the scope in each cell.
    np.maximum.at(height, (ix, iy), elev)
    density = _blur(density)
    # Spread the free-surface height into the cells the blur filled.
    finite = np.isfinite(height) & (height > -1e8)
    height = np.where(finite, height, np.nan)
    for _ in range(FILM_SMOOTH):
        acc = np.zeros_like(height)
        weight = np.zeros_like(height)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            rolled = _shift2(height, di, dj, np.nan)
            ok = np.isfinite(rolled)
            acc = np.where(ok, acc + rolled, acc)
            weight = np.where(ok, weight + 1.0, weight)
        have = np.isfinite(height) & (weight > 0.0)
        mixed = 0.4 * height + 0.6 * (acc / np.maximum(weight, 1.0))
        height = np.where(have, mixed, height)
        # The blur joins particles. Carry the free surface into those cells.
        empty = ~np.isfinite(height) & (weight > 0.0) & (density >= FILM_DENSITY)
        height = np.where(empty, acc / np.maximum(weight, 1.0), height)
    show = (density >= FILM_DENSITY) & np.isfinite(height)
    if int(show.sum()) < 2:
        return None
    ii, jj = np.nonzero(show)
    corner = -np.ones((nx + 1, ny + 1), dtype=np.int32)
    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        corner[ii + di, jj + dj] = 0
    ci, cj = np.nonzero(corner == 0)
    corner[ci, cj] = np.arange(len(ci), dtype=np.int32)
    elev_c = np.zeros(len(ci), dtype=np.float64)
    weight = np.zeros(len(ci), dtype=np.float64)
    for di, dj in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
        i = ci + di
        j = cj + dj
        valid = (i >= 0) & (j >= 0) & (i < nx) & (j < ny)
        sample = np.full(len(ci), np.nan)
        if np.any(valid):
            sample[valid] = height[i[valid], j[valid]]
        ok = valid & np.isfinite(sample)
        elev_c[ok] += sample[ok]
        weight[ok] += 1.0
    if not np.any(weight > 0.0):
        return None
    elev_c /= np.maximum(weight, 1.0)
    x_c = minx + ci.astype(np.float64) * cell
    y_c = miny + cj.astype(np.float64) * cell
    verts = (
        origin
        + right * (x_c * MM_TO_SIM)[:, None]
        + across * (y_c * MM_TO_SIM)[:, None]
        - view * (elev_c * MM_TO_SIM)[:, None]
    )
    step = cell
    gx = np.zeros(len(ci), dtype=np.float64)
    gy = np.zeros(len(ci), dtype=np.float64)
    right_of = corner[np.clip(ci + 1, 0, nx), cj]
    left_of = corner[np.clip(ci - 1, 0, nx), cj]
    up_of = corner[ci, np.clip(cj + 1, 0, ny)]
    down_of = corner[ci, np.clip(cj - 1, 0, ny)]
    has_x = (ci + 1 <= nx) & (ci - 1 >= 0) & (right_of >= 0) & (left_of >= 0)
    has_y = (cj + 1 <= ny) & (cj - 1 >= 0) & (up_of >= 0) & (down_of >= 0)
    gx[has_x] = (elev_c[right_of[has_x]] - elev_c[left_of[has_x]]) / (2.0 * step)
    gy[has_y] = (elev_c[up_of[has_y]] - elev_c[down_of[has_y]]) / (2.0 * step)
    tx = right - view * gx[:, None]
    ty = across - view * gy[:, None]
    normal = np.cross(tx, ty)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(length, 1e-8)
    flip = (normal @ view) > 0.0
    normal[flip] *= -1.0
    tris = []
    for a, b, c in (
        (corner[ii, jj], corner[ii, jj + 1], corner[ii + 1, jj + 1]),
        (corner[ii, jj], corner[ii + 1, jj + 1], corner[ii + 1, jj]),
    ):
        tris.append(np.stack((a, b, c), axis=1))
    tris = np.concatenate(tris, axis=0).astype(np.int32)
    return {
        "verts": np.ascontiguousarray(verts, dtype=np.float32),
        "tris": tris,
        "normals": np.ascontiguousarray(normal, dtype=np.float32),
    }
