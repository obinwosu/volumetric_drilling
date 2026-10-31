"""A thin water depth on the bone.

The jet adds water in a patch as wide as the stream. The grid stays on the
camera-facing bone. The waterline falls into a hole and toward the bottom of
the microscope image, fills a cavity up to the downhill lip, then runs off the
bone. A thickness cap only stops a closed hole from growing into a mass.
Suction deletes depth. Bone the water has reached stays wet. The drawn sheet
is a smoothed finer mesh. The sim grid stays coarse.

World positions are simulator metres. Depth and the grid are millimetres.
"""

from __future__ import annotations

import math

import numpy as np

from fluid.occupancy import MM_TO_SIM

CELL_MM = 0.5
SPAN_MM = 40.0
# Safety for a pit with no lip. A real lip spills before this.
MAX_DEPTH_MM = 6.0
MIN_DEPTH_MM = 0.04
# The march lands up to one cell inside the first voxel. This lifts the
# drawn sheet back to the near side so it coats the bone instead of sitting in it.
SURFACE_BIAS_MM = 0.45
# Drop per millimetre toward the bottom of the microscope image. Holes still
# pool, because their floor is deeper along the view. Shallow dents run off.
GRAVITY_SLOPE = 0.45
# Sim stays on the coarse grid. The drawn sheet is finer and smoothed.
VISUAL_SUBDIV = 2
SMOOTH_PASSES = 5
SMOOTH_BLEND = 0.6
# Presence is full. Translucency comes from the water shader, not from a dark scrim.
SHEET_OPACITY = 1.0
# Visible volume at the tuner's 80 cc/s. The clinical flow would fill a cavity in one frame.
VISUAL_MM3_PER_S = 18.0
REFERENCE_CC_PER_S = 80.0
RELAX_PASSES = 8
SAMPLE_BUDGET = 480


def _basis(down):
    axis = np.asarray(down, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    else:
        axis = axis / length
    helper = np.array((1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(helper, axis)
    right = right / float(np.linalg.norm(right))
    across = np.cross(axis, right)
    return right, across, axis


def _shift(field, di, dj, fill):
    """Roll toward +di, +dj. The wrapped edge is replaced with ``fill``."""
    out = np.roll(np.array(field, copy=True), (int(di), int(dj)), axis=(0, 1))
    if di > 0:
        out[:di, :] = fill
    elif di < 0:
        out[di:, :] = fill
    if dj > 0:
        out[:, :dj] = fill
    elif dj < 0:
        out[:, dj:] = fill
    return out


def _at_neighbor(field, di, dj, fill):
    """Neighbor (i+di, j+dj) as seen from cell i. The grid edge has no neighbor."""
    return _shift(field, -int(di), -int(dj), fill)


def _smooth_nan(field, passes: int, blend: float):
    """Soften stairs. A straight slope is unchanged. Missing samples stay missing."""
    out = np.array(field, dtype=np.float64, copy=True)
    for _ in range(int(passes)):
        acc = np.zeros_like(out)
        weight = np.zeros_like(out)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = _at_neighbor(out, di, dj, np.nan)
            ok = np.isfinite(out) & np.isfinite(nb)
            acc = np.where(ok, acc + nb, acc)
            weight = np.where(ok, weight + 1.0, weight)
        mean = acc / np.maximum(weight, 1.0)
        mixed = (1.0 - blend) * out + blend * mean
        out = np.where(weight > 0.0, mixed, out)
    return out


def _bilinear(field, fi, fj):
    """Height at fractional cell indices. Missing corners drop out of the average."""
    n = field.shape[0]
    i0 = np.floor(fi).astype(np.int32)
    j0 = np.floor(fj).astype(np.int32)
    ti = fi - i0
    tj = fj - j0
    acc = np.zeros(len(fi), dtype=np.float64)
    weight = np.zeros(len(fi), dtype=np.float64)
    for di, dj, tw in (
        (0, 0, (1.0 - ti) * (1.0 - tj)),
        (1, 0, ti * (1.0 - tj)),
        (0, 1, (1.0 - ti) * tj),
        (1, 1, ti * tj),
    ):
        ii = i0 + di
        jj = j0 + dj
        inside = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < n)
        sampled = np.full(len(fi), np.nan)
        if np.any(inside):
            sampled[inside] = field[ii[inside], jj[inside]]
        ok = inside & np.isfinite(sampled) & (tw > 1e-8)
        acc = np.where(ok, acc + sampled * tw, acc)
        weight = np.where(ok, weight + tw, weight)
    return acc / np.maximum(weight, 1e-8), weight > 1e-8


class SurfaceFilm:
    """Fixed horizontal grid. Elevation increases against gravity."""

    def __init__(self, cell_mm: float = CELL_MM, span_mm: float = SPAN_MM):
        self.cell_mm = float(cell_mm)
        self.n = max(3, int(round(float(span_mm) / self.cell_mm)))
        self.bed = np.full((self.n, self.n), np.nan, dtype=np.float64)
        self.depth = np.zeros((self.n, self.n), dtype=np.float64)
        self.wet = np.zeros((self.n, self.n), dtype=bool)
        self.sampled = np.zeros((self.n, self.n), dtype=bool)
        self.origin = None
        self.right = None
        self.across = None
        self.down = None
        # Gravitational waterline. None keeps the old bed-only flow used by the cup tests.
        self.level = None
        self.time = 0.0
        self.spill_mm3 = 0.0
        self.suction_mm3 = 0.0
        self._e_hi = None
        self._e_lo = None
        self._sheet_visible = False
        self._wet_visible = False
        self._sheet_bound = False
        self._wet_bound = False
        self._since_sample = 0.0

    @property
    def any_wet(self) -> bool:
        return bool(self.wet.any())

    @property
    def volume_mm3(self) -> float:
        return float(self.depth.sum() * self.cell_mm * self.cell_mm)

    def place(self, origin, down) -> None:
        """Center the grid on the first hit. Later hits reuse it."""
        if self.origin is not None:
            return
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3).copy()
        self.right, self.across, self.down = _basis(down)

    def _cell(self, world):
        if self.origin is None:
            return None
        rel = np.asarray(world, dtype=np.float64).reshape(3) - self.origin
        x = float(np.dot(rel, self.right) / MM_TO_SIM)
        y = float(np.dot(rel, self.across) / MM_TO_SIM)
        i = int(np.rint(x / self.cell_mm + (self.n - 1) / 2.0))
        j = int(np.rint(y / self.cell_mm + (self.n - 1) / 2.0))
        if i < 0 or j < 0 or i >= self.n or j >= self.n:
            return None
        return i, j

    def _world(self, i, j, elevation_mm):
        x = (float(i) - (self.n - 1) / 2.0) * self.cell_mm
        y = (float(j) - (self.n - 1) / 2.0) * self.cell_mm
        return (
            self.origin
            + self.right * (x * MM_TO_SIM)
            + self.across * (y * MM_TO_SIM)
            - self.down * (float(elevation_mm) * MM_TO_SIM)
        )

    def covers(self, points) -> np.ndarray:
        """True on bone the sheet has already wet. The latch does not dry."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        keep = np.zeros(len(points), dtype=bool)
        if self.origin is None or not self.any_wet:
            return keep
        rel = points - self.origin
        x = rel @ self.right / MM_TO_SIM
        y = rel @ self.across / MM_TO_SIM
        i = np.rint(x / self.cell_mm + (self.n - 1) / 2.0).astype(np.int32)
        j = np.rint(y / self.cell_mm + (self.n - 1) / 2.0).astype(np.int32)
        inside = (i >= 0) & (j >= 0) & (i < self.n) & (j < self.n)
        if np.any(inside):
            keep[inside] = self.wet[i[inside], j[inside]]
        return keep

    def deposit(self, hit, jet_mm: float, volume_mm3: float) -> None:
        """Add volume on the jet footprint. The pool then spreads on its own."""
        if volume_mm3 <= 0.0 or self.origin is None:
            return
        cell = self._cell(hit)
        if cell is None:
            return
        radius = max(0.5 * float(jet_mm), self.cell_mm * 0.5)
        ii, jj = np.mgrid[0 : self.n, 0 : self.n]
        x = (ii - (self.n - 1) / 2.0) * self.cell_mm
        y = (jj - (self.n - 1) / 2.0) * self.cell_mm
        hit_i, hit_j = cell
        hx = (hit_i - (self.n - 1) / 2.0) * self.cell_mm
        hy = (hit_j - (self.n - 1) / 2.0) * self.cell_mm
        mask = (x - hx) ** 2 + (y - hy) ** 2 <= radius ** 2
        mask &= np.isfinite(self.bed)
        chosen = np.argwhere(mask)
        if len(chosen) == 0:
            if not np.isfinite(self.bed[hit_i, hit_j]):
                return
            chosen = np.array([[hit_i, hit_j]], dtype=np.int32)
        share = float(volume_mm3) / (len(chosen) * self.cell_mm * self.cell_mm)
        self.depth[chosen[:, 0], chosen[:, 1]] += share
        self.wet[chosen[:, 0], chosen[:, 1]] = True

    def suction(self, tip, radius_mm: float) -> None:
        """Delete depth inside the suction tip. Wet bone stays wet."""
        if tip is None or self.origin is None or radius_mm <= 0.0 or not np.any(self.depth > 0.0):
            return
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        ii, jj = np.nonzero(self.depth > 0.0)
        worlds = np.stack([self._world(int(i), int(j), self.bed[i, j] + self.depth[i, j]) for i, j in zip(ii, jj)])
        close = np.linalg.norm(worlds - tip, axis=1) <= float(radius_mm) * MM_TO_SIM
        if not np.any(close):
            return
        taken = self.depth[ii[close], jj[close]]
        self.suction_mm3 += float(taken.sum() * self.cell_mm * self.cell_mm)
        self.depth[ii[close], jj[close]] = 0.0

    def _rebuild_level(self, gravity) -> None:
        """Waterline falls into a hole and toward the bottom of the image.

        ``bed`` is depth into the scene, so a drilled floor is already low.
        Image-down tilts that waterline so the downhill lip spills first and
        the sheet keeps running down the bone.
        """
        g = np.asarray(gravity, dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(g))
        if length < 1e-12 or self.origin is None:
            self.level = None
            return
        g = g / length
        half = (self.n - 1) / 2.0
        ii, jj = np.mgrid[0 : self.n, 0 : self.n]
        x = (ii - half) * self.cell_mm
        y = (jj - half) * self.cell_mm
        screen = float(np.dot(self.right, g)) * x + float(np.dot(self.across, g)) * y
        finite = np.isfinite(self.bed)
        self.level = np.where(finite, self.bed - GRAVITY_SLOPE * screen, np.nan)

    def _relax(self) -> None:
        area = self.cell_mm * self.cell_mm
        bed = self.bed if self.level is None else self.level
        depth = self.depth
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            finite = np.isfinite(bed)
            surface = np.where(finite, bed + depth, -np.inf)
            bed_n = _at_neighbor(bed, di, dj, np.nan)
            surface_n = _at_neighbor(surface, di, dj, -np.inf)
            sampled_n = _at_neighbor(self.sampled.astype(np.float64), di, dj, 1.0) > 0.5
            neighbor_known = np.isfinite(bed_n)
            # The grid edge is an outlet. An unsampled column is not: water waits.
            outlet = finite & (depth > 0.0) & ~neighbor_known & sampled_n
            sill = np.where(neighbor_known, np.maximum(surface_n, bed_n), np.inf)
            can = finite & neighbor_known & (surface > bed_n) & (surface > surface_n + 1e-6)
            gap = np.clip(surface - sill, 0.0, None)
            give = np.where(can, np.minimum(depth, 0.5 * gap), 0.0)
            give = np.where(outlet, depth, give)
            if not np.any(give > 0.0):
                continue
            self.spill_mm3 += float(give[outlet].sum() * area)
            depth -= give
            depth += _shift(np.where(can, give, 0.0), di, dj, 0.0)
            np.maximum(depth, 0.0, out=depth)
            self.wet |= depth > MIN_DEPTH_MM
        over = np.clip(depth - MAX_DEPTH_MM, 0.0, None)
        if np.any(over > 0.0):
            self.spill_mm3 += float(over.sum() * area)
            depth -= over

    def settle(self, passes: int = 48) -> None:
        for _ in range(int(passes)):
            self._relax()

    def _mark_sample(self, hit) -> None:
        """Columns around the inlet and just ahead of water that is already moving."""
        want = np.zeros((self.n, self.n), dtype=bool)
        cell = self._cell(hit) if hit is not None else None
        ii, jj = np.mgrid[0 : self.n, 0 : self.n]
        if cell is not None:
            dist = np.maximum(np.abs(ii - cell[0]), np.abs(jj - cell[1]))
            want |= dist <= 10
        moving = self.depth > MIN_DEPTH_MM
        if np.any(moving):
            grown = moving.copy()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                grown |= _shift(moving.astype(np.float64), di, dj, 0.0) > 0.5
            want |= grown
        want &= ~self.sampled
        chosen = np.argwhere(want)
        if len(chosen) > SAMPLE_BUDGET:
            if cell is None:
                chosen = chosen[:SAMPLE_BUDGET]
            else:
                order = np.maximum(np.abs(chosen[:, 0] - cell[0]), np.abs(chosen[:, 1] - cell[1]))
                chosen = chosen[np.argsort(order)[:SAMPLE_BUDGET]]
        return chosen

    def _elevation_bounds(self, frame):
        if self._e_hi is not None:
            return
        counts = np.asarray(frame.counts, dtype=np.float64)
        corners = []
        for i in (0.0, max(counts[0] - 1.0, 0.0)):
            for j in (0.0, max(counts[1] - 1.0, 0.0)):
                for k in (0.0, max(counts[2] - 1.0, 0.0)):
                    corners.append(frame.index_to_world(np.array([i, j, k], dtype=np.float64)))
        rel = np.stack(corners) - self.origin
        elev = -(rel @ self.down) / MM_TO_SIM
        self._e_hi = float(elev.max()) + self.cell_mm
        self._e_lo = float(elev.min()) - self.cell_mm

    def sample(self, frame, occupied, hit, refresh: bool) -> None:
        if frame is None or occupied is None or self.origin is None:
            return
        if refresh:
            live = self.depth > MIN_DEPTH_MM
            self.sampled[live] = False
        chosen = self._mark_sample(hit)
        if len(chosen) == 0:
            return
        self._elevation_bounds(frame)
        steps = max(2, int(math.ceil((self._e_hi - self._e_lo) / self.cell_mm)))
        elevations = np.linspace(self._e_hi, self._e_lo, steps)
        found = np.full(len(chosen), np.nan, dtype=np.float64)
        pending = np.ones(len(chosen), dtype=bool)
        counts = np.asarray(frame.counts, dtype=np.int32)
        half = (self.n - 1) / 2.0
        x = (chosen[:, 0] - half) * self.cell_mm
        y = (chosen[:, 1] - half) * self.cell_mm
        horiz = (
            self.origin
            + self.right * (x * MM_TO_SIM)[:, None]
            + self.across * (y * MM_TO_SIM)[:, None]
        )
        for elevation in elevations:
            if not np.any(pending):
                break
            pts = horiz[pending] - self.down * (float(elevation) * MM_TO_SIM)
            ijk = np.floor(frame.world_to_index(pts)).astype(np.int32)
            inside = np.all((ijk >= 0) & (ijk < counts), axis=1)
            occ = np.zeros(len(ijk), dtype=bool)
            if np.any(inside):
                sel = ijk[inside]
                occ[inside] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
            if not np.any(occ):
                continue
            slots = np.flatnonzero(pending)
            hit_slots = slots[occ]
            found[hit_slots] = float(elevation)
            pending[hit_slots] = False
        self.bed[chosen[:, 0], chosen[:, 1]] = found
        self.sampled[chosen[:, 0], chosen[:, 1]] = True

    def step(
        self,
        dt: float,
        hit,
        jet_mm: float,
        cc_per_s: float,
        down,
        frame=None,
        occupied=None,
        suction_tip=None,
        suction_radius_mm: float = 0.0,
        gravity=None,
    ) -> None:
        dt = float(max(dt, 0.0))
        if dt <= 0.0:
            return
        self.time += dt
        if hit is not None:
            self.place(hit, down)
        elif self.origin is None:
            return
        if frame is not None and occupied is not None:
            self._since_sample += dt
            refresh = self._since_sample >= 0.3
            if refresh:
                self._since_sample = 0.0
            self.sample(frame, occupied, hit, refresh=refresh)
        if gravity is not None:
            self._rebuild_level(gravity)
        else:
            self.level = None
        if hit is not None and dt > 0.0:
            rate = VISUAL_MM3_PER_S * max(float(cc_per_s), 0.0) / REFERENCE_CC_PER_S
            self.deposit(hit, jet_mm, rate * dt)
        self.suction(suction_tip, suction_radius_mm)
        for _ in range(RELAX_PASSES):
            self._relax()

    def sheet_mesh(self):
        """A coating on the bone, not a stack of tiles facing the camera."""
        show = (self.depth > MIN_DEPTH_MM) & np.isfinite(self.bed)
        if not np.any(show) or self.origin is None:
            return None
        return self._coating(show)

    def _coating(self, mask):
        """Smoothed water surface. Depth stays in the sim grid, not in the facet size."""
        height = np.full((self.n, self.n), np.nan)
        height[mask] = self.bed[mask] + self.depth[mask] + SURFACE_BIAS_MM
        near = np.zeros_like(mask)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near |= _shift(mask.astype(np.float64), di, dj, 0.0) > 0.5
        shore = near & ~mask & np.isfinite(self.bed)
        height[shore] = self.bed[shore] + SURFACE_BIAS_MM
        height = _smooth_nan(height, SMOOTH_PASSES, SMOOTH_BLEND)
        return self._triangulate(height, mask)

    def _triangulate(self, height, mask):
        sub = VISUAL_SUBDIV
        n = self.n
        ii, jj = np.nonzero(mask)
        cn = n * sub + 1
        corner = -np.ones((cn, cn), dtype=np.int32)
        for a in range(sub + 1):
            for b in range(sub + 1):
                corner[ii * sub + a, jj * sub + b] = 0
        ci, cj = np.nonzero(corner == 0)
        corner[ci, cj] = np.arange(len(ci), dtype=np.int32)
        # Corner (i, j) sits on the low edge of cell i. Fractional index matches that.
        fi = ci.astype(np.float64) / sub - 0.5
        fj = cj.astype(np.float64) / sub - 0.5
        elev, ok = _bilinear(height, fi, fj)
        if not np.all(ok):
            # A corner with no bone sample stays on the nearest wet height.
            elev = np.where(ok, elev, np.nanmean(elev))
        half = (n - 1) / 2.0
        x = (ci.astype(np.float64) / sub - 0.5 - half) * self.cell_mm
        y = (cj.astype(np.float64) / sub - 0.5 - half) * self.cell_mm
        horiz = (
            self.origin
            + self.right * (x * MM_TO_SIM)[:, None]
            + self.across * (y * MM_TO_SIM)[:, None]
        )
        verts = horiz - self.down * (elev * MM_TO_SIM)[:, None]
        step = self.cell_mm / sub
        gx = np.zeros(len(ci), dtype=np.float64)
        gy = np.zeros(len(ci), dtype=np.float64)
        right_of = corner[np.clip(ci + 1, 0, cn - 1), cj]
        left_of = corner[np.clip(ci - 1, 0, cn - 1), cj]
        up_of = corner[ci, np.clip(cj + 1, 0, cn - 1)]
        down_of = corner[ci, np.clip(cj - 1, 0, cn - 1)]
        has_x = (ci + 1 < cn) & (ci - 1 >= 0) & (right_of >= 0) & (left_of >= 0)
        has_y = (cj + 1 < cn) & (cj - 1 >= 0) & (up_of >= 0) & (down_of >= 0)
        gx[has_x] = (elev[right_of[has_x]] - elev[left_of[has_x]]) / (2.0 * step)
        gy[has_y] = (elev[up_of[has_y]] - elev[down_of[has_y]]) / (2.0 * step)
        tx = self.right - self.down * gx[:, None]
        ty = self.across - self.down * gy[:, None]
        normal = np.cross(tx, ty)
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(length, 1e-8)
        # Geometric winding below points at the camera. Keep the shading normal with it.
        flip = (normal @ self.down) > 0.0
        normal[flip] *= -1.0
        tris = []
        for a in range(sub):
            for b in range(sub):
                c00 = corner[ii * sub + a, jj * sub + b]
                c10 = corner[ii * sub + a + 1, jj * sub + b]
                c01 = corner[ii * sub + a, jj * sub + b + 1]
                c11 = corner[ii * sub + a + 1, jj * sub + b + 1]
                # cross(across, right) = -down, so this face points at the camera.
                tris.append(np.stack((c00, c01, c11), axis=1))
                tris.append(np.stack((c00, c11, c10), axis=1))
        tris = np.concatenate(tris, axis=0).astype(np.int32)
        return {
            "verts": np.ascontiguousarray(verts, dtype=np.float32),
            "tris": tris,
            "normals": np.ascontiguousarray(normal, dtype=np.float32),
        }

    def prime(self, viewer) -> None:
        """Author both meshes before the first render so the materials stay live."""
        self._log(viewer, "fluid/sheet", self._seed_mesh(), water=True, hidden=True)
        self._log(viewer, "fluid/wet_bone", self._seed_mesh(), water=False, hidden=True)

    def draw(self, viewer) -> None:
        sheet = self.sheet_mesh()
        if sheet is None:
            self._hide(viewer, "fluid/sheet", water=True)
        else:
            self._log(viewer, "fluid/sheet", sheet, water=True, hidden=False)
        # No second mesh on the bone. That overlay cut the surface and drew the black lines.
        self._hide(viewer, "fluid/wet_bone", water=False)

    def _seed_mesh(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        return {
            "verts": verts,
            "tris": np.array([[0, 1, 2]], dtype=np.int32),
            "normals": np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        }

    def _log(self, viewer, name: str, mesh, water: bool, hidden: bool) -> None:
        import warp as wp

        import rtx_materials

        color = (1.0, 1.0, 1.0) if water else (1.0, 0.976, 0.859)
        opacity = SHEET_OPACITY if water else 1.0
        viewer.log_mesh(
            name,
            wp.array(mesh["verts"], dtype=wp.vec3),
            wp.array(np.ascontiguousarray(mesh["tris"].reshape(-1)), dtype=wp.int32),
            normals=wp.array(mesh["normals"], dtype=wp.vec3),
            hidden=hidden,
            backface_culling=True,
            dynamic=True,
            color=color,
            opacity=opacity,
        )
        flag = "_sheet_visible" if water else "_wet_visible"
        setattr(self, flag, not hidden)
        bound = "_sheet_bound" if water else "_wet_bound"
        if getattr(self, bound) or hidden:
            return
        prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify(name)))
        if water:
            rtx_materials.bind_water(viewer, prim)
        else:
            rtx_materials.bind_wet_bone(viewer, prim)
        setattr(self, bound, True)

    def _hide(self, viewer, name: str, water: bool) -> None:
        flag = "_sheet_visible" if water else "_wet_visible"
        if not getattr(self, flag):
            return
        self._log(viewer, name, self._seed_mesh(), water=water, hidden=True)
