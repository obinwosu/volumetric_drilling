"""Stage 1 bone debris for the RTX microscope.

AMBF voxel removals stay the source of truth. This module only watches the
batches the viewer already applies to the bone. It does not remove voxels,
move the drill, or touch the irrigation emitter.

Removed volume becomes a white paste in the cut, not one grain per voxel.
The parcels fall a short way and then stay until irrigation or the suction
tip reaches them. A lifetime and a count ceiling remain so the microscope
does not keep drawing paste forever if neither instrument clears it.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

from fluid.occupancy import MM_TO_SIM, VolumeFrame
from fluid.outlet import quat_axes

CONFIG_PATH = Path(__file__).resolve().parent / "debris.json"
PRIM = "debris/bone"

DEFAULTS = {
    "enabled": True,
    "emission_scale": 1.0,
    "parcels_per_mm3": 8.0,
    "max_active": 480,
    "max_spawn_per_frame": 24,
    "lifetime_s": 8.0,
    "speed_mm_s": 2.0,
    "drag_per_s": 3.0,
    "scatter": 0.25,
    "spread_voxels": 0.45,
    "radius_mm": 0.22,
    "clump_mm": 0.5,
    "shaft_bias": 0.0,
    "flute_bias": 0.0,
    "gravity_mm_s2": 80.0,
    "settle_s": 0.18,
    "color": [0.98, 0.93, 0.88],
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = dict(DEFAULTS)
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return config
    if isinstance(loaded, dict):
        config.update(loaded)
    return _sane(config)


def _sane(config: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(config)
    out["enabled"] = bool(out["enabled"])
    out["emission_scale"] = max(0.0, float(out["emission_scale"]))
    out["parcels_per_mm3"] = max(0.0, float(out["parcels_per_mm3"]))
    out["max_active"] = max(1, int(out["max_active"]))
    out["max_spawn_per_frame"] = max(0, int(out["max_spawn_per_frame"]))
    out["lifetime_s"] = max(0.02, float(out["lifetime_s"]))
    out["speed_mm_s"] = max(0.0, float(out["speed_mm_s"]))
    out["drag_per_s"] = max(0.0, float(out["drag_per_s"]))
    out["scatter"] = max(0.0, float(out["scatter"]))
    out["spread_voxels"] = max(0.0, float(out["spread_voxels"]))
    out["radius_mm"] = max(0.02, float(out["radius_mm"]))
    out["clump_mm"] = max(out["radius_mm"], float(out.get("clump_mm", out["radius_mm"])))
    out["shaft_bias"] = float(out["shaft_bias"])
    out["flute_bias"] = float(out["flute_bias"])
    out["gravity_mm_s2"] = max(0.0, float(out.get("gravity_mm_s2", 0.0)))
    out["settle_s"] = max(0.0, float(out.get("settle_s", 0.0)))
    color = out["color"]
    out["color"] = (float(color[0]), float(color[1]), float(color[2]))
    return out


def _mul_quat(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def head_quat(shaft_quat, spin: float):
    """Shaft pose, then the burr's spin about local +X."""
    half = 0.5 * float(spin)
    return _mul_quat(shaft_quat, (math.sin(half), 0.0, 0.0, math.cos(half)))


def eject_direction(shaft_quat, spin: float, shaft_bias: float, flute_bias: float) -> np.ndarray:
    """World direction of the initial throw. Columns of the pose are local axes."""
    axes = quat_axes(head_quat(shaft_quat, spin))
    direction = axes[:, 0] * float(shaft_bias) + axes[:, 1] * float(flute_bias)
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        direction = np.asarray(axes[:, 1], dtype=np.float64)
        length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return np.array((0.0, 1.0, 0.0), dtype=np.float64)
    return direction / length


def _gravity(down, mm_s2: float) -> wp.vec3:
    """Gentle fall in sim units. ``down`` is the microscope-down direction."""
    axis = np.array((0.0, 0.0, -1.0) if down is None else down, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        axis = np.array((0.0, 0.0, -1.0), dtype=np.float64)
        length = 1.0
    axis = axis / length * (float(mm_s2) * MM_TO_SIM)
    return wp.vec3(float(axis[0]), float(axis[1]), float(axis[2]))


def suction_radius_mm(size_fr: float) -> float:
    """Invisible capture radius. 12 Fr is 3 mm, and smaller tubes are smaller."""
    return 0.75 * (float(size_fr) / 3.0)


def _zone_active(zones) -> bool:
    if not zones:
        return False
    if zones.get("spheres") or zones.get("segment") is not None:
        return True
    film = zones.get("film")
    return film is not None and bool(getattr(film, "any_wet", False))


def outside_zones(points, zones) -> np.ndarray:
    """True where a point is outside the jet, the suction tip, and wet bone."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    keep = np.ones(len(points), dtype=bool)
    if not zones:
        return keep
    for center, radius in zones.get("spheres") or []:
        origin = np.asarray(center, dtype=np.float64).reshape(3)
        keep &= np.linalg.norm(points - origin, axis=1) > float(radius)
    segment = zones.get("segment")
    if segment is not None:
        start, end, radius = segment
        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        axis = end - start
        length2 = float(np.dot(axis, axis))
        if length2 > 1e-16:
            along = np.clip(((points - start) @ axis) / length2, 0.0, 1.0)
            closest = start + along[:, None] * axis
            keep &= np.linalg.norm(points - closest, axis=1) > float(radius)
    film = zones.get("film")
    if film is not None and bool(getattr(film, "any_wet", False)):
        keep &= ~film.covers(points)
    return keep


def parcels_from_volume(mm3: float, config: dict, carry: float) -> tuple[int, float]:
    """Whole parcels earned by removed volume. The fractional remainder is kept.

    A frame never stores a backlog of unreleased parcels, so a large bite cannot
    keep emitting after the cut is over. ``carry`` stays in ``[0, 1)``.
    """
    rate = float(config["parcels_per_mm3"]) * float(config["emission_scale"])
    carry = float(carry) + max(0.0, float(mm3)) * rate
    earned = int(math.floor(carry))
    carry -= float(earned)
    return earned, carry


@wp.kernel
def _spawn(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    born: wp.array(dtype=float),
    origins: wp.array(dtype=wp.vec3),
    speeds: wp.array(dtype=wp.vec3),
    sizes: wp.array(dtype=float),
    cursor: int,
    capacity: int,
    now: float,
):
    i = wp.tid()
    slot = (cursor + i) % capacity
    positions[slot] = origins[i]
    velocities[slot] = speeds[i]
    radii[slot] = sizes[i]
    born[slot] = now


@wp.kernel
def _integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    born: wp.array(dtype=float),
    head: int,
    capacity: int,
    dt: float,
    drag: float,
    gravity: wp.vec3,
    settle_s: float,
    now: float,
):
    i = wp.tid()
    slot = (head + i) % capacity
    if now - born[slot] >= settle_s:
        velocities[slot] = wp.vec3(0.0, 0.0, 0.0)
        return
    vel = velocities[slot] + gravity * dt
    vel = vel * wp.exp(-drag * dt)
    velocities[slot] = vel
    positions[slot] = positions[slot] + vel * dt


@wp.kernel
def _pack(
    positions: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    packed: wp.array(dtype=wp.vec3),
    packed_radii: wp.array(dtype=float),
    head: int,
    capacity: int,
):
    i = wp.tid()
    slot = (head + i) % capacity
    packed[i] = positions[slot]
    packed_radii[i] = radii[slot]


@wp.kernel
def _pack_live(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    born: wp.array(dtype=float),
    packed: wp.array(dtype=wp.vec3),
    packed_vel: wp.array(dtype=wp.vec3),
    packed_radii: wp.array(dtype=float),
    packed_born: wp.array(dtype=float),
    head: int,
    capacity: int,
):
    i = wp.tid()
    slot = (head + i) % capacity
    packed[i] = positions[slot]
    packed_vel[i] = velocities[slot]
    packed_radii[i] = radii[slot]
    packed_born[i] = born[slot]


@wp.kernel
def _rewrite(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    born: wp.array(dtype=float),
    origins: wp.array(dtype=wp.vec3),
    speeds: wp.array(dtype=wp.vec3),
    sizes: wp.array(dtype=float),
    births: wp.array(dtype=float),
):
    i = wp.tid()
    positions[i] = origins[i]
    velocities[i] = speeds[i]
    radii[i] = sizes[i]
    born[i] = births[i]


class BoneDebris:
    """GPU ring of debris parcels. The live count stays on the CPU."""

    def __init__(self, frame: VolumeFrame, config: dict | None = None, config_path: Path = CONFIG_PATH):
        self.frame = frame
        self.config_path = Path(config_path)
        self.config = _sane(config if config is not None else load_config(self.config_path))
        self._mtime = self._config_mtime()
        spacing = np.asarray(frame.dimensions, dtype=np.float64) / np.asarray(frame.counts, dtype=np.float64)
        self.voxel_mm3 = float(np.prod(spacing / MM_TO_SIM))
        self.voxel_edge = float(np.min(spacing))
        self._alloc(self.config["max_active"])
        self.head = 0
        self.count = 0
        self.birth = np.zeros(self.capacity, dtype=np.float64)
        self.time = 0.0
        self.carry = 0.0
        self.pending: list[tuple[int, list]] = []
        self._visible = False
        self._readbacks = 0
        self.removed_mm3 = 0.0
        self.spawned = 0
        self.retired = 0
        self.dropped = 0
        self.last_voxels = 0
        self.last_mm3 = 0.0
        self.last_spawned = 0
        self.last_retired = 0
        self.step_ms = 0.0
        self.draw_ms = 0.0
        self.launch_ms = 0.0
        self.washed = 0
        self._zones = None
        self._slurry_bound = False
        self._paste_mtime = None
        self._report_size("loaded")

    def _report_size(self, reason: str) -> None:
        print(
            f"Bone debris {reason}: {self.config['radius_mm']:.2f}–{self.config['clump_mm']:.2f} mm grains, "
            f"{self.config['parcels_per_mm3']:g} per mm3, cap {self.config['max_active']}",
            flush=True,
        )

    def _alloc(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.positions = wp.zeros(self.capacity, dtype=wp.vec3)
        self.velocities = wp.zeros(self.capacity, dtype=wp.vec3)
        self.radii = wp.zeros(self.capacity, dtype=float)
        self.born = wp.zeros(self.capacity, dtype=float)
        self.packed = wp.zeros(self.capacity, dtype=wp.vec3)
        self.packed_vel = wp.zeros(self.capacity, dtype=wp.vec3)
        self.packed_radii = wp.zeros(self.capacity, dtype=float)
        self.packed_born = wp.zeros(self.capacity, dtype=float)

    def _config_mtime(self):
        try:
            return self.config_path.stat().st_mtime
        except OSError:
            return None

    def reload_if_changed(self) -> None:
        mtime = self._config_mtime()
        if mtime is None or mtime == self._mtime:
            return
        self._mtime = mtime
        previous = self.capacity
        self.config = load_config(self.config_path)
        self._report_size("reloaded")
        if self.config["max_active"] != previous:
            self._alloc(self.config["max_active"])
            self.head = 0
            self.count = 0
            self.birth = np.zeros(self.capacity, dtype=np.float64)
            self.carry = 0.0

    def note_removals(self, batches) -> None:
        """Live removal batches. ``batches`` are ``(stamp, wall, serial, voxels)``."""
        if not batches:
            return
        for _stamp, _wall, serial, voxels in batches:
            if voxels:
                self.pending.append((int(serial), voxels))

    def step(self, dt: float, tip, shaft_quat, spin: float, down=None, zones=None) -> None:
        started = time.perf_counter()
        self.reload_if_changed()
        dt = max(0.0, float(dt))
        if not self.config["enabled"]:
            self.pending.clear()
            self.count = 0
            self.head = 0
            self.last_spawned = 0
            self.last_retired = 0
            self.step_ms = (time.perf_counter() - started) * 1e3
            self.launch_ms = 0.0
            return
        self.time += dt
        self._zones = zones
        retired = self._expire()
        spawned, mm3, voxels = self._emit(tip, shaft_quat, spin)
        retired += self._clear_zones()
        launch = time.perf_counter()
        if self.count > 0 and dt > 0.0:
            wp.launch(
                _integrate,
                dim=self.count,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.born,
                    int(self.head),
                    int(self.capacity),
                    float(dt),
                    float(self.config["drag_per_s"]),
                    _gravity(down, float(self.config["gravity_mm_s2"])),
                    float(self.config["settle_s"]),
                    float(self.time),
                ],
            )
        self.launch_ms = (time.perf_counter() - launch) * 1e3
        self.last_retired = retired
        self.last_spawned = spawned
        self.last_mm3 = mm3
        self.last_voxels = voxels
        self.step_ms = (time.perf_counter() - started) * 1e3

    def draw(self, viewer) -> None:
        started = time.perf_counter()
        if not self.config["enabled"] or self.count <= 0:
            if self._visible:
                viewer.log_points(PRIM, None, hidden=True)
                self._visible = False
            self.draw_ms = (time.perf_counter() - started) * 1e3
            return
        wp.launch(
            _pack,
            dim=self.count,
            inputs=[
                self.positions,
                self.radii,
                self.packed,
                self.packed_radii,
                int(self.head),
                int(self.capacity),
            ],
        )
        viewer.log_points(
            PRIM,
            wp.clone(self.packed[: self.count]),
            radii=wp.clone(self.packed_radii[: self.count]),
            colors=self.config["color"],
        )
        self._bind_slurry(viewer)
        self._visible = True
        self.draw_ms = (time.perf_counter() - started) * 1e3

    def sample_positions(self) -> np.ndarray:
        """Test-only copy. The live draw path does not call this."""
        self._readbacks += 1
        if self.count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        wp.launch(
            _pack,
            dim=self.count,
            inputs=[
                self.positions,
                self.radii,
                self.packed,
                self.packed_radii,
                int(self.head),
                int(self.capacity),
            ],
        )
        packed = self.packed.numpy()
        return np.asarray(packed[: self.count], dtype=np.float32)

    def snapshot(self) -> dict:
        mm3 = self.removed_mm3
        return {
            "enabled": bool(self.config["enabled"]),
            "active": int(self.count),
            "spawned": int(self.spawned),
            "retired": int(self.retired),
            "dropped": int(self.dropped),
            "last_voxels": int(self.last_voxels),
            "last_mm3": float(self.last_mm3),
            "last_spawned": int(self.last_spawned),
            "last_retired": int(self.last_retired),
            "removed_mm3": float(mm3),
            "parcels_per_mm3": (float(self.spawned) / mm3) if mm3 > 1e-9 else 0.0,
            "voxel_mm3": float(self.voxel_mm3),
            "step_ms": float(self.step_ms),
            "draw_ms": float(self.draw_ms),
            "launch_ms": float(self.launch_ms),
            "readbacks": int(self._readbacks),
        }

    def _expire(self) -> int:
        life = float(self.config["lifetime_s"])
        retired = 0
        while self.count > 0 and (self.time - float(self.birth[self.head])) >= life:
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
            retired += 1
        self.retired += retired
        return retired

    def _clear_zones(self) -> int:
        zones = self._zones or {}
        if self.count <= 0 or not _zone_active(zones):
            return 0
        wp.launch(
            _pack_live,
            dim=self.count,
            inputs=[
                self.positions,
                self.velocities,
                self.radii,
                self.born,
                self.packed,
                self.packed_vel,
                self.packed_radii,
                self.packed_born,
                int(self.head),
                int(self.capacity),
            ],
        )
        count = int(self.count)
        origins = np.asarray(self.packed.numpy()[:count], dtype=np.float64)
        keep = outside_zones(origins, zones)
        removed = int(count - int(keep.sum()))
        if removed <= 0:
            return 0
        kept_count = int(keep.sum())
        if kept_count <= 0:
            self.head = 0
            self.count = 0
            self.retired += removed
            self.washed += removed
            return removed
        speeds = np.asarray(self.packed_vel.numpy()[:count], dtype=np.float32)[keep]
        sizes = np.asarray(self.packed_radii.numpy()[:count], dtype=np.float32)[keep]
        births = np.asarray(self.packed_born.numpy()[:count], dtype=np.float32)[keep]
        kept = origins[keep].astype(np.float32)
        self._spawn_origins = wp.array(np.ascontiguousarray(kept), dtype=wp.vec3)
        self._spawn_speeds = wp.array(np.ascontiguousarray(speeds), dtype=wp.vec3)
        self._spawn_sizes = wp.array(np.ascontiguousarray(sizes), dtype=float)
        self._spawn_births = wp.array(np.ascontiguousarray(births), dtype=float)
        wp.launch(
            _rewrite,
            dim=kept_count,
            inputs=[
                self.positions,
                self.velocities,
                self.radii,
                self.born,
                self._spawn_origins,
                self._spawn_speeds,
                self._spawn_sizes,
                self._spawn_births,
            ],
        )
        self.birth[:] = 0.0
        self.birth[:kept_count] = births
        self.head = 0
        self.count = kept_count
        self.retired += removed
        self.washed += removed
        return removed

    def _bind_slurry(self, viewer) -> None:
        if not hasattr(viewer, "_qualify"):
            return
        import rtx_materials

        try:
            mtime = rtx_materials.PASTE_PATH.stat().st_mtime
        except OSError:
            mtime = None
        if self._slurry_bound:
            if mtime is not None and mtime != self._paste_mtime:
                rtx_materials.apply_params(viewer, "slurry", rtx_materials.slurry_params())
                self._paste_mtime = mtime
            return
        qualified = viewer._qualify(PRIM)
        instancer = viewer._ensure_point_batch_primitive(qualified)
        if instancer is None:
            return
        sphere = instancer.GetPrim().GetChild("sphere")
        if not sphere:
            return
        rtx_materials.bind_slurry(viewer, sphere)
        self._slurry_bound = True
        self._paste_mtime = mtime

    def _emit(self, tip, shaft_quat, spin: float) -> tuple[int, float, int]:
        pending = self.pending
        self.pending = []
        if not pending or shaft_quat is None or tip is None:
            return 0, 0.0, 0
        groups = []
        total_mm3 = 0.0
        total_voxels = 0
        counts = np.asarray(self.frame.counts, dtype=np.int64)
        for serial, voxels in pending:
            idx = np.asarray(voxels, dtype=np.int64)
            if idx.ndim != 2 or idx.shape[1] != 3:
                idx = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
            inside = np.all((idx >= 0) & (idx < counts), axis=1)
            idx = idx[inside]
            if len(idx) == 0:
                continue
            mm3 = float(len(idx)) * self.voxel_mm3
            total_mm3 += mm3
            total_voxels += int(len(idx))
            groups.append((int(serial), idx))
        self.removed_mm3 += total_mm3
        earned, self.carry = parcels_from_volume(total_mm3, self.config, self.carry)
        room = self.capacity - self.count
        n = min(earned, int(self.config["max_spawn_per_frame"]), room)
        self.dropped += max(0, earned - n)
        if n <= 0 or not groups:
            return 0, total_mm3, total_voxels
        weights = np.array([len(idx) for _serial, idx in groups], dtype=np.float64)
        weights /= weights.sum()
        shares = np.floor(weights * n).astype(np.int32)
        shares[int(np.argmax(weights))] += n - int(shares.sum())
        direction = eject_direction(
            shaft_quat, spin, self.config["shaft_bias"], self.config["flute_bias"]
        )
        speed = float(self.config["speed_mm_s"]) * MM_TO_SIM
        scatter = float(self.config["scatter"]) * speed
        spread = float(self.config["spread_voxels"]) * self.voxel_edge
        axes = quat_axes(head_quat(shaft_quat, spin))
        side = axes[:, 1]
        up = axes[:, 2]
        origins = np.empty((n, 3), dtype=np.float32)
        speeds = np.empty((n, 3), dtype=np.float32)
        sizes = np.empty(n, dtype=np.float32)
        lo = float(self.config["radius_mm"]) * MM_TO_SIM
        hi = float(self.config["clump_mm"]) * MM_TO_SIM
        cursor = 0
        rng = np.random.default_rng(int(groups[0][0]) & 0xFFFFFFFF)
        for share, (_serial, idx) in zip(shares, groups):
            if share <= 0:
                continue
            pick = rng.integers(0, len(idx), size=int(share))
            centers = self.frame.index_to_world(idx[pick].astype(np.float64) + 0.5)
            jitter = (rng.random((share, 2)) - 0.5) * (2.0 * spread)
            points = centers + side * jitter[:, 0:1] + up * jitter[:, 1:2]
            noise = (rng.random((share, 2)) - 0.5) * (2.0 * scatter)
            velocity = direction * speed + side * noise[:, 0:1] + up * noise[:, 1:2]
            # Square root spends more of the parcels toward the large end, so the
            # bite reads as a paste. The large end stays about half a millimetre.
            sizes[cursor : cursor + share] = (lo + (hi - lo) * np.sqrt(rng.random(share))).astype(np.float32)
            origins[cursor : cursor + share] = points.astype(np.float32)
            speeds[cursor : cursor + share] = velocity.astype(np.float32)
            cursor += int(share)
        keep = outside_zones(origins[:n], self._zones)
        blocked = int(n - int(keep.sum()))
        self.washed += blocked
        if blocked:
            origins = origins[:n][keep]
            speeds = speeds[:n][keep]
            sizes = sizes[:n][keep]
            n = int(keep.sum())
        if n <= 0:
            return 0, total_mm3, total_voxels
        start = (self.head + self.count) % self.capacity
        # Kept alive so the queued spawn kernel still owns these arrays next frame.
        self._spawn_origins = wp.array(np.ascontiguousarray(origins), dtype=wp.vec3)
        self._spawn_speeds = wp.array(np.ascontiguousarray(speeds), dtype=wp.vec3)
        self._spawn_sizes = wp.array(np.ascontiguousarray(sizes), dtype=float)
        wp.launch(
            _spawn,
            dim=n,
            inputs=[
                self.positions,
                self.velocities,
                self.radii,
                self.born,
                self._spawn_origins,
                self._spawn_speeds,
                self._spawn_sizes,
                int(start),
                int(self.capacity),
                float(self.time),
            ],
        )
        for i in range(n):
            slot = (start + i) % self.capacity
            self.birth[slot] = self.time
        self.count += n
        self.spawned += n
        return n, total_mm3, total_voxels
