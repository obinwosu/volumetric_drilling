"""Tuner irrigation bench: a 1 mm jet on the frozen bone.

The airborne look is either packed 0.2 mm spheres or a 1 mm tube. Contact
drops most of that jet. A short downhill film is all that stays, and it is
gone in tens of milliseconds.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from fluid.moving_inlet import disk_offsets
from fluid.cannula import cannula_frame
from fluid.outlet import MM_TO_SIM, outlet_frame
from fluid.occupancy import VolumeFrame

IRRIGATION_PATH = Path(__file__).resolve().parents[1] / "rtx_view" / "tuner_irrigation.json"

DRAW_TUBE = "tube"
DRAW_SPHERES = "spheres"
SPHERE_MM = 0.2
JET_MM = 1.0
CC_PER_S = 80.0
FILM_MS = 50.0
FILM_KEEP = 0.1
RUNOFF_M_S = 2.5
MAX_FILM = 512
MISS_MM = 400.0
# The burr sits this far off the bone, toward the camera.
STANDOFF_MM = 25.0
HIT_MM = 400.0
RIGHT_TRANSTEMPORAL_QUAT = (-0.380217, 0.0698274, 0.131694, 0.912807)

DEFAULTS = {
    "draw": DRAW_TUBE,
    "jet_mm": JET_MM,
    "particle_mm": SPHERE_MM,
    "cc_per_s": CC_PER_S,
    "film_ms": FILM_MS,
    "offset_mm": 2.0,
    "proximal_mm": 0.0,
    "opacity": 0.22,
    "enabled": True,
}


def screen_down(quat=RIGHT_TRANSTEMPORAL_QUAT) -> np.ndarray:
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    xz, yz = x * z * s, y * z * s
    wx, wy = w * x * s, w * y * s
    xx, yy = x * x * s, y * y * s
    up = np.array((xz + wy, yz - wx, 1.0 - (xx + yy)), dtype=np.float64)
    return -up / float(np.linalg.norm(up))


def quat_from_matrix(matrix) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def quat_aiming_at_bone(forward, right) -> tuple[float, float, float, float]:
    """Shaft points at the camera, jet at the bone (camera forward)."""
    emit = np.asarray(forward, dtype=np.float64)
    emit = emit / float(np.linalg.norm(emit))
    shaft = -emit
    right_v = np.asarray(right, dtype=np.float64)
    right_v = right_v - shaft * float(np.dot(right_v, shaft))
    right_v = right_v / float(np.linalg.norm(right_v))
    tangent = np.cross(shaft, right_v)
    tangent = tangent / float(np.linalg.norm(tangent))
    return quat_from_matrix(np.column_stack((shaft, right_v, tangent)))


def first_hit(origin, direction, frame: VolumeFrame, occupied, max_mm: float = HIT_MM):
    """First occupied voxel along a ray. None if the ray misses."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return None
    direction = direction / length
    step = 0.25 * MM_TO_SIM
    counts = frame.counts.astype(np.int32)
    point = origin.copy()
    n = max(1, int(math.ceil(float(max_mm) * MM_TO_SIM / step)))
    for _ in range(n):
        point = point + direction * step
        ijk = np.floor(frame.world_to_index(point)).astype(np.int32)
        if np.any(ijk < 0) or np.any(ijk >= counts):
            continue
        if occupied[int(ijk[0]), int(ijk[1]), int(ijk[2])]:
            return point
    return None


def packed_column(outlet, hit, emit, right, tangent, jet_mm: float, particle_mm: float, phase: float = 0.0):
    """0.2 mm spheres packed in the 1 mm jet, from the nozzle to the bone."""
    outlet = np.asarray(outlet, dtype=np.float64).reshape(3)
    hit = np.asarray(hit, dtype=np.float64).reshape(3)
    emit = np.asarray(emit, dtype=np.float64).reshape(3)
    emit = emit / float(np.linalg.norm(emit))
    right = np.asarray(right, dtype=np.float64).reshape(3)
    tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
    offsets = disk_offsets(jet_mm, particle_mm)
    length = max(0.0, float(np.dot(hit - outlet, emit)))
    spacing = float(particle_mm) * MM_TO_SIM
    n_along = max(1, int(math.ceil(length / spacing)))
    shift = (float(phase) % 1.0) * spacing
    along = np.arange(n_along, dtype=np.float64) * spacing + shift
    along = along[along <= length]
    if len(along) == 0:
        along = np.array([0.0], dtype=np.float64)
    seeds = outlet[None, :] + offsets[:, 0:1] * right[None, :] + offsets[:, 1:2] * tangent[None, :]
    points = seeds[:, None, :] + along[None, :, None] * emit
    return np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float32)


def cylinder_mesh(start, end, radius, slices: int = 16):
    """One open tube from the nozzle to the hit."""
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        return None
    z = axis / length
    helper = np.array((1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0), dtype=np.float64)
    x = np.cross(helper, z)
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    angles = np.linspace(0.0, 2.0 * math.pi, slices, endpoint=False)
    ring = np.stack((np.cos(angles), np.sin(angles)), axis=1) * float(radius)
    bottom = start + ring[:, 0:1] * x + ring[:, 1:2] * y
    top = end + ring[:, 0:1] * x + ring[:, 1:2] * y
    verts = np.concatenate((bottom, top), axis=0).astype(np.float32)
    tris = []
    for i in range(slices):
        j = (i + 1) % slices
        tris.append((i, j, slices + j))
        tris.append((i, slices + j, slices + i))
    tris = np.asarray(tris, dtype=np.int32)
    side = ring[:, 0:1] * x + ring[:, 1:2] * y
    side = side / np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
    normals = np.concatenate((side, side), axis=0).astype(np.float32)
    return {"verts": verts, "tris": tris, "normals": normals}


# Sides around the tube. Twelve reads as a faceted rod under the microscope.
STREAM_SLICES = 24


def stream_mesh(start, end, radius, time: float, cc_per_s: float, slices: int = STREAM_SLICES):
    """Same one tube, with necks and bulges so it is not a straight rod.

    The nozzle and the bone hit stay put. A station is about 1.5 mm, capped
    so a long jet does not grow the mesh.
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        return None
    z = axis / length
    helper = np.array((1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0), dtype=np.float64)
    x = np.cross(helper, z)
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    length_mm = length / MM_TO_SIM
    stations = int(np.clip(round(length_mm / 1.5), 4, 18))
    scales = stream_scales(stations, length_mm, time, cc_per_s)
    along = np.linspace(0.0, 1.0, stations)
    envelope = np.sin(math.pi * along)
    travel = float(time) * visual_scroll_mm_s(cc_per_s)
    wander = envelope * (0.05 * float(radius)) * np.sin((along * length_mm - travel) * (2.0 * math.pi / 7.0))
    angles = np.linspace(0.0, 2.0 * math.pi, slices, endpoint=False)
    ring = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    verts = np.empty((stations * slices, 3), dtype=np.float64)
    normals = np.empty((stations * slices, 3), dtype=np.float64)
    for i in range(stations):
        center = start + z * (along[i] * length) + x * float(wander[i])
        spoke = ring[:, 0:1] * x + ring[:, 1:2] * y
        sl = i * slices
        verts[sl : sl + slices] = center + spoke * (float(radius) * float(scales[i]))
        normals[sl : sl + slices] = spoke
    tris = []
    for i in range(stations - 1):
        base = i * slices
        nxt = base + slices
        for s in range(slices):
            j = (s + 1) % slices
            tris.append((base + s, base + j, nxt + j))
            tris.append((base + s, nxt + j, nxt + s))
    return {
        "verts": verts.astype(np.float32),
        "tris": np.asarray(tris, dtype=np.int32),
        "normals": normals.astype(np.float32),
    }


def splash_volume_mm3() -> float:
    """Disc plus three droplets. Kept between 1 and 2 mm³."""
    puddle = math.pi * (PUDDLE_R_MM ** 2) * PUDDLE_H_MM
    droplet = math.pi * (DROP_R_MM ** 2) * DROP_H_MM
    return puddle + 3.0 * droplet


def _coin(center, normal, x_axis, y_axis, radius, height, slices: int = 10):
    angles = np.linspace(0.0, 2.0 * math.pi, slices, endpoint=False)
    ring = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    spoke = ring[:, 0:1] * x_axis + ring[:, 1:2] * y_axis
    bottom = center + spoke * float(radius)
    top = bottom + normal * float(height)
    verts = np.concatenate((bottom, top), axis=0)
    normals = np.concatenate((spoke, np.repeat(normal.reshape(1, 3), slices, axis=0)), axis=0)
    tris = []
    for i in range(slices):
        j = (i + 1) % slices
        tris.append((i, j, slices + j))
        tris.append((i, slices + j, slices + i))
        tris.append((slices + i, slices + j, slices))  # filled below; center added by caller
    return verts, np.asarray(tris, dtype=np.int32), normals


def splash_mesh(hit, emit):
    """Tiny splash where the jet meets bone. One mesh, no free particles."""
    hit = np.asarray(hit, dtype=np.float64).reshape(3)
    direction = np.asarray(emit, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        direction = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    else:
        direction = direction / length
    normal = -direction
    helper = np.array((1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0), dtype=np.float64)
    x_axis = np.cross(helper, normal)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(normal, x_axis)
    scale = MM_TO_SIM
    parts = [(hit, PUDDLE_R_MM * scale, PUDDLE_H_MM * scale)]
    for turn in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0):
        offset = x_axis * math.cos(turn) + y_axis * math.sin(turn)
        parts.append((hit + offset * (DROP_DIST_MM * scale), DROP_R_MM * scale, DROP_H_MM * scale))
    verts = []
    tris = []
    normals = []
    slices = 10
    for center, radius, height in parts:
        coin_v, coin_t, coin_n = _coin(center, normal, x_axis, y_axis, radius, height, slices)
        base = sum(len(item) for item in verts)
        # Drop the placeholder cap triangles from _coin (they reference a missing center).
        side = coin_t[: slices * 2]
        top = coin_v[slices:]
        cap_center = top.mean(axis=0)
        cap_index = base + len(coin_v)
        coin_v = np.concatenate((coin_v, cap_center.reshape(1, 3)), axis=0)
        coin_n = np.concatenate((coin_n, normal.reshape(1, 3)), axis=0)
        cap = []
        for i in range(slices):
            j = (i + 1) % slices
            cap.append((base + slices + i, base + slices + j, cap_index))
        verts.append(coin_v)
        normals.append(coin_n)
        tris.append(side + base)
        tris.append(np.asarray(cap, dtype=np.int32))
    return {
        "verts": np.concatenate(verts, axis=0).astype(np.float32),
        "tris": np.concatenate(tris, axis=0).astype(np.int32),
        "normals": np.concatenate(normals, axis=0).astype(np.float32),
    }


def load_irrigation(path: Path) -> dict:
    settings = dict(DEFAULTS)
    if not path.is_file():
        return settings
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return settings
    for key, value in DEFAULTS.items():
        if key in saved:
            settings[key] = type(value)(saved[key]) if not isinstance(value, bool) else bool(saved[key])
    if settings["draw"] not in (DRAW_TUBE, DRAW_SPHERES):
        settings["draw"] = DRAW_TUBE
    if isinstance(saved.get("water"), dict):
        settings["water"] = dict(saved["water"])
    return settings


def _json_ready(value):
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(format(value, ".6g"))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_irrigation(path: Path, settings: dict) -> None:
    payload = {key: settings[key] for key in DEFAULTS}
    if isinstance(settings.get("water"), dict):
        payload["water"] = dict(settings["water"])
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    temporary.replace(path)


def water_from_irrigation(settings: dict, fallback: dict) -> dict:
    water = dict(fallback)
    saved = settings.get("water")
    if isinstance(saved, dict):
        water.update(saved)
    if "opacity" in settings:
        water["geometry_opacity"] = float(settings["opacity"])
    return water


def visual_scroll_mm_s(cc_per_s: float) -> float:
    """How fast the packed beads crawl. Flow of 80 cc/s is 40 mm/s on screen."""
    return max(5.0, 40.0 * float(cc_per_s) / CC_PER_S)


# The live jet stays one mesh. Its width only moves inside this fraction of the set flow.
FLOW_SPREAD = 0.20
# A puddle plus three droplets, together about 1.3 mm³. Nothing travels off the bone.
PUDDLE_R_MM = 1.05
PUDDLE_H_MM = 0.30
DROP_R_MM = 0.30
DROP_H_MM = 0.36
DROP_DIST_MM = 1.20
SPLASH_CLEAR_MM = DROP_DIST_MM + DROP_R_MM + 0.15


def stream_scales(stations: int, length_mm: float, time: float, cc_per_s: float) -> np.ndarray:
    """Thickness along the jet, within ±20% of the set width, traveling with the flow."""
    count = max(2, int(stations))
    length_mm = max(float(length_mm), 1e-3)
    travel = float(time) * visual_scroll_mm_s(cc_per_s)
    distance = np.linspace(0.0, length_mm, count)
    coarse = np.sin((distance - travel) * (2.0 * math.pi / 5.0))
    fine = np.sin((distance - travel * 1.6) * (2.0 * math.pi / 1.7) + 0.7)
    wave = 0.62 * coarse + 0.38 * fine
    return 1.0 + (FLOW_SPREAD * 0.75) * wave


def _index_occupied(frame: VolumeFrame, occupied, points):
    counts = frame.counts.astype(np.int32)
    ijk = np.floor(frame.world_to_index(points)).astype(np.int32)
    inside = np.all((ijk >= 0) & (ijk < counts), axis=1)
    hit = np.zeros(len(points), dtype=bool)
    if np.any(inside):
        sel = ijk[inside]
        hit[inside] = occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
    return hit


class IrrigationBench:
    """Packed jet plus a short downhill film. No OpenMaelstrom in this window."""

    def __init__(self, occupied, volume_frame: VolumeFrame, down=None):
        self.occupied = np.ascontiguousarray(occupied, dtype=bool)
        self.frame = volume_frame
        self.down = np.asarray(down if down is not None else screen_down(), dtype=np.float64)
        self.down = self.down / float(np.linalg.norm(self.down))
        self.settings = dict(DEFAULTS)
        self.film_pos = np.zeros((MAX_FILM, 3), dtype=np.float64)
        self.film_vel = np.zeros((MAX_FILM, 3), dtype=np.float64)
        self.film_age = np.full(MAX_FILM, -1.0, dtype=np.float64)
        self._cursor = 0
        self.time = 0.0
        self.last_frame = None
        self.last_hit = None
        self.hit_mm = 0.0
        self.live_film = 0
        self.live_spheres = 0
        self.draw_ms = 0.0
        self._water_tube_bound = False
        self._water_point_names = set()
        self._tube_visible = False
        self._splash_visible = False
        self._splash_bound = False
        self._spheres_logged = False
        self._film_logged = False

    def apply(self, settings: dict) -> None:
        self.settings.update(settings)

    def step(self, dt: float, tip, quat, size_mm: int | None = None) -> dict:
        dt = float(max(dt, 0.0))
        self.time += dt
        jet_mm = float(self.settings["jet_mm"])
        if size_mm is None:
            # Tuner preview: a jet aimed at the bone, not a drill nozzle.
            offset_mm = float(self.settings["offset_mm"])
            proximal_mm = float(self.settings["proximal_mm"])
            frame = outlet_frame(tip, quat, offset_mm=offset_mm, proximal_mm=proximal_mm, azimuth_deg=0.0)
        else:
            frame = cannula_frame(tip, quat, int(size_mm))
        self.last_frame = frame
        hit = first_hit(frame["outlet"], frame["emit"], self.frame, self.occupied)
        film_ms = float(self.settings.get("film_ms") or 0.0)
        if hit is None:
            if film_ms <= 0.0:
                self.last_hit = None
                self.hit_mm = 0.0
                self.film_age[:] = -1.0
                self.live_film = 0
                return frame
            hit = frame["outlet"] + frame["emit"] * (MISS_MM * MM_TO_SIM)
        self.last_hit = hit
        self.hit_mm = float(np.linalg.norm(hit - frame["outlet"]) / MM_TO_SIM)
        if film_ms <= 0.0:
            self.film_age[:] = -1.0
            self.live_film = 0
            return frame
        film_s = max(0.01, film_ms * 1e-3)
        self._age_film(dt, film_s)
        if dt > 0.0:
            self._birth_film(hit, dt, film_s)
        return frame

    def _age_film(self, dt: float, film_s: float) -> None:
        live = self.film_age >= 0.0
        if not np.any(live):
            self.live_film = 0
            return
        self.film_age[live] += dt
        self.film_pos[live] += self.film_vel[live] * dt
        inside = np.zeros(len(self.film_age), dtype=bool)
        inside[live] = _index_occupied(self.frame, self.occupied, self.film_pos[live])
        if np.any(inside):
            self.film_pos[inside] += self.down * (0.4 * MM_TO_SIM)
        self.film_age[self.film_age > film_s] = -1.0
        self.live_film = int(np.count_nonzero(self.film_age >= 0.0))

    def _birth_film(self, hit, dt: float, film_s: float) -> None:
        speed = RUNOFF_M_S * 1000.0 * MM_TO_SIM
        flow = max(0.05, float(self.settings["cc_per_s"]) / CC_PER_S)
        owed = max(2, int(round(12.0 * dt / (1.0 / 60.0) * flow)))
        owed = min(owed, 16)
        jitter = 0.25 * MM_TO_SIM
        for i in range(owed):
            slot = self._cursor % MAX_FILM
            self._cursor += 1
            self.film_pos[slot] = hit + (np.random.rand(3) - 0.5) * 2.0 * jitter
            self.film_vel[slot] = self.down * speed
            self.film_age[slot] = 0.0

    def draw(self, viewer, splash: bool = True) -> None:
        started = time.perf_counter()
        if not self.settings.get("enabled", True) or self.last_frame is None or self.last_hit is None:
            self.hide(viewer)
            self.draw_ms = (time.perf_counter() - started) * 1e3
            return
        frame = self.last_frame
        hit = self.last_hit
        draw = self.settings["draw"]
        jet_mm = float(self.settings["jet_mm"])
        particle_mm = float(self.settings["particle_mm"])
        # Crawl the packed beads with flow so the Flow slider is visible.
        scroll_mm_s = visual_scroll_mm_s(self.settings["cc_per_s"])
        spacing = max(particle_mm, 1e-6)
        phase = (self.time * scroll_mm_s / spacing) if spacing else 0.0
        if draw == DRAW_SPHERES:
            points = packed_column(
                frame["outlet"],
                hit,
                frame["emit"],
                frame["right"],
                frame["tangent"],
                jet_mm,
                particle_mm,
                phase=phase,
            )
            self.live_spheres = int(len(points))
            self._draw_spheres(viewer, points, 0.5 * particle_mm * MM_TO_SIM)
            self._hide_tube(viewer)
            self._hide_splash(viewer)
        else:
            self.live_spheres = 0
            mesh = stream_mesh(
                frame["outlet"],
                hit,
                0.5 * jet_mm * MM_TO_SIM,
                self.time,
                float(self.settings["cc_per_s"]),
            )
            self._draw_tube(viewer, mesh)
            if splash:
                self._draw_splash(viewer, splash_mesh(hit, frame["emit"]))
            else:
                self._hide_splash(viewer)
            self._hide_spheres(viewer)
        if float(self.settings.get("film_ms") or 0.0) > 0.0:
            film = self.film_pos[self.film_age >= 0.0]
            self._draw_film(viewer, film)
        else:
            self._hide_film(viewer)
        self.draw_ms = (time.perf_counter() - started) * 1e3

    def hide(self, viewer) -> None:
        self._hide_spheres(viewer)
        self._hide_tube(viewer)
        self._hide_splash(viewer)
        self._hide_film(viewer)
        self.live_spheres = 0

    def prime(self, viewer) -> None:
        """Author water prims before the first render so slider writes stay live.

        Meshes logged after RTX starts are copied into a frozen runtime prim.
        The water material and later jet/opacity edits never reach that copy.
        """
        start = np.zeros(3, dtype=np.float64)
        end = np.array((0.0, 0.0, 0.02), dtype=np.float64)
        mesh = cylinder_mesh(start, end, 0.5 * JET_MM * MM_TO_SIM)
        self._draw_tube(viewer, mesh)
        seed = np.zeros((1, 3), dtype=np.float32)
        radius = 0.5 * SPHERE_MM * MM_TO_SIM
        self._draw_spheres(viewer, seed, radius)
        self._draw_film(viewer, seed)
        self._draw_splash(viewer, splash_mesh(start, np.array((0.0, 0.0, -1.0))))
        self.hide(viewer)

    def _wp(self):
        import warp as wp

        return wp

    def _water_look(self):
        water = self.settings.get("water") if isinstance(self.settings.get("water"), dict) else {}
        color = water.get("diffuse_reflection_color", (0.62, 0.82, 0.94))
        opacity = self.settings.get("opacity", water.get("geometry_opacity", 0.22))
        return tuple(float(channel) for channel in color), float(opacity)

    def _bind_water_prim(self, viewer, prim) -> None:
        if prim is None or not prim.IsValid():
            return
        import rtx_materials

        rtx_materials.bind_water(viewer, prim)

    def _bind_points(self, viewer, name: str) -> None:
        if name in self._water_point_names:
            return
        qualified = viewer._qualify(name)
        instancer = viewer._ensure_point_batch_primitive(qualified)
        if instancer is None:
            return
        sphere = instancer.GetPrim().GetChild("sphere")
        self._bind_water_prim(viewer, sphere)
        self._water_point_names.add(name)

    def _draw_spheres(self, viewer, points, radius) -> None:
        if len(points) == 0:
            self._hide_spheres(viewer)
            return
        wp = self._wp()
        color, _opacity = self._water_look()
        viewer.log_points(
            "fluid/spheres",
            wp.array(np.ascontiguousarray(points), dtype=wp.vec3),
            radii=float(radius),
            colors=color,
        )
        self._spheres_logged = True
        self._bind_points(viewer, "fluid/spheres")

    def _hide_spheres(self, viewer) -> None:
        if self._spheres_logged:
            viewer.log_points("fluid/spheres", None, hidden=True)

    def _draw_tube(self, viewer, mesh) -> None:
        if mesh is None:
            self._hide_tube(viewer)
            return
        wp = self._wp()
        color, opacity = self._water_look()
        points = wp.array(np.ascontiguousarray(mesh["verts"]), dtype=wp.vec3)
        indices = wp.array(np.ascontiguousarray(mesh["tris"].reshape(-1)), dtype=wp.int32)
        normals = wp.array(np.ascontiguousarray(mesh["normals"]), dtype=wp.vec3)
        viewer.log_mesh(
            "fluid/tube",
            points,
            indices,
            normals=normals,
            hidden=False,
            backface_culling=True,
            dynamic=True,
            color=color,
            opacity=opacity,
        )
        if not self._water_tube_bound:
            prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify("fluid/tube")))
            self._bind_water_prim(viewer, prim)
            self._water_tube_bound = True
        self._tube_visible = True

    def _hide_tube(self, viewer) -> None:
        if not self._tube_visible:
            return
        wp = self._wp()
        empty = {
            "verts": np.zeros((3, 3), dtype=np.float32),
            "tris": np.array([[0, 1, 2]], dtype=np.int32),
            "normals": np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        }
        points = wp.array(empty["verts"], dtype=wp.vec3)
        indices = wp.array(empty["tris"].reshape(-1), dtype=wp.int32)
        normals = wp.array(empty["normals"], dtype=wp.vec3)
        viewer.log_mesh(
            "fluid/tube",
            points,
            indices,
            normals=normals,
            hidden=True,
            backface_culling=False,
            dynamic=True,
        )
        self._tube_visible = False

    def _draw_splash(self, viewer, mesh) -> None:
        if mesh is None:
            self._hide_splash(viewer)
            return
        wp = self._wp()
        color, opacity = self._water_look()
        points = wp.array(np.ascontiguousarray(mesh["verts"]), dtype=wp.vec3)
        indices = wp.array(np.ascontiguousarray(mesh["tris"].reshape(-1)), dtype=wp.int32)
        normals = wp.array(np.ascontiguousarray(mesh["normals"]), dtype=wp.vec3)
        viewer.log_mesh(
            "fluid/splash",
            points,
            indices,
            normals=normals,
            hidden=False,
            backface_culling=False,
            dynamic=True,
            color=color,
            opacity=min(0.55, max(opacity, 0.35)),
        )
        if not self._splash_bound:
            prim = viewer.stage.GetPrimAtPath(viewer._get_path(viewer._qualify("fluid/splash")))
            self._bind_water_prim(viewer, prim)
            self._splash_bound = True
        self._splash_visible = True

    def _hide_splash(self, viewer) -> None:
        if not self._splash_visible:
            return
        wp = self._wp()
        empty = {
            "verts": np.zeros((3, 3), dtype=np.float32),
            "tris": np.array([[0, 1, 2]], dtype=np.int32),
            "normals": np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        }
        viewer.log_mesh(
            "fluid/splash",
            wp.array(empty["verts"], dtype=wp.vec3),
            wp.array(empty["tris"].reshape(-1), dtype=wp.int32),
            normals=wp.array(empty["normals"], dtype=wp.vec3),
            hidden=True,
            backface_culling=False,
            dynamic=True,
        )
        self._splash_visible = False

    def _draw_film(self, viewer, points) -> None:
        if len(points) == 0:
            self._hide_film(viewer)
            return
        wp = self._wp()
        color, _opacity = self._water_look()
        radius = 0.5 * float(self.settings["particle_mm"]) * MM_TO_SIM
        viewer.log_points(
            "fluid/film",
            wp.array(np.ascontiguousarray(points.astype(np.float32)), dtype=wp.vec3),
            radii=float(radius),
            colors=color,
        )
        self._film_logged = True
        self._bind_points(viewer, "fluid/film")

    def _hide_film(self, viewer) -> None:
        if self._film_logged:
            viewer.log_points("fluid/film", None, hidden=True)
