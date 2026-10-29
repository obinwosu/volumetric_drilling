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
        self._spheres_logged = False
        self._film_logged = False

    def apply(self, settings: dict) -> None:
        self.settings.update(settings)

    def step(self, dt: float, tip, quat) -> dict:
        dt = float(max(dt, 0.0))
        self.time += dt
        jet_mm = float(self.settings["jet_mm"])
        offset_mm = float(self.settings["offset_mm"])
        proximal_mm = float(self.settings["proximal_mm"])
        frame = outlet_frame(tip, quat, offset_mm=offset_mm, proximal_mm=proximal_mm, azimuth_deg=0.0)
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

    def draw(self, viewer) -> None:
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
        else:
            self.live_spheres = 0
            mesh = cylinder_mesh(frame["outlet"], hit, 0.5 * jet_mm * MM_TO_SIM)
            self._draw_tube(viewer, mesh)
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
            backface_culling=False,
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
