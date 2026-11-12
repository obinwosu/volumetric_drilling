"""Quest hands (WiVRn, XR_EXT_hand_tracking) drawn as simple capsules in the RTX view.

The OpenXR eyepiece plugin writes 26 joints per hand, in OpenXR LOCAL space
(metres, Y up), to /dev/shm/fivrs_hands every headset frame.

Where the hands go in the scene
-------------------------------
Head placement (default): the hands are placed relative to your head, as if
the microscope looked at your lap. A hand held about 40 cm ahead and 30 cm
below the eyes shows at the focus, scaled so both hands fit the current field
of view at any zoom. Moving the head moves them, like looking down at them.

Touch placement (FIVRS_HANDS_MODE=touch): life size and tied to the drill.
The Touch already maps real motion into the scene: the drill moves by the
stylus motion turned into the camera frame and multiplied by a gain (about
12 scene units per metre, a little over life size). The hands use that same
mapping, anchored on the drill, so they move exactly as the drill would:

    scene = drill_tip + gain * R_cam @ (device_point - stylus_point)

That needs each hand joint in the Touch's device frame. The headset and the
Touch do not share a frame, so it is learned: while you hold the stylus and
move it, the fingertips that pinch it move with the stylus. Pairs of
(pinch point in OpenXR, stylus position and rotation in the device frame) are
fitted to a rigid transform plus the pinch's offset in the stylus frame. The
fit is learned in both modes, refreshes whenever the stylus moves, and is kept
in /tmp/fivrs-rtx/hand_calibration.json. Until there is a fit, touch mode
falls back to the head placement. At life size a hand is far wider than the
microscope's ~40 mm field, so zoom out to see it.

The hands show while the stylus sits in its inkwell and go away once it is
lifted out. FIVRS_HANDS_ALWAYS=1 keeps them on (handy to judge the fit), and
FIVRS_HANDS=0 turns them off.
"""

from __future__ import annotations

import json
import os
import pathlib
import struct
import time
from collections import deque

import numpy as np

HANDS_PATH = pathlib.Path("/dev/shm/fivrs_hands")
CALIBRATION_PATH = pathlib.Path("/tmp/fivrs-rtx/hand_calibration.json")
MAGIC = 0x44485646  # "FVHD"
JOINTS = 26
HEADER = 64
HAND_BYTES = 8 + JOINTS * 8 * 4 + JOINTS * 4
FILE_BYTES = HEADER + 2 * HAND_BYTES
# XrSpaceLocationFlags
ORIENTATION_VALID = 0x1
POSITION_VALID = 0x2
# A frame older than this means the headset stopped sending.
STALE_S = 0.5

# XR_EXT_hand_tracking joint indices.
PALM, WRIST = 0, 1
THUMB_TIP = 5
INDEX_PROXIMAL, INDEX_TIP = 7, 10
MIDDLE_PROXIMAL = 12
LITTLE_PROXIMAL = 22
_CHAINS = (
    (WRIST, 2, 3, 4, 5),
    (WRIST, 6, 7, 8, 9, 10),
    (WRIST, 11, 12, 13, 14, 15),
    (WRIST, 16, 17, 18, 19, 20),
    (WRIST, 21, 22, 23, 24, 25),
)
BONES = tuple(
    [(chain[i], chain[i + 1]) for chain in _CHAINS for i in range(len(chain) - 1)]
    # Across the knuckles, across the metacarpal bases, and the thumb web.
    + [(7, 12), (12, 17), (17, 22), (6, 11), (11, 16), (16, 21), (3, 7)]
)
# The pinch on the stylus: thumb tip and index tip.
GRIP_JOINTS = (THUMB_TIP, INDEX_TIP)

# Touch stylus position in its inkwell, chai device frame (m). The same point
# OpenHaptics calibrates to, read back from drill_state.json with the stylus docked.
INKWELL_M = np.array([-0.088, 0.0, -0.066])
INKWELL_RADIUS_M = 0.015
# Scene units per metre of stylus travel when nothing has been measured yet:
# DrillManager adds velocity * 0.8 / workspace scale (66.7) per 1 kHz step.
DEFAULT_GAIN = 12.0
# Head placement: this point in the head frame (m; x right, y up, -z forward)
# lands on the microscope focus. About where hands rest in the lap.
HEAD_REF_M = np.array([0.0, -0.30, -0.40])
# Head placement fits this much real width (both hands side by side, m) to the
# microscope's field of view, so both hands stay in view at any zoom.
HEAD_VIEW_SPAN_M = 0.35
PALM_HALF_THICKNESS_M = 0.011
MIN_RADIUS_M = 0.004
GLOVE_COLOR = (0.83, 0.76, 0.64)
GLOVE_ROUGHNESS = 0.45


def enabled() -> bool:
    return os.environ.get("FIVRS_HANDS", "1") not in ("0", "false")


def _env_vec(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        values = np.array([float(v) for v in raw.split(",")], dtype=np.float64)
    except ValueError:
        return default
    return values if values.shape == (3,) else default


# ---------------------------------------------------------------- math


def quat_to_mat(q):
    """(x, y, z, w) to a 3x3 rotation."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def mat_to_quat(m):
    """3x3 rotation to (x, y, z, w)."""
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quats_from_x(directions):
    """Unit quaternions (N, 4) turning +X onto each direction (N, 3)."""
    d = np.asarray(directions, dtype=np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    # Half-way vector between +X and d: q = (cross(x, h), dot(x, h)).
    h = d + np.array([1.0, 0.0, 0.0])
    norm = np.linalg.norm(h, axis=1, keepdims=True)
    opposite = norm[:, 0] < 1e-9
    h = h / np.maximum(norm, 1e-12)
    q = np.zeros((len(d), 4))
    q[:, 0] = 0.0
    q[:, 1] = -h[:, 2]
    q[:, 2] = h[:, 1]
    q[:, 3] = h[:, 0]
    # +X onto -X: half a turn about Z.
    q[opposite] = (0.0, 0.0, 1.0, 0.0)
    return q


def kabsch(source, target):
    """Rotation A and offset b with target ~= A @ source + b (least squares)."""
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    h = (source - sc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    a = vt.T @ np.diag((1.0, 1.0, d)) @ u.T
    return a, tc - a @ sc


def camera_rotation(state):
    """main_camera rotation from drill_state.json: columns are back, right, up."""
    look = np.array([state["lx"], state["ly"], state["lz"]], dtype=np.float64)
    up = np.array([state["ux"], state["uy"], state["uz"]], dtype=np.float64)
    look /= max(np.linalg.norm(look), 1e-9)
    right = np.cross(look, up)
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, look)
    return np.column_stack((-look, right, up))


def drill_rotation(state):
    return quat_to_mat((state["qx"], state["qy"], state["qz"], state["qw"]))


def stylus_position(state):
    try:
        return np.array([float(state["hx"]), float(state["hy"]), float(state["hz"])])
    except (KeyError, TypeError, ValueError):
        return None


def stylus_docked(state, inkwell=None, radius=INKWELL_RADIUS_M):
    """True in the inkwell, False out of it, None when there is no Touch."""
    h = stylus_position(state) if state else None
    if h is None:
        return None
    inkwell = INKWELL_M if inkwell is None else inkwell
    return bool(np.linalg.norm(h - inkwell) <= radius)


def _has_camera(state):
    return state is not None and all(k in state for k in ("lx", "ly", "lz", "ux", "uy", "uz"))


# ---------------------------------------------------------------- reading


class HandsFrame:
    __slots__ = ("wall", "session", "head_pos", "head_rot", "hands")

    def __init__(self, wall, session, head_pos, head_rot, hands):
        self.wall = wall
        self.session = session
        self.head_pos = head_pos
        self.head_rot = head_rot
        # Per hand (left, right): None or {"pos": (26, 3), "radius": (26,)}.
        self.hands = hands


def parse(buffer: bytes):
    """One complete snapshot of the file, or None."""
    if len(buffer) < FILE_BYTES:
        return None
    magic, _version, seq, wall, head_flags, session = struct.unpack_from("<IIQdII", buffer, 0)
    if magic != MAGIC or seq == 0 or seq & 1:
        return None
    head = np.frombuffer(buffer, dtype="<f4", count=7, offset=32).astype(np.float64)
    head_pos = head_rot = None
    if head_flags & ORIENTATION_VALID and head_flags & POSITION_VALID:
        head_pos = head[:3].copy()
        head_rot = quat_to_mat(head[3:7])
    hands = []
    for h in range(2):
        base = HEADER + h * HAND_BYTES
        (active,) = struct.unpack_from("<I", buffer, base)
        joints = np.frombuffer(buffer, dtype="<f4", count=JOINTS * 8, offset=base + 8).reshape(JOINTS, 8)
        flags = np.frombuffer(buffer, dtype="<u4", count=JOINTS, offset=base + 8 + JOINTS * 32)
        valid = (flags & POSITION_VALID) != 0
        if not active or not valid.all():
            hands.append(None)
            continue
        hands.append(
            {
                "pos": joints[:, :3].astype(np.float64),
                "radius": np.maximum(joints[:, 7].astype(np.float64), MIN_RADIUS_M),
            }
        )
    return HandsFrame(float(wall), int(session), head_pos, head_rot, hands)


class HandsFile:
    """Seqlock reader for /dev/shm/fivrs_hands. pread only: never mmap, so a
    file replaced under us cannot fault the RTX window."""

    def __init__(self, path=HANDS_PATH):
        self.path = pathlib.Path(path)
        self._fd = None
        self._inode = None
        self._checked = 0.0

    def _reopen(self, now):
        if now - self._checked < 0.5 and self._fd is not None:
            return
        self._checked = now
        try:
            inode = os.stat(self.path).st_ino
        except OSError:
            self.close()
            return
        if inode != self._inode or self._fd is None:
            self.close()
            try:
                self._fd = os.open(self.path, os.O_RDONLY)
                self._inode = inode
            except OSError:
                self._fd = None

    def read(self):
        self._reopen(time.monotonic())
        if self._fd is None:
            return None
        for _ in range(3):
            try:
                buffer = os.pread(self._fd, FILE_BYTES, 0)
                again = os.pread(self._fd, 8, 8)
            except OSError:
                self.close()
                return None
            if len(again) == 8 and buffer[8:16] == again:
                return parse(buffer)
        return None

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self._inode = None


# ---------------------------------------------------------------- mapping


class GainEstimator:
    """Scene units of drill travel per metre of stylus travel, measured live."""

    def __init__(self, default=DEFAULT_GAIN):
        self.default = default
        self._ratios = deque(maxlen=240)
        self._previous = None

    @property
    def gain(self):
        if len(self._ratios) < 20:
            return self.default
        return float(np.clip(np.median(self._ratios), 2.0, 60.0))

    def update(self, state):
        h = stylus_position(state) if state else None
        if h is None:
            self._previous = None
            return
        try:
            p = np.array([float(state["px"]), float(state["py"]), float(state["pz"])])
        except (KeyError, TypeError, ValueError):
            return
        held = any(state.get(k) for k in ("cam_clutch", "device_clutch", "orbit", "paused", "drill_locked"))
        previous, self._previous = self._previous, (h, p)
        if previous is None or held:
            return
        dh = float(np.linalg.norm(h - previous[0]))
        if dh < 1e-3:
            return
        self._ratios.append(float(np.linalg.norm(p - previous[1])) / dh)


class StylusCalibration:
    """OpenXR LOCAL -> Touch device frame, learned from the hand that holds the stylus.

    Model: A @ pinch_xr + b = stylus + R_stylus @ offset, with A a rotation,
    b a translation and offset the pinch point in the stylus frame.
    """

    MIN_SAMPLES = 60
    SPACING_M = 0.003
    MAX_RMS_M = 0.025
    # Pull the offset gently toward zero: with little stylus rotation it trades off against b.
    OFFSET_RIDGE = 1e-4

    def __init__(self, path=CALIBRATION_PATH):
        self.path = pathlib.Path(path) if path else None
        self.samples = deque(maxlen=400)
        self.session = None
        self.result = None  # {"A", "b", "offset", "rms", "hand", "session"}
        self._new = 0
        self._solved_at = 0.0
        self.load()

    def load(self):
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text())
            self.result = {
                "A": np.array(data["A"], dtype=np.float64).reshape(3, 3),
                "b": np.array(data["b"], dtype=np.float64),
                "offset": np.array(data["offset"], dtype=np.float64),
                "rms": float(data["rms"]),
                "hand": int(data["hand"]),
                "session": data.get("session"),
            }
        except (OSError, ValueError, KeyError, TypeError):
            self.result = None

    def save(self):
        if self.path is None or self.result is None:
            return
        r = self.result
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "A": r["A"].tolist(),
                        "b": r["b"].tolist(),
                        "offset": r["offset"].tolist(),
                        "rms": r["rms"],
                        "hand": r["hand"],
                        "session": r["session"],
                        "wall": time.time(),
                    }
                )
            )
            tmp.replace(self.path)
        except OSError:
            pass

    def add(self, frame, state):
        """Record a sample while the stylus is out of its inkwell."""
        if frame is None or not _has_camera(state) or stylus_docked(state) is not False:
            return
        if frame.session != self.session:
            self.samples.clear()
            self.session = frame.session
        h = stylus_position(state)
        if self.samples and np.linalg.norm(h - self.samples[-1][0]) < self.SPACING_M:
            return
        r_stylus = camera_rotation(state).T @ drill_rotation(state)
        grips = []
        for hand in frame.hands:
            grips.append(None if hand is None else hand["pos"][list(GRIP_JOINTS)].mean(axis=0))
        if all(g is None for g in grips):
            return
        self.samples.append((h, r_stylus, grips))
        self._new += 1

    @classmethod
    def fit(cls, pinch, stylus, rotations, iterations=8):
        n = len(pinch)
        # Start from the linear problem M @ pinch + b - R_i @ offset = stylus with M
        # any 3x3 (15 unknowns, one least squares), then snap M to a rotation and
        # refine. Alternating from offset = 0 alone takes ~100 steps to settle.
        design = np.zeros((n, 3, 15))
        for k in range(3):
            design[:, k, 3 * k : 3 * k + 3] = pinch
            design[:, k, 9 + k] = 1.0
        design[:, :, 12:] = -rotations
        ridge = np.sqrt(n * cls.OFFSET_RIDGE) * np.eye(15)[12:]
        solution = np.linalg.lstsq(
            np.vstack((design.reshape(3 * n, 15), ridge)),
            np.concatenate((stylus.reshape(-1), np.zeros(3))),
            rcond=None,
        )[0]
        offset = solution[12:]
        for _ in range(iterations):
            target = stylus + np.einsum("nij,j->ni", rotations, offset)
            a, b = kabsch(pinch, target)
            residual = pinch @ a.T + b - stylus
            # R^T R = I, so the ridge least squares offset is a scaled mean.
            offset = np.einsum("nji,nj->i", rotations, residual) / (n * (1.0 + cls.OFFSET_RIDGE))
        error = pinch @ a.T + b - stylus - np.einsum("nij,j->ni", rotations, offset)
        rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        return a, b, offset, rms

    def maybe_solve(self, now=None, force=False):
        """Refit at most once a second when new samples came in. True if the result changed."""
        now = time.monotonic() if now is None else now
        if not force and (self._new < 10 or now - self._solved_at < 1.0):
            return False
        self._solved_at = now
        self._new = 0
        best = None
        for hand in range(2):
            rows = [(h, r, g[hand]) for h, r, g in self.samples if g[hand] is not None]
            if len(rows) < self.MIN_SAMPLES:
                continue
            stylus = np.array([row[0] for row in rows])
            spread = np.sqrt(np.sort(np.linalg.eigvalsh(np.cov(stylus.T))))
            if spread[1] < 0.01:
                continue  # moved along a line only: the rotation is not pinned down
            pinch = np.array([row[2] for row in rows])
            rotations = np.array([row[1] for row in rows])
            a, b, offset, rms = self.fit(pinch, stylus, rotations)
            if rms <= self.MAX_RMS_M and np.linalg.norm(offset) < 0.2 and (best is None or rms < best["rms"]):
                best = {"A": a, "b": b, "offset": offset, "rms": rms, "hand": hand, "session": self.session}
        if best is None:
            return False
        self.result = best
        self.save()
        return True

    def to_device(self, points):
        r = self.result
        return points @ r["A"].T + r["b"]


def map_to_scene(frame, state, calibration, gain, mode=None):
    """Hand joints in scene units. Returns (list of (26, 3) or None, scene units per metre, mode)."""
    if not _has_camera(state):
        return [None, None], gain, None
    r_cam = camera_rotation(state)
    h = stylus_position(state)
    tip = np.array([float(state["px"]), float(state["py"]), float(state["pz"])])
    if mode == "touch" and calibration is not None and calibration.result is not None and h is not None:
        out = []
        for hand in frame.hands:
            if hand is None:
                out.append(None)
                continue
            device = calibration.to_device(hand["pos"])
            out.append(tip + gain * ((device - h) @ r_cam.T))
        return out, gain, "touch"
    if frame.head_rot is None:
        return [None, None], gain, None
    # Head placement: head frame (x right, y up, -z forward) onto the camera
    # (right, up, -look), with HEAD_REF_M at the focus.
    look = -r_cam[:, 0]
    camera = np.array([float(state["cx"]), float(state["cy"]), float(state["cz"])])
    distance = max(float(np.dot(tip - camera, look)), 0.5)
    focus = camera + look * distance
    fov = float(state.get("fov") or 0.0)
    if 0.01 < fov < 3.0:
        # Microscope view height at the focus, in scene units, per HEAD_VIEW_SPAN_M.
        gain = 2.0 * distance * np.tan(0.5 * fov) / HEAD_VIEW_SPAN_M
    to_camera = np.column_stack((r_cam[:, 1], r_cam[:, 2], -look))
    out = []
    for hand in frame.hands:
        if hand is None:
            out.append(None)
            continue
        local = (hand["pos"] - frame.head_pos) @ frame.head_rot  # rows: head_rot.T @ p
        out.append(focus + gain * ((local - HEAD_REF_M) @ to_camera.T))
    return out, gain, "head"


# ---------------------------------------------------------------- drawing


def _unit_sphere(stacks=8, slices=12):
    points = []
    for i in range(stacks + 1):
        phi = np.pi * (i / stacks - 0.5)
        for j in range(slices):
            theta = 2.0 * np.pi * j / slices
            points.append((np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)))
    tris = []
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = i * slices + (j + 1) % slices
            c = (i + 1) * slices + j
            d = (i + 1) * slices + (j + 1) % slices
            tris.append((a, c, b))
            tris.append((b, c, d))
    points = np.asarray(points, dtype=np.float32)
    return points, np.asarray(tris, dtype=np.int32), points.copy()


def _unit_tube(slices=12):
    """Open cylinder of radius 1 from x=0 to x=1. The joint spheres cap it."""
    theta = 2.0 * np.pi * np.arange(slices) / slices
    ring = np.column_stack((np.zeros(slices), np.cos(theta), np.sin(theta)))
    points = np.vstack((ring, ring + (1.0, 0.0, 0.0))).astype(np.float32)
    normals = np.vstack((ring, ring)).astype(np.float32)
    tris = []
    for j in range(slices):
        a, b = j, (j + 1) % slices
        tris.append((a, b, a + slices))
        tris.append((b, b + slices, a + slices))
    return points, np.asarray(tris, dtype=np.int32), normals


def hand_instances(points, radii_m, scale):
    """Sphere and tube instances for one hand in scene units.

    Returns (sphere_xforms (27, 7), sphere_scales (27, 3), tube_xforms (B, 7), tube_scales (B, 3)).
    The last sphere is the palm, flattened."""
    radii = radii_m * scale
    spheres = np.zeros((JOINTS + 1, 7))
    spheres[:JOINTS, :3] = points
    spheres[:JOINTS, 6] = 1.0
    sphere_scales = np.zeros((JOINTS + 1, 3))
    sphere_scales[:JOINTS] = radii[:, None]
    # Palm: an ellipsoid across the knuckles, from the wrist to the middle knuckle.
    along = points[MIDDLE_PROXIMAL] - points[WRIST]
    across = points[INDEX_PROXIMAL] - points[LITTLE_PROXIMAL]
    length = max(np.linalg.norm(along), 1e-9)
    ez = along / length
    across_flat = across - np.dot(across, ez) * ez
    width = max(np.linalg.norm(across_flat), 1e-9)
    ex = across_flat / width
    ey = np.cross(ez, ex)
    spheres[JOINTS, :3] = 0.5 * (points[WRIST] + points[MIDDLE_PROXIMAL])
    spheres[JOINTS, 3:] = mat_to_quat(np.column_stack((ex, ey, ez)))
    sphere_scales[JOINTS] = (0.5 * width + radii[INDEX_PROXIMAL], PALM_HALF_THICKNESS_M * scale, 0.5 * length)

    a = np.array([bone[0] for bone in BONES])
    b = np.array([bone[1] for bone in BONES])
    span = points[b] - points[a]
    lengths = np.linalg.norm(span, axis=1)
    tubes = np.zeros((len(BONES), 7))
    tubes[:, :3] = points[a]
    tubes[:, 3:] = quats_from_x(span)
    tube_radius = 0.95 * np.minimum(radii[a], radii[b])
    tube_scales = np.column_stack((lengths, tube_radius, tube_radius))
    return spheres, sphere_scales, tubes, tube_scales


class HandRenderer:
    SPHERE_MESH = "hands/sphere"
    TUBE_MESH = "hands/tube"
    SPHERES = "hands/joints"
    TUBES = "hands/bones"

    def __init__(self, viewer):
        import warp as wp

        self._wp = wp
        for name, (points, tris, normals) in (
            (self.SPHERE_MESH, _unit_sphere()),
            (self.TUBE_MESH, _unit_tube()),
        ):
            # Prototypes stay hidden; only the instances draw.
            viewer.log_mesh(
                name,
                wp.array(points, dtype=wp.vec3),
                wp.array(tris.reshape(-1), dtype=wp.int32),
                normals=wp.array(normals, dtype=wp.vec3),
                color=GLOVE_COLOR,
                roughness=GLOVE_ROUGHNESS,
                metallic=0.0,
                hidden=True,
            )
        self.shown = False
        self._parked = None

    def draw(self, viewer, hands_scene, radii, scale):
        wp = self._wp
        sphere_x, sphere_s, tube_x, tube_s = [], [], [], []
        anchor = next(p for p in hands_scene if p is not None)
        for points, r in zip(hands_scene, radii):
            if points is None:
                # Fixed instance counts: a lost hand is shrunk to nothing beside the other.
                points, r = anchor, np.full(JOINTS, MIN_RADIUS_M)
                parts = hand_instances(points, r, scale)
                parts = (parts[0], parts[1] * 1e-6, parts[2], parts[3] * 1e-6)
            else:
                parts = hand_instances(points, r, scale)
            sphere_x.append(parts[0])
            sphere_s.append(parts[1])
            tube_x.append(parts[2])
            tube_s.append(parts[3])
        for name, mesh, xforms, scales in (
            (self.SPHERES, self.SPHERE_MESH, sphere_x, sphere_s),
            (self.TUBES, self.TUBE_MESH, tube_x, tube_s),
        ):
            viewer.log_instances(
                name,
                mesh,
                wp.array(np.ascontiguousarray(np.vstack(xforms), dtype=np.float32), dtype=wp.transform),
                wp.array(np.ascontiguousarray(np.vstack(scales), dtype=np.float32), dtype=wp.vec3),
                None,
                None,
                hidden=False,
            )
        self.shown = True

    def hide(self, viewer):
        if not self.shown:
            return
        for name, mesh in ((self.SPHERES, self.SPHERE_MESH), (self.TUBES, self.TUBE_MESH)):
            viewer.log_instances(name, mesh, None, None, None, None, hidden=True)
        self.shown = False


class Hands:
    """Read, place and draw the hands once per RTX frame."""

    def __init__(self, viewer, path=HANDS_PATH, calibration_path=CALIBRATION_PATH):
        self.file = HandsFile(path)
        self.calibration = StylusCalibration(calibration_path)
        self.gain = GainEstimator()
        self.renderer = HandRenderer(viewer)
        self.always = os.environ.get("FIVRS_HANDS_ALWAYS", "0") == "1"
        self.mode = os.environ.get("FIVRS_HANDS_MODE") or None
        global INKWELL_M
        INKWELL_M = _env_vec("FIVRS_INKWELL_M", INKWELL_M)
        self._status = None
        if self.calibration.result is not None:
            r = self.calibration.result
            print(f"hands: Touch fit loaded ({r['rms'] * 1e3:.0f} mm rms)", flush=True)

    def _say(self, status):
        if status != self._status:
            print(f"hands: {status}", flush=True)
            self._status = status

    def update(self, viewer, state):
        frame = self.file.read()
        if frame is None or time.time() - frame.wall > STALE_S:
            self.renderer.hide(viewer)
            self._say("no headset hand data")
            return
        self.gain.update(state)
        self.calibration.add(frame, state)
        if self.calibration.maybe_solve():
            r = self.calibration.result
            print(
                f"hands: Touch fit from the {'left' if r['hand'] == 0 else 'right'} hand, "
                f"{r['rms'] * 1e3:.0f} mm rms, pinch {np.round(r['offset'] * 1e3).tolist()} mm "
                f"in the stylus frame, gain {self.gain.gain:.1f}",
                flush=True,
            )
        docked = stylus_docked(state)
        if not self.always and docked is False:
            self.renderer.hide(viewer)
            self._say("hidden (stylus out of the inkwell)")
            return
        hands_scene, scale, mode = map_to_scene(frame, state, self.calibration, self.gain.gain, self.mode)
        if all(p is None for p in hands_scene):
            self.renderer.hide(viewer)
            self._say("no hands tracked")
            return
        radii = [None if hand is None else hand["radius"] for hand in frame.hands]
        self.renderer.draw(viewer, hands_scene, radii, scale)
        tracked = "+".join(name for name, p in zip(("left", "right"), hands_scene) if p is not None)
        self._say(f"showing {tracked}, placed by the {'Touch fit' if mode == 'touch' else 'head'}")
