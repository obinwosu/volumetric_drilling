"""Constrained-tip outlet used by the prescribed stream and the SPH inlet.

Drill local +X is the shaft (tip toward the handle). Local +Y is a stable
right, not camera-right. The jet is parallel to the shaft, toward the tip.
"""

from __future__ import annotations

import math

import numpy as np

MM_TO_SIM = 0.01007
OFFSET_MM = 5.0
PROXIMAL_MM = 10.0
AZIMUTH_DEG = 0.0
JET_MM = 1.0
PARTICLE_MM = 1.0
CC_PER_S = 1.0


def particles_per_second(cc_per_s: float = CC_PER_S, particle_mm: float = PARTICLE_MM) -> float:
    volume = float(particle_mm) ** 3
    if volume <= 0.0:
        raise ValueError("particle_mm must be positive")
    return float(cc_per_s) * 1000.0 / volume


def jet_speed_mm_s(cc_per_s: float = CC_PER_S, jet_mm: float = JET_MM) -> float:
    area = float(jet_mm) ** 2
    if area <= 0.0:
        raise ValueError("jet_mm must be positive")
    return float(cc_per_s) * 1000.0 / area


def outlet_frame(tip, quat, offset_mm=OFFSET_MM, azimuth_deg=AZIMUTH_DEG, proximal_mm=PROXIMAL_MM):
    """World outlet and unit axes from the constrained tip pose."""
    axes = quat_axes(quat)
    shaft = np.asarray(axes[:, 0], dtype=np.float64)
    shaft_n = float(np.linalg.norm(shaft))
    shaft = shaft / shaft_n if shaft_n > 1e-12 else np.array((1.0, 0.0, 0.0))
    right0 = np.asarray(axes[:, 1], dtype=np.float64)
    right0 = right0 - shaft * float(np.dot(right0, shaft))
    if float(np.linalg.norm(right0)) < 1e-12:
        right0 = np.asarray(axes[:, 2], dtype=np.float64)
        right0 = right0 - shaft * float(np.dot(right0, shaft))
    right0 = right0 / float(np.linalg.norm(right0))
    up0 = np.cross(shaft, right0)
    up0 = up0 / float(np.linalg.norm(up0))
    angle = math.radians(azimuth_deg)
    right = math.cos(angle) * right0 + math.sin(angle) * up0
    right = right / float(np.linalg.norm(right))
    tangent = np.cross(shaft, right)
    tangent = tangent / float(np.linalg.norm(tangent))
    emit = -shaft
    tip = np.asarray(tip, dtype=np.float64)
    along = shaft * (float(proximal_mm) * MM_TO_SIM)
    side = right * (float(offset_mm) * MM_TO_SIM)
    outlet = tip + along + side
    delta = outlet - tip
    return {
        "tip": tip,
        "outlet": outlet,
        "shaft": shaft,
        "right": right,
        "radial": right,
        "tangent": tangent,
        "emit": emit,
        "proximal_mm": float(np.dot(delta, shaft) / MM_TO_SIM),
        "radial_mm": float(np.dot(delta, right) / MM_TO_SIM),
        "offset_mm": float(np.linalg.norm(delta) / MM_TO_SIM),
        "emit_dot_shaft": float(np.dot(emit, shaft)),
        "emit_dot_tangent": float(np.dot(emit, tangent)),
        "emit_dot_right": float(np.dot(emit, right)),
    }


def quat_axes(quat) -> np.ndarray:
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )
