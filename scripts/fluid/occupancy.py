"""Load the AMBF occupancy dump and the volume pose used to place it.

This is the same index space the RTX bone uses. It does not touch the
rendered mesh or the haptic loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Same millimetre as DrillManager / the RTX burr. Kept here so occupancy
# and the frozen SDF bake do not import the prescribed stream.
MM_TO_SIM = 0.01007


@dataclass(frozen=True)
class VolumeFrame:
    """World placement of the occupancy grid."""

    translation: np.ndarray
    rotation: np.ndarray
    dimensions: np.ndarray
    counts: np.ndarray

    def world_to_index(self, world) -> np.ndarray:
        local = (np.asarray(world, dtype=np.float64) - self.translation) @ self.rotation
        return (local + 0.5 * self.dimensions) * (self.counts / self.dimensions)

    def index_to_world(self, index) -> np.ndarray:
        local = np.asarray(index, dtype=np.float64) * (self.dimensions / self.counts) - 0.5 * self.dimensions
        return local @ self.rotation.T + self.translation


def load_occupancy_dump(path: Path) -> np.ndarray:
    """Return occupied[x, y, z] from /tmp/fivrs-rtx/occupancy.bin."""
    raw = Path(path).read_bytes()
    width, height, depth, bpp = np.frombuffer(raw[:16], dtype=np.int32)
    payload = np.frombuffer(raw[16:], dtype=np.uint8)
    if bpp == 4:
        rgba = payload.reshape(int(depth), int(height), int(width), 4)
        alpha = np.transpose(rgba[:, :, :, 3], (2, 1, 0))
    elif bpp == 1:
        alpha = np.transpose(payload.reshape(int(depth), int(height), int(width)), (2, 1, 0))
    else:
        raise RuntimeError(f"Unexpected occupancy bpp {bpp}")
    return np.ascontiguousarray(alpha > 10)


def volume_frame_from_adf(path: Path, counts) -> VolumeFrame:
    """Pose and size from a FIVRS anatomy ADF."""
    data = yaml.safe_load(Path(path).read_text())
    volume = _first_volume(data)
    location = volume.get("location") or {}
    position = location.get("position") or {}
    orientation = location.get("orientation") or {}
    dimensions = volume.get("dimensions") or {}
    translation = np.array(
        [float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))],
        dtype=np.float64,
    )
    rotation = rpy_matrix(
        float(orientation.get("r", 0.0)),
        float(orientation.get("p", 0.0)),
        float(orientation.get("y", 0.0)),
    )
    size = np.array(
        [float(dimensions.get("x", 1.0)), float(dimensions.get("y", 1.0)), float(dimensions.get("z", 1.0))],
        dtype=np.float64,
    )
    return VolumeFrame(translation, rotation, size, np.asarray(counts, dtype=np.float64))


def identity_frame(counts, dimensions=(1.0, 1.0, 1.0)) -> VolumeFrame:
    return VolumeFrame(
        np.zeros(3, dtype=np.float64),
        np.eye(3),
        np.asarray(dimensions, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """AMBF afMatrix3d::setRPY, column-vector convention."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def read_drill_state(path: Path) -> dict:
    try:
        import json

        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def drill_pose(state: dict):
    try:
        tip = np.array([float(state["px"]), float(state["py"]), float(state["pz"])], dtype=np.float64)
        quat = (float(state["qx"]), float(state["qy"]), float(state["qz"]), float(state["qw"]))
        return tip, quat, bool(state.get("on"))
    except (KeyError, TypeError, ValueError):
        return None, None, False


def _first_volume(data: dict) -> dict:
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("VOLUME ") and isinstance(value, dict):
            return value
    raise ValueError("no VOLUME block in anatomy ADF")


# Re-export so callers can stay in sim millimetres.
__all__ = [
    "MM_TO_SIM",
    "VolumeFrame",
    "drill_pose",
    "identity_frame",
    "load_occupancy_dump",
    "read_drill_state",
    "rpy_matrix",
    "volume_frame_from_adf",
]
