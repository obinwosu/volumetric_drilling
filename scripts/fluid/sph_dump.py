"""Host dump of Maelstrom particle positions for the RTX overlay.

consoleParticles writes this after a step. The viewer only reads it.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

MAGIC = b"FIVRSPH\0"
HEADER_STRUCT = struct.Struct("<8sIIIf")
DEFAULT_PATH = Path("/tmp/fivrs-rtx/sph_positions.bin")
FRESH_S = 2.0


def write_positions(path: Path, xyz, radius: float, generation: int, flags: int = 0) -> Path:
    xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(HEADER_STRUCT.pack(MAGIC, int(generation), int(xyz.shape[0]), int(flags), float(radius)))
        handle.write(xyz.tobytes())
    tmp.replace(path)
    return path


def read_positions(path: Path) -> dict | None:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) < HEADER_STRUCT.size or raw[:8] != MAGIC:
        return None
    magic, generation, count, flags, radius = HEADER_STRUCT.unpack_from(raw, 0)
    need = HEADER_STRUCT.size + int(count) * 12
    if len(raw) < need:
        return None
    xyz = np.frombuffer(raw, dtype=np.float32, offset=HEADER_STRUCT.size, count=int(count) * 3).reshape(int(count), 3)
    return {
        "generation": int(generation),
        "count": int(count),
        "flags": int(flags),
        "radius": float(radius),
        "xyz": xyz.copy(),
        "mtime": path.stat().st_mtime,
    }


def dump_is_fresh(path: Path, max_age: float = FRESH_S) -> bool:
    path = Path(path)
    try:
        return (time.time() - path.stat().st_mtime) <= float(max_age)
    except OSError:
        return False
