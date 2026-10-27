#!/usr/bin/env python3
"""Bake a frozen occupancy dump into a Maelstrom kind=2 VDB.

The occupancy grid stays authoritative. This writes a world-axis-aligned
OpenVDB level set named "surface" plus a static boundary config. It does
not change the bone renderer, the haptic loop, or the prescribed stream.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluid.maelstrom_boundary import recommended_band
from fluid.occupancy import (
    MM_TO_SIM,
    VolumeFrame,
    identity_frame,
    load_occupancy_dump,
    volume_frame_from_adf,
)
from fluid.sdf_proxy import _sample_field

HERE = Path(__file__).resolve().parent
WRITER_SRC = HERE / "write_surface_vdb.cpp"
WRITER_BIN = HERE / "_build" / "write_surface_vdb"
VFX_INCLUDE = Path.home() / "opt" / "vfx" / "include"
VFX_LIB = Path.home() / "opt" / "vfx" / "lib" / "x86_64-linux-gnu"
DENSE_MAGIC = b"FIVRSF32"
DEFAULT_PARTICLE_RADIUS = 0.5 * MM_TO_SIM
KIND_STATIC = 2


@dataclass
class FrozenBone:
    """World-axis-aligned truncated SDF covering the occupancy AABB."""

    frame: VolumeFrame
    occupied: np.ndarray
    origin: np.ndarray
    spacing: float
    distance: np.ndarray
    band: float
    particle_radius: float
    occupied_count: int
    bake_ms: float

    @property
    def shape(self):
        return tuple(int(v) for v in self.distance.shape)

    @property
    def world_max(self) -> np.ndarray:
        return self.origin + self.spacing * (np.asarray(self.shape, dtype=np.float64) - 1.0)

    def sample_world(self, xyz) -> float:
        q = (np.asarray(xyz, dtype=np.float64) - self.origin) / self.spacing
        return float(_sample_field(self.distance, q))

    def write_vdb(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dense = path.with_suffix(".dense.bin")
        write_dense(dense, self.origin, self.spacing, self.distance)
        writer = ensure_writer()
        subprocess.run([str(writer), "write", str(dense), str(path)], check=True)
        dense.unlink(missing_ok=True)
        return path

    def write_config(self, path: Path, vdb_name: str = "bone_surface.vdb") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        domain = path.parent / "domain.obj"
        write_domain_obj(domain, self.origin, self.world_max, pad=8.0 * self.spacing)
        path.write_text(kind2_config(vdb_name, self.particle_radius) + "\n")
        return path

    def report(self) -> dict:
        return {
            "occupancy_shape": [int(v) for v in self.occupied.shape],
            "occupied": int(self.occupied_count),
            "vdb_shape": list(self.shape),
            "spacing": float(self.spacing),
            "origin": [float(v) for v in self.origin],
            "world_max": [float(v) for v in self.world_max],
            "band": float(self.band),
            "particle_radius": float(self.particle_radius),
            "kind": KIND_STATIC,
            "bake_ms": float(self.bake_ms),
            "inside_fraction": float(np.mean(self.distance < 0.0)),
        }


def volume_world_aabb(frame: VolumeFrame):
    corners = []
    for ix in (0.0, float(frame.counts[0])):
        for iy in (0.0, float(frame.counts[1])):
            for iz in (0.0, float(frame.counts[2])):
                corners.append(frame.index_to_world((ix, iy, iz)))
    pts = np.stack(corners, axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def occupancy_signed_distance(occupied: np.ndarray, frame: VolumeFrame, band: float) -> np.ndarray:
    """Positive outside, negative inside, distances in world units."""
    occupied = np.asarray(occupied, dtype=bool)
    spacing = frame.dimensions / frame.counts
    if not np.any(occupied):
        return np.full(occupied.shape, float(band), np.float32)
    if np.all(occupied):
        return np.full(occupied.shape, -float(band), np.float32)
    outside = ndimage.distance_transform_edt(~occupied, sampling=spacing)
    inside = ndimage.distance_transform_edt(occupied, sampling=spacing)
    return np.clip(outside - inside, -band, band).astype(np.float32)


def resample_world(field: np.ndarray, frame: VolumeFrame, origin, spacing: float, shape, band: float):
    nx, ny, nz = (int(v) for v in shape)
    grid = np.stack(np.meshgrid(
        origin[0] + np.arange(nx, dtype=np.float64) * spacing,
        origin[1] + np.arange(ny, dtype=np.float64) * spacing,
        origin[2] + np.arange(nz, dtype=np.float64) * spacing,
        indexing="ij",
    ), axis=-1)
    index = frame.world_to_index(grid.reshape(-1, 3))
    values = np.asarray(_sample_field(field, index), dtype=np.float32)
    counts = frame.counts
    outside = (
        (index[:, 0] < 0.0)
        | (index[:, 1] < 0.0)
        | (index[:, 2] < 0.0)
        | (index[:, 0] > counts[0] - 1.0)
        | (index[:, 1] > counts[1] - 1.0)
        | (index[:, 2] > counts[2] - 1.0)
    )
    values[outside] = float(band)
    return np.ascontiguousarray(values.reshape(nx, ny, nz))


def bake_frozen_bone(
    occupied: np.ndarray,
    frame: VolumeFrame | None = None,
    target_res: int = 256,
    particle_radius: float | None = None,
    band: float | None = None,
) -> FrozenBone:
    occupied = np.ascontiguousarray(np.asarray(occupied, dtype=bool))
    if occupied.ndim != 3:
        raise ValueError("occupied must be (x, y, z)")
    if frame is None:
        frame = identity_frame(occupied.shape)
    radius = DEFAULT_PARTICLE_RADIUS if particle_radius is None else float(particle_radius)
    lo, hi = volume_world_aabb(frame)
    extent = hi - lo
    longest = float(np.max(extent))
    if longest <= 0.0:
        raise ValueError("volume AABB is empty")
    spacing = longest / max(int(target_res) - 1, 1)
    if band is None:
        band_voxels = recommended_band(spacing, radius)
        band = float(band_voxels) * spacing
    pad = max(spacing, float(band))
    origin = lo - pad
    world_max = hi + pad
    shape = tuple(int(np.ceil((world_max[i] - origin[i]) / spacing)) + 1 for i in range(3))
    started = time.perf_counter()
    field = occupancy_signed_distance(occupied, frame, band)
    distance = resample_world(field, frame, origin, spacing, shape, band)
    bake_ms = (time.perf_counter() - started) * 1000.0
    return FrozenBone(
        frame=frame,
        occupied=occupied,
        origin=np.asarray(origin, dtype=np.float64),
        spacing=float(spacing),
        distance=distance,
        band=float(band),
        particle_radius=radius,
        occupied_count=int(np.count_nonzero(occupied)),
        bake_ms=bake_ms,
    )


def write_dense(path: Path, origin, spacing: float, distance: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = (int(v) for v in distance.shape)
    layout = np.ascontiguousarray(np.transpose(distance, (2, 1, 0)))
    with path.open("wb") as handle:
        handle.write(DENSE_MAGIC)
        handle.write(struct.pack("<iii", nx, ny, nz))
        handle.write(struct.pack("<ddd", float(origin[0]), float(origin[1]), float(origin[2])))
        handle.write(struct.pack("<d", float(spacing)))
        handle.write(np.ascontiguousarray(layout, dtype=np.float32).tobytes())
    return path


def inspect_vdb(path: Path) -> dict:
    writer = ensure_writer()
    out = subprocess.check_output([str(writer), "inspect", str(path)], text=True)
    parsed = {}
    for line in out.splitlines():
        key, _, rest = line.partition(" ")
        parts = rest.split()
        if not parts:
            parsed[key] = ""
        elif len(parts) == 1:
            token = parts[0]
            try:
                parsed[key] = float(token) if any(c in token for c in ".eE") else int(token)
            except ValueError:
                parsed[key] = token
        else:
            try:
                parsed[key] = [float(v) for v in parts]
            except ValueError:
                parsed[key] = parts
    parsed["raw"] = out
    return parsed


def write_domain_obj(path: Path, lo, hi, pad: float = 0.0) -> Path:
    lo = np.asarray(lo, dtype=np.float64) - pad
    hi = np.asarray(hi, dtype=np.float64) + pad
    corners = [
        (hi[0], lo[1], hi[2]),
        (lo[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]),
        (lo[0], hi[1], hi[2]),
        (lo[0], lo[1], lo[2]),
        (hi[0], lo[1], lo[2]),
        (lo[0], hi[1], lo[2]),
        (hi[0], hi[1], lo[2]),
    ]
    faces = (
        (1, 4, 2), (1, 3, 4), (5, 8, 6), (5, 7, 8),
        (7, 3, 8), (7, 4, 3), (6, 2, 5), (6, 1, 2),
        (6, 3, 1), (6, 8, 3), (2, 7, 5), (2, 4, 7),
    )
    lines = ["# frozen-bone domain", "g domain"]
    lines.extend(f"v {x} {y} {z}" for x, y, z in corners)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def kind2_config(vdb_name: str, particle_radius: float) -> str:
    payload = {
        "modules": {
            "pressure": "DFSPH",
            "volumeBoundary": True,
            "volumeInlets": False,
            "adaptive": False,
            "vorticity": "None",
            "drag": "None",
            "tension": "None",
        },
        "boundary_volumes": {
            "volume1": {
                "file": vdb_name,
                "kind": KIND_STATIC,
                "position": "0 0 0",
                "angle": "0 0 0",
                "density": 998,
            }
        },
        "particle_settings": {
            "radius": float(particle_radius),
            "density": 998,
        },
        "simulation_settings": {
            "boundaryObject": "domain.obj",
            "domainWalls": "x+-y+-z+-",
            "numptcls": 200000,
            "auto_boundary": True,
            "timestep_min": 0.0001,
            "timestep_max": 0.006,
            "neighborlimit": 120,
            "external_force": "0 0 0 0",
        },
    }
    return json.dumps(payload, indent=2)


def ensure_writer() -> Path:
    WRITER_BIN.parent.mkdir(parents=True, exist_ok=True)
    if WRITER_BIN.exists() and WRITER_BIN.stat().st_mtime >= WRITER_SRC.stat().st_mtime:
        return WRITER_BIN
    includes = [f"-I{VFX_INCLUDE}"]
    lib_dirs = [f"-L{VFX_LIB}", "-Wl,-rpath," + str(VFX_LIB)]
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        str(WRITER_SRC),
        "-o",
        str(WRITER_BIN),
        *includes,
        *lib_dirs,
        "-lopenvdb",
        "-ltbb",
        "-lblosc",
        "-lboost_iostreams",
        "-lImath-3_1",
    ]
    env = os.environ.copy()
    env["LIBRARY_PATH"] = str(VFX_LIB) + os.pathsep + env.get("LIBRARY_PATH", "")
    subprocess.run(cmd, check=True, env=env)
    return WRITER_BIN


def main():
    runtime = Path("/tmp/fivrs-rtx")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupancy", type=Path, default=runtime / "occupancy.bin")
    parser.add_argument("--anatomy-adf", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=runtime / "frozen_bone")
    parser.add_argument("--target-res", type=int, default=256)
    parser.add_argument("--particle-radius", type=float, default=DEFAULT_PARTICLE_RADIUS)
    args = parser.parse_args()

    occupied = load_occupancy_dump(args.occupancy)
    if args.anatomy_adf is None:
        frame = identity_frame(occupied.shape)
    else:
        frame = volume_frame_from_adf(args.anatomy_adf, occupied.shape)
    bone = bake_frozen_bone(
        occupied,
        frame,
        target_res=args.target_res,
        particle_radius=args.particle_radius,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vdb = bone.write_vdb(out / "bone_surface.vdb")
    bone.write_config(out / "config.json")
    report = bone.report()
    report["vdb"] = str(vdb)
    report["vdb_bytes"] = vdb.stat().st_size
    report["inspect"] = inspect_vdb(vdb)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
