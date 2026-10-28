"""GPU block meshes match the CPU midpoint surface, including shared faces."""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from surface import extract_block
from warp_mesh import WarpMesher


def _cpu_block(occupied, colors, block, smooth):
    mesh, _phases = extract_block(occupied, colors, block, smooth=smooth)
    return mesh.verts, mesh.tris, mesh.colors, mesh.index_normals


def _triangles(verts, tris):
    found = set()
    for tri in tris:
        found.add(tuple(tuple(np.round(verts[int(i)], 5)) for i in tri))
    return found


def _check(cpu, gpu, label):
    cpu_tris = _triangles(cpu[0], cpu[1])
    gpu_tris = _triangles(gpu[0], gpu[1])
    if cpu_tris != gpu_tris:
        raise SystemExit(f"{label}: triangle set {len(gpu_tris)} != {len(cpu_tris)}, only cpu {len(cpu_tris - gpu_tris)} only gpu {len(gpu_tris - cpu_tris)}")
    if not np.array_equal(np.sort(cpu[2], axis=0), np.sort(gpu[2], axis=0)):
        # Colors travel with vertices; compare per matched triangle instead.
        cpu_colors = {}
        for tri, row in zip(cpu[1], cpu[2][cpu[1][:, 0]]):
            pass
        mismatch = 0
        gpu_at = {tuple(np.round(v, 5)): c for v, c in zip(gpu[0], gpu[2])}
        for v, c in zip(cpu[0], cpu[2]):
            other = gpu_at[tuple(np.round(v, 5))]
            if not np.array_equal(other, c):
                mismatch += 1
        if mismatch:
            raise SystemExit(f"{label}: {mismatch} vertex colors differ")
    print(f"{label}: {len(cpu_tris)} triangles match", flush=True)


def main():
    occupied = np.zeros((96, 80, 70), np.uint8)
    occupied[4:90, 4:74, 4:40] = 1
    colors = np.zeros(occupied.shape + (3,), np.uint8)
    colors[..., 0] = 255
    colors[..., 1] = 249
    colors[..., 2] = 219
    colors[40:, :, :, 0] = 110
    colors[40:, :, :, 1] = 184
    colors[40:, :, :, 2] = 209
    mesher = WarpMesher()
    mesher.upload_volume(occupied, colors)
    mesher.warmup(occupied, colors)
    blocks = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 1)]
    for block in blocks:
        cpu = _cpu_block(occupied, colors, block, smooth=False)
        gpu, phases = mesher.extract_block(block, smooth=False, prepare_pose=None)
        _check(cpu, gpu, f"extract {block} {phases['extract_ms']:.2f} ms")
    _shared_face_matches(occupied, colors, mesher)
    for block in blocks:
        cpu = _cpu_block(occupied, colors, block, smooth=True)
        gpu, phases = mesher.extract_block(block, smooth=True, prepare_pose=None)
        delta = _vertex_delta(cpu[0], gpu[0])
        print(f"smooth {block}: max vertex delta {delta:.6f} voxels, overflow {mesher.smooth_overflows}", flush=True)
        if delta > 1.0e-3:
            raise SystemExit(f"smooth {block} moved too far from the CPU surface")
    gpu_many, many_phases = mesher.mesh_blocks(blocks, smooth=True, prepare_pose=None)
    for block, gpu in gpu_many:
        cpu = _cpu_block(occupied, colors, block, smooth=True)
        delta = _vertex_delta(cpu[0], gpu[0])
        print(f"batched {block}: max vertex delta {delta:.6f} extract {many_phases['extract_ms']:.2f} smooth {many_phases['smooth_ms']:.2f}", flush=True)
        if delta > 1.0e-3:
            raise SystemExit(f"batched {block} moved too far from the CPU surface")
    from surface import index_to_local, local_to_world

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pose = (np.array([10.0, 8.0, 7.0]), np.array([96.0, 80.0, 70.0]), rotation, np.array([1.0, 2.0, 3.0]))
    posed, _phases = mesher.mesh_blocks([(1, 0, 0)], smooth=False, prepare_pose=pose)
    cpu = _cpu_block(occupied, colors, (1, 0, 0), smooth=False)
    expected = local_to_world(index_to_local(cpu[0], pose[0], pose[1]), pose[2], pose[3]).astype(np.float32)
    expected_normals = (np.asarray(cpu[3], dtype=np.float64) @ pose[2].T).astype(np.float32)
    length = np.linalg.norm(expected_normals, axis=1, keepdims=True)
    expected_normals = expected_normals / np.maximum(length, 1e-8)
    got = posed[0][1][3]
    got_normals = posed[0][1][4]
    # posed is a list of (block, tuple)
    got = posed[0][1][3]
    got_normals = posed[0][1][4]
    world_delta = _vertex_delta(expected, got)
    normal_delta = _vertex_delta(expected_normals, got_normals)
    print(f"world delta {world_delta:.6f} normal delta {normal_delta:.6f}", flush=True)
    if world_delta > 1.0e-4 or normal_delta > 1.0e-3:
        raise SystemExit("GPU world positions or normals do not match the CPU prepare")
    print("warp mesh matches", flush=True)


def _face_pairs(raw, smoothed, face=64.0):
    """Smoothed positions of the raw vertices that sat on the shared plane."""
    if raw[3] is None or smoothed[0] is None or len(raw[0]) != len(smoothed[0]):
        return np.zeros((0, 3)), np.zeros((0, 3))
    select = np.abs(raw[0][:, 0] - face) < 1e-4
    return raw[0][select], smoothed[0][select]


def _shared_face_matches(occupied, colors, mesher):
    """A curved surface must leave the raw face plane, and both blocks must agree."""
    sphere = np.zeros_like(occupied)
    xs, ys, zs = np.ogrid[0 : sphere.shape[0], 0 : sphere.shape[1], 0 : sphere.shape[2]]
    sphere[((xs - 48) ** 2 + (ys - 40) ** 2 + (zs - 35) ** 2) <= 30 ** 2] = 1
    mesher.upload_volume(sphere, colors)
    left_raw = _cpu_block(sphere, colors, (1, 0, 0), smooth=False)
    right_raw = _cpu_block(sphere, colors, (2, 0, 0), smooth=False)
    left = _cpu_block(sphere, colors, (1, 0, 0), smooth=True)
    right = _cpu_block(sphere, colors, (2, 0, 0), smooth=True)
    raw_l, moved_l = _face_pairs(left_raw, left)
    raw_r, moved_r = _face_pairs(right_raw, right)
    right_at = {tuple(np.round(v, 5)): p for v, p in zip(raw_r, moved_r)}
    deltas = []
    drift = []
    for origin, moved in zip(raw_l, moved_l):
        other = right_at.get(tuple(np.round(origin, 5)))
        if other is None:
            continue
        deltas.append(float(np.linalg.norm(moved - other)))
        drift.append(float(abs(moved[0] - 64.0)))
    if len(deltas) < 8:
        raise SystemExit(f"shared face only matched {len(deltas)} vertices")
    worst = max(deltas)
    mean_drift = float(np.mean(drift))
    if worst > 1.0e-3:
        raise SystemExit(f"shared face disagrees by {worst:.6f} voxels")
    if mean_drift < 1.0e-3:
        raise SystemExit("shared face stayed on the raw marching-cubes plane")
    gpu_l, _ = mesher.extract_block((1, 0, 0), smooth=True, prepare_pose=None)
    gpu_r, _ = mesher.extract_block((2, 0, 0), smooth=True, prepare_pose=None)
    gpu_delta = max(_vertex_delta(left[0], gpu_l[0]), _vertex_delta(right[0], gpu_r[0]))
    if gpu_delta > 1.0e-3:
        raise SystemExit(f"GPU shared face is {gpu_delta:.6f} voxels from the CPU surface")
    print(
        f"shared face: {len(deltas)} vertices agree within {worst:.6f}, mean drift {mean_drift:.4f} voxels",
        flush=True,
    )
    mesher.upload_volume(occupied, colors)


def _vertex_delta(cpu, gpu):
    if len(cpu) != len(gpu):
        # Match by rounded pre-smooth identity is not available. Use nearest.
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return 1.0e9
        tree = cKDTree(cpu)
        dist, _ = tree.query(gpu, k=1)
        return float(np.max(dist)) if len(dist) else 0.0
    # Same compact order is not guaranteed. Nearest distance is the appearance check.
    from scipy.spatial import cKDTree

    tree = cKDTree(cpu)
    dist, _ = tree.query(gpu, k=1)
    back, _ = tree.query(cpu, k=1) if False else (None, None)
    reverse = cKDTree(gpu).query(cpu, k=1)[0]
    return float(max(np.max(dist), np.max(reverse)))


if __name__ == "__main__":
    main()
