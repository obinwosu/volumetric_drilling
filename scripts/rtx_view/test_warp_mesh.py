"""GPU block meshes match the CPU midpoint surface, including shared faces."""

from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from surface import (
    _border_slab,
    _compact_used,
    _owns_cell,
    block_voxel_bounds,
    cell_corners,
    marching_cubes,
    shared_face_pins,
    smooth_pinned,
)
from warp_mesh import STAGE, WarpMesher


def _cpu_block(occupied, colors, block, smooth):
    shape = occupied.shape
    bx, by, bz = block
    i0, i1 = block_voxel_bounds(bx, shape[0])
    j0, j1 = block_voxel_bounds(by, shape[1])
    k0, k1 = block_voxel_bounds(bz, shape[2])
    slab, slab_colors = _border_slab(occupied, colors, i0, i1, j0, j1, k0, k1)
    verts, tris, _keys, cells, vert_colors = marching_cubes(slab, slab_colors, origin=(i0, j0, k0))
    if len(tris):
        ci, cj, ck = cell_corners(cells)
        keep = _owns_cell(ci, bx, shape[0]) & _owns_cell(cj, by, shape[1]) & _owns_cell(ck, bz, shape[2])
        verts, tris, vert_colors = _compact_used(verts, tris[keep], vert_colors)
    if smooth and len(tris):
        verts = smooth_pinned(verts, tris, shared_face_pins(verts, shape))
    return verts, tris, vert_colors


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
    # Shared-face vertices are the raw midpoint on both blocks.
    left, _ = mesher.extract_block((1, 0, 0), smooth=STAGE >= 2, prepare_pose=None)
    right, _ = mesher.extract_block((2, 0, 0), smooth=STAGE >= 2, prepare_pose=None)
    face = 64.0
    left_face = {tuple(np.round(v, 5)) for v in left[0] if abs(v[0] - face) < 1e-5}
    right_face = {tuple(np.round(v, 5)) for v in right[0] if abs(v[0] - face) < 1e-5}
    if not left_face or left_face != right_face:
        raise SystemExit(f"shared face {len(left_face)} vs {len(right_face)} differ")
    print(f"shared face: {len(left_face)} vertices match", flush=True)
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
    from surface import index_to_local, local_to_world, vertex_normals

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pose = (np.array([10.0, 8.0, 7.0]), np.array([96.0, 80.0, 70.0]), rotation, np.array([1.0, 2.0, 3.0]))
    posed, _phases = mesher.mesh_blocks([(1, 0, 0)], smooth=False, prepare_pose=pose)
    cpu = _cpu_block(occupied, colors, (1, 0, 0), smooth=False)
    expected = local_to_world(index_to_local(cpu[0], pose[0], pose[1]), pose[2], pose[3]).astype(np.float32)
    expected_normals = vertex_normals(expected.astype(np.float64), cpu[1]).astype(np.float32)
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
