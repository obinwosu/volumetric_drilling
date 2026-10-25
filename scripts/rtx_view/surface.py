"""Bone surface from the FIVRS occupancy grid.

The grid is the same voxel index space AMBF uses for contact. A vertex at
index (i, j, k) is the local point idx * dimensions / count - dimensions / 2.
Binary marching cubes puts each vertex on the midpoint of an edge between an
occupied voxel and an empty one, so neighboring cells share that point and a
regional rebuild does not open a crack. A clamped smooth then takes the stair
steps off without moving a vertex by more than ``max_step`` voxel widths.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from mc_tables import TRI_TABLE

CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int32,
)
EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)
_TRI = np.asarray(TRI_TABLE, dtype=np.int16)
# Edge keys and cell ids are packed into unsigned fields. A one-voxel empty
# border sits at index -1, so the stored values are shifted up by these biases.
KEY_BIAS = 32
CELL_BIAS = 4
_CELL_MASK = (1 << 21) - 1


def _edge_colors(occupied: np.ndarray, colors: np.ndarray | None, keys: np.ndarray, origin) -> np.ndarray:
    """RGB of the occupied endpoint of each edge. Empty voxels are not a label."""
    ox0, oy0, oz0 = (int(origin[0]), int(origin[1]), int(origin[2]))
    axis = ((keys >> 48) & 3).astype(np.int32)
    x = ((keys >> 32) & 0xFFFF).astype(np.int32) - KEY_BIAS - ox0
    y = ((keys >> 16) & 0xFFFF).astype(np.int32) - KEY_BIAS - oy0
    z = (keys & 0xFFFF).astype(np.int32) - KEY_BIAS - oz0
    x2 = x.copy()
    y2 = y.copy()
    z2 = z.copy()
    x2[axis == 0] += 1
    y2[axis == 1] += 1
    z2[axis == 2] += 1
    inside = occupied[x, y, z].astype(bool)
    if colors is None:
        out = np.empty((len(keys), 3), np.uint8)
        out[:] = (248, 248, 216)
        return out
    return np.ascontiguousarray(np.where(inside[:, None], colors[x, y, z], colors[x2, y2, z2]), dtype=np.uint8)


def marching_cubes(occupied: np.ndarray, colors: np.ndarray | None = None, origin=(0, 0, 0)):
    """Return vertices, triangles, edge keys, cell ids, and a label color per vertex.

    ``occupied`` is boolean, indexed (x, y, z), True inside tissue. ``colors``
    is uint8 RGB in the same index space. ``origin`` is the index of voxel
    (0, 0, 0) of this block inside the full volume, so a regional extract
    shares edge keys with the mesh built from the whole volume. Triangles are
    wound so the normal points out of the tissue.
    """
    occ = np.asarray(occupied, dtype=np.uint8)
    nx, ny, nz = occ.shape
    empty_v = np.zeros((0, 3), np.float64)
    empty_t = np.zeros((0, 3), np.int32)
    empty_k = np.zeros((0,), np.int64)
    empty_c = np.zeros((0,), np.int64)
    empty_rgb = np.zeros((0, 3), np.uint8)
    if min(nx, ny, nz) < 2:
        return empty_v, empty_t, empty_k, empty_c, empty_rgb

    bits = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.uint16)
    for bit, (dx, dy, dz) in enumerate(CORNERS):
        bits |= occ[dx : dx + nx - 1, dy : dy + ny - 1, dz : dz + nz - 1].astype(np.uint16) << bit
    active = (bits != 0) & (bits != 255)
    if not np.any(active):
        return empty_v, empty_t, empty_k, empty_c, empty_rgb

    ii, jj, kk = np.nonzero(active)
    case = bits[ii, jj, kk].astype(np.int32)
    rows = _TRI[case]
    used = rows >= 0
    counts = used.sum(axis=1)
    flat_edge = rows[used].astype(np.int32)
    cell = np.repeat(np.arange(len(ii), dtype=np.int32), counts)

    corner_a = EDGES[flat_edge, 0]
    corner_b = EDGES[flat_edge, 1]
    origin_a = CORNERS[corner_a]
    origin_b = CORNERS[corner_b]
    edge_origin = np.minimum(origin_a, origin_b)
    axis = np.argmax(np.abs(origin_b - origin_a), axis=1).astype(np.int64)
    ox0, oy0, oz0 = (int(origin[0]), int(origin[1]), int(origin[2]))
    ox = ii[cell].astype(np.int64) + edge_origin[:, 0] + ox0
    oy = jj[cell].astype(np.int64) + edge_origin[:, 1] + oy0
    oz = kk[cell].astype(np.int64) + edge_origin[:, 2] + oz0
    key = (axis << 48) | ((ox + KEY_BIAS) << 32) | ((oy + KEY_BIAS) << 16) | (oz + KEY_BIAS)
    uniq, inverse = np.unique(key, return_inverse=True)
    u_axis = (uniq >> 48) & 3
    u_x = ((uniq >> 32) & 0xFFFF).astype(np.float64) - KEY_BIAS
    u_y = ((uniq >> 16) & 0xFFFF).astype(np.float64) - KEY_BIAS
    u_z = (uniq & 0xFFFF).astype(np.float64) - KEY_BIAS
    verts = np.stack([u_x, u_y, u_z], axis=1)
    verts[np.arange(len(verts)), u_axis.astype(np.int32)] += 0.5
    # Table winding points into the bone. Swap so the displayed normal faces out.
    tris = inverse.reshape(-1, 3).astype(np.int32)
    tris[:, [1, 2]] = tris[:, [2, 1]]
    packed_cell = (
        (ii.astype(np.int64) + ox0 + CELL_BIAS)
        | ((jj.astype(np.int64) + oy0 + CELL_BIAS) << 21)
        | ((kk.astype(np.int64) + oz0 + CELL_BIAS) << 42)
    )
    tri_cell = np.repeat(packed_cell, counts // 3)
    return verts, tris, uniq, tri_cell, _edge_colors(occ, colors, uniq, origin)


def smooth_surface(verts: np.ndarray, tris: np.ndarray, iterations: int = 4, lam: float = 0.5, max_step: float = 0.4):
    """Pull each vertex toward its neighbors, but never by more than ``max_step``."""
    if len(verts) == 0 or iterations <= 0:
        return verts
    n = len(verts)
    a = tris[:, [0, 1, 2]].reshape(-1)
    b = tris[:, [1, 2, 0]].reshape(-1)
    weights = np.ones(len(a), dtype=np.float64)
    matrix = sparse.coo_matrix((weights, (a, b)), shape=(n, n)).tocsr()
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    degree[degree == 0.0] = 1.0
    out = np.array(verts, dtype=np.float64, copy=True)
    for _ in range(iterations):
        mean = matrix.dot(out) / degree[:, None]
        delta = (mean - out) * lam
        length = np.linalg.norm(delta, axis=1)
        scale = np.ones(n, dtype=np.float64)
        over = length > max_step
        scale[over] = max_step / length[over]
        out = out + delta * scale[:, None]
    return out


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    if len(tris) == 0:
        return np.zeros_like(verts)
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(verts)
    for corner in range(3):
        np.add.at(normals, tris[:, corner], face)
    length = np.linalg.norm(normals, axis=1)
    length[length == 0.0] = 1.0
    return normals / length[:, None]


def cell_corners(tri_cell: np.ndarray):
    """Minimum corner of each triangle's cell, in volume index space.

    The low face of the volume is cell -1 and the high face is cell ``count - 1``,
    where the neighbor outside the scan is empty.
    """
    cell = np.asarray(tri_cell)
    ci = (cell & _CELL_MASK).astype(np.int64) - CELL_BIAS
    cj = ((cell >> 21) & _CELL_MASK).astype(np.int64) - CELL_BIAS
    ck = ((cell >> 42) & _CELL_MASK).astype(np.int64) - CELL_BIAS
    return ci, cj, ck


def _border_slab(occupied: np.ndarray, colors: np.ndarray | None, i0, i1, j0, j1, k0, k1):
    """Voxels ``i0..i1`` inclusive. An index outside the scan is empty.

    Marching cubes only emits a face between an occupied voxel and an empty
    one. The scan is solid up to its crop, so without this empty border the
    outer face is missing and the corner of the volume is an open cut.
    """
    nx, ny, nz = occupied.shape
    slab = np.zeros((i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1), np.uint8)
    rgb = None if colors is None else np.zeros(slab.shape + (3,), np.uint8)
    ia0, ia1 = max(0, i0), min(nx, i1 + 1)
    ib0, ib1 = max(0, j0), min(ny, j1 + 1)
    ic0, ic1 = max(0, k0), min(nz, k1 + 1)
    if ia0 < ia1 and ib0 < ib1 and ic0 < ic1:
        slab[ia0 - i0 : ia1 - i0, ib0 - j0 : ib1 - j0, ic0 - k0 : ic1 - k0] = occupied[ia0:ia1, ib0:ib1, ic0:ic1]
        if rgb is not None:
            rgb[ia0 - i0 : ia1 - i0, ib0 - j0 : ib1 - j0, ic0 - k0 : ic1 - k0] = colors[ia0:ia1, ib0:ib1, ic0:ic1]
    return slab, rgb


def index_to_local(index_xyz: np.ndarray, dimensions, counts) -> np.ndarray:
    """AMBF afVolume::voxelIndexToLocalPos, for continuous index coordinates."""
    dim = np.asarray(dimensions, dtype=np.float64)
    count = np.asarray(counts, dtype=np.float64)
    return index_xyz * (dim / count) - dim * 0.5


def local_to_world(local_xyz: np.ndarray, rotation: np.ndarray, translation) -> np.ndarray:
    """Same placement as DrillManager::volumeLocalToWorld.

    ``rotation`` is the volume pose matrix. The simulator places a column
    vector as ``translation + rotation * local``. These coordinates are rows,
    so the rotation is transposed.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    return local_xyz @ rotation.T + np.asarray(translation, dtype=np.float64)


class BoneMesh:
    """One surface for the whole volume. Drilling replaces only the touched cells.

    Edge keys are shared, so the border of a rebuilt patch uses the vertices
    already in the mesh. Those border vertices stay where the last smooth left
    them. New vertices inside the patch are smoothed toward that border.
    """

    def __init__(self, occupied: np.ndarray, colors: np.ndarray | None = None):
        self.shape = tuple(int(v) for v in occupied.shape)
        self._edge = {}
        nx, ny, nz = self.shape
        slab, slab_colors = _border_slab(occupied, colors, -1, nx, -1, ny, -1, nz)
        verts, tris, keys, cells, vert_colors = marching_cubes(slab, slab_colors, origin=(-1, -1, -1))
        self.verts = verts
        self.tris = tris
        self.tri_cell = cells
        self.colors = vert_colors
        for index, key in enumerate(keys.tolist()):
            self._edge[int(key)] = index
        self.verts = smooth_surface(self.verts, self.tris)

    def replace_cells(self, occupied: np.ndarray, i0, i1, j0, j1, k0, k1, colors: np.ndarray | None = None):
        """Rebuild every cell whose corner voxel lies in the half-open box."""
        nx, ny, nz = self.shape
        # Cell -1 is the low crop face and cell count-1 is the high crop face.
        i0 = max(-1, min(nx - 1, int(i0)))
        j0 = max(-1, min(ny - 1, int(j0)))
        k0 = max(-1, min(nz - 1, int(k0)))
        i1 = max(i0 + 1, min(nx, int(i1)))
        j1 = max(j0 + 1, min(ny, int(j1)))
        k1 = max(k0 + 1, min(nz, int(k1)))
        ci, cj, ck = cell_corners(self.tri_cell)
        inside = (ci >= i0) & (ci < i1) & (cj >= j0) & (cj < j1) & (ck >= k0) & (ck < k1)
        keep = ~inside
        kept_tris = self.tris[keep]
        kept_cells = self.tri_cell[keep]
        slab, slab_colors = _border_slab(occupied, colors, i0, i1, j0, j1, k0, k1)
        new_verts, new_tris, new_keys, new_cells, new_colors = marching_cubes(
            slab, slab_colors, origin=(i0, j0, k0)
        )
        if len(new_tris) == 0 and len(kept_tris) == 0:
            self.verts = np.zeros((0, 3), np.float64)
            self.tris = np.zeros((0, 3), np.int32)
            self.tri_cell = np.zeros((0,), np.int64)
            self.colors = np.zeros((0, 3), np.uint8)
            return
        remap = np.empty(len(new_keys), dtype=np.int32)
        extra = []
        extra_colors = []
        extra_ids = []
        base = len(self.verts)
        for local, key in enumerate(new_keys.tolist()):
            key = int(key)
            found = self._edge.get(key)
            if found is None:
                found = base + len(extra)
                self._edge[key] = found
                extra.append(new_verts[local])
                extra_colors.append(new_colors[local])
                extra_ids.append(found)
            remap[local] = found
        if extra:
            self.verts = np.vstack([self.verts, np.asarray(extra, dtype=np.float64)])
            self.colors = np.vstack([self.colors, np.asarray(extra_colors, dtype=np.uint8)])
        mapped = remap[new_tris] if len(new_tris) else np.zeros((0, 3), np.int32)
        locked = np.zeros(len(self.verts), dtype=bool)
        if len(kept_tris):
            locked[np.unique(kept_tris)] = True
        if len(kept_tris):
            self.tris = np.vstack([kept_tris, mapped])
            self.tri_cell = np.concatenate([kept_cells, new_cells])
        else:
            self.tris = mapped
            self.tri_cell = new_cells
        self._smooth_new(extra_ids, locked)

    def _smooth_new(self, free_ids, locked):
        if not free_ids or len(self.tris) == 0:
            return
        free = np.asarray(free_ids, dtype=np.int32)
        touch = np.isin(self.tris, free).any(axis=1)
        patch = self.tris[touch]
        if len(patch) == 0:
            return
        used = np.unique(patch)
        local_of = -np.ones(len(self.verts), dtype=np.int32)
        local_of[used] = np.arange(len(used))
        local_tris = local_of[patch]
        a = local_tris[:, [0, 1, 2]].reshape(-1)
        b = local_tris[:, [1, 2, 0]].reshape(-1)
        weights = np.ones(len(a), dtype=np.float64)
        matrix = sparse.coo_matrix((weights, (a, b)), shape=(len(used), len(used))).tocsr()
        degree = np.asarray(matrix.sum(axis=1)).ravel()
        degree[degree == 0.0] = 1.0
        current = self.verts[used].copy()
        pinned = locked[used]
        original = current.copy()
        for _ in range(4):
            mean = matrix.dot(current) / degree[:, None]
            delta = (mean - current) * 0.5
            length = np.linalg.norm(delta, axis=1)
            scale = np.ones(len(used), dtype=np.float64)
            over = length > 0.4
            scale[over] = 0.4 / np.maximum(length[over], 1e-8)
            moved = current + delta * scale[:, None]
            moved[pinned] = original[pinned]
            current = moved
        self.verts[used] = current
