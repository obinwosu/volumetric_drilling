"""Diamond burr mesh. Unit radius is the outermost abrasive grain.

Valleys are displaced inward, then the whole mesh is scaled so the highest
grain sits on the unit sphere. Scaling the mesh to a nominal radius therefore
makes the outside diameter that size, and the grit does not add to it.
"""

from __future__ import annotations

import pathlib

import numpy as np

MESH_PATH = pathlib.Path(__file__).resolve().parents[2] / "ADF" / "meshes" / "high_res" / "diamond_burr.obj"
# One step coarser than the first mesh. Subdivision 5 was 20k facets in the
# scope foreground and cost about 10 fps in the path tracer. Subdivision 4
# still resolves the grit and matches the cutting burr's triangle count.
SUBDIV = 4
AMPLITUDE = 0.085
FREQUENCY = 9.0


def _icosphere(subdiv: int):
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int32,
    )
    for _ in range(subdiv):
        verts, faces = _subdivide(verts, faces)
    return verts, faces


def _subdivide(verts, faces):
    midpoint = {}
    extra = []

    def mid(a, b):
        key = (int(a), int(b)) if a < b else (int(b), int(a))
        slot = midpoint.get(key)
        if slot is None:
            point = verts[a] + verts[b]
            point /= np.linalg.norm(point)
            slot = len(verts) + len(extra)
            midpoint[key] = slot
            extra.append(point)
        return slot

    tris = []
    for a, b, c in faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        tris.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    points = np.vstack((verts, np.asarray(extra, dtype=np.float64))) if extra else verts
    return points, np.asarray(tris, dtype=np.int32)


def _hash3(ix, iy, iz):
    n = ix.astype(np.uint64) * np.uint64(374761393)
    n ^= iy.astype(np.uint64) * np.uint64(668265263)
    n ^= iz.astype(np.uint64) * np.uint64(1274126177)
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n ^= n >> np.uint64(16)
    return (n & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(0xFFFFFFFF)


def _value_noise(points):
    base = np.floor(points).astype(np.int64)
    frac = points - base
    fade = frac * frac * (3.0 - 2.0 * frac)
    corners = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                corners.append(_hash3(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz))
    c000, c100, c010, c110, c001, c101, c011, c111 = corners
    x00 = c000 * (1.0 - fade[:, 0]) + c100 * fade[:, 0]
    x10 = c010 * (1.0 - fade[:, 0]) + c110 * fade[:, 0]
    x01 = c001 * (1.0 - fade[:, 0]) + c101 * fade[:, 0]
    x11 = c011 * (1.0 - fade[:, 0]) + c111 * fade[:, 0]
    y0 = x00 * (1.0 - fade[:, 1]) + x10 * fade[:, 1]
    y1 = x01 * (1.0 - fade[:, 1]) + x11 * fade[:, 1]
    return y0 * (1.0 - fade[:, 2]) + y1 * fade[:, 2]


def _grain(direction):
    coarse = _value_noise(direction * FREQUENCY)
    fine = _value_noise(direction * (FREQUENCY * 2.13) + 3.7)
    return np.clip(0.75 * coarse + 0.25 * fine, 0.0, 1.0)


def mesh(subdiv: int = SUBDIV, amplitude: float = AMPLITUDE):
    """Positions, triangles, and face normals. Max radius is 1."""
    points, faces = _icosphere(subdiv)
    direction = points / np.linalg.norm(points, axis=1, keepdims=True)
    radii = 1.0 - float(amplitude) * _grain(direction)
    radii *= 1.0 / float(np.max(radii))
    points = direction * radii[:, None]
    edge_a = points[faces[:, 1]] - points[faces[:, 0]]
    edge_b = points[faces[:, 2]] - points[faces[:, 0]]
    normal = np.cross(edge_a, edge_b)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal /= np.maximum(length, 1e-12)
    outward = np.sum(normal * points[faces[:, 0]], axis=1) < 0.0
    if np.any(outward):
        faces = faces.copy()
        faces[outward] = faces[outward][:, [0, 2, 1]]
        normal[outward] *= -1.0
    return (
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(normal, dtype=np.float32),
    )


def write_obj(path: pathlib.Path = MESH_PATH, subdiv: int = SUBDIV, amplitude: float = AMPLITUDE) -> None:
    points, faces, normals = mesh(subdiv, amplitude)
    lines = [
        "# Diamond burr. Radius 1 is the outermost grain. Grooves are displaced inward.",
        f"# subdiv {subdiv} amplitude {amplitude:g} frequency {FREQUENCY:g}",
    ]
    for point in points:
        lines.append(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
    for normal in normals:
        lines.append(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
    for index, (a, b, c) in enumerate(faces, start=1):
        lines.append(f"f {a + 1}//{index} {b + 1}//{index} {c + 1}//{index}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_obj()
    points, faces, _normals = mesh()
    radius = np.linalg.norm(points, axis=1)
    print(
        f"wrote {MESH_PATH}  {len(faces)} tris  "
        f"radius {radius.min():.4f}..{radius.max():.4f}"
    )
