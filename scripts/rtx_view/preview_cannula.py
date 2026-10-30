"""Side views of the four drills, the stainless cannula, and the jet to the burr."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fluid.cannula import CLIP_MM, MESH_MM, SIZES, load_obj, local_frame, mesh_path

OUT = Path("/tmp/fivrs_cannula_preview.png")


def _shaft_faces(size: int):
    path = mesh_path(size).parent / f"{size}mm.OBJ"
    verts = []
    faces = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                corners = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for index in range(1, len(corners) - 1):
                    faces.append((corners[0], corners[index], corners[index + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _paint(image, zbuf, verts_mm, tris, rgb, x0, y0, scale, x_max):
    """Orthographic view along Z. +X to the right, +Y up. Closer is larger Z."""
    height, width = zbuf.shape
    color = np.asarray(rgb, dtype=np.float64)
    keep = []
    for tri in tris:
        tri = verts_mm[tri]
        if tri[:, 0].min() > x_max:
            continue
        keep.append(tri)
    if not keep:
        return
    tris_mm = np.asarray(keep, dtype=np.float64)
    light = np.array((0.15, 0.45, 0.88), dtype=np.float64)
    light /= np.linalg.norm(light)
    for tri in tris_mm:
        edge_a = tri[1] - tri[0]
        edge_b = tri[2] - tri[0]
        normal = np.cross(edge_a, edge_b)
        length = float(np.linalg.norm(normal))
        if length < 1e-10:
            continue
        shade = abs(float(np.dot(normal / length, light)))
        shade = 0.38 + 0.62 * shade
        sx = x0 + tri[:, 0] * scale
        sy = y0 - tri[:, 1] * scale
        depth = tri[:, 2]
        min_x = max(int(np.floor(sx.min())), 0)
        max_x = min(int(np.ceil(sx.max())), width - 1)
        min_y = max(int(np.floor(sy.min())), 0)
        max_y = min(int(np.ceil(sy.max())), height - 1)
        if min_x >= max_x or min_y >= max_y:
            continue
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        denom = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
        if abs(denom) < 1e-8:
            continue
        w0 = ((sy[1] - sy[2]) * (grid_x - sx[2]) + (sx[2] - sx[1]) * (grid_y - sy[2])) / denom
        w1 = ((sy[2] - sy[0]) * (grid_x - sx[2]) + (sx[0] - sx[2]) * (grid_y - sy[2])) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(mask):
            continue
        zee = w0 * depth[0] + w1 * depth[1] + w2 * depth[2]
        closer = mask & (zee > zbuf[min_y : max_y + 1, min_x : max_x + 1])
        if not np.any(closer):
            continue
        zbuf[min_y : max_y + 1, min_x : max_x + 1][closer] = zee[closer]
        painted = (color * shade * 255.0).astype(np.uint8)
        image[min_y : max_y + 1, min_x : max_x + 1][closer] = painted


def _draw_line(draw, a, b, fill, width, x0, y0, scale):
    ax = x0 + a[0] * scale
    ay = y0 - a[1] * scale
    bx = x0 + b[0] * scale
    by = y0 - b[1] * scale
    draw.line((ax, ay, bx, by), fill=fill, width=width)


def render_one(size: int, width: int, height: int) -> Image.Image:
    shaft_pts, shaft_tris = _shaft_faces(size)
    cannula_pts, cannula_tris = load_obj(mesh_path(size))
    shaft_mm = shaft_pts / MESH_MM
    cannula_mm = cannula_pts / MESH_MM
    frame = local_frame(size)
    outlet = frame["outlet"] / MESH_MM
    emit = frame["emit"]
    radius = 0.5 * size
    clip = CLIP_MM[size]
    x_lo = -radius - 3.0
    x_hi = clip + 14.0
    y_lo, y_hi = -12.0, 14.0
    scale = min((width - 36) / (x_hi - x_lo), (height - 36) / (y_hi - y_lo))
    x0 = 18 - x_lo * scale
    y0 = 18 + y_hi * scale
    image = np.full((height, width, 3), 244, dtype=np.uint8)
    zbuf = np.full((height, width), -1e9, dtype=np.float64)
    _paint(image, zbuf, shaft_mm, shaft_tris[::2], (0.55, 0.57, 0.60), x0, y0, scale, x_hi + 2)
    _paint(image, zbuf, cannula_mm, cannula_tris, (0.86, 0.89, 0.92), x0, y0, scale, x_hi + 2)
    picture = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(picture)
    steps = 48
    previous = None
    for index in range(steps + 1):
        angle = 2.0 * np.pi * index / steps
        point = (radius * np.cos(angle), radius * np.sin(angle))
        if previous is not None:
            _draw_line(draw, previous, point, (40, 44, 48), 2, x0, y0, scale)
        previous = point
    reach = float(np.linalg.norm(frame["target"] - frame["outlet"]) / MESH_MM) + 6.0
    end = outlet + emit * reach
    _draw_line(draw, outlet, end, (32, 112, 186), 3, x0, y0, scale)
    ox = x0 + outlet[0] * scale
    oy = y0 - outlet[1] * scale
    draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill=(20, 90, 160))
    draw.text((12, 8), f"{size} mm    clip {clip:.0f} mm    nozzle 7.5 mm from the burr center", fill=(20, 24, 28))
    return picture


def main():
    panels = [render_one(size, 980, 280) for size in SIZES]
    sheet = Image.new("RGB", (980, 280 * len(panels) + 8 * (len(panels) - 1)), (255, 255, 255))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * (280 + 8)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
