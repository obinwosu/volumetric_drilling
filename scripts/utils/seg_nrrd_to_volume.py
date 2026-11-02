#!/usr/bin/env python3
"""Convert a 3D Slicer .seg.nrrd into an AMBF/FIVRS RGBA label volume.

Writes one RGBA PNG per slice (``plane00<i>.png``, the layout the old FIVRS
seg_nrrd_to_pngs.py produced), a volume ADF, a study GUI icon, and the
segment names for the RTX tuner in ADF/color/volume_labels.yaml.

Each voxel takes the color of exactly one segment. Where Slicer layers
overlap, the smaller segment wins, so ossicles stay visible inside a bone
print. Colors must be unique, because the tuner and the RTX view identify
a structure only by its RGB.

Example:
    python3 scripts/utils/seg_nrrd_to_volume.py \\
        -n ~/ambf_delta.seg.nrrd --name ambf_delta --laterality right \\
        --bone bd_print --eac "Outer ear Canal" --factor 2
"""

import argparse
import gzip
import pathlib
import re
import sys

import numpy as np
import yaml
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
LABELS_PATH = REPO / "ADF" / "color" / "volume_labels.yaml"
METERS_PER_AU = 0.049664
BONE_HEX = "fff9db"
EAC_HEX = "ffe1d6"


def read_nrrd(path):
    header = {}
    with open(path, "rb") as handle:
        magic = handle.readline()
        if not magic.startswith(b"NRRD"):
            raise SystemExit(f"{path} is not a NRRD file")
        while True:
            line = handle.readline()
            if line in (b"", b"\n", b"\r\n"):
                break
            text = line.decode("latin1").rstrip("\r\n")
            if text.startswith("#"):
                continue
            separator = ":=" if ":=" in text else ": "
            key, value = text.split(separator, 1)
            header[key.strip()] = value.strip()
        payload = handle.read()
    encoding = header.get("encoding", "raw")
    if encoding in ("gzip", "gz"):
        payload = gzip.decompress(payload)
    elif encoding != "raw":
        raise SystemExit(f"Unsupported NRRD encoding {encoding}")
    dtypes = {"unsigned char": np.uint8, "uchar": np.uint8, "uint8": np.uint8,
              "short": np.int16, "unsigned short": np.uint16, "int": np.int32}
    dtype = dtypes.get(header["type"])
    if dtype is None:
        raise SystemExit(f"Unsupported NRRD type {header['type']}")
    sizes = [int(value) for value in header["sizes"].split()]
    data = np.frombuffer(payload, dtype=dtype).reshape(sizes[::-1])
    return header, data


def spatial_layers(header, data):
    """(layers, x, y, z) with x the fastest NRRD axis, and voxel spacing in mm."""
    directions = re.findall(r"\(([^)]*)\)", header["space directions"])
    spacing = np.array([np.linalg.norm([float(v) for v in d.split(",")]) for d in directions])
    if data.ndim == 3:
        data = data[..., np.newaxis]
    return np.transpose(data, (3, 2, 1, 0)), spacing


def read_segments(header):
    segments = []
    for key in header:
        match = re.fullmatch(r"Segment(\d+)_ID", key)
        if not match:
            continue
        prefix = f"Segment{match.group(1)}_"
        rgb = [float(value) for value in header[prefix + "Color"].split()][:3]
        segments.append({
            "index": int(match.group(1)),
            "name": header.get(prefix + "Name", f"segment_{match.group(1)}"),
            "layer": int(header.get(prefix + "Layer", 0)),
            "label": int(header.get(prefix + "LabelValue", 1)),
            "rgb": [int(round(channel * 255)) for channel in rgb],
        })
    segments.sort(key=lambda item: item["index"])
    return segments


def hex_of(rgb):
    return "".join(f"{channel:02x}" for channel in rgb)


def clean_name(text):
    cleaned = (text or "").strip().lower().replace(" ", "_")
    return "".join(char for char in cleaned if char.isalnum() or char == "_")


def read_label_names():
    names = {}
    if LABELS_PATH.is_file():
        labels = (yaml.safe_load(LABELS_PATH.read_text()) or {}).get("labels") or {}
        names = {str(key).lower(): str(value) for key, value in labels.items()}
    return names


def assign_colors(segments, bone, eac, known):
    """Bone and EAC take the plugin and tuner colors. Every hex is unique."""
    taken = {}
    for segment in segments:
        if segment["name"] == bone:
            segment["rgb"], segment["alias"] = [255, 249, 219], "bone"
        elif segment["name"] == eac:
            segment["rgb"], segment["alias"] = [255, 225, 214], "eac"
        else:
            segment["alias"] = clean_name(segment["name"])
    for segment in segments:
        rgb = list(segment["rgb"])
        reserved = segment["alias"] in ("bone", "eac")
        step = 0
        while not reserved and (
            hex_of(rgb) in taken or (hex_of(rgb) in known and known[hex_of(rgb)] != segment["alias"])
        ):
            step += 1
            channel = step % 3
            rgb = list(segment["rgb"])
            delta = (step + 2) // 3
            rgb[channel] = rgb[channel] - delta if rgb[channel] - delta >= 0 else rgb[channel] + delta
        if hex_of(rgb) in taken:
            raise SystemExit(f"{segment['name']} and {taken[hex_of(rgb)]} share color #{hex_of(rgb)}")
        if rgb != segment["rgb"]:
            print(f"  {segment['name']}: #{hex_of(segment['rgb'])} is taken, using #{hex_of(rgb)}")
        segment["rgb"] = rgb
        segment["hex"] = hex_of(rgb)
        taken[segment["hex"]] = segment["name"]


def compose_labels(layers, segments):
    """One id per voxel (1-based segment order). Smaller segments paint last."""
    for segment in segments:
        segment["count"] = int(np.count_nonzero(layers[segment["layer"]] == segment["label"]))
    labels = np.zeros(layers.shape[1:], dtype=np.uint8)
    for rank, segment in enumerate(sorted(segments, key=lambda item: -item["count"]), start=1):
        segment["id"] = rank
        labels[layers[segment["layer"]] == segment["label"]] = rank
    return labels


def crop(labels, margin):
    occupied = np.argwhere(labels > 0)
    low = np.maximum(occupied.min(axis=0) - margin, 0)
    high = np.minimum(occupied.max(axis=0) + 1 + margin, labels.shape)
    return labels[low[0]:high[0], low[1]:high[1], low[2]:high[2]]


def downsample(labels, factor, bone_id):
    """Thin structures survive: a block keeps its smallest segment if any voxel has it.

    Bone and anything at least as large (ids up to bone_id) only fill a block
    they cover by half or more, so bulk surfaces do not swell.
    """
    if factor == 1:
        return labels
    shape = [size // factor * factor for size in labels.shape]
    trimmed = labels[: shape[0], : shape[1], : shape[2]]
    blocks = trimmed.reshape(shape[0] // factor, factor, shape[1] // factor, factor, shape[2] // factor, factor)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(shape[0] // factor, shape[1] // factor, shape[2] // factor, -1)
    fine = np.where(blocks > bone_id, blocks, 0).max(axis=-1)
    bulk = np.zeros(fine.shape, dtype=np.uint8)
    best = np.zeros(fine.shape, dtype=np.int32)
    for bulk_id in range(1, bone_id + 1):
        votes = np.count_nonzero(blocks == bulk_id, axis=-1)
        wins = (votes * 2 >= factor ** 3) & (votes > best)
        bulk[wins] = bulk_id
        best = np.maximum(best, votes)
    return np.where(fine > 0, fine, bulk).astype(np.uint8)


def write_slices(labels, segments, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("plane00*.png"):
        stale.unlink()
    palette = np.zeros((256, 4), dtype=np.uint8)
    for segment in segments:
        palette[segment["id"]] = [*segment["rgb"], 255]
    for index in range(labels.shape[2]):
        # Rows are NRRD x and columns NRRD y, as plt.imsave(data[:, :, i]) wrote them.
        Image.fromarray(palette[labels[:, :, index]], "RGBA").save(folder / f"plane00{index}.png")


def write_icon(labels, segments, path):
    palette = np.zeros((256, 3), dtype=np.uint8)
    for segment in segments:
        palette[segment["id"]] = segment["rgb"]
    first = np.argmax(labels > 0, axis=2)
    seen = np.take_along_axis(labels, first[..., np.newaxis], axis=2)[..., 0]
    shade = 1.0 - 0.6 * first / max(labels.shape[2] - 1, 1)
    image = (palette[seen] * shade[..., np.newaxis]).astype(np.uint8)
    image[seen == 0] = 0
    icon = Image.fromarray(image, "RGB")
    icon.thumbnail((256, 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    icon.save(path)


def write_adf(path, folder, count, dimensions, laterality, volume_name):
    rel = pathlib.Path("..") / folder.relative_to(REPO)
    text = f"""# AMBF Version: (0.1)
bodies: []
joints: []
volumes: [VOLUME {volume_name}]
high resolution path: ./meshes/high_res/
low resolution path: ./meshes/low_res/
ignore inter-collision: true
namespace: /ambf/env/

VOLUME {volume_name}:
  name: {volume_name}
  laterality: {laterality}
  location:
    position: {{x: -6.040592783505154, y: 0.0, z: 0.0}}  # placed 745mm away from camera given 1AU = 0.049664m
    orientation: {{r: 1.57079, p: 1.4, y: 0}}
  scale: 1.0
  # Physical size in AU (1 AU = 49.664 mm): x = image columns, y = image rows, z = slices.
  dimensions: {{x: {dimensions[0]:.6f}, y: {dimensions[1]:.6f}, z: {dimensions[2]:.6f}}}
  images:
    path: {rel.as_posix()}
    prefix: plane00
    format: png
    count: {count}
  shaders:
    path: ./shaders/dvr/
    vertex: shader.vs
    fragment: shader.fs
"""
    path.write_text(text)


def write_label_names(segments, source):
    known = read_label_names()
    additions = [s for s in segments if s["hex"] not in known]
    if not additions:
        return
    lines = [f"  # {source}"]
    for segment in additions:
        r, g, b = segment["rgb"]
        lines.append(f"  {segment['hex']}: {segment['alias']:<24} # {r},{g},{b}  {segment['name']}")
    text = LABELS_PATH.read_text().rstrip("\n") + "\n" + "\n".join(lines) + "\n"
    LABELS_PATH.write_text(text)
    print(f"Added {len(additions)} names to {LABELS_PATH.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--nrrd", required=True, type=lambda p: pathlib.Path(p).expanduser())
    parser.add_argument("--name", required=True, help="Volume id, used for folders and ADF names")
    parser.add_argument("--laterality", choices=("left", "right"), required=True)
    parser.add_argument("--bone", required=True, help="Segment name that is drillable bone")
    parser.add_argument("--eac", default="", help="Segment name of the external auditory canal")
    parser.add_argument("--factor", type=int, default=1, help="Integer downsampling factor")
    parser.add_argument("--margin", type=int, default=2, help="Empty voxels kept around the anatomy")
    args = parser.parse_args()

    print(f"Reading {args.nrrd}")
    header, data = read_nrrd(args.nrrd)
    layers, spacing = spatial_layers(header, data)
    segments = read_segments(header)
    names = {segment["name"] for segment in segments}
    for wanted in (args.bone, args.eac):
        if wanted and wanted not in names:
            raise SystemExit(f"No segment named {wanted!r}. Segments: {sorted(names)}")
    assign_colors(segments, args.bone, args.eac, read_label_names())
    labels = compose_labels(layers, segments)
    del layers, data
    bone_id = next(s["id"] for s in segments if s["name"] == args.bone)
    labels = downsample(crop(labels, args.margin), args.factor, bone_id)
    voxel_mm = spacing * args.factor

    tag = f"{args.name}_{args.factor * spacing.min():.3f}mm".replace(".", "p")
    folder = REPO / "resources" / "volumes" / tag
    adf = REPO / "ADF" / f"volume_{tag}.yaml"
    icon = REPO / "scripts" / "study_gui" / "assets" / f"{args.name}.png"
    rows, cols, count = labels.shape
    dimensions = (cols * voxel_mm[1], rows * voxel_mm[0], count * voxel_mm[2])
    dimensions = [mm / 1000.0 / METERS_PER_AU for mm in dimensions]

    print(f"Writing {count} slices of {cols}x{rows} at {voxel_mm[0]:.3f} mm to {folder.relative_to(REPO)}")
    write_slices(labels, segments, folder)
    volume_name = "mastoidectomy_volume_high" if max(labels.shape) > 384 else "mastoidectomy_volume_low"
    write_adf(adf, folder, count, dimensions, args.laterality, volume_name)
    write_icon(labels, segments, icon)
    write_label_names(segments, args.nrrd.name)

    present = np.bincount(labels.ravel(), minlength=256)
    print(f"{'segment':<26}{'hex':<9}{'tuner name':<26}voxels")
    for segment in segments:
        print(f"{segment['name']:<26}{segment['hex']:<9}{segment['alias']:<26}{present[segment['id']]}")
    print(f"ADF {adf.relative_to(REPO)}  dimensions {[round(d, 4) for d in dimensions]} AU")
    print(f"Icon {icon.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
