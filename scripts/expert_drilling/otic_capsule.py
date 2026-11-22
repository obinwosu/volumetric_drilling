#!/usr/bin/env python3
"""UR5 specimens: a hollow inner ear inside a colored otic capsule.

The inner ear segmentation stands for the membranous labyrinth, so in the simulator it is an empty
cavity: its voxels are cleared from the volume (no rendering, no haptic contact). The mask is kept
in ``otic_capsule.npz`` beside the slices, because recordings and the capsule color still need it.

The file also holds, per voxel, how far the bone is from the cavity wall, measured **through bone
only** so the color can never jump across air to another bony surface:

    through_bone   any solid bone
    dense_bone     only bone at least DENSE_FRACTION as dense on CT as the bone hugging the cavity,
                   so the color follows the capsule and stops at softer or pneumatized bone

    python3 scripts/expert_drilling/otic_capsule.py          # every specimen in specimens.yaml
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
import shutil
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import prepare_specimens as ps  # noqa: E402

SIDECAR = "otic_capsule.npz"
BACKUP_ROOT = REPO / "resources" / "intermediate_volumes" / "ur5_before_hollow"
RULES = ("through_bone", "dense_bone")
MAX_MM = 6.0
FAR_CODE = 65535
MM_PER_CODE = 0.01
DENSE_FRACTION = 0.8
HUG_MM = 1.0
EAR_RGB = np.array(ps.INNER_EAR_RGB, np.uint8)
CT_FILES = {"TB1": "TB12_CT.nrrd", "TB2": "TB12_CT.nrrd", "TB3": "TB345_CT.nrrd",
            "TB4": "TB345_CT.nrrd", "TB5": "TB345_CT.nrrd"}


def read_ct(path):
    """(array x, y, z as float32, LPS -> index matrix) of a gzip NRRD of doubles or shorts."""
    raw = pathlib.Path(path).read_bytes()
    cut = raw.find(b"\n\n")
    header = {}
    for line in raw[:cut].decode("latin1").splitlines():
        if ": " in line and not line.startswith("#"):
            key, value = line.split(": ", 1)
            header[key] = value
    kinds = {"double": "<f8", "float": "<f4", "short": "<i2", "signed short": "<i2", "unsigned short": "<u2"}
    if header.get("encoding") not in ("gzip", "gz") or header.get("type") not in kinds:
        raise ValueError(f"{path}: expected a gzip NRRD")
    sizes = [int(v) for v in header["sizes"].split()]
    data = np.frombuffer(gzip.decompress(raw[cut + 2:]), kinds[header["type"]]).reshape(sizes[::-1])
    directions = np.array([[float(v) for v in d.split(",")] for d in re.findall(r"\(([^)]*)\)", header["space directions"])])
    origin = np.array([float(v) for v in re.findall(r"\(([^)]*)\)", header["space origin"])[0].split(",")])
    index_to_lps = np.eye(4)
    index_to_lps[:3, :3] = directions.T
    index_to_lps[:3, 3] = origin
    return data.transpose(2, 1, 0).astype(np.float32), np.linalg.inv(index_to_lps)


def ct_on_voxels(ct, lps_to_index, t_lps_voxel, shape):
    """The CT value at each specimen voxel."""
    grid = np.indices(shape).reshape(3, -1)
    matrix = lps_to_index @ np.asarray(t_lps_voxel, float)
    index = matrix[:3, :3] @ grid + matrix[:3, 3:]
    index = np.rint(index).astype(np.intp)
    for axis in range(3):
        index[axis] = np.clip(index[axis], 0, ct.shape[axis] - 1)
    return ct[index[0], index[1], index[2]].reshape(shape)


def through_bone_distance(allowed, seeds, voxel_mm, max_mm=MAX_MM):
    """Shortest path in mm from the seed voxels, stepping only through allowed voxels.

    Steps go to face and edge neighbours; an edge step needs one of the two voxels it passes
    between to be allowed too, so a path cannot squeeze through a diagonal gap in the bone.
    """
    out = np.full(allowed.shape, np.inf, np.float32)
    seeds = seeds & allowed
    if not seeds.any():
        return out
    reach = int(np.ceil(max_mm / voxel_mm)) + 1
    region = ndi.binary_dilation(seeds, iterations=reach) & allowed
    ids = np.full(allowed.shape, -1, np.int64)
    count = int(region.sum())
    ids[region] = np.arange(count)
    rows, cols, weights = [], [], []
    steps = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    where = np.argwhere(region)
    shape = np.array(allowed.shape)
    for step in steps:
        delta = np.array(step)
        target = where + delta
        inside = np.all((target >= 0) & (target < shape), axis=1)
        a, b = where[inside], target[inside]
        good = region[b[:, 0], b[:, 1], b[:, 2]]
        a, b = a[good], b[good]
        if int(np.abs(delta).sum()) == 2:
            axes = np.flatnonzero(delta)
            first, second = a.copy(), a.copy()
            first[:, axes[0]] += delta[axes[0]]
            second[:, axes[1]] += delta[axes[1]]
            open_path = allowed[first[:, 0], first[:, 1], first[:, 2]] | allowed[second[:, 0], second[:, 1], second[:, 2]]
            a, b = a[open_path], b[open_path]
        rows.append(ids[a[:, 0], a[:, 1], a[:, 2]])
        cols.append(ids[b[:, 0], b[:, 1], b[:, 2]])
        weights.append(np.full(len(a), float(np.linalg.norm(delta)) * voxel_mm))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    # One extra node, joined to every seed at (almost) no cost, makes this a multi-source search.
    source = count
    seed_ids = ids[seeds & region]
    rows = np.concatenate([rows, np.full(len(seed_ids), source)])
    cols = np.concatenate([cols, seed_ids])
    weights = np.concatenate([weights, np.full(len(seed_ids), 1e-6)])
    graph = coo_matrix((weights, (rows, cols)), shape=(count + 1, count + 1)).tocsr()
    distance = dijkstra(graph, directed=False, indices=source, limit=max_mm)
    out[region] = distance[:count].astype(np.float32)
    return out


def face_neighbors(mask):
    return ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(3, 1))


def capsule_fields(bone, ear, voxel_mm, ct_values=None):
    """{rule: distance in mm through bone from the cavity wall}."""
    seeds = bone & face_neighbors(ear)
    fields = {"through_bone": through_bone_distance(bone, seeds, voxel_mm)}
    info = {"wall_voxels": int(seeds.sum())}
    if ct_values is not None:
        hug = bone & (fields["through_bone"] <= HUG_MM)
        reference = float(np.median(ct_values[hug])) if hug.any() else np.nan
        if np.isfinite(reference):
            threshold = DENSE_FRACTION * reference
            dense = bone & (ct_values >= threshold)
            fields["dense_bone"] = through_bone_distance(dense, seeds & dense, voxel_mm)
            info.update(capsule_hu=round(reference, 1), dense_threshold_hu=round(threshold, 1))
    return fields, info


def encode(field):
    codes = np.full(field.shape, FAR_CODE, np.uint16)
    finite = np.isfinite(field)
    codes[finite] = np.clip(np.rint(field[finite] / MM_PER_CODE), 0, FAR_CODE - 1).astype(np.uint16)
    return codes


def decode(codes):
    out = codes.astype(np.float32) * MM_PER_CODE
    out[codes == FAR_CODE] = np.inf
    return out


def slices_folder(adf):
    adf = pathlib.Path(adf)
    doc = yaml.safe_load(adf.read_text())
    return (adf.parent / doc[doc["volumes"][0]]["images"]["path"]).resolve()


def sidecar_path(adf):
    return slices_folder(adf) / SIDECAR


def load_sidecar(adf):
    """{"ear": bool array x y z, "fields": {rule: mm}} or None for volumes without one."""
    path = sidecar_path(adf)
    if not path.is_file():
        return None
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        ear = np.unpackbits(data["ear"])[: int(np.prod(shape))].reshape(shape).astype(bool)
        fields = {rule: decode(data[rule]) for rule in RULES if rule in data}
    return {"ear": ear, "fields": fields}


def hollow(adf, ct_values_for=None):
    """Clear the inner ear from the slices and write the sidecar. Safe to run again."""
    adf = pathlib.Path(adf)
    folder = slices_folder(adf)
    path = folder / SIDECAR
    volume = ps.read_adf_volume(adf)
    color_ear = volume["alpha"] & np.all(volume["rgb"] == EAR_RGB, axis=-1)
    if path.is_file():
        ear = load_sidecar(adf)["ear"]
    else:
        ear = color_ear
    if not ear.any():
        raise ValueError(f"{adf.name}: no inner ear")
    bone = volume["alpha"] & ~ear
    voxel_mm = float(np.mean(volume["dimensions_au"] * ps.MM_PER_AU / np.array(ear.shape)))
    ct_values = ct_values_for(ear.shape) if ct_values_for is not None else None
    fields, info = capsule_fields(bone, ear, voxel_mm, ct_values)
    arrays = {rule: encode(field) for rule, field in fields.items()}
    partial = path.with_name("otic_capsule_part.npz")
    np.savez_compressed(partial, shape=np.array(ear.shape), ear=np.packbits(ear), **arrays)
    partial.replace(path)
    if color_ear.any():
        names = sorted((p for p in folder.glob("plane00*.png")), key=lambda p: int(re.search(r"plane00(\d+)", p.name).group(1)))
        for name in names:
            image = np.array(Image.open(name).convert("RGBA"))
            hit = (image[..., 3] > 10) & np.all(image[..., :3] == EAR_RGB, axis=-1)
            if hit.any():
                image[hit] = 0
                Image.fromarray(image, "RGBA").save(name)
    info.update(inner_ear_voxels=int(ear.sum()), rules=list(fields))
    return info


def main():
    specimens = ps.load_manifest()
    cts = {}
    for specimen in specimens:
        adf = REPO / specimen["adf_path"]
        folder = slices_folder(adf)
        backup = BACKUP_ROOT / folder.name
        if not backup.exists():
            backup.mkdir(parents=True)
            for item in folder.glob("plane00*.png"):
                shutil.copy2(item, backup / item.name)
            shutil.copy2(adf, backup / adf.name)
        name = CT_FILES.get(specimen.get("bone_id", ""))
        ct_path = pathlib.Path.home() / "Downloads" / name if name else None
        reader = None
        if ct_path is not None and ct_path.is_file():
            if name not in cts:
                cts[name] = read_ct(ct_path)
            ct, lps_to_index = cts[name]
            reader = lambda shape, s=specimen: ct_on_voxels(ct, lps_to_index, s["T_lps_voxel"], shape)
        print(specimen["name"], json.dumps(hollow(adf, reader)), flush=True)


if __name__ == "__main__":
    main()
