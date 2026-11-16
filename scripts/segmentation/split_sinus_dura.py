#!/usr/bin/env python3
"""Recolour an existing tbone-nnU-Net volume: merged "Sinus + Dura" -> sigmoid sinus and dura.

  ../tbone-seg-nnunetv2/.venv/bin/python scripts/segmentation/split_sinus_dura.py \\
      resources/volumes/left_dehiscence_ct_0p196mm resources/intermediate_volumes/segmentation/<run>

Volumes written before 2026-11-15 gave the model's Sigmoid_Sinus (13) and Dura (14) one colour,
6eb8d1. The run's segmentation.nii.gz (0.098 mm, z y x) still has them apart. Each simulator
voxel of that colour is one factor^3 block of it (tbone_worker.write_volume: crop to the occupied
box with a 2-voxel margin, then downsample), so the block's 13 / 14 count decides its colour.
Only those voxels change; every other voxel, and the alpha, stays as it is.

Not possible for volumes whose source already merged them (the JHU anatomies A-U label
"Sigmoid sinus + dura" as one structure; AMBF Delta's seg.nrrd has one Sinus_Dura segment).
"""
import argparse
import pathlib
import re
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MERGED = (110, 184, 209)


def stack_files(folder: pathlib.Path):
    return sorted(folder.glob("plane00*.png"), key=lambda p: int(re.findall(r"plane00(\d+)", p.name)[0]))


def split(folder: pathlib.Path, segmentation: pathlib.Path, factor: int = 2, dry_run: bool = False):
    import SimpleITK as sitk

    import tbone_worker

    sinus_rgb = np.array(tbone_worker.LABELS[13][1], np.uint8)
    dura_rgb = np.array(tbone_worker.LABELS[14][1], np.uint8)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(segmentation))).transpose(2, 1, 0)  # x y z = row col slice
    occupied = np.argwhere(labels > 0)
    low = np.maximum(occupied.min(axis=0) - 2, 0)  # seg_nrrd_to_volume.crop(margin=2)
    files = stack_files(folder)
    first = np.array(Image.open(files[0]))
    shape = (first.shape[0], first.shape[1], len(files))
    block = labels[low[0]:low[0] + shape[0] * factor, low[1]:low[1] + shape[1] * factor,
                   low[2]:low[2] + shape[2] * factor]
    if block.shape != tuple(s * factor for s in shape):
        raise SystemExit(f"{segmentation} does not cover {folder} ({block.shape} vs {shape} x {factor})")
    blocks = block.reshape(shape[0], factor, shape[1], factor, shape[2], factor)
    sinus = (blocks == 13).sum(axis=(1, 3, 5))
    dura = (blocks == 14).sum(axis=(1, 3, 5))
    totals = {"sinus": 0, "dura": 0, "unresolved": 0}
    for index, path in enumerate(files):
        image = np.array(Image.open(path).convert("RGBA"))
        merged = (image[..., 3] > 0) & np.all(image[..., :3] == MERGED, axis=-1)
        if not merged.any():
            continue
        s, d = sinus[:, :, index], dura[:, :, index]
        to_sinus = merged & (s > d)
        to_dura = merged & (d >= s) & (d > 0)
        totals["sinus"] += int(to_sinus.sum())
        totals["dura"] += int(to_dura.sum())
        totals["unresolved"] += int((merged & ~to_sinus & ~to_dura).sum())
        if dry_run:
            continue
        image[to_sinus, :3] = sinus_rgb
        image[to_dura, :3] = dura_rgb
        Image.fromarray(image, "RGBA").save(path)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("volume_folder", type=pathlib.Path)
    parser.add_argument("run_folder", type=pathlib.Path, help="folder with the run's segmentation.nii.gz")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    totals = split(args.volume_folder, args.run_folder / "segmentation.nii.gz", args.factor, args.dry_run)
    print(f"{args.volume_folder.name}: {totals['sinus']} voxels -> sigmoid sinus, {totals['dura']} -> dura, "
          f"{totals['unresolved']} left merged" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
