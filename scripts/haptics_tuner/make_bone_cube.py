#!/usr/bin/env python3
"""Writes the plain bone cube the haptics tuner drills.

A 32 mm cube of one bone label at 0.25 mm voxels, with air around it. Sizes are in
the simulator's millimetres (1 mm = 0.01007 sim units), the unit the burrs use.
"""
import pathlib

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
MM_TO_SIM = 0.01007
VOXEL_MM = 0.25
BONE_RGBA = (255, 249, 219, 255)  # the plugin's m_boneColor
VOXELS = 136
MARGIN = 4
FOLDER = REPO / "resources" / "volumes" / "haptics_cube_0p250mm"
ADF = REPO / "ADF" / "volume_haptics_cube.yaml"

ADF_TEXT = """# AMBF Version: (0.1)
# Written by scripts/haptics_tuner/make_bone_cube.py
bodies: []
joints: []
volumes: [VOLUME mastoidectomy_volume_low]
high resolution path: ./meshes/high_res/
low resolution path: ./meshes/low_res/
ignore inter-collision: true
namespace: /ambf/env/

VOLUME mastoidectomy_volume_low:
  name: mastoidectomy_volume_low
  laterality: right
  location:
    position: {{x: -6.040592783505154, y: 0.0, z: 0.0}}
    orientation: {{r: 1.57079, p: 1.4, y: 0}}
  scale: 1.0
  dimensions: {{x: {size:.6f}, y: {size:.6f}, z: {size:.6f}}}
  images:
    path: ../resources/volumes/{folder}
    prefix: plane00
    format: png
    count: {count}
  shaders:
    path: ./shaders/dvr/
    vertex: shader.vs
    fragment: shader.fs
"""


def build(folder=FOLDER, adf=ADF, voxels=VOXELS, margin=MARGIN):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("plane00*.png"):
        old.unlink()
    empty = np.zeros((voxels, voxels, 4), dtype=np.uint8)
    solid = empty.copy()
    solid[margin:voxels - margin, margin:voxels - margin] = BONE_RGBA
    for index in range(voxels):
        inside = margin <= index < voxels - margin
        Image.fromarray(solid if inside else empty, "RGBA").save(folder / f"plane00{index}.png", compress_level=1)
    pathlib.Path(adf).write_text(ADF_TEXT.format(
        size=voxels * VOXEL_MM * MM_TO_SIM, folder=folder.name, count=voxels))
    return folder


def ready():
    return (ADF.is_file() and (FOLDER / f"plane00{VOXELS - 1}.png").is_file()
            and not (FOLDER / f"plane00{VOXELS}.png").exists())


if __name__ == "__main__":
    print(f"Wrote {build()}")
