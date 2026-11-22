#!/usr/bin/env python3
"""A CT of one or more cadaveric temporal bones -> bone + inner ear -> UR5 specimens.

Runs in the tbone-seg venv; the launcher's UR5 tab starts it under a hard memory cap and reads
its stdout (``@progress <0-1> <text>``, ``@result <json>``, ``@error <text>``).

  run <ct file> --work <folder>     split the scan into bones and segment each one
  add <work> --bone N --name NAME   turn one accepted bone into a UR5 specimen

Each bone is segmented twice. Blocks lie at arbitrary angles in the scanner, so the first pass
finds the inner ear, its shape gives the side and the anatomical pose, and the second pass runs
in that pose (TB12: inner ear Dice 0.70 -> 0.77 and 0.80 -> 0.84 against hand labels). Only bone
and inner ear are kept: the model's other labels are unreliable on dissected specimens.

The first pass runs without test-time mirroring: it only has to locate the inner ear.

Memory: class scores (about 100 bytes per voxel) stay on the GPU and only the winning label comes
back to host memory. A box too large for the free GPU memory is segmented in overlapping pieces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "utils"), str(REPO / "scripts" / "expert_drilling")]
import prepare_specimens as ps  # noqa: E402
import tbone_worker as tw  # noqa: E402

SPACING = tw.MODEL_SPACING_MM
MARGIN_MM = 3.0
DENSE_HU = 300
BONE_LABEL, LABYRINTH_LABEL = 1, 5
MIN_PIECE_FRACTION = 0.15   # of the largest dense piece
MAX_BONES = 4
MAX_SCAN_VOXELS = 400_000_000
MAX_OUTPUT_VOXELS_PER_SIDE = 384
GPU_BYTES_PER_VOXEL = 100
GPU_RESERVE_GIB = 1.5
OVERLAP_VOXELS = 136        # more than one 128-voxel model patch
MIN_EAR_VOXELS = 150
CONFIDENT_FIT_MM = 0.6
CONFIDENT_FIT_RATIO = 0.75


def say_result(payload):
    print("@result " + json.dumps(payload), flush=True)


# ---------------------------------------------------------------- split
def split_bones(ct):
    """The large dense pieces of the scan, each a separate bone, highest slices first."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi

    array = sitk.GetArrayFromImage(ct).transpose(2, 1, 0)
    dense = ndi.binary_opening(array > DENSE_HU, iterations=1)
    labels, _ = ndi.label(dense, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if not sizes.any():
        raise SystemExit(f"Nothing above {DENSE_HU} HU in this scan. Is it a CT in Hounsfield units?")
    keep = [int(k) for k in np.argsort(-sizes)[:MAX_BONES] if sizes[k] >= MIN_PIECE_FRACTION * sizes.max()]
    pieces = []
    for label in keep:
        where = np.argwhere(labels == label)
        sample = where[:: max(1, len(where) // 40000)].astype(float)
        points = np.array([ct.TransformContinuousIndexToPhysicalPoint(v.tolist()) for v in sample])
        pieces.append({"low": where.min(0).tolist(), "high": where.max(0).tolist(), "points": points})
    # Numbered along the axis the bones are stacked on, from the high index end.
    centers = np.array([(np.array(p["low"]) + np.array(p["high"])) / 2.0 for p in pieces])
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    return sorted(pieces, key=lambda p: -(p["low"][axis] + p["high"][axis]))


def box_image(ct, piece, direction):
    """A box just around this bone, axes along direction, at the model's spacing."""
    import SimpleITK as sitk

    direction = np.asarray(direction, float).reshape(3, 3)
    local = piece["points"] @ direction
    low, high = local.min(0) - MARGIN_MM, local.max(0) + MARGIN_MM
    count = np.ceil((high - low) / SPACING).astype(int)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([SPACING] * 3)
    resample.SetSize([int(v) for v in count])
    resample.SetOutputOrigin((direction @ low).tolist())
    resample.SetOutputDirection(direction.ravel().tolist())
    resample.SetDefaultPixelValue(-1000)
    resample.SetInterpolator(sitk.sitkBSpline)
    return sitk.Cast(sitk.Clamp(resample.Execute(ct), sitk.sitkFloat32, -1024, 4000), sitk.sitkInt16)


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, tbone_dir, work, max_gpu_gib=None):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("This needs the NVIDIA GPU, and CUDA is not available.")
        os.environ.setdefault("nnUNet_results", str(tbone_dir / "nnUNet_results"))
        os.environ.setdefault("nnUNet_raw", str(work))
        os.environ.setdefault("nnUNet_preprocessed", str(work))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        folder = tbone_dir / tw.MODEL_SUBDIR
        if not (folder / "fold_0" / "checkpoint_final.pth").is_file():
            raise SystemExit(f"Model not found in {folder}. Run scripts/segmentation/setup_tbone_seg.sh")
        tw.allow_checkpoint_numpy(torch)
        self.torch = torch
        self.tbone_dir = tbone_dir
        self.max_gpu_gib = max_gpu_gib
        self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                         perform_everything_on_device=True, device=torch.device("cuda"),
                                         verbose=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(str(folder), use_folds=(0,),
                                                            checkpoint_name="checkpoint_final.pth")
        network = self.predictor.network
        network = network._orig_mod if hasattr(network, "_orig_mod") else network
        network.load_state_dict(self.predictor.list_of_parameters[0])
        self.predictor.network.eval()
        self.stats = self.predictor.plans_manager.foreground_intensity_properties_per_channel["0"]
        self.pieces = 0

    def voxel_budget(self):
        torch = self.torch
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 2**30
        if self.max_gpu_gib is not None:
            free = min(free, float(self.max_gpu_gib))
        budget = int((free - GPU_RESERVE_GIB) * 2**30 / GPU_BYTES_PER_VOXEL)
        smallest = (2 * OVERLAP_VOXELS) ** 3
        if budget < smallest:
            raise SystemExit(f"Only {free:.1f} GiB of GPU memory is free. Close the simulator and try again.")
        return budget

    def _predict(self, array):
        torch = self.torch
        self.pieces += 1
        data = torch.from_numpy(array[None].copy())
        with torch.no_grad():
            logits = self.predictor.predict_sliding_window_return_logits(data)
            labels = torch.argmax(logits.to(torch.device("cuda")), 0).to(torch.uint8).cpu().numpy()
        del logits, data
        torch.cuda.empty_cache()
        return labels

    def _region(self, array, budget):
        if array.size <= budget:
            return self._predict(array)
        axis = int(np.argmax(array.shape))
        length = array.shape[axis]
        if length <= 2 * OVERLAP_VOXELS:
            raise SystemExit("The GPU has too little free memory for this bone.")
        middle = length // 2
        half = OVERLAP_VOXELS // 2
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, middle + half)
        second[axis] = slice(middle - half, length)
        left = self._region(array[tuple(first)], budget)
        right = self._region(array[tuple(second)], budget)
        out = np.empty(array.shape, np.uint8)
        keep_left = [slice(None)] * 3
        keep_right = [slice(None)] * 3
        put_left = [slice(None)] * 3
        put_right = [slice(None)] * 3
        keep_left[axis] = slice(0, middle)
        put_left[axis] = slice(0, middle)
        keep_right[axis] = slice(half, None)
        put_right[axis] = slice(middle, length)
        out[tuple(put_left)] = left[tuple(keep_left)]
        out[tuple(put_right)] = right[tuple(keep_right)]
        return out

    def segment(self, image, mirror=True):
        """mirror=False is about 8x less GPU work: enough to locate the inner ear."""
        import SimpleITK as sitk

        self.predictor.use_mirroring = mirror
        plans = self.predictor.plans_manager
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        array = np.clip(array, self.stats["percentile_00_5"], self.stats["percentile_99_5"])
        array = (array - self.stats["mean"]) / max(self.stats["std"], 1e-8)
        labels = self._region(array.transpose(plans.transpose_forward), self.voxel_budget())
        labels = tw.postprocess(labels.transpose(plans.transpose_backward), self.tbone_dir)
        result = sitk.GetImageFromArray(labels)
        result.CopyInformation(image)
        return result


# ---------------------------------------------------------------- pose
def references_in_lps():
    """Reference labyrinths in their own LPS frames (voxel x = j, y = rows - 1 - i, z = k)."""
    from scipy import ndimage as ndi

    out = {}
    for side, name in ps.REFERENCES.items():
        volume = ps.read_adf_volume(REPO / "ADF" / name)
        mask = volume["alpha"] & np.all(volume["rgb"] == np.array(ps.INNER_EAR_RGB, np.uint8), axis=-1)
        shape = np.array(mask.shape)
        step = volume["dimensions_au"] * ps.MM_PER_AU / shape
        where = np.argwhere(mask & ~ndi.binary_erosion(mask))
        points = np.column_stack([(shape[1] - 1 - where[:, 1]) * step[1], where[:, 0] * step[0], where[:, 2] * step[2]])
        out[side] = _thin(points)
    return out


def _thin(points):
    keep = np.unique(np.floor(points / ps.FIT_SPACING_MM).astype(np.int64), axis=0, return_index=True)[1]
    points = points[keep]
    if len(points) > 2500:
        points = points[np.random.default_rng(0).choice(len(points), 2500, replace=False)]
    return points


def largest_component(mask):
    from scipy import ndimage as ndi

    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def labyrinth_pose(labels_image, references):
    """(side, rotation scan -> reference, fit distances), or None if no inner ear was found."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi

    array = sitk.GetArrayFromImage(labels_image).transpose(2, 1, 0) == LABYRINTH_LABEL
    if array.sum() < MIN_EAR_VOXELS * (0.35 / SPACING) ** 3 * 0.1:
        return None
    array = largest_component(array)
    where = np.argwhere(array & ~ndi.binary_erosion(array))
    direction = np.array(labels_image.GetDirection()).reshape(3, 3)
    points = (direction @ (where * SPACING).T).T + np.array(labels_image.GetOrigin())
    points = _thin(points)
    fits = {side: ps.fit_labyrinth(points, target) for side, target in references.items()}
    side = min(fits, key=lambda s: fits[s][0])
    return side, fits[side][1], {s: round(float(f[0]), 3) for s, f in fits.items()}


# ---------------------------------------------------------------- output
def output_grid(ct, piece):
    """The scan's own grid when it is isotropic and small enough, else an isotropic one."""
    import SimpleITK as sitk

    spacing = np.array(ct.GetSpacing())
    extent = (np.array(piece["high"]) - np.array(piece["low"]) + 1) * spacing
    needed = float(extent.max()) / MAX_OUTPUT_VOXELS_PER_SIDE
    if np.ptp(spacing) < 1e-3 and spacing.mean() >= needed:
        return ct
    target = max(float(spacing.min()), needed)
    size = [int(np.ceil(n * s / target)) for n, s in zip(ct.GetSize(), spacing)]
    grid = sitk.Image(size, sitk.sitkUInt8)
    grid.SetSpacing([target] * 3)
    grid.SetOrigin(ct.GetOrigin())
    grid.SetDirection(ct.GetDirection())
    return grid


def clean(labels):
    """Bone + inner ear only: one inner ear, and the piece of solid that holds it."""
    ear = largest_component(labels == LABYRINTH_LABEL) if (labels == LABYRINTH_LABEL).any() else labels == LABYRINTH_LABEL
    bone = (labels == BONE_LABEL) & ~ear
    solid = largest_component(bone | ear) if (bone | ear).any() else bone
    return bone & solid, ear


def preview(ct_array, bone, ear, path):
    from PIL import Image

    where = np.argwhere(bone | ear)
    low, high = np.maximum(where.min(0) - 5, 0), np.minimum(where.max(0) + 6, bone.shape)
    center = np.round(np.argwhere(ear).mean(0)).astype(int) if ear.any() else (low + high) // 2
    cuts = ((slice(low[0], high[0]), slice(low[1], high[1]), center[2]),
            (slice(low[0], high[0]), center[1], slice(low[2], high[2])),
            (center[0], slice(low[1], high[1]), slice(low[2], high[2])))
    tiles = []
    for cut in cuts:
        base = (np.clip((ct_array[cut] + 1000.0) / 3400.0, 0, 1) * 255).astype(np.uint8)
        plain = np.stack([base] * 3, -1).astype(float)
        marked = plain.copy()
        marked[bone[cut]] = marked[bone[cut]] * 0.45 + np.array((232, 222, 190)) * 0.55
        marked[ear[cut]] = marked[ear[cut]] * 0.25 + np.array((235, 60, 45)) * 0.75
        tiles.append((plain, marked))
    scale = max(1, int(round(150 / max(max(t[0].shape[:2]) for t in tiles))))
    width = sum(t[0].shape[0] for t in tiles) * scale + 8 * (len(tiles) - 1)
    height = max(t[0].shape[1] for t in tiles) * scale
    sheet = Image.new("RGB", (width, height * 2 + 8), (12, 22, 32))
    x = 0
    for plain, marked in tiles:
        for row, array in enumerate((plain, marked)):
            image = Image.fromarray(np.transpose(array.astype(np.uint8), (1, 0, 2)))
            image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
            sheet.paste(image, (x, row * (height + 8)))
        x += plain.shape[0] * scale + 8
    sheet.save(path)


# ---------------------------------------------------------------- commands
def run(args):
    import SimpleITK as sitk

    source = pathlib.Path(args.ct).expanduser()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tw.progress(0.01, "Reading the scan")
    ct = sitk.ReadImage(str(source))
    if ct.GetDimension() != 3 or ct.GetNumberOfComponentsPerPixel() != 1:
        raise SystemExit("Expected a single 3D grayscale CT.")
    if ct.GetNumberOfPixels() > MAX_SCAN_VOXELS:
        raise SystemExit(f"This scan has {ct.GetNumberOfPixels():,} voxels; the limit is {MAX_SCAN_VOXELS:,}.")
    ct = sitk.Cast(ct, sitk.sitkFloat32)
    tw.progress(0.03, "Finding the bones")
    pieces = split_bones(ct)
    tw.progress(0.05, f"Found {len(pieces)} bone{'s' if len(pieces) != 1 else ''}. Loading the model")
    model = Model(pathlib.Path(args.tbone_dir), work, args.max_gpu_gib)
    references = references_in_lps()
    bones = []
    passes = 2 * len(pieces)
    done = 0

    def step(text):
        tw.progress(0.08 + 0.88 * done / passes, text)

    for number, piece in enumerate(pieces, start=1):
        step(f"Bone {number} of {len(pieces)}: finding the inner ear")
        first = model.segment(box_image(ct, piece, np.eye(3)), mirror=args.first_pass_mirroring)
        done += 1
        pose = labyrinth_pose(first, references)
        final, warning = first, ""
        side, fit = None, {}
        if pose is None:
            warning = "No inner ear found."
            done += 1
        else:
            side, rotation, fit = pose
            step(f"Bone {number} of {len(pieces)}: segmenting in the anatomical pose ({side} ear)")
            final = model.segment(box_image(ct, piece, rotation.T))
            done += 1
            # The side and the fit shown to the user come from the final, cleaner inner ear.
            checked = labyrinth_pose(final, references)
            if checked is not None:
                side, _, fit = checked
        grid = output_grid(ct, piece)
        labels = sitk.GetArrayFromImage(sitk.Resample(final, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0)).transpose(2, 1, 0)
        bone, ear = clean(labels)
        ct_on_grid = ct if grid is ct else sitk.Resample(ct, grid, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
        ct_array = sitk.GetArrayFromImage(ct_on_grid).transpose(2, 1, 0)
        out = np.zeros(bone.shape, np.uint8)
        out[bone], out[ear] = 1, 2
        image = sitk.GetImageFromArray(out.transpose(2, 1, 0))
        image.CopyInformation(ct_on_grid)
        labels_path = work / f"bone{number}_labels.nii.gz"
        sitk.WriteImage(image, str(labels_path), True)
        preview_path = work / f"bone{number}_preview.png"
        if bone.any():
            preview(ct_array, bone, ear, preview_path)
        voxel_ml = float(np.prod(ct_on_grid.GetSpacing())) / 1000.0
        if int(ear.sum()) < MIN_EAR_VOXELS and not warning:
            warning = "The inner ear is very small; check it."
        if fit and not warning:
            best, other = min(fit.values()), max(fit.values())
            if best > CONFIDENT_FIT_MM or best / max(other, 1e-6) > CONFIDENT_FIT_RATIO:
                warning = "The inner ear's shape fits poorly; check the side and the segmentation."
        bones.append({"number": number, "side": side, "fit_mm": fit, "warning": warning,
                      "bone_ml": round(int(bone.sum()) * voxel_ml, 2), "inner_ear_ml": round(int(ear.sum()) * voxel_ml, 3),
                      "voxel_mm": round(float(ct_on_grid.GetSpacing()[0]), 4), "labels": str(labels_path),
                      "preview": str(preview_path) if bone.any() else "", "usable": bool(bone.any() and ear.any())})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {"source": str(source), "source_file": source.name, "source_sha256": digest, "work": str(work),
              "pieces_segmented": model.pieces, "bones": bones}
    (work / "result.json").write_text(json.dumps(result, indent=1))
    tw.progress(1.0, "Done")
    say_result(result)


def add(args):
    import SimpleITK as sitk

    work = pathlib.Path(args.work)
    result = json.loads((work / "result.json").read_text())
    bone_info = next((b for b in result["bones"] if b["number"] == args.bone), None)
    if bone_info is None or not bone_info["usable"]:
        raise SystemExit(f"Bone {args.bone} has no usable segmentation.")
    name = args.name.strip()
    slug = "ur5_" + (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "specimen")
    known = ps.load_manifest()
    if any(s["name"].casefold() == name.casefold() for s in known):
        raise SystemExit(f"There is already a specimen called {name}.")
    if any(pathlib.Path(s["adf_path"]).stem.startswith(f"volume_{slug}_") for s in known):
        raise SystemExit(f"The name {name} is too close to an existing specimen's.")
    image = sitk.ReadImage(bone_info["labels"])
    labels = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    spacing = np.array(image.GetSpacing())
    directions = (np.array(image.GetDirection()).reshape(3, 3) @ np.diag(spacing)).T
    entry = ps.write_specimen(labels == 1, labels == 2, float(spacing.mean()), directions, np.array(image.GetOrigin()),
                              slug, name, f"{name}", ps.load_references(), result["source_file"], result["source_sha256"])
    import otic_capsule

    reader = None
    source = pathlib.Path(result["source"])
    if source.suffix == ".nrrd" and source.is_file():
        ct_values, lps_to_index = otic_capsule.read_ct(source)
        reader = lambda shape: otic_capsule.ct_on_voxels(ct_values, lps_to_index, entry["T_lps_voxel"], shape)
    otic_capsule.hollow(REPO / entry["adf_path"], reader)
    entry["segmentation"] = "automatic: tbone nnU-Net fold 0, bone + inner ear (ur5_ct_worker.py)"
    entry["source_bone"] = args.bone
    ps.add_to_manifest(entry)
    say_result(entry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("ct")
    one.add_argument("--work", required=True)
    one.add_argument("--tbone_dir", default=str(tw.DEFAULT_TBONE_DIR))
    one.add_argument("--max_gpu_gib", type=float, default=None, help="Use at most this much GPU memory")
    one.add_argument("--first_pass_mirroring", action="store_true", help="Slower first pass, for comparison")
    two = sub.add_parser("add")
    two.add_argument("work")
    two.add_argument("--bone", type=int, required=True)
    two.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    try:
        run(args) if args.command == "run" else add(args)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(f"@error {exc.code}", flush=True)
            return 1
    except Exception as exc:
        traceback.print_exc()
        text = "The GPU ran out of memory." if "out of memory" in str(exc).lower() else str(exc)
        print(f"@error {text}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
