#!/usr/bin/env python3
"""CT -> temporal bone segmentation -> FIVRS volume, for the launcher's "Add a volume" tab.

Runs in the tbone-seg-nnunetv2 venv (torch, nnunetv2, SimpleITK), not in the
launcher's Python. The launcher starts it with QProcess and reads its stdout:
lines starting with "@" are machine-readable (``@progress <0-1> <text>``,
``@result <json>``, ``@error <text>``); everything else is log text.

Model: andy-s-ding/tbone-seg-nnunetv2, Dataset101_TemporalBone, 3d_fullres
nnUNetTrainer, 5 cross-validation folds. Its training scans were ~50 mm cubes
around one temporal bone at ~0.098 mm (dataset_fingerprint.json), so the input
is cropped to a box around the chosen ear and resampled to the model spacing
before inference (what the repo's nifti_resampler.py does). Feeding a whole
head at 0.1 mm would be ~2000^3 voxels.

Commands:
  scan <dicom_dir>                     list the DICOM series below a folder
  load <dicom_dir> <series_uid> <work> read one series, reorient to LPS, save
                                       work/ct.nii.gz and a small preview
  run <work> --center X Y Z --size MM --folds 0 [1 ...] --name N --laterality L
                                       crop, resample, segment, write the volume
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "utils"))

DATASET = "Dataset101_TemporalBone"
MODEL_SUBDIR = f"nnUNet_results/{DATASET}/nnUNetTrainer__nnUNetPlans__3d_fullres"
DEFAULT_TBONE_DIR = REPO.parent / "tbone-seg-nnunetv2"
MODEL_SPACING_MM = 0.098  # median training spacing (plans.json 3d_fullres)
DEFAULT_CROP_MM = 50.0  # training scans: 48-53 mm per side
PREVIEW_MAX = 320  # preview voxels on the longest side
METERS_PER_AU = 0.049664

# Model label -> (name, color). Colors are tbone_colortable.csv as the JHU
# anatomies A-I store them (channels truncated, e.g. malleus e900ff), so
# materials in the tuner already match.
# Sigmoid sinus and dura share one color there ("Sinus + Dura").
LABELS = {
    1: ("Bone", (255, 249, 219)),
    2: ("Malleus", (233, 0, 255)),
    3: ("Incus", (0, 255, 149)),
    4: ("Stapes", (63, 0, 255)),
    5: ("Bony Labyrinth", (91, 123, 91)),
    6: ("IAC", (244, 142, 52)),
    7: ("Superior Vestibular Nerve", (255, 191, 135)),
    8: ("Inferior Vestibular Nerve", (121, 70, 24)),
    9: ("Cochlear Nerve", (219, 244, 52)),
    10: ("Facial Nerve", (244, 214, 49)),
    11: ("Chorda Tympani", (151, 131, 29)),
    12: ("ICA", (216, 100, 79)),
    13: ("Sinus + Dura", (110, 184, 209)),
    14: ("Sinus + Dura", (110, 184, 209)),
    15: ("Vestibular Aqueduct", (91, 98, 123)),
    16: ("Mandible", (100, 0, 0)),
    17: ("EAC", (255, 225, 214)),
}
BONE_LABEL = 1

# ADF pose of the JHU anatomies (volume_256_RT143.yaml / volume_256_LT143.yaml).
# Their label arrays are in LPS index order, as this script writes them.
ORIENTATION = {
    "right": "{r: 1.57079, p: 1.4, y: 0}",
    "left": "{r: 1.57079, p: 1.8, y: 3.14}",
}


def say(kind, payload=""):
    print(f"@{kind} {payload}", flush=True)


def progress(fraction, text):
    say("progress", f"{fraction:.3f} {text}")


# ---------------------------------------------------------------- DICOM


def scan(dicom_dir: pathlib.Path):
    """Every series below dicom_dir: uid, folder, slice count, description (no patient data)."""
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalWarningDisplay(False)  # folders without DICOM are expected
    series = []
    folders = [dicom_dir] + sorted(p for p in dicom_dir.rglob("*") if p.is_dir())
    for folder in folders:
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid)
            if not files:
                continue
            info = {"uid": uid, "folder": str(folder), "slices": len(files)}
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(files[0])
                reader.LoadPrivateTagsOff()
                reader.ReadImageInformation()

                def tag(key):
                    return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""

                info.update(
                    modality=tag("0008|0060"),
                    description=tag("0008|103e"),
                    slice_thickness=tag("0018|0050"),
                    kernel=tag("0018|1210"),
                    rows=tag("0028|0010"),
                    columns=tag("0028|0011"),
                    pixel_spacing=tag("0028|0030"),
                )
            except RuntimeError as exc:
                info["note"] = str(exc).splitlines()[0]
            series.append(info)
    return series


def read_series(folder: str, uid: str):
    import SimpleITK as sitk

    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, uid)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetDimension() == 4:
        image = image[:, :, :, 0]
    # LPS index order: i toward the patient's left, j posterior, k superior.
    # The JHU training scans and FIVRS anatomies are stored this way.
    image = sitk.DICOMOrient(image, "LPS")
    return sitk.Cast(image, sitk.sitkInt16)


def laterality_from_text(text: str) -> str:
    """'left' / 'right' when a series name says so ("Left Dehiscence", "RT temporal", "L IAC"), else ''."""
    words = re.findall(r"[a-z]+", (text or "").lower())
    left = any(w in ("left", "lt", "l", "lft", "links", "gauche", "izq", "izquierdo") for w in words)
    right = any(w in ("right", "rt", "r", "rgt", "rechts", "droit", "der", "derecho") for w in words)
    return "left" if left and not right else "right" if right and not left else ""


def laterality_from_geometry(image) -> str:
    """A single-ear scan sits off the midline: +x is the patient's left in LPS. '' for a whole head."""
    size = np.array(image.GetSize())
    extent = size * np.array(image.GetSpacing())
    if extent[0] >= 110:
        return ""
    center = image.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2.0).tolist())
    if abs(center[0]) < 15:  # too close to the midline to tell
        return ""
    return "left" if center[0] > 0 else "right"


def guess_ear_centers(image):
    """Rough temporal bone centers (LPS mm) for each side: the densest bone mass in each half.

    The petrous bone is the thickest dense bone in the head; a 20 mm box filter of
    the bone mask peaks on it. Only a starting point; the proctor moves the box.
    """
    import SimpleITK as sitk
    from scipy import ndimage

    array_zyx = sitk.GetArrayViewFromImage(image)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = np.array(spacing_xyz[::-1], dtype=float)
    extent = np.array(array_zyx.shape) * spacing_zyx
    center_index = (np.array(array_zyx.shape) - 1) / 2.0
    if extent[2] < 110:  # already one ear (a temporal bone scan): the volume center
        point = list(image.TransformContinuousIndexToPhysicalPoint(center_index[::-1].tolist()))
        return {"left": point, "right": point}
    bone = (array_zyx > 900).astype(np.float32)
    size = np.maximum(1, np.round(20.0 / spacing_zyx)).astype(int)
    density = ndimage.uniform_filter(bone, size=tuple(size), mode="constant")
    mid = array_zyx.shape[2] // 2
    centers = {}
    # LPS: index i grows toward the patient's left, so the right ear is in the low-i half.
    for side, sl in (("right", slice(0, mid)), ("left", slice(mid, None))):
        half = density[:, :, sl]
        k, j, i = np.unravel_index(np.argmax(half), half.shape)
        i += sl.start or 0
        centers[side] = list(image.TransformIndexToPhysicalPoint([int(i), int(j), int(k)]))
    return centers


def series_description(folder: str, uid: str) -> list:
    import SimpleITK as sitk

    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, uid)
    reader = sitk.ImageFileReader()
    reader.SetFileName(files[0])
    reader.ReadImageInformation()
    parts = [reader.GetMetaData(key).strip() for key in ("0008|103e", "0018|0015", "0008|1030")
             if reader.HasMetaDataKey(key)]  # series description, body part, study description
    return [p for p in parts if p]


def load(dicom_dir: str, uid: str, work: pathlib.Path):
    import SimpleITK as sitk

    work.mkdir(parents=True, exist_ok=True)
    progress(0.1, "Reading the DICOM series")
    image = read_series(dicom_dir, uid)
    sitk.WriteImage(image, str(work / "ct.nii.gz"), True)
    progress(0.6, "Making the preview")
    spacing = np.array(image.GetSpacing())
    size = np.array(image.GetSize())
    factor = max(1.0, float((size * spacing).max() / PREVIEW_MAX / spacing.min()))
    preview_spacing = np.full(3, spacing.min() * factor)
    preview_size = np.maximum(1, np.round(size * spacing / preview_spacing)).astype(int)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(preview_spacing.tolist())
    resample.SetSize([int(v) for v in preview_size])
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(-1024)
    resample.SetInterpolator(sitk.sitkLinear)
    small = resample.Execute(image)
    array = sitk.GetArrayFromImage(small).astype(np.int16)  # z, y, x
    np.save(work / "preview.npy", array)
    progress(0.8, "Guessing where the ears are")
    centers = guess_ear_centers(small)
    texts = series_description(dicom_dir, uid)
    description = texts[0] if texts else ""
    # The series name wins over the study name ("TEMPORAL BONES" studies hold both sides).
    laterality = next((side for side in map(laterality_from_text, texts) if side), "")
    source = "series name"
    if not laterality:
        laterality, source = laterality_from_geometry(image), "scan position"
    meta = {
        "description": description,
        "laterality": laterality,
        "laterality_source": source if laterality else "",
        "dicom_dir": dicom_dir,
        "series_uid": uid,
        "size": [int(v) for v in size],
        "spacing": spacing.tolist(),
        "origin": list(image.GetOrigin()),
        "preview_spacing": list(small.GetSpacing()),
        "preview_origin": list(small.GetOrigin()),
        "preview_direction": list(small.GetDirection()),
        "ear_centers": centers,
    }
    (work / "ct.json").write_text(json.dumps(meta, indent=2))
    progress(1.0, "Loaded")
    say("result", json.dumps({"work": str(work), **meta}))


# ---------------------------------------------------------------- crop + inference


def crop_and_resample(work: pathlib.Path, center_mm, size_mm: float, spacing_mm: float):
    """A size_mm cube around center_mm (LPS), resampled to spacing_mm with B-splines."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(work / "ct.nii.gz"))
    count = int(round(size_mm / spacing_mm))
    origin = np.array(center_mm, dtype=float) - (count - 1) * spacing_mm / 2.0
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([spacing_mm] * 3)
    resample.SetSize([count] * 3)
    resample.SetOutputOrigin(origin.tolist())
    resample.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resample.SetDefaultPixelValue(-1024)  # outside the scan is air
    resample.SetInterpolator(sitk.sitkBSpline)
    cropped = sitk.Cast(sitk.Clamp(resample.Execute(image), sitk.sitkFloat32, -1024, 4000), sitk.sitkInt16)
    sitk.WriteImage(cropped, str(work / "crop_0000.nii.gz"), True)
    return cropped


class TqdmProgress:
    """nnU-Net's sliding window prints tqdm bars on stderr; turn them into @progress lines."""

    def __init__(self, low, high, label):
        self.low, self.high, self.label = low, high, label
        self.stream = sys.stderr
        self.last = -1.0

    def write(self, text):
        for match in re.finditer(r"(\d+)/(\d+)", text):
            done, total = int(match.group(1)), int(match.group(2))
            if total:
                fraction = self.low + (self.high - self.low) * done / total
                if fraction - self.last >= 0.005 or done == total:
                    self.last = fraction
                    progress(fraction, f"{self.label} ({done}/{total} tiles)")
        return len(text)

    def flush(self):
        pass


def allow_checkpoint_numpy(torch):
    """PyTorch >= 2.6 loads checkpoints with weights_only=True. The 2024 checkpoints also hold
    a few numpy scalars (logged metrics); allow those types and nothing else."""
    try:
        multiarray = np._core.multiarray
    except AttributeError:
        multiarray = np.core.multiarray
    # Pickled under numpy 1.x module names.
    allowed = [(multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype]
    allowed += [type(np.dtype(name)) for name in ("float64", "float32", "int64", "int32", "bool")]
    torch.serialization.add_safe_globals(allowed)


def segment(work: pathlib.Path, cropped, folds, tbone_dir: pathlib.Path, mirroring: bool, device: str):
    import SimpleITK as sitk
    import torch

    # nnunetv2.paths warns at import when these are unset; inference needs none of them.
    os.environ.setdefault("nnUNet_results", str(tbone_dir / "nnUNet_results"))
    os.environ.setdefault("nnUNet_raw", str(work))
    os.environ.setdefault("nnUNet_preprocessed", str(work))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model = tbone_dir / MODEL_SUBDIR
    if not (model / "plans.json").is_file():
        raise SystemExit(f"Model not found in {model}")
    missing = [f for f in folds if not (model / f"fold_{f}" / "checkpoint_final.pth").is_file()]
    if missing:
        raise SystemExit(f"Missing folds {missing} in {model}")

    allow_checkpoint_numpy(torch)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; running on the CPU (slow).", flush=True)
        torch_device = torch.device("cpu")
    # A 50 mm box keeps ~12 GiB of logits on the GPU (RTX 5090). Smaller cards (the lab's
    # RTX 5060), or a GPU the running simulator already fills, keep the logits in host
    # memory instead: slower, but it fits.
    on_device = torch_device.type == "cuda"
    if on_device:
        free = torch.cuda.mem_get_info(torch_device)[0] / 2**30
        need = 14.0 * (cropped.GetNumberOfPixels() / 510 ** 3)
        if free < need:
            print(f"GPU has {free:.1f} GiB free (about {need:.0f} needed on the GPU); "
                  "keeping the result in host memory.", flush=True)
            on_device = False
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirroring,
                                perform_everything_on_device=on_device,
                                device=torch_device, verbose=False, allow_tqdm=True)
    progress(0.12, f"Loading the model ({len(folds)} fold{'s' if len(folds) != 1 else ''})")
    predictor.initialize_from_trained_model_folder(str(model), use_folds=tuple(folds),
                                                   checkpoint_name="checkpoint_final.pth")

    # Same axis order as training (SimpleITKIO): array z, y, x and spacing reversed.
    array = sitk.GetArrayFromImage(cropped).astype(np.float32)[None]
    props = {"spacing": list(cropped.GetSpacing())[::-1]}

    # One tqdm bar per fold; spread them over 0.15-0.85.
    fold_index = {"n": 0}
    original = predictor.predict_sliding_window_return_logits

    def per_fold(data):
        n = fold_index["n"]
        low = 0.15 + 0.7 * n / len(folds)
        high = 0.15 + 0.7 * (n + 1) / len(folds)
        fold_index["n"] += 1
        sys.stderr = TqdmProgress(low, high, f"Segmenting, fold {n + 1} of {len(folds)}")
        try:
            return original(data)
        finally:
            sys.stderr = sys.__stderr__

    predictor.predict_sliding_window_return_logits = per_fold
    started = time.time()
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    labels = predictor.predict_single_npy_array(array, props, None, None, False)
    seconds = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch_device.type == "cuda" else 0.0
    print(f"Inference {seconds:.1f} s, peak GPU memory {peak:.1f} GiB", flush=True)
    return np.asarray(labels).astype(np.uint8), seconds, peak


def postprocess(labels, tbone_dir: pathlib.Path):
    """The model's own postprocessing (keep the largest component where that helped in cross-validation)."""
    from batchgenerators.utilities.file_and_folder_operations import load_pickle
    from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing

    pkl = tbone_dir / MODEL_SUBDIR / "crossval_results_folds_0_1_2_3_4" / "postprocessing.pkl"
    if not pkl.is_file():
        print("No postprocessing.pkl; skipping postprocessing.", flush=True)
        return labels
    pp_fns, pp_kwargs = load_pickle(str(pkl))
    return apply_postprocessing(labels, pp_fns, pp_kwargs).astype(np.uint8)


# ---------------------------------------------------------------- FIVRS volume


def slug(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return text or "segmented"


def write_volume(labels_zyx: np.ndarray, spacing_mm: float, name: str, laterality: str, factor: int,
                 repo: pathlib.Path = REPO):
    """Label map (model labels, z y x) -> PNG slices, ADF and icon, like seg_nrrd_to_volume.py."""
    import seg_nrrd_to_volume as conv

    labels_xyz = np.ascontiguousarray(np.transpose(labels_zyx, (2, 1, 0)))  # LPS i, j, k
    # One segment per color; sigmoid sinus and dura merge.
    segments, by_hex = [], {}
    for value, (seg_name, rgb) in LABELS.items():
        key = conv.hex_of(rgb)
        if key not in by_hex:
            by_hex[key] = {"name": seg_name, "rgb": list(rgb), "hex": key, "values": []}
            segments.append(by_hex[key])
        by_hex[key]["values"].append(value)
    known = conv.read_label_names()
    for segment in segments:
        segment["alias"] = known.get(segment["hex"], conv.clean_name(segment["name"]))
        segment["count"] = int(np.isin(labels_xyz, segment["values"]).sum())
    present = [s for s in segments if s["count"] > 0]
    if not any(s["hex"] == conv.BONE_HEX for s in present):
        raise SystemExit("The segmentation has no bone. Is the box on the ear?")
    # Larger segments first; smaller ones paint over them (as compose_labels does).
    present.sort(key=lambda s: -s["count"])
    ids = np.zeros(len(LABELS) + 1, dtype=np.uint8)
    for rank, segment in enumerate(present, start=1):
        segment["id"] = rank
        ids[segment["values"]] = rank
    composed = ids[labels_xyz]
    bone_id = next(s["id"] for s in present if s["hex"] == conv.BONE_HEX)
    composed = conv.downsample(conv.crop(composed, 2), factor, bone_id)
    voxel_mm = spacing_mm * factor

    tag = f"{slug(name)}_{voxel_mm:.3f}mm".replace(".", "p")
    folder = repo / "resources" / "volumes" / tag
    adf = repo / "ADF" / f"volume_{tag}.yaml"
    icon = repo / "scripts" / "study_gui" / "assets" / f"{slug(name)}.png"
    rows, cols, count = composed.shape
    dims = [cols * voxel_mm / 1000.0 / METERS_PER_AU, rows * voxel_mm / 1000.0 / METERS_PER_AU,
            count * voxel_mm / 1000.0 / METERS_PER_AU]
    conv.write_slices(composed, present, folder)
    volume_name = "mastoidectomy_volume_high" if max(composed.shape) > 384 else "mastoidectomy_volume_low"
    conv.write_adf(adf, folder, count, dims, laterality, volume_name)
    # write_adf poses a right ear; a left ear takes the LT anatomies' pose.
    text = adf.read_text().replace(f"orientation: {ORIENTATION['right']}", f"orientation: {ORIENTATION[laterality]}")
    adf.write_text(text)
    conv.write_icon(composed, present, icon)
    # No write_label_names: the tuner (tuner_materials.json "names") already names these colors.
    counts = {s["name"]: s["count"] for s in present}
    return {"adf": str(adf), "icon": str(icon), "folder": str(folder), "slices": count,
            "shape": [int(v) for v in composed.shape], "voxel_mm": voxel_mm, "segments": counts}


CROP_LANDMARKS = {"incus": 3, "malleus": 2, "bony_labyrinth": 5, "facial_nerve": 10, "eac": 17,
                  "ica": 12, "sinus_dura": (13, 14), "mandible": 16}


def crop_record(labels_zyx: np.ndarray, image, center_mm, size_mm: float) -> dict:
    """Where the landmarks landed in the box, for learning an automatic crop later.

    Offsets are LPS mm from the box center (x patient-left, y posterior, z superior).
    edge_mm: how close each landmark comes to the nearest box face (0 = cut off).
    bone_on_faces: fraction of each box face that is bone, i.e. how much bone the box cut.
    """
    spacing = np.array(image.GetSpacing())  # x, y, z
    center_index = (np.array(labels_zyx.shape[::-1]) - 1) / 2.0
    landmarks = {}
    for name, values in CROP_LANDMARKS.items():
        mask = np.isin(labels_zyx, values)
        if not mask.any():
            continue
        zyx = np.argwhere(mask)
        xyz = zyx[:, ::-1]
        low, high = xyz.min(0), xyz.max(0)
        edge = np.minimum(low, np.array(labels_zyx.shape[::-1]) - 1 - high) * spacing
        landmarks[name] = {"offset_mm": np.round((xyz.mean(0) - center_index) * spacing, 2).tolist(),
                           "edge_mm": round(float(edge.min()), 2), "voxels": int(len(zyx))}
    bone = labels_zyx == BONE_LABEL
    faces = {}
    for axis_zyx, axis in ((2, "x"), (1, "y"), (0, "z")):
        faces[f"-{axis}"] = round(float(np.take(bone, 0, axis_zyx).mean()), 3)
        faces[f"+{axis}"] = round(float(np.take(bone, -1, axis_zyx).mean()), 3)
    return {"center_mm": [round(float(v), 3) for v in center_mm], "size_mm": size_mm,
            "landmarks": landmarks, "bone_on_faces": faces}


def run(args):
    import SimpleITK as sitk

    work = pathlib.Path(args.work)
    tbone_dir = pathlib.Path(args.tbone_dir)
    folds = [int(f) for f in args.folds]
    progress(0.02, f"Cropping a {args.size:.0f} mm box and resampling to {MODEL_SPACING_MM} mm")
    cropped = crop_and_resample(work, args.center, args.size, MODEL_SPACING_MM)
    print(f"Crop {cropped.GetSize()} voxels at {MODEL_SPACING_MM} mm", flush=True)
    labels, seconds, peak = segment(work, cropped, folds, tbone_dir, not args.no_mirroring, args.device)
    if not args.no_postprocessing:
        progress(0.87, "Postprocessing")
        labels = postprocess(labels, tbone_dir)
    seg = sitk.GetImageFromArray(labels)
    seg.CopyInformation(cropped)
    sitk.WriteImage(seg, str(work / "segmentation.nii.gz"), True)
    progress(0.9, "Writing the simulator volume")
    result = write_volume(labels, MODEL_SPACING_MM, args.name, args.laterality, args.factor, pathlib.Path(args.repo))
    result.update(work=str(work), folds=folds, seconds=round(seconds, 1), peak_gpu_gib=round(peak, 2),
                  name=args.name, laterality=args.laterality, crop_mm=args.size, center=list(args.center),
                  mirroring=not args.no_mirroring,
                  crop=crop_record(labels, cropped, args.center, args.size))
    (work / "result.json").write_text(json.dumps(result, indent=2))
    progress(1.0, "Done")
    say("result", json.dumps(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan")
    p.add_argument("dicom_dir")
    p = sub.add_parser("load")
    p.add_argument("dicom_dir")
    p.add_argument("series_uid")
    p.add_argument("work")
    p = sub.add_parser("run")
    p.add_argument("work")
    p.add_argument("--center", type=float, nargs=3, required=True, help="Box center, LPS mm")
    p.add_argument("--size", type=float, default=DEFAULT_CROP_MM, help="Box side, mm")
    p.add_argument("--folds", nargs="+", default=["0"], help="Model folds 0-4 (more = ensemble, slower)")
    p.add_argument("--name", required=True)
    p.add_argument("--laterality", choices=("left", "right"), required=True)
    p.add_argument("--factor", type=int, default=2, help="Downsampling for the simulator (1 = 0.1 mm, 2 = 0.2 mm)")
    p.add_argument("--no-mirroring", action="store_true", help="Skip test-time mirroring (8x faster)")
    p.add_argument("--no-postprocessing", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tbone-dir", default=os.environ.get("TBONE_SEG_DIR", str(DEFAULT_TBONE_DIR)))
    p.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            series = scan(pathlib.Path(args.dicom_dir).expanduser())
            say("result", json.dumps({"series": series}))
        elif args.command == "load":
            load(args.dicom_dir, args.series_uid, pathlib.Path(args.work))
        else:
            run(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            say("error", str(exc.code))
            return 1
    except Exception as exc:  # report to the launcher, not just a traceback
        traceback.print_exc()
        say("error", f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
