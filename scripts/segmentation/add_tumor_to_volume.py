#!/usr/bin/env python3
"""Make an anatomy with a tumor: temporal bone CT + tumor segmented on a registered MRI.

  ../tbone-seg-nnunetv2/.venv/bin/python scripts/segmentation/add_tumor_to_volume.py \\
      --ct ~/Downloads/pcseg_dicom --registration resources/intermediate_volumes/registration/<stamp> \\
      --name "Left Dehiscence CT + VS" --laterality left

The tumor comes from the Register MRI tab (<registration>/tumor/tumor_mri.nii.gz, parts on the
MRI's grid) and mri_to_ct.tfm. The CT here may be any reconstruction in the registration CT's
DICOM frame of reference (checked), e.g. the one-ear "Left Dehiscence" recon of the bilateral
CT used as the registration target.

Why a new segmentation: the first anatomy's 50 mm box held only about half of the tumor (it
was centred on the temporal bone; the CPA part lies medial to it, the middle fossa part
anterior). The box here holds the temporal bone box of an earlier run (--ear-box-from, the
"Add a volume" work folder) and the whole tumor plus a margin, and is segmented by the same
tbone nnU-Net (fold 0, mirroring, the model's postprocessing), as tbone_worker.py does.

With --ear-box-from, the labels inside that box are the existing anatomy's, copied voxel for
voxel (the box is snapped to its grid); the model only fills the area outside it (or, with
--outside-from, an earlier run of this script does).

The tumor is one more label (TUMOR_RGB, named in ADF/color/volume_labels.yaml). It only
takes voxels that are empty, IAC contents, the vestibular / cochlear nerves or dura: bone,
labyrinth, facial nerve, sigmoid sinus, carotid and ossicles stay as segmented.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "registration"))
sys.path.insert(0, str(REPO / "scripts" / "utils"))

TUMOR_LABEL = 18
TUMOR_NAME = "Vestibular schwannoma"
TUMOR_ALIAS = "vestibular_schwannoma"
TUMOR_RGB = (190, 110, 170)
# Model labels the tumor may replace: empty, IAC, superior / inferior vestibular and cochlear nerves,
# and dura (14). The model draws dura as a ~3 mm band it cannot see on a bone-kernel CT; the tumor is
# seen on the contrast MRI. On the first patient 1.8 mL of the tumor's base lay in that band.
REPLACEABLE = (0, 6, 7, 8, 9, 14)
MARGIN_MM = 3.0
TUMOR_SMOOTH_MM = 1.0
MAX_PIECE_VOXELS = 80_000_000  # ~220 bytes a voxel: 89 M fitted in 20 GB, 109 M did not
OVERLAP_MM = 8.0


def box_of(points_lps, margin):
    return np.min(points_lps, axis=0) - margin, np.max(points_lps, axis=0) + margin


def crop_box(ct, low, high, spacing):
    """The CT resampled to `spacing` over the LPS box [low, high] (axis-aligned)."""
    import SimpleITK as sitk

    count = np.ceil((np.asarray(high) - np.asarray(low)) / spacing).astype(int)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([spacing] * 3)
    resample.SetSize([int(v) for v in count])
    resample.SetOutputOrigin([float(v) for v in low])
    resample.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resample.SetDefaultPixelValue(-1024)
    resample.SetInterpolator(sitk.sitkBSpline)
    return sitk.Cast(sitk.Clamp(resample.Execute(ct), sitk.sitkFloat32, -1024, 4000), sitk.sitkInt16)


def segment_in_pieces(work, cropped, tbone_dir, max_voxels=MAX_PIECE_VOXELS, overlap_mm=OVERLAP_MM):
    """tbone nnU-Net on a box larger than the 50 mm it was made for, without running out of host
    memory: the whole 62 x 62 x 51 mm box (208 M voxels) went to 24.7 GB and was killed on this
    30 GB machine. Split along the longest axis into pieces of at most max_voxels (the 50 mm cube
    is 133 M; 124 M still went past 20 GB) that overlap by overlap_mm, segment each, and keep each piece's own half of every
    overlap. Returns (labels z y x, seconds, peak GiB)."""
    import tbone_worker

    array_shape = np.array(cropped.GetSize())[::-1]  # z, y, x
    total = int(np.prod(array_shape))
    axis = int(np.argmax(array_shape))
    length = array_shape[axis]
    overlap = int(round(overlap_mm / tbone_worker.MODEL_SPACING_MM))
    across = total // length  # voxels in one slice across the split axis
    pieces = 1
    while pieces < length and (int(np.ceil(length / pieces)) + (overlap if pieces > 1 else 0)) * across > max_voxels:
        pieces += 1  # the overlap counts: a 109 M piece (12 mm overlap included) was killed at 20 GB
    edges = np.linspace(0, length, pieces + 1).round().astype(int)  # where each piece's own part ends
    labels = np.zeros(array_shape, np.uint8)
    seconds = peak = 0.0
    for n in range(pieces):
        lo, hi = max(0, edges[n] - overlap // 2), min(length, edges[n + 1] + overlap // 2)
        index = [0, 0, 0]
        size = list(cropped.GetSize())
        xyz_axis = 2 - axis
        index[xyz_axis], size[xyz_axis] = int(lo), int(hi - lo)
        piece = cropped[index[0]:index[0] + size[0], index[1]:index[1] + size[1], index[2]:index[2] + size[2]]
        print(f"Piece {n + 1} of {pieces}: {piece.GetSize()}", flush=True)
        part, s, p = tbone_worker.segment(work, piece, [0], tbone_dir, True, "cuda")
        seconds, peak = seconds + s, max(peak, p)
        keep = [slice(None)] * 3
        keep[axis] = slice(edges[n] - lo, edges[n + 1] - lo)
        target = [slice(None)] * 3
        target[axis] = slice(edges[n], edges[n + 1])
        labels[tuple(target)] = part[tuple(keep)]
        del part
    return labels, seconds, peak


def amb_rotation(roll, pitch, yaw):
    """AMBF afMatrix3d::setRPY (core/ambf_math/afMath.h): Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, cp, cy, sr, sp, sy = np.cos(roll), np.cos(pitch), np.cos(yaw), np.sin(roll), np.sin(pitch), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def adf_pose(path: pathlib.Path):
    import re

    text = path.read_text()
    number = r"([-\d.eE+]+)"
    position = re.search(rf"position: \{{x: {number}, y: {number}, z: {number}\}}", text)
    orientation = re.search(rf"orientation: \{{r: {number}, p: {number}, y: {number}\}}", text)
    dims = re.search(rf"dimensions: \{{x: {number}, y: {number}, z: {number}\}}", text)
    return (np.array([float(v) for v in position.groups()]), [float(v) for v in orientation.groups()],
            np.array([float(v) for v in dims.groups()]), text)


def keep_pose(new_adf: pathlib.Path, old_adf: pathlib.Path, offset_rows_cols_slices, old_shape, new_shape):
    """Move the new volume so the old volume's voxels land where they did in the world (the ear
    stays where saved views expect it). offset: old voxel (row, col, slice) -> new.
    Volume local x = image columns, y = rows, z = slices; local = index * dim / count - dim / 2."""
    old_position, old_rpy, old_dims, _ = adf_pose(old_adf)
    _, new_rpy, new_dims, text = adf_pose(new_adf)
    rows, cols, slices = offset_rows_cols_slices
    old_counts = np.array([old_shape[1], old_shape[0], old_shape[2]], float)  # cols, rows, slices
    new_counts = np.array([new_shape[1], new_shape[0], new_shape[2]], float)
    old_centre_in_new = np.array([cols, rows, slices], float) + old_counts / 2.0
    local = old_centre_in_new * new_dims / new_counts - new_dims / 2.0
    position = old_position - amb_rotation(*new_rpy) @ local  # the old centre was at old_position
    import re
    text = re.sub(r"position: \{x: [^}]*\}.*", "position: {x: %.9f, y: %.9f, z: %.9f}  # keeps the ear where %s has it"
                  % (*position, old_adf.name), text, count=1)
    new_adf.write_text(text)
    return position


def tumor_on(grid, registration: pathlib.Path):
    """Union of the tumor parts on `grid` (uint8 0/1), smoothed so thick MRI slices do not step."""
    import SimpleITK as sitk

    labels = sitk.ReadImage(str(registration / "tumor" / "tumor_mri.nii.gz"))
    transform = sitk.ReadTransform(str(registration / "mri_to_ct.tfm"))
    # sigma 1 mm: the automatic part comes from a 2.5 mm-slice C+ T1 and was terraced at 0.4 mm.
    mask = sitk.SmoothingRecursiveGaussian(sitk.Cast(labels > 0, sitk.sitkFloat32), TUMOR_SMOOTH_MM)
    on_grid = sitk.Resample(mask, grid, transform, sitk.sitkLinear, 0.0)
    return (sitk.GetArrayFromImage(on_grid) > 0.5).astype(np.uint8)


def tumor_points(registration: pathlib.Path, step=3):
    """Tumor voxel centres in CT (LPS) space, every `step`-th voxel."""
    import SimpleITK as sitk

    labels = sitk.ReadImage(str(registration / "tumor" / "tumor_mri.nii.gz"))
    inverse = sitk.ReadTransform(str(registration / "mri_to_ct.tfm")).GetInverse()
    index = np.argwhere(sitk.GetArrayViewFromImage(labels) > 0)[::step]
    return np.array([inverse.TransformPoint(labels.TransformIndexToPhysicalPoint([int(i), int(j), int(k)]))
                     for k, j, i in index])


def add_label_name():
    import seg_nrrd_to_volume as conv

    conv.write_label_names([{"hex": conv.hex_of(TUMOR_RGB), "rgb": list(TUMOR_RGB), "alias": TUMOR_ALIAS,
                             "name": TUMOR_NAME}], "add_tumor_to_volume.py")


def main():
    import SimpleITK as sitk

    import mri_ct_worker
    import tbone_worker

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ct", required=True, help="CT DICOM folder (or NIfTI) to segment")
    parser.add_argument("--ct-uid", default=None)
    parser.add_argument("--registration", required=True, type=pathlib.Path)
    parser.add_argument("--ear-box-from", type=pathlib.Path, default=None,
                        help="'Add a volume' work folder whose crop box must stay inside (crop_0000.nii.gz)")
    parser.add_argument("--outside-from", type=pathlib.Path, default=None,
                        help="earlier run of this script whose segmentation.nii.gz covers the box: reuse it "
                             "outside the ear box instead of running the model again")
    parser.add_argument("--keep-pose-of", type=pathlib.Path, default=None,
                        help="the anatomy ADF made from --ear-box-from: place the new volume so its ear stays put")
    parser.add_argument("--name", required=True)
    parser.add_argument("--laterality", choices=("left", "right"), required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--tbone-dir", default=str(tbone_worker.DEFAULT_TBONE_DIR))
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    start = time.time()

    registration = json.loads((args.registration / "registration.json").read_text())
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(args.registration / "ct.nii.gz"))
    reader.ReadImageInformation()
    frame = mri_ct_worker.frame_uid(args.ct, args.ct_uid)
    print(f"CT frame of reference {'checked' if frame else 'unknown (not DICOM)'}", flush=True)
    ct, description = mri_ct_worker.read_image(args.ct, args.ct_uid)
    registered_ct = registration.get("ct_frame_uid")
    if registered_ct and frame and frame != registered_ct:
        raise SystemExit("The CT is not in the registration CT's frame of reference.")

    points = tumor_points(args.registration)
    low, high = box_of(points, MARGIN_MM)
    if args.ear_box_from:
        ear = sitk.ReadImage(str(args.ear_box_from / "crop_0000.nii.gz"))
        ear_low = np.array(ear.GetOrigin())
        ear_high = ear_low + np.array(ear.GetSize()) * np.array(ear.GetSpacing())
        low, high = np.minimum(low, ear_low), np.maximum(high, ear_high)
    # Stay inside the CT.
    ct_low = np.array(ct.GetOrigin()) - np.array(ct.GetSpacing()) / 2  # ct is LPS-oriented, identity-like
    ct_high = ct_low + np.array(ct.GetSize()) * np.array(ct.GetSpacing())
    low, high = np.maximum(low, ct_low), np.minimum(high, ct_high)
    ear = None
    if args.ear_box_from:
        # Snap the box to the ear box's grid, an even number of voxels out, so the existing labels
        # are copied without resampling and the 2x downsampling blocks line up with the old volume's.
        ear = sitk.ReadImage(str(args.ear_box_from / "segmentation.nii.gz"))
        step2 = 2 * tbone_worker.MODEL_SPACING_MM
        ear_origin = np.array(ear.GetOrigin())
        low = ear_origin - np.ceil(np.maximum(ear_origin - low, 0) / step2 - 1e-6) * step2
    print(f"Box {np.round(low, 1)} .. {np.round(high, 1)} mm ({np.round(high - low, 1)} mm)", flush=True)

    work = REPO / "resources" / "intermediate_volumes" / "segmentation" / time.strftime("%Y%m%d_%H%M%S_tumor")
    work.mkdir(parents=True, exist_ok=True)
    cropped = crop_box(ct, low, high, tbone_worker.MODEL_SPACING_MM)
    sitk.WriteImage(cropped, str(work / "crop_0000.nii.gz"), True)
    if args.outside_from:
        earlier = sitk.ReadImage(str(args.outside_from / "segmentation.nii.gz"))
        earlier = sitk.ChangeLabel(earlier, {TUMOR_LABEL: 0})  # the tumor is written again below
        labels = sitk.GetArrayFromImage(sitk.Resample(earlier, cropped, sitk.Transform(),
                                                      sitk.sitkNearestNeighbor, 0)).astype(np.uint8)
        print(f"Outside the ear box: labels of {args.outside_from.name} (no new inference)", flush=True)
    else:
        print(f"Crop {cropped.GetSize()} at {tbone_worker.MODEL_SPACING_MM} mm; segmenting", flush=True)
        labels, seconds, peak = segment_in_pieces(work, cropped, pathlib.Path(args.tbone_dir))
        labels = tbone_worker.postprocess(labels, pathlib.Path(args.tbone_dir))
        print(f"Segmented in {seconds:.0f} s (GPU peak {peak:.1f} GiB)", flush=True)
    if ear is not None:
        # Inside the ear box: the anatomy's existing labels, voxel for voxel.
        offset = np.round((np.array(ear.GetOrigin()) - low) / tbone_worker.MODEL_SPACING_MM).astype(int)[::-1]
        old = sitk.GetArrayFromImage(ear)
        end = np.minimum(offset + old.shape, labels.shape)
        labels[offset[0]:end[0], offset[1]:end[1], offset[2]:end[2]] = \
            old[:end[0] - offset[0], :end[1] - offset[1], :end[2] - offset[2]]
        print(f"Inside the ear box: the existing labels of {args.ear_box_from.name}", flush=True)

    tumor = tumor_on(cropped, args.registration)
    take = (tumor > 0) & np.isin(labels, REPLACEABLE)
    kept = int(((tumor > 0) & ~take).sum())
    labels = labels.copy()
    labels[take] = TUMOR_LABEL
    voxel_ml = tbone_worker.MODEL_SPACING_MM ** 3 / 1000.0
    print(f"Tumor {tumor.sum() * voxel_ml:.2f} mL on the crop; {take.sum() * voxel_ml:.2f} mL written, "
          f"{kept * voxel_ml:.2f} mL left to bone / other structures", flush=True)
    seg = sitk.GetImageFromArray(labels)
    seg.CopyInformation(cropped)
    sitk.WriteImage(seg, str(work / "segmentation.nii.gz"), True)

    tbone_worker.LABELS[TUMOR_LABEL] = (TUMOR_NAME, TUMOR_RGB)
    add_label_name()
    result = tbone_worker.write_volume(labels, tbone_worker.MODEL_SPACING_MM, args.name, args.laterality,
                                       args.factor, REPO)
    if ear is not None and args.keep_pose_of:
        def crop_low(array_zyx):
            occupied = np.argwhere(array_zyx > 0)
            return np.maximum(occupied.min(axis=0) - 2, 0)  # as seg_nrrd_to_volume.crop, z y x here
        old_labels = sitk.GetArrayFromImage(ear)
        ear_offset = np.round((np.array(ear.GetOrigin()) - low) / tbone_worker.MODEL_SPACING_MM).astype(int)[::-1]
        fine = ear_offset + crop_low(old_labels) - crop_low(labels)  # z y x, at the model spacing
        if np.any(fine % args.factor):
            print(f"Warning: offset {fine} is not a whole number of simulator voxels", flush=True)
        k, j, i = fine // args.factor
        old_folder = (args.keep_pose_of.parent / re.search(r"path: (\S+)\n    prefix", args.keep_pose_of.read_text())
                      .group(1)).resolve()
        from PIL import Image
        first = Image.open(next(old_folder.glob("plane00*.png")))
        old_shape = (first.height, first.width, len(list(old_folder.glob("plane00*.png"))))
        position = keep_pose(pathlib.Path(result["adf"]), args.keep_pose_of, (i, j, k), old_shape, result["shape"])
        result["position_au"] = position.tolist()
        print(f"Placed so the ear stays where {args.keep_pose_of.name} has it (offset rows {i}, cols {j}, "
              f"slices {k})", flush=True)
    result.update(work=str(work), ct=description, registration=str(args.registration), box_low_mm=low.tolist(),
                  box_high_mm=high.tolist(), tumor_ml=float(take.sum() * voxel_ml), seconds=time.time() - start)
    (work / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("adf", "icon", "shape", "voxel_mm", "tumor_ml")}), flush=True)


if __name__ == "__main__":
    main()
