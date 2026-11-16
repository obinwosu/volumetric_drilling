#!/usr/bin/env python3
"""Rigid MRI -> CT registration for the launcher's "Register MRI" tab.

Runs in the tbone-seg-nnunetv2 venv (SimpleITK), like scripts/segmentation/tbone_worker.py,
and talks to the launcher the same way: stdout lines ``@progress <0-1> <text>``,
``@result <json>``, ``@error <text>``; anything else is log text.

The temporal bone is rigid, so the model is rigid (3 rotations + 3 translations). The
match is edge directions around the temporal bones, found by a brute-force search on
the GPU and refined with Powell's method; why, and what failed: rigid_register.py.

Typical inputs: a thin bone CT (0.2-0.6 mm) and either a heavily T2-weighted
CISS / FIESTA (0.4-0.8 mm) or a post-contrast T1 (thicker slices). Both work the
same way; for thick slices the last level stops at about the slice spacing.

Outputs in <work>:
  ct.nii.gz, mri.nii.gz       the scans as read (LPS)
  mri_to_ct.tfm               ITK transform (CT point -> MRI point). 3D Slicer loads it;
                              put the MRI under it and the MRI lands on the CT.
  mri_on_ct.nii.gz            the MRI resampled onto the CT grid
  preview_ct.npy, preview_mri.npy  the two on one small grid, for the launcher's views
                              (MRI 1-255; 0 where the MRI has no data)
  registration.json           transform, metric, candidates, timings

Commands:
  scan <dicom_dir>                                  list series (no patient data)
  run --ct DIR [--ct-uid UID] --mri DIR [--mri-uid UID] --work DIR
                                                    register; DIR may also be a NIfTI/NRRD file
  preview <work>                                    redo mri_on_ct + previews from mri_to_ct.tfm
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "segmentation"))
sys.path.insert(0, str(HERE))

CT_PADDING = -1500.0  # below this: outside the reconstruction circle
PREVIEW_MAX = 320


def say(kind, payload=""):
    print(f"@{kind} {payload}", flush=True)


def progress(fraction, text):
    say("progress", f"{fraction:.3f} {text}")


# ---------------------------------------------------------------- reading


def series_files(folder, uid):
    import SimpleITK as sitk

    if not uid:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or ()
        if not ids:  # look below
            for sub in sorted(p for p in pathlib.Path(folder).rglob("*") if p.is_dir()):
                ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(sub)) or ()
                if ids:
                    folder = sub
                    break
        if len(ids) != 1:
            raise SystemExit(f"{folder}: {len(ids)} series; choose one (--ct-uid / --mri-uid)")
        uid = ids[0]
    return sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid)


def read_image(source, uid=None):
    """A DICOM series (folder + uid) or an image file, as float32 in LPS order. Also returns a
    short description (series name) for the result."""
    import SimpleITK as sitk

    path = pathlib.Path(source).expanduser()
    description = ""
    if path.is_dir():
        files = series_files(path, uid)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        image = reader.Execute()
        if reader.HasMetaDataKey(0, "0008|103e"):
            description = reader.GetMetaData(0, "0008|103e").strip()
    else:
        image = sitk.ReadImage(str(path))
        description = path.name
    if image.GetDimension() == 4:
        image = image[:, :, :, 0]
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0)
    image = sitk.DICOMOrient(image, "LPS")
    return sitk.Cast(image, sitk.sitkFloat32), description


def frame_uid(source, uid=None) -> str:
    """The series' DICOM frame of reference UID ('' for an image file). Series that share it are
    in the same scanner world, so one registration serves them all (C+ T1 next to C+ FIESTA)."""
    import SimpleITK as sitk

    path = pathlib.Path(source).expanduser()
    if not path.is_dir():
        return ""
    reader = sitk.ImageFileReader()
    reader.SetFileName(series_files(path, uid)[0])
    reader.ReadImageInformation()
    return reader.GetMetaData("0020|0052").strip() if reader.HasMetaDataKey("0020|0052") else ""


# ---------------------------------------------------------------- pyramid


def isotropic(image, spacing_mm, default, smooth=True):
    """image resampled to isotropic spacing_mm over its own extent, blurred first so the
    coarse level is an average, not a sample."""
    import SimpleITK as sitk

    spacing = np.array(image.GetSpacing())
    if smooth:
        sigma = np.sqrt(np.maximum(0.0, (spacing_mm / 2.0) ** 2 - (spacing / 2.0) ** 2))
        if sigma.max() > 0:
            image = sitk.SmoothingRecursiveGaussian(image, [float(max(s, 1e-3)) for s in sigma])
    size = np.array(image.GetSize())
    out_size = np.maximum(1, np.round(size * spacing / spacing_mm)).astype(int)
    # Keep the same physical extent: first output voxel centre half a voxel in from the edge.
    direction = np.array(image.GetDirection()).reshape(3, 3)
    corner = np.array(image.GetOrigin()) - direction @ (spacing / 2.0)
    origin = corner + direction @ np.full(3, spacing_mm / 2.0)
    return sitk.Resample(image, [int(v) for v in out_size], sitk.Transform(), sitk.sitkLinear,
                         origin.tolist(), [float(spacing_mm)] * 3, image.GetDirection(), float(default),
                         sitk.sitkFloat32)



def ct_valid_mask(ct):
    """Real CT voxels: inside the reconstruction circle (outside it scanners write -2048 or so)."""
    import SimpleITK as sitk

    return sitk.BinaryThreshold(ct, CT_PADDING, 1e9, 1, 0)


def mri_unit(mri):
    """MRI scaled so its 0.5-99.5 percentiles are 0-1 (scanners differ by orders of magnitude)."""
    import SimpleITK as sitk

    array = sitk.GetArrayViewFromImage(mri)
    low, high = np.percentile(array[::2, ::2, ::2], [0.5, 99.5])
    return sitk.Clamp(sitk.ShiftScale(mri, -low, 1.0 / max(high - low, 1e-6)), sitk.sitkFloat32, 0.0, 1.0)


# ---------------------------------------------------------------- registration


def to_transform(params, center):
    """rigid_register pose -> ITK rigid transform (CT point -> MRI point)."""
    import SimpleITK as sitk

    import rigid_register

    transform = sitk.Euler3DTransform()
    transform.SetCenter([float(v) for v in center])
    transform.SetMatrix([float(v) for v in rigid_register.rotation(params[:3]).ravel()], 1e-6)
    transform.SetTranslation([float(v) for v in params[3:]])
    return transform


def temporal_bones(ct, tbone_worker):
    """The densest bone in each half of the CT (the petrous bones), or the middle of a one-ear scan."""
    import SimpleITK as sitk

    small = isotropic(ct, 1.0, -1024.0)
    centers = tbone_worker.guess_ear_centers(sitk.Cast(small, sitk.sitkInt16))
    left, right = np.array(centers["left"]), np.array(centers["right"])
    return [left] if np.linalg.norm(left - right) < 1.0 else [right, left]


def register(ct, mri, log=print):
    import rigid_register

    import tbone_worker

    ears = temporal_bones(ct, tbone_worker)
    log("temporal bones at " + "; ".join(", ".join(f"{v:.0f}" for v in e) for e in ears) + " (LPS mm)")
    params, center, record = rigid_register.register(ct, ct_valid_mask(ct), mri_unit(mri), isotropic, ears=ears,
                                                   progress=progress, log=log)
    record["temporal_bones_mm"] = [list(map(float, e)) for e in ears]
    record.update(pitch_roll_yaw_deg=[float(v) for v in params[:3]], translation_mm=[float(v) for v in params[3:]],
                  center_mm=[float(v) for v in center])
    return to_transform(params, center), record

# ---------------------------------------------------------------- outputs


def write_resampled(work, ct, mri, transform):
    """The MRI on the CT grid, and the small common-grid previews the launcher shows."""
    import SimpleITK as sitk

    on_ct = sitk.Resample(mri, ct, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    sitk.WriteImage(sitk.Cast(sitk.Clamp(on_ct, sitk.sitkFloat32, -32768, 32767), sitk.sitkInt16),
                    str(work / "mri_on_ct.nii.gz"), True)
    # Small common grid for the launcher: the CT's extent, isotropic.
    spacing = np.array(ct.GetSpacing())
    size = np.array(ct.GetSize())
    step = float((size * spacing).max() / PREVIEW_MAX)
    grid = isotropic(ct, step, -1024.0, smooth=True)
    preview_ct = sitk.GetArrayFromImage(grid).astype(np.int16)
    smooth_mri = isotropic(mri, max(step, float(np.min(mri.GetSpacing()))), 0.0)
    covered = sitk.Image(mri.GetSize(), sitk.sitkUInt8) + 1  # 1 inside the MRI's field of view
    covered.CopyInformation(mri)
    preview_mri = sitk.GetArrayFromImage(sitk.Resample(smooth_mri, grid, transform, sitk.sitkLinear, 0.0))
    inside = sitk.GetArrayFromImage(sitk.Resample(covered, grid, transform, sitk.sitkNearestNeighbor, 0)) > 0
    low, high = np.percentile(preview_mri[inside], [1, 99.5]) if inside.any() else (0, 1)
    preview_mri = (np.clip((preview_mri - low) / max(high - low, 1e-6), 0, 1) * 254 + 1).astype(np.uint8)
    preview_mri[~inside] = 0  # 0 = no MRI here
    np.save(work / "preview_ct.npy", preview_ct)
    np.save(work / "preview_mri.npy", preview_mri)
    return {"preview_spacing_mm": step, "preview_origin_mm": list(grid.GetOrigin()),
            "preview_direction": list(grid.GetDirection()),
            "matrix": list(transform.GetMatrix())}


def write_outputs(work, ct, mri, transform, record, descriptions):
    import SimpleITK as sitk

    progress(0.95, "Writing the results")
    work.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(ct, sitk.sitkInt16), str(work / "ct.nii.gz"), True)
    sitk.WriteImage(mri, str(work / "mri.nii.gz"), True)
    sitk.WriteTransform(transform, str(work / "mri_to_ct.tfm"))
    return {
        "work": str(work.resolve()),
        "ct_description": descriptions[0],
        "mri_description": descriptions[1],
        "ct_spacing_mm": list(ct.GetSpacing()),
        "mri_spacing_mm": list(mri.GetSpacing()),
        **write_resampled(work, ct, mri, transform),
        **record,
    }


def run(args):
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    start = time.time()
    progress(0.01, "Reading the CT")
    ct, ct_name = read_image(args.ct, args.ct_uid)
    progress(0.03, "Reading the MRI")
    mri, mri_name = read_image(args.mri, args.mri_uid)
    print(f"CT  {ct_name}: {ct.GetSize()} at {np.round(ct.GetSpacing(), 3).tolist()} mm")
    print(f"MRI {mri_name}: {mri.GetSize()} at {np.round(mri.GetSpacing(), 3).tolist()} mm")
    transform, record = register(ct, mri)
    work = pathlib.Path(args.work)
    result = write_outputs(work, ct, mri, transform, record, (ct_name, mri_name))
    result["seconds"] = time.time() - start
    result["mri_frame_uid"] = frame_uid(args.mri, args.mri_uid)
    result["ct_frame_uid"] = frame_uid(args.ct, args.ct_uid)
    (work / "registration.json").write_text(json.dumps(result, indent=2))
    progress(1.0, "Registered")
    say("result", json.dumps({k: v for k, v in result.items() if k != "candidates"}))


def preview(args):
    """Redo the resampled MRI and previews from mri_to_ct.tfm, e.g. after it was adjusted and
    saved over in 3D Slicer."""
    import SimpleITK as sitk

    work = pathlib.Path(args.work)
    progress(0.1, "Reading")
    ct = sitk.ReadImage(str(work / "ct.nii.gz"), sitk.sitkFloat32)
    mri = sitk.ReadImage(str(work / "mri.nii.gz"), sitk.sitkFloat32)
    transform = sitk.ReadTransform(str(work / "mri_to_ct.tfm"))
    progress(0.4, "Resampling the MRI")
    path = work / "registration.json"
    result = json.loads(path.read_text()) if path.is_file() else {"work": str(work.resolve())}
    result.update(write_resampled(work, ct, mri, transform), transform_edited=time.strftime("%Y-%m-%d %H:%M"))
    path.write_text(json.dumps(result, indent=2))
    progress(1.0, "Reloaded")
    say("result", json.dumps({k: v for k, v in result.items() if k != "candidates"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan")
    p.add_argument("dicom_dir")
    p = sub.add_parser("run")
    p.add_argument("--ct", required=True, help="CT DICOM folder or image file")
    p.add_argument("--ct-uid", default=None)
    p.add_argument("--mri", required=True, help="MRI DICOM folder or image file")
    p.add_argument("--mri-uid", default=None)
    p.add_argument("--work", required=True)
    p = sub.add_parser("preview")
    p.add_argument("work")
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            import SimpleITK as sitk

            import tbone_worker

            sitk.ProcessObject.SetGlobalWarningDisplay(False)
            say("result", json.dumps({"series": tbone_worker.scan(pathlib.Path(args.dicom_dir).expanduser())}))
        elif args.command == "preview":
            preview(args)
        else:
            run(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            say("error", str(exc.code))
            return 1
    except Exception as exc:
        traceback.print_exc()
        say("error", f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
