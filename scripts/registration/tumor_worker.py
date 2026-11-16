#!/usr/bin/env python3
"""Tumor segmentation on an MRI that is registered to the CT: automatic + click-to-edit.

Runs in ../nninteractive/.venv (nnInteractive + nnU-Net v2 + torch; set up by
scripts/segmentation/setup_tumor_seg.sh) and stays running: the launcher's "Register MRI"
tab writes one JSON command per line to stdin and reads ``@progress`` / ``@result`` /
``@error`` lines back, as with the other workers. Models load on first use and stay loaded.

Two models:
* automatic: Kujawa et al. 2024 vestibular schwannoma nnU-Net v2 (Zenodo 10363647,
  CC BY 4.0), scripts/segmentation/vs_seg_run.py; contrast T1 or T2 (CISS/FIESTA) by the
  series name. Good on the CPA/IAC part; it gave ~0 probability to the first patient's
  middle fossa part (an atypical extension), which is what the clicks are for.
* clicks: nnInteractive (DKFZ 2025; weights CC BY-NC-SA 4.0, non-commercial). A positive
  click grows the current part, a negative one (Shift+click) removes. It ignores voxel
  spacing, so a series with thick slices is resampled to 0.5 mm isotropic first
  (C+ T1 0.18x0.18x2.5 mm: one click gave the middle fossa part, 4.2 mL, cleanly).

The tumor series must share the registered MRI's DICOM frame of reference (same session,
e.g. C+ T1 next to the C+ FIESTA that was registered): then mri_to_ct.tfm applies to it.

Commands (one JSON object per line):
  {"cmd": "scan", "folder": F}                        list DICOM series (no patient data)
  {"cmd": "open", "work": W, "folder": F, "uid": U}   series to segment; restores saved parts
  {"cmd": "auto"}                                     automatic model -> a new part
  {"cmd": "click", "lps": [x, y, z], "positive": true} CT-space point -> current part
  {"cmd": "new_part"} / {"cmd": "select", "index": i} / {"cmd": "delete", "index": i}
  {"cmd": "undo"}                                     undo the last change to the current part
  {"cmd": "save"}                                     also write the CT-grid label file
  {"cmd": "quit"}

Outputs in <work>/tumor/: tumor_mri.nii.gz (labels 1..n on the segmentation grid),
tumor_on_ct.nii.gz (labels on the CT grid), preview_tumor.npy (labels on the tab's
preview grid), tumor.json (parts, volumes, series, models).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "segmentation"))

ISO_MM = 0.5  # nnInteractive grid when the series is anisotropic
ISO_IF_ABOVE_MM = 0.6
MIN_PART_ML = 0.02  # automatic: drop specks
UNDO_DEPTH = 20
T2_WORDS = ("fiesta", "ciss", "t2", "drive", "space", "cube", "fse")


def say(kind, payload=""):
    print(f"@{kind} {payload}", flush=True)


def progress(fraction, text):
    say("progress", f"{fraction:.3f} {text}")


def model_for(description: str) -> str:
    """'t2' for CISS / FIESTA / other T2 series, else 't1' (contrast T1)."""
    text = (description or "").lower()
    return "t2" if any(word in text for word in T2_WORDS) else "t1"


def pack(mask):
    return zlib.compress(np.packbits(mask.ravel()).tobytes(), 1), mask.shape


def unpack(blob):
    data, shape = blob
    return np.unpackbits(np.frombuffer(zlib.decompress(data), np.uint8))[: int(np.prod(shape))].reshape(shape)


class TumorWorker:
    def __init__(self):
        import SimpleITK as sitk

        sitk.ProcessObject.SetGlobalWarningDisplay(False)
        self.sitk = sitk
        self.session = None  # nnInteractive, loaded on the first click
        self.grid = None  # the segmentation grid (sitk image, float32)
        self.parts = []  # [{"mask": uint8 array on the grid, "source": "auto"/"clicks", "undo": [...]}]
        self.current = -1
        self.session_part = None  # which part nnInteractive's buffer holds
        self.target = None

    # ------------------------------------------------------------ open

    def open(self, work, folder, uid):
        import mri_ct_worker

        sitk = self.sitk
        self.work = pathlib.Path(work)
        self.out = self.work / "tumor"
        self.out.mkdir(exist_ok=True)
        registration = json.loads((self.work / "registration.json").read_text())
        self.transform = sitk.ReadTransform(str(self.work / "mri_to_ct.tfm"))
        progress(0.2, "Reading the MRI")
        image, self.description = mri_ct_worker.read_image(folder, uid)
        self.series = {"folder": str(folder), "uid": uid, "description": self.description}
        frame = mri_ct_worker.frame_uid(folder, uid)
        registered = registration.get("mri_frame_uid")
        warning = ""
        if registered and frame and frame != registered:
            raise SystemExit("This series is not in the registered MRI's frame of reference (a different "
                             "session?): register it to the CT on its own first.")
        if not registered:
            warning = "Could not check that this series shares the registered MRI's frame of reference."
        self.native = image
        spacing = np.array(image.GetSpacing())
        if spacing.max() > ISO_IF_ABOVE_MM:
            self.grid = mri_ct_worker.isotropic(image, ISO_MM, 0.0, smooth=False)
        else:
            self.grid = image
        self.parts, self.current, self.session_part = [], -1, None
        saved = self.out / "tumor_mri.nii.gz"
        meta = self.out / "tumor.json"
        if saved.is_file() and meta.is_file() and json.loads(meta.read_text()).get("series", {}).get("uid") == uid:
            labels = sitk.GetArrayFromImage(sitk.ReadImage(str(saved)))
            info = json.loads(meta.read_text())
            if labels.shape == sitk.GetArrayViewFromImage(self.grid).shape:
                for n, part in enumerate(info.get("parts", []), start=1):
                    self.parts.append({"mask": (labels == n).astype(np.uint8), "source": part.get("source", "?"),
                                       "undo": []})
                self.current = len(self.parts) - 1
        self._preview_reference(registration)
        if self.session is not None:
            self._set_session_image()
        return self._state(f"Opened {self.description}" + (f" ({warning})" if warning else ""))

    def _preview_reference(self, registration):
        sitk = self.sitk
        shape = np.load(self.work / "preview_ct.npy", mmap_mode="r").shape
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self.work / "ct.nii.gz"))
        reader.ReadImageInformation()
        step = registration["preview_spacing_mm"]
        ref = sitk.Image([int(shape[2]), int(shape[1]), int(shape[0])], sitk.sitkUInt8)
        ref.SetSpacing([step] * 3)
        ref.SetOrigin(registration["preview_origin_mm"])
        ref.SetDirection(reader.GetDirection())
        self.preview_ref = ref
        self.ct_info = reader

    # ------------------------------------------------------------ models

    def _load_session(self):
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        from nnInteractive.model_management import ensure_model_available, get_default_model_id

        progress(0.1, "Loading the click model (first time only)")
        path = ensure_model_available(get_default_model_id())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        session = nnInteractiveInferenceSession(device=device, use_torch_compile=False, verbose=False,
                                                torch_n_threads=os.cpu_count(), do_autozoom=True)
        session.initialize_from_trained_model_folder(str(path))
        self.session = session
        self._set_session_image()

    def _set_session_image(self):
        import torch

        array = self.sitk.GetArrayFromImage(self.grid).astype(np.float32)[None]
        self.session.set_image(array)
        self.target = torch.zeros(array.shape[1:], dtype=torch.uint8)
        self.session.set_target_buffer(self.target)
        self.session_part = None

    def _sync_session(self):
        """Make nnInteractive's buffer hold the current part."""
        if self.session is None:
            self._load_session()
        if self.session_part == self.current:
            return
        self.session.reset_interactions()
        mask = self.parts[self.current]["mask"]
        if mask.any():
            self.session.add_initial_seg_interaction(mask, run_prediction=False)
        else:
            self.target.zero_()
        self.session_part = self.current

    # ------------------------------------------------------------ commands

    def auto(self):
        import vs_seg_run

        sitk = self.sitk
        model = model_for(self.description)
        native = self.out / "mri_native.nii.gz"
        sitk.WriteImage(self.native, str(native), True)
        progress(0.2, f"Automatic model ({'T2' if model == 't2' else 'contrast T1'}) on the GPU")
        vs_seg_run.segment(native, model, self.out / "auto.nii.gz")
        prob = sitk.ReadImage(str(self.out / "auto_prob.nii.gz"))
        on_grid = sitk.GetArrayFromImage(sitk.Resample(prob, self.grid, sitk.Transform(), sitk.sitkLinear, 0.0))
        mask = (on_grid > 0.5).astype(np.uint8)
        mask = self._drop_specks(mask)
        if not mask.any():
            return self._state("The automatic model found no tumor: click on it instead.")
        self.parts.append({"mask": mask, "source": f"auto ({model})", "undo": []})
        self.current = len(self.parts) - 1
        self.session_part = None
        return self._state("Automatic model done", changed=True)

    def _drop_specks(self, mask):
        from scipy import ndimage

        labels, n = ndimage.label(mask)
        if n == 0:
            return mask
        voxel_ml = float(np.prod(self.grid.GetSpacing())) / 1000.0
        sizes = ndimage.sum(mask, labels, range(1, n + 1)) * voxel_ml
        keep = [i + 1 for i, s in enumerate(sizes) if s >= MIN_PART_ML]
        return np.isin(labels, keep).astype(np.uint8)

    def click(self, lps, positive):
        if self.current < 0:
            self.new_part(quiet=True)
        point = self.transform.TransformPoint([float(v) for v in lps])  # CT -> MRI
        index = self.grid.TransformPhysicalPointToContinuousIndex(point)
        size = self.grid.GetSize()
        if not all(-0.5 <= index[a] <= size[a] - 0.5 for a in range(3)):
            return self._state("That point is outside the MRI.")
        part = self.parts[self.current]
        part["undo"] = (part["undo"] + [pack(part["mask"])])[-UNDO_DEPTH:]
        self._sync_session()
        progress(0.5, "Segmenting")
        coordinates = tuple(int(round(index[a])) for a in (2, 1, 0))  # array order
        self.session.add_point_interaction(coordinates, include_interaction=bool(positive))
        part["mask"] = self.target.cpu().numpy().astype(np.uint8).copy()
        if part["source"] == "new":
            part["source"] = "clicks"
        elif "clicks" not in part["source"]:
            part["source"] += " + clicks"
        return self._state("Added" if positive else "Removed", changed=True)

    def new_part(self, quiet=False):
        shape = self.sitk.GetArrayViewFromImage(self.grid).shape
        self.parts.append({"mask": np.zeros(shape, np.uint8), "source": "new", "undo": []})
        self.current = len(self.parts) - 1
        return None if quiet else self._state("New part: click on it")

    def select(self, index):
        if 0 <= index < len(self.parts):
            self.current = index
        return self._state(f"Part {self.current + 1}")

    def delete(self, index):
        if 0 <= index < len(self.parts):
            del self.parts[index]
            self.current = min(self.current, len(self.parts) - 1)
            self.session_part = None
        return self._state("Deleted", changed=True)

    def undo(self):
        if self.current < 0 or not self.parts[self.current]["undo"]:
            return self._state("Nothing to undo")
        part = self.parts[self.current]
        part["mask"] = unpack(part["undo"].pop())
        self.session_part = None  # resync from the restored mask on the next click
        return self._state("Undone", changed=True)

    # ------------------------------------------------------------ outputs

    def _labels(self):
        shape = self.sitk.GetArrayViewFromImage(self.grid).shape
        labels = np.zeros(shape, np.uint8)
        for n, part in enumerate(self.parts, start=1):
            labels[(part["mask"] > 0) & (labels == 0)] = n
        return labels

    def _write(self, ct_grid=False):
        sitk = self.sitk
        labels = self._labels()
        image = sitk.GetImageFromArray(labels)
        image.CopyInformation(self.grid)
        sitk.WriteImage(image, str(self.out / "tumor_mri.nii.gz"), True)
        preview = sitk.Resample(image, self.preview_ref, self.transform, sitk.sitkNearestNeighbor, 0)
        np.save(self.out / "preview_tumor.npy", sitk.GetArrayFromImage(preview).astype(np.uint8))
        if ct_grid:
            progress(0.8, "Writing the tumor on the CT grid")
            ct = sitk.Image(self.ct_info.GetSize(), sitk.sitkUInt8)
            ct.SetSpacing(self.ct_info.GetSpacing())
            ct.SetOrigin(self.ct_info.GetOrigin())
            ct.SetDirection(self.ct_info.GetDirection())
            out = np.zeros(sitk.GetArrayViewFromImage(ct).shape, np.uint8)
            for n in range(1, len(self.parts) + 1):
                # Smooth before resampling so the CT-grid surface is not blocky (thick MRI slices).
                part = sitk.Cast(image == n, sitk.sitkFloat32)
                part = sitk.SmoothingRecursiveGaussian(part, ISO_MM * 0.8)
                on_ct = sitk.GetArrayFromImage(sitk.Resample(part, ct, self.transform, sitk.sitkLinear, 0.0))
                out[(on_ct > 0.5) & (out == 0)] = n
            result = sitk.GetImageFromArray(out)
            result.CopyInformation(ct)
            sitk.WriteImage(result, str(self.out / "tumor_on_ct.nii.gz"), True)
        voxel_ml = float(np.prod(self.grid.GetSpacing())) / 1000.0
        info = {
            "series": self.series,
            "grid_spacing_mm": list(self.grid.GetSpacing()),
            "parts": [{"label": n, "source": p["source"], "ml": float((p["mask"] > 0).sum() * voxel_ml)}
                      for n, p in enumerate(self.parts, start=1)],
            "models": {"auto": "Kujawa et al. 2024 VS nnU-Net v2 (Zenodo 10363647, CC BY 4.0)",
                       "clicks": "nnInteractive v1.0 (DKFZ, CC BY-NC-SA 4.0)"},
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ct_grid_written": ct_grid or (self.out / "tumor_on_ct.nii.gz").is_file(),
        }
        (self.out / "tumor.json").write_text(json.dumps(info, indent=2))
        return info

    def _state(self, message, changed=False, ct_grid=False):
        info = self._write(ct_grid=ct_grid) if (changed or ct_grid) else None
        voxel_ml = float(np.prod(self.grid.GetSpacing())) / 1000.0 if self.grid is not None else 0
        return {
            "message": message,
            "description": getattr(self, "description", ""),
            "parts": [{"source": p["source"], "ml": float((p["mask"] > 0).sum() * voxel_ml),
                       "undo": len(p["undo"])} for p in self.parts],
            "current": self.current,
            "preview": str(self.out / "preview_tumor.npy") if (self.out / "preview_tumor.npy").is_file() else "",
            "ct_file": str(self.out / "tumor_on_ct.nii.gz") if (info or {}).get("ct_grid_written") else "",
        }

    def save(self):
        return self._state("Saved", ct_grid=True)


def main():
    worker = TumorWorker()
    say("ready", "")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
            name = command.pop("cmd")
            if name == "quit":
                break
            if name == "scan":
                import tbone_worker

                result = {"message": "", "series": tbone_worker.scan(pathlib.Path(command["folder"]).expanduser())}
                say("result", json.dumps({"cmd": name, **result}))
                continue
            if name == "open":
                result = worker.open(command["work"], command["folder"], command.get("uid"))
            elif name == "auto":
                result = worker.auto()
                result = worker.save() | {"message": result["message"]}
            elif name == "click":
                result = worker.click(command["lps"], command.get("positive", True))
            elif name == "new_part":
                result = worker.new_part()
            elif name == "select":
                result = worker.select(int(command["index"]))
            elif name == "delete":
                result = worker.delete(int(command["index"]))
            elif name == "undo":
                result = worker.undo()
            elif name == "save":
                result = worker.save()
            else:
                raise ValueError(f"unknown command {name!r}")
            progress(1.0, result["message"])
            say("result", json.dumps({"cmd": name, **result}))
        except SystemExit as exc:
            say("error", json.dumps({"cmd": locals().get("name", "?"), "error": str(exc.code)}))
        except Exception as exc:  # keep running: the tab shows the error
            traceback.print_exc()
            say("error", json.dumps({"cmd": locals().get("name", "?"), "error": f"{type(exc).__name__}: {exc}"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
