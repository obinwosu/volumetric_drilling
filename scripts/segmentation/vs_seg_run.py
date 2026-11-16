#!/usr/bin/env python3
"""Vestibular schwannoma segmentation with the Kujawa et al. 2024 nnU-Net v2 models.

Models: "Deep learning for automatic segmentation of vestibular schwannoma: a retrospective
study from multi-centre routine MRI" (Front. Comput. Neurosci. 2024), Zenodo 10.5281/zenodo.10363647
(CC BY 4.0). The Multi-Centre Routine Clinical set is unpacked in ../vs-seg-kujawa:
  Dataset910_VSMCRCT1  contrast T1; the authors' best is the 3d_fullres + 2d ensemble
  Dataset911_VSMCRCT2  T2 (CISS / FIESTA); 3d_fullres
Zenodo says to skip nnU-Net's post-processing step.

Runs in the tbone-seg venv (nnU-Net v2, torch cu128). PyTorch >= 2.6 loads checkpoints
weights-only; the checkpoints' numpy scalars are allow-listed as in tbone_worker.py.

  vs_seg_run.py --image t1c.nii.gz --model t1 --out tumor_t1.nii.gz
  vs_seg_run.py --image fiesta.nii.gz --model t2 --out tumor_t2.nii.gz
"""
import argparse
import os
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_MODELS = REPO.parent / "vs-seg-kujawa" / "Multi-Centre-Routine-Clinical-(MC-RC)-models"
MODELS = {
    "t1": ("Dataset910_VSMCRCT1", ("3d_fullres", "2d")),
    "t2": ("Dataset911_VSMCRCT2", ("3d_fullres",)),
}


def segment(image: pathlib.Path, model: str, out: pathlib.Path, models_dir=DEFAULT_MODELS, folds=(0, 1, 2, 3, 4)):
    import SimpleITK as sitk
    import torch

    sys.path.insert(0, str(HERE))
    from tbone_worker import allow_checkpoint_numpy

    allow_checkpoint_numpy(torch)
    os.environ.setdefault("nnUNet_raw", tempfile.gettempdir())
    os.environ.setdefault("nnUNet_preprocessed", tempfile.gettempdir())
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dataset, configs = MODELS[model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = []
    for config in configs:
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, device=device,
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(
            str(pathlib.Path(models_dir) / dataset / f"nnUNetTrainer__nnUNetPlans__{config}"), use_folds=folds,
            checkpoint_name="checkpoint_final.pth")
        print(f"{dataset} {config}: predicting", flush=True)
        _, probs = _predict(predictor, image)
        probabilities.append(probs)
    mean = np.mean(probabilities, axis=0)  # the authors' ensemble: average of the two models' softmax
    labels = np.argmax(mean, axis=0).astype(np.uint8)
    reference = sitk.ReadImage(str(image))
    seg = sitk.GetImageFromArray(labels)
    seg.CopyInformation(reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(seg, str(out), True)
    prob = sitk.GetImageFromArray(mean[1].astype(np.float32))
    prob.CopyInformation(reference)
    sitk.WriteImage(prob, str(out).replace(".nii.gz", "_prob.nii.gz"), True)
    voxel = np.prod(reference.GetSpacing())
    print(f"tumor: {int(labels.sum())} voxels, {labels.sum() * voxel / 1000:.2f} mL", flush=True)
    return out


def _predict(predictor, image):
    """(segmentation, probabilities (classes, z, y, x)) for one image, in its own grid."""
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    data, properties = SimpleITKIO().read_images([str(image)])
    seg, probs = predictor.predict_single_npy_array(data, properties, None, None, True)
    return seg, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=pathlib.Path)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS))
    args = parser.parse_args()
    segment(args.image, args.model, args.out, args.models_dir)


if __name__ == "__main__":
    main()
