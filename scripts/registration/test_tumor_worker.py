"""Tumor worker on a synthetic scan (no patient data). Runs in the nnInteractive venv, GPU:

../nninteractive/.venv/bin/python scripts/registration/test_tumor_worker.py

A bright 10 mm ball in a dim head-sized block stands in for an enhancing tumor; the
registration is the identity. One click in the ball must segment about the ball (nnInteractive),
Undo must bring the part back to empty, a negative click must shrink it, and Save must write
the labels on the CT grid. The automatic model (VS nnU-Net) is checked on the real patient
only; see CLAUDE.md.
"""
import json
import pathlib
import sys
import tempfile

import numpy as np
import SimpleITK as sitk

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tumor_worker as tw  # noqa: E402

BALL_MM = 5.0


def scan(spacing, size_mm, value_inside, value_outside, ball=True):
    n = np.round(np.array(size_mm) / np.array(spacing)).astype(int)
    origin = -np.array(size_mm) / 2.0
    k, j, i = np.meshgrid(*[np.arange(v) for v in n[::-1]], indexing="ij")
    p = np.stack([i, j, k], -1) * np.array(spacing) + origin + np.array(spacing) / 2
    r = np.linalg.norm(p - [6.0, -4.0, 2.0], axis=-1)
    image = np.full(r.shape, value_outside, np.float32)
    if ball:
        image[r < BALL_MM] = value_inside
    image += np.random.default_rng(0).normal(0, value_inside * 0.03 if ball else 20, image.shape).astype(np.float32)
    out = sitk.GetImageFromArray(image)
    out.SetSpacing([float(s) for s in spacing])
    out.SetOrigin([float(v) for v in origin])
    return out


def make_work(folder: pathlib.Path):
    ct = scan((0.5, 0.5, 0.5), (60, 60, 40), 0, 40, ball=False)
    sitk.WriteImage(sitk.Cast(ct, sitk.sitkInt16), str(folder / "ct.nii.gz"))
    sitk.WriteTransform(sitk.Euler3DTransform(), str(folder / "mri_to_ct.tfm"))
    step = 1.0
    np.save(folder / "preview_ct.npy", np.zeros((40, 60, 60), np.int16))
    (folder / "registration.json").write_text(json.dumps({
        "preview_spacing_mm": step, "preview_origin_mm": [-29.5, -29.5, -19.5],
        "preview_direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]}))
    # Thick slices like the C+ T1 (0.3 x 0.3 x 2.5 mm): the worker resamples to 0.5 mm isotropic.
    mri = scan((0.3, 0.3, 2.5), (60, 60, 40), 1500, 300)
    sitk.WriteImage(mri, str(folder / "t1c.nii.gz"))
    return folder / "t1c.nii.gz"


def test_helpers():
    assert tw.model_for("C+ AX 3D FIESTA") == "t2" and tw.model_for("AX CISS") == "t2"
    assert tw.model_for("C+ AX T1 2MM") == "t1" and tw.model_for("T1 POST GAD") == "t1"
    mask = (np.random.default_rng(1).random((7, 9, 11)) > 0.5).astype(np.uint8)
    assert np.array_equal(tw.unpack(tw.pack(mask)), mask)


def test_click_undo_save():
    expected_ml = 4.0 / 3.0 * np.pi * BALL_MM ** 3 / 1000.0
    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        image = make_work(folder)
        worker = tw.TumorWorker()
        state = worker.open(str(folder), str(image), None)
        assert state["parts"] == [] and np.allclose(worker.grid.GetSpacing(), tw.ISO_MM)
        worker.new_part()
        state = worker.click([6.0, -4.0, 2.0], True)
        ml = state["parts"][0]["ml"]
        assert 0.6 * expected_ml < ml < 1.5 * expected_ml, (ml, expected_ml)
        preview = np.load(state["preview"])
        assert preview.shape == (40, 60, 60) and preview.max() == 1
        assert worker.undo()["parts"][0]["ml"] == 0.0
        worker.click([6.0, -4.0, 2.0], True)
        smaller = worker.click([6.0, -4.0, 6.0], False)["parts"][0]["ml"]  # remove the top
        assert smaller <= ml + 1e-6, (smaller, ml)
        state = worker.save()
        on_ct = sitk.GetArrayFromImage(sitk.ReadImage(state["ct_file"]))
        ct_ml = (on_ct == 1).sum() * 0.125 / 1000.0
        assert abs(ct_ml - smaller) < 0.25 * expected_ml, (ct_ml, smaller)
        info = json.loads((folder / "tumor" / "tumor.json").read_text())
        assert info["parts"][0]["source"] == "clicks"
        # reopening restores the saved part
        again = tw.TumorWorker()
        assert len(again.open(str(folder), str(image), None)["parts"]) == 1
        # outside the MRI: refused, nothing changes
        assert "outside" in worker.click([500.0, 0.0, 0.0], True)["message"]


if __name__ == "__main__":
    test_helpers()
    print("ok test_helpers")
    test_click_undo_save()
    print("ok test_click_undo_save")
