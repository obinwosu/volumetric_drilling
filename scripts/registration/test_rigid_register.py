"""MRI -> CT registration on a synthetic head (no patient data). Runs in the tbone-seg venv:

../tbone-seg-nnunetv2/.venv/bin/python scripts/registration/test_rigid_register.py

The phantom: an ellipsoid skull, two dense petrous blocks each holding a fluid-filled
"labyrinth" (a ring and two small balls) and an "IAC" tube, mastoid air cells. The CT
shows bone bright and fluid / air dark; the MRI shows fluid bright, bone and air dark,
brain mid-gray. The MRI is a thinner slab, tilted 35 degrees nose-down, turned and
moved 200 mm away, as the first real pair was. The registration must put the two
labyrinths back within 1 mm. On a GPU ~10 s; the CPU grid is coarser and slower.

Also checks: the rotation convention against ITK, and the ITK transform the worker
writes (CT point -> MRI point).

The real pair (bilateral dehiscence CT + C+ FIESTA, not in the repo) gave pitch -41.3,
roll +1.0, yaw +0.4 deg; the same MRI blurred to 0.9 x 0.9 x 3 mm (T1-like) and the CT
cut to one ear both came back within 0.35 mm of it.
"""
import pathlib
import sys
import tempfile

import numpy as np
import SimpleITK as sitk

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "segmentation"))

import mri_ct_worker as worker  # noqa: E402
import rigid_register as rr  # noqa: E402

EARS = (np.array([-38.0, -5.0, 0.0]), np.array([38.0, -5.0, 0.0]))  # phantom labyrinths, LPS mm


def phantom(spacing, size_mm, kind, transform=None):
    """kind 'ct' (HU) or 'mri' (0..1000). transform: CT point -> MRI point (the image is drawn
    in MRI space by mapping each MRI voxel back into the phantom)."""
    n = np.round(np.array(size_mm) / np.array(spacing)).astype(int)
    origin = -np.array(size_mm) / 2.0
    k, j, i = np.meshgrid(*[np.arange(v) for v in n[::-1]], indexing="ij")
    p = np.stack([i, j, k], -1) * np.array(spacing) + origin
    if transform is not None:  # MRI voxel -> phantom (CT) point
        inverse = transform.GetInverse()
        flat = p.reshape(-1, 3)
        m = np.array(inverse.GetMatrix()).reshape(3, 3)
        offset = np.array(inverse.TransformPoint((0.0, 0.0, 0.0)))
        p = (flat @ m.T + offset).reshape(p.shape)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.sqrt((x / 75) ** 2 + (y / 92) ** 2 + (z / 70) ** 2)
    head, brain = r < 1.0, r < 0.9
    bone = head & ~brain
    air = np.zeros_like(head)
    fluid = np.zeros_like(head)
    for side, ear in zip((-1, 1), EARS):
        d = p - ear
        bone |= (np.abs(d[..., 0]) < 14) & (np.abs(d[..., 1]) < 9) & (np.abs(d[..., 2]) < 8)  # petrous block
        ring = np.abs(np.sqrt(d[..., 0] ** 2 + d[..., 2] ** 2) - 3.5) < 0.8
        fluid |= ring & (np.abs(d[..., 1] - 2) < 0.8)  # a "semicircular canal"
        fluid |= np.linalg.norm(d - [0, -4, 0], axis=-1) < 2.0  # "cochlea"
        fluid |= np.linalg.norm(d - [side * 3, 0, -2], axis=-1) < 1.5  # "vestibule"
        iac = (side * d[..., 0] < -2) & (side * d[..., 0] > -14) & (d[..., 1] ** 2 + d[..., 2] ** 2 < 2.2 ** 2)
        fluid |= iac
        cells = (np.sin(d[..., 0] * 1.3) * np.sin(d[..., 1] * 1.1) * np.sin(d[..., 2] * 1.7) > 0.4)
        air |= cells & (np.linalg.norm(d - [side * 22, 10, 0], axis=-1) < 9)  # mastoid air cells
    if kind == "ct":
        image = np.where(head, 40.0, -1000.0)
        image[bone] = 1600.0
        image[fluid] = 10.0
        image[air] = -900.0
        image += np.random.default_rng(1).normal(0, 60, image.shape)
    else:
        image = np.where(head, 300.0, 0.0)
        image[brain] = 380.0
        image[bone] = 40.0
        image[air] = 0.0
        image[fluid] = 1000.0
        image += np.random.default_rng(2).normal(0, 15, image.shape)
    out = sitk.GetImageFromArray(image.astype(np.float32))
    out.SetSpacing([float(s) for s in spacing])
    out.SetOrigin([float(v) for v in origin])
    return out


def test_rotation_matches_itk():
    angles = [-35.0, 3.0, 5.0]
    ours = rr.rotation(angles)
    assert np.allclose(ours @ ours.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(ours), 1.0)
    # pitch alone turns +y (posterior) toward +z (superior) for positive angles, about x
    assert np.allclose(rr.rotation([90.0, 0, 0]) @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    t = worker.to_transform(np.r_[angles, 1.0, 2.0, 3.0], np.array([10.0, 20.0, 30.0]))
    p = np.array([5.0, -7.0, 11.0])
    expected = ours @ (p - [10, 20, 30]) + [10, 20, 30] + [1, 2, 3]
    assert np.allclose(t.TransformPoint(p.tolist()), expected, atol=1e-9)


def test_registers_tilted_slab():
    ct = phantom((0.5, 0.5, 0.5), (150, 150, 50), "ct")
    # Drawn with CT point p -> R p + (4, 0, 0); then the MRI's grid is moved by (0, 210, -60), as if
    # the scanner's coordinates were somewhere else entirely.
    t_drawn = sitk.Euler3DTransform()
    t_drawn.SetMatrix([float(v) for v in rr.rotation([-35.0, 3.0, 5.0]).ravel()], 1e-6)
    t_drawn.SetTranslation((4.0, 0.0, 0.0))
    mri = phantom((0.6, 0.6, 0.8), (170, 190, 60), "mri", t_drawn)
    mri.SetOrigin([v + o for v, o in zip(mri.GetOrigin(), (0.0, 210.0, -60.0))])
    transform, record = worker.register(ct, mri, log=lambda *_: None)
    for ear in EARS:
        found = np.array(transform.TransformPoint(ear.tolist()))
        expected = np.array(t_drawn.TransformPoint(ear.tolist())) + [0.0, 210.0, -60.0]
        error = np.linalg.norm(found - expected)
        assert error < 1.0, (error, record["pitch_roll_yaw_deg"], record["translation_mm"])
    assert abs(record["pitch_roll_yaw_deg"][0] + 35.0) < 1.5, record["pitch_roll_yaw_deg"]
    return transform, record, ct, mri


def test_outputs_and_preview():
    transform, record, ct, mri = test_registers_tilted_slab()
    with tempfile.TemporaryDirectory() as folder:
        work = pathlib.Path(folder)
        result = worker.write_outputs(work, ct, mri, transform, record, ("ct", "mri"))
        for name in ("ct.nii.gz", "mri.nii.gz", "mri_to_ct.tfm", "mri_on_ct.nii.gz", "preview_ct.npy",
                     "preview_mri.npy"):
            assert (work / name).is_file(), name
        back = sitk.ReadTransform(str(work / "mri_to_ct.tfm"))
        p = EARS[0].tolist()
        assert np.allclose(back.TransformPoint(p), transform.TransformPoint(p), atol=1e-6)
        on_ct = sitk.GetArrayFromImage(sitk.ReadImage(str(work / "mri_on_ct.nii.gz")))
        ct_array = sitk.GetArrayFromImage(ct)
        # fluid is bright on the resampled MRI where the CT has fluid inside bone
        index = ct.TransformPhysicalPointToIndex((EARS[0] + [0, -4, 0]).tolist())[::-1]
        assert on_ct[index] > 600 and ct_array[index] < 400, (on_ct[index], ct_array[index])
        preview_mri = np.load(work / "preview_mri.npy")
        assert preview_mri.shape == np.load(work / "preview_ct.npy").shape
        assert (preview_mri == 0).any() and (preview_mri > 0).any()  # the thin MRI slab does not cover all
        assert result["preview_spacing_mm"] > 0


if __name__ == "__main__":
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    test_rotation_matches_itk()
    print("ok test_rotation_matches_itk")
    test_outputs_and_preview()
    print("ok test_registers_tilted_slab + test_outputs_and_preview")
