"""Tests for the "Register MRI" tab (no GPU, no scans, no worker process).

QT_QPA_PLATFORM=offscreen python3 scripts/study_gui/test_register_mri_tab.py

The registration itself: scripts/registration/test_rigid_register.py (tbone-seg venv).
"""
import json
import os
import pathlib
import stat
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PyQt5 import QtWidgets  # noqa: E402

import register_mri_tab as tab  # noqa: E402
import tumor_panel  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def fake_result(folder: pathlib.Path, **extra):
    ct = np.full((40, 60, 50), -1000, np.int16)
    ct[10:30, 10:50, 10:40] = 1500
    mri = np.zeros((40, 60, 50), np.uint8)
    mri[5:35] = 128  # a slab: 0 above and below = no MRI
    np.save(folder / "preview_ct.npy", ct)
    np.save(folder / "preview_mri.npy", mri)
    result = {"work": str(folder), "preview_spacing_mm": 1.0, "preview_origin_mm": [-25.0, -30.0, -20.0],
              "temporal_bones_mm": [[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], "pitch_roll_yaw_deg": [-41.3, 1.0, 0.4],
              "score_1mm": 0.46, "runner_up_score_1mm": 0.25, "starts_agreeing": 3, "starts_tried": 24,
              "seconds": 30.0, "device": "cuda", **extra}
    (folder / "registration.json").write_text(json.dumps(result))
    return result


def test_confidence():
    sure, text = tab.confidence({"score_1mm": 0.46, "runner_up_score_1mm": 0.25, "starts_agreeing": 3,
                                 "starts_tried": 24})
    assert sure and "3 of 24" in text and "54%" in text
    assert not tab.confidence({"score_1mm": 0.30, "runner_up_score_1mm": 0.29, "starts_agreeing": 5})[0]
    assert not tab.confidence({"score_1mm": 0.46, "runner_up_score_1mm": 0.10, "starts_agreeing": 1})[0]
    assert tab.confidence({"score_1mm": 0.46, "runner_up_score_1mm": None, "starts_agreeing": 4})[0]


def test_fuse_modes():
    ct = np.array([[-1000, 2000], [500, 500]], np.int16)
    mri = np.array([[200, 0], [255, 1]], np.uint8)
    for mode in ("blend", "checker", "color"):
        rgb = tab.fuse(ct, mri, mode, 0.5, 1.0)
        assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8, mode
    only_ct = tab.fuse(ct, mri, "blend", 0.0, 1.0)
    assert only_ct[0, 1].tolist() == [255, 255, 255] and only_ct[0, 0].tolist() == [0, 0, 0]
    # where there is no MRI (0), the checkerboard shows the CT
    assert tab.fuse(ct, mri, "checker", 0.5, 100.0)[0, 1].tolist() == [255, 255, 255]


def test_find_slicer():
    with tempfile.TemporaryDirectory() as folder:
        launcher = pathlib.Path(folder) / "Slicer"
        launcher.write_text("#!/bin/sh\n")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
        assert tab.find_slicer(str(launcher)) == str(launcher)
        launcher.chmod(stat.S_IRUSR | stat.S_IWUSR)  # not executable: not used
        assert tab.find_slicer(str(launcher)) != str(launcher)


def test_one_folder_fills_both_lists():
    panel = tab.RegisterMriPanel(None, tbone_dir="/nonexistent")
    series = [{"uid": "1", "folder": "/a", "slices": 261, "modality": "CT", "description": "Bone 0.4"},
              {"uid": "2", "folder": "/a", "slices": 40, "modality": "CT", "description": "Scout"},
              {"uid": "3", "folder": "/a", "slices": 152, "modality": "MR", "description": "AX FIESTA"}]
    panel._scanned("ct", "/a", series)
    assert [s["uid"] for s in panel.series["ct"]] == ["1", "2"]  # most slices first
    assert [s["uid"] for s in panel.series["mri"]] == ["3"]
    assert panel._chosen("ct")["uid"] == "1" and panel._chosen("mri")["uid"] == "3"
    assert panel.folder_labels["mri"].text() == "/a"
    # a later MRI-only folder replaces just the MRI list
    panel._scanned("mri", "/b", [{"uid": "9", "folder": "/b", "slices": 30, "modality": "MR", "description": "T1"}])
    assert panel._chosen("mri")["uid"] == "9" and panel._chosen("ct")["uid"] == "1"


def test_show_result_and_navigate():
    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        panel = tab.RegisterMriPanel(None, tbone_dir="/nonexistent")
        panel.resize(1200, 800)
        panel.show_result(fake_result(folder))
        assert "Looks right" in panel.result_label.text() and "-41.3" in panel.result_label.text()
        assert panel.cursor == [20, 30, 15]  # right temporal bone: (-10, 0, 0) mm from origin (-25, -30, -20)
        panel._go_to_ear(1)
        assert panel.cursor == [20, 30, 35]
        panel._move(0, 100)
        assert panel.cursor[0] == 39  # clamped to the last slice
        panel._clicked("axial", 5.0, 7.0)
        assert panel.cursor[1:] == [7, 5]
        for index in range(panel.mode_combo.count()):
            panel.mode_combo.setCurrentIndex(index)
        assert panel.axial.image is not None and panel.sagittal.image is not None
        assert panel.folder_button.isEnabled() and panel.reload_button.isEnabled()
        unsure = fake_result(folder, runner_up_score_1mm=0.45, transform_edited="2026-11-15 20:00")
        panel.show_result(unsure)
        assert "Not sure" in panel.result_label.text() and "edited" in panel.result_label.text()


def test_slicer_setting_saved():
    saved = {}
    panel = tab.RegisterMriPanel(None, tbone_dir="/nonexistent", save=lambda k, v: saved.__setitem__(k, v))
    with tempfile.TemporaryDirectory() as folder:
        launcher = pathlib.Path(folder) / "Slicer"
        launcher.write_text("#!/bin/sh\n")
        launcher.chmod(0o755)
        original = QtWidgets.QFileDialog.getOpenFileName
        QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(launcher), ""))
        try:
            panel._choose_slicer()
        finally:
            QtWidgets.QFileDialog.getOpenFileName = original
        assert saved == {tab.SLICER_SETTING: str(launcher)} and panel._slicer() == str(launcher)


def test_fuse_tints_tumor_parts():
    ct = np.zeros((4, 4), np.int16)
    mri = np.zeros((4, 4), np.uint8)
    labels = np.zeros((4, 4), np.uint8)
    labels[0, 0], labels[1, 1] = 1, 2
    rgb = tab.fuse(ct, mri, "blend", 0.0, 1.0, labels)
    assert rgb[0, 0, 0] > rgb[0, 0, 2] and rgb[1, 1, 1] > rgb[1, 1, 2] and rgb[2, 2].tolist() == rgb[3, 3].tolist()
    assert rgb[0, 0].tolist() != rgb[1, 1].tolist()  # parts get different colours


def test_point_index_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        panel = tab.RegisterMriPanel(None, tbone_dir="/nonexistent", tumor_python="/nonexistent")
        result = fake_result(pathlib.Path(folder), preview_direction=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        panel.show_result(result)
        point = panel._point_of([20, 30, 15])
        assert np.allclose(point, [-10.0, 0.0, 0.0]) and panel._index_of(point) == [20, 30, 15]


def test_tumor_series_order_and_clicks():
    series = [{"uid": "f", "modality": "MR", "description": "C+ AX 3D FIESTA", "slices": 152, "folder": "/f"},
              {"uid": "t", "modality": "MR", "description": "C+ AX T1 2MM", "slices": 25, "folder": "/t"},
              {"uid": "x", "modality": "MR", "description": "AX T2 FLAIR", "slices": 30, "folder": "/x"},
              {"uid": "c", "modality": "CT", "description": "Bone", "slices": 261, "folder": "/c"}]
    assert [s["uid"] for s in tumor_panel.tumor_series_order(series)] == ["t", "f", "x"]
    with tempfile.TemporaryDirectory() as folder:
        panel = tab.RegisterMriPanel(None, tbone_dir="/nonexistent", tumor_python="/nonexistent")
        panel.show_result(fake_result(pathlib.Path(folder)))
        panel.tumor.offer_series(series)
        assert panel.tumor.series_combo.count() == 3 and "Not installed" in panel.tumor.status.text()
        assert not panel.tumor.auto_button.isEnabled()  # nothing opened, worker not installed
        assert panel.tumor.click([0, 0, 0], True) is False  # click mode off: the view just moves
        # a saved tumor preview is drawn over the views
        tumor_dir = pathlib.Path(folder) / "tumor"
        tumor_dir.mkdir()
        labels = np.zeros((40, 60, 50), np.uint8)
        labels[20, 30, 15] = 1
        np.save(tumor_dir / "preview_tumor.npy", labels)
        panel.tumor.set_work(folder + "/")  # a different string: reloads
        assert panel.tumor.preview is not None and panel.tumor.preview.max() == 1


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
