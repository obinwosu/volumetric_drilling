"""Named views round-trip through the same pose the tuner camera uses."""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import camera_views


def test_saved_right_view_is_the_tuner_home_offset():
    views = camera_views.load_views()
    right = next(view for view in views if view["id"] == "right")
    position, right_axis, up, forward = camera_views.pose_from_view(right)
    translation, quat = camera_views.view_offset(position, right_axis, up, forward)
    assert np.allclose(translation, right["t"], atol=1e-5)
    stored = np.asarray(right["q"], dtype=np.float64)
    if float(np.dot(quat, stored)) < 0.0:
        quat = -quat
    assert np.allclose(quat, stored, atol=1e-5)
    assert np.allclose(position, camera_views.HOME_POSITION + np.asarray(right["t"]), atol=1e-6)


def test_empty_slot_stays_empty_until_marked_valid():
    views = camera_views.load_views()
    left_eac = next(view for view in views if view["id"] == "left_eac")
    assert left_eac["valid"] is False
    assert left_eac["label"] == "Left EAC"


def test_new_names_get_their_own_ids():
    assert camera_views.unique_id("Right attic", []) == "right_attic"
    assert camera_views.unique_id("Left transtemporal", ["left_transtemporal"]) == "left_transtemporal_2"
    assert camera_views.unique_id("   ", ["view"]) == "view_2"


def test_save_rewrites_every_slot():
    views = camera_views.load_views()
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "canonical_views.yaml"
        camera_views.save_views(views, path)
        loaded = camera_views.load_views(path)
    assert [view["id"] for view in loaded] == [view["id"] for view in views]
    for original, saved in zip(views, loaded):
        assert saved["label"] == original["label"]
        assert saved["laterality"] == original["laterality"]
        assert saved["valid"] is original["valid"]
        assert np.allclose(saved["t"], original["t"], atol=1e-6)
        assert np.allclose(saved["q"], original["q"], atol=1e-6)


if __name__ == "__main__":
    test_saved_right_view_is_the_tuner_home_offset()
    test_empty_slot_stays_empty_until_marked_valid()
    test_new_names_get_their_own_ids()
    test_save_rewrites_every_slot()
    print("camera view tests passed")
