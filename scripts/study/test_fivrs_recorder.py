"""Recorder output opens in the Mastoid feedback system's reader.

Run with the feedback environment (it has h5py and the reader):
    feedback_system/.venv/bin/python scripts/study/test_fivrs_recorder.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "rtx_view"))
sys.path.insert(0, str(REPO / "feedback_system" / "src"))

import fivrs_recorder  # noqa: E402
import rtx_record  # noqa: E402

VOLUME_ADF = REPO / "ADF" / "volume_256_RT143.yaml"
NRRD = REPO / "resources" / "volumes" / "nrrd_header.pkl"


def _drive(recorder, seconds=4.0, frame_hz=10.0, pose_hz=100.0, t0=1_800_000_000.0):
    recorder.on_volume_info([1.0, 1.0, 1.0], [256, 256, 256])
    recorder.on_burr(t0, 6)
    steps = int(seconds * pose_hz)
    every = int(pose_hz / frame_hz)
    for k in range(steps):
        t = t0 + k / pose_hz
        drill = (-5.9 + 0.001 * k, 0.01, 0.02, 0.0, 0.0, 0.0, 1.0)
        recorder.on_pose("mastoidectomy_drill", t, drill)
        recorder.on_pose("main_camera", t, (1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))
        recorder.on_force(t, (0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
        if k % 7 == 0:
            recorder.on_voxels(t, [(100, 120, 130 + k % 5)], [(0.9, 0.9, 0.8, 1.0)])
        if k % every == 0:
            image = np.full((360, 640, 3), k % 255, np.uint8)
            recorder.on_frame(t, (1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), 0.1795, image)


def test_share_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "share"
        writer = rtx_record.RecordShare(path, slots=4)
        reader = rtx_record.RecordReader(path)
        image = np.zeros((36, 64, 3), np.uint8)
        writer.write(image, 1.0, np.zeros(7), 0.2)  # before the reader attached: an earlier recording
        assert reader.poll() == []
        for k in range(3):
            image[:] = k
            writer.write(image, 2.0 + k, np.arange(7, dtype=float), 0.2)
        frames = reader.poll()
        assert [f[0] for f in frames] == [2.0, 3.0, 4.0]
        assert frames[2][3][0, 0, 0] == 2
        for k in range(10):  # more than the ring holds
            writer.write(image, 10.0 + k, np.zeros(7), 0.2)
        frames = reader.poll()
        assert frames[-1][0] == 19.0 and reader.dropped > 0
        writer.close()
        reader.close()


def test_rgba_to_recorded_bgr():
    rgba = np.zeros((720, 1280, 4), np.uint8)
    rgba[..., 0] = 200  # red
    bgr = rtx_record.to_record_image(rgba)
    assert bgr.shape == (360, 640, 3)
    assert bgr[0, 0, 2] == 200 and bgr[0, 0, 0] == 0


def test_chunks_match_the_v3_layout_and_open_in_the_reader():
    from evidence_system.hdf5_reader import RecordingReader

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "recording"
        recorder = fivrs_recorder.Recorder(out, VOLUME_ADF, NRRD, chunk_frames=15)
        _drive(recorder)
        recorder.close()
        files = sorted(out.glob("*.hdf5"))
        assert len(files) == 3, files  # 40 frames at 15 per file
        with h5py.File(files[0]) as f:
            cont = f["vision_data/continuous_data"]
            assert cont["l_img"].shape == (15, 360, 640, 3)
            assert cont["pose_mastoidectomy_volume"].shape == (15, 7)
            assert abs(f["metadata_v3/voxel_volume"][()] - (1000 / 256) ** 3 * 0.049202688 ** 3) < 1e-9
            assert f["metadata_v3/voxel_volume"].attrs["units"] == "mm^3, millimeters cubed"
            readme = f["metadata_v3/README"][()].decode()
            # The feedback system's volume check requires these conventions to be stated.
            for fragment in ("All position information is in meters", "[qx, qy, qz, qw]",
                             "T_world_obj", "Depth in CV convention", "T_cv_ambf"):
                assert fragment in readme, fragment
            assert "vision_data/continuous_data/depth" not in f
            removed = f["physics_data/intermittent_data/voxels_removed/voxel_removed"][()]
            stamps = f["physics_data/intermittent_data/voxels_removed/voxel_time_stamp"][()]
            assert removed[:, 0].max() == len(stamps) - 1  # first column indexes this file's batches
            drill = f["physics_data/continuous_data/high_frequency_poses/mastoidectomy_drill_pose/pose"][()]
            assert abs(drill[0, 0] - (-5.9 * 0.049202688)) < 1e-9  # meters
        reader = RecordingReader(out)
        assert reader.schema == "v3"
        manifest = reader.manifest()
        assert manifest, "reader produced no manifest"


if __name__ == "__main__":
    test_share_round_trip()
    test_rgba_to_recorded_bgr()
    test_chunks_match_the_v3_layout_and_open_in_the_reader()
    print("ok")
