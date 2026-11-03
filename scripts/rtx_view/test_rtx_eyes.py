"""Headset eyes: microscope stereo geometry and the shared-memory images."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rtx_eyes


def test_eyes_sit_a_base_apart_and_aim_at_the_focus():
    position = np.array([1.0, 2.0, 3.0])
    right = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    forward = np.array([1.0, 0.0, 0.0])
    base, focus = 0.23, 2.5
    left_mat, right_mat = rtx_eyes.eye_matrices(position, right, up, forward, base, focus)
    if not np.isclose(np.linalg.norm(right_mat[3, :3] - left_mat[3, :3]), base):
        raise SystemExit("the eyes should sit one stereo base apart")
    if float((right_mat[3, :3] - left_mat[3, :3]) @ right) <= 0.0:
        raise SystemExit("the right eye should be to the scope's right")
    target = position + forward * focus
    for mat in (left_mat, right_mat):
        look = -mat[2, :3]
        to_focus = (target - mat[3, :3]) / np.linalg.norm(target - mat[3, :3])
        if float(look @ to_focus) < 1.0 - 1e-9:
            raise SystemExit("each eye should look at the focus point")
        frame = mat[:3, :3]
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-9):
            raise SystemExit("eye frames should stay orthonormal")
        if float(mat[1, :3] @ up) < 0.99:
            raise SystemExit("eyes should keep the scope's up")


def test_shared_images_round_trip_bottom_row_first():
    path = pathlib.Path("/dev/shm/fivrs_rtx_eyes_unittest")
    share = rtx_eyes.EyeShare(path)
    try:
        height, width = 6, 8
        left = np.zeros((height, width, 4), np.uint8)
        right = np.zeros((height, width, 4), np.uint8)
        left[0] = 200  # top row
        right[-1] = 50  # bottom row
        share.write(left, right, 0.18)
        seq, got_left, got_right = rtx_eyes.read(path)
        if seq != 2:
            raise SystemExit(f"a complete frame should leave an even sequence, got {seq}")
        if not (got_left[-1] == 200).all() or not (got_right[0] == 50).all():
            raise SystemExit("images should be stored bottom row first, as OpenGL reads them")
        share.write(np.zeros((12, 10, 4), np.uint8), np.zeros((12, 10, 4), np.uint8))
        seq, got_left, _ = rtx_eyes.read(path)
        if got_left.shape != (12, 10, 4) or seq != 2:
            raise SystemExit("a new window size should start a fresh file")
    finally:
        share.close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_eyes_sit_a_base_apart_and_aim_at_the_focus()
    test_shared_images_round_trip_bottom_row_first()
    print("ok")
