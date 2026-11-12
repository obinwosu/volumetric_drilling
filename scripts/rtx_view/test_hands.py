"""Quest hand joints: file format, the Touch fit, and placement in the scene."""

from __future__ import annotations

import os
import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hands

PLUGIN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "plugin"
    / "ambf_openxr_eyepiece"
    / "plugins"
    / "object"
    / "ambf_openxr_eyepiece.cpp"
)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return hands.quat_to_mat((axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)))


def _random_rotation(rng, max_angle=np.pi):
    return _rotation(rng.normal(size=3), rng.uniform(-max_angle, max_angle))


def _fake_hand(rng, center):
    """26 joints around a center, wrist behind, fingers ahead."""
    pos = center + rng.normal(scale=0.03, size=(hands.JOINTS, 3))
    pos[hands.WRIST] = center + (0.0, 0.0, 0.05)
    pos[hands.MIDDLE_PROXIMAL] = center + (0.0, 0.0, -0.03)
    pos[hands.INDEX_PROXIMAL] = center + (0.03, 0.0, -0.03)
    pos[hands.LITTLE_PROXIMAL] = center + (-0.03, 0.0, -0.025)
    return pos


def _buffer(seq, wall, head_pose, hand_pos, session=7):
    """Build the file exactly as the plugin lays it out."""
    out = bytearray(hands.FILE_BYTES)
    struct.pack_into("<IIQdII", out, 0, hands.MAGIC, 1, seq, wall, 0x3 if head_pose is not None else 0, session)
    if head_pose is not None:
        struct.pack_into("<7f", out, 32, *head_pose)
    for h, pos in enumerate(hand_pos):
        base = hands.HEADER + h * hands.HAND_BYTES
        if pos is None:
            continue
        struct.pack_into("<I", out, base, 1)
        joints = np.zeros((hands.JOINTS, 8), dtype="<f4")
        joints[:, :3] = pos
        joints[:, 6] = 1.0
        joints[:, 7] = 0.008
        out[base + 8 : base + 8 + joints.nbytes] = joints.tobytes()
        flags = np.full(hands.JOINTS, 0xF, dtype="<u4")
        offset = base + 8 + hands.JOINTS * 32
        out[offset : offset + flags.nbytes] = flags.tobytes()
    return bytes(out)


def _state(tip, r_cam, r_stylus, stylus, camera=(0.0, 0.0, 3.0)):
    look = -r_cam[:, 0]
    up = r_cam[:, 2]
    q = hands.mat_to_quat(r_cam @ r_stylus)
    return {
        "px": tip[0], "py": tip[1], "pz": tip[2],
        "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
        "lx": look[0], "ly": look[1], "lz": look[2],
        "ux": up[0], "uy": up[1], "uz": up[2],
        "cx": camera[0], "cy": camera[1], "cz": camera[2],
        "hx": stylus[0], "hy": stylus[1], "hz": stylus[2],
    }


def test_layout_matches_the_plugin():
    assert hands.HAND_BYTES == 944 and hands.FILE_BYTES == 64 + 2 * 944
    text = PLUGIN.read_text()
    assert "64 + hand * 944" in text
    assert '"/dev/shm/fivrs_hands"' in text and "0x44485646u" in text


def test_quaternion_helpers():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = _random_rotation(rng)
        assert np.allclose(hands.quat_to_mat(hands.mat_to_quat(r)), r, atol=1e-9)
    directions = np.vstack((rng.normal(size=(40, 3)), [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]))
    for d, q in zip(directions, hands.quats_from_x(directions)):
        turned = hands.quat_to_mat(q) @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(turned, d / np.linalg.norm(d), atol=1e-9)


def test_file_round_trip_and_torn_frames():
    rng = np.random.default_rng(2)
    left = _fake_hand(rng, np.array([-0.1, -0.3, -0.4]))
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "fivrs_hands"
        path.write_bytes(_buffer(4, 123.5, (0, 1.2, 0, 0, 0, 0, 1), [left, None]))
        reader = hands.HandsFile(path)
        frame = reader.read()
        assert frame is not None and frame.wall == 123.5 and frame.session == 7
        assert np.allclose(frame.hands[0]["pos"], left, atol=1e-6)
        assert frame.hands[1] is None
        assert np.allclose(frame.head_pos, (0, 1.2, 0), atol=1e-6)
        # Odd sequence: the plugin is mid-write.
        path.write_bytes(_buffer(5, 124.0, None, [left, None]))
        reader.close()
        assert hands.HandsFile(path).read() is None
        # A replaced file (new inode) is picked up.
        replacement = pathlib.Path(folder) / "next"
        replacement.write_bytes(_buffer(2, 200.0, None, [None, left]))
        os.replace(replacement, path)
        reader._checked = -1.0
        frame = reader.read()
        assert frame is not None and frame.wall == 200.0 and frame.hands[1] is not None


def test_fit_recovers_the_headset_to_touch_transform():
    rng = np.random.default_rng(3)
    a_true = _random_rotation(rng)
    b_true = np.array([0.3, -0.5, 0.9])
    offset_true = np.array([0.055, 0.004, -0.008])
    stylus = rng.uniform(-0.06, 0.06, size=(200, 3)) + (-0.02, 0.0, 0.05)
    rotations = np.array([_random_rotation(rng, 0.7) for _ in range(200)])
    device_pinch = stylus + np.einsum("nij,j->ni", rotations, offset_true)
    pinch_xr = (device_pinch - b_true) @ a_true  # A^T (x - b)
    pinch_xr += rng.normal(scale=0.003, size=pinch_xr.shape)
    a, b, offset, rms = hands.StylusCalibration.fit(pinch_xr, stylus, rotations)
    assert rms < 0.008, rms  # 3 mm per axis is about 5 mm in 3D
    assert np.allclose(a, a_true, atol=0.03)
    assert np.linalg.norm(offset - offset_true) < 0.008
    probe = pinch_xr.mean(axis=0) + rng.uniform(-0.15, 0.15, size=(20, 3))  # where the hands are
    mapped = probe @ a.T + b
    assert np.max(np.linalg.norm(mapped - (probe @ a_true.T + b_true), axis=1)) < 0.01


def test_calibration_learns_from_the_held_stylus_and_places_the_hand_on_the_drill():
    rng = np.random.default_rng(4)
    a_true = _random_rotation(rng)
    b_true = np.array([0.1, 0.2, -0.3])
    offset_true = np.array([0.06, 0.0, 0.0])
    r_cam = _random_rotation(rng)
    gain = 12.0
    calibration = hands.StylusCalibration(path=None)
    template = _fake_hand(rng, np.zeros(3))
    for i in range(150):
        stylus = rng.uniform(-0.05, 0.05, size=3) + (0.0, 0.0, 0.05)
        r_stylus = _random_rotation(rng, 0.6)
        pinch_device = stylus + r_stylus @ offset_true
        pinch_xr = a_true.T @ (pinch_device - b_true)
        right = template - template[list(hands.GRIP_JOINTS)].mean(axis=0) + pinch_xr
        left = right + (0.25, 0.0, 0.0) + rng.normal(scale=0.05, size=3)  # not holding anything
        frame = hands.parse(_buffer(2, 1.0, (0, 0, 0, 0, 0, 0, 1), [left, right]))
        tip = np.array([0.5, 0.1, -0.2]) + 0.01 * i
        calibration.add(frame, _state(tip, r_cam, r_stylus, stylus))
    assert calibration.maybe_solve(force=True)
    assert calibration.result["hand"] == 1
    assert calibration.result["rms"] < 0.002, calibration.result
    # Holding the stylus, the pinch lands at the same offset on the drill.
    stylus = np.array([0.01, 0.02, 0.06])
    r_stylus = _random_rotation(rng, 0.5)
    tip = np.array([-5.1, 0.26, -0.3])
    pinch_xr = a_true.T @ (stylus + r_stylus @ offset_true - b_true)
    right = template - template[list(hands.GRIP_JOINTS)].mean(axis=0) + pinch_xr
    frame = hands.parse(_buffer(2, 1.0, None, [None, right]))
    placed, scale, mode = hands.map_to_scene(
        frame, _state(tip, r_cam, r_stylus, stylus), calibration, gain, mode="touch"
    )
    assert mode == "touch" and placed[0] is None and scale == gain
    pinch_scene = placed[1][list(hands.GRIP_JOINTS)].mean(axis=0)
    expected = tip + gain * (r_cam @ r_stylus @ offset_true)
    assert np.linalg.norm(pinch_scene - expected) < 0.02 * gain * 0.06
    # Joint spacing grows by the gain.
    d_xr = np.linalg.norm(right[hands.WRIST] - right[hands.MIDDLE_PROXIMAL])
    d_scene = np.linalg.norm(placed[1][hands.WRIST] - placed[1][hands.MIDDLE_PROXIMAL])
    assert abs(d_scene - gain * d_xr) < 1e-6


def test_docked_stylus_adds_no_samples():
    rng = np.random.default_rng(5)
    calibration = hands.StylusCalibration(path=None)
    frame = hands.parse(_buffer(2, 1.0, None, [_fake_hand(rng, np.zeros(3)), None]))
    for _ in range(20):
        calibration.add(frame, _state(np.zeros(3), np.eye(3), np.eye(3), hands.INKWELL_M + rng.normal(scale=0.003, size=3)))
    assert len(calibration.samples) == 0
    assert hands.stylus_docked({"hx": -0.088, "hy": 0.0, "hz": -0.066}) is True
    assert hands.stylus_docked({"hx": 0.0, "hy": 0.0, "hz": 0.0}) is False
    assert hands.stylus_docked({"px": 0.0}) is None


def test_head_placement_puts_the_lap_at_the_focus():
    rng = np.random.default_rng(6)
    r_cam = _random_rotation(rng)
    camera = np.array([1.0, 2.0, 3.0])
    look = -r_cam[:, 0]
    tip = camera + 2.3 * look
    head_rot = _random_rotation(rng)
    head_pos = np.array([0.0, 1.2, 0.1])
    lap = head_pos + head_rot @ hands.HEAD_REF_M
    hand = _fake_hand(rng, lap)
    hand[0] = lap
    hand[1] = lap + head_rot @ np.array([0.0, 0.1, 0.0])  # 10 cm toward the top of the head
    q = hands.mat_to_quat(head_rot)
    frame = hands.parse(_buffer(2, 1.0, (*head_pos, *q), [hand, None]))
    state = _state(tip, r_cam, np.eye(3), np.array([0.0, 0.0, 0.0]), camera=camera)
    placed, scale, mode = hands.map_to_scene(frame, state, None, 12.0)
    assert mode == "head" and scale == 12.0
    assert np.allclose(placed[0][0], tip, atol=1e-5)
    # Up in the head frame is up in the microscope image.
    assert np.allclose(placed[0][1] - placed[0][0], 1.2 * r_cam[:, 2], atol=1e-5)
    # With the camera's field of view known, HEAD_VIEW_SPAN_M fills the view height.
    state["fov"] = 0.18
    placed, scale, mode = hands.map_to_scene(frame, state, None, 12.0)
    view_height = 2.0 * 2.3 * np.tan(0.09)
    assert abs(scale * hands.HEAD_VIEW_SPAN_M - view_height) < 1e-6
    assert np.allclose(placed[0][0], tip, atol=1e-5)
    # Touch mode without a fit falls back to the head.
    assert hands.map_to_scene(frame, state, hands.StylusCalibration(path=None), 12.0, mode="touch")[2] == "head"


def test_instances_have_fixed_counts_and_a_flat_palm():
    rng = np.random.default_rng(7)
    points = _fake_hand(rng, np.zeros(3)) * 12.0
    spheres, sphere_scales, tubes, tube_scales = hands.hand_instances(points, np.full(hands.JOINTS, 0.008), 12.0)
    assert spheres.shape == (hands.JOINTS + 1, 7) and sphere_scales.shape == (hands.JOINTS + 1, 3)
    assert tubes.shape == (len(hands.BONES), 7) and tube_scales.shape == (len(hands.BONES), 3)
    palm = sphere_scales[-1]
    assert palm[1] < palm[0] and palm[1] < palm[2]
    # Each tube runs from its first joint to its second.
    for (a, b), xform, scale in zip(hands.BONES, tubes, tube_scales):
        end = xform[:3] + hands.quat_to_mat(xform[3:]) @ np.array([scale[0], 0.0, 0.0])
        assert np.allclose(end, points[b], atol=1e-6)


def test_gain_is_measured_from_free_motion():
    estimator = hands.GainEstimator()
    assert estimator.gain == hands.DEFAULT_GAIN
    for i in range(40):
        h = np.array([0.002 * i, 0.0, 0.0])
        estimator.update({"px": 9.0 * h[0], "py": 0.0, "pz": 0.0, "hx": h[0], "hy": h[1], "hz": h[2]})
    assert abs(estimator.gain - 9.0) < 1e-6
    # Clutched motion moves the stylus but not the drill, and is not counted.
    for i in range(40, 200):
        estimator.update({"px": 0.0, "py": 0.0, "pz": 0.0, "hx": 0.002 * i, "hy": 0.0, "hz": 0.0, "device_clutch": True})
    assert abs(estimator.gain - 9.0) < 1e-6


if __name__ == "__main__":
    test_layout_matches_the_plugin()
    test_quaternion_helpers()
    test_file_round_trip_and_torn_frames()
    test_fit_recovers_the_headset_to_touch_transform()
    test_calibration_learns_from_the_held_stylus_and_places_the_hand_on_the_drill()
    test_docked_stylus_adds_no_samples()
    test_head_placement_puts_the_lap_at_the_focus()
    test_instances_have_fixed_counts_and_a_flat_palm()
    test_gain_is_measured_from_free_motion()
    print("hand tests passed")
