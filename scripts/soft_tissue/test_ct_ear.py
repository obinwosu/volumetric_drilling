"""CT ear: two-point alignment along the canal's S-curve chord.

    middle_ear_newton/.venv/bin/python scripts/soft_tissue/test_ct_ear.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ct_ear  # noqa: E402
import envelope as ev  # noqa: E402

ADF = ev.ROOT / "ADF" / "volume_256_RT143.yaml"


def test_register_puts_meatus_on_target_and_canal_on_the_chord():
    vol = ev.Volume(ADF)
    f = vol.frame
    rng = np.random.default_rng(3)
    marks = dict(meatus=np.array([-70.0, 23.0, -120.0]), hole=np.array([-60.0, 30.0, -118.0]), superior=np.array([0.1, 0.0, 1.0]))
    target = ct_ear.meatus_target(vol, 14.0, 3.0, 2.0)
    T, info = ct_ear.register(vol, marks, target)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0  # a rotation, no mirror or scale
    w = lambda p: R @ p + T[:3, 3]  # noqa: E731
    assert np.allclose(w(marks["meatus"]), target, atol=1e-9)
    # the deep canal point lies on the line from the meatus target back to the EAC opening
    d = w(marks["hole"]) - target
    chord = f.canal_lateral_end - target
    assert np.linalg.norm(np.cross(d, chord)) / (np.linalg.norm(d) * np.linalg.norm(chord)) < 1e-9 and d @ chord > 0
    # superior stays as close to the volume's as the chord allows
    up = R @ marks["superior"]
    up = up / np.linalg.norm(up)
    assert up @ f.superior > 0.5
    # the meatus sits down and forward of the canal's end, as the S-curve has it
    r = target - f.canal_lateral_end
    assert r @ f.superior < 0 and r @ f.anterior > 0 and r @ f.lateral > 0
    del rng, info


def test_register_without_a_canal_uses_lateral_and_superior():
    vol = ev.Volume(ADF)
    f = vol.frame
    marks = dict(meatus=np.array([-70.0, 23.0, -120.0]), lateral=np.array([-0.82, -0.54, 0.21]), superior=np.array([0.17, 0.11, 0.98]))
    target = ct_ear.meatus_target(vol)
    T, info = ct_ear.register(vol, marks, target)
    R = T[:3, :3]
    assert info["by"] == "lateral and superior"
    assert np.allclose(R @ marks["meatus"] + T[:3, 3], target)
    lat = marks["lateral"] / np.linalg.norm(marks["lateral"])
    assert np.allclose(R @ lat, f.lateral, atol=1e-9)


if __name__ == "__main__":
    test_register_without_a_canal_uses_lateral_and_superior()
    print("test_register_without_a_canal_uses_lateral_and_superior ok")
    test_register_puts_meatus_on_target_and_canal_on_the_chord()
    print("test_register_puts_meatus_on_target_and_canal_on_the_chord ok")
