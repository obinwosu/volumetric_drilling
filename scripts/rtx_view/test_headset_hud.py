"""Headset text: pause menu, study panel and timer drawn into eye images.

    middle_ear_newton/.venv/bin/python scripts/rtx_view/test_headset_hud.py [preview.png]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import headset_hud  # noqa: E402

W, H = 1152, 864


def _point_at(tmp: pathlib.Path):
    headset_hud.PAUSE_PATH = tmp / "pause_hud.txt"
    headset_hud.STUDY_PATH = tmp / "study_hud.json"
    headset_hud.STATE_PATH = tmp / "drill_state.json"


def _eyes():
    left = np.full((H, W, 4), 90, np.uint8)
    return left, left.copy()


def test_nothing_drawn_without_files():
    with tempfile.TemporaryDirectory() as tmp:
        _point_at(pathlib.Path(tmp))
        hud = headset_hud.HeadsetHud()
        left, right = _eyes()
        hud.draw(left, right)
        assert (left == 90).all() and (right == 90).all()


def test_study_panel_and_timer_draw_identically_in_both_eyes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _point_at(tmp)
        (tmp / "study_hud.json").write_text(json.dumps({
            "panel": {"title": "Tutorial 2 of 8: Move the microscope",
                      "lines": ["Hold the microscope button and move your head to move the view.",
                                "Release the button to fix the view."],
                      "progress": [True, False, False, False, False, False, False, False]},
            "timer": {"remaining_s": 125.0, "running": False, "written_wall": time.time()},
        }))
        hud = headset_hud.HeadsetHud()
        left, right = _eyes()
        hud.draw(left, right)
        assert (left != 90).any()
        assert np.array_equal(left, right)  # same pixels: text sits at the field's depth
        assert (left[: int(H * 0.05)] == 90).all()  # a margin above the top band
        # The middle of the view, where the bone is, stays clear of study text.
        assert (left[int(H * 0.42) : int(H * 0.58), int(W * 0.4) : int(W * 0.6)] == 90).all()


def test_pause_menu_only_while_simulator_reports_paused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _point_at(tmp)
        (tmp / "pause_hud.txt").write_text("PAUSED\n\nVIEWS\n  > Home\n")
        (tmp / "drill_state.json").write_text(json.dumps({"paused": False}))
        hud = headset_hud.HeadsetHud()
        assert hud.layers(W, H) == []  # stale file from a resumed session
        state = tmp / "drill_state.json"
        state.write_text(json.dumps({"paused": True}))
        # The HUD re-reads on an mtime change; two writes this close can share one
        # coarse filesystem timestamp, so move it on explicitly.
        stamp = state.stat().st_mtime_ns + 10_000_000
        os.utime(state, ns=(stamp, stamp))
        hud._state_checked = 0.0
        layers = hud.layers(W, H)
        assert len(layers) == 1
        rgba, x, y = layers[0]
        assert 0 <= x and x + rgba.shape[1] <= W and 0 <= y and y + rgba.shape[0] <= H


def test_running_timer_counts_down_from_its_write_time():
    assert headset_hud.format_clock(600) == "10:00"
    assert headset_hud.format_clock(59.2) == "1:00"
    assert headset_hud.format_clock(-3) == "0:00"


def test_blend_is_fast_enough_for_every_headset_frame():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _point_at(tmp)
        (tmp / "study_hud.json").write_text(json.dumps({
            "panel": {"title": "Tutorial", "lines": ["x" * 80] * 4},
            "timer": {"remaining_s": 300.0, "running": True, "written_wall": time.time()},
        }))
        hud = headset_hud.HeadsetHud()
        left, right = _eyes()
        hud.draw(left, right)  # first frame renders and caches the text
        start = time.perf_counter()
        for _ in range(20):
            hud.draw(left, right)
        per_frame_ms = (time.perf_counter() - start) / 20 * 1e3
        assert per_frame_ms < 8.0, per_frame_ms


def test_loading_card_hides_the_bone_in_both_eyes():
    with tempfile.TemporaryDirectory() as tmp:
        _point_at(pathlib.Path(tmp))
        hud = headset_hud.HeadsetHud()
        hud.loading = ("Loading the bone", 0.42)
        left, right = _eyes()
        hud.draw(left, right)
        assert np.array_equal(left, right)
        corner = left[5, 5, :3]
        assert tuple(int(v) for v in corner) == headset_hud.LOADING_BACKGROUND
        hud.loading = None
        left, right = _eyes()
        hud.draw(left, right)
        assert (left == 90).all()


def preview(path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _point_at(tmp)
        from PIL import Image

        (tmp / "study_hud.json").write_text(json.dumps({
            "panel": {
                "title": "Tutorial 3 of 8: Zoom",
                "lines": ["Hold the left pedal and press Stylus 2 to zoom in, Stylus 1 to zoom out."],
                "progress": [True, True, False, False, False, False, False, False],
            },
            "timer": {"remaining_s": 432.0, "running": False, "written_wall": time.time()},
        }))
        hud = headset_hud.HeadsetHud()
        left, right = _eyes()
        hud.draw(left, right)
        (tmp / "pause_hud.txt").write_text(
            "PAUSED\n\nVIEWS  (RIGHT)\n    Home\n    Antrum  saved\n    Facial recess  empty\n\n"
            "BURR\n    Type          Cutting\n    Size          4mm\n          1mm\n        > 2mm\n          4mm\n          6mm\n\n"
            "Next option          Stylus 2  or  right pedal (c)\nChoose               Left pedal (a)\n"
            "Back                 Middle pedal (b)\n")
        (tmp / "drill_state.json").write_text(json.dumps({"paused": True}))
        hud2 = headset_hud.HeadsetHud()
        pl, pr = _eyes()
        hud2.draw(pl, pr)
        both = np.concatenate([left[..., :3], pl[..., :3]], axis=1)
        Image.fromarray(both).save(path)


if __name__ == "__main__":
    test_nothing_drawn_without_files()
    test_study_panel_and_timer_draw_identically_in_both_eyes()
    test_pause_menu_only_while_simulator_reports_paused()
    test_running_timer_counts_down_from_its_write_time()
    test_blend_is_fast_enough_for_every_headset_frame()
    test_loading_card_hides_the_bone_in_both_eyes()
    if len(sys.argv) > 1:
        preview(sys.argv[1])
    print("ok")
