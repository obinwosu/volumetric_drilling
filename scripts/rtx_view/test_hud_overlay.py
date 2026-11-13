"""Headset text overlay for AMBF's eyes: compositing and the shared-memory layout.

    python3 scripts/rtx_view/test_hud_overlay.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import headset_hud  # noqa: E402
import hud_overlay  # noqa: E402


def test_over_matches_the_rtx_blend_on_an_opaque_picture():
    rng = np.random.default_rng(1)
    layer = rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)
    picture = rng.integers(0, 256, (40, 50, 4), dtype=np.uint8)
    picture[..., 3] = 255
    # RTX arm: text blended straight into the eye image.
    rtx = picture.copy()
    headset_hud.blend(rtx, layer, 7, 5)
    # AMBF arm: text composed on a transparent overlay, then blended over the eye (GL SRC_ALPHA).
    canvas = np.zeros_like(picture)
    hud_overlay.over(canvas, layer, 7, 5)
    a = canvas[..., 3:4].astype(np.float32) / 255.0
    gl = picture[..., :3] * (1 - a) + canvas[..., :3] * a
    assert np.abs(gl - rtx[..., :3]).max() <= 2.0


def test_cover_is_opaque_and_share_is_bottom_row_first():
    path = pathlib.Path(tempfile.mkdtemp()) / "overlay"
    share = hud_overlay.OverlayShare(8, 4, path)
    layer = np.zeros((1, 8, 4), np.uint8)
    layer[..., :] = (255, 0, 0, 255)
    image = hud_overlay.compose(True, [(layer, 0, 0)], 8, 4)
    assert (image[..., 3] == 255).all() and tuple(image[0, 0, :3]) == (255, 0, 0)
    share.write(image)
    data = path.read_bytes()
    assert data[0:4] == b"FVHU"
    seq = int(np.frombuffer(data[8:16], np.uint64)[0])
    flags = int(np.frombuffer(data[24:28], np.uint32)[0])
    assert seq % 2 == 0 and seq > 0 and flags == 1
    pixels = np.frombuffer(data[64:], np.uint8).reshape(4, 8, 4)
    assert tuple(pixels[-1, 0, :3]) == (255, 0, 0)  # top row of the image is the last row
    share.write(None)
    assert int(np.frombuffer(path.read_bytes()[24:28], np.uint32)[0]) == 0
    share.close()
    assert not path.exists()


def test_rtx_start_up_card(preview=None):
    """The launcher's rtx_loading.json becomes an opaque card with stage, percent and log lines."""
    import json
    path = pathlib.Path(tempfile.mkdtemp()) / "rtx_loading.json"
    assert hud_overlay.read_loading(path) is None
    info = {"heading": "Initializing RTX View", "label": "Meshing bone", "percent": 62,
            "lines": ["Loading intact volume", "Closed occupancy 256x256x253",
                      "Meshing union of 12 structures " + "x" * 200, "Initial surface 1.2M triangles"]}
    path.write_text(json.dumps(info))
    assert hud_overlay.read_loading(path) == info
    path.write_text("{half written")
    assert hud_overlay.read_loading(path) is None  # never a crash on a partial file
    rgba, x, y = hud_overlay.loading_card(info, 1152, 864)
    assert 0 <= x and 0 <= y and rgba.shape[1] <= 1152 * 0.75 and rgba.shape[0] < 864
    image = hud_overlay.compose(True, [(rgba, x, y)], 1152, 864)
    assert (image[..., 3] == 255).all()  # opaque: nothing of AMBF shows behind it
    if preview:
        from PIL import Image
        Image.fromarray(image[..., :3]).save(preview)


if __name__ == "__main__":
    test_over_matches_the_rtx_blend_on_an_opaque_picture()
    test_cover_is_opaque_and_share_is_bottom_row_first()
    test_rtx_start_up_card(sys.argv[1] if len(sys.argv) > 1 else None)
    print("ok")
