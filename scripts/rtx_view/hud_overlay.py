"""Study text for the headset when AMBF (CHAI3D) draws the eyes, not the RTX window.

    python3 scripts/rtx_view/hud_overlay.py [--size 1152x864]

The graphics study shows the same task with two renderers. With the RTX window
the headset text (study prompts, timer, pause menu) is drawn into the eye images
by headset_hud. With AMBF's own microscope eyes (AMBF_OPENXR_CHAI_EYES=1) there
is no Python in the picture path, so this process draws the very same layers
with headset_hud.HeadsetHud into one transparent image and the OpenXR eyepiece
plugin blends it over both eyes. Same pixels, same places, as in the RTX arm.

While the RTX view starts (before its first headset frame) the launcher writes
/tmp/fivrs-rtx/rtx_loading.json; this process then draws an "Initializing RTX
View" card (stage, progress, the last log lines) that the eyepiece shows on
black, so the participant is not left in the dark between sessions:
    {"label": str, "percent": int | null, "lines": [str], "heading": str}

Shared memory /dev/shm/fivrs_hud_overlay (little endian), read by
ambf_openxr_eyepiece.cpp:
    0   4  magic "FVHU"
    4   4  version (1)
    8   8  sequence: odd while writing, even when complete
    16  4  width
    20  4  height
    24  4  flags: bit 0 = something to draw
    28 36  reserved
    64     RGBA8, straight alpha, bottom row first (OpenGL)
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import pathlib
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import headset_hud  # noqa: E402

SHARE_PATH = pathlib.Path("/dev/shm/fivrs_hud_overlay")
MAGIC = b"FVHU"
VERSION = 1
HEADER = 64
SIZE = (1152, 864)  # the RTX window's headset eyes (fivrs_rtx_view.py --headset)
RATE_HZ = 30.0
LOADING_PATH = headset_hud.RUNTIME / "rtx_loading.json"
LOG_CHARS = 64  # longer log lines are cut (a long path would run past the card)


def read_loading(path=LOADING_PATH):
    """The launcher's RTX start-up status, or None."""
    try:
        info = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def loading_card(info: dict, width: int, height: int):
    """(rgba, x, y): the RTX start-up card, centred."""
    font_px = max(14, height // 34)
    percent = info.get("percent")
    stage = str(info.get("label") or "Starting")
    lines = [(str(info.get("heading") or "Initializing RTX View"), headset_hud.ACCENT, True),
             (stage + (f"  {int(percent)}%" if percent is not None else ""), headset_hud.TEXT, True),
             ("", headset_hud.DIM, False)]
    for line in list(info.get("lines") or [])[-6:]:
        line = str(line).strip()
        lines.append((line if len(line) <= LOG_CHARS else line[:LOG_CHARS - 1] + "\u2026", headset_hud.DIM, False))
    lines += [("", headset_hud.DIM, False), ("This may take a minute.", headset_hud.DIM, False)]
    rgba = headset_hud.render_panel(lines, int(width * 0.7), font_px)
    return rgba, (width - rgba.shape[1]) // 2, (height - rgba.shape[0]) // 2


def over(canvas: np.ndarray, layer: np.ndarray, x: int, y: int) -> None:
    """Porter-Duff 'over' of a straight-alpha RGBA layer onto a straight-alpha RGBA canvas."""
    h, w = layer.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = layer[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0
    dst = canvas[y0:y1, x0:x1].astype(np.float32) / 255.0
    sa, da = src[..., 3:4], dst[..., 3:4]
    out_a = sa + da * (1.0 - sa)
    rgb = (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / np.maximum(out_a, 1e-6)
    canvas[y0:y1, x0:x1, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    canvas[y0:y1, x0:x1, 3:4] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def compose(cover: bool, layers, width: int, height: int) -> np.ndarray:
    """The overlay for one frame, top row first. A cover card is opaque, as in the RTX arm."""
    canvas = np.zeros((height, width, 4), np.uint8)
    if cover:
        canvas[..., :3] = headset_hud.LOADING_BACKGROUND
        canvas[..., 3] = 255
    for rgba, x, y in layers:
        over(canvas, rgba, x, y)
    return canvas


class OverlayShare:
    def __init__(self, width: int, height: int, path=SHARE_PATH):
        self.path = pathlib.Path(path)
        self.width, self.height = width, height
        size = HEADER + width * height * 4
        partial = self.path.with_suffix(".part")
        with open(partial, "wb") as handle:
            handle.truncate(size)
        os.replace(partial, self.path)
        with open(self.path, "r+b") as handle:
            self._map = mmap.mmap(handle.fileno(), size)
        self._map[0:4] = MAGIC
        self._map[4:8] = np.uint32(VERSION).tobytes()
        self._map[16:20] = np.uint32(width).tobytes()
        self._map[20:24] = np.uint32(height).tobytes()
        self._image = np.frombuffer(self._map, np.uint8, count=width * height * 4,
                                    offset=HEADER).reshape(height, width, 4)
        self._seq = 0

    def write(self, image: np.ndarray | None) -> None:
        """image: (H, W, 4) top row first, or None for nothing to draw."""
        self._seq += 1
        self._map[8:16] = np.uint64(self._seq).tobytes()
        self._map[24:28] = np.uint32(0 if image is None else 1).tobytes()
        if image is not None:
            np.copyto(self._image, image[::-1])
        self._seq += 1
        self._map[8:16] = np.uint64(self._seq).tobytes()

    def close(self):
        self._image = None
        self._map.close()
        self.path.unlink(missing_ok=True)


def signature(cover: bool, layers):
    """Layers are cached arrays, so identity + place tells whether the frame changed."""
    return (cover, tuple((id(rgba), x, y) for rgba, x, y in layers))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", default=f"{SIZE[0]}x{SIZE[1]}")
    args = parser.parse_args(argv)
    width, height = (int(v) for v in args.size.lower().split("x"))
    headset_hud.PAUSE_PATH.unlink(missing_ok=True)  # an old pause menu must not show
    hud = headset_hud.HeadsetHud()
    share = OverlayShare(width, height)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    print(f"Headset text overlay {width}x{height} -> {SHARE_PATH}", flush=True)
    last = None
    card = (None, None)  # (loading json, rendered card)
    try:
        while not stop:
            started = time.monotonic()
            try:
                loading = read_loading()
                if loading is not None:
                    text = json.dumps(loading, sort_keys=True)
                    if card[0] != text:
                        card = (text, loading_card(loading, width, height))
                    cover, layers = True, [card[1]]
                else:
                    cover, layers = hud.frame(width, height)
                key = signature(cover, layers)
                if key != last:
                    share.write(compose(cover, layers, width, height) if (cover or layers) else None)
                    last = key
            except Exception as exc:  # text must never stop the session
                print(f"Headset text failed: {exc}", flush=True)
                time.sleep(1.0)
            time.sleep(max(0.0, 1.0 / RATE_HZ - (time.monotonic() - started)))
    finally:
        share.close()


if __name__ == "__main__":
    main()
