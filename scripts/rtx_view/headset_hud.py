"""Text in the headset: the simulator pause menu and the study tutorial and timer.

The headset shows the RTX image, not AMBF's own panels, so text is drawn into
both eye images just before they go to the headset (rtx_eyes.Eyes.overlay).
The same pixels in both eyes put the text at the depth of the field centre,
where the eyes already converge, so reading it needs no refocusing.

Sources, all under /tmp/fivrs-rtx:
- pause_hud.txt  written by the simulator while its pause menu is open
                 (double-tap stylus button 1). Shown only while drill_state.json
                 also reports the simulator paused, so a stale file is ignored.
- study_hud.json written by the study controller (scripts/study/study_session.py):
    {"panel": {"title": str, "lines": [str], "progress": [done], "checklist": [[label, done]],
               "footer": str} | null,
     "timer": {"remaining_s": float, "running": bool, "written_wall": float} | null,
     "banner": str | null}

While the RTX window is loading or restoring the bone it sets ``loading`` to
(label, fraction); the eyes then show a plain card instead of the bone being built.

Recorded frames are copied before this runs (rtx_record), so recordings have no text.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
PAUSE_PATH = RUNTIME / "pause_hud.txt"
STUDY_PATH = RUNTIME / "study_hud.json"
STATE_PATH = RUNTIME / "drill_state.json"
FONT_DIR = pathlib.Path("/usr/share/fonts/truetype/dejavu")

PANEL_FILL = (12, 24, 36, 205)
TEXT = (236, 242, 248, 255)
DIM = (150, 170, 190, 255)
DONE = (120, 220, 140, 255)
ACCENT = (255, 205, 90, 255)
LOADING_BACKGROUND = (8, 14, 20)


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont

    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold.ttf" if bold else ".ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def _read_file(path: pathlib.Path, cache: dict):
    """(content, changed). Re-reads only when the file's mtime changes."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        changed = cache.get(path) is not None
        cache[path] = None
        return None, changed
    entry = cache.get(path)
    if entry is not None and entry[0] == mtime:
        return entry[1], False
    try:
        text = path.read_text()
    except OSError:
        return (entry[1] if entry else None), False
    cache[path] = (mtime, text)
    return text, True


def render_panel(lines, width: int, font_px: int, mono: bool = False):
    """lines: list of (text, colour, bold). Returns an RGBA uint8 array.

    Proportional text wraps to ``width``. Monospaced text (the pause menu's
    aligned columns) is never wrapped; the panel grows to its longest line.
    """
    from PIL import Image, ImageDraw

    pad = font_px
    gap = int(font_px * 0.35)
    fonts = {bold: _font(font_px, bold, mono) for bold in (False, True)}
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    wrapped = []
    if mono:
        wrapped = [(text, colour, fonts[bold]) for text, colour, bold in lines]
        longest = max((measure.textlength(text, font=font) for text, _c, font in wrapped), default=0)
        width = max(width, int(longest) + 2 * pad)
        lines = []
    for text, colour, bold in lines:
        font = fonts[bold]
        if not text:
            wrapped.append(("", colour, font))
            continue
        words, line = text.split(" "), ""
        for word in words:
            trial = (line + " " + word) if line else word
            if measure.textlength(trial, font=font) <= width - 2 * pad or not line:
                line = trial
            else:
                wrapped.append((line, colour, font))
                line = word
        wrapped.append((line, colour, font))
    line_h = font_px + gap
    height = 2 * pad + line_h * len(wrapped) - gap
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=font_px, fill=PANEL_FILL)
    y = pad
    for text, colour, font in wrapped:
        draw.text((pad, y), text, fill=colour, font=font)
        y += line_h
    return np.asarray(image, dtype=np.uint8)


def blend(image: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend an RGBA overlay into an RGBA/RGB image in place, clipped to the image."""
    h, w = overlay.shape[:2]
    H, W = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = overlay[y0 - y : y1 - y, x0 - x : x1 - x]
    dst = image[y0:y1, x0:x1, :3]
    alpha = src[..., 3:4].astype(np.uint16)
    mixed = dst.astype(np.uint16) * (255 - alpha) + src[..., :3].astype(np.uint16) * alpha
    dst[...] = ((mixed + 127) // 255).astype(np.uint8)


def study_lines(panel: dict):
    lines = []
    if panel.get("title"):
        lines.append((str(panel["title"]), ACCENT, True))
    for text in panel.get("lines") or []:
        lines.append((str(text), TEXT, False))
    progress = panel.get("progress") or []
    if progress:
        lines.append(("  ".join("●" if done else "○" for done in progress), DONE, False))
    checklist = panel.get("checklist") or []
    if checklist:
        lines.append(("", TEXT, False))
        for label, done in checklist:
            lines.append((("✓  " if done else "○  ") + str(label), DONE if done else TEXT, bool(done) is False))
    if panel.get("footer"):
        lines.append(("", TEXT, False))
        lines.append((str(panel["footer"]), DIM, False))
    return lines


def format_clock(seconds: float) -> str:
    seconds = max(0, int(np.ceil(seconds)))
    return f"{seconds // 60:d}:{seconds % 60:02d}"


class HeadsetHud:
    """Builds the overlay from the runtime files and draws it into eye images."""

    def __init__(self):
        self._files = {}
        self._study = {}
        self._pause_text = None
        self._paused = False
        self._state_checked = 0.0
        self._cache = {}
        self.loading = None  # (label, fraction 0..1) while the bone is loading

    def _loading_layer(self, width: int, height: int):
        label, fraction = self.loading
        pct = int(max(0.0, min(1.0, float(fraction))) * 100)
        font_px = max(14, height // 32)
        lines = [(label, ACCENT, True), (f"{pct}%", TEXT, True), ("This may take a minute.", DIM, False)]
        rgba = self._cached(("loading", label, pct, width, height),
                            lambda: render_panel(lines, int(width * 0.4), int(font_px * 1.2)))
        return rgba, (width - rgba.shape[1]) // 2, (height - rgba.shape[0]) // 2

    def _poll(self, now: float):
        text, changed = _read_file(STUDY_PATH, self._files)
        if changed:
            try:
                self._study = json.loads(text) if text else {}
            except json.JSONDecodeError:
                pass
        self._pause_text, _ = _read_file(PAUSE_PATH, self._files)
        if now - self._state_checked > 0.1:
            self._state_checked = now
            state, _ = _read_file(STATE_PATH, self._files)
            try:
                self._paused = bool(json.loads(state).get("paused")) if state else False
            except json.JSONDecodeError:
                pass

    def _cached(self, key, build):
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) > 64:
                self._cache.clear()
            hit = build()
            self._cache[key] = hit
        return hit

    def layers(self, width: int, height: int, now_wall: float | None = None):
        """List of (rgba, x, y) to draw on an eye image of this size."""
        now_wall = time.time() if now_wall is None else now_wall
        self._poll(time.monotonic())
        out = []
        font_px = max(14, height // 32)
        if self._pause_text and self._paused:
            lines = [(line, ACCENT if i == 0 else TEXT, i == 0) for i, line in enumerate(self._pause_text.rstrip().split("\n"))]
            panel_w = int(width * 0.4)
            rgba = self._cached(("pause", self._pause_text, width, height),
                                lambda: render_panel(lines, panel_w, font_px, mono=True))
            out.append((rgba, (width - rgba.shape[1]) // 2, (height - rgba.shape[0]) // 2))
            return out  # the pause menu covers everything else
        study = self._study or {}
        # The bone fills the middle of the view, so the timer and study text sit
        # in a band along the top; only the pause menu and short banners centre.
        top = int(height * 0.06)
        timer = study.get("timer")
        if timer:
            remaining = float(timer.get("remaining_s", 0.0))
            if timer.get("running"):
                remaining -= max(0.0, now_wall - float(timer.get("written_wall", now_wall)))
            label = format_clock(remaining)
            colour = ACCENT if remaining <= 60 else TEXT
            rgba = self._cached(("timer", label, colour, height),
                                lambda: render_panel([(label, colour, True)], font_px * 5, font_px))
            out.append((rgba, (width - rgba.shape[1]) // 2, top))
            top += rgba.shape[0] + int(font_px * 0.6)
        banner = study.get("banner")
        if banner:
            rgba = self._cached(("banner", banner, width, height),
                                lambda: render_panel([(banner, ACCENT, True)], int(width * 0.6), int(font_px * 1.4)))
            out.append((rgba, (width - rgba.shape[1]) // 2, (height - rgba.shape[0]) // 2))
        panel = study.get("panel")
        if panel:
            key = ("panel", json.dumps(panel, sort_keys=True), width, height)
            # Wide and a little smaller than the timer, so it takes few lines of the top band.
            rgba = self._cached(key, lambda: render_panel(study_lines(panel), int(width * 0.70),
                                                          max(12, int(font_px * 0.85))))
            out.append((rgba, (width - rgba.shape[1]) // 2, top))
        return out

    def draw(self, left: np.ndarray, right: np.ndarray) -> None:
        height, width = left.shape[:2]
        if self.loading is not None:
            rgba, x, y = self._loading_layer(width, height)
            for eye in (left, right):
                eye[..., :3] = LOADING_BACKGROUND
                blend(eye, rgba, x, y)
            return
        for rgba, x, y in self.layers(width, height):
            blend(left, rgba, x, y)
            blend(right, rgba, x, y)


def install(viewer) -> HeadsetHud | None:
    """Attach to the headset eyes, if this window drives a headset."""
    eyes = getattr(viewer, "_eyes", None)
    if eyes is None:
        return None
    # A pause file left by an earlier session must not show.
    PAUSE_PATH.unlink(missing_ok=True)
    hud = HeadsetHud()
    eyes.overlay = hud.draw
    print("Headset text ready (pause menu, study prompts)", flush=True)
    return hud
