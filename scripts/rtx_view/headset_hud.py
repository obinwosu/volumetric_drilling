"""Text in the headset: the simulator pause menu and the study tutorial and timer.

The headset shows the RTX image, not AMBF's own panels, so text is drawn into
both eye images just before they go to the headset (rtx_eyes.Eyes.overlay).
The same pixels in both eyes put the text at the depth of the field centre,
where the eyes already converge, so reading it needs no refocusing.

Sources, all under /tmp/fivrs-rtx:
- pause_hud.txt  written by the simulator while its pause menu is open
                 (hold the left pedal). Shown only while drill_state.json
                 also reports the simulator paused, so a stale file is ignored.
- study_hud.json written by the study controller (scripts/study/study_session.py):
    {"panel": {"title": str, "lines": [str], "progress": [done], "checklist": [[label, done]],
               "footer": str, "done": bool, "advance": str,
               "pictures": [{"device": "stylus" | "pedals" | "scope" | "headset",
                             "marks": {region: label}}],
               "look": {"target": [x, y, z], "radius": float}} | null,
     "timer": {"remaining_s": float, "running": bool, "written_wall": float} | null,
     "banner": str | null,
     "menu_help": panel | null}   shown above the pause menu while it is open

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
IMAGE_DIR = pathlib.Path(__file__).resolve().parent / "hud_images"

# Button regions on the photos in hud_images (pixels of the saved image).
PICTURES = {
    # Stylus upright. The study uses one stylus button, the "action button" (the
    # plugin's switch 0), nearer the stylus's active tip; the rig owner confirmed the
    # circled button is the right one. The other button is broken on this rig's Touch.
    "stylus": {"file": "stylus.png", "regions": {
        "action": ("ellipse", (24, 56, 95, 126)),
    }},
    "pedals": {"file": "pedals.png", "regions": {
        "left": ("rect", (15, 49, 249, 326)),
        "middle": ("rect", (374, 4, 531, 254)),
        "right": ("rect", (670, 29, 887, 300)),
    }},
}

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


def render_panel(lines, width: int, font_px: int, mono: bool = False, fill: bool = True):
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
    if fill:
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
        done = bool(panel.get("done"))
        lines.append((("✓  " if done else "") + str(panel["title"]), DONE if done else ACCENT, True))
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


_photo_cache = {}


def _photo(name: str):
    from PIL import Image

    if name not in _photo_cache:
        _photo_cache[name] = Image.open(IMAGE_DIR / PICTURES[name]["file"]).convert("RGBA")
    return _photo_cache[name]


def _label_pill(draw, text, centre_x, y, font, colour=ACCENT, anchor_top=True):
    """Dark pill with text centred on centre_x, its top (or bottom) at y."""
    w = draw.textlength(text, font=font)
    h = font.size
    pad = max(3, h // 3)
    top = y if anchor_top else y - h - 2 * pad
    box = (centre_x - w / 2 - pad, top, centre_x + w / 2 + pad, top + h + 2 * pad)
    draw.rounded_rectangle(box, radius=pad + 2, fill=(10, 16, 24, 235), outline=colour, width=2)
    draw.text((box[0] + pad, top + pad * 0.6), text, fill=colour, font=font)


def _render_stylus(marks: dict, height: int, font):
    """Upright stylus with what to do written to the right of each marked button."""
    from PIL import Image, ImageDraw

    photo = _photo("stylus")
    height = int(height * 1.5)  # narrow upright photo: taller than the others so the buttons read
    scale = height / photo.height
    body = photo.resize((max(1, int(photo.width * scale)), height), Image.LANCZOS)
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    label_w = max([measure.textlength(str(v), font=font) for v in marks.values()] + [0])
    left = int(font.size * 0.4)
    right = int(label_w + font.size * 3.2) if marks else 0
    image = Image.new("RGBA", (left + body.width + right, height), (0, 0, 0, 0))
    image.alpha_composite(body, (left, 0))
    draw = ImageDraw.Draw(image)
    line = max(3, height // 60)
    for region, (shape, box) in PICTURES["stylus"]["regions"].items():
        x0, y0, x1, y1 = (v * scale for v in box)
        cy = (y0 + y1) / 2
        if region in marks:
            draw.ellipse((left + x0, y0, left + x1, y1), outline=ACCENT, width=line)
            ty = min(max(cy, font.size), height - font.size)
            x = left + body.width + font.size * 1.2
            draw.line((left + x1, cy, x, ty), fill=ACCENT, width=line)
            text = str(marks[region])
            w = draw.textlength(text, font=font)
            _label_pill(draw, text, x + w / 2 + font.size * 0.35, ty - font.size * 0.85, font)
    return np.asarray(image, dtype=np.uint8)


def render_picture(picture: dict, height: int, font_px: int):
    """One device picture with the buttons to use outlined and labelled. RGBA array."""
    from PIL import Image, ImageDraw

    device = picture.get("device")
    marks = picture.get("marks") or {}
    font = _font(max(10, font_px), bold=True)
    if device == "stylus":
        return _render_stylus(marks, height, font)
    if device in PICTURES:
        photo = _photo(device)
        scale = height / photo.height
        image = photo.resize((max(1, int(photo.width * scale)), height), Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        line = max(3, height // 45)
        for region, label in marks.items():
            shape, box = PICTURES[device]["regions"][str(region)]
            box = [v * scale for v in box]
            if shape == "ellipse":
                draw.ellipse(box, outline=ACCENT, width=line)
            else:
                draw.rounded_rectangle(box, radius=line * 3, outline=ACCENT, width=line)
            # Keep the label's pill inside the photo (the outer pedals sit at its edges).
            half = draw.textlength(str(label), font=font) / 2 + max(3, font.size // 3) + 2
            cx = min(max((box[0] + box[2]) / 2, half), image.width - half)
            _label_pill(draw, str(label), cx, min(box[3] - font.size * 2.2, height - font.size * 2.2), font)
        return np.asarray(image, dtype=np.uint8)
    # No photo for the microscope rig button or the headset: a labelled tile.
    width = int(height * 1.2)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    marked = bool(marks)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=height // 8, fill=(40, 52, 66, 255),
                           outline=ACCENT if marked else DIM, width=max(3, height // 45))
    words = {"scope": ["Microscope", "handle", "button"], "headset": ["Quest", "headset"]}.get(device, [str(device)])
    title = _font(max(10, int(font_px * 0.9)), bold=True)
    y = height * 0.10
    for word in words:
        w = draw.textlength(word, font=title)
        draw.text(((width - w) / 2, y), word, fill=TEXT, font=title)
        y += title.size * 1.25
    for label in marks.values():
        _label_pill(draw, str(label), width / 2, height - font.size * 1.9 - height * 0.06, font)
    return np.asarray(image, dtype=np.uint8)


def render_study_panel(panel: dict, width: int, font_px: int):
    """The tutorial panel: text on the left, device pictures on the right."""
    from PIL import Image

    pictures = panel.get("pictures") or []
    if not pictures:
        return render_panel(study_lines(panel), width, font_px)
    pad = font_px
    pic_h = int(font_px * 6.5)
    rendered = [render_picture(p, pic_h, int(font_px * 0.75)) for p in pictures]
    pics_w = sum(r.shape[1] for r in rendered) + pad * (len(rendered) - 1)
    max_pics_w = int(width * 0.45)
    if pics_w > max_pics_w:  # shrink the pictures, never the text
        pic_h = int(pic_h * max_pics_w / pics_w)
        rendered = [render_picture(p, pic_h, int(font_px * 0.75)) for p in pictures]
        pics_w = sum(r.shape[1] for r in rendered) + pad * (len(rendered) - 1)
    text_w = width - pics_w - pad
    text = render_panel(study_lines(panel), text_w, font_px, fill=False)
    height = max([text.shape[0]] + [r.shape[0] + 2 * pad for r in rendered])
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    from PIL import ImageDraw
    ImageDraw.Draw(canvas).rounded_rectangle((0, 0, width - 1, height - 1), radius=font_px, fill=PANEL_FILL)
    canvas.alpha_composite(Image.fromarray(text), (0, 0))
    x = text_w
    for r in rendered:
        canvas.alpha_composite(Image.fromarray(r), (x, (height - r.shape[0]) // 2))
        x += r.shape[1] + pad
    return np.asarray(canvas, dtype=np.uint8)


def camera_from_state(state: dict):
    """(position, look, up, vertical fov) from drill_state.json, or None."""
    try:
        c = np.array([state["cx"], state["cy"], state["cz"]], float)
        look = np.array([state["lx"], state["ly"], state["lz"]], float)
        up = np.array([state["ux"], state["uy"], state["uz"]], float)
        fov = float(state["fov"])
    except (KeyError, TypeError, ValueError):
        return None
    if fov <= 0 or not np.all(np.isfinite(np.concatenate([c, look, up]))):
        return None
    look = look / np.linalg.norm(look)
    up = up - look * float(up @ look)  # square to the look direction
    return c, look, up / np.linalg.norm(up), fov


def project(point, camera):
    """Screen position of a world point in half-heights from the centre (x right, y down), or None if behind."""
    c, look, up, fov = camera
    d = np.asarray(point, float) - c
    depth = float(d @ look)
    if depth <= 1e-9:
        return None
    right = np.cross(look, up)
    t = np.tan(fov / 2.0)
    return float(d @ right) / depth / t, -float(d @ up) / depth / t


def _ring(radius_px: int, colour, line: int, cross: bool = False, dot: bool = False):
    from PIL import Image, ImageDraw

    size = 2 * radius_px + 2 * line + 2
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    m = line + 1
    draw.ellipse((m, m, size - m - 1, size - m - 1), outline=(0, 0, 0, 160), width=line + 2)
    draw.ellipse((m, m, size - m - 1, size - m - 1), outline=colour, width=line)
    mid = size // 2
    if cross:
        arm = radius_px // 3
        draw.line((mid - arm, mid, mid + arm, mid), fill=colour, width=line)
        draw.line((mid, mid - arm, mid, mid + arm), fill=colour, width=line)
    if dot:
        r = max(3, radius_px // 4)
        draw.ellipse((mid - r, mid - r, mid + r, mid + r), fill=colour)
    return np.asarray(image, dtype=np.uint8)


# Timer corner: the headset image edges are blurred and partly hidden by the lenses.
HEADSET_TIMER_INSET = (0.20, 0.18)
DESKTOP_TIMER_INSET = (0.02, 0.03)


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
        self._state = {}
        self._state_checked = 0.0
        self._cache = {}
        self.loading = None  # (label, fraction 0..1) while the bone is loading
        self.timer_inset = HEADSET_TIMER_INSET  # (from the right, from the top) as fractions

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
        # Every frame while a look target is up (it follows the camera), else 10 Hz.
        looking = bool(((self._study or {}).get("panel") or {}).get("look"))
        if looking or now - self._state_checked > 0.1:
            self._state_checked = now
            state, changed = _read_file(STATE_PATH, self._files)
            if changed:
                try:
                    self._state = json.loads(state) if state else {}
                except json.JSONDecodeError:
                    pass
            self._paused = bool(self._state.get("paused"))

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
            y = (height - rgba.shape[0]) // 2
            help_panel = (self._study or {}).get("menu_help")
            if help_panel:
                # Study tutorial: how to use the menu, in the top band, the menu below it.
                key = ("menu_help", json.dumps(help_panel, sort_keys=True), width, height)
                tip = self._cached(key, lambda: render_study_panel(help_panel, int(width * 0.86),
                                                                   max(12, int(font_px * 0.85))))
                top = int(height * 0.06)
                out.append((tip, (width - tip.shape[1]) // 2, top))
                y = max(y, top + tip.shape[0] + int(font_px * 0.5))
            out.append((rgba, (width - rgba.shape[1]) // 2, y))
            return out  # the pause menu covers everything else
        study = self._study or {}
        panel = study.get("panel") or {}
        if panel.get("cover"):
            # A full-screen card (the study introduction): centred on a plain
            # background, and it hides a bone reset going on behind it.
            lines = study_lines(panel)
            if self.loading is not None:
                pct = int(max(0.0, min(1.0, float(self.loading[1]))) * 100)
                lines.append((f"Preparing the bone {pct}%", DIM, False))
            key = ("cover", json.dumps(panel, sort_keys=True), len(lines), lines[-1][0], width, height)
            # Long reads ("compact") are drawn smaller so the whole card fits the view.
            cover_px = max(12, int(font_px * 0.8)) if panel.get("compact") else font_px
            rgba = self._cached(key, lambda: render_panel(lines, int(width * 0.62), cover_px))
            return [(rgba, (width - rgba.shape[1]) // 2, (height - rgba.shape[0]) // 2)]
        # The bone fills the middle of the view, so the study text sits in a band
        # along the top and the timer in the top-right corner.
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
            # Top right, out of the way of the study text. The headset keeps it
            # further in, where both eyes see it sharply through the lenses.
            inset_x, inset_y = self.timer_inset
            out.append((rgba, width - int(width * inset_x) - rgba.shape[1], int(height * inset_y)))
        banner = study.get("banner")
        if banner:
            rgba = self._cached(("banner", banner, width, height),
                                lambda: render_panel([(banner, ACCENT, True)], int(width * 0.6), int(font_px * 1.4)))
            out.append((rgba, (width - rgba.shape[1]) // 2, (height - rgba.shape[0]) // 2))
        panel = study.get("panel")
        if panel:
            shown = {k: v for k, v in panel.items() if k not in ("look", "advance", "compact")}
            key = ("panel", json.dumps(shown, sort_keys=True), panel.get("compact"), width, height)
            # Wide and a little smaller than the timer, so it takes few lines of the top band.
            # Long reads ("compact") are drawn smaller again so they fit the view.
            panel_px = max(12, int(font_px * (0.68 if panel.get("compact") else 0.85)))
            rgba = self._cached(key, lambda: render_study_panel(shown, int(width * 0.86), panel_px))
            out.append((rgba, (width - rgba.shape[1]) // 2, top))
            top += rgba.shape[0] + int(font_px * 0.5)
            if panel.get("advance"):
                text = str(panel["advance"])
                pill = self._cached(("advance", text, height),
                                    lambda: render_panel([(text, DONE, True)], int(width * 0.45), font_px))
                out.append((pill, (width - pill.shape[1]) // 2, top))
            if panel.get("look"):
                out.extend(self._look_layers(panel["look"], width, height))
        return out

    def _look_layers(self, look: dict, width: int, height: int):
        """Centre circle (where the microscope points) and the target to bring into it."""
        radius = float(look.get("radius", 0.15))
        r_px = max(8, int(radius * height / 2))
        line = max(2, height // 300)
        cx, cy = width // 2, height // 2
        camera = camera_from_state(self._state)
        spot = project(look.get("target", (0, 0, 0)), camera) if camera else None
        inside = spot is not None and np.hypot(*spot) <= radius
        colour = DONE if inside else TEXT
        centre = self._cached(("centre", r_px, colour), lambda: _ring(r_px, colour, line, cross=True))
        out = [(centre, cx - centre.shape[1] // 2, cy - centre.shape[0] // 2)]
        if spot is not None and not inside:
            t_px = max(6, r_px // 2)
            target = self._cached(("target", t_px), lambda: _ring(t_px, ACCENT, line + 1, dot=True))
            # Kept on screen at the edge when it is out of view, so it shows the way.
            margin = target.shape[0]
            x = int(np.clip(cx + spot[0] * height / 2, margin, width - margin)) - target.shape[1] // 2
            y = int(np.clip(cy + spot[1] * height / 2, margin, height - margin)) - target.shape[0] // 2
            out.append((target, x, y))
        return out

    def frame(self, width: int, height: int):
        """(cover the picture?, layers) for one frame."""
        layers = self.layers(width, height)
        panel = (self._study or {}).get("panel") or {}
        paused = bool(self._pause_text and self._paused)
        if panel.get("cover") and not paused:
            return True, layers
        if self.loading is not None:
            return True, [self._loading_layer(width, height)]
        return False, layers

    def draw(self, left: np.ndarray, right: np.ndarray) -> None:
        height, width = left.shape[:2]
        cover, layers = self.frame(width, height)
        for eye in (left, right):
            if cover:
                eye[..., :3] = LOADING_BACKGROUND
            for rgba, x, y in layers:
                blend(eye, rgba, x, y)


def install_desktop(viewer) -> HeadsetHud:
    """Draw the same text into the desktop window (study without the headset, for development).

    Each frame with text is copied to the CPU, drawn on and copied back before the
    window shows it; frames without text go straight through.
    """
    import warp as wp

    PAUSE_PATH.unlink(missing_ok=True)
    hud = HeadsetHud()
    hud.timer_inset = DESKTOP_TIMER_INSET
    blit = viewer._blit_to_window
    warned = []

    def blit_with_text(pixels):
        try:
            height, width = pixels.shape[:2]
            cover, layers = hud.frame(width, height)
            if layers or cover:
                image = pixels.numpy()
                if cover:
                    image[..., :3] = LOADING_BACKGROUND
                for rgba, x, y in layers:
                    blend(image, rgba, x, y)
                pixels = wp.array(image, dtype=wp.vec4ub, device=pixels.device)
        except Exception as exc:  # text must never stop the picture
            if not warned:
                print(f"Desktop study text failed: {exc}", flush=True)
                warned.append(True)
        blit(pixels)

    viewer._blit_to_window = blit_with_text
    print("Desktop study text ready (pause menu, study prompts)", flush=True)
    return hud


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
