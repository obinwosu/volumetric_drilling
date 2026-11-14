"""One slide per tutorial step: the study's own card drawn over a microscope frame.

    python3 scripts/study/make_tutorial_slides.py [--background PNG] [--out DIR]

Each card comes from study_session.StudySession.hud() and is drawn by the same
code that draws it in the headset (scripts/rtx_view/headset_hud.py), so the
slides show what the participant reads. The last slide is the pause menu help
shown above the menu while it is open. Writes slide_NN_<step>.png and
tutorial_steps.pdf (16:9, one slide per page).

The default background, docs/tutorial_slides/scene.png, is the RTX start-up
microscope view of Anatomy A (volume_256_RT143.yaml) from the tuner.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "rtx_view"))
import headset_hud  # noqa: E402
import study_session as ss  # noqa: E402

SLIDE_W, SLIDE_H = 1920, 1080
DEFAULT_BACKGROUND = REPO / "docs" / "tutorial_slides" / "scene.png"
DEFAULT_OUT = REPO / "docs" / "tutorial_slides"


class _QuietIO:
    """Enough of SessionIO for StudySession to build its cards; touches nothing."""

    def now(self):
        return 0.0

    def read_state(self):
        return {}

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def step_cards():
    """[(file stem, study_hud.json dict)] for every tutorial page, then the pause menu help."""
    session = ss.StudySession(_QuietIO(), "SLIDES", pathlib.Path(tempfile.mkdtemp()) / "x", {})
    cards = []
    for index, step in enumerate(ss.TUTORIAL):
        session.phase = "tutorial"
        session.step = index
        hud = session.hud(0.0)
        hud["timer"] = None
        cards.append((step.key, hud))
    cards.append(("pause_menu_help", {"panel": ss.MENU_HELP, "timer": None, "banner": None}))
    return cards


def draw(background: Image.Image, hud: dict) -> Image.Image:
    image = np.array(background.convert("RGB").resize((SLIDE_W, SLIDE_H), Image.LANCZOS))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        headset_hud.STUDY_PATH = tmp / "study_hud.json"
        headset_hud.PAUSE_PATH = tmp / "pause_hud.txt"
        headset_hud.CURSOR_PATH = tmp / "pause_cursor.txt"
        headset_hud.STATE_PATH = tmp / "drill_state.json"
        headset_hud.STUDY_PATH.write_text(json.dumps(hud))
        hud_drawer = headset_hud.HeadsetHud()
        for rgba, x, y in hud_drawer.layers(SLIDE_W, SLIDE_H):
            headset_hud.blend(image, rgba, x, y)
    return Image.fromarray(image)


def main(argv=None) -> pathlib.Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--background", type=pathlib.Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    background = Image.open(args.background)
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("slide_*.png"):
        old.unlink()
    slides = []
    for number, (key, hud) in enumerate(step_cards()):
        slide = draw(background, hud)
        slide.save(args.out / f"slide_{number:02d}_{key}.png")
        slides.append(slide)
    pdf = args.out / "tutorial_steps.pdf"
    slides[0].save(pdf, save_all=True, append_images=slides[1:], resolution=144.0)
    print(f"{len(slides)} slides -> {pdf}")
    return pdf


if __name__ == "__main__":
    main()
