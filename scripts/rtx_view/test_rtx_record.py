"""The recorder copies a frame out of ovrtx's render results.

    middle_ear_newton/.venv/bin/python scripts/rtx_view/test_rtx_record.py

ovrtx returns a dict-like set without .get(); the recorder once called .get()
and silently recorded no RTX frames.
"""
from __future__ import annotations

import contextlib
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rtx_record  # noqa: E402


class Products:
    """Only what ovrtx's RenderProductSetOutputs offers: in, [], iteration, no .get()."""

    def __init__(self, items):
        self._items = items

    def __contains__(self, key):
        return key in self._items

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)


class Var:
    def __init__(self, rgba):
        self.rgba = rgba

    @contextlib.contextmanager
    def map(self, device=None):
        yield self.rgba


class Frame:
    def __init__(self, rgba):
        self.render_vars = {"LdrColor": Var(rgba)}


class Product:
    def __init__(self, rgba):
        self.frames = [Frame(rgba)]


def test_records_a_frame():
    tmp = pathlib.Path(tempfile.mkdtemp())
    rtx_record.ACTIVE_PATH = tmp / "record_on"
    rtx_record.ACTIVE_PATH.touch()
    written = []

    class Share:
        def write(self, image, stamp, pose7, fov):
            written.append(image.shape)

    rtx_record.RecordShare = Share

    class Viewer:
        _render_product_path = "/Render/Main"
        camera = type("Cam", (), {"fov": 10.0})()

        def _accept_render(self, products):
            pass

    class Bridge:
        camera_stamped = (1.0, [0, 0, 0, 0, 0, 0, 1])

    viewer = Viewer()
    rtx_record.install(viewer, Bridge())
    rgba = np.zeros((72, 128, 4), np.uint8)
    viewer._accept_render(Products({"/Render/Main": Product(rgba)}))
    assert written, "no frame recorded"
    viewer._accept_render(Products({}))  # no product this step: nothing, no error


if __name__ == "__main__":
    test_records_a_frame()
    print("ok")
