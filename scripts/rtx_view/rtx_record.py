"""RTX frames for the study recorder.

While a recording runs, the RTX window copies about ten frames a second into
/dev/shm/fivrs_rtx_record. The recorder (scripts/study/fivrs_recorder.py)
reads them and writes them to the HDF5 file as ``l_img``, so the review video
shows what the resident saw. Nothing is copied unless the recorder has created
/dev/shm/fivrs_record_on, so the window costs nothing outside a recording.

Shared memory layout (little endian):
    0   4  magic "FVRC"
    4   4  version (1)
    8   4  slot count
    12  4  width
    16  4  height
    20  4  reserved
    24  8  frames written so far (the newest frame is in slot (count - 1) % slots)
    32 32  reserved
    64     slots. Each slot:
            0  8  sequence: odd while writing, even when complete
            8  8  camera stamp, seconds (the camera pose this frame was drawn from)
           16 56  camera pose: x y z qx qy qz qw, simulator units
           72  4  vertical field of view, radians
           76  4  reserved
           80     image: BGR8, top row first
"""
from __future__ import annotations

import mmap
import os
import pathlib
import time

import numpy as np

SHARE_PATH = pathlib.Path("/dev/shm/fivrs_rtx_record")
ACTIVE_PATH = pathlib.Path("/dev/shm/fivrs_record_on")
MAGIC = b"FVRC"
VERSION = 1
HEADER = 64
SLOT_HEADER = 80
SLOTS = 32
RATE_HZ = 10.0
# The window renders 1280 x 720; every second pixel keeps files near the expert recordings' size.
MAX_WIDTH = 640


def _slot_size(width: int, height: int) -> int:
    return SLOT_HEADER + width * height * 3


class RecordShare:
    """Writer side, used by the RTX window."""

    def __init__(self, path=SHARE_PATH, slots: int = SLOTS):
        self.path = pathlib.Path(path)
        self.slots = int(slots)
        self._map = None
        self._shape = None
        self._count = 0

    def _open(self, width: int, height: int):
        size = HEADER + self.slots * _slot_size(width, height)
        if self._map is not None:
            self._map.close()
        partial = self.path.with_suffix(".part")
        with open(partial, "wb") as handle:
            handle.truncate(size)
        os.replace(partial, self.path)
        with open(self.path, "r+b") as handle:
            self._map = mmap.mmap(handle.fileno(), size)
        header = bytearray(HEADER)
        header[0:4] = MAGIC
        header[4:8] = np.uint32(VERSION).tobytes()
        header[8:12] = np.uint32(self.slots).tobytes()
        header[12:16] = np.uint32(width).tobytes()
        header[16:20] = np.uint32(height).tobytes()
        self._map[0:HEADER] = bytes(header)
        self._shape = (height, width)
        self._count = 0

    def write(self, bgr: np.ndarray, stamp: float, pose7, fov_y: float) -> None:
        """bgr: (H, W, 3) uint8, top row first."""
        height, width = bgr.shape[:2]
        if self._shape != (height, width):
            self._open(width, height)
        m = self._map
        slot = self._count % self.slots
        start = HEADER + slot * _slot_size(width, height)
        seq = int(np.frombuffer(m[start : start + 8], np.uint64)[0])
        m[start : start + 8] = np.uint64(seq + 1).tobytes()  # odd: writing
        m[start + 8 : start + 16] = np.float64(stamp).tobytes()
        m[start + 16 : start + 72] = np.asarray(pose7, np.float64).reshape(7).tobytes()
        m[start + 72 : start + 76] = np.float32(fov_y).tobytes()
        m[start + SLOT_HEADER : start + _slot_size(width, height)] = np.ascontiguousarray(bgr, np.uint8).tobytes()
        m[start : start + 8] = np.uint64(seq + 2).tobytes()  # even: complete
        self._count += 1
        m[24:32] = np.uint64(self._count).tobytes()

    def close(self):
        if self._map is not None:
            self._map.close()
            self._map = None


class RecordReader:
    """Reader side, used by the recorder. Returns each completed frame once."""

    def __init__(self, path=SHARE_PATH):
        self.path = pathlib.Path(path)
        self._map = None
        self._inode = None
        self._seen = None
        self.dropped = 0

    def _attach(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        if self._map is not None and stat.st_ino == self._inode:
            return True
        if self._map is not None:
            self._map.close()
            self._map = None
        if stat.st_size < HEADER:
            return False
        with open(self.path, "rb") as handle:
            self._map = mmap.mmap(handle.fileno(), stat.st_size, access=mmap.ACCESS_READ)
        if self._map[0:4] != MAGIC:
            self._map.close()
            self._map = None
            return False
        self._inode = stat.st_ino
        # Frames already in the file belong to an earlier recording.
        self._seen = int(np.frombuffer(self._map[24:32], np.uint64)[0])
        return True

    def poll(self):
        """List of (stamp, pose7, fov_y, bgr) written since the last call."""
        if not self._attach():
            return []
        m = self._map
        slots = int(np.frombuffer(m[8:12], np.uint32)[0])
        width = int(np.frombuffer(m[12:16], np.uint32)[0])
        height = int(np.frombuffer(m[16:20], np.uint32)[0])
        count = int(np.frombuffer(m[24:32], np.uint64)[0])
        if count < self._seen:
            self._seen = 0
        if count - self._seen > slots - 1:
            # Keep one slot of margin: the writer may be filling the oldest one.
            self.dropped += count - self._seen - (slots - 1)
            self._seen = count - (slots - 1)
        out = []
        size = _slot_size(width, height)
        for index in range(self._seen, count):
            start = HEADER + (index % slots) * size
            before = int(np.frombuffer(m[start : start + 8], np.uint64)[0])
            if before % 2:
                self.dropped += 1
                continue
            stamp = float(np.frombuffer(m[start + 8 : start + 16], np.float64)[0])
            pose = np.frombuffer(m[start + 16 : start + 72], np.float64).copy()
            fov = float(np.frombuffer(m[start + 72 : start + 76], np.float32)[0])
            image = np.frombuffer(m[start + SLOT_HEADER : start + size], np.uint8).reshape(height, width, 3).copy()
            after = int(np.frombuffer(m[start : start + 8], np.uint64)[0])
            if after != before:
                self.dropped += 1
                continue
            out.append((stamp, pose, fov, image))
        self._seen = count
        return out

    def close(self):
        if self._map is not None:
            self._map.close()
            self._map = None


def to_record_image(rgba: np.ndarray) -> np.ndarray:
    """Renderer RGBA (top row first) to BGR at no more than MAX_WIDTH."""
    step = max(1, int(np.ceil(rgba.shape[1] / MAX_WIDTH)))
    return np.ascontiguousarray(rgba[::step, ::step, 2::-1])


def install(viewer, bridge, rate_hz: float = RATE_HZ):
    """Copy the window's frame for the recorder while /dev/shm/fivrs_record_on exists."""
    share = RecordShare()
    period = 1.0 / float(rate_hz)
    state = {"next": 0.0, "warned": False, "announced": False}
    accept = viewer._accept_render

    def accept_render(products):
        accept(products)
        if products is None or not ACTIVE_PATH.exists():
            return
        now = time.monotonic()
        if now < state["next"]:
            return
        stamped = getattr(bridge, "camera_stamped", None)
        if stamped is None:
            return
        state["next"] = now + period
        try:
            from ovrtx import Device

            # ovrtx's result set is dict-like but has no .get().
            path = viewer._render_product_path
            if path not in products:
                return
            product = products[path]
            rgba = None
            for frame in product.frames:
                for var in frame.render_vars.values():
                    with var.map(device=Device.CPU) as mapping:
                        rgba = np.array(np.from_dlpack(mapping), copy=True)
                    break
                break
            if rgba is None:
                return
            stamp, pose7 = stamped
            fov = float(np.radians(getattr(viewer.camera, "fov", 10.29)))
            share.write(to_record_image(rgba), stamp, pose7, fov)
            if not state["announced"]:
                print(f"Recording RTX frames at {rate_hz:g} Hz", flush=True)
                state["announced"] = True
        except Exception as exc:
            if not state["warned"]:
                print(f"RTX recording frame failed: {exc}", flush=True)
                state["warned"] = True

    viewer._accept_render = accept_render
    return share
