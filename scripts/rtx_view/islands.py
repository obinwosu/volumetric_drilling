"""Floating bone islands: find them, and have AMBF delete them.

Small pieces of bone that are not connected to the rest of the volume can't
be drilled away (the drill never registers contact with a speck), and they
catch water and go dark in the RTX view. Two kinds:

* already in the volume when it loads, and
* cut loose while drilling.

The RTX window finds them here and asks AMBF to delete them through the
plugin's delete request (/tmp/fivrs-rtx/delete_sequence.bin). AMBF stays in
charge of the voxels: it clears them, its haptics stop seeing them, and it
publishes the removal back like any drill cut, so both views agree.

Only pieces made entirely of bone, up to a size limit, are removed. Small
pieces of other labels (nerve, carotid) sit inside their own structures and
are left alone.
"""

from __future__ import annotations

import os
import pathlib
import struct
import threading
import time

import numpy as np
from scipy import ndimage

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
REQUEST_PATH = RUNTIME / "delete_sequence.bin"
MAGIC = 0x51455346
# Largest piece removed, in voxels. Larger detached pieces are left alone.
MAX_VOXELS = int(os.environ.get("FIVRS_ISLAND_MAX_VOXELS", "2000"))
# How far around a cut to look. A piece wider than this that is cut loose is missed.
MARGIN = 24
_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)
_NEIGHBOURS = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)


def _packed(rgb) -> int:
    return int(rgb[0]) | (int(rgb[1]) << 8) | (int(rgb[2]) << 16)


def _pack_colors(colors) -> np.ndarray:
    c = np.asarray(colors)
    return c[..., 0].astype(np.int64) | (c[..., 1].astype(np.int64) << 8) | (c[..., 2].astype(np.int64) << 16)


def _bone_pieces(labels, count, colors, bone, keep, max_voxels, offset=(0, 0, 0)):
    """Index arrays (in the full volume) of the labelled pieces that are all bone and small."""
    if count == 0 or not len(keep):
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    found = []
    slices = ndimage.find_objects(labels)
    for label in keep:
        label = int(label)
        if label <= 0 or sizes[label] > max_voxels or slices[label - 1] is None:
            continue
        box = slices[label - 1]
        inside = labels[box] == label
        if not np.all(_pack_colors(colors[box][inside]) == bone):
            continue
        idx = np.argwhere(inside) + np.array([s.start for s in box]) + np.asarray(offset)
        found.append(idx.astype(np.int32))
    return found


def islands_at_load(solid, colors, bone_rgb, max_voxels=MAX_VOXELS):
    """Bone pieces not connected to the largest piece of the volume."""
    labels, count = ndimage.label(solid, structure=_CONNECTIVITY)
    if count <= 1:
        return []
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    keep = [label for label in range(1, count + 1) if label != main]
    return _bone_pieces(labels, count, colors, _packed(bone_rgb), keep, max_voxels)


def cut_loose(solid, colors, bone_rgb, cut, max_voxels=MAX_VOXELS, margin=MARGIN):
    """Bone pieces that a cut has just separated. ``cut`` is (N, 3) cleared voxels.

    Looks in a box around the cut. A piece that touches the side of the box
    may still be connected beyond it, so it is kept; one wholly inside is loose.
    """
    cut = np.asarray(cut, np.int64).reshape(-1, 3)
    if not len(cut):
        return []
    shape = np.asarray(solid.shape)
    lo = np.maximum(cut.min(axis=0) - margin, 0)
    hi = np.minimum(cut.max(axis=0) + margin + 1, shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    crop = solid[box]
    labels, count = ndimage.label(crop, structure=_CONNECTIVITY)
    if count == 0:
        return []
    # Pieces next to the cut.
    near = (cut - lo)[:, None, :] + _NEIGHBOURS[None, :, :]
    near = near.reshape(-1, 3)
    inside = np.all((near >= 0) & (near < np.asarray(crop.shape)), axis=1)
    near = near[inside]
    touched = np.unique(labels[near[:, 0], near[:, 1], near[:, 2]])
    touched = touched[touched > 0]
    if not len(touched):
        return []
    # Pieces reaching a side of the box that is not the edge of the volume might continue past it.
    open_sides = set()
    for axis in range(3):
        for side, at_edge in ((0, lo[axis] == 0), (-1, hi[axis] == shape[axis])):
            if at_edge:
                continue
            face = np.take(labels, side, axis=axis)
            open_sides.update(np.unique(face[face > 0]).tolist())
    keep = [label for label in touched if int(label) not in open_sides]
    return _bone_pieces(labels, count, colors[box], _packed(bone_rgb), keep, max_voxels, offset=lo)


def write_request(pieces, start_id: int, path=REQUEST_PATH) -> bool:
    """One batch per piece, all released at once. False if AMBF has not taken the last one."""
    path = pathlib.Path(path)
    if path.exists() or not pieces:
        return False
    blob = struct.pack("<II", MAGIC, len(pieces))
    for k, piece in enumerate(pieces):
        name = f"island-{start_id + k}".encode("ascii")
        indices = np.ascontiguousarray(np.asarray(piece, np.int32).reshape(-1), dtype="<i4")
        blob += struct.pack("<I", len(name)) + name
        blob += struct.pack("<dI", 0.0, len(indices) // 3)
        blob += indices.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(blob)
    temporary.replace(path)
    return True


class IslandTracker:
    """Keeps its own copy of what is solid (every label) and queues islands for AMBF."""

    def __init__(self, occupied, colors, bone_rgb, max_voxels=MAX_VOXELS):
        self.solid = np.array(occupied, dtype=bool, copy=True)
        self.colors = colors
        self.bone_rgb = tuple(int(c) for c in bone_rgb)
        self.max_voxels = int(max_voxels)
        self._pending = []
        self._lock = threading.Lock()
        self._next_id = 0
        self.removed = 0
        self.check_ms = 0.0

    def scan_at_load(self):
        """Whole volume, on a background thread: it can take a few seconds on large volumes."""

        def run():
            started = time.perf_counter()
            pieces = islands_at_load(self.solid, self.colors, self.bone_rgb, self.max_voxels)
            voxels = sum(len(p) for p in pieces)
            print(
                f"Floating bone islands at load: {len(pieces)} ({voxels} voxels), sent to AMBF to delete "
                f"in {time.perf_counter() - started:.1f} s",
                flush=True,
            )
            self._queue(pieces)

        threading.Thread(target=run, name="fivrs-islands", daemon=True).start()

    def on_cut(self, xs, ys, zs):
        """Called by the surface worker after it clears voxels."""
        if not len(xs):
            return
        started = time.perf_counter()
        cut = np.stack((np.asarray(xs), np.asarray(ys), np.asarray(zs)), axis=1)
        self.solid[cut[:, 0], cut[:, 1], cut[:, 2]] = False
        # Spread-out cuts (a resync replay) are checked region by region, so no box gets large.
        groups = [cut]
        if np.prod(np.ptp(cut, axis=0) + 2 * MARGIN + 1) > 4_000_000:
            keys = cut // 64
            _unique, which = np.unique(keys, axis=0, return_inverse=True)
            groups = [cut[which.reshape(-1) == k] for k in range(int(which.max()) + 1)]
        pieces = []
        for group in groups:
            pieces += cut_loose(self.solid, self.colors, self.bone_rgb, group, self.max_voxels)
        self.check_ms = (time.perf_counter() - started) * 1e3
        if pieces:
            # Count them as gone now, so the next cut does not find them again.
            for piece in pieces:
                self.solid[piece[:, 0], piece[:, 1], piece[:, 2]] = False
            self._queue(pieces)

    def _queue(self, pieces):
        if not pieces:
            return
        with self._lock:
            self._pending.extend(pieces)

    def pump(self, path=REQUEST_PATH):
        """From the frame loop: hand queued islands to AMBF once it has taken the last request."""
        with self._lock:
            if not self._pending or pathlib.Path(path).exists():
                return
            pieces, self._pending = self._pending, []
        if write_request(pieces, self._next_id, path):
            self._next_id += len(pieces)
            self.removed += sum(len(p) for p in pieces)
        else:
            with self._lock:
                self._pending = pieces + self._pending
