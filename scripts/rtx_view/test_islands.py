"""Floating bone islands: found at load and when a cut frees them; other labels left alone."""

from __future__ import annotations

import pathlib
import struct
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import islands

BONE = (255, 249, 219)
NERVE = (170, 0, 1)


def _block():
    """A bone block with a peg standing on a one-voxel neck, a floating speck, and a nerve speck."""
    shape = (64, 64, 64)
    solid = np.zeros(shape, bool)
    colors = np.zeros(shape + (3,), np.uint8)
    solid[8:56, 8:56, 8:30] = True
    solid[30:34, 30:34, 30] = False
    solid[31, 31, 30] = True          # neck
    solid[29:35, 29:35, 31:35] = True  # peg, 144 voxels, above the neck
    solid[10, 10, 40] = True          # floating bone speck
    solid[50, 50, 45] = True          # floating nerve speck
    colors[solid] = BONE
    colors[50, 50, 45] = NERVE
    return solid, colors


def test_islands_at_load_are_bone_only():
    solid, colors = _block()
    pieces = islands.islands_at_load(solid, colors, BONE)
    found = {tuple(map(tuple, p)) for p in pieces}
    if found != {((10, 10, 40),)}:
        raise SystemExit(f"expected only the bone speck, got {found}")


def test_a_cut_through_the_neck_frees_the_peg():
    solid, colors = _block()
    tracker = islands.IslandTracker(solid, colors, BONE)
    tracker.on_cut([31], [31], [30])
    if len(tracker._pending) != 1 or len(tracker._pending[0]) != 144:
        raise SystemExit(f"cutting the neck should free the 144-voxel peg, got {[len(p) for p in tracker._pending]}")
    # A cut somewhere in the block frees nothing.
    tracker._pending.clear()
    tracker.on_cut([20, 21], [20, 20], [29, 29])
    if tracker._pending:
        raise SystemExit("a cut into solid bone should not free anything")


def test_large_or_open_pieces_stay():
    solid, colors = _block()
    # A big slab joined to the block by one voxel: cutting it frees a piece over the size limit.
    solid[8:56, 8:56, 50:60] = True
    colors[8:56, 8:56, 50:60] = BONE
    solid[40, 40, 30:50] = True
    colors[40, 40, 30:50] = BONE
    tracker = islands.IslandTracker(solid, colors, BONE, max_voxels=2000)
    tracker.on_cut([40], [40], [45])
    if any(len(p) > 2000 for p in tracker._pending):
        raise SystemExit("pieces over the size limit must stay")


def test_request_file_matches_the_plugin_reader():
    pieces = [np.array([[1, 2, 3], [4, 5, 6]], np.int32), np.array([[7, 8, 9]], np.int32)]
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "delete_sequence.bin"
        if not islands.write_request(pieces, 5, path):
            raise SystemExit("the first request should be written")
        if islands.write_request(pieces, 7, path):
            raise SystemExit("a second request must wait until AMBF has taken the first")
        data = path.read_bytes()
    # Same reads as afVolmetricDrillingPlugin::loadRtxTrial.
    magic, count = struct.unpack_from("<II", data, 0)
    at = 8
    batches = []
    for _ in range(count):
        (length,) = struct.unpack_from("<I", data, at)
        at += 4
        name = data[at : at + length].decode()
        at += length
        release, voxels = struct.unpack_from("<dI", data, at)
        at += 12
        indices = np.frombuffer(data, "<i4", voxels * 3, at).reshape(-1, 3)
        at += voxels * 12
        batches.append((name, release, indices.tolist()))
    if magic != 0x51455346 or at != len(data):
        raise SystemExit("request layout does not match the plugin")
    if batches != [("island-5", 0.0, [[1, 2, 3], [4, 5, 6]]), ("island-6", 0.0, [[7, 8, 9]])]:
        raise SystemExit(f"unexpected batches {batches}")


if __name__ == "__main__":
    test_islands_at_load_are_bone_only()
    test_a_cut_through_the_neck_frees_the_peg()
    test_large_or_open_pieces_stay()
    test_request_file_matches_the_plugin_reader()
    print("ok")
