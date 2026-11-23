#!/usr/bin/env python3
"""Turn a recorded session into a coded case a phone can review.

    python3 scripts/remote_review/package_case.py <session folder> [--base <recording base>]

Only what the page needs is written: structure labels at full resolution (and a half-resolution
fallback for phones short of memory), when each voxel was drilled, the drill pose, the curves,
a small H.264 video, and times counted from the session start. Names,
dates, anatomy names and file paths stay out; the link to the source is in private/cases.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import secrets
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "session_review"))
import remote_store as store  # noqa: E402
import session_data as data  # noqa: E402

BLOCK = 2
AIR, NEVER = 0, 65535
SMALL_VOXELS = 30000  # smaller structures keep every block they touch, so thin nerves survive
PATH_HZ = 20.0
VIDEO_FPS = 20
VIDEO_SIZE = (960, 540)
VIDEO_KBPS = 750
MAX_CURVE_POINTS = 12000


def removal_times(session) -> tuple[np.ndarray, np.ndarray]:
    """(structure ids, u16 time each voxel is drilled: 0 air, 65535 never) of the only epoch."""
    voxels = session.voxels
    if len(voxels.epochs) != 1:
        raise ValueError("This session has a volume reset, which remote review does not handle yet.")
    voxels.structures_at(0.0)
    ids = voxels._ids
    order = voxels._order
    times = np.full(ids.shape, NEVER, np.uint16)
    times[ids == 0] = AIR
    removed = order < np.iinfo(np.int64).max
    if removed.any():
        firsts = voxels.batches[:, voxels.col["first"]].astype(np.int64)
        batch = np.clip(np.searchsorted(firsts, order[removed], side="right") - 1, 0, len(firsts) - 1)
        seconds = np.clip(voxels.times[batch] - session.t0, 0.0, session.duration)
        times[removed & (ids > 0)] = encode_time(seconds[ids[removed] > 0], session.duration)
    return ids, times


def encode_time(seconds, duration):
    return np.clip(np.rint(np.asarray(seconds) / max(duration, 1e-6) * 65533.0) + 1, 1, 65534).astype(np.uint16)


def _blocks(array):
    shape = np.array(array.shape)
    padded = np.pad(array, [(0, int(n % BLOCK)) for n in shape])
    nx, ny, nz = np.array(padded.shape) // BLOCK
    return padded.reshape(nx, BLOCK, ny, BLOCK, nz, BLOCK).transpose(0, 2, 4, 1, 3, 5).reshape(nx, ny, nz, BLOCK ** 3)


def downsample(ids, times, table):
    """Half resolution. Large structures need half a block; small ones keep any block they touch."""
    label_blocks, time_blocks = _blocks(ids), _blocks(times)
    out_ids = np.zeros(label_blocks.shape[:3], np.uint8)
    out_times = np.zeros(label_blocks.shape[:3], np.uint16)
    for entry in sorted(table, key=lambda e: -e["voxels"]):
        same = label_blocks == entry["id"]
        needed = 1 if entry["voxels"] < SMALL_VOXELS else BLOCK ** 3 // 2
        chosen = same.sum(-1) >= needed
        if not chosen.any():
            continue
        alive = np.where(same[chosen], time_blocks[chosen], 0)
        # The block goes when fewer than `needed` of its voxels are left.
        rank = BLOCK ** 3 - needed
        out_times[chosen] = np.partition(alive, rank, axis=1)[:, rank]
        out_ids[chosen] = entry["id"]
    return out_ids, out_times


def write_video(session, path) -> None:
    width, height = VIDEO_SIZE
    pipeline = (
        f"fdsrc fd=0 ! rawvideoparse width={width} height={height} format=bgr framerate={VIDEO_FPS}/1 "
        f"! videoconvert ! video/x-raw,format=I420 "
        f"! x264enc speed-preset=veryfast bitrate={VIDEO_KBPS} key-int-max={VIDEO_FPS} bframes=0 "
        f"! video/x-h264,profile=main ! h264parse ! mp4mux faststart=true ! filesink location={path}")
    process = subprocess.Popen(["gst-launch-1.0", "-q", *pipeline.split()], stdin=subprocess.PIPE)
    last, resized = -1, None
    try:
        for k in range(int(session.duration * VIDEO_FPS) + 1):
            index = session.video.index_at(k / VIDEO_FPS)
            if index != last:
                resized = cv2.resize(session.video.frame(index, session.video.eyes[0]), VIDEO_SIZE,
                                     interpolation=cv2.INTER_AREA)
                last = index
            process.stdin.write(resized.tobytes())
            if k % (VIDEO_FPS * 60) == 0:
                print(f"video {k / VIDEO_FPS:.0f} s of {session.duration:.0f} s", flush=True)
    finally:
        process.stdin.close()
    if process.wait() != 0 or not path.exists():
        raise RuntimeError("The video encoder failed.")


def to_voxels(session, points):
    """World points to the page's voxel grid, where voxel i is centred at i + 0.5."""
    volume = session.meta["volume"]
    rotation, position = np.asarray(volume["R_world_volume"], float), np.asarray(volume["p_world_volume"], float)
    dims = np.asarray(volume["dimensions_sim"], float)
    local = (np.asarray(points, float) - position) @ rotation
    return (local + dims / 2.0) / dims * np.asarray(session.voxels.shape, float) + 0.5


def drill_track(session, moments):
    """Per moment: burr centre in voxels, then the unit shaft direction in the volume's axes."""
    drill = session.drill
    position = np.column_stack([np.interp(moments, session.drill_t, drill[:, 2 + k]) for k in range(3)])
    nearest = np.clip(np.searchsorted(session.drill_t, moments), 0, len(drill) - 1)
    x, y, z, w = drill[nearest, 5], drill[nearest, 6], drill[nearest, 7], drill[nearest, 8]
    axis = np.column_stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    axis = axis @ np.asarray(session.meta["volume"]["R_world_volume"], float)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    return np.column_stack([to_voxels(session, position), axis])


def thin_curve(channel):
    """(t, y) with at most MAX_CURVE_POINTS, keeping each stretch's lowest and highest value."""
    keep = np.isfinite(channel.t) & np.isfinite(channel.y)
    t, y = channel.t[keep], channel.y[keep]
    if channel.step and len(t) > 1:
        changed = np.r_[True, y[1:] != y[:-1]]
        changed[-1] = True
        t, y = t[changed], y[changed]
    if len(t) <= MAX_CURVE_POINTS:
        return t, y
    if channel.step:
        index = np.unique(np.linspace(0, len(t) - 1, MAX_CURVE_POINTS).astype(np.int64))
        return t[index], y[index]
    cuts = np.unique(np.linspace(0, len(t), MAX_CURVE_POINTS // 2, endpoint=False).astype(np.int64))
    low, high = np.minimum.reduceat(y, cuts), np.maximum.reduceat(y, cuts)
    return np.repeat(t[cuts], 2), np.column_stack([low, high]).ravel()


def write_curves(session, path) -> list[dict]:
    described, offset = [], 0
    with gzip.open(path, "wb", compresslevel=6) as handle:
        for key, channel in session.channels.items():
            t, y = thin_curve(channel)
            if not len(t):
                continue
            handle.write(t.astype("<f4").tobytes() + y.astype("<f4").tobytes())
            described.append({"key": key, "label": channel.label, "unit": channel.unit, "group": channel.group,
                              "step": bool(channel.step), "count": int(len(t)), "offset": offset})
            offset += 8 * len(t)
    return described


def case_description(session, code, table, half_shape, curves) -> dict:
    volume = session.meta["volume"]
    dims = np.asarray(volume["dimensions_sim"], float)
    spacing_mm = dims / np.asarray(session.voxels.shape, float) * data.dataset.MM_PER_SIM
    burrs = [{"t": start, "stop": stop, "size_mm": size, "type": kind} for start, stop, size, kind in session.burr_spans()]
    view = None
    camera = session.camera_at(0.0)
    if camera is not None:
        _, look, up, fov = camera
        rotation = np.asarray(volume["R_world_volume"], float)
        view = {"look": (np.asarray(look) @ rotation).tolist(), "up": (np.asarray(up) @ rotation).tolist(),
                "fov": float(fov)}
    summary = session.summary
    return {"code": code, "version": secrets.token_hex(4), "side": session.meta["specimen"].get("laterality") or "", "duration": session.duration,
            "shape": [int(n) for n in session.voxels.shape], "half_shape": [int(n) for n in half_shape],
            "spacing_mm": spacing_mm.tolist(), "burrs": burrs, "view": view, "path_hz": PATH_HZ,
            "video_fps": VIDEO_FPS, "strokes": int(summary.get("strokes") or 0),
            "stroke_spans": [[round(a, 3), round(b, 3)] for a, b in session.stroke_spans()],
            "voxels_removed": int(summary.get("voxels_removed") or 0),
            "inner_ear_voxels_removed": int(summary.get("inner_ear_voxels_removed") or 0), "curves": curves,
            "structures": [{"id": e["id"], "name": e["name"], "color": list(e["color"])} for e in table]}


def package(folder, base, video=True) -> str:
    root = store.root_of(base)
    folder = pathlib.Path(folder).resolve()
    code = store.code_for_source(root, folder) or store.new_code(root)
    session = data.ReviewSession(folder)
    work = root / "cases" / f".{code}.building"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(mode=0o700)
    try:
        if video and (session.video is None or not session.video.count):
            raise ValueError("This session has no video.")
        ids, times = removal_times(session)
        table = [dict(e) for e in session.voxels.structures]
        small_ids, small_times = downsample(ids, times, table)
        present = set(np.unique(small_ids).tolist())
        missing = [e["name"] for e in table if e["id"] not in present]
        if missing:
            raise RuntimeError(f"Lost in packaging: {missing}")
        for name, array, kind in (("labels", ids, "<u1"), ("times", times, "<u2"),
                                  ("labels_half", small_ids, "<u1"), ("times_half", small_times, "<u2")):
            with gzip.open(work / f"{name}.bin.gz", "wb", compresslevel=6) as handle:
                handle.write(array.astype(kind).tobytes(order="F"))
        moments = np.arange(0.0, session.duration + 1.0 / PATH_HZ, 1.0 / PATH_HZ)
        with gzip.open(work / "path.bin.gz", "wb") as handle:
            handle.write(drill_track(session, moments).astype("<f4").tobytes())
        curves = write_curves(session, work / "curves.bin.gz")
        earlier_video = root / "cases" / code / "video.mp4"
        if video and earlier_video.exists():  # a session's recording never changes
            shutil.copy2(earlier_video, work / "video.mp4")
        elif video:
            write_video(session, work / "video.mp4")
        (work / "case.json").write_text(json.dumps(case_description(session, code, table, small_ids.shape, curves)))
        final = root / "cases" / code
        shutil.rmtree(final, ignore_errors=True)
        work.replace(final)
        store.register_case(root, code, folder)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        session.close()
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session")
    parser.add_argument("--base", default=None)
    args = parser.parse_args()
    folder = pathlib.Path(args.session).resolve()
    base = args.base or next(p.parent for p in folder.parents if p.name in {n for n, _ in data.DATA_FOLDERS})
    print("@code", package(folder, base), flush=True)


if __name__ == "__main__":
    main()
