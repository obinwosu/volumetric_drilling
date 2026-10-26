#!/usr/bin/env python3
"""Time a known deletion under three hand-offs.

The deletion is 2,000 occupied voxels. Physics mode publishes one message per
cleared batch, which is one voxel per physics step. Graphics mode clears on
that same schedule and publishes the accumulated voxels once per graphics
frame. The drop trials omit 40 of those voxels from the messages. Dump
recovery reloads the occupancy file. Index recovery sends the missed voxels.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from measure_bone_delay import _read_jsonl, _restore_bone, _wait_for, _write_sequence

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
TOTAL = 2000
DROP_AT = 200
DROP_COUNT = 40
REPEATS = 2


def _walk(occupied, seed, count):
    nx, ny, nz = occupied.shape
    x, y, z = seed
    found = []
    seen = set()
    while len(found) < count:
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            raise RuntimeError("ran out of voxels")
        key = (int(x), int(y), int(z))
        if occupied[key] and key not in seen:
            found.append(key)
            seen.add(key)
        x += 1
        if x >= nx - 32:
            x = seed[0]
            y += 1
        if y >= ny - 32:
            y = seed[1]
            z += 1
    return found


def _seed(occupied):
    from scipy.ndimage import uniform_filter

    density = uniform_filter(occupied.astype(np.float32), size=9)
    density = np.where(occupied, density, 0.0)
    margin = 32
    density[:margin] = 0.0
    density[-margin:] = 0.0
    density[:, :margin] = 0.0
    density[:, -margin:] = 0.0
    density[:, :, :margin] = 0.0
    density[:, :, -margin:] = 0.0
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(density)), density.shape))


def _batches(voxels, drop):
    batches = []
    cursor = 0
    release = 0.0
    step = 0.001
    while cursor < len(voxels):
        if drop and cursor == DROP_AT:
            chunk = voxels[cursor : cursor + DROP_COUNT]
            batches.append(("drop|drop", release, np.asarray(chunk, dtype=np.int32)))
            cursor += DROP_COUNT
        else:
            batches.append((f"v{cursor:04d}", release, np.asarray([voxels[cursor]], dtype=np.int32)))
            cursor += 1
        release += step
    return batches


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _summarize(ambf_rows, rtx_rows):
    published = [row for row in ambf_rows if not row.get("dropped")]
    dropped = [row for row in ambf_rows if row.get("dropped")]
    presented = [
        row
        for row in rtx_rows
        if row.get("kind") == "presented" and row.get("display_accept_wall") is not None
    ]
    by_serial = {}
    for event in presented:
        for serial in event.get("serials") or []:
            by_serial.setdefault(int(serial), event)
    lags = []
    for row in published:
        event = by_serial.get(int(row.get("message_serial") or 0))
        if event is None:
            continue
        lags.append((event["display_accept_wall"] - row["wall_clear"]) * 1e3)
    surfaces = [row for row in rtx_rows if row.get("kind") == "surface"]
    recoveries = [row for row in surfaces if row.get("reason") in ("gap", "catchup")]
    shown = [row for row in presented if row.get("reason") in ("gap", "catchup")]
    reload_ms = None
    settled_ms = None
    if dropped and shown:
        origin = dropped[0]["wall_clear"]
        reload_ms = (shown[0]["display_accept_wall"] - origin) * 1e3
        settled_ms = (shown[-1]["display_accept_wall"] - origin) * 1e3
    first = recoveries[0] if recoveries else {}
    publish_span = None
    if published:
        publish_span = (published[-1]["wall_publish"] - published[0]["wall_clear"]) * 1e3
    messages = len({int(row["message_serial"]) for row in published if row.get("message_serial")})
    return {
        "voxels_logged": int(sum(row["n"] for row in ambf_rows)),
        "dropped_voxels": int(sum(row["n"] for row in dropped)),
        "messages": messages,
        "publish_span_ms": publish_span,
        "displayed_batches": len(lags),
        "display_median_ms": _percentile(lags, 0.5),
        "display_worst_ms": max(lags) if lags else None,
        "recoveries": len(recoveries),
        "recovery_reason": first.get("reason"),
        "reload_ms": reload_ms,
        "settled_ms": settled_ms,
        "load_ms": first.get("load_ms"),
        "diff_ms": first.get("diff_ms"),
        "extract_ms": first.get("extract_ms"),
        "worker_ms": (
            ((first.get("worker_done") or 0) - (first.get("worker_start") or 0)) * 1e3 if first else None
        ),
        "blocks": first.get("blocks"),
    }


def _run_once(name, publish, recovery, drop, voxels):
    (RUNTIME / "publish_mode").write_text(publish + "\n")
    (RUNTIME / "recovery_mode").write_text(recovery + "\n")
    for leftover in ("resync.request", "catchup.request", "catchup.bin", "delete_sequence.bin"):
        path = RUNTIME / leftover
        if path.exists():
            path.unlink()
    _restore_bone()
    batches = _batches(voxels, drop)
    expected = {("drop" if item[0].endswith("|drop") else item[0]) for item in batches}
    ambf_before = len(_read_jsonl(RUNTIME / "ambf_timing.jsonl"))
    rtx_before = len(_read_jsonl(RUNTIME / "rtx_timing.jsonl"))
    started = time.time()
    _write_sequence(batches)

    def published():
        rows = _read_jsonl(RUNTIME / "ambf_timing.jsonl")[ambf_before:]
        found = {row["id"] for row in rows}
        if expected <= found:
            return rows
        return None

    try:
        ambf_rows = _wait_for(published, 20.0, f"{name} did not finish publishing")
    except TimeoutError:
        ambf_rows = _read_jsonl(RUNTIME / "ambf_timing.jsonl")[ambf_before:]
    ambf_rows = [row for row in ambf_rows if row["id"] in expected or row["id"] == "drop"]

    def displayed():
        rows = _read_jsonl(RUNTIME / "rtx_timing.jsonl")[rtx_before:]
        summary = _summarize(ambf_rows, rows)
        if summary["dropped_voxels"] and summary["reload_ms"] is None:
            return None
        if summary["displayed_batches"] < max(1, int(0.9 * len([row for row in ambf_rows if not row.get("dropped")]))):
            return None
        return rows

    try:
        rtx_rows = _wait_for(displayed, 25.0, f"{name} did not reach the RTX window")
        complete = True
    except TimeoutError:
        rtx_rows = _read_jsonl(RUNTIME / "rtx_timing.jsonl")[rtx_before:]
        complete = False
    summary = _summarize(ambf_rows, rtx_rows)
    summary["complete"] = complete
    summary["wall_s"] = time.time() - started
    return summary


def main():
    from fivrs_rtx_view import _load_occupancy

    adf = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("ADF/volume_256_RT143.yaml")
    print("Loading occupancy", flush=True)
    occupied, _colors = _load_occupancy(adf)
    seed = _seed(occupied)
    voxels = _walk(occupied, seed, TOTAL)
    print(f"Seed {seed}  {len(voxels)} voxels, drop {DROP_COUNT} at index {DROP_AT}", flush=True)
    plan = (
        ("physics_steady", "physics", "dump", False),
        ("graphics_steady", "graphics", "dump", False),
        ("physics_dump", "physics", "dump", True),
        ("graphics_dump", "graphics", "dump", True),
        ("physics_indices", "physics", "indices", True),
        ("graphics_indices", "graphics", "indices", True),
    )
    report = {"seed": list(seed), "voxels": TOTAL, "dropped_voxels": DROP_COUNT, "repeats": REPEATS, "trials": {}}
    for name, publish, recovery, drop in plan:
        trials = []
        for repeat in range(1, REPEATS + 1):
            print(f"{name} {repeat}", flush=True)
            trial = _run_once(f"{name}-{repeat}", publish, recovery, drop, voxels)
            trial["repeat"] = repeat
            trials.append(trial)
            print(json.dumps(trial), flush=True)
        report["trials"][name] = trials
    (RUNTIME / "reload_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {RUNTIME / 'reload_report.json'}", flush=True)


if __name__ == "__main__":
    main()
