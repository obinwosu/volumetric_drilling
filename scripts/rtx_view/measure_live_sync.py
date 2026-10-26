#!/usr/bin/env python3
"""Time the live graphics-frame deletion path, including a withheld batch.

Counts are separate:

- messages: ROS batches AMBF published (one per graphics frame)
- scripted groups: trial rows, several of which share one message
- frames: presented RTX events that contain those messages
- voxels: coordinates cleared in AMBF

A displayed frame is not proof that a withheld voxel was meshed. After the
sequence settles, the RTX worker writes its occupancy and every cleared
coordinate is checked there.
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
REPEATS = 2
DROP_AT = 200
DROP_COUNT = 40


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "median_ms": _percentile(values, 0.5),
        "p90_ms": _percentile(values, 0.9),
        "worst_ms": max(values),
    }


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


def _script(voxels, per_step, drop):
    batches = []
    release = 0.0
    cursor = 0
    step_index = 0
    while cursor < len(voxels):
        if drop and cursor == DROP_AT:
            chunk = voxels[cursor : cursor + DROP_COUNT]
            batches.append(("drop|drop", release, np.asarray(chunk, dtype=np.int32)))
            cursor += DROP_COUNT
            release += 0.001
            continue
        chunk = voxels[cursor : cursor + per_step]
        batches.append((f"v{step_index:04d}", release, np.asarray(chunk, dtype=np.int32)))
        cursor += per_step
        step_index += 1
        release += 0.001
    return batches


def _audit(voxels):
    target = RUNTIME / "rtx_occupied.npy"
    ready = target.with_suffix(".ready")
    target.unlink(missing_ok=True)
    ready.unlink(missing_ok=True)
    (RUNTIME / "rtx_occupied.request").write_text("audit\n")
    _wait_for(lambda: ready.is_file(), 30.0, "RTX did not export occupancy")
    occupied = np.load(target)
    still = 0
    for x, y, z in voxels:
        if occupied[int(x), int(y), int(z)]:
            still += 1
    return {"voxels_checked": len(voxels), "still_occupied": still}


def _summarize(ambf_rows, rtx_rows, started_wall):
    published = [row for row in ambf_rows if not row.get("dropped")]
    dropped = [row for row in ambf_rows if row.get("dropped")]
    receipts = [
        row
        for row in rtx_rows
        if row.get("kind") == "receipt" and row.get("wall", 0) >= started_wall - 1.0
    ]
    surfaces = [
        row
        for row in rtx_rows
        if row.get("kind") == "surface" and (row.get("serials") or row.get("reason") in ("gap", "catchup"))
    ]
    presented = [
        row
        for row in rtx_rows
        if row.get("kind") == "presented" and row.get("display_accept_wall")
    ]
    receipt_by_serial = {}
    for row in receipts:
        receipt_by_serial.setdefault(int(row["serial"]), row)
    surface_by_serial = {}
    for row in surfaces:
        for serial in row.get("serials") or []:
            surface_by_serial.setdefault(int(serial), row)
    presented_by_serial = {}
    for row in presented:
        for serial in row.get("serials") or []:
            presented_by_serial.setdefault(int(serial), row)

    deletion = []
    queue_wait = []
    mesh = []
    first_frame = []
    total = []
    recovery = []
    unmatched = 0
    for row in published:
        serial = int(row.get("message_serial") or 0)
        receipt = receipt_by_serial.get(serial)
        surface = surface_by_serial.get(serial)
        event = presented_by_serial.get(serial)
        if (
            receipt is None
            or surface is None
            or event is None
            or event.get("attribute_wall") is None
            or receipt.get("stamp") is None
            or surface.get("worker_start") is None
        ):
            unmatched += 1
            continue
        if receipt.get("recovered"):
            continue
        deletion.append((receipt["wall"] - row["wall_clear"]) * 1e3)
        queue_wait.append((surface["worker_start"] - receipt["stamp"]) * 1e3)
        mesh.append(float(surface.get("mesh_ms") or 0.0))
        first_frame.append((event["display_accept_wall"] - event["attribute_wall"]) * 1e3)
        total.append((event["display_accept_wall"] - row["wall_clear"]) * 1e3)
    for row in dropped:
        serial = int(row.get("message_serial") or 0)
        event = presented_by_serial.get(serial)
        if event is None:
            continue
        recovery.append((event["display_accept_wall"] - row["wall_clear"]) * 1e3)

    fallback = [
        row
        for row in surfaces
        if row.get("reason") == "gap" and row.get("worker_start")
    ]
    messages = {int(row["message_serial"]) for row in published if row.get("message_serial")}
    received = {serial for serial in messages if serial in receipt_by_serial and not receipt_by_serial[serial].get("recovered")}
    frame_ids = []
    seen_frames = set()
    for event in presented:
        serials = [int(serial) for serial in (event.get("serials") or [])]
        relevant = [serial for serial in serials if serial in messages or serial in {int(row.get("message_serial") or 0) for row in dropped}]
        if not relevant or id(event) in seen_frames or event.get("attribute_wall") is None:
            continue
        seen_frames.add(id(event))
        frame_ids.append((event["display_accept_wall"] - event["attribute_wall"]) * 1e3)
    dropped_shown = sum(1 for row in dropped if int(row.get("message_serial") or 0) in presented_by_serial)
    return {
        "scripted_groups": len(ambf_rows),
        "voxels_logged": int(sum(row["n"] for row in ambf_rows)),
        "dropped_voxels": int(sum(row["n"] for row in dropped)),
        "messages_published": len(messages),
        "messages_received": len(received),
        "frames_presented": len(frame_ids),
        "first_rendered_frame_per_frame": _stats(frame_ids),
        "groups_without_a_frame": unmatched,
        "dropped_groups_displayed": dropped_shown,
        "deletion_to_receipt": _stats(deletion),
        "queue_wait": _stats(queue_wait),
        "mesh_work": _stats(mesh),
        "first_rendered_frame": _stats(first_frame),
        "total_clear_to_display": _stats(total),
        "recovery_clear_to_display": _stats(recovery),
        "occupancy_file_fallbacks": len(fallback),
        "fallback_load_ms": fallback[0].get("load_ms") if fallback else None,
    }


def _run_once(name, voxels, per_step, drop):
    for leftover in (
        "resync.request",
        "catchup.request",
        "catchup.bin",
        "delete_sequence.bin",
        "publish_mode",
        "recovery_mode",
    ):
        path = RUNTIME / leftover
        if path.exists():
            path.unlink()
    _restore_bone()
    batches = _script(voxels, per_step, drop)
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
        ambf_rows = _wait_for(published, 30.0, f"{name} did not finish clearing")
        published_ok = True
    except TimeoutError:
        ambf_rows = _read_jsonl(RUNTIME / "ambf_timing.jsonl")[ambf_before:]
        published_ok = False
    ambf_rows = [row for row in ambf_rows if row["id"] in expected or row["id"] == "drop"]

    def displayed():
        rows = _read_jsonl(RUNTIME / "rtx_timing.jsonl")[rtx_before:]
        summary = _summarize(ambf_rows, rows, started)
        if summary["groups_without_a_frame"] > 0:
            return None
        if drop and summary["dropped_groups_displayed"] < 1:
            return None
        return rows

    try:
        rtx_rows = _wait_for(displayed, 40.0, f"{name} did not reach the RTX window")
        complete = True
    except TimeoutError:
        rtx_rows = _read_jsonl(RUNTIME / "rtx_timing.jsonl")[rtx_before:]
        complete = False
    # Let the last presented frame land in the worker before the occupancy snapshot.
    time.sleep(0.4)
    summary = _summarize(ambf_rows, _read_jsonl(RUNTIME / "rtx_timing.jsonl")[rtx_before:], started)
    summary["complete"] = complete and published_ok
    summary["wall_s"] = time.time() - started
    summary["audit"] = _audit([(int(v[0]), int(v[1]), int(v[2])) for batch in batches for v in batch[2]])
    return summary


def main():
    from fivrs_rtx_view import _load_occupancy

    adf = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("ADF/volume_256_RT143.yaml")
    print("Loading occupancy", flush=True)
    occupied, _colors = _load_occupancy(adf)
    seed = _seed(occupied)
    steady_voxels = _walk(occupied, seed, 2000)
    sustained_voxels = _walk(occupied, seed, 2000 * 6)
    print(f"Seed {seed}", flush=True)
    plan = (
        ("controlled_steady", steady_voxels, 1, False),
        ("controlled_withhold", steady_voxels, 1, True),
        ("sustained_drill", sustained_voxels, 6, False),
    )
    report = {
        "seed": list(seed),
        "definitions": {
            "messages": "ROS batches published, one per graphics frame",
            "scripted_groups": "trial rows; many share one message",
            "frames": "presented RTX events containing those batches",
            "voxels": "coordinates cleared; audit checks the RTX occupancy afterward",
            "deletion_to_receipt": "AMBF clear wall clock to the ROS callback, published groups only",
            "queue_wait": "callback to the start of the mesh rebuild that included the batch",
            "mesh_work": "extract + edge + smooth + prepare for that rebuild",
            "first_rendered_frame": "attribute write to the completed frame handed to the window",
            "recovery": "withheld clear to the frame that displayed the catch-up batch",
        },
        "trials": {},
    }
    for name, voxels, per_step, drop in plan:
        trials = []
        for repeat in range(1, REPEATS + 1):
            print(f"{name} {repeat}", flush=True)
            trial = _run_once(name, voxels, per_step, drop)
            trial["repeat"] = repeat
            trials.append(trial)
            print(json.dumps(trial), flush=True)
        report["trials"][name] = trials
    for leftover in ("publish_mode", "recovery_mode"):
        path = RUNTIME / leftover
        if path.exists():
            path.unlink()
    (RUNTIME / "live_sync_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {RUNTIME / 'live_sync_report.json'}", flush=True)


if __name__ == "__main__":
    main()
