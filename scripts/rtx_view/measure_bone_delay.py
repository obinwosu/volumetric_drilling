#!/usr/bin/env python3
"""Measure the current AMBF-to-RTX bone update, then print the breakdown.

Each trial restores the original volume, reloads that bone into RTX, and
deletes a predetermined set of occupied voxels. The simulator clears those
voxels and publishes them with the same calls the drill uses. The camera and
the RTX render settings are left alone.

Writing mesh attributes is recorded separately from the first completed RTX
frame. Monitor vblank is not measurable here.
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fivrs_rtx_view import _load_occupancy

RUNTIME = pathlib.Path("/tmp/fivrs-rtx")
MAGIC = 0x51455346
REPEATS = 3


def _write_sequence(batches):
    blob = bytearray()
    blob += struct.pack("<II", MAGIC, len(batches))
    for name, release_s, indices in batches:
        raw = name.encode("ascii")
        blob += struct.pack("<I", len(raw))
        blob += raw
        blob += struct.pack("<dI", float(release_s), len(indices))
        if len(indices):
            blob += np.ascontiguousarray(indices, dtype="<i4").tobytes()
    path = RUNTIME / "delete_sequence.bin"
    temporary = RUNTIME / "delete_sequence.tmp"
    temporary.write_bytes(blob)
    temporary.replace(path)


def _occupied_box(occupied, seed, radius):
    nx, ny, nz = occupied.shape
    x0, y0, z0 = seed
    found = []
    for x in range(max(0, x0 - radius), min(nx, x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(ny, y0 + radius + 1)):
            for z in range(max(0, z0 - radius), min(nz, z0 + radius + 1)):
                if occupied[x, y, z]:
                    found.append((x, y, z))
    return np.asarray(found, dtype=np.int32)


def _occupied_ball(occupied, seed, radius):
    nx, ny, nz = occupied.shape
    x0, y0, z0 = seed
    limit = radius * radius
    found = []
    for x in range(max(0, x0 - radius), min(nx, x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(ny, y0 + radius + 1)):
            for z in range(max(0, z0 - radius), min(nz, z0 + radius + 1)):
                if (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 > limit:
                    continue
                if occupied[x, y, z]:
                    found.append((x, y, z))
    return np.asarray(found, dtype=np.int32)


def _drill_path(occupied, seed):
    """Forty bites of ten voxels, one every 25 ms, walking through solid bone."""
    nx, ny, nz = occupied.shape
    x, y, z = seed
    taken = []
    batches = []
    for batch_index in range(40):
        bite = []
        while len(bite) < 10:
            if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
                break
            if occupied[x, y, z] and (x, y, z) not in taken:
                bite.append((x, y, z))
                taken.append((x, y, z))
            x += 1
            if x >= nx:
                x = seed[0]
                y += 1
            if y >= ny:
                y = seed[1]
                z += 1
        if len(bite) < 10:
            raise RuntimeError("ran out of occupied voxels for the drilling sequence")
        batches.append((f"hold-{batch_index:02d}", batch_index * 0.025, np.asarray(bite, dtype=np.int32)))
    return batches


def _choose_sites(occupied):
    from scipy.ndimage import uniform_filter

    density = uniform_filter(occupied.astype(np.float32), size=9)
    density = np.where(occupied, density, 0.0)
    # The crop face is solid, so the raw peak sits on the boundary. Stay
    # inside the scan, where a burr would actually cut.
    margin = 32
    density[:margin] = 0.0
    density[-margin:] = 0.0
    density[:, :margin] = 0.0
    density[:, -margin:] = 0.0
    density[:, :, :margin] = 0.0
    density[:, :, -margin:] = 0.0
    seed = tuple(int(v) for v in np.unravel_index(int(np.argmax(density)), density.shape))
    small = _occupied_box(occupied, seed, 2)
    large = _occupied_ball(occupied, seed, 14)
    stream = _drill_path(occupied, seed)
    if len(small) < 20 or len(large) < 1000:
        raise RuntimeError(f"seed {seed} is not solid enough: small {len(small)} large {len(large)}")
    return seed, small, large, stream


def _wait_for(predicate, timeout, label):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError(label)


def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _restore_bone():
    done = RUNTIME / "restore.done"
    if done.exists():
        done.unlink()
    ready = RUNTIME / "mesh_ready.json"
    previous = 0
    if ready.is_file():
        previous = int(json.loads(ready.read_text()).get("generation") or 0)
    (RUNTIME / "restore.request").write_text("restore\n")
    _wait_for(lambda: done.is_file(), 15.0, "AMBF did not restore the volume")
    (RUNTIME / "reload.request").write_text("reload\n")

    def generation():
        if not ready.is_file():
            return None
        try:
            payload = json.loads(ready.read_text())
        except json.JSONDecodeError:
            return None
        if int(payload.get("generation") or 0) > previous:
            return payload
        return None

    _wait_for(generation, 90.0, "RTX did not finish reloading the restored bone")
    time.sleep(0.5)


def _run_batches(name, batches):
    for row_name, _release, indices in batches:
        if not row_name.startswith(name):
            raise RuntimeError(f"batch {row_name} is outside trial {name}")
    before = len(_read_jsonl(RUNTIME / "ambf_timing.jsonl"))
    _write_sequence(batches)
    expected = {item[0] for item in batches}

    def published():
        rows = _read_jsonl(RUNTIME / "ambf_timing.jsonl")[before:]
        found = {row["id"] for row in rows}
        if expected <= found:
            return rows
        return None

    ambf_rows = _wait_for(published, 20.0, f"{name} was not published")
    serials = {row["id"]: int(row["serial"]) for row in ambf_rows if row["id"] in expected}

    def presented():
        rows = _read_jsonl(RUNTIME / "rtx_timing.jsonl")
        have = set()
        for row in rows:
            if row.get("kind") != "presented":
                continue
            have.update(int(serial) for serial in row.get("serials") or [])
        if set(serials.values()) <= have:
            return rows
        return None

    try:
        rtx_rows = _wait_for(presented, 60.0, f"{name} did not reach a completed RTX frame")
        complete = True
    except TimeoutError:
        rtx_rows = _read_jsonl(RUNTIME / "rtx_timing.jsonl")
        complete = False
    return {
        "name": name,
        "complete": complete,
        "ambf": [row for row in ambf_rows if row["id"] in expected],
        "rtx": rtx_rows,
        "serials": serials,
    }


def _match(rows, kind, serial):
    found = [row for row in rows if row.get("kind") == kind and serial in {int(s) for s in row.get("serials") or []}]
    if kind == "receipt":
        found = [row for row in rows if row.get("kind") == "receipt" and int(row.get("serial", -1)) == serial]
    return found[-1] if found else None


def _batch_record(trial, ambf):
    serial = int(ambf["serial"])
    rows = trial["rtx"]
    receipt = _match(rows, "receipt", serial)
    surface = _match(rows, "surface", serial)
    presented = _match(rows, "presented", serial)
    record = {
        "id": ambf["id"],
        "n": ambf["n"],
        "serial": serial,
        "ambf_clear_ms": ambf["clear_ms"],
        "complete": presented is not None and presented.get("display_accept_wall") is not None,
    }
    if receipt is None or surface is None:
        record["missing"] = "receipt" if receipt is None else "surface"
        return record
    record["transport_ms"] = (receipt["wall"] - ambf["wall_publish"]) * 1e3
    record["queue_wait_ms"] = surface.get("queue_wait_ms")
    record["voxel_clear_ms"] = surface.get("voxel_clear_ms")
    record["copy_ms"] = surface.get("copy_ms")
    record["extract_ms"] = surface.get("extract_ms")
    record["edge_ms"] = surface.get("edge_ms")
    record["rewrite_ms"] = surface.get("rewrite_ms")
    record["stitch_ms"] = surface.get("stitch_ms")
    record["smooth_ms"] = surface.get("smooth_ms")
    record["prepare_ms"] = surface.get("prepare_ms")
    record["kept_tris"] = surface.get("kept_tris")
    record["new_tris"] = surface.get("new_tris")
    record["blocks"] = surface.get("blocks")
    record["triangles"] = surface.get("triangles")
    record["batches_in_job"] = surface.get("batches")
    record["replace_calls"] = surface.get("replace_calls")
    record["pending_batches"] = receipt.get("pending_batches")
    record["inbox"] = receipt.get("inbox")
    record["worker_busy"] = receipt.get("worker_busy")
    record["reason"] = surface.get("reason")
    if presented is None:
        record["missing"] = "presented"
        return record
    record["host_log_ms"] = presented.get("host_log_ms")
    record["upload_spread_ms"] = presented.get("upload_spread_ms")
    record["attribute_write_ms"] = presented.get("attribute_write_ms")
    record["frame_ms_max"] = presented.get("frame_ms_max")
    record["stalls_over_50ms"] = presented.get("stalls_over_50ms")
    record["stalls_over_100ms"] = presented.get("stalls_over_100ms")
    record["attribute_wall"] = presented.get("attribute_wall")
    record["product_ready_wall"] = presented.get("product_ready_wall")
    record["display_accept_wall"] = presented.get("display_accept_wall")
    if presented.get("display_accept_wall") is None:
        record["missing"] = "display_accept"
        return record
    record["gpu_frame_ms"] = (presented["product_ready_wall"] - presented["attribute_wall"]) * 1e3
    record["display_handoff_ms"] = (presented["display_accept_wall"] - presented["product_ready_wall"]) * 1e3
    record["total_ms"] = (presented["display_accept_wall"] - ambf["wall_clear"]) * 1e3
    accounted = (
        record["ambf_clear_ms"]
        + record["transport_ms"]
        + record["queue_wait_ms"]
        + record["voxel_clear_ms"]
        + record["copy_ms"]
        + record["extract_ms"]
        + record["edge_ms"]
        + record["rewrite_ms"]
        + record["smooth_ms"]
        + record["prepare_ms"]
        + record["upload_spread_ms"]
        + record["gpu_frame_ms"]
        + record["display_handoff_ms"]
    )
    record["unaccounted_ms"] = record["total_ms"] - accounted
    return record


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _family_summary(name, records):
    done = [row for row in records if row.get("total_ms") is not None]
    totals = [row["total_ms"] for row in done]
    summary = {
        "name": name,
        "trials": len(records),
        "completed": len(done),
        "median_ms": _median(totals) if totals else None,
        "worst_ms": max(totals) if totals else None,
        "best_ms": min(totals) if totals else None,
    }
    if not done:
        summary["records"] = records
        return summary
    worst = max(done, key=lambda row: row["total_ms"])
    median_row = min(done, key=lambda row: abs(row["total_ms"] - summary["median_ms"]))
    summary["median_trial"] = median_row["id"]
    summary["worst_trial"] = worst["id"]
    summary["median_breakdown_ms"] = _breakdown(median_row)
    summary["worst_breakdown_ms"] = _breakdown(worst)
    summary["frame_ms_max"] = max(row.get("frame_ms_max") or 0.0 for row in done)
    summary["stalls_over_50ms"] = max(row.get("stalls_over_50ms") or 0 for row in done)
    summary["stalls_over_100ms"] = max(row.get("stalls_over_100ms") or 0 for row in done)
    summary["max_pending_batches"] = max(row.get("pending_batches") or 0 for row in done)
    summary["max_inbox"] = max(row.get("inbox") or 0 for row in done)
    summary["max_batches_in_job"] = max(row.get("batches_in_job") or 0 for row in done)
    summary["max_replace_calls"] = max(row.get("replace_calls") or 0 for row in done)
    summary["max_queue_wait_ms"] = max(row.get("queue_wait_ms") or 0.0 for row in done)
    summary["worker_busy_at_receipt"] = any(row.get("worker_busy") for row in done)
    summary["backlog"] = bool(
        summary["max_batches_in_job"] > 1
        or summary["max_inbox"] > 0
        or summary["max_pending_batches"] > 1
        or summary["worker_busy_at_receipt"]
        or summary["max_queue_wait_ms"] > 50.0
    )
    summary["records"] = records
    return summary


def _breakdown(row):
    keys = (
        "ambf_clear_ms",
        "transport_ms",
        "queue_wait_ms",
        "voxel_clear_ms",
        "copy_ms",
        "extract_ms",
        "edge_ms",
        "rewrite_ms",
        "smooth_ms",
        "prepare_ms",
        "blocks",
        "upload_spread_ms",
        "host_log_ms",
        "attribute_write_ms",
        "gpu_frame_ms",
        "display_handoff_ms",
        "unaccounted_ms",
        "total_ms",
    )
    return {key: row.get(key) for key in keys}


def main():
    adf = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("ADF/volume_256_RT143.yaml")
    print("Loading occupancy to choose voxels", flush=True)
    occupied, _colors = _load_occupancy(adf)
    seed, small, large, stream = _choose_sites(occupied)
    print(
        f"Seed {seed}  small {len(small)} voxels  large {len(large)} voxels  "
        f"stream {len(stream)} x {len(stream[0][2])} voxels",
        flush=True,
    )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for name in ("ambf_timing.jsonl", "rtx_timing.jsonl", "restore.request", "reload.request", "delete_sequence.bin"):
        path = RUNTIME / name
        if path.exists():
            path.unlink()

    print("Warmup", flush=True)
    _restore_bone()
    _run_batches("warmup", [("warmup", 0.0, small)])

    families = {
        "small": [],
        "large": [],
        "stream": [],
    }
    for repeat in range(1, REPEATS + 1):
        for kind, batches in (
            ("small", [("small", 0.0, small)]),
            ("large", [("large", 0.0, large)]),
            ("stream", [(f"stream-{index:02d}", release, indices) for index, (_name, release, indices) in enumerate(stream)]),
        ):
            print(f"{kind} {repeat}", flush=True)
            _restore_bone()
            renamed = []
            for batch_name, release, indices in batches:
                renamed.append((f"{kind}-{repeat:02d}-{batch_name}", release, indices))
            # Single-batch names would become small-01-small. Keep them short.
            if kind != "stream":
                renamed = [(f"{kind}-{repeat:02d}", 0.0, batches[0][2])]
            trial = _run_batches(f"{kind}-{repeat:02d}", renamed)
            for ambf in trial["ambf"]:
                families[kind].append(_batch_record(trial, ambf))

    report = {
        "seed": list(seed),
        "small_voxels": int(len(small)),
        "large_voxels": int(len(large)),
        "stream_batches": len(stream),
        "stream_voxels_per_batch": int(len(stream[0][2])),
        "stream_spacing_s": 0.025,
        "repeats_after_warmup": REPEATS,
        "not_measured": [
            "Monitor vblank and the scanout of the updated pixels are not visible to this process.",
            "display_accept_wall is ViewerRTX._accept_render for the first completed render product submitted after the last mesh attributes of that batch were written.",
            "attribute_write_ms is _update_ovrtx_mesh_points. host_log_ms is log_mesh. Neither one means the updated bone was shown.",
            "The AMBF window's own 3D-texture upload is a separate graphics-thread update and is not on this RTX path.",
        ],
        "small": _family_summary("small", families["small"]),
        "large": _family_summary("large", families["large"]),
        "stream_first_bite": _family_summary(
            "stream_first",
            [row for row in families["stream"] if row["id"].endswith("-stream-00")],
        ),
        "stream_last_bite": _family_summary(
            "stream_last",
            [row for row in families["stream"] if row["id"].endswith("-stream-39")],
        ),
        "stream_all_bites": _family_summary("stream_all", families["stream"]),
    }
    (RUNTIME / "delay_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in report if key != "small" and not str(key).startswith("stream") and key != "large"}, indent=2))
    for key in ("small", "large", "stream_first_bite", "stream_last_bite", "stream_all_bites"):
        item = report[key]
        printable = {name: item[name] for name in item if name != "records"}
        print(f"\n{key}")
        print(json.dumps(printable, indent=2))
    print(f"\nWrote {RUNTIME / 'delay_report.json'}", flush=True)


if __name__ == "__main__":
    main()
