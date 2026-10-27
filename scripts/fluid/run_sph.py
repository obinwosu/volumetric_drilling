#!/usr/bin/env python3
"""Headless SPH sidecar: one occupancy bake, moving inlet, no RTX draw."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluid.occupancy import load_occupancy_dump, volume_frame_from_adf
from fluid.sph import bake_solver, step_from_drill, write_state

RUNTIME = Path("/tmp/fivrs-rtx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occupancy", type=Path, default=RUNTIME / "occupancy.bin")
    parser.add_argument("--anatomy-adf", type=Path, required=True)
    parser.add_argument("--drill-state", type=Path, default=RUNTIME / "drill_state.json")
    parser.add_argument("--state-out", type=Path, default=RUNTIME / "sph.json")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--force-emit", action="store_true")
    args = parser.parse_args()

    wp.init()
    occupied = load_occupancy_dump(args.occupancy)
    frame = volume_frame_from_adf(args.anatomy_adf, occupied.shape)
    solver, bake_ms = bake_solver(occupied, frame, downsample=args.downsample)
    print(
        f"SPH bake {occupied.shape} → {solver.proxy.shape} in {bake_ms:.1f} ms, "
        f"band {solver.proxy.band}",
        flush=True,
    )
    times = []
    for i in range(int(args.steps)):
        step_from_drill(solver, args.drill_state, force_emit=args.force_emit)
        times.append(solver.step_ms)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == args.steps:
            print(
                f"step {i + 1}/{args.steps} live {solver.live} "
                f"emitted {solver.emitted} {solver.step_ms:.2f} ms",
                flush=True,
            )
    write_state(
        args.state_out,
        solver,
        {
            "bake_ms": bake_ms,
            "steps": int(args.steps),
            "force_emit": bool(args.force_emit),
            "step_ms_avg": float(sum(times) / max(len(times), 1)),
            "wall": time.time(),
        },
    )
    print(f"wrote {args.state_out}", flush=True)
    print(json.dumps(solver.snapshot(), indent=2), flush=True)


if __name__ == "__main__":
    main()
