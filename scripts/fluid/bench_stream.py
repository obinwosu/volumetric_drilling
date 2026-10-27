#!/usr/bin/env python3
"""Stream cost without the bone renderer. Times compact draw vs a full-buffer upload."""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fluid.rtx_stream import MAX_PARTICLES, PrescribedStream, rate_rows


def _ms(fn):
    start = time.perf_counter()
    result = fn()
    wp.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def main():
    wp.init()
    tip = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    quat = (0.0, 0.0, 0.0, 1.0)
    idle = PrescribedStream(cc_per_s=1.0, max_particles=MAX_PARTICLES, gravity=False)
    _, idle_step_ms = _ms(lambda: idle.step(0.016, tip, quat, False))
    idle_live, idle_compact_ms = _ms(idle.compact)

    fat = wp.zeros(131072, dtype=wp.vec3)
    _, fat_download_ms = _ms(lambda: fat.numpy())

    emitting = PrescribedStream(cc_per_s=1.0, max_particles=MAX_PARTICLES, gravity=False)
    emitting.step(0.05, tip, quat, True)
    live, compact_ms = _ms(emitting.compact)
    _, live_download_ms = _ms(lambda: emitting.packed.numpy()[:live].copy())

    report = {
        "rate_rows": rate_rows(),
        "capacity": MAX_PARTICLES,
        "idle_live": idle_live,
        "idle_step_ms": round(idle_step_ms, 3),
        "idle_compact_ms": round(idle_compact_ms, 3),
        "old_131k_numpy_ms": round(fat_download_ms, 3),
        "emit_0.05s_at_1ccs_live": live,
        "compact_ms": round(compact_ms, 3),
        "live_download_ms": round(live_download_ms, 3),
        "note": (
            "ViewerRTX.log_points calls points.numpy() every frame. "
            "The previous 131072-point upload did that while the drill was off."
        ),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
