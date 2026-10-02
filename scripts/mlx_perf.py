#!/usr/bin/env python3
"""Timing + memory harness for MLX workloads on Apple Silicon.

Import from a bench script:

    import sys
    sys.path.insert(0, "<path-to-mlx-perf-skill>/scripts")
    import mlx_perf

    def step():
        out = model(x)
        mx.eval(out)

    stats = mlx_perf.bench(step, n=30, warmup=5)
    mlx_perf.report(stats)

Or run a selftest to verify the harness works on this machine:

    python mlx_perf.py --selftest
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx


def memory_snapshot() -> dict:
    """Active (live arrays), cache (reusable buffers), peak (high-water) in MB."""
    return {
        "active_mb": round(mx.get_active_memory() / 1e6, 1),
        "cache_mb": round(mx.get_cache_memory() / 1e6, 1),
        "peak_mb": round(mx.get_peak_memory() / 1e6, 1),
    }


def bench(fn, n: int = 20, warmup: int = 10, prewarm_s: float = 1.0,
          label: str = "") -> dict:
    """Time fn() n times after warmup. fn must do its own mx.eval, or rely
    on the trailing mx.synchronize() here. Returns a JSON-friendly dict with
    per-iteration and aggregate stats plus memory high-water.

    Apple GPUs ramp clocks only under sustained load: on a cold machine a
    millisecond-scale kernel can read ~4x its steady-state time, and a
    handful of warmup iterations is not enough sustained work to spin the
    clocks up. Hence prewarm_s: keep running fn for at least this many
    seconds before the timed loop. Pass prewarm_s=0 for fns that are
    already seconds-scale. `clock_ramp_suspected` flags runs that likely
    still include ramp; when set, raise prewarm_s and re-run.
    """
    for _ in range(warmup):
        fn()
        mx.synchronize()
    if prewarm_s > 0:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < prewarm_s:
            fn()
            mx.synchronize()

    mx.reset_peak_memory()
    per_iter = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        per_iter.append(round((time.perf_counter() - t0) * 1000.0, 3))

    ordered = sorted(per_iter)
    third = max(1, n // 3)
    head = statistics.fmean(per_iter[:third])
    tail = statistics.fmean(per_iter[-third:])
    stats = {
        "label": label or getattr(fn, "__name__", "fn"),
        "n": n,
        "warmup": warmup,
        "min_ms": ordered[0],
        "p50_ms": round(statistics.median(ordered), 3),
        "p90_ms": round(ordered[max(0, int(0.9 * n) - 1)], 3),
        "mean_ms": round(statistics.fmean(per_iter), 3),
        "max_ms": ordered[-1],
        "clock_ramp_suspected": bool(head > 1.5 * tail),
        "per_iter_ms": per_iter,
    }
    stats.update(memory_snapshot())
    return stats


def report(stats: dict) -> None:
    """Human-readable one-block summary."""
    ramp = "  [clock_ramp_suspected: re-run with more warmup]" if stats.get("clock_ramp_suspected") else ""
    print(
        f"{stats['label']}: n={stats['n']} warmup={stats['warmup']}  "
        f"p50={stats['p50_ms']:.2f}ms p90={stats['p90_ms']:.2f}ms "
        f"mean={stats['mean_ms']:.2f}ms min={stats['min_ms']:.2f}ms "
        f"max={stats['max_ms']:.2f}ms  mem(active/cache/peak)="
        f"{stats['active_mb']}/{stats['cache_mb']}/{stats['peak_mb']} MB{ramp}"
    )


def _selftest() -> int:
    dev = mx.device_info()
    print(f"device: {dev.get('architecture', '?')}  mlx device ok")

    a = mx.random.normal((1024, 1024))
    b = mx.random.normal((1024, 1024))

    def matmul_1024():
        mx.eval(a @ b)

    def matmul_chain():  # heavier: forces several dispatches per eval
        c = a
        for _ in range(8):
            c = c @ b
        mx.eval(c)

    for fn, name in ((matmul_1024, "matmul_1024"), (matmul_chain, "matmul_chain_x8")):
        stats = bench(fn, n=20, label=name)
        report(stats)
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="bench two matmul workloads to verify the harness")
    ap.add_argument("--json", action="store_true", help="selftest output as JSON")
    args = ap.parse_args()
    if not args.selftest:
        ap.error("nothing to do: pass --selftest (import bench() for real workloads)")
    if args.json:
        a = mx.random.normal((1024, 1024))
        stats = bench(lambda: mx.eval(a @ a), n=20, warmup=3, label="matmul_1024")
        print(json.dumps(stats))
    else:
        return _selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
