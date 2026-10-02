#!/usr/bin/env python3
"""Build a REPORT.md skeleton from mlx_perf bench JSON stats.

Usage (from a run directory `profile/<run_name>/`):

    python <skill>/scripts/make_report.py stats/*.json \
        [--baseline ../<prev_run>/stats/decode_4k.json] > REPORT.md

Fills environment, stats tables, flags, and baseline deltas. The Diagnosis /
Ranked recommendations / Next measurement sections are left for the agent to
complete — with numbers cited, per the skill's golden rule.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def stats_row(s: dict, base: dict | None = None) -> str:
    def delta(key):
        if not base or base.get(key, 0) == 0:
            return ""
        d = (s[key] - base[key]) / base[key] * 100.0
        return f" ({d:+.1f}% vs baseline)"

    return (
        f"| {s['label']} | {s['n']} | {s['p50_ms']:.2f}{delta('p50_ms')} "
        f"| {s['p90_ms']:.2f}{delta('p90_ms')} | {s['mean_ms']:.2f} "
        f"| {s['active_mb']} | {s['cache_mb']} | {s['peak_mb']} |"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stats", nargs="+", help="bench JSON file(s) from mlx_perf.bench()")
    ap.add_argument("--baseline", help="baseline bench JSON for delta column")
    ap.add_argument("--question", default="", help="the question this run answers")
    args = ap.parse_args()

    base = load(args.baseline) if args.baseline else None
    rows = [stats_row(load(p), base) for p in args.stats]
    any_ramp = any(load(p).get("clock_ramp_suspected") for p in args.stats)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"""# Perf report — {now}

## Question

{args.question or "<state the question this run answers — 'why is X slow?'>"}

## Environment

<device (mx.device_info architecture), mlx version, model + quant + shapes,
machine thermal state (just-idle vs sustained-load), macOS version>

## Measurements

| label | n | p50 ms | p90 ms | mean ms | active MB | cache MB | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}
{f"{'⚠ clock_ramp_suspected — prewarm was too short; numbers above are inflated. Re-run with larger prewarm_s before diagnosing.' if any_ramp else ''}"
}
## Diagnosis

<match signals against references/playbook.md; name 1–2 dominant causes,
citing the exact numbers that identified them>

## Ranked recommendations

1. <fix — evidence (numbers above) — expected impact>
2. ...

## Next measurement

<how you will verify the top recommendation: harness variant to bench,
capture or powermetrics run if escalation is needed>
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
