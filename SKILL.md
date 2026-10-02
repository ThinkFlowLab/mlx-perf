---
name: mlx-perf
description: |
  Profile, diagnose, and report on MLX workloads on Apple Silicon, in the
  spirit of ncu-report-skill but for Metal: run-directory discipline, tiered
  measurement (timing harness, GPU capture, SoC counters), a signal→cause→fix
  playbook, and an evidence-cited REPORT.md. Use when profiling MLX kernels,
  diagnosing why a step/generation is slow, measuring token latency or memory
  footprint, or A/B-comparing a change. Not for web performance or CUDA.
---

# mlx-perf

**Measure → Diagnose → Plan, in that order. Never guess.** A slow step has
one identifiable reason on Apple Silicon too; don't hypothesize before
numbers exist, and don't recommend a fix that doesn't match a diagnosis.
Every conclusion cites a measured value.

There is no ncu on macOS; per-kernel hardware counters effectively don't
exist outside Xcode. The three tiers below are the whole toolbox, and tier
1 answers more questions than people expect — escalate only when the
playbook says the needed signal lives in a deeper tier.

## Workflow

1. **Fresh run directory, never reused:**
   `profile/<run_name>/` (in the repo being profiled) containing
   `harness.py`, `stats/` (bench JSON), and the final `REPORT.md`.
   Suggest gitignoring `profile/`.
2. **State the question first.** "Why is decode 30 ms/token at 4k context?"
   not "make it fast". The question picks shapes, context lengths, and
   which tier can answer it.
3. **Build a standalone harness** importing this skill's
   `scripts/mlx_perf.py`; run it via `bench()` and dump JSON per variant
   (`stats/<label>.json`). Tier-1 rules below.
4. **Diagnose with the playbook** — read
   [references/playbook.md](references/playbook.md) and match signals to a
   cause. Usually 1–2 causes dominate.
5. **Escalate tier only if the playbook demands it:** per-kernel truth →
   [references/capture.md](references/capture.md); throttling/bandwidth →
   [references/deep-dive.md](references/deep-dive.md).
6. **Write REPORT.md** via `scripts/make_report.py` (fills stats tables,
   flags, baseline deltas) then complete the Diagnosis / Ranked
   recommendations / Next-measurement sections by hand, citing numbers.

## Tier 1: timing harness (always start here)

```python
import sys; sys.path.insert(0, "<this-skill>/scripts")
from mlx_perf import bench

def step():                 # the code under test; eval its outputs
    out = model(x)
    mx.eval(out)

stats = bench(step, n=30, label="decode_4k")   # dict → stats/<label>.json
```

Rules that make numbers trustworthy:

- **MLX is lazy.** Graph-building calls queue work; only `mx.eval(arrays)`
  or `mx.synchronize()` execute it. Time the eval bracket. `bench()`
  synchronizes after each call.
- **Clock ramp (DVFS).** Apple GPUs raise clocks only under sustained load:
  the same matmul read 4.2 ms cold vs 1.1 ms warm on this very machine.
  `bench()` pre-warms `prewarm_s=1.0` (raise for short kernels) and flags
  `clock_ramp_suspected`. A/B comparisons pre-warm identically or interleave.
- **p50 and p90, never the mean alone.** windowserver et al. spike the tail;
  the median is the workload, the p90 is the user experience.
- **Memory**: `mx.reset_peak_memory()` brackets the region;
  `mx.get_active_memory()` (live arrays) vs `mx.get_cache_memory()`
  (reusable buffers — growth is not a leak) vs peak. `mx.metal.get_*`
  spellings are deprecated.
- **Generation loops**: bench one decode step at fixed context AND
  end-to-end generation separately; TTFT and inter-token latency have
  different bottlenecks.

## Verified API facts (mlx 0.32.x)

- **`mx.profiler` / `mx.metrics` do not exist in MLX** (that's JAX — don't
  reach for them). Per-kernel attribution is tier-2 capture only.
- `mx.metal.start_capture(path)` raises `Capture layer is not inserted`
  unless the process was launched with `METAL_CAPTURE_ENABLED=1`
  (references/capture.md).

## REPORT.md shape

Environment (device_info, mlx version, machine thermal state), the question,
stats table(s) with baseline deltas if any, flags, **Diagnosis** (playbook
match, cited numbers), **Ranked recommendations** (evidence + expected
impact), **Next measurement** (how the recommendation will be verified).
A report without numbers in its conclusions is not done.
