# mlx-perf

An agent skill for profiling MLX workloads on Apple Silicon — the macOS
analog of `ncu`-style profile → diagnose → report workflows (in the spirit
of [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill), but
for Metal, where no Nsight Compute exists).

**Measure → Diagnose → Plan, in that order. Never guess.** The skill gives
an agent (ZCode, Claude Code, Codex, …) three measurement tiers, a
signal→cause→fix playbook, and an evidence-cited `REPORT.md` output:

1. **Timing + memory harness** (fully autonomous) — bracket the `mx.eval`,
   p50/p90 stats, `mx.get_peak_memory` accounting. Handles the GPU
   clock-ramp trap: on a cold machine the same kernel reads ~4x slow until
   clocks spin up, so the harness pre-warms ~1s of sustained load and flags
   `clock_ramp_suspected`.
2. **Metal GPU capture** — `mx.metal.start_capture` recorded headless
   (requires launching with `METAL_CAPTURE_ENABLED=1`), per-kernel truth
   read in Xcode.
3. **SoC counters** — `powermetrics` / `xctrace` for throttling, residency,
   and frequency questions.

## Install

```sh
git clone https://github.com/ThinkFlowLab/mlx-perf.git ~/.zcode/skills/mlx-perf
# or ~/.claude/skills/mlx-perf, or your agent's skills directory
```

## Use

Ask your agent (or invoke the skill directly):

> profile my decode step at 4k context with mlx-perf — why is it 30 ms/token?

The harness is also usable standalone:

```python
import sys; sys.path.insert(0, "mlx-perf/scripts")
from mlx_perf import bench

stats = bench(step, n=30, label="decode_4k")   # p50/p90/mean + memory dict
```

`python scripts/mlx_perf.py --selftest` verifies the harness on your machine.

## Structure

| Path | What |
| --- | --- |
| `SKILL.md` | workflow: run-dir discipline, tier routing, report shape |
| `scripts/mlx_perf.py` | `bench()` / `memory_snapshot()` / `--selftest` |
| `scripts/make_report.py` | bench JSON stats → `REPORT.md` with baseline deltas |
| `references/playbook.md` | signal → likely cause → fix, four signal families |
| `references/capture.md` | tier 2: GPU capture recipe + Xcode reading guide |
| `references/deep-dive.md` | tier 3: powermetrics / xctrace / MLX memory knobs |

## Verified against

mlx 0.32.x (arm64), Apple Silicon (`applegpu_g16g`), Python 3.13. Facts
baked into the skill that are easy to get wrong elsewhere: MLX has no
`mx.profiler`/`mx.metrics` (that's JAX); `mx.metal.get_*_memory` is
deprecated in favor of `mx.get_*_memory`; time the eval bracket, not graph
construction.

## Related

- [ThinkFlowLab/vllm-omni-mlx](https://github.com/ThinkFlowLab/vllm-omni-mlx) —
  the project this was built for; human-readable methodology in its
  `docs/profiling.md`.
