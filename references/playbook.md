# Diagnosis playbook — signal → likely cause → fix

Match measured signals to a cause before proposing anything. Usually 1–2
causes dominate; name them with the numbers that identified them.

## Timing signals (tier 1)

| Signal | Likely cause | Fix direction |
| --- | --- | --- |
| `clock_ramp_suspected`, or per-iter times trend downward | GPU clocks still ramping (DVFS); prewarm too short | raise `prewarm_s`, re-run; don't diagnose anything else from this run |
| p50 fine, p90 ≫ p50 (tail spikes) | other GPU clients (windowserver), or thermal | report p90 honestly; verify with powermetrics frequency trend; for A/B use medians |
| Two variants differ only on cold runs | machine state, not your change | pre-warm identically or interleave variants in one process |
| TTFT ≫ n·ITL | prefill-bound, not decode-bound | attack prompt processing; prompt cache (skip re-prefill), not decode tricks |
| ITL grows with context length | attention/KV cost scaling | capture (tier 2) to confirm attention kernels dominate; KV quant, sliding window, cache |
| decode step slow, stable, memory flat | kernel-bound or bandwidth-bound | tier 2 capture: which kernel, is it compute- or copy-bound |

## Memory signals (tier 1)

| Signal | Likely cause | Fix direction |
| --- | --- | --- |
| `cache_mb` grows across runs/variants | normal MLX buffer cache, NOT a leak | only a problem if footprint matters: `mx.metal.set_cache_limit` |
| `active_mb` never returns down after a phase | retained arrays (e.g. logits, generated tokens held per-step) | free per-step products; only keep the running generation state |
| `peak_mb` ≫ `active_mb` end-state | transient spike (prefill, big temp) | bracket with `reset_peak_memory` to locate; chunk the spike source |
| allocation failures / wired-limit errors near big models | unified-memory wired cap | `mx.metal.set_wired_limit`, or smaller quant |

## Capture signals (tier 2 — read references/capture.md first)

| Signal | Likely cause | Fix direction |
| --- | --- | --- |
| gaps between kernels, empty encoder | CPU-bound: Python/MLX encoding can't feed the GPU | fewer, larger ops; `mx.compile` the step; avoid per-token Python work |
| one kernel dominates (usually gemm) | compute-bound | quantization, shapes/batching of the gemm, smaller dtype |
| long copy/memcpy kernels around compute | data movement: dtype casts, reshapes, host transfers | restructure to keep data resident; cast once, not per-step |
| attention kernels grow with context | KV/attention scaling | KV quant, cache reuse |

## SoC signals (tier 3 — references/deep-dive.md)

| Signal | Likely cause | Fix direction |
| --- | --- | --- |
| GPU HW residency ~100%, wall-clock slow | GPU-bound — kernel or bandwidth | tier 2 to see which kernel |
| residency low | gaps — CPU-bound or sync stalls | tier 2 empty-encoder check |
| GPU frequency sags across a long run | thermal throttling | cool-down between runs; report both hot and cold numbers |
