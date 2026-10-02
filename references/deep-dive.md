# Tier 3: OS counters — powermetrics, xctrace, and neighbors

Escalate here when the question is about the SoC, not the code: thermal
throttling during long runs, GPU frequency behavior, or hardware counters
that captures don't expose. All of these are sudo- or Xcode-bound: ask the
user to run them (or confirm sudo) rather than assuming.

## powermetrics (GPU frequency / power / residency)

```bash
# run alongside the bench loop for ~15s
sudo powermetrics --samplers gpu_power -i 500 -n 30
```

Parse the text output; the interesting lines are `GPU HW active residency`
(GPU busy %) and `GPU Frequency`. Interpretation:

- Residency near 100% with slow wall-clock → compute- or bandwidth-bound;
  the GPU is fed, the kernel is the problem (go back to Tier 2 capture).
- Residency low → gaps between dispatches: CPU-bound encoding, sync stalls,
  or data movement. Look at the capture's empty-encoder gaps.
- Frequency sagging over a long run → thermal throttling; compare first vs
  last third of the samples.

Add `--samplers cpu_power` when the workload mixes Metal and Python-heavy
phases. Output is text on stdout; the agent can parse it if the user pipes
it to a file: `sudo powermetrics ... | tee /tmp/gpu.txt`.

## xctrace / Instruments (hardware counters)

```bash
xcrun xctrace record --template 'Game Performance' --attach <pid> \
    --time-limit 10s --output /tmp/perf.trace
```

This is the truest ncu analog (per-kernel counters via the Game Performance
template) but GUI-first: `xctrace export --input /tmp/perf.trace --toc` lists
extractable tables and `--xpath` dumps them as XML — clunky but parseable
when a specific counter table is needed. Requires full Xcode, not just CLT.

## MLX-side knobs that change measurements

- `mx.metal.set_wired_limit(bytes)` — allow the GPU to wire more memory for
  large models; failures to allocate show up as slowdowns before errors.
- `mx.metal.set_cache_limit(bytes)` / `mx.metal.clear_cache()` — cap or drop
  the buffer cache when measuring true memory footprint.
- `MLX_DISABLE_COMPILE=1` — isolate kernel-fusion effects when A/B-testing
  transformations.

## When stock tooling isn't enough

- [OpenSourceWTF/mlx-profiler](https://github.com/OpenSourceWTF/mlx-profiler) —
  community MLX fork with command-buffer-level instrumentation (encoding vs
  GPU execution overlap). Diagnostic builds, not for production numbers.
- `MLX_METAL_DEBUG` CMake option — better kernel names in captures (custom
  builds of mlx itself).
