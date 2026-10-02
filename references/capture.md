# Tier 2: GPU frame capture (.gputrace)

Use when Tier 1 timing shows where the time goes but not why — e.g. a step
is slow, memory bandwidth looks saturated, or you need per-kernel times.
The agent records the capture headless; a human opens it in Xcode (nothing
on macOS parses .gputrace programmatically).

## The one trap

`mx.metal.start_capture(path)` fails with:

```
RuntimeError: [metal::start_capture] Failed to start: Capture layer is not inserted.
```

unless the **process is launched with `METAL_CAPTURE_ENABLED=1`**. You cannot
set it after start. Relaunch the harness with the env var:

```bash
METAL_CAPTURE_ENABLED=1 .venv/bin/python harness.py
```

## Recipe

1. Reuse the Tier 1 harness but shrink it: **one** iteration of the hot
   region inside the capture bracket (traces grow ~MBs per ms of captured
   GPU work; a full generation loop produces an unopenable trace).
2. Bracket exactly the interesting region:

   ```python
   mx.metal.start_capture("/tmp/step.gputrace")
   out = step()          # one decode step / forward pass
   mx.eval(out)
   mx.metal.stop_capture()
   ```

3. Verify the bundle exists: `ls -la /tmp/step.gputrace` (a directory with
   `capture`, `index`, `delta-*` files).
4. Hand off: `open /tmp/step.gputrace` opens Xcode's capture view. Tell the
   user what to look for (below). Do not bury them in a cold directory of
   captured files — one named trace, one sentence of guidance.

## What the human should look for in Xcode

- The kernel list shows per-kernel GPU durations. MLX names its kernels:
  `gemm`/matmul variants, `rms_norm`, `rope`, quantized gemm (`quantized_*`),
  attention, and memcpy/copy kernels. Sort by duration.
- Big gaps between kernels with an empty encoder = CPU-bound: the GPU is
  waiting for Python/MLX to encode work. That's a "reduce Python overhead /
  fuse ops" answer, not a "make the kernel faster" answer.
- Long copy kernels bracketing compute = data is being moved (reshapes,
  dtype casts, host transfers) — restructuring beats kernel tuning.
- Per-kernel memory throughput in the capture is the closest thing to ncu's
  bandwidth counters on macOS.

## Caveats

- Captured runs are for attribution only, never for timing numbers — the
  capture layer adds overhead.
- Fixed inputs (seeded) keep the captured kernels representative.
- If kernel names are opaque in the trace, MLX built with `MLX_METAL_DEBUG`
  improves labels (relevant only for custom builds, see references/deep-dive.md).
