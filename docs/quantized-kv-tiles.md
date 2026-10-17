# Experimental quantized KV tile loading

Implemented for an isolated comparison against the existing MMA attention path.
Not built or tested by the implementation agent. Performance and numerical
correctness remain to be validated on hardware.

## Enable and disable

Keep your existing server arguments and combine the option with the existing
CUDA argument. Repeating `-cuda` replaces its earlier value.

```sh
# Candidate
-cuda offload-batch-size=16,fa-quant-tile=1

# Control (also the default when fa-quant-tile is omitted)
-cuda offload-batch-size=16,fa-quant-tile=0
```

The option is set when the CUDA backend is created. It does not change CPU
expert workers, resident cache policy, KV quantization, or microbatch sizing.
It can be combined with the existing `--moe-resident` options.

Look for actual dispatch messages, not only the startup option acknowledgement:

```text
fa-quant-tile active K=q4_0 V=q4_0 head=256 query=1 gqa=8 (MMA, no full KV expansion)
fa-quant-tile active K=q4_0 V=q8_0 head=256 query=1 gqa=8 (MMA, no full KV expansion)
```

Each type combination logs once per backend. A startup message alone does not
prove that the request's shape selected the optimized path.

## Scope and ownership

The first specialization supports the measured Qwen path: one query token,
256-wide heads, Q4_0 K, Q4_0/Q8_0 V, and grouped query heads in multiples of
eight. It requires the existing MMA-capable NVIDIA path, a mask, default
precision, zero ALiBi/softcap, and no sliding-window hint. Other cases retain
the existing implementation, including multi-token prompt processing.

K/V remain quantized in their existing device buffers. The loader expands a
32-row tile into the same padded FP16 shared-memory tile the existing MMA
kernel consumes. Each lane owns one adjacent pair per loop iteration. K and V
retain independent row/head strides, including the model's interleaved-head
layout and the mixed Q4/Q8 case. Existing synchronization protects tile reuse.
The legacy F16 specializations retain their previous layout assumptions.

The grouped-head MMA arithmetic, softmax, attention sinks, stream-K partitioning
and reduction are reused. Hadamard transformations remain outside this kernel.
No persistent expanded cache, new worker, or background allocation is added.

This avoids full-view K/V conversion during eligible decode calls. Prefill
still uses the old conversion path, and its idle buffers may remain in the
CUDA memory pool. Lower persistent VRAM use or more resident experts is not
guaranteed by this change alone.

## Build and correctness gate (user-run)

From the repository root, using the existing configured CUDA build:

```sh
cmake --build build --target llama-server -j14
cmake --build build --target test-cuda-fa-quant-tile -j14
./build/bin/test-cuda-fa-quant-tile
```

The manual test target is available when `GGML_CUDA` and `LLAMA_BUILD_TESTS` are
enabled. It compares identical quantized inputs through separate off/on CUDA
backends and fails if the tiled path was expected but never selected. It covers
Q4/Q4 and Q4/Q8 at 256, 257, 2048, 20000, 50000 and 65536 actual tokens, padded
and sparse masks, both KV layouts, sinks, repeated graphs with changed inputs,
and fallback shapes/types. It rejects non-finite output and checks each value
with `abs_error <= 1e-5 + 1e-4*abs(reference)`.

The test is a numerical gate, not a speed benchmark. It reuses the old
conversion-plus-MMA implementation as reference. It does not establish
whole-model generation quality by itself.

## Performance acceptance

Compare warmed 20k and 50k prompts with the same grammar, request, sampling,
thread settings and fixed expert budget. Use an alternating off/on order and
multiple repetitions. Check other CPU/GPU use and close Firefox before timing.
Record filled context, prompt and decode timings, resident GPU share, and the
actual dispatch messages. Compare the complete conversion-plus-attention cost
and end-to-end decode speed. Removing conversion kernels alone is not a win
if the attention kernel becomes slower.

Keep the change only if numerical checks pass and the repeated speed difference
exceeds run-to-run variation. Extra unpacking instructions can outweigh the
saved VRAM traffic. No throughput improvement is claimed before that comparison.
