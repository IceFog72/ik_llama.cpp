# Quantized KV tile loading for decode attention

`-cuda fa-quant-tile=1` makes decode attention read the quantized KV cache
directly. The old path converts the whole K and V views to f16 on every
attention call. Opt-in, default off. Built and measured on an RTX 2060
(Turing) with Qwen3.6-35B-A3B; see Results.

## Enable and disable

Add the option to the existing CUDA argument. Repeating `-cuda` replaces its
earlier value, so combine options with commas.

```sh
-cuda fa-quant-tile=1    # tiled path
-cuda fa-quant-tile=0    # old path (also the default)
```

The option is set when the CUDA backend is created. It does not change CPU
expert workers, the resident cache policy, KV quantization, or microbatch
sizing, and it combines with the `--moe-resident` options.

Look for the dispatch messages, not only the startup acknowledgement:

```text
fa-quant-tile active K=q4_0 V=q4_0 head=256 query<=2 gqa=8 (MMA, no full KV expansion)
fa-quant-tile active K=q4_0 V=q8_0 head=256 query<=2 gqa=8 (MMA, no full KV expansion)
```

Each type combination logs once per backend. A startup message alone does not
prove that the request's shape selected the tiled path.

## Scope

The specialization covers the measured Qwen path: one query token, or two for
the MTP verification batch (`--spec-type mtp:n_max=1`), 256-wide heads, Q4_0 K,
Q4_0/Q8_0 V, and grouped query heads in multiples of eight. It requires the
MMA-capable NVIDIA path, a mask, default precision, zero ALiBi/softcap, and no
sliding-window hint. Other cases keep the existing implementation, including
prompt processing and draft batches longer than two tokens.

## How it works

K and V stay quantized in their device buffers. The loader expands a 32-row
tile into the same padded f16 shared-memory tile the existing MMA kernel
consumes. Each lane owns one adjacent pair per loop iteration. K and V keep
independent row/head strides, including the model's interleaved-head layout
and the mixed Q4/Q8 case. Existing synchronization protects tile reuse. The
f16 specializations keep their previous layout assumptions.

The grouped-head MMA arithmetic, softmax, attention sinks, stream-K
partitioning and reduction are reused. No persistent expanded cache, new
worker, or background allocation is added.

The tiled kernel runs 2-warp blocks, and Turing has no asynchronous copies, so
one stream-K block per SM leaves it latency-bound. The tiled path launches two
blocks per SM. Four or eight did not help further on the RTX 2060.

Hadamard rotation (`--k-cache-hadamard`, `--v-cache-hadamard`) happens in the
graph outside this kernel. Q and K are rotated before attention, V before it is
stored, and the output is rotated back. The kernel reads the rotated Q4_0 data,
so the option works unchanged with rotation on.

Prefill still uses the conversion path, and its idle buffers may remain in the
CUDA memory pool. The option does not by itself free VRAM for experts.

## Correctness test

From the repository root, with a configured CUDA build:

```sh
cmake --build build --target test-cuda-fa-quant-tile -j14
./build/bin/test-cuda-fa-quant-tile
```

The test target is available when `GGML_CUDA` and `LLAMA_BUILD_TESTS` are
enabled. It runs identical quantized inputs through separate off/on CUDA
backends and fails if the tiled path was expected but never selected.

The tiled path splits the KV range across more blocks than the old path, so its
f16 partial sums combine in a different order. Where it is selected, both
outputs are measured against a double-precision reference, and the tiled
maximum error must not exceed 1.25 times the old path's. Fallback shapes must
still match the old path to `1e-5 + 1e-4*abs(reference)`.

It covers Q4/Q4 and Q4/Q8 at 256, 257, 2048, 20000, 50000 and 65536 actual
tokens, two queries at 257, 2048, 28000 and 65536 tokens, padded and sparse
masks, both KV layouts, sinks, repeated graphs with changed inputs, and
fallback shapes/types. It rejects non-finite output. It is a numerical gate,
not a speed benchmark, and does not by itself establish whole-model quality.

## Results

RTX 2060, Qwen3.6-35B-A3B attention shape (head 256, 16 query heads, 2 KV
heads), K Q4_0. Time for one attention call:

| Context | Query tokens | Old path | Tiled |
|---|---|---|---|
| 1,024 | 1 | 64 us | 74 us |
| 1,024 | 2 | 71 us | 80 us |
| 8,192 | 2 | 271 us | 168 us |
| 28,160 | 1 | 759 us | 290 us |
| 28,160 | 2 | 649 us | 301 us |
| 28,160 | 2 (V Q8_0) | 675 us | 283 us |
| 57,344 | 2 | 1,305 us | 561 us |

At about 1K context the tiled path costs about 10 us per call, roughly 0.1 ms
per decode step over Qwen's 10 attention layers.

End to end, a 28K-token request on the same build with one-draft MTP decoded at
36.4–38.5 t/s tiled against 32.5–33.3 t/s on the old path, with draft acceptance
unchanged. Waiting for attention fell from 372 to 269 us per layer, the same as
at short context.

To compare on another machine, use warmed prompts of 20K tokens or more, the
same request and sampling, and alternate off/on runs. Close other GPU users,
such as a browser, and record free VRAM, because the expert cache sizes itself
from it.
