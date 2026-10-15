# Plan 004: Reduce exposed CPU handoff latency without predictor VRAM

## Decision

The direct new-policy run reached 33.67–33.68 t/s after warming. Its cumulative deferred join was about 340 us/layer; CPU job turnaround about 321 us/layer. These counters do not distinguish arithmetic from scheduling. Source shows unconditional condition-variable parking of the CPU coordinator and persistent team after each short graph.

First candidate: bounded hot waiting before the existing predicate/mutex waits, confined to the new resident executor. This adds no predictor, model arithmetic, or GPU allocation. Preserve the arithmetic and existing lifecycle while testing whether repeated OS wake-ups expose measurable latency. This is a hypothesis to benchmark, not a proven bottleneck or promised speedup.

CPU-only MTP remains a later option: it consumes additional CPU/RAM bandwidth, needs its own placement/state path, and must be measured against this already exposed CPU work. Do not allocate an MTP predictor on GPU or silently enable drafting.

## Implementation

1. Commit the existing n-gram and early-GPU work separately before starting this candidate.
2. Publish an advisory atomic generation for the CPU team; use a bounded 500 us pause loop before parking on its existing condition variable.
3. For the new-policy CPU coordinator, similarly poll its atomic busy indication for at most 500 us before taking the mutex and using the original predicate wait.
4. Keep all descriptor reads, completion/status publication, inactive-worker checks and teardown under their existing mutex ownership. Polling is advisory; it does not grant permission to read a job or reuse storage.
5. Add a stress test alternating 1/8/2/16/1/4 active threads, immediate descriptor reuse, occasional idle gaps beyond the spin deadline, and destruction after a hot completion.

The intended benefit is avoiding kernel wake-up latency across short adjacent layer gaps. The tradeoff is extra CPU time/power spent polling; higher CPU utilization is not proof of improvement. Threads park after timeout, so idle server use must settle. Existing active worker count remains controlled by -t.

## Verification

- Direct execution, no Luna, as requested by the user.
- Resource preflight before build/tests; never terminate another application to obtain a result.
- Preserve the pre-change executable and shared libraries before building for an actual before/after comparison.
- Run existing CPU masked-quantized arithmetic checks, changing-team stress, CUDA batches and prediction fixtures. Retain known broader baseline failures separately; do not claim they are fixed.
- Use the same saved approximately 16k conversation with two prior exchanges for both executable/library sets. Warm once, then repeat three cached two-paragraph requests with the same seeds. Confirm the loaded library paths and actual executable hashes.
- Compare decode throughput and CPU join/turnaround counters while recording competing CPU/GPU activity, paging, clocks and memory. No speculative decoding and no extra GPU model.
- Reject or revise the candidate if it regresses throughput or fails ownership/numerical checks. Do not promote CPU utilization itself as success. A single pair supports only a preliminary result.

## Status

Candidate implemented. Final focused build, changing-team CPU stress/default arithmetic, CUDA resident batches and CUDA prediction checks passed. The coordinator captures the hot-wait policy at thread creation; idle polling does not race later graph-scan writes to the policy field.

Baseline binaries are preserved under `.skills/benchmarks/2026-10-15-cpu-handoff/before/`. The first baseline run is explicitly excluded: the user confirmed Firefox activity. The rerun samples other processes throughout inference and holds requests during heavy competing CPU activity.

The direct before/after comparison completed. Three cached responses measured 33.94/33.21/32.56 t/s before and 37.59/37.30/37.43 t/s after. Decode-time-weighted rates were 33.21 and 37.44 t/s: an observed 12.8% increase. Cumulative CPU turnaround fell 305.8 -> 234.1 us/layer, deferred join 329.4 -> 249.6 us/deferred-layer, while router wait stayed approximately 289 -> 288 us/layer. Mean cached server CPU rose 349% -> 687% (one logical core = 100%); the polling cost is real. No predictor or additional model GPU allocation was added. Cache allocations were 2018/2024 MiB within the same cap.

All candidate responses stopped normally with two readable paragraphs. Inputs and seeds matched, but generated sequences/lengths differed. This is a fixed 16,910-token conversation replay with two earlier exchanges, not a fresh sequential chat chain. Background CPU activity was lower after (sampled top-eight mean 54% versus 79%), and some paging remained, so the single sequential pair is preliminary evidence, not a guaranteed speedup. No 40 t/s result is claimed. Both servers exited 0.

Decision: retain the bounded-handoff candidate under the existing new-policy opt-in. Full commands, actual library paths, source/binary provenance, request/response files, competing-process samples and caveats: `.skills/benchmarks/2026-10-15-cpu-handoff/RESULTS.md`.
