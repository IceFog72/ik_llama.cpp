# Plan 003: Reduce exposed CPU/GPU handoff time

Date: 2026-10-15. Starting source: 0e4e40b0 plus the separate, uncommitted n-gram repair. This plan records the full optimization sequence. The first implementation is stages 1–2 below; later stages have explicit gates rather than being silently bundled into a concurrency patch.

## Target and evidence

Improve correct warmed emitted throughput with a configured 65,536-token context. A reported 32 t/s corresponds to 31.25 ms/token; 40 t/s requires 25 ms/token. Saving 6.25 ms/token averages about 156 microseconds over 40 routed layers. This is a target, not a prediction.

The matching recent launcher capture is `/mnt/LLM/Text/llama-launcher/logs/llama-server-3172495.log`: new policy, cap 2048 MiB, ubatch 512, threads 8/8, CUDA offload threshold 16. Its final response measured 28.32 t/s near 21.8k context. Cumulative shutdown counters: router wait 334.7 us/layer, CPU turnaround 331.5 us/layer, deferred join 323.4 us/deferred-layer, GPU enqueue 15.7 us/layer, mapping wait 2.2 us/layer. These averages are not an additive device timeline. GPU expert-reference share 52.3% is not GPU utilization.

The old-policy CUPTI capture in Plan 001 does not measure the new executor. Eight working logical threads on a 16-thread CPU can produce approximately 50% CPU utilization; utilization alone does not identify a bottleneck.

## Contract and scope

- Apply execution changes only to automatic residency with `--moe-resident-profiler new`.
- Preserve actual router IDs, expert weights, quantization formats, reduction order, KV/recurrent state, Hadamard settings and workspace lifecycle.
- Preserve old-policy and residency-disabled execution.
- Keep all graph/cache/mapping descriptors alive until their consumers finish. Never remove a barrier without replacing its ownership guarantee.
- Do not increase ubatch on follow-ups or change the 65k context to obtain a speed result.
- Preserve unrelated n-gram changes. No server rebuild by the parent agent; the user rebuilds llama-server.

## Stage 1: GPU hits before the host routing wait

Status: source implemented; focused CUDA numerical checks passed; wider baseline failures remain unresolved.

Previous partial-layer order: copy IDs/activation, wait, classify on CPU, submit CPU, upload mapped IDs, enqueue GPU experts.

New order:

1. Queue original IDs and activations to pinned host buffers.
2. Record the input-ready event immediately after these copies.
3. If the layer has cached experts, map the actual selected IDs using the existing device cache lookup and enqueue the masked GPU FFN.
4. Only then wait on the input-ready event. This event does not include the expert kernels queued after it.
5. Partition the original IDs for CPU misses, submit the CPU job, and retain original IDs separately from its mutable masked input.
6. Join at the existing proven consumer and merge CPU results through the existing scatter/full-result path.

This removes the redundant host-to-device mapped-ID upload on partial cached layers. Layers with no resident experts enqueue no GPU FFN. Partially cached layers whose actual selection is all-miss can enqueue a masked FFN; CPU output overwrites all its rows in stream order. Count these cases explicitly to measure wasted work. Do not claim this cost is free.

Safety: no promotion/publication changes mappings between device mapping and the existing final result fence; the background uploader writes only its reserved unavailable slot. An error after early enqueue synchronizes the GPU before returning. Host readback scratch cannot be overwritten until input-ready. CPU masked IDs and expert-list storage remain immutable until join.

Counters: `early_gpu layers=... empty_routes=...`. Existing `router_wait` now starts after enqueue, so a reduction can merely reflect the moved host work; use total useful throughput and a device timeline to establish improvement.

## Stage 2: Cover CPU work with independent work

Status: source implemented for policy bookkeeping; existing shared-branch placement retained.

Move token-route learning and popularity updates after CPU submission. These touch policy state, not the CPU graph's IDs, active-expert list or arithmetic buffers. Retain the actual original route in task-owned reusable storage so CPU masking cannot pollute prediction learning.

The graph builder already expands shared expert work between routed DOWN and reduction. The scheduler already validates intervening splits and joins before the first marked dependent consumer, another expert invocation, or a CPU split. Do not add a second scheduling mechanism. Existing numerical tests cover an independent branch, an intervening CPU branch, and a failing independent branch followed by retry.

If join latency stays high, obtain split durations and attribution before moving further work: no cross-layer precomputation may consume a residual whose routed contribution is unfinished.

## Stage 3: Selected-expert all-hit execution

Status: planned, not implemented by stages 1–2.

Whole-layer residency currently bypasses host routing, but selected-expert all-hit on a partially cached layer still reads back inputs. Stage 1 already enqueues its GPU work before that readback wait and skips the CPU graph when CPU misses are zero.

Next design: device route classification produces a miss flag/count and compact miss rows. Let all-hit GPU work continue without host participation on its device critical path. Queue or condition CPU-input transfers without a new mandatory host round trip. Preserve actual IDs and make all-hit, mixed and all-miss cases explicit. A flag copied to the host followed by a wait is not a host-free fast path.

Gate: demonstrate reduced all-hit critical-path cost without penalizing common mixed/all-miss routes. Check repeated IDs, invalid/masked IDs and widths 1–4.

## Stage 4: Reduce CPU result transfer and graph overhead

Status: planned, gated on stages 1–2 measurements.

CPU DOWN already skips quantizing GPU-owned rows. `cpu_masked_quant_rows` counts masked rows, not proof of redundant conversion. Persistent CPU workers and unique miss lists also already exist.

Investigate weighted CPU reduction to one hidden-size vector per token instead of packing every missing expert output. This requires passing the correct current-layer routing weights, preserving optional expert scales/biases and joining at the correct graph seam. Partial CPU reduction changes floating-point summation order; prove tolerances and generation quality rather than promising bit identity.

Compare existing generic graph execution against a narrow fused active-expert CPU path only if measured wake/barrier/graph overhead remains exposed. Reuse existing vector-dot kernels and bounded worker ownership. Strata's 64-row task layout and worker-owned GPU submission are reference designs, not automatic replacements.

## Stage 5: Batched cache maintenance

Status: planned, gated on admission/upload costs and hit rate.

The current uploader supports one pending expert and rebuilds/uploads a full mapping on reservation and publication. Replace this only after measuring its exposed cost. Proposed bounded batch: own immutable source/destination descriptors and pinned staging until one completion event; reserve all victims before dispatch; update changed mapping ranges; publish only completed entries; protect active and predicted useful slots.

Do not overwrite a live slot merely because its host mapping was removed: fence its last GPU reader. Retain safe shrink, graph reset, allocation-failure fallback and teardown. Track bytes uploaded, admitted/rejected experts, hit rate, useful forecast use and exposed waits. No unbounded upload queue or per-layer cache rebuild.

## Stage 6: Existing lookup and MTP speculation

Status: existing mechanisms available; separate n-gram repair remains uncommitted. Live comparison pending.

Compare none, lookup, one-draft MTP, and lookup-to-MTP fallback independently, using Plan 001's explicit stage flags. Include draft runtime, verification, rollback, recurrent checkpoints and expert-cache displacement in useful emitted tokens/s. Do not mix speculative emitted throughput with single-token decode speed.

Strata's engine-poly defaults enable lookup/model drafting. Historical launches used tail-skip 6, but the latest recorded launch omitted it. Check the exact comparison run. Expert omission changes arithmetic and must not be silently imported as a faithful optimization.

## Verification and acceptance

1. Inspect current process CPU deltas, GPU compute/graphics activity, RAM, paging, VRAM and GPU clocks/temperature before every test. Luna performs builds/tests; notify the user before running them. Do not terminate competing applications.
2. Build only the focused `test-moe-resident` target, not llama-server. Run default CPU checks plus existing CUDA resident batch/prediction checks. Add assertions that early device dispatch and partially resident all-miss routes are exercised; retain numerical comparison against native CPU/GPU references.
3. Exercise widths 1–4, combined/separate UP/GATE, mixed quantization, duplicate/invalid IDs, all-hit/mixed/all-miss routes, cache failure, stale bank identity, growth/release, independent-branch failure/retry and CPU-branch defer rejection where supported by existing fixtures.
4. User rebuilds the server. Record binary/library provenance. With original flags and no profiler preload, run a neutral approximately 16k prompt, two real follow-ups and regeneration of the third request. Require useful two-paragraph output, no corruption/empty response and correct history reuse.
5. Compare paired warmed rounds against the previous binary with identical flags, sampling and resource conditions. Report emitted throughput, request wall time, cache allocation, CPU join, early empty-route fraction and contending work. Do not count changed counter timing as a speedup.
6. Only progress into later architectural stages using measured evidence. Stages 3–6 are explicitly not claimed complete by the first dispatch patch.

## Validation record

Luna's focused build, default CPU checks, CUDA resident batches (widths 2–4) and prediction checks passed. The broader CUDA suite exposed a stale white-box fixture: manually injected new-policy width-1 caches omitted the device lookup table. The backtrace reached a null lookup-buffer interface before early dispatch. The fixture now initializes lookup state for all new-policy injected caches.

After the fixture repair the broader `--cuda` mode completes, but reports failures in combined-bank growth numerical checks, failed-allocation fallback numerical checks, eviction expectations and legacy timing-driven zero-fill assertions. Luna built a separate HEAD executor/test overlay with only the same fixture repair: it reproduces all those failure categories, including the identical failed-allocation row mismatch counts. Failure counts in growth/eviction cases differ between executions, so this is category-level baseline attribution, not a deterministic proof that every failure is identical. These wider correctness issues remain unresolved and must not be described as passing or caused by external GPU contention.

Artifacts and exact commands: `.skills/benchmarks/2026-10-15-resident-early-dispatch/`; separate pre-change comparison: `baseline/`. No tracked production files were rolled back for the comparison.

An external graphics application occupied approximately 3.2 GiB VRAM and used the GPU during those synthetic checks. Their timing readings are not throughput evidence.

After the user requested direct testing without Luna, the parent completed the server build and a real approximately 16k chat chain using the posted fixed-ubatch/capped new-policy flags and no speculation. Decode rates were 31.34, 32.20, 33.67 and 33.68 t/s for the initial prompt, two follow-ups and third-turn regeneration. Weighted warm decode was 33.18 t/s. All returned two readable paragraphs and stopped normally. Some answer overinterpretation was observed; no empty output or multilingual corruption appeared. The configured context remained 65,536. The supervised server exited 0.

The live run had no other GPU compute process, but desktop activity and some paging remained. Exposed cumulative counters remained router wait 286.1 us/layer and deferred CPU join 339.8 us/deferred-layer. A paired pre-change live comparison was not performed, so improvement attributable to this patch is not established. Neither a 40 t/s result nor resolution of the broader baseline CUDA failures is claimed. Full live evidence: `.skills/benchmarks/2026-10-15-direct-resident-live/RESULTS.md`.
