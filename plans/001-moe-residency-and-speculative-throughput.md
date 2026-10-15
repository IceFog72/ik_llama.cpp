# Plan 001: Improve warm throughput with existing speculation and a small lookup fix

Rewritten 2026-10-15 against committed baseline 0e4e40b0. Implementation branch: feat/ngram-context-fix-2026-10-15. This replaces the previous nine-stage architecture plan.

## Goal and evidence

Improve correct emitted throughput above the approximately 30 t/s warmed baseline, aiming for 40 t/s while keeping configured context 65,536, useful two-paragraph responses, follow-up continuity and regeneration. Preserve the working automatic initial-ubatch-to-512 handoff and existing KV/Hadamard/sampling settings.

The quieter cached control measured 28.88 t/s. Its profiled replay measured 25.99 t/s with approximately 10% profiling slowdown. Of the captured window, kernels occupied 42.4%, copies 28.9%, CPU expert graphs outside recorded GPU activity 4.4%, and the remainder 24.4%. CPU expert graphs overlapped GPU activity for 85.5% of their duration. Expert weights dominated H2D traffic, and recorded copies did not overlap kernels. The remainder is not proven to be entirely host launch overhead.

Evidence: .skills/benchmarks/2026-10-15-committed-adaptive-profile/RESULTS.md and quiet-run/. Sequential chat baseline: .skills/benchmarks/2026-10-15-committed-adaptive-rollback/RESULTS.md.

Earlier Strata measurements reached roughly 41 t/s with drafting versus 32 t/s without drafting on warmed cached prompts. Those are different engine runs, not a speed guarantee for this checkpoint. They justify testing existing speculation before designing another scheduler.

## Minimal scope

1. Measure the already implemented one-draft MTP path against no speculation.
2. Repair the existing RAM-only n-gram context cache; compare lookup alone and the existing lookup-to-MTP fallback.
3. Implement only the particular remaining bottleneck demonstrated by the winning configuration.

No new feature-selector framework, adaptive chain controller, persistent lookup database, external draft model, width-8 verifier, worker system, upload queue, lazy KV allocator, or graph-arena rewrite is part of the initial implementation.

## What is implemented

Source changes implemented; the focused CPU regression test passed:

- common/ngram-cache.cpp and .h: one context synchronization helper owns no new global state. It incrementally updates complete accepted history, preserving n-grams across append boundaries; edits, truncation and carry replacement rebuild counts.
- common/speculative.cpp: rebuild the context-only lookup cache once at request start; retain an owned history vector rather than allocating/copying the entire prompt for every lookup; use effective runtime n_max instead of the hardcoded eight-token draft limit.
- Proposals never enter the history vector or learned counts. The next draft call observes actual committed target tokens; accept(count) stays a no-op because a count cannot identify actual accepted/replacement tokens.
- tests/test-ngram-cache.cpp: regression cases compare incremental counts with full rebuilds, including repeated calls, multi-token append, same-length edits, regeneration, shifted history, replaced carry and rejected proposals.
- docs/speculative.md: document bounded lookup and the already supported MTP fallback chain.

The helper compares the supplied prefix for correctness. This remains a linear read, but eliminates the old per-token full-history allocation/copy and avoids a new fingerprint/epoch framework. Context memory is tied to supplied history. Existing optional loaded corpus/dynamic caches are retained; automatic persistence is not added.

No target weights, routing, CPU worker execution, CUDA code, residency policy, KV/recurrent checkpoints, sampler, grammar, server defaults or launcher presets are changed. The selected ngram-cache speculative stage is the opt-in. No speculation remains the default.

## Step 1 — Compare existing MTP before adding a batched residency engine

Use the same rebuilt server, model and request/sampling settings. Run no speculation and then one-token MTP independently:

~~~text
--spec-type none

--spec-type mtp:n_max=1,p_min=0.0
~~~

Keep the base launch otherwise identical:

~~~text
--jinja -fa on --no-context-shift -ngl 999 -np 1
-ctk q4_0 -ctv q4_0 -ctv-first q8_0,4 -ctv-last q8_0,4
-t 8 -tb 8 -b 2048 -wgt 16 -cuda offload-batch-size=16 --no-mmap
--moe-resident auto --moe-resident-profiler old --moe-resident-ubatch 512
-cmoe --ctx-size 65536 --k-cache-hadamard --v-cache-hadamard
~~~

Omit -ub for automatic initial sizing. Do not change threads, sampling or memory caps at the same time as speculation.

The current old residency executor accepts width 1 only (ggml-moe-resident-ffn.inc); one draft needs target width 2. The general target verification path already exists. Record which path runs, its transfers and its net rate before deciding to add a different executor. Falling outside this residency fast path is an observation, not automatically a failed configuration.

Include predictor weights, state/workspace and displaced target cache space in MTP's cost. A chain using lookup does not eliminate the allocated predictor's memory cost. Do not assume the experimental new policy's predictor cap applies to old mode.

Gate: model loads at configured 65k context, useful outputs, correct continuation/regeneration, and measured emitted rate including draft and commit costs. Compare against no-MTP baseline; acceptance rate alone is not success.

## Step 2 — Validate the small lookup repair and compare existing modes

Focused source regression gates:

~~~sh
# Only if the relevant build is authorized; testing is performed by Luna.
cmake --build build --target test-ngram-cache -j2
build/bin/test-ngram-cache
~~~

Observed 2026-10-15: Luna ran both commands successfully (exit 0), recompiling ngram-cache.cpp and speculative.cpp and relinking the common library and test. Test output: PASS for history edits, append boundaries, regeneration, limits, and committed-only learning. Logs and artifact hashes: .skills/benchmarks/2026-10-15-ngram-context-fix/. The server executable was not rebuilt; its earlier timestamp was recorded.

A test build is not a server rebuild; check dependency changes and do not silently build llama-server. If a server rebuild is needed, tell the user. The user normally rebuilds the server; a later explicit instruction for Luna to build overrides that preference.

Check the stage uses runtime n_max directly and that its caller retains the existing overall draft cap. Existing helper tests cover zero, one, two and three proposed tokens. Static inspection of stage wiring is not an integration-speed result.

Compare these modes against Step 1:

~~~text
--spec-type ngram-cache:n_max=1,n_min=1

--spec-type ngram-cache:n_max=1,n_min=1
--spec-type mtp:n_max=1,p_min=0.0
~~~

The existing stage chain already falls through when no usable lookup proposal exists. Keep that control flow. Do not add autotuning: explicit stage chains currently disable it.

Use repeated structured fields/neutral grammar literals and original prose as separate workloads. Lookup may help repeated labels and copied facts but have low hit rate on novel prose. All proposed tokens still pass through the target sampler/grammar.

Gate: full-context rebuild and split append yield identical cache counts; changed histories leave no obsolete context counts; proposal generation cannot modify committed history; useful emitted throughput improves without relaxed sampling or repeated-token degeneration.

## Step 3 — Change only what remains measurably expensive

After the comparison, select at most one first change:

| Measured result | Next action |
| --- | --- |
| Existing MTP or lookup chain already improves useful throughput | Keep the existing implementation and measured settings. No new engine. |
| Width-2 verification loses residency and increases weight traffic enough to erase its benefit | Add narrow width-2 residency support, preserving the existing numerical and checkpoint paths; prove all token/ID/scratch dimensions and per-prefix rollback. |
| Foreground weight fills dominate the winning mode | Evaluate a zero-fill admission candidate in isolation; retain current hits and compare bytes per emitted token and total speed. Do not enable the entire experimental new policy to obtain one choice. |
| Necessary copies still serialize with demonstrably independent GPU work | Design one bounded ready-event transfer overlap change, with explicit slot and pinned-source lifetime. Different streams alone do not make a kernel safe to read unfinished weights. |

No speculative branch is implemented without its gate. CPU thread/worker changes are not the first response to the measured warm trace. Keep one draft; longer windows require new evidence and every-prefix rollback proof.

## Verification protocol

No benchmark result is claimed for the new source until the rebuilt binary is tested.

Before every live request, notify the user of the test window and inspect actual per-process CPU/GPU load, free RAM, active paging, VRAM, clocks and temperature. Luna performs tests. Do not kill unrelated applications. Record server and linked-library hashes and exact argv.

Use approximately 16k useful prompt tokens, two follow-ups built from actual prior answers exactly once, and regenerate the third request from identical history. Keep configured context 65,536 and requested two-paragraph answers. Use the user's sampler/grammar settings consistently, with a neutral content fixture. Include a long-lived context test before promoting a mode.

Record server emitted decode rate, client elapsed throughput, prompt reuse, generated length, accepted/drafted tokens, stage hit/fallback counts and actual resident path. Collect expert transfer bytes only when needed to explain a failed performance result; avoid profiling every speed trial. Repeat paired warmed comparisons three times when selecting a winner. Stop profiling during final speed measurement.

A response that is empty, garbled, loops, loses continuity or changes the sampling contract fails regardless of reported speed. Same seed is not byte-identity proof across CPU/GPU numeric paths. Recurrent/KV and sampler state must match the committed prefix if any later verifier code is changed.

## Status and stop boundary

- Simplified plan: rewritten.
- Small n-gram source fix and regression cases: implemented; focused CPU build/test passed through Luna, with common/speculative.cpp compiled as part of the test build.
- Existing MTP/no-spec/lookup/fallback comparison: pending the matching server build and live test window.
- Width-2 residency, admission and overlap code: deliberately conditional, not implemented.
- 40 t/s objective: not demonstrated.

Stop adding code after the small lookup fix until measurements select the next change. This is the intended scope boundary, not permission to claim the overall performance objective complete.

Related flow diagrams: [002](002-moe-data-flow.md). They distinguish current execution from optional future work.
