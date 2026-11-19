# Implementation plans

Rewritten 2026-10-15 against baseline `0e4e40b0`, following the user's request to simplify and implement. These documents describe status and scope; read direct user instructions before execution.

| Plan | Purpose | Status |
| --- | --- | --- |
| [001](001-moe-residency-and-speculative-throughput.md) | Compare existing speculation; small n-gram history/draft-limit repair; only then select a measured bottleneck fix | SOURCE FIX + CPU REGRESSION PASS; LIVE COMPARISON PENDING |
| [002](002-moe-data-flow.md) | Current memory/compute/conversion/synchronization flow and proposed speculative flow | DOCUMENTATION |
| [003](003-resident-dispatch-overlap.md) | Full handoff optimization sequence; first patch moves GPU dispatch before the host wait and policy learning after CPU submission | LIVE 33.7 T/S; GAIN UNPROVEN; WIDER CUDA FAILURES ALSO REPRODUCE ON BASELINE |
| [004](004-cpu-handoff-latency.md) | Bounded CPU coordinator/team hot waiting without predictor VRAM | FOCUSED CHECKS PASS; OBSERVED 33.21 -> 37.44 T/S; SINGLE-PAIR LIMITS DOCUMENTED |
| [005](005-decode-layer-timeline.md) | Opt-in per-layer decode timeline (`GGML_MOE_RESIDENT_TIMELINE=1`) + 5-run matrix (baseline, timeline, prefetch 0/1, -t 6); picks the next fix from measured buckets. Planned at `21c8c246`, 2026-11-17 | DONE: FIVE RUNS VERIFIED; REPORT IN `.unlazy/timeline/REPORT.md` |

Plan 001 replaces the nine-stage architecture proposal with three steps. Preserve the committed approximately 30 t/s baseline and its workspace lifecycle. No new engine is justified until existing one-draft MTP and lookup are compared.

Deferred: a feature-flag framework, adaptive chain controller, persistent lookup database, discarded experiments, forced width-8 speculation, external draft models, lazy KV/VMM redesign and unbounded expert forecasting. Width-2 residency, admission and transfer overlap remain conditional on measurements.

2026-11-17: plan 005 added. Rationale: at 28K context, CPU expert work is 36% of the ~1,025 us layer cycle and already reads ~28.8 GB/s; the other ~64% (router wait, host handoff, GPU segment) has no breakdown. Kernel rewrites are rejected until 005's REPORT shows CPU compute is not bandwidth-bound.
