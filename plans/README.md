# Implementation plans

Rewritten 2026-10-15 against baseline `0e4e40b0`, following the user's request to simplify and implement. These documents describe status and scope; read direct user instructions before execution.

| Plan | Purpose | Status |
| --- | --- | --- |
| [001](001-moe-residency-and-speculative-throughput.md) | Compare existing speculation; small n-gram history/draft-limit repair; only then select a measured bottleneck fix | SOURCE FIX + CPU REGRESSION PASS; LIVE COMPARISON PENDING |
| [002](002-moe-data-flow.md) | Current memory/compute/conversion/synchronization flow and proposed speculative flow | DOCUMENTATION |
| [003](003-resident-dispatch-overlap.md) | Full handoff optimization sequence; first patch moves GPU dispatch before the host wait and policy learning after CPU submission | FOCUSED CHECKS PASS; WIDER CUDA FAILURES ALSO REPRODUCE ON BASELINE; SPEED UNMEASURED |

Plan 001 replaces the nine-stage architecture proposal with three steps. Preserve the committed approximately 30 t/s baseline and its workspace lifecycle. No new engine is justified until existing one-draft MTP and lookup are compared.

Deferred: a feature-flag framework, adaptive chain controller, persistent lookup database, discarded experiments, forced width-8 speculation, external draft models, lazy KV/VMM redesign and unbounded expert forecasting. Width-2 residency, admission and transfer overlap remain conditional on measurements.
