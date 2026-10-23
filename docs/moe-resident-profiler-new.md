# Reuse-aware MoE residency

Select the opt-in policy with:

```sh
--moe-resident auto --moe-resident-profiler new
```

An explicit expert cap works too, for example `--moe-resident 1536`. The
README's "Recommended flags" lists the full measured setup, including
`--moe-resident-prefetch 2` and `-cuda fa-quant-tile=1`.
`--moe-resident-profiler old` remains the default. Residency omitted or set
to zero disables both policies. No other launch settings need to change.
The complete-FFN executor requires an eligible single-CUDA-plus-CPU layout.
The new policy supports one to four tokens, allowing up to three MTP drafts to
be verified with the current token. Larger batches retain ordinary execution.

For one-draft MTP verification, use the existing speculation option:

```sh
--moe-resident auto --moe-resident-profiler new --spec-type mtp:n_max=1,p_min=0.0
```

All target tokens use their actual router IDs. CPU kernels can group repeated
experts across the verification window; GPU hits keep using resident weights. Routing
readback, CPU dispatch and result upload operate on the batch. Masking and
packed-result scattering cover all expert rows of every token. This does not
use the predictor's expert IDs as a substitute for target routing, and does
not change draft acceptance, target arithmetic or sampling.
The graph builder compacts the batched top-k view before FFN execution: the
sort result's original expert-count stride is not a packed route buffer.
Before constructing an embedded MTP companion, target warm-up expert storage
is reclaimed so companion KV/compute allocations fit. Routing history and the
target graph arena remain available; residency refills after those allocations.
The tagged two-to-four-token complete-FFN graph can use existing CUDA graph replay
on Turing and newer devices. Ordinary batched expert graphs keep their prior
capture restrictions; the resident fused implementation launches each token
plane without host route readback. `new CUDA FFN graph replay active tokens=N`
confirms a successful captured launch. Cache resize/reclaim invalidates captures
before changing the bank storage or dimensions.

## Admission and reuse

The new policy separates current execution from cache admission. Resident
experts run on GPU. Misses execute both expert projections on CPU, without
forcing a weight upload for the current token. A bounded asynchronous promotion
can install a missing expert for future tokens. There is at most one promotion
in flight, and it uses the existing cache rather than a second expert arena.

GPU prompt routing accumulates an integer histogram in its existing execution
stream. Histogram collection does not split prefill graphs or read routes back
for every prompt token. At the generation boundary, a small histogram readback
seeds a bounded routing history: at most 64 tokens of count mass per layer.
CPU observations are accumulated directly when available. Decode observations
update the history, with a 128-model-pass half-life for older counts, matching
the recent-use horizon in the selected `strata-poly` engine. All eligible FFN
layers advance history age, including fully resident layers. The live target
graph determines the pass width, excluding extra MTP predictor layers.
The histogram accepts both original `blk.N.*` bank names and scheduler staging
names such as `CUDA0#blk.N.*#0`. A `new history layers=... references=...` line
reports nonempty GPU history collection. A subsequent cache log's `history=`
field counts the ranked candidates, not the number of prompt tokens.

Preload and capacity allocation rank experts by reuse count times observed CPU
cost, divided by their padded storage size. Live layouts retain at least one
route's worth of slots. Before CPU costs are observed, the policy uses a common
initial estimate. Replacing a live victim requires at least twice its recent
reuse score plus two uses, as well as a doubled cost-weighted victim margin
and an upload-cost estimate; completed timed uploads update that estimate.
An empty slot can bootstrap without evicting a resident expert.
Victim ranking is refreshed once per approximate model-layer sweep. Each
admission rechecks up to 64 ranked candidates against current scores, active
route protection and in-flight ownership. This removes a full cache scan with
exponential score evaluation from every layer's admission decision.

The original boundary preload uses routing history. Boundary
preload fills vacant slots; it does not replace every occupied slot with the
latest global ranking. Live admission considers the current layer's misses,
with one promotion in flight. The selected `strata-poly` engine starts from a
saved count profile and chooses recent-use swaps across layers, with up to 64
pending uploads by default. Those global candidate and concurrency mechanisms
are not implemented by this policy.

## Token-conditioned expert prediction

The new policy also learns per-token target-layer routing associations from
actual one-to-four-token target decode routes. The table holds at most 512 token
IDs; per-layer expert counters decay after 32 observations. For Qwen's 41-layer,
256-expert layout, dense byte counters occupy about 5 MiB of ordinary CPU RAM
when the table fills, plus container overhead. This adds no predictor tensor,
second expert arena or GPU routing-history allocation.

MTP passes the carry token and its first three actual draft proposals to the
target cache before verification begins. Without MTP, the chosen input token
IDs trigger the same forecast before the target graph runs. These token IDs
condition predictions of target-layer expert IDs; the appended predictor's own
expert IDs are not reused as target expert IDs. This is an online association
predictor, not an additional neural router model. It needs repeated decode
observations and is not trained from the aggregate prefill histogram.

A forecast needs at least four observations and 75% empirical consistency.
All confident forecasts remain in the protection and accuracy plan. At most
two experts per layer enter the upload queue, ordered by layer deadline,
missing-before-resident status, then confidence. Residency checks include the
current bank identity. Forecast probability contributes a short-horizon priority term of
32 times probability alongside decayed reuse. Predictive replacements need a
10% value margin plus the measured upload-cost term; ordinary miss admission
retains its doubled reuse/value thresholds. Confident future residents are
protected until that layer's result is queued, then their protection expires.
Each scheduling point attempts at most two admissions and permits only the
existing single background upload in flight. Predictions cannot exceed the
current cache capacity or the user cap.

Binding to actual verification tokens removes stale forecasts when drafts are
shortened/replaced. Planned IDs and metadata are copied; no predictor context or
request storage is retained. Upload-queue consumption is separate from the
accuracy plan. The background worker retains its existing owned descriptors,
staging buffer and victim fence. Completion covers result and lookup DMA before
their host sources can be reused. Actual router IDs always control inference;
an inaccurate forecast can waste a cache slot/upload, but cannot select a wrong
expert for arithmetic.

`new prediction` reports calls, forecast candidates, selected forecasts checked
against actual route unions, matches, predictive uploads, first actual uses of
predictively uploaded slots, and learned token count. Upload/use counters are
separate from forecast accuracy. The actual server rebuilt successfully; default,
`--cuda-layout`, `--cuda-resident-batch`, and `--cuda-resident-prediction` passed.
The last mode teaches both already-resident and deliberately incorrect missing
experts, confirms real predictive uploads/uses, and checks all actual result rows
against their native CPU/GPU reference branch. It also exercises error/retry,
stale-plan binding, old/off gating and bounded history. These are correctness
checks, not model-level forecast accuracy or throughput evidence. The first
full-model prediction run measured 32.24 t/s over three warmed follow-ups.
It checked 49,819 forecasts against actual target routes, with 41,838 matches;
15,571 predictive uploads had 7,157 first actual uses. GPU route share was
41.3%, and weight upload traffic increased substantially without a demonstrated
speed improvement. That run preceded the separation of full resident protection
from the two-entry upload shortlist and the CPU preparation changes below.
The combined changes were then measured with both `-t 16` and `-tb 16`, as
requested. Three warmed follow-ups averaged 18.42 t/s. The native team logged
16 workers, but process-wide execution samples were approximately 9–11 core
equivalents. Target CPU completion averaged 1,676.9 us/layer and the deferred
join still waited 1,660.8 us/layer. GPU route share was 41.7%; that counter is
the fraction of routed expert rows handled on GPU, not SM utilization. The
preceding incomplete 8-thread run had CPU completion of 692.5 us/layer and
join wait of 657.3 us/layer. Desktop activity and swap deltas limit comparisons,
but the 16-thread run demonstrates neither a speed gain nor the 40 t/s target.
The existing shared-branch overlap does not implement Strata's two-half
cross-layer wavefront; nearly all CPU completion time remains exposed at the
join. Evidence is in
[the prediction results](../.skills/benchmarks/2026-10-14-luna-build-test/warm-diagnosis/mtp-expert-prediction/RESULTS.md).

## Next-layer prefetch

`--moe-resident-prefetch N` (0 to 8, default 0) adds a forecast-driven upload
ring. It requires `--moe-resident` and `--moe-resident-profiler new`; 0 builds
the same graph as before.

For one-to-four-token batches, each layer's graph gains one small GPU node:
the next layer's router applied to `rms_norm(this layer's FFN input)` times the
next layer's `ffn_norm`. Its top-k IDs are read back with the routing data. On
Qwen3.6-35B-A3B this forecast matched the real next-layer top-2 at 97.6% and
top-4 at 94.7%. The forecast IDs never enter arithmetic. The real router still
selects every expert.

After this layer's CPU join, a dedicated CUDA stream uploads up to N forecast
experts that are not resident. They go straight from pinned host weights into
`2*N` extra VRAM ring slots. Starting after the join matters: the upload then
overlaps the next layer's GPU attention while the CPU is idle. Uploads that
overlapped CPU experts slowed each CPU expert by 14-25% through host DRAM
contention. When the next layer is routed, its ring entries join the slot
mapping. A still-running upload makes only the GPU half of that layer wait on
the upload event. A used entry that rates higher than its victim moves into the
cache by a device copy, with no host traffic. A forecast upload waits on a
compute-stream event recorded after the last readers of its ring slot.

Pinned host banks (`CUDA_Host`) also let ordinary admission DMA directly from
the source weights. This skips the staging memcpy. Pageable banks keep the
staged path, and the ring stays inactive (`skipped` counts this).
The new policy's CPU join spins up to 500 us before it parks.

On 2026-10-23, Qwen3.6-35B-A3B UD-Q4_K_M ran on an RTX 2060 and a Ryzen 7 2700X
with `-cmoe --moe-resident 2048`. The prompt had 9,578 tokens, then 256 greedy
tokens were generated. Mean decode speed over interleaved runs:

| `--moe-resident-prefetch` | runs | t/s | `gpu_share` |
|---|---|---|---|
| 0 | 4 | 30.2 | 13% |
| 1 | 2 | 33.4 | 38% |
| 2 | 4 | 33.9 | 48-54% |
| 3 | 2 | 29.6 | 57-58% |

N=3 transfers about 5.4 MB per layer. That exceeds the idle window, and router
wait grows from about 220 to 350 us/layer. The useful N depends on PCIe speed,
expert size and attention time. Output stayed coherent. The text is not
bit-identical: a different CPU/GPU split changes floating-point summation
order, and two baseline runs already differ.
Shutdown logs report
`new prefetch uploads= installed= waited= hits= unused= promoted= skipped= bytes=`.
`waited` counts installs that made the GPU wait for the upload.

## CPU execution

Eligible CPU misses use a persistent native worker team owned by the scheduler.
It executes the existing GGML graph plan and kernels, with the requested CPU
thread count and native GGML barriers. It does not enter an OpenMP parallel
region for each expert graph. The submitting dispatcher is worker zero; all
workers complete before graph-plan stack data or scratch can be reused.
The ordinary graph executor and old residency policy keep their original path.

The new-policy CPU job also owns one sorted, unique list of its actual missing
expert IDs for the entire verification batch. Up/gate and down projections share
that list, so the IQK chunk dispatcher addresses only participating experts
instead of repeatedly scanning the entire expert bank. Down-projection input
quantization skips masked rows already handled on GPU. The original grouped
matrix kernels and actual target route IDs still determine all arithmetic;
forecast IDs do not enter this list. Its storage remains immutable until the
CPU job joins. Untagged nodes retain the original preparation and dispatch.

The combined source rebuilt successfully and passed default, layout, batched
numerical, and prediction checks. At widths two, three and four the batched
fixture showed fewer unique experts than CPU rows and nonzero skipped
quantization rows, while retaining the native branch comparisons. These checks
establish correctness of the exercised paths, not a model throughput gain.

## Routing and result transfers

A device table maps original layer/expert IDs to physical cache slots. A fully
resident layer uses that table on GPU and skips host routing/activation readback.
Partially resident layers transfer their small routing and activation inputs
through ordinary asynchronous copies into pinned host scratch. A scoped event
wait makes those inputs visible before the CPU job starts.

CPU misses start before GPU submission. The actual cache-hit IDs are uploaded
from a separate half of pinned ID scratch, and their GPU FFN overlaps the CPU
job. An all-miss route submits no GPU FFN. This replaces the earlier mapped-host
readback kernels and unconditional early masked GPU submission for partial
layers; a partial all-hit route still transfers routing and activation inputs.

CPU result rows are packed into one host upload and scattered to their original
positions on GPU. GPU rows stay intact. Expert selection, result row order,
weighting, reduction, KV storage, recurrent state and sampling are unchanged.
Routes without any cache hit bypass the GPU FFN and upload the CPU result
directly, including all-miss routes in partially resident layers.

For small eligible graphs with GPU shared-expert weights, graph construction
places the independent shared-expert branch between the routed projections and
their reduction. This happens before allocation, so the allocator retains the
routed output until its consumer. The scheduler validates that every intervening
split runs on the same GPU and does not consume the routed result. It can then
queue that branch while the routed CPU job is running, joining and uploading
the CPU rows before the reduction. A CPU split or another routed FFN prevents
deferral. Error exits join the worker and synchronize GPU users before releasing
the pending owner; the next invocation cannot reuse its scratch early.
This mechanism is gated by auto residency and profiler=new. It does not split
the attention graph or recurrent/KV state into speculative halves.
The independent shared branch carries a private graph marker so it remains
eligible for CUDA graph replay on Turing after it is separated from the next
router. Replay still requires an active new-policy histogram, at most four
tokens, no ordinary expert operations, and the existing node compatibility and
pointer-update checks. Successful launches report `new CUDA shared graph replay
active tokens=N nodes=M`.

## Device join (experimental)

`--moe-resident-device-join` (new profiler only, default off) queues a layer's
CPU-result upload and the following GPU work before the CPU experts finish.
The CPU worker packs its result rows and then stores a flag in pinned memory.
A GPU-side wait on that flag releases the queued work, so the host does not
block on the CPU join before submitting it.

With `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1`, the wait is a hardware
`cuStreamWaitValue32`. Otherwise a small polling kernel waits, with a 10 s
timeout. GeForce drivers ship with `NVreg_EnableStreamMemOPs=0`, and the CUDA
driver then emulates stream waits by blocking the host thread, which defeats
the purpose. `sudo scripts/nvidia-stream-memops.sh enable|disable|check` sets,
removes or checks that module option. A reboot applies it.

No speedup has been measured with the polling kernel, and the hardware path
has not been measured yet. The recommended setup leaves the flag off.

## Promotion and resizing ownership

An admitted victim is removed from the device lookup and marked unavailable.
Active route slots cannot be victims. The upload stream waits for the current
compute/result fence before writing the victim's storage. Its identity becomes
visible only after the upload completion event. Source weight identities are
checked before a subsequent route can reuse a slot. Reset, cache destruction,
workspace release and scheduler destruction finish outstanding uploads first.
Every publication marks the device lookup dirty, including publication from a
graph scan; the next dispatch refreshes it after fencing the reusable DMA source.
Legacy projection caches are released before measuring the complete-FFN budget.
The cache is planned from the decode graph's expert layers. A partial graph,
such as a prefill ubatch whose last layer is a one-token FFN, does not create
or shrink it. See [moe-resident-ubatch.md](moe-resident-ubatch.md).

A dedicated host upload worker stages the admitted expert's three banks into
one reusable pinned RAM buffer, then queues their copies on the upload stream.
The generation caller does not perform the staging copies or weight upload
calls. The worker publishes an atomic queued marker only after recording the
upload completion event, preventing a stale event from publishing a new job.
Reset/resize waits for both host submission and GPU completion; destruction
joins the host worker before freeing its events, staging buffer or backend.
There is still only one promotion in flight and no second GPU expert arena.

On CUDA devices supporting VMM, each cache bank has a fixed virtual address.
Budget refresh changes committed physical pages without allocating a second
cache or reuploading unchanged slots. On shrink, valuable tail slots can move
into the retained prefix using device-to-device copies before pages are freed.
Physical page rounding and histogram/lookup storage are deducted from the
cache budget. This policy keeps a 256 MiB free-memory margin, additional to
the already allocated model, full context and compute workspace.

Without VMM, a fitting ordinary cache is retained and growth is deferred.
Required shrink can rebuild it. A full adaptive workspace release still reclaims
expert storage for the initial large prompt after a full target-context reset;
returning to generation preloads from retained routing history. Generation and
follow-ups then stay within `--moe-resident-ubatch N` until another full reset.
A follow-up above `N` is chunked rather than reclaiming experts for a larger
workspace. Stable resizing does not promise to retain expert bytes while their
physical VRAM is needed by another allocation.

## Status and diagnostics

### Current results (2026-10-23)

RTX 2060 (6 GB) with a Ryzen 7 2700X, Qwen3.6-35B-A3B UD-Q4_K_M, `-cmoe
--moe-resident 2048 --moe-resident-profiler new --moe-resident-prefetch 2
-cuda fa-quant-tile=1`, no `-ub`, `-t 8 -tb 8`:

| Workload | Prefill | Decode |
|---|---|---|
| Three-turn chat, no MTP | 701 t/s | 41–44 t/s |
| Three-turn chat, MTP `n_max=1` | 430–590 t/s | 45–48 t/s |
| 28K-token chat request, MTP | about 440 t/s | 36–38 t/s |

`gpu_share` is about 64–75% on short chats and about 60% at 28K context. At
28K, CPU expert time (`cpu_wait` about 380–450 us/layer) and lower draft
acceptance (about 0.70 against 0.80) are the remaining costs. Router wait is
back to its short-context level (about 270 us/layer) with `fa-quant-tile=1`.

`-tb 16` on this 8-core CPU dropped MTP decode from 45–48 to about 20 t/s and
prefill from 592 to 435 t/s. MTP verification runs CPU experts with the batch
thread count, so `-tb` above the physical core count makes SMT siblings compete
for the same cores and DRAM bandwidth.

### Earlier validation (2026-10-14)

On 2026-10-14 the incremental CUDA server build and rebuilt residency test
target passed. The default test compares persistent CPU execution against
ordinary execution for quantized masked/repeated rows across thread counts
1, 4, 2, 8 and 1. Existing `--cuda-layout` numerical tests also passed; those
exercise the old policy. The new `--cuda-resident-batch` test passed numerical
comparisons against native CPU/CUDA branches for two-to-four-token mixed quantized and
merged-bank FFNs, masked/repeated expert rows, changing routes and asynchronous
promotion/eviction. These checks do not prove complete MTP recurrent/KV state
equivalence; end-to-end MTP validation is recorded separately below.

The subsequent shared-expert overlap change rebuilt the actual CUDA server and
test target. All three focused modes passed again. The batched numerical
fixture exercised deferred GPU work at widths two, three and four, injected a
GPU failure while CPU work was pending, retried successfully, and verified
immediate completion when an intervening branch uses the CPU backend. The
full-model run measured 32.84, 33.38 and 31.82 t/s on warm follow-ups, 32.96 t/s
weighted, with 58,441 deferred jobs and queued GPU splits. Desktop activity and
incomplete concurrent sampler coverage limit the comparison with preceding
runs. This remains below 40 t/s. The subsequent shared-branch capture addition
passed its build and numerical checks and measured 32.13 t/s weighted across
three warmed follow-ups, with replay logged on both one- and two-token branches.
Those results are separate from
[the overlap validation](../.skills/benchmarks/2026-10-14-luna-build-test/warm-diagnosis/mtp-shared-overlap/RESULTS.md).

A neutral chat with a 15,913-token initial prompt, 65536 configured context,
and two actual follow-ups exercised the native team and host upload worker.
It produced visible English answers at 25.79, 28.40 and 27.41 t/s with reasoning
disabled. Other CPU and desktop GPU activity was present. No paired old-policy
measurement was made. Commands,
logs, requests, outputs and limitations are recorded in
[the validation results](../.skills/benchmarks/2026-10-14-luna-build-test/RESULTS.md).

The later 15,947-token prompt and three actual follow-ups, using one MTP draft,
confirmed captured layer-head router graphs of 64/65 nodes on RTX 2060. Warm
follow-ups measured 31.64, 32.68 and 31.23 t/s (32.02 weighted). Four-token MTP
verification measured 28.48 t/s weighted before router capture. These runs do
not meet the 40 t/s target and desktop load varied. A `-t16 -tb8` control still
used an eight-worker resident team: multi-token MTP verification selects the
batch thread count, so `-tb` is relevant to its CPU expert work. Full evidence:
[router replay](../.skills/benchmarks/2026-10-14-luna-build-test/warm-diagnosis/mtp-router-capture/RESULTS.md),
[thread control](../.skills/benchmarks/2026-10-14-luna-build-test/warm-diagnosis/mtp-threads16/RESULTS.md).

Cache allocation logs include `moe-resident new cache=`, slot count and
`stable=yes|no`. Shutdown diagnostics include `all_hit_layers`, `promotions`,
`rejected`, `d2h_activation`, `packed_h2d_results`, and actual `d2d_weights`.
These supplement the existing cache hits, fills and host-weight upload counters.
`new stats` lines appear every 64 model-layer sweeps, at a later generation
boundary and at shutdown. Values are cumulative across the scheduler lifetime:
`gpu_share` is the fraction of selected expert references computed on GPU;
`empty_gpu` counts layer invocations with no resident expert; `map_uploads`
counts lookup uploads. `cpu_team=native` identifies this executor version.
`cpu_only` counts routes that submit no GPU FFN; `dma_routes` counts partial
routes using asynchronous readback, and `mapped_routes` counts fully resident
routes using GPU lookup without host readback.

Host timers report router-event wait, CPU dispatch-through-completion elapsed
time, GPU FFN enqueue and lookup-source wait in microseconds per eligible layer.
`admission` measures caller-side replacement decisions/submission; `upload_host`
measures staging and CUDA enqueue on the separate upload worker, amortized over
eligible layers. `ffn_host` measures the complete caller-side FFN invocation,
including initial cache setup. These are not GPU kernel timings or an end-to-end
model breakdown. CPU/GPU and the upload worker overlap; CPU elapsed also covers
concurrent GPU enqueue. The timers must not be added as independent costs.
For deferred jobs, CPU completion is timestamped by the worker, so admission
does not mistake a later consumer join for extra CPU execution time. A separate
`new overlap` line reports deferred layer count, intervening GPU splits and
remaining join wait per deferred layer. A positive split count proves that GPU
work was queued before the join; it does not measure GPU execution overlap or
establish a throughput improvement.
