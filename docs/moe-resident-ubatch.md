# Adaptive MoE compute workspace

`--moe-resident-ubatch N` enables a compute-workspace handoff under
automatic MoE residency. `N` is the generation and follow-up microbatch ceiling in tokens,
not an expert-cache budget. `--moe-resident [MiB|auto]` still controls experts.

With `--moe-resident auto` or a MiB cap and no `-ub`, the handoff is on by
default. The ceiling is then the smallest multiple of 512 at which the GPU
backend offloads host-resident expert ops (see below), at least 512. It stays
off with `-np` above 1, graph or attention split modes, an evaluation callback,
`-b` below 512 or no GPU, so those keep the fixed workspace.

| Options | Workspace policy |
| --- | --- |
| No residency, or `-ub` given, or `--moe-resident-ubatch 0` | Ordinary fixed workspace policy |
| `--moe-resident auto` (or a MiB cap) without `-ub` | Automatic prefill workspace, then the offload-derived ceiling until full context reset |
| Residency omitted, or `--moe-resident 0` | Adaptation disabled, even if the workspace option is present |
| `--moe-resident auto --moe-resident-ubatch 512 -ub 2048` | Initial prefill ceiling up to 2048, generation and follow-up ceiling 512 until full context reset |
| `--moe-resident auto --moe-resident-ubatch 512` without `-ub` | Choose the largest measured initial prefill workspace fitting available GPU memory, in multiples of 512, up to `-b`; then retain the 512 ceiling until full context reset |
| `--moe-resident 1536 --moe-resident-ubatch 512` | Same automatic prefill policy, with a separate 1536 MiB expert cap per CUDA backend |

Explicit `-ub` is an upper limit. Both policies can fall back to a smaller
prefill ceiling when memory is insufficient. The automatic policy only chooses
multiples of 512; its smallest candidate is the generation ceiling rounded up to
512. `-b` remains the logical batch limit: increase it if you want automatic
prefill sizing to consider a larger microbatch. `llama_n_ubatch()` reports this
configured upper limit, while execution uses the context's active ceiling.

For example, add this to the existing Qwen launch arguments and omit `-ub`:

```sh
-b 3072 --ctx-size 65536 -np 1 -cmoe \
--moe-resident auto --moe-resident-ubatch 512 \
-cuda offload-batch-size=16
```

To bound prefill explicitly, add `-ub 2048`. Keep the existing KV quantization,
Hadamard, Flash Attention, thread and MTP arguments. A residency cap remains
optional; adaptation never exceeds an explicit cap. Freed memory may remain
unused if that cap, expert layout, the number of distinct experts or reserved
CUDA headroom prevents further growth. Automatic residency, with or without
adaptive ubatching, sizes complete-expert slots from the byte budget up to all
experts in the eligible layers. The former limit of twice the active-expert
count per layer is removed for this cache.

## Expert offload batch

Prefill with host-resident experts is dominated by the offload path: for each
microbatch the CUDA backend uploads a layer's expert banks and computes there.
That pays off only above some batch. The backend's rule
`ubatch * n_expert_used >= offload-batch-size * n_expert` depends on the model,
and 512 tokens qualify for Qwen3.6-35B-A3B only with `offload-batch-size=16`.

Under automatic residency, unless `-cuda` sets `offload-batch-size` or
`offload-batch-size-per-byte`, context creation measures it once per model on
the first host-expert layer: the upload of a 64 MiB slice of the real expert
weights, and the CPU expert graph at 64 and 512 tokens. Expert ops offload from
the batch at which the CPU estimate exceeds the upload time (GPU compute after
the upload is not counted). Dense ops keep the `offload-batch-size` rule. The log
line `expert offload: upload ... -> offload from N tokens` reports the result,
and `adaptive generation ceiling 512 -> N` reports a raised ceiling.

## Refresh at the start of a response

`llama-server` calls `llama_moe_resident_begin_generation()` once per response
after prompt evaluation, including follow-ups and regenerations. This applies
to both `--moe-resident auto` and `--moe-resident N`; disabled residency is a
no-op. Other API callers can issue the same notification after prefill and
before starting their generation loop. With adaptive ubatching this notification
also latches the small workspace ceiling for the rest of the context lifetime.
The next decode shrinks a remaining large initial-prompt workspace before
uploading input or modifying KV state. The notification itself does not discard
the last prompt's host output used for first-token sampling.

The notification finishes pending computation/readback, invalidates CUDA
captures, releases idle CUDA scratch and stale legacy bank pools, and requests
a fresh graph. Complete-expert capacity is checked after that real graph's
arena is allocated, before its computation. The memory calculation includes
the existing cache as reclaimable memory. Explicit caps remain in force.
`--moe-resident-profiler new` keeps a 256 MiB free-memory margin for both
initial-workspace sizing and expert-cache sizing. The old policy retains at
least 256 MiB or one eighth of available memory, with a 512 MiB minimum when
adaptive ubatching is enabled. The margin is additional to already allocated
model, full context, compute buffers, histogram/lookup and physical page padding.

A fitting cache is retained. Growth of less than 64 MiB is deferred to avoid
reuploading the working set for small driver-memory fluctuations. Larger
growth or a required reduction rebuilds the cache and restores valid previous
expert identities, hottest first, from the original host weights. A previous
allocation failure is retried on the next response. Refresh does not occur on
every generated token and does not change KV or recurrent state.

## Allocation and lifetime

The implementation currently supports Qwen3.5 MoE on one CUDA backend with
causal attention, one sequence and ordinary layer scheduling. Evaluation
callbacks and graph/attention split modes are rejected when adaptation is
requested. Embedded MTP is supported by coordinating the linked target and
predictor workspaces. External draft contexts keep their ordinary workspace
policy; the automatic budget sees their existing allocations as occupied VRAM.

Only the initial prompt after context creation or a full target-context reset
may select a workspace larger than `N`. Within that phase, batches above `N`
can use the selected large workspace; smaller batches can shrink it early.
Once generation starts, every later batch is processed in microbatches of at
most `N`, including large follow-ups. Follow-ups do not release the expert
cache to obtain a larger prompt workspace.

`llama_kv_cache_clear()` and successful full-range sequence removal that leaves
the target KV empty re-enable initial-prompt growth. Partial removal, ordinary
regeneration and checkpoint restore of a populated conversation do not. An MTP
predictor reset cannot re-enable target growth. API clients should notify the
generation boundary explicitly; as a fallback, a single-token target decode in
an already populated context latches the small ceiling too.

At a transition, both linked contexts finish outstanding computation and
readback. The scheduler invalidates reusable graph descriptors, remembers the
complete-expert cache identities and reuse order in RAM, releases expert and
physical graph allocations, and invalidates CUDA captures. Idle legacy CUDA
pool buffers or idle VMM physical mappings are released. Checked-out scratch
is retained. Maximum graph metadata capacity remains in RAM.

Fresh worst-case graphs use the full configured context. The predictor's
warmup and one-token prediction shapes are both measured/reserved. Graph
measurement does not execute inference or allocate the graph arena. Automatic
prefill sizing checks candidates in descending 512-token steps, because kernel
and graph choices need not have monotonic memory requirements. It leaves the
policy's free-memory margin for transient CUDA scratch and other allocations.
It reserves the selected target and predictor arenas
before replanning residency from the remaining physical memory.

The complete-expert cache is allocated lazily on an eligible decode graph.
It uses actual remaining VRAM, the same headroom rule and any user cap. Previous
experts are refilled from the original host weights, hottest first, after bank
identity and layout checks. Slots become usable after uploads finish. Old and
new VRAM caches never coexist. Legacy bank caches are dropped at the handoff;
their entries are not preserved by the complete-expert working-set mechanism.

Conversation KV, recurrent state, sampler state, host output storage and model
weights are not resized or converted. Refill adds host-to-device traffic and
transition latency. The extra RAM state is expert identity/order metadata;
this does not add another full host copy of the model.

## Diagnostics and validation status

Logs report selected ceiling, number of coordinated contexts, full context
length, graph arena bytes before/after, released idle scratch and handoff time.
Cache logs report capacity, budget, reserve and restored expert count.

Recoverable graph-arena allocation failures retry a smaller prefill ceiling.
A failed generation reservation or real graph allocation returns a decode
error. This does not make all CUDA OOMs recoverable: transient kernel pools
still use the backend's existing fatal CUDA error handling. Headroom is a
conservative policy, not a proof against other processes taking VRAM.

On 2026-10-12, Luna exercised an earlier implementation of the user-built executable with the Qwen3.6
35B-A3B UD-Q4_K_M GGUF, a 65,536-token configured context, Q4 KV with Q8
first/last layers, both Hadamard options and no MTP. Fixed 512, adaptive with
an explicit 512 ceiling, and adaptive automatic prefill up to 2048 produced
coherent short conversations. Automatic prefill also processed a 16,050-token
prompt and two factual follow-ups coherently. Logs confirmed the graph arena
grew from 537.02 to 772.08 MiB, then shrank to 537.02 MiB for generation.
Requests and responses are saved under
`.skills/benchmarks/2026-10-12-adaptive-corruption/`.

For that long prompt, the fixed-512 control took 56.870 seconds for prefill
and decoded the first response at 32.43 tokens/s. Automatic prefill took
22.612 seconds and decoded at 33.41 tokens/s. The two follow-up decode rates
were 27.95/29.71 tokens/s for fixed 512 and 29.69/30.91 for automatic prefill.
These are single-run observations with temperature zero and seed 4242, not a
repeated performance study or a comparison against PolyStrata. Both runs
still used the old 656-slot expert-cache limit.

These runs did not reproduce the reported garbled output. They do not establish
correctness for the original failing conversation or the embedded MTP path.
The subsequent changes lifting the automatic cache's twice-top-k slot limit
and refreshing its budget per server response are source-only and require a
rebuild and runtime check. No throughput gain is claimed for those unbuilt
changes. Runtime acceptance must cover repeated responses with and without
adaptive ubatching, explicit caps, disabled residency, and embedded MTP.
