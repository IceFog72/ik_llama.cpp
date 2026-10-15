# Data flow: current Qwen MoE execution and proposed speculation

Source: committed `0e4e40b0`. Current means the measured launch with old residency, automatic residency/initial ubatch, generation ceiling 512, no MTP or other speculation. Proposed diagrams are labeled separately. Arrows express dependencies and data movement, not proportional elapsed time.

## 1. What A3B means and where memory lives

The inspected GGUF has 41 blocks, including one MTP predictor tail. The target graph loops over **40 transformer blocks**. It selects **8 of 256 routed experts per block and token**, plus the shared expert path. A3B describes approximately three billion active parameters per token across the model; it does not mean three billion parameters are permanently in VRAM or that there is a separate A3B layer group. Attention, router, shared expert, embedding and output weights are also used.

```mermaid
flowchart LR
  subgraph DISK[SSD — load at startup]
    FILE[GGUF — quantized model weights]
  end
  subgraph RAM[CPU RAM]
    HOSTW[Pinned original routed-expert weights<br/>about 19 GiB in this run]
    HOSTIO[Token history, sampler, grammar<br/>CPU expert scratch and host readback]
    LOOKUP[N-gram cache if enabled<br/>CPU token statistics — not expert weights]
  end
  subgraph VRAM[GPU VRAM — approximately 6 GiB device]
    BASE[Persistent shared model weights<br/>embedding, attention/recurrent projections<br/>routers, shared FFNs, output head]
    STATE[Persistent state<br/>full-attention K/V + recurrent state<br/>logical context 65,536]
    WORK[Temporary compute workspace<br/>larger initial prefill, smaller generation]
    EXP[Resident routed-expert cache<br/>subset of host weights<br/>about 1,845 MiB in the baseline]
    FREE[Free allocation headroom<br/>512 MiB reserve in this old adaptive policy]
  end
  FILE --> HOSTW
  FILE --> BASE
  HOSTW -->|Selected UP/GATE/DOWN weight bytes over PCIe<br/>only when a chosen GPU expert misses| EXP
  HOSTIO <-->|Token/activation/ID/result/logit transfers| WORK
  LOOKUP -.->|Proposed token IDs only| HOSTIO
  BASE --> WORK
  EXP --> WORK
  STATE <--> WORK
```

The expert cache stores the original quantized weights with device layout/padding. A cache fill does not convert the entire model to FP16. CPU-computed experts read their original weights in RAM and return results; their full weights need not cross PCIe. A cache hit reads existing VRAM weights without another RAM upload.

## 2. Startup, prefill and workspace handoff

```mermaid
flowchart TD
  A[Load committed model and options] --> B[Place shared model weights on GPU<br/>place original routed experts in pinned RAM]
  B --> C[Allocate persistent KV/recurrent state<br/>configured capacity 65,536]
  C --> D[Initial-prompt phase<br/>measure feasible compute workspace<br/>omitted -ub: choose in steps of 512, up to -b]
  D --> E[Prefill prompt in microbatches<br/>measured initial ceiling: 2,048]
  E --> F[Initial prompt completes<br/>preserve final prompt output for sampling]
  F --> S[Safe workspace handoff<br/>finish outstanding work/readback<br/>invalidate captures that reference old scratch<br/>release/reallocate compute buffers]
  S --> G[Generation ceiling 512<br/>measured CUDA compute 772 MiB → 537 MiB]
  G --> H[Recalculate expert residency budget<br/>grow/refill cache if useful and safe]
  H --> I[Generate token after token<br/>follow-up prefills also stay at or below 512]
  I --> J{Full target context reset?}
  J -->|No| I
  J -->|Yes| D
```

Only the compute workspace changes size. Existing KV/recurrent state is not transformed from an ubatch-2048 format into an ubatch-512 format. Ubatch is how many token rows a compute graph may process at once, not the length or format of the stored conversation. Normal decoding here uses one token per target call despite the 512 workspace ceiling.

## 3. One complete target token

```mermaid
flowchart TD
  INPUT[CPU: prompt token IDs or last sampled token] -->|H2D token IDs| EMB[GPU: embedding row lookup<br/>working hidden vector width 2,048]
  EMB --> BLOCK[Enter target block i — repeat 40 times]
  BLOCK --> AN[GPU: attention normalization and projections]
  AN --> KIND{Recurrent/linear or full-attention block?}
  KIND -->|Recurrent/linear| LIN[GPU: linear-attention / delta-net computation<br/>read/update recurrent and convolution state]
  KIND -->|Full attention| ATT[GPU: Q/K/V, positional transforms<br/>read/update K/V; Flash Attention]
  LIN --> RES1[GPU: attention output projection + residual]
  ATT --> RES1
  RES1 --> FN[GPU: FFN normalization]
  FN --> ROUTE[GPU router<br/>256 expert scores → top 8 IDs and weights]
  FN --> SHARED[GPU shared expert<br/>UP/GATE → SiLU gate → DOWN<br/>shared gating if present]
  FN --> ROUTED[Hybrid routed expert computation<br/>expanded in diagram 4]
  ROUTE --> ROUTED
  ROUTED --> REDUCE[GPU: apply router weights<br/>reduce the 8 expert output rows]
  SHARED --> ADD[GPU: add routed + shared outputs<br/>and FFN residual]
  REDUCE --> ADD
  ADD --> NEXT{More target blocks?}
  NEXT -->|Yes: i + 1| BLOCK
  NEXT -->|No| HEAD[GPU: final normalization + output projection<br/>vocabulary logits]
  HEAD -->|D2H logits; wait for readback| SAMPLE[CPU: penalties, filters, grammar and sampler<br/>choose next token]
  SAMPLE --> EMIT[Emit accepted token; append committed history]
  EMIT -->|Continue generation| INPUT
```

The shared and routed branches are a mathematical dependency diagram. Their independence does not mean the current scheduler overlaps their device kernels. The measured device kernel and copy timelines were serialized. The predictor tail is not part of this 40-block loop; it runs only when MTP is selected.

## 4. Inside one routed MoE layer — current old residency path

```mermaid
flowchart TD
  GX[GPU: normalized activation FP32<br/>2,048 values = 8 KiB] --> READ
  GI[GPU: actual top-8 expert IDs, INT32<br/>8 values = 32 B<br/>routing weights remain on GPU] --> READ
  READ[D2H: activation + IDs to pinned host scratch] --> S1[S1: CPU waits for GPU backend completion<br/>host must not inspect unfinished copies]
  S1 --> PLAN[CPU: lookup actual expert IDs in resident cache<br/>choose GPU hits, selected GPU fills and CPU misses<br/>protect slots used by this invocation]
  PLAN --> CPUIDS[CPU mask: actual CPU experts<br/>GPU-owned rows marked invalid]
  PLAN --> GPUIDS[GPU slot IDs: cache positions<br/>CPU-owned rows marked invalid]
  PLAN --> FILLS[Select cache misses to upload<br/>old policy considers at least one fill when misses exist]

  subgraph CPU[CPU worker — overlaps part of GPU work]
    CPUIDS --> CQ[Convert activation to CPU dot-product packing<br/>e.g. q8-family scratch required by the quantized kernel]
    CW[Original quantized expert weights in pinned RAM] --> CUP
    CQ --> CUP[Selected UP and GATE projections<br/>quantized weight dot products → FP32 values]
    CUP --> CA[SiLU gate × UP<br/>expert intermediate width 512]
    CA --> CDQ[Pack intermediate for CPU DOWN dot product]
    CW --> CD[Selected DOWN projection]
    CDQ --> CD
    CD --> COUT[FP32 expert output rows in host scratch<br/>logical 8 × 2,048 values]
  end

  subgraph GPU[GPU compute stream]
    FILLS --> DMA[H2D required expert UP/GATE/DOWN banks<br/>about 1.8–2 MiB per complete expert<br/>weights remain in their quantized formats]
    DMA --> READY[Stream ordering: required fills must finish<br/>before the expert graph consumes them]
    GPUIDS --> IDDMA[H2D masked slot-ID array]
    IDDMA --> GU
    READY --> GU[Cached + newly filled GPU experts<br/>UP/GATE → SiLU gate → DOWN<br/>activation packing / quantized dot products as required]
    GU --> GOUT[FP32 expert rows in VRAM<br/>CPU-owned rows excluded/cleared]
  end

  COUT --> S2[S2: CPU scheduler joins worker<br/>host result must be complete before upload]
  S2 --> RDMA[H2D CPU output scratch<br/>64 KiB full 8-row buffer in this old single-token path]
  GOUT --> MERGE[GPU masked overwrite<br/>put CPU rows into their original row positions]
  RDMA --> MERGE
  MERGE --> WEIGHT[GPU: routing-weight multiply and reduction<br/>then shared FFN/residual merge]
  WEIGHT --> DONE[Record completion event<br/>preserve buffers until upload/consumer finishes]
  DONE --> S3[S3: later dependent readback / layer boundary<br/>must observe those writes before reuse]
```

Optional paths: with no CPU-owned expert, skip CPU compute/result upload. With no GPU-owned expert, upload the CPU result directly. A valid GPU cache hit does not pass through the weight-upload node. Cache eviction changes which expert occupies a slot; it does not change the original RAM bank. Slot mapping alone is not proof that its bytes have finished uploading.

CPU and GPU do not independently add the same expert contribution twice. Their masks partition expert rows; the GPU first restores the full row order, then applies the original routing weights and reduction.

## 5. Attention cache conversions stay on the GPU

```mermaid
flowchart LR
  H[GPU hidden activation] --> QKV[Attention projections + positional transforms]
  QKV --> Q[Q query]
  QKV --> K[K key]
  QKV --> V[V value]
  Q --> QH[Q Hadamard rotation<br/>when K rotation enabled]
  K --> KH[K Hadamard rotation]
  KH --> KSTORE[Quantize/store K in q4_0 KV cache]
  V --> VH[V Hadamard rotation]
  VH --> VSTORE[Quantize/store V in configured type<br/>q4_0; configured edge-layer overrides q8_0]
  KSTORE --> READK[Read/unpack cached K as required<br/>temporary attention compute representation]
  VSTORE --> READV[Read/unpack cached V as required]
  QH --> FA[Flash Attention]
  READK --> FA
  READV --> FA
  FA --> UNDO[Undo V-space Hadamard rotation]
  UNDO --> OUT[Attention output projection + residual]
```

This is the full-attention branch, not every recurrent block. Quantized KV stays in its stored format; temporary unpacking does not convert the entire persistent context. The actual CUDA attention kernel chooses its working precision/layout. K/V cache Hadamard transforms do not transform routed expert weights. Recurrent state has its own representation; `-ctk/-ctv` do not imply all persistent state is q4.

## 6. Proposed speculative flow — token lookup first, optional MTP fallback

```mermaid
flowchart TD
  HISTORY[Committed CPU token history + current carry token] --> LOOK[CPU n-gram cache lookup<br/>propose likely next token IDs]
  LOOK --> USABLE{Enough useful draft tokens?}
  USABLE -->|Yes| BATCH[Build verification batch<br/>carry + proposed token IDs]
  USABLE -->|No; MTP configured| MTP[Predictor tail uses target hidden state<br/>plus token embedding → draft token IDs<br/>additional compute and memory cost]
  USABLE -->|No; MTP absent or not worthwhile| ONE[Target-only next-token step]
  MTP --> BATCH
  BATCH --> SAVE[Save bounded speculative checkpoint<br/>KV/recurrent state + sampler/grammar position]
  SAVE --> VERIFY[Target verifies through all 40 blocks<br/>initial width 2: carry + 1 draft]
  VERIFY --> ROUTES[Each target layer computes its own real routes<br/>union duplicate experts across verification rows<br/>upload a weight once when safely reusable]
  ROUTES --> LOGITS[Target logits for verification positions]
  LOGITS --> ACCEPT[CPU target sampler / grammar validates tokens<br/>find accepted prefix and next target token]
  ACCEPT --> COMMIT[Commit accepted prefix<br/>restore or discard rejected suffix state<br/>update committed n-gram history only]
  COMMIT --> EMIT[Emit validated tokens]
  ONE --> EMIT
  EMIT --> HISTORY
```

N-gram lookup does not predict real expert IDs, and an MTP tail's expert IDs are not target-layer expert IDs. The target still runs and validates proposals. Speed comes from useful accepted tokens per target verification, amortized weights/dispatch, and eventual transfer overlap—not from bypassing target validation.

The current stable residency path only handles width 1; its batched `new` path is separate. The general target verifier already exists. The simplified plan first measures that existing path; a new width-2 residency executor is conditional on a measured loss of benefit. It does not switch on the discarded experimental engine. An explicit `n_max=1` means one draft, not a width-8 verifier.

## 7. Current and desired transfer scheduling

```mermaid
flowchart LR
  subgraph CURRENT[Current observed device schedule — serialized]
    A[GPU kernels] --> B[Required weight DMA] --> C[GPU expert kernels] --> D[Readback / synchronization] --> E[Next dependent work]
  end
  subgraph PROPOSED[Proposed — only where dependencies permit]
    P[Actual routing known] --> R[Ready cached-expert or independent GPU work]
    P --> U[Upload stream: prepare missing weights]
    U --> EV[Per-upload ready event]
    R --> J[Dependent expert computation / merge]
    EV --> J
  end
```

This proposal requires separate readiness and leases. A required expert kernel cannot read a slot until its upload finishes. If all required experts are consumed by one indivisible kernel, merely moving the copy to another stream does not create overlap: launch a safe ready subset or use other truly independent work. Measure whether the extra partition/launch cost is worthwhile.

## Conversion and transfer ledger

| Item | Location and conversion | Crosses RAM/VRAM? |
| --- | --- | --- |
| Token IDs | CPU INT32 -> GPU IDs for embedding | H2D, small |
| Shared model weights | GGUF quantized/native storage -> persistent GPU buffers at load | Startup H2D, then reused |
| Original routed weights | Native quantized banks retained in pinned RAM | No transfer for CPU expert compute |
| GPU expert fill | Quantized bank slices -> padded VRAM cache slot | H2D on selected miss; dominant traffic |
| Router decisions | GPU top-8 IDs INT32 -> CPU cache planner | D2H 32 B per eligible single-token layer; wait before host read |
| FFN activation | GPU FP32 -> pinned CPU FP32 -> CPU dot-product packing | D2H 8 KiB; conversion is CPU scratch |
| CPU expert intermediate | FP32 UP/GATE -> gated activation -> dot-product packing -> FP32 DOWN | CPU-local |
| GPU expert intermediate | Kernel-specific activation packing, quantized matrix products, FP32 output rows | GPU-local |
| CPU expert outputs | Full FP32 row buffer -> GPU masked merge | H2D 64 KiB when CPU branch runs in old path |
| Full-attention K/V | Rotated working values -> stored q4_0/q8_0; unpacked as needed for attention | GPU-local in this configuration |
| Vocabulary logits | GPU FP32 output -> CPU sampling | D2H, roughly 1 MiB per token in measured trace |
| N-gram proposal | CPU cache -> proposed token IDs | Only the IDs later go to target GPU verification |
| MTP proposal | Predictor compute + target hidden-state handling -> proposed IDs | Placement-dependent; measure actual hidden/draft transfers |

Sources: `src/graphs/build_qwen35.cpp`, `src/llama-build-context.cpp`, `ggml/src/ggml-moe-resident-ffn.inc`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml.c`, `common/speculative.cpp`, the profiled server metadata, and the raw CUPTI capture. Timelines are schematic; measured unions and profiler overhead are recorded in the profile RESULTS.md.
