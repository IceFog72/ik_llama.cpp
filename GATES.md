# Gates: resident expert cache

OWNS: ggml/include/ggml-backend.h, ggml/src/ggml-backend.cpp, src/llama.cpp, src/llama-dflash.cpp, src/llama-build-context.cpp, src/llama-build-context.h, src/llama-model.h, bench_baseline.sh, GATES.md

Scope: add a bounded GPU resident expert LRU to the existing active-expert copy path, preserve mixed hit/miss and graph shape correctness, and verify cache-off equivalence plus cache-on runtime evidence.

- [x] B2-G1: source implements an explicit scheduler resident-cache path with LRU state, host-miss fallback, and device-copy hits
  CHECK: python3 -c "from pathlib import Path; p=Path('ggml/src/ggml-backend.cpp').read_text(); required=['moe_resident_slots','ggml_backend_sched_moe_resident_get_pool','ggml_backend_sched_moe_resident_prepare','ggml_backend_sched_moe_resident_copy','last_used','ggml_backend_tensor_set_async','ggml_backend_tensor_copy_async','GGML_BACKEND_BUFFER_USAGE_COMPUTE']; assert all(x in p for x in required); print('B2 source checks passed')"
  EXPECT: B2 source checks passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B2 source checks passed` from the listed source check.

- [x] B2-G2: work-copy CLI builds after the scheduler and wiring changes
  CHECK: cmake --build /mnt/Kingstone_SSD/build-ft --target llama-cli ft-bench-bw -j12 && echo B2 build passed
  EXPECT: B2 build passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B2 build passed` after building `llama-cli` and `ft-bench-bw`.

- [x] B2-G3: cache disabled preserves the existing short generation output and exits successfully
  # Force the CUDA MoE offload decision for this short decode so the cache path is exercised; G3 and G4 use identical runtime settings.
  CHECK: /mnt/Kingstone_SSD/build-ft/bin/llama-cli -m /mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 'The capital of France is' -n 20 --no-display-prompt --cpu-moe -ngl 99 -c 2048 -ctk q4_0 -ctv q4_0 -fa on --temp 0 -s 42 --no-warmup -t 16 -tb 16 --prefetch-experts --prefetch-experts-threads 4 --cuda-params offload-batch-size=0 --moe-resident 0 2>&1 | tee /tmp/b2-off.log >/dev/null && grep -q 'Paris' /tmp/b2-off.log && echo B2 cache-off passed
  EXPECT: B2 cache-off passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B2 cache-off passed`; `/tmp/b2-off.log` contained `Paris.` and exited 0.

- [x] B2-G4: cache enabled allocates resident slots, records mixed hits/misses, and completes the same short generation
  CHECK: /mnt/Kingstone_SSD/build-ft/bin/llama-cli -m /mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 'The capital of France is' -n 20 --no-display-prompt --cpu-moe -ngl 99 -c 2048 -ctk q4_0 -ctv q4_0 -fa on --temp 0 -s 42 --no-warmup -t 16 -tb 16 --prefetch-experts --prefetch-experts-threads 4 --cuda-params offload-batch-size=0 --moe-resident 64 2>&1 | tee /tmp/b2-on.log >/dev/null && grep -q 'Paris' /tmp/b2-on.log && grep -Eq 'resident.*hits=[1-9].*miss(es)?=[1-9]' /tmp/b2-on.log && echo B2 cache-on passed
  EXPECT: B2 cache-on passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B2 cache-on passed`; `/tmp/b2-on.log` contained `Paris.` and resident stats including `hits=205 misses=315`.

- [x] B2-G5: final diff has no whitespace errors and no accidental generated artifacts
  CHECK: git diff --check && test -z "$(git status --short --untracked-files=all | awk '$1 == \"??\" && $2 != \"GATES.md\" { print }')" && echo B2 diff passed
  EXPECT: B2 diff passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B2 diff passed` from `git diff --check` plus the unexpected-untracked-file check.

## Slice B3: adaptive per-layer resident cache

OWNS: common/common.cpp, common/common.h, ggml/include/ggml-backend.h, ggml/include/ggml-cuda.h, ggml/src/ggml-backend.cpp, ggml/src/ggml-cuda.cu, include/llama.h, src/llama.cpp, src/llama-build-context.cpp, src/llama-build-context.h, src/llama-cparams.h, src/llama-dflash.cpp, src/llama-model.h, GATES.md

Scope: make `--moe-resident N|auto` sufficient to enable active-MoE CUDA placement, partition each bounded bank-pool slot budget across model layers, and keep an LRU within each layer. `--moe-resident 0` remains the behavior-preserving default.

Non-goals: full FreeToken CPU/hybrid expert execution or changes to the stock tree.

- [x] B3-G1: source has explicit per-layer resident ownership and flag-driven active-MoE offload
  CHECK: python3 -c "from pathlib import Path; p=Path('ggml/src/ggml-backend.cpp').read_text(); required=['moe_resident_layers','layer_banks','moe_resident_get_layer','force_moe_offload','layer_slots','bank_role','auto_slots']; assert all(x in p for x in required); print('B3 source checks passed')"
  EXPECT: B3 source checks passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B3 source checks passed`; the obsolete graph-resident no-op hook was removed because the scheduler input-copy path is the live implementation.

- [x] B3-G2: work-copy builds after the per-layer cache changes
  CHECK: cmake --build /mnt/Kingstone_SSD/build-ft --target llama-server llama-cli ft-bench-bw -j12 && echo B3 build passed
  EXPECT: B3 build passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B3 build passed`; `llama-server`, `llama-cli`, and `ft-bench-bw` linked successfully after correcting the model pointer access and removing the stale graph hook.

- [x] B3-G3: flag disabled preserves the short deterministic completion
  # Keep cache-off on the same forced CUDA path for an apples-to-apples output comparison; B3-G4 proves the resident flag removes this forcing requirement.
  CHECK: /mnt/Kingstone_SSD/build-ft/bin/llama-cli -m /mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 'The capital of France is' -n 20 --no-display-prompt --cpu-moe -ngl 99 -c 2048 -ctk q4_0 -ctv q4_0 -fa on --temp 0 -s 42 --no-warmup -t 16 -tb 16 --prefetch-experts --prefetch-experts-threads 4 --cuda-params offload-batch-size=0 --moe-resident 0 2>&1 | tee /tmp/b3-off.log >/dev/null && grep -q 'Paris' /tmp/b3-off.log && echo B3 cache-off passed
  EXPECT: B3 cache-off passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B3 cache-off passed`; the command exited 0 and generated `Paris.`.

- [x] B3-G4: flag enabled forces the CUDA active-MoE path without `--cuda-params`
  CHECK: /mnt/Kingstone_SSD/build-ft/bin/llama-cli -m /mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 'The capital of France is' -n 20 --no-display-prompt --cpu-moe -ngl 99 -c 2048 -ctk q4_0 -ctv q4_0 -fa on --temp 0 -s 42 --no-warmup -t 16 -tb 16 --prefetch-experts --prefetch-experts-threads 4 --moe-resident auto 2>&1 | tee /tmp/b3-on.log >/dev/null && grep -q 'Paris' /tmp/b3-on.log && grep -Eq 'moe-resident.*layers=[1-9].*hits=[1-9].*miss(es)?=[1-9]' /tmp/b3-on.log && echo B3 cache-on passed
  EXPECT: B3 cache-on passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B3 cache-on passed`; `/tmp/b3-on.log` showed separate bank-role pools with `layers=41`, nonzero hits/misses, and `disabled=no` without `--cuda-params`.

- [x] B3-G5: cache-on and cache-off produce identical deterministic short output
  CHECK: python3 -c "from pathlib import Path; import re; a=Path('/tmp/b3-off.log').read_text(); b=Path('/tmp/b3-on.log').read_text(); pick=lambda s: re.search(r'(?m)^ Paris\\..*?(?=^main: prompt eval time)', s, re.S).group(0); assert pick(a)==pick(b); print('B3 output equivalence passed')"
  EXPECT: B3 output equivalence passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B3 output equivalence passed` for `/tmp/b3-off.log` and `/tmp/b3-on.log`.

- [x] B3-G6: B3 diff has no whitespace errors or unexpected untracked files
  CHECK: git diff --check && test -z "$(git status --short --untracked-files=all | awk '$1 == \"??\" && $2 != \"GATES.md\" { print }')" && echo B3 diff passed
  EXPECT: B3 diff passed
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `B3 diff passed`; `git diff --check` was clean and no unexpected untracked files were present.

- [x] B3-G7: user-shaped 128K context run completes with automatic sizing and deterministic output
  CHECK: FT `llama-cli` with `-cmoe -ngl 999 -c 131072 -b 3072 -ub 1536`, fixed seed/greedy sampling, and `--moe-resident auto`; compare against the same forced-CUDA `--moe-resident 0` command.
  EXPECT: both runs complete, output matches, and auto reports enabled pools without allocation failure.
  EVIDENCE: `/tmp/b3-user-forced-off.log` and `/tmp/b3-user-auto.log` matched; auto reported four enabled pools (`slots=220/198/145/108`, `layers=41`, `disabled=no`) after separating up/gate/down bank roles. At 64 generated tokens, forced cache-off reached `9.28 tok/s` and auto reached `8.16 tok/s`; auto recorded low hit rates (`235/21011`, `90/21156`, `5/19665`, `0/1576`) and was not a production speedup.

- [x] B3-G8: cache-off perplexity remains unchanged
  CHECK: `/mnt/Kingstone_SSD/build-ft/bin/llama-perplexity -m /mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f /mnt/Kingstone_SSD/b_eval/corpus_big.txt --cpu-moe -ngl 99 -c 2048 --moe-resident 0`
  EXPECT: baseline PPL remains `3.1931 +/- 0.07007`
  EVIDENCE: `/tmp/b3-ppl-big-final.log`: `Final estimate: PPL over 10 chunks for n_ctx=2048 = 3.1931 +/- 0.07007`.

- [x] B3-G9: explicit fixed positive slot count remains functional
  CHECK: repeat B3-G4 with `--moe-resident 64`, require `mode=fixed`, nonzero hit/miss counters, and compare the deterministic completion with cache-off.
  EXPECT: fixed-slot run exits successfully and preserves the completion.
  EVIDENCE: `B3 fixed-slot correctness and output equivalence passed`; four fixed pools reported `slots=64`, `layers=41`, `disabled=no`, and nonzero misses.

- [x] B3-G10: resident mode forces active-expert placement even when general active-only offload is disabled
  CHECK: repeat B3-G4 with `--no-offload-only-active-experts`, require enabled resident pools and compare the deterministic completion.
  EXPECT: `--moe-resident auto` still uses the active-MoE CUDA copy path and preserves output.
  EVIDENCE: `/tmp/b3-auto-no-ooae.log` matched `/tmp/b3-on.log`; four auto pools reported nonzero hits/misses and `disabled=no`.

## B3 repair: route state, identity, and bounded policy

Scope: repair the resident staging cache before any new speed claim. Preserve `--moe-resident 0`, keep `/mnt/LLM/Text/ik_llama.cpp` untouched, and defer true CPU/GPU hybrid execution to B4.

- [x] B3R-G1: baseline and repro are recorded
  CHECK: git status --short --branch && git log -1 --oneline && test -f /tmp/diag-0.stdout && test -f /tmp/diag-auto.stdout && echo B3R baseline recorded
  EXPECT: B3R baseline recorded
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `f002449e slice B3: add adaptive resident expert cache`; `/tmp/diag-0.stdout` and `/tmp/diag-auto.stdout` are present. Repair work remains uncommitted on `ft-slice-a-profiler`.

- [x] B3R-G2: route state persists across scheduler splits
  CHECK: source/test gate must prove the same route tensor is read and synchronized once per scheduler pass when shared by up/gate/down nodes.
  EXPECT: no duplicate route read or synchronize for the shared route tensor.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: route state is passed by reference through `ggml_backend_sched_copy_inputs()`. `/tmp/b3r-auto-window.log` reports `routes reads=1280 syncs=1280 reuses=2560`.

- [ ] B3R-G3: route IDs are validated before bitset indexing
  CHECK: source/test gate must cover negative, sentinel, and `id >= n_expert` values without OOB access; ordered IDs and bitset must agree.
  EXPECT: invalid IDs are rejected/ignored safely and valid representations match.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: real C++ CUDA-copy tests reject ordered IDs `[-1, 8]` for `n_expert=8` before lookups/copies. This is not a test of route-tensor ingestion or its bitset construction; that seam still needs invalid/sentinel/representation tests. Earlier Python/string checks are supporting evidence, not completion.

- [x] B3R-G4: resident keys use real layer identity and explicit bank roles
  CHECK: source/test gate must cover `blk.<N>` parsing, unknown names, and role separation without encounter-order assignment.
  EXPECT: layer N cannot be accounted to layer M; unknown identity bypasses residency safely.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `tests/test-moe-resident.cpp` compiles the actual scheduler implementation and exercises valid, negative, out-of-range, overflowing, and unknown layer identities, down-role recognition, and unknown-role rejection. All bank roles are separate in the implementation; full-expert coordinated residency is not implemented.

- [x] B3R-G5: auto uses one backend-wide VRAM budget
  CHECK: source/test gate must show total resident allocation stays within one backend budget as pools appear.
  EXPECT: pool creation cannot multiply the auto budget; allocation failure falls back safely.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: real CUDA tests allocate two bank pools against one byte budget, reject a third pool when exhausted, and verify budget accounting returns to zero on free. An injected buffer allocator failure sets auto unavailable without consuming bytes; new-graph residency requests then return false. The original reused-graph re-split claim is superseded by the B4 fallback repair below. That path corrupts scheduler copy references. The repair requires a fresh caller graph and tests both output values and CPU placement after rebuilding.

- [ ] B3R-G6: underprovisioned layers avoid guaranteed thrash
  CHECK: compiled history checks for stable/rotating visits and expiry; real CUDA checks for hot/cold mixed routes, active-victim protection, and capacity-2/working-set-3 bypass; viable per-layer quota checks.
  EXPECT: bypass or selective admission prevents endless zero-value admit/evict traffic.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: the actual C++ implementation failed 42 history assertions and three CUDA ordering/eviction assertions before repair; both now pass. History ages per layer, records one visit regardless of batch frequency, saturates at two, and clears safely on rollover/geometry changes. CUDA mixed copies preserve active expert/padding bytes, reject cold eviction, protect later active hits, and bypass oversized routes before lookups. Quotas leave uncached layers at zero rather than undersizing every layer. These are correctness checks, not a performance result. Gate remains open: the model's 210-MiB DOWN banks occur only at layers 34/38/39, but their geometry pool assigns its quota to layers 0–11; the 64-token auto run allocates 80.44 MiB to that pool with zero lookups. Eligible-layer planning must be geometry-aware.

- [x] B3R-G7: matched CUDA-path deterministic equivalence
  CHECK: cache-off and cache-on use identical CUDA active-MoE placement, fixed seed, and greedy sampling.
  EXPECT: deterministic output matches.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: current cache-off CUDA active-expert run (`--cuda-params offload-batch-size=0 --moe-resident 0`) and resident-auto run used identical seed/sampler/prompt and matched the complete 32-token generated block byte-for-byte. `/tmp/b3r-cuda-cacheoff.log` and `/tmp/b3r-auto-window.log`.

- [x] B3R-G8: cache-off PPL and behavior remain unchanged
  CHECK: build, cache-off short generation, and established PPL baseline.
  EXPECT: PPL remains `3.1931 +/- 0.07007`; resident disabled allocates no cache.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: full target build passed; cache-off short generation exited 0; `/tmp/b3r-ppl.log` reports `Final estimate: PPL over 10 chunks for n_ctx=2048 = 3.1931 +/- 0.07007`. Cache-off log has no resident pool summary.

- [ ] B3R-G9: server/API regression and canonical 64K A/B
  CHECK: replay the exact frontend request with cache off and auto, record utilization, transfers, latency, and resident counters; request JSON is currently unavailable locally.
  EXPECT: gate remains pending until the request is supplied and both modes are repeated at least five times.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: blocked: exact frontend request JSON not found locally.

- [ ] B3R-G10: executable scheduler-policy and CUDA-copy regression checks
  CHECK: build `test-moe-resident`; run `ctest -R '^test-moe-resident$'`, `test-moe-resident --cuda`, host ASan/UBSan, and CUDA UBSan.
  EXPECT: zero failures; no sanitizer errors in the instrumented scheduler implementation.
  CWD: /mnt/LLM/Text/ik_llama_ft
  EVIDENCE: `ctest -R '^test-moe-resident$'` passes; host test binary and `--cuda` variant both report 0 failures. Standalone host ASan+UBSan (`-std=c++20 -fsanitize=address,undefined`, `/tmp/tmr-asan`) passes with 0 failures, covering the new replan flag assertions. CUDA initialization failed under ASan despite free VRAM; GPU ASan coverage is not claimed. Other ggml/CUDA shared-library internals are not sanitizer-instrumented by the standalone test command. `llama-perplexity` rebuilt (10:15) for the PPL re-check.

### Paper-alignment review (B3R policy repair)

Reference: [FreeToken, §§3.2, 3.3, 4.1–4.2](https://arxiv.org/pdf/2608.16157).

Confirmed defects repaired: admission expiry used global layer-read epochs; intra-batch frequency falsely established temporal reuse; cache traversal discarded route order; a miss could evict a later active hit. Regression tests compile the actual `ggml-backend.cpp`, and the optional CUDA test executes real H2D/D2D copies without a model. It is a non-Windows white-box test, not a second policy implementation or a new public API.

Remaining implementation/acceptance gaps:
- The same-graph re-split claim was incorrect. Runtime demotion now requests a fresh caller graph. FBR-G1–G4 below prove output preservation, sticky rebuild signaling, fresh CPU placement, and server allocator-failure fallback.
- Bank pools are independent, not complete-expert slots sharing one logical expert-to-slot mapping. Full-expert readiness and coordinated role admission remain open.
- Quotas target the first N model layers, even when that pool's geometry only exists later. `/tmp/b3r-policy-auto.log` places the 210-MiB DOWN banks at layers 34/38/39; the corresponding 98-slot pool reserves layers 0–11, allocates 80.44 MiB, and records zero lookups. Root cause: `prepare()` distributes across all model layers before `get_layer()` rejects layers with zero quota. Registering geometry-specific eligible layers before quota allocation is the preferred fix; do not relabel actual layer identities by encounter order.
- Invalid route-tensor ingestion, multi-GPU affinity, and pipeline-copy lifetime behavior still need direct coverage. Single-device copy tests do not prove them.
- Server/API replay, final cache-off PPL, and canonical warmed 64K/128K repeated medians are required before release acceptance. Short deterministic generation is not evidence of a production speedup.

Deliberate B3 scope differences, not fixes delivered here: the paper runs cache decisions on device and executes resident hits directly, chooses hybrid CPU/GPU miss execution with its bandwidth policy, and rebuilds runtime residency at scheduler safe points. B3 remains a host-controlled staging cache with extra D2D copies and routing synchronization. These architectural differences preclude calling this FreeToken-equivalent or claiming its performance.

## B4 fallback repair: fresh graph after residency demotion

Problem: re-splitting an allocated graph after residency demotion reuses source pointers into the scheduler's reset copy context. A three-bank CUDA repro changes the first output from 16 to 48 while reporting success. The original B3R-G5 re-split claim above is superseded.

Behavior: the scheduler reports a pending caller graph rebuild. Main llama and DFlash reuse gates honor it. Direct scheduler computes preserve the old graph and placement until the caller supplies a fresh graph. Reset preserves the signal; successful fresh allocation clears it.

- [x] FBR-G1: regression catches the original corruption and preserves expert outputs after repair.
  CHECK: compile `tests/test-moe-resident.cpp` against the actual scheduler and run `--cuda`.
  EVIDENCE: native CUDA regression passes with zero failures. Independent Luna verification restores only the old `graph_compute_async` guard in a temporary source copy and observes eight failures, including corrupted output and lost rebuild signal. Logs: `/mnt/Kingstone_SSD/moe-auto-review/luna-green.log` and `luna-oldguard-red.log`.

- [x] FBR-G2: runtime demotion and resident allocator failure preserve old graph results, then use CPU on a fresh graph.
  CHECK: `test_cuda_rebuild` exercises explicit demotion and an injected backend resident allocation failure, all three bank outputs, old graph reuse, sticky signal through reset, fresh CPU placement, and fresh CPU graph reuse.
  EVIDENCE: native CUDA and CUDA UBSan runs pass. Host ASan/UBSan passes. CUDA ASan cannot initialize the driver (`out of memory`) before entering the regression. Logs: `/mnt/Kingstone_SSD/moe-auto-review/regression.log`, `ubsan-cuda.log`, `asan-host.log`, and `asan.log`.

- [x] FBR-G3: normal release build and focused regression gates pass.
  CHECK: `cmake --build /mnt/Kingstone_SSD/build-ft --target test-moe-resident llama-server -j12`; release host/CUDA tests; `ctest -R '^test-moe-resident$' --output-on-failure`.
  EVIDENCE: build exits 0; host and CUDA report zero failures; CTest passes. Logs: `/mnt/Kingstone_SSD/moe-auto-review/build.log`, `release-host.log`, `final-release-cuda.log`, and `final-ctest.log`.

- [x] FBR-G4: exact user server defaults survive a resident allocation failure.
  CHECK: `/mnt/Kingstone_SSD/moe-auto-review/server-smoke.py` uses Qwen3.6 UD Q4_K_M, `-cmoe -b 3072 -ub 1536 --ctx-size 65536`, all user cache/sampler flags, and two 128-token completion requests per OFF/AUTO/injected-failure AUTO mode. Request overrides: temperature 0, seed 42, cache_prompt false.
  EVIDENCE: all six requests succeed. A test-only LD_PRELOAD hook rejects one CUDA resident-shaped allocation of 75,563,008 bytes (128 UP slots). The log confirms `moe-resident auto disabled: device allocation failed`. Both failure-mode completions match OFF byte-for-byte. Normal AUTO completes both requests. These are short-prompt functional checks under a 64K context configuration, not a long-context benchmark. Logs, JSON responses, and summary are under `/mnt/Kingstone_SSD/moe-auto-review/`.


## B5 per-expert CPU/CUDA decode

Changed behavior: for supported single-token decode with one CUDA backend and CPU, auto selects complete-bank cache hits plus bandwidth-selected cache fills for CUDA. Other routed experts execute in place on CPU. Negative ID masks preserve output row positions, including repeated IDs. CUDA launches before the native CPU kernel; only CPU output rows upload before the existing graph tail runs. The partition is retained across projections in one layer. All-GPU routes use the original graph directly. Prefill, pipeline copies, multiple accelerator backends, biased fused operators, and unsupported layouts retain their previous execution path.

The scheduler owns reusable pinned activation/output/ID scratch and CUDA ID scratch. A successful preparation marker gates mixed execution. Calibration measures the deployed quantized CPU kernel and actual expert transfers once per operator/type/geometry, then rounds the miss quota using the measured PCIe/host ratio, with at least one cache fill. Warmup's all-expert route is excluded. This is host control, not FreeToken's device control plane.

Confirmed supporting defects repaired: CUDA MMQ counted negative IDs as lower-index experts and left quantization row mappings uninitialized. Its compaction now handles route references directly, including duplicate IDs, without token-sized shared scratch. CPU ordinary/fused MoE row maps now reserve room for repeated expert IDs. Combined fused-bank work planning sizes activation quantization from the activation type. Auto quotas honor planned capacities above 32 slots per layer. Empty scans no longer seal a zero-layer plan. Scheduler construction/destruction now uses C++ lifetime rules; host ASan exposed the prior vector allocation leaks.

- [x] B5-G1: native mixed arithmetic and downstream merging.
  CHECK: release `test-moe-resident --cuda`, ordinary and fused Q4_0, combined gate/up bank, duplicate IDs, top-k 8/16/24, repeated cache use, and a fused graph with a downstream SQR.
  EVIDENCE: `split-final-cuda.log` reports zero failures. Every mixed row matches its full native CPU or CUDA reference at `1e-5 + 1e-5*abs(reference)`. CPU repeated/masked F32 routes also match an independent dot-product oracle. CPU/CUDA activation quantization differs, so a guessed cross-backend tolerance is not the merge oracle. Test ratios are injected at 0.2 to deterministically exercise both branches; model logs separately prove runtime calibration and mixed execution.

- [x] B5-G2: failure contracts.
  CHECK: partial decode scratch allocation failure, cache allocation failure during mixed compute, fresh CPU graphs after demotion, and prior fallback regression.
  EVIDENCE: `split-final-cuda.log` passes. Failed device-ID scratch preparation retains ordinary CUDA compute with no prepared marker. A resident allocation failure still computes selected GPU rows via direct H2D and other rows on CPU, then requests a fresh caller graph. Final model failure replay passes twice and matches OFF byte-for-byte: `split-final-failure.log`, `split-final-failure-summary.json`, and `auto-fail-server.log`.

- [x] B5-G3: release and memory verification.
  CHECK: build `llama-server` and `test-moe-resident`; host/CUDA tests; CTest; host ASan/UBSan; CUDA UBSan; CUDA compute-sanitizer memcheck; independent Luna execution/review.
  EVIDENCE: final release CUDA and memcheck logs report zero failures and zero CUDA memory errors. Host ASan/UBSan passes with leak checking after the scheduler lifetime fix. Standalone sanitizers instrument the test/scheduler translation unit, not every shared ggml/CUDA-library instruction. Detailed final records: `/mnt/Kingstone_SSD/moe-auto-review/`.

- [x] B5-G4: user defaults and warmed alternating measurement.
  CHECK: exact user Qwen server flags, append `--moe-resident 0|auto`; OFF/AUTO/OFF/AUTO, one warmup plus five measured 128-token requests per server.
  EVIDENCE: all 24 requests succeed. Pair medians: 26.007/26.723 tokens/s (+2.8%), then 30.919/30.855 (-0.2%). Pooled medians: 28.171/29.217 (+3.7%). The baseline shift prevents a reliable speedup claim. AUTO logs report 13,716 mixed operators per block, 49,622–52,218 CPU references and 57,510–60,106 GPU references, with 47.15–49.62 GB of expert-weight H2D avoided by CPU assignment. These counters cover the selected CUDA layers and exclude calibration transfers. Context is configured at 65,536 but reaches 139 tokens in these short requests. `split-bench.py`, `split-bench-summary.json`, and `split-{off,auto}{1,2}-server.log` preserve commands and evidence.

Retained limits: the initial auto budget is still a fraction of free VRAM and the placement plan still chooses a layer prefix (9/41 for this model/run). Cache hits require all planned roles, and decode admissions use one shared route partition, but storage remains independent bank pools. CUDA hits still copy D2D into full-bank staging tensors. Multi-GPU/pipeline mixed execution, device routing, a global complete-expert cache, direct resident-bank execution, long-context behavior, and reliable production speedup are not proved here. Native test launches disable core dumps locally. The initial failed prototype produced an 18.5-GB system core dump, which the user removed; the directory is now empty.

## B6 complete-expert decode implementation

Acceptance: supported single-GPU fused single-token MoE decode reads compact resident slots directly, shares complete experts across layers, executes the CPU UP/GATE and DOWN together, returns CPU expert outputs in one upload, uses stable task descriptors and a persistent worker, and chooses fills from measured concurrent FFN latency. Model host banks remain the source of truth. Routing row order, duplicate references, existing weighting/reduction, fixed/off modes, unsupported graphs, and the fresh-graph fallback contract remain correct.

- [x] B6-G1: direct-slot execution, global replacement, complete-bank identity, no resident-hit D2D weight copies.
  EVIDENCE: scheduler-private complete slots share one `(layer, expert)` mapping across all banks. CUDA regression forces cross-layer eviction with an eight-slot cache and replaces a live expert's DOWN bank with different quantized data. Fresh native references agree. Model logs report `d2d_weights=0` for the complete FFN path. Legacy warmup/fallback copies are counted separately.
- [x] B6-G2: full CPU FFN and concurrent CUDA path match their native branch references, including masks, duplicates, combined banks and downstream weighting.
  EVIDENCE: separate and combined Q4_0 banks plus Q4_K UP/GATE with heterogeneous Q5_K/Q6_K DOWN banks, repeated/masked IDs, both execution branches, downstream weighting/SQR, and reused scheduler graphs match the branch-specific native oracle at `1e-5 + 1e-5*abs(reference)`. CPU and CUDA use different native activation quantization, so mixed generation need not be byte-identical to all-CPU generation.
- [x] B6-G3: allocation failure, geometry/graph changes, scratch reuse, reset/free, and active-slot protection remain safe.
  EVIDENCE: CUDA regression injects complete-cache allocation failure after graph allocation, preserves current graph outputs, requests a rebuild, and validates subsequent CPU graphs. Growing one layer's intermediate dimension on the same scheduler releases old storage before compute, allocates larger shared strides, and still matches fresh native references. Native CUDA and compute-sanitizer pass through reset/free. Callback, split-mode and unsupported-operator exclusions are source-gated; those exclusions and multi-GPU behavior have no new direct runtime coverage.
- [x] B6-G4: live post-reservation memory budget and real concurrent latency calibration are observable.
  EVIDENCE: allocation logs report free memory after runtime reservation, remaining budget, retained headroom, slot count and allocated bytes. Complete storage releases legacy caches before sizing and is reclaimed before runtime reservation growth. Calibration logs measure CPU worker + weight DMA + CUDA projections + result merge together, by quantized types/geometry/operator parameters and miss count. Each candidate uses three observations and an EWMA, followed by periodic probes. At least one miss is admitted when there are misses; this policy is not a proof of a globally optimal split.
- [x] B6-G5: release build, host/CUDA regression, CUDA memcheck, user-command short/long-context functional and paired performance checks. Luna runs read-only verification only after implementation is complete. Native test processes suppress core dumps locally and verify the limit.

  EVIDENCE: release server build3 and final test build5 succeed. Luna ran the completed tests read-only: host/CUDA zero failures, CTest 1/1, CUDA compute-sanitizer zero errors (`b6/luna-done-{host,cuda,ctest,memcheck}.log`). Final changed-route reuse, source-bank replacement, shared-stride geometry growth and mixed quantization regressions pass. Host ASan/UBSan with leak checking and CUDA UBSan pass. CUDA ASan/UBSan also passes with `ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=1:halt_on_error=1`; default shadow-gap protection fails CUDA initialization before the regression. Standalone sanitizer instrumentation covers the test/scheduler translation unit, not every shared C/CUDA-library instruction. CPU-only scheduler syntax and `git diff --check` pass.

  PERFORMANCE: the accepted model benchmark uses the exact user server flags, appending OFF/AUTO and port 8091. Requests override temperature=0, seed=42, cache_prompt=false. OFF/AUTO/AUTO/OFF blocks each use one warmup and five measured completions, every prompt processes 9,123 tokens and every completion generates 128 tokens. Decode medians: OFF1 28.288 / AUTO1 35.041 tokens/s (+23.87%), AUTO2 34.390 / OFF2 27.769 (+23.85%). Median total request time falls 5.40% and 4.99%. Median prefill throughput remains 662–669 tokens/s. All 24 requests succeed. The prompt consists of repeated reference records followed by a TCP/UDP question; this proves a repeatable gain for that workload, not all workloads or 64K populated-context performance. AUTO output text differs from OFF, so this is not a quality-parity or perplexity result. Earlier short-prompt medians are superseded by this minimum-8K run. Commands, responses, limits, server logs and metrics: `/mnt/Kingstone_SSD/moe-auto-review/b6/long/`; driver: `b6/long-bench.py`.

- [x] B6-G6: minimum-8K model replay after an injected complete-cache allocation failure.
  EVIDENCE: a test-only allocator callback rejects the startup complete cache allocation of 1,102,325,760 bytes after runtime buffers are reserved. The server logs `CPU fallback`, remains healthy, and completes two 9,123-token prompt / 128-token generation requests. Both completions match the OFF reference byte-for-byte. `b6/long-failure.log`, `b6/auto-fail-server.log` and `b6/split-final-failure-summary.json` preserve evidence. Initial hooks were armed after health or targeted the wrong size and therefore did not inject failure; those attempts are not acceptance evidence. The final hook targets the startup allocator callback and leaves the larger runtime buffer allocation intact. Test server processes are stopped, core limits were zero, and `/var/lib/systemd/coredump/` remains empty.

Retained scope: single CUDA backend plus CPU, one pipeline copy, supported fused single-token UP/GATE + DOWN. Unsupported graphs retain prior paths. Control decisions still run on the host, with a combined routing/activation device wait per FFN. This is not an implementation of the paper's device control plane. No perplexity/quality parity, multi-GPU mixed execution, or populated 64K context claim is made.
