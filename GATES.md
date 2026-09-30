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
  EVIDENCE: real CUDA tests allocate two bank pools against one byte budget, reject a third pool when exhausted, and verify budget accounting returns to zero on free. An injected buffer allocator failure sets auto unavailable without consuming bytes; new-graph residency requests then return false. The replan half is proven at two levels: the unit test shows a runtime layer demotion on an allocated graph sets `moe_resident_replan_required` (and an idle scheduler does not), and `graph_compute_async()` honors that flag by synchronizing, resetting, and re-splitting the same reused graph at the next safe compute boundary before `is_alloc` is set again.

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
- Auto becoming unavailable is covered two ways: an injected device-allocation failure disables auto and new-graph requests return false, and a runtime layer demotion on an allocated graph sets `moe_resident_replan_required` so `graph_compute_async()` re-splits the same reused graph at the next safe compute boundary (unit test proves the flag state machine; the reset/re-split path itself is stock scheduler code).
- Bank pools are independent, not complete-expert slots sharing one logical expert-to-slot mapping. Full-expert readiness and coordinated role admission remain open.
- Quotas target the first N model layers, even when that pool's geometry only exists later. `/tmp/b3r-policy-auto.log` places the 210-MiB DOWN banks at layers 34/38/39; the corresponding 98-slot pool reserves layers 0–11, allocates 80.44 MiB, and records zero lookups. Root cause: `prepare()` distributes across all model layers before `get_layer()` rejects layers with zero quota. Registering geometry-specific eligible layers before quota allocation is the preferred fix; do not relabel actual layer identities by encounter order.
- Invalid route-tensor ingestion, multi-GPU affinity, and pipeline-copy lifetime behavior still need direct coverage. Single-device copy tests do not prove them.
- Server/API replay, final cache-off PPL, and canonical warmed 64K/128K repeated medians are required before release acceptance. Short deterministic generation is not evidence of a production speedup.

Deliberate B3 scope differences, not fixes delivered here: the paper runs cache decisions on device and executes resident hits directly, chooses hybrid CPU/GPU miss execution with its bandwidth policy, and rebuilds runtime residency at scheduler safe points. B3 remains a host-controlled staging cache with extra D2D copies and routing synchronization. These architectural differences preclude calling this FreeToken-equivalent or claiming its performance.
