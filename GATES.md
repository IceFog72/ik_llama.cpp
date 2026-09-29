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
