# Gates: Slice B2 resident expert cache

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
