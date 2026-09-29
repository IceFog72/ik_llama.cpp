#!/bin/bash
# Baseline suite: stock ik_llama.cpp vs FT fork. Same model, same prompts.
# Run BEFORE building FT changes (baseline numbers) and AFTER (comparison).
# Usage: ./bench_baseline.sh [outdir]  (default: /mnt/Kingstone_SSD/b_eval/baseline)
set -u
OUT="${1:-/mnt/Kingstone_SSD/b_eval/baseline}"
mkdir -p "$OUT"
STOCK_BIN="${STOCK_BIN:-/mnt/Kingstone_SSD/build-ik-llama/bin}"
M=/mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
export GGML_CUDA_NO_PINNED=1

C="--no-display-prompt --cpu-moe -ngl 99 -c 8192 -ctk q4_0 -ctv q4_0 -fa on --temp 0 -s 42 --no-warmup -t 16 -tb 16 --prefetch-experts --prefetch-experts-threads 4"

P_SHORT="The capital of France is"
P_TRAIN="Solve step by step: A train travels 120 km at 80 km/h, then returns the same 120 km at 60 km/h. What is the average speed for the whole trip? Show your work."
P_TUESDAY="Mr. Smith has two children. At least one of them is a boy born on a Tuesday. What is the probability that both children are boys? State assumptions, show the full sample-space computation. Give the exact fraction."

gen() { # name prompt ntokens
  local name="$1" prompt="$2" n="$3"
  echo "### START gen/$name $(date +%H:%M:%S)"
  "$STOCK_BIN/llama-cli" -m "$M" -p "$prompt" -n "$n" $C > "$OUT/gen_$name.log" 2>&1
  echo "### END gen/$name exit=$? $(date +%H:%M:%S)"
  grep -E "prompt eval time|eval time" "$OUT/gen_$name.log" | tail -2
}
ppl() { # name corpus
  local name="$1" corpus="$2"
  echo "### START ppl/$name $(date +%H:%M:%S)"
  "$STOCK_BIN/llama-perplexity" -m "$M" -f "$corpus" --cpu-moe -ngl 99 -c 2048 > "$OUT/ppl_$name.log" 2>&1
  echo "### END ppl/$name exit=$? $(date +%H:%M:%S)"
  grep -E "Final estimate" "$OUT/ppl_$name.log"
}
bench() { # pp tg
  local pp="$1" tg="$2"
  echo "### START bench/pp${pp}_tg${tg} $(date +%H:%M:%S)"
  "$STOCK_BIN/llama-bench" -m "$M" -p "$pp" -n "$tg" $C -o json > "$OUT/bench_pp${pp}_tg${tg}.json" 2>"$OUT/bench_pp${pp}_tg${tg}.err"
  echo "### END bench/pp${pp}_tg${tg} exit=$? $(date +%H:%M:%S)"
}

gen short "$P_SHORT" 20
gen train "$P_TRAIN" 400
gen tuesday "$P_TUESDAY" 2000
ppl big /mnt/Kingstone_SSD/b_eval/corpus_big.txt
bench 512 128
bench 128 128
echo ALLDONE $(date +%H:%M:%S)
