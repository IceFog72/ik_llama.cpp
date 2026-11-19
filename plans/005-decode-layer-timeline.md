# Plan 005: Measure where each resident decode layer spends its time, then choose the fix

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in "STOP conditions" occurs, stop and report. Do not
> improvise. When done, update the status row for this plan in
> `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 21c8c246..HEAD -- ggml/src/ggml-backend.cpp ggml/src/ggml-moe-resident-new.inc ggml/src/ggml-cuda/moe-resident.h`
> If any of these files changed, compare the "Current state" excerpts with the
> live code before you continue. Treat any mismatch as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M (about 120 changed lines, then 5 model runs of about 2 minutes each)
- **Risk**: LOW. The new code is diagnostic only, and none of it runs unless an environment variable is set.
- **Depends on**: none
- **Category**: perf
- **Planned at**: commit `21c8c246`, 2026-11-17

## Why this matters

Decode on the reference machine (RTX 2060 6 GiB, Ryzen 7 2700X, Qwen3.6-35B-A3B
Q4_K_M, 28,000-token prompt, MTP with one draft token) runs at **42.79 t/s**.
That is 146 two-token verification passes of about **41.0 ms** each, or roughly
**1,025 us per MoE layer** across 40 layers. The existing counters split each layer
into the following parts (`.unlazy/improve/model-long.log`, final `reason=shutdown layers=5840` line):

| Part of a layer cycle | us/layer | Share | What it is |
|---|---:|---:|---|
| `cpu_wait` (CPU expert job) | 372.9 | 36% | 5.62 unique experts × 1.91 MB = 10.7 MB read per layer, about **28.8 GB/s** |
| `router_wait` | 265.9 | 26% | Host blocked until the GPU has finished attention and the router |
| Other host time in `ffn_host` (885.8 − 372.9 − 265.9) | 247.0 | 24% | Not broken down: mapping, enqueue, dispatch, join, admission |
| Outside the FFN (≈1,025 − 885.8) | ≈140 | 14% | Scheduler between splits, output head, sampling, MTP draft |

The CPU part is close to the 2700X's practical DRAM read bandwidth, so faster
kernels can recover at most about 20% of 36%. The other **≈64% is handoff and
GPU-segment time**, and nothing measures its pieces. A short-prompt run
(`.unlazy/improve/model-after.log`, 14 prompt tokens) still shows
`router_wait=221.2`, so about 80% of the router wait does not depend on context
length. The prefetch ring sends about 147 MB per pass over PCIe (≈11 ms at the
measured 13.1 GB/s). Whether that traffic ever stalls the GPU is also unmeasured.

The repo's own handoff forbids choosing a fix from aggregate counters
("Do not rewrite kernels from aggregate `cpu_wait` alone"). This plan adds
one opt-in per-layer timeline that splits the 64% into named host intervals and
GPU-measured intervals. It then runs a fixed 5-run matrix. **No speedup is
expected from this plan itself.** Its output is the evidence that selects the
real fix (see "Decision table").

## Current state

Files involved:

- `ggml/src/ggml-backend.cpp`: defines `struct ggml_backend_sched_moe_ffn` (line 1377), the CPU job worker thread (`submit()`, about line 1992), `finish()` (about line 2032), the stats printer `report_new()` (line 1613), and the destructor that frees events (about line 2060). It `#include`s `ggml-cuda/moe-resident.h` (line 10).
- `ggml/src/ggml-moe-resident-new.inc`: per-layer decode logic. `ggml_backend_sched_moe_new_compute()` starts at line 940. `resident_finish_ffn()` starts at line 849. The whole file sits inside `#ifdef GGML_USE_CUDA` (line 3), with stubs after `#else` (line 1185).
- `ggml/src/ggml-cuda/moe-resident.h:30-31`: these helpers already exist. Use them and do not add CUDA code:
  ```cpp
  extern "C" GGML_API ggml_backend_event_t ggml_backend_cuda_moe_timed_event(ggml_backend_t backend);
  extern "C" GGML_API double ggml_backend_cuda_moe_elapsed_us(ggml_backend_event_t start, ggml_backend_event_t end);
  ```
  `ggml_backend_cuda_moe_timed_event` creates an event with timing enabled.
  Ordinary `ggml_backend_event_new` events are created with
  `cudaEventDisableTiming` and **cannot** be passed to the elapsed helper.
  `ggml_backend_cuda_moe_elapsed_us` requires both events to have completed.

Excerpts to confirm before editing:

`ggml/src/ggml-backend.cpp` (~1986–2025), the CPU job worker:
```cpp
    int64_t cpu_finished_us = 0; // published with busy=false under mutex
...
                    auto * current = job;
                    lock.unlock();
                    auto result = new_policy
                            ? ggml_backend_cpu_graph_compute_executor(cpu, &current->host_graph,
...
                    lock.lock();
                    status = result; cpu_finished_us = ggml_time_us(); busy = false;
```

`ggml/src/ggml-moe-resident-new.inc` (~1044–1075), the partial-layer path:
```cpp
    ++f.fallback_routes;
    ggml_backend_event_record(f.router_ready);
    if (!f.upload.pending && f.upload_fence) {
...
    // Router/activation copies and their event precede late forecast waits.
    // CPU misses can run while the GPU waits for its own expert weights.
    for (auto event : ring_waits) ggml_backend_event_wait(gpu, event);
...
    if (has_resident) {
        const int64_t started = ggml_time_us();
        ggml_backend_cuda_moe_route(gpu, t.up->src[3], device_map, &t.device_ids, experts);
        status = resident_gpu_ffn(f, t, gpu);
        f.gpu_enqueue_us += ggml_time_us() - started;
...
    const int64_t router_started = ggml_time_us();
    ggml_backend_event_synchronize(f.router_ready);
    f.router_wait_us += ggml_time_us() - router_started;
```
Further down in the same function, `f.submit(t);` is the single call that starts the CPU job (about line 1146).

`resident_finish_ffn()` (~864–936), the host-join path and the end of the layer:
```cpp
    if (cpu_count || !gpu_count) {
        const int64_t join_started = ggml_time_us();
        const auto cpu_status = f.finish();
...
    ggml_backend_event_record(f.completion); f.pending = true;
    ++f.layers; f.cpu_refs += cpu_count; f.gpu_refs += gpu_count;
...
    f.host_layer_us += ggml_time_us() - host_started;
    if (f.layers % (64ull * sched->moe_resident_layers) == 0) f.report_new("periodic");
    call.owner = nullptr;
    return GGML_STATUS_SUCCESS;
```

A layer whose experts are all on the GPU (`if (complete) { ... return GGML_STATUS_SUCCESS; }`,
~1003–1027) returns early and never reaches the CPU path. The timeline skips such
layers.

Repo conventions to match:

- **Environment-variable gate, read once.** Exemplar: `ggml/src/ggml-backend.cpp:5066-5069`:
  ```cpp
  static const size_t moe_ahead = [] {
      const char * env = getenv("GGML_MOE_PREFETCH_AHEAD");
      return env ? (size_t) std::max(0, atoi(env)) : (size_t) 3;
  }();
  ```
- **Stats output.** Write one `fprintf(stderr, "ggml_backend_sched: moe-resident new <name> key=value ...\n")` line inside `report_new()`, guarded by a counter being non-zero. Follow the `overlap deferred_layers=` line (~1637–1640) as the pattern.
- **Comments.** Write short comments that say why, not what. Do not add a header banner.
- **User rule for this fork: every fork change sits behind a flag, and with no flag set the code must behave exactly like stock.** Here the gate is the environment variable. With it unset, nothing new allocates, records events, or prints.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Incremental build (never run a full rebuild; the user forbids it) | `cmake --build build --target llama-cli test-moe-resident -j8 2>&1 \| tail -3` | exit 0, last lines show linking succeeded |
| CPU-only test | `./build/bin/test-moe-resident` | exit 0 |
| CUDA fixtures | `./build/bin/test-moe-resident --cuda-resident-batch` and `--cuda`, `--ring-order`, `--resident-history` | each exits 0 |
| GPU state | `nvidia-smi --query-gpu=name,utilization.gpu,memory.free,memory.total,clocks.sm --format=csv,noheader` | utilization ≤ 5 %, free ≥ 5000 MiB |
| RAM | `free -m \| sed -n 2p` | "available" ≥ 20000 |

Run `ulimit -c 0` in the same shell before every native test. Two CTest
failures already exist and are not yours: `test-chat-template` and
`test-eval-callback`. Do not run full CTest; it is not needed here.

## Scope

**In scope** (modify only these):
- `ggml/src/ggml-backend.cpp`: struct fields, the gate function, the worker start timestamp, the report line, and freeing the events.
- `ggml/src/ggml-moe-resident-new.inc`: timestamps and event records in `ggml_backend_sched_moe_new_compute()` and `resident_finish_ffn()`.
- `.unlazy/timeline/` (create it): the benchmark script, logs, and `REPORT.md`. `.unlazy/` is git-excluded.
- `plans/README.md`: the status row only.

**Out of scope** (do NOT touch):
- Any `.cu` or `.cuh` file. The helpers already exist, and the user forbids CUDA recompiles and full rebuilds.
- `common/`, `src/`, `examples/`: no CLI flag plumbing. Use the environment variable.
- Changing any scheduling, admission, prefetch, or ordering behaviour. This plan only measures. Even a fix that looks obvious goes into REPORT.md, not into the code.
- `.unlazy/improve/` evidence files. Do not overwrite them, because they hold the validated baseline. Do not run `.unlazy/improve/model-bench.py`: it overwrites `model-<label>.log`.
- The device-join path (`call.gated`). The timeline skips gated layers.

## Git workflow

- Stay on branch `moe-resident-dev`. Do **not** create a branch; the user dislikes extra branches.
- Commit only if the operator says commits are allowed, as one commit, for example:
  `perf(moe): add opt-in decode layer timeline (GGML_MOE_RESIDENT_TIMELINE)`, ending with the operator's attribution trailer if they give one.
  Match the style of `git log --oneline -5` (for example `fix(moe): read CPU inputs before late prefetch waits (6c)`).
- Never push. Do not touch the untracked `.skills/` directory.

## Steps

### Step 1: Add the gate, state, and report line in `ggml-backend.cpp`

1. Add the gate near the top of the file, after the includes, as a free function:
   ```cpp
   // Opt-in diagnostic: per-layer decode timeline for the new resident policy.
   static bool ggml_moe_resident_timeline_enabled() {
       static const bool enabled = [] {
           const char * env = getenv("GGML_MOE_RESIDENT_TIMELINE");
           return env && atoi(env) != 0;
       }();
       return enabled;
   }
   ```
2. In `struct ggml_backend_sched_moe_ffn`, add these fields directly after the line
   `uint64_t token_layers = 0, batch2_layers = 0, batch3_layers = 0, batch4_layers = 0;` (~1612):
   ```cpp
   // GGML_MOE_RESIDENT_TIMELINE: host intervals (us sums) and GPU intervals.
   struct timeline_stats {
       uint64_t layers = 0, pre_router_us = 0, dispatch_us = 0, wake_us = 0, cpu_compute_us = 0,
               join_react_us = 0, post_join_us = 0, between_us = 0, between_layers = 0;
       double gpu_segment_us = 0, ring_stall_us = 0, gpu_hits_us = 0;
       uint64_t gpu_segment_samples = 0, gpu_hits_samples = 0;
   } tl;
   struct timeline_events {
       ggml_backend_event_t routed = nullptr, wait_begin = nullptr, wait_end = nullptr,
               hits_end = nullptr, joined = nullptr;
       int layer = -1; bool has_wait = false, has_hits = false, joined_valid = false;
   } tl_ev[2];
   int tl_cur = 0, tl_last_layer = -1;
   int64_t tl_last_finish_us = 0, tl_router_done_us = 0, tl_submit_us = 0;
   ```
3. Next to `int64_t cpu_finished_us = 0;` (~1986), add
   `int64_t cpu_begin_us = 0; // worker start; read only after finish()`.
   In the worker lambda, directly after `lock.unlock();` and before `auto result = ...`, add:
   `cpu_begin_us = ggml_time_us();`
   (The host reads it only after `finish()` takes the mutex, so the existing `busy=false` handoff already orders the read after this write.)
4. In `report_new()`, directly after the `if (deferred_layers) fprintf(...)` block, add:
   ```cpp
   if (tl.layers) fprintf(stderr,
           "ggml_backend_sched: moe-resident new timeline layers=%llu pre_router=%.1f dispatch=%.1f wake=%.1f cpu_compute=%.1f join_react=%.1f post_join=%.1f between=%.1f ring_stall=%.1f us/layer gpu_segment=%.1f us/sample (n=%llu) gpu_hits=%.1f us/sample (n=%llu)\n",
           (unsigned long long)tl.layers,
           double(tl.pre_router_us) / tl.layers, double(tl.dispatch_us) / tl.layers,
           double(tl.wake_us) / tl.layers, double(tl.cpu_compute_us) / tl.layers,
           double(tl.join_react_us) / tl.layers, double(tl.post_join_us) / tl.layers,
           tl.between_layers ? double(tl.between_us) / tl.between_layers : 0.0,
           tl.ring_stall_us / tl.layers,
           tl.gpu_segment_samples ? tl.gpu_segment_us / tl.gpu_segment_samples : 0.0,
           (unsigned long long)tl.gpu_segment_samples,
           tl.gpu_hits_samples ? tl.gpu_hits_us / tl.gpu_hits_samples : 0.0,
           (unsigned long long)tl.gpu_hits_samples);
   ```
5. In the destructor, next to the line that frees `router_ready` and the other events (~2060), add:
   ```cpp
   for (auto & e : tl_ev) for (auto ev : {e.routed, e.wait_begin, e.wait_end, e.hits_end, e.joined}) ggml_backend_event_free(ev);
   ```
   `ggml_backend_event_free(nullptr)` is a no-op (ggml-backend.cpp:434-437).

**Verify**: `cmake --build build --target test-moe-resident -j8 2>&1 | tail -3` → exit 0.

### Step 2: Record the intervals in `ggml-moe-resident-new.inc`

All new code is wrapped in `if (tl) { ... }`, where `const bool tl = ggml_moe_resident_timeline_enabled();`
is declared once at the top of each function that uses it. Add a small helper above
`ggml_backend_sched_moe_new_compute()`:

```cpp
static ggml_backend_event_t resident_tl_event(ggml_backend_event_t & e, ggml_backend_t gpu) {
    if (!e) e = ggml_backend_cuda_moe_timed_event(gpu);
    return e;
}
```

In `ggml_backend_sched_moe_new_compute()` (partial-layer path only):

a. In the `if (complete) { ... }` block, before its `return GGML_STATUS_SUCCESS;`, add
   `if (tl) f.tl_last_layer = -1;`. A fully resident layer breaks the chain, so no
   cross-layer interval is computed across it.

b. Directly after `ggml_backend_event_record(f.router_ready);` add:
   ```cpp
   auto * tle = tl ? &f.tl_ev[f.tl_cur] : nullptr;
   if (tle) {
       tle->has_wait = tle->has_hits = tle->joined_valid = false; tle->layer = t.layer;
       ggml_backend_event_record(resident_tl_event(tle->routed, gpu));
   }
   ```
c. Replace the single line `for (auto event : ring_waits) ggml_backend_event_wait(gpu, event);`
   **in the partial path** (the one after the `upload_fence` block, not the one in `if (complete)`) with:
   ```cpp
   if (tle && !ring_waits.empty()) ggml_backend_event_record(resident_tl_event(tle->wait_begin, gpu));
   for (auto event : ring_waits) ggml_backend_event_wait(gpu, event);
   if (tle && !ring_waits.empty()) { ggml_backend_event_record(resident_tl_event(tle->wait_end, gpu)); tle->has_wait = true; }
   ```
   The order of the waits is unchanged. The new lines only add records.
d. Inside `if (has_resident) { ... }`, directly after `status = resident_gpu_ffn(f, t, gpu);`, add:
   `if (tle && status == GGML_STATUS_SUCCESS) { ggml_backend_event_record(resident_tl_event(tle->hits_end, gpu)); tle->has_hits = true; }`
e. Directly after `f.router_wait_us += ggml_time_us() - router_started;` add the harvest step.
   At this point `router_ready` is complete. Everything the previous layer queued sits
   before it on the same stream, so the previous layer's events are complete too.
   ```cpp
   if (tle) {
       f.tl_router_done_us = ggml_time_us();
       f.tl.pre_router_us += router_started - host_started;
       auto & prev = f.tl_ev[f.tl_cur ^ 1];
       const bool chained = f.tl_last_layer >= 0 && f.tl_last_layer + 1 == t.layer && prev.layer == f.tl_last_layer;
       if (chained) {
           f.tl.between_us += host_started - f.tl_last_finish_us; ++f.tl.between_layers;
           if (prev.joined_valid) {
               ggml_backend_event_synchronize(tle->routed); // recorded right after router_ready
               f.tl.gpu_segment_us += ggml_backend_cuda_moe_elapsed_us(prev.joined, tle->routed);
               ++f.tl.gpu_segment_samples;
           }
       }
       if (prev.layer >= 0 && prev.has_wait) f.tl.ring_stall_us += ggml_backend_cuda_moe_elapsed_us(prev.wait_begin, prev.wait_end);
       if (prev.layer >= 0 && prev.has_hits) {
           f.tl.gpu_hits_us += ggml_backend_cuda_moe_elapsed_us(prev.has_wait ? prev.wait_end : prev.routed, prev.hits_end);
           ++f.tl.gpu_hits_samples;
       }
       prev.layer = -1; prev.has_wait = prev.has_hits = prev.joined_valid = false; // harvested once
   }
   ```
   Note on `ring_stall`: this counts GPU time spent waiting on the previous layer's late
   prefetch uploads. It is harvested one layer late, so the first and last layers of a
   run can be missed. That is acceptable.
f. Directly before the single `f.submit(t);` call, add `if (tl) f.tl_submit_us = ggml_time_us();`

In `resident_finish_ffn()`:

g. Inside `if (cpu_count || !gpu_count) {`, directly after `const auto cpu_status = f.finish();`, add:
   ```cpp
   const int64_t tl_joined_us = ggml_time_us();
   if (tl && !call.gated) {
       f.tl.dispatch_us += std::max<int64_t>(0, f.tl_submit_us - f.tl_router_done_us);
       f.tl.wake_us += std::max<int64_t>(0, f.cpu_begin_us - f.tl_submit_us);
       f.tl.cpu_compute_us += std::max<int64_t>(0, f.cpu_finished_us - f.cpu_begin_us);
       f.tl.join_react_us += std::max<int64_t>(0, tl_joined_us - f.cpu_finished_us);
   }
   ```
   Declare `int64_t tl_joined_us = 0;` before the `if (cpu_count || !gpu_count)` block
   and assign it inside the block, so step h can read it.
h. Directly before the final `f.host_layer_us += ggml_time_us() - host_started;` at the end of
   `resident_finish_ffn()`, add:
   ```cpp
   if (tl && !call.gated && tl_joined_us) {
       auto & cur = f.tl_ev[f.tl_cur];
       ggml_backend_event_record(resident_tl_event(cur.joined, gpu)); cur.joined_valid = true;
       const int64_t now = ggml_time_us();
       f.tl.post_join_us += now - tl_joined_us;
       f.tl_last_finish_us = now; f.tl_last_layer = t.layer; ++f.tl.layers;
       f.tl_cur ^= 1;
   } else if (tl) {
       f.tl_last_layer = -1;
   }
   ```
   Note: `cur.joined` is recorded after the result upload and scatter have been queued.
   The next layer's attention and router therefore come after it on the stream, so
   `gpu_segment` = (time from the result being enqueued to the next router being ready)
   **as the GPU saw it**. If the host is slow to enqueue the next attention split,
   GPU idle time is included. Compare it with `between` to tell the two apart.

**Verify**:
- `cmake --build build --target llama-cli test-moe-resident -j8 2>&1 | tail -3` → exit 0
- `grep -c "ggml_moe_resident_timeline_enabled" ggml/src/ggml-moe-resident-new.inc` → `2` (one per function)
- `git diff --stat` lists only `ggml/src/ggml-backend.cpp` and `ggml/src/ggml-moe-resident-new.inc`.

### Step 3: Prove default behaviour is unchanged and the timeline prints

```bash
ulimit -c 0; mkdir -p .unlazy/timeline
for a in "" --cuda --cuda-resident-batch --ring-order --resident-history; do ./build/bin/test-moe-resident $a > /dev/null 2>&1; echo "off $a -> $?"; done
for a in --cuda --cuda-resident-batch; do GGML_MOE_RESIDENT_TIMELINE=1 ./build/bin/test-moe-resident $a > .unlazy/timeline/fixture$a.log 2>&1; echo "on $a -> $?"; grep -c "new timeline" .unlazy/timeline/fixture$a.log; done
./build/bin/test-moe-resident --cuda-resident-batch 2>&1 | grep -c "new timeline"
```
**Expected**: every `-> 0`. At least one of the two `on` runs prints a count ≥ 1. The last
command (env unset) prints `0`. If neither `on` fixture prints a timeline line, read which
fixture prints `new stats` (`grep -l "new stats" .unlazy/timeline/fixture*.log`). If none does, the
fixtures never reach `report_new`. That is acceptable. Note it in REPORT.md and rely on
Step 4's model run.

### Step 4: Run the 5-run matrix with the real model

Create `.unlazy/timeline/bench.py`. It is a copy of the pattern in
`.unlazy/improve/model-bench.py`, writes only into `.unlazy/timeline/`, and always uses
the long prompt:

```python
import os, pathlib, resource, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[2]
label, extra = sys.argv[1], sys.argv[2:]
env = dict(os.environ)
if label != 'r0-baseline': env['GGML_MOE_RESIDENT_TIMELINE'] = '1'
print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.free,memory.total,clocks.sm', '--format=csv,noheader'], text=True).strip(), flush=True)
print(subprocess.check_output(['free', '-m'], text=True).splitlines()[1], flush=True)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
cmd = ['./build/bin/llama-cli', '-m', '/mnt/Kingstone_SSD/GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', '--jinja', '-fa', 'on', '--no-context-shift', '-ngl', '999', '-np', '1', '-ctk', 'q4_0', '-ctv', 'q4_0', '-ctv-first', 'q8_0,4', '-ctv-last', 'q8_0,4', '--k-cache-hadamard', '--v-cache-hadamard', '-t', '8', '-tb', '8', '-b', '3072', '-wgt', '1', '-cmoe', '--ctx-size', '65536', '--moe-resident', '2048', '--moe-resident-profiler', 'new', '--moe-resident-prefetch', '2', '-cuda', 'fa-quant-tile=1', '--spec-type', 'mtp:n_max=1,p_min=0.0', '-f', str(root / '.unlazy/improve/long-prompt.txt'), '-n', '256', '--seed', '1']
i = 0
while i < len(extra):  # override "--flag value" pairs already in cmd
    if extra[i] in cmd: cmd[cmd.index(extra[i]) + 1] = extra[i + 1]
    else: cmd += extra[i:i + 2]
    i += 2
out = root / '.unlazy/timeline' / (label + '.out'); err = root / '.unlazy/timeline' / (label + '.log')
with out.open('w') as o, err.open('w') as e:
    p = subprocess.Popen(cmd, cwd=root, stdout=o, stderr=e, env=env)
    line = next(x for x in (pathlib.Path('/proc') / str(p.pid) / 'limits').read_text().splitlines() if x.startswith('Max core file size'))
    assert line.split()[4:6] == ['0', '0'], line
    code = p.wait(timeout=900)
print('exit', code, flush=True)
for x in err.read_text(errors='replace').splitlines():
    if x.startswith('main: ') or 'reason=shutdown layers=' in x or 'new timeline' in x or 'new prefetch' in x or 'statistics mtp' in x:
        print(x)
sys.exit(code)
```

Run these one at a time and never two model processes at once. Before **each** run,
take the GPU and RAM readings from "Commands you will need". If GPU utilization is above
5% or free VRAM is below 5000 MiB, STOP and ask the user. Do not kill other programs
yourself.

| Run | Command | Purpose |
|---|---|---|
| R0 | `python3 .unlazy/timeline/bench.py r0-baseline` | Reference throughput with the timeline off |
| R1 | `python3 .unlazy/timeline/bench.py r1-timeline` | Decomposition; overhead check against R0 |
| R2 | `python3 .unlazy/timeline/bench.py r2-prefetch0 --moe-resident-prefetch 0` | Is the prefetch ring worth its PCIe traffic? |
| R3 | `python3 .unlazy/timeline/bench.py r3-prefetch1 --moe-resident-prefetch 1` | Intermediate point |
| R4 | `python3 .unlazy/timeline/bench.py r4-threads6 -t 6` | Fewer spinning CPU threads |

**Verify after each run**:
- `exit 0`
- a `main: prompt eval time = ... / 28000 tokens` line (the prompt count must be exactly 28000)
- an `eval time = ... / 256 tokens` line
- for R1–R4: a `new timeline` line from the last `reason=shutdown` group.
  Use the line whose `layers=` value is in the thousands. The line with about 145 layers belongs to the MTP context.

### Step 5: Write `.unlazy/timeline/REPORT.md`

Contents (numbers only from the logs; label every derived number as derived):

1. A table with one row per run: t/s, `cache=` MiB, `gpu_share`, accepted drafts
   (`#acc drafts`), `router_wait`, `cpu_wait`, `ffn_host`, and prefetch `bytes`.
2. For R1, the per-layer budget: `pre_router`, `router_wait` (from the stats line),
   `dispatch`, `wake`, `cpu_compute`, `join_react`, `post_join`, `between`, `gpu_segment`,
   `ring_stall`, `gpu_hits`.
   Also report the derived CPU bandwidth = `unique_experts / layers × 1.911 MB / cpu_compute`.
   1.911 MB is the average expert size: ring `bytes / uploads` in the baseline log.
3. The R1/R0 throughput ratio. If R1 is more than 3% slower, flag that the timeline perturbs the result.
4. Which row of the "Decision table" below applies, with the numbers that put it there.
5. Machine state recorded before each run.

Do **not** implement the chosen fix. Stop after REPORT.md.

## Decision table (for the REPORT, not for implementing)

Use R1 values in us/layer. "Handoff" = `dispatch + wake + join_react + post_join + between`.

| If R1 shows | Then the biggest lever is | Likely next plan |
|---|---|---|
| R2 or R3 is ≥ 3% faster than R0 | The prefetch ring costs more than it saves | Make the ring cost-aware or limit it to entries that arrive in time |
| `ring_stall` ≥ 50 | Late uploads stall the GPU half | Same as above, or fall back to the CPU for an entry that is still late |
| Handoff ≥ 150 | Host serialization between CPU and GPU | Trim the host work on the critical path (`post_join`/`dispatch`), or measure device join with hardware stream memops (needs a user-side driver option; see `docs/moe-resident-profiler-new.md` "Device join") |
| `gpu_segment` ≥ 250 and `between` < 50 | The GPU attention and router chain itself | Profile kernel count and duration for one layer of the router graph. Fusion work is probably upstream territory. |
| `cpu_compute` ≥ 300 and derived bandwidth ≥ 30 GB/s | DRAM bandwidth on the CPU side | Only a higher GPU share helps (VRAM budget, admission), not kernels |
| R4 ≥ 3% faster than R0 | CPU thread contention | Change the recommended `-t` in the docs (no code) |

More than one row can apply. Rank them by us/layer saved, as estimated from the numbers.

## Test plan

- No new test file. The change is diagnostic and off by default.
- Step 3 covers: default-off parity (existing fixtures unchanged), the gate turning on, and no crash or leak at teardown with timed events allocated.
- Step 4 R0 is the regression check for the default path: its t/s should be within about 5% of the 42.79 baseline at equal `cache=`. Different free VRAM changes `cache=` and makes the numbers incomparable, so note it rather than calling it a regression.

## Done criteria

- [ ] `cmake --build build --target llama-cli test-moe-resident -j8` exits 0
- [ ] Step 3: all `off` runs exit 0. The env-unset `--cuda-resident-batch` run prints `0` timeline lines.
- [ ] `.unlazy/timeline/r0-baseline.log` … `r4-threads6.log` exist. Each has an `eval time = ... / 256 tokens` line and a prompt of `28000 tokens`.
- [ ] `r1-timeline.log` contains `moe-resident new timeline layers=` with layers > 1000
- [ ] `.unlazy/timeline/REPORT.md` exists with sections 1–5
- [ ] `git status --short` shows changes only to `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-moe-resident-new.inc`, `plans/README.md`, plus the existing untracked `.skills/`
- [ ] `plans/README.md` row 005 updated

## STOP conditions

- The drift check shows a change and the excerpts no longer match.
- Any `off` fixture in Step 3 fails. Default behaviour must not change.
- The build needs to recompile `.cu` files, or the build command starts compiling hundreds of objects. Stop and ask; the user forbids full rebuilds.
- A model run exits non-zero, prints `Failed to sample`, NaN, or `device join timed out`, or its prompt count is not 28000.
- The GPU is busy or VRAM is short before a run (see Step 4).
- `ggml_backend_cuda_moe_elapsed_us` aborts with a CUDA error. This means an event was not complete or was created without timing. Report the call site.
- You feel the urge to change scheduling or prefetch logic. That is the next plan, not this one.

## Maintenance notes

- The timeline depends on one compute stream ordering for `router_ready`, the ring waits, the hits, and the result upload. If a future change moves hits or results to another stream, `gpu_segment` and `gpu_hits` lose their meaning. Re-derive the harvest argument in step 2e.
- Gated (device-join) layers are skipped on purpose. Supporting them needs the worker-side gate timestamp, not `finish()`.
- The two `ggml_backend_event_synchronize(tle->routed)` calls per layer cost a few microseconds and exist only when the variable is set. R1 vs R0 quantifies that cost.
- Keep the variable name stable. Later plans compare against R1 numbers.
