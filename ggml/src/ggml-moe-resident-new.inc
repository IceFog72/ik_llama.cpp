// Opt-in policy. This file is included after complete-FFN helpers and never
// executes unless both automatic residency and profiler=new are selected.
#ifdef GGML_USE_CUDA
static ggml_tensor resident_vector(ggml_backend_buffer_t buffer, ggml_type type, size_t n) {
    ggml_tensor t = {};
    t.type = type;
    t.ne[0] = int64_t(n); t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = t.nb[2] = t.nb[3] = n * t.nb[0];
    ggml_backend_sched_moe_ffn_bind(t, buffer);
    return t;
}

static void resident_history(ggml_backend_sched_t sched) {
    auto & f = *sched->moe_ffn;
    const size_t n = size_t(sched->moe_resident_layers) * sched->moe_resident_n_expert;
    // The target graph excludes a model's extra predictor layers. Keep the
    // last decode width while scanning prefill, which has no eligible tasks.
    if (!f.graph_tasks.empty()) f.history_layers = std::max<size_t>(1, f.graph_tasks.size() / 2);
    else if (f.popularity.empty()) f.history_layers = std::max(1, sched->moe_resident_layers);
    if (f.popularity.size() != n) {
        f.popularity.assign(n, 0);
        f.recent_popularity.assign(n, 0);
        f.observed_at.assign(n, f.policy_clock);
        f.mapping.assign(n, -1);
        f.cpu_cost.assign(size_t(sched->moe_resident_layers), 100.0);
        f.cpu_exposed_cost.assign(size_t(sched->moe_resident_layers), 100.0);
        f.cpu_marginal_cost.assign(size_t(sched->moe_resident_layers), 50.0);
        f.cpu_samples.resize(size_t(sched->moe_resident_layers));
    }
}

static void ggml_backend_sched_moe_new_collect(ggml_backend_sched_t sched) {
    auto & f = *sched->moe_ffn;
    resident_history(sched);
    if (!f.histogram_dirty || !f.histogram) return;
    ggml_backend_synchronize(sched->backends[0]);
    std::vector<uint32_t> counts(f.popularity.size());
    auto histogram = resident_vector(f.histogram, GGML_TYPE_I32, counts.size());
    ggml_backend_tensor_get(&histogram, counts.data(), 0, counts.size() * sizeof(uint32_t));
    const int experts = sched->moe_resident_n_expert;
    uint64_t references = 0;
    int observed_layers = 0;
    for (int layer = 0; layer < sched->moe_resident_layers; ++layer) {
        uint64_t total = 0;
        for (int e = 0; e < experts; ++e) total += counts[size_t(layer) * experts + e];
        if (!total) continue;
        references += total;
        ++observed_layers;
        // A long prompt seeds a bounded history, rather than dominating all
        // later decode observations simply because it contains more tokens.
        const double scale = std::min(1.0, double(64 * sched->moe_resident_n_expert_used) / total);
        for (int e = 0; e < experts; ++e) {
            const size_t i = size_t(layer) * experts + e;
            f.popularity[i] = f.session_score(layer, e, experts) * 0.5 + counts[i] * scale;
            f.recent_popularity[i] = f.recent_score(layer, e, experts) * 0.5 + counts[i] * scale / 16;
            f.observed_at[i] = f.policy_clock;
        }
    }
    ggml_backend_buffer_clear(f.histogram, 0);
    f.histogram_dirty = false;
    if (references) fprintf(stderr,
            "ggml_backend_sched: moe-resident new history layers=%d/%d references=%llu\n",
            observed_layers, sched->moe_resident_layers, (unsigned long long)references);
}

static void ggml_backend_sched_moe_new_observe(ggml_backend_sched_t sched,
        const ggml_tensor * node, ggml_backend_t backend) {
    auto & f = *sched->moe_ffn;
    const auto found = f.observations.find(node);
    resident_history(sched);
    // Prefill routing is summarized on-device; no per-prompt-token readback.
    if (ggml_backend_is_cuda(backend)) {
        if (!f.histogram) {
            f.histogram = ggml_backend_buft_alloc_buffer(sched->bufts[0], f.popularity.size() * sizeof(uint32_t));
            if (!f.histogram) return;
            ggml_backend_buffer_clear(f.histogram, 0);
        }
        auto * counts = static_cast<uint32_t *>(ggml_backend_buffer_get_base(f.histogram));
        ggml_backend_cuda_moe_profile(backend, counts, sched->moe_resident_layers, sched->moe_resident_n_expert);
        f.profile_backend = backend;
        f.histogram_dirty = true;
    } else if (found != f.observations.end()) {
        const auto * ids = node->src[3];
        if (!ids || !ids->buffer || !ggml_is_contiguous(ids) || !ggml_backend_buffer_is_host(ids->buffer)) return;
        const auto * values = static_cast<const int32_t *>(ids->data);
        const int experts = sched->moe_resident_n_expert;
        const double scale = std::min(1.0, 64.0 / std::max<int64_t>(1, ids->ne[1]));
        for (int64_t j = 0; j < ggml_nelements(ids); ++j) if (values[j] >= 0 && values[j] < experts) {
            const size_t i = size_t(found->second) * experts + values[j];
            f.popularity[i] = f.session_score(found->second, values[j], experts) + scale;
            f.recent_popularity[i] = f.recent_score(found->second, values[j], experts) + scale;
            f.observed_at[i] = f.policy_clock;
        }
    }
}

static void resident_bank(ggml_backend_sched_moe_ffn & f, size_t group, int bank,
        ggml_tensor & tensor) {
    auto & g = f.groups[group];
    ggml_backend_sched_moe_ffn_bind(tensor, f.virtual_cache ? g.storage[bank] : f.cache);
    if (!f.virtual_cache) tensor.data = static_cast<uint8_t *>(tensor.data) + g.offsets[bank];
    // Prefetch ring entries follow the cache slots in the same storage.
    const size_t slots = g.slots.size() + (f.rings.empty() ? 0 : size_t(f.ring_size));
    tensor.ne[2] = int64_t(slots);
    tensor.nb[2] = g.strides[bank]; tensor.nb[3] = tensor.nb[2] * slots;
}

static void resident_upload(ggml_backend_sched_moe_ffn & f,
        ggml_backend_sched_moe_ffn::task & t, int expert, size_t position, ggml_backend_t backend) {
    auto & g = f.groups[t.group];
    for (int b = 0; b < 3; ++b) if (t.banks[b]) {
        auto destination = *t.banks[b];
        resident_bank(f, t.group, b, destination);
        ggml_moe_prefetch_wait(t.banks[b]);
        ggml_backend_tensor_set_async(backend, &destination,
                static_cast<const uint8_t *>(t.banks[b]->data) + size_t(expert) * t.banks[b]->nb[2],
                position * g.strides[b], t.banks[b]->nb[2]);
        f.host_bytes += t.banks[b]->nb[2];
    }
}

static void resident_mapping(ggml_backend_sched_t sched, ggml_backend_t gpu, bool incremental = false) {
    auto & f = *sched->moe_ffn;
    f.mapping_scratch.assign(f.mapping.size(), -1);
    for (auto & g : f.groups) for (size_t p = 0; p < g.slots.size(); ++p) {
        const auto & s = g.slots[p];
        if (s.layer >= 0 && s.expert >= 0)
            f.mapping_scratch[size_t(s.layer) * sched->moe_resident_n_expert + s.expert] = int32_t(p);
    }
    for (size_t gi = 0; gi < f.rings.size(); ++gi) for (size_t i = 0; i < f.rings[gi].size(); ++i) {
        const auto & r = f.rings[gi][i];
        if (r.installed) f.mapping_scratch[size_t(r.layer) * sched->moe_resident_n_expert + r.expert] =
                int32_t(f.groups[gi].slots.size() + i);
    }
    auto map = resident_vector(f.slot_map, GGML_TYPE_I32, f.mapping.size());
    // Keep the reusable DMA source at its stable address. Callers fence the
    // previous source before this update; batch retirement fences reservation.
    for (size_t begin = 0; begin < f.mapping.size();) {
        if (incremental && f.mapping[begin] == f.mapping_scratch[begin]) { ++begin; continue; }
        size_t end = begin + 1;
        if (!incremental) end = f.mapping.size();
        else while (end < f.mapping.size() && f.mapping[end] != f.mapping_scratch[end]) ++end;
        std::copy(f.mapping_scratch.begin() + begin, f.mapping_scratch.begin() + end, f.mapping.begin() + begin);
        const size_t bytes = (end - begin) * sizeof(int32_t);
        ggml_backend_tensor_set_async(gpu, &map, f.mapping.data() + begin, begin * sizeof(int32_t), bytes);
        f.mapping_bytes += bytes;
        begin = end;
    }
    f.mapping_dirty = false;
    ++f.mapping_uploads;
}

struct resident_candidate { int layer, expert; size_t group; double value; };
static std::vector<resident_candidate> resident_candidates(ggml_backend_sched_t sched) {
    auto & f = *sched->moe_ffn;
    std::vector<resident_candidate> candidates;
    for (const auto & t : f.tasks) if (t && f.live_task(*t)) {
        size_t bytes = 0; for (size_t stride : f.groups[t->group].strides) bytes += stride;
        if (!bytes) continue;
        for (int e = 0; e < sched->moe_resident_n_expert; ++e) {
            const double value = f.reuse_value(t->layer, e, sched->moe_resident_n_expert, bytes);
            if (value > 0) candidates.push_back({t->layer, e, t->group, value});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const resident_candidate & a, const resident_candidate & b) {
        return a.value > b.value;
    });
    return candidates;
}

static bool ggml_backend_sched_moe_new_cache(ggml_backend_sched_t sched,
        ggml_backend_t gpu, bool resize) {
    auto & f = *sched->moe_ffn;
    // A prefill ubatch can expose only its last layer as a one-token FFN, which
    // creates the cache with a one-layer plan. Re-plan once a graph has more.
    const size_t live_tasks = f.graph_tasks.size() / 2;
    if (f.cache && !resize) {
        if (live_tasks <= f.planned_tasks) return true;
        resize = true;
    }
    if (f.disabled) return false;
    f.finish(); f.wait_io(); f.finish_upload();
    f.ring_reset(); // ring entries are rebuilt after any capacity change
    ggml_backend_synchronize(gpu);
    // Prefill can have allocated ordinary projection caches. They are not
    // used by complete-FFN decode and must not reduce its available budget.
    auto & pools = sched->moe_resident[0];
    for (int p = 0; p < pools.n_pools; ++p)
        ggml_backend_sched_moe_resident_clear_storage(&pools.pools[p], &sched->moe_resident_budget[0]);
    ggml_backend_cuda_trim_compute(gpu); // captures also embed bank dimensions
    resident_history(sched);
    ggml_backend_sched_moe_new_collect(sched);
    const int device = ggml_backend_cuda_get_device(gpu);
    if (!ggml_backend_sched_moe_decode_buffer(f.slot_map, sched->bufts[0], f.mapping.size() * sizeof(int32_t))) {
        // Same CPU fallback as a failed cache plan; the next response retries.
        f.remember_cache(); f.clear_cache(); f.disabled = true; return false;
    }
    size_t free_bytes = 0, total = 0;
    ggml_backend_cuda_get_device_memory(device, &free_bytes, &total);
    f.budget_bytes = ggml_backend_sched_moe_ffn_budget(sched, free_bytes + f.allocation_bytes, f.reserve_bytes);
    const size_t gran = ggml_backend_cuda_moe_cache_granularity(device);
    size_t padding = ggml_backend_buffer_get_size(f.slot_map) +
            (f.histogram ? ggml_backend_buffer_get_size(f.histogram) : 0);
    for (const auto & g : f.groups) for (size_t stride : g.strides) if (stride) padding += gran;
    const size_t ring = size_t(f.ring_size); // prefetch entries per group, outside cache capacity
    for (const auto & g : f.groups) for (size_t stride : g.strides) padding += ring * stride;
    std::vector<size_t> counts;
    size_t logical_bytes = 0;
    if (f.budget_bytes <= padding || !f.plan_cache(f.budget_bytes - padding,
            sched->moe_resident_n_expert_used, sched->moe_resident_n_expert, counts, logical_bytes)) {
        f.remember_cache(); f.clear_cache(); f.disabled = true; return false;
    }
    const auto candidates = resident_candidates(sched);
    if (!candidates.empty()) {
        // Allocate additional capacity by measured reuse value, keeping enough
        // slots for one complete route in every live layout.
        std::vector<size_t> learned(f.groups.size(), 0), seen(f.groups.size(), 0), sizes(f.groups.size(), 0);
        size_t used = 0;
        for (const auto & t : f.tasks) if (t && f.live_task(*t)) learned[t->group] = size_t(sched->moe_resident_n_expert_used);
        for (size_t gi = 0; gi < f.groups.size(); ++gi) {
            for (size_t stride : f.groups[gi].strides) sizes[gi] += stride;
            used += learned[gi] * sizes[gi];
        }
        const size_t limit = f.budget_bytes - padding;
        if (used <= limit) {
            for (const auto & c : candidates) {
                if (++seen[c.group] <= size_t(sched->moe_resident_n_expert_used)) continue;
                if (sizes[c.group] <= limit - used) { ++learned[c.group]; used += sizes[c.group]; }
            }
            // Spare slots remain useful for new observations, even if the
            // prompt exposed fewer experts than the memory budget can hold.
            bool added = true;
            while (added) {
                added = false;
                for (size_t gi = 0; gi < learned.size(); ++gi) {
                    size_t layers = 0;
                    for (const auto & t : f.tasks) if (t && f.live_task(*t) && t->group == gi) ++layers;
                    if (sizes[gi] && learned[gi] < layers * size_t(sched->moe_resident_n_expert) && sizes[gi] <= limit - used) {
                        ++learned[gi]; used += sizes[gi]; added = true;
                    }
                }
            }
            counts = std::move(learned);
            logical_bytes = used;
        }
    }
    if (f.cache && !f.virtual_cache) {
        // Platforms without VMM keep a fitting cache; growth is deferred.
        if (f.allocation_bytes <= f.budget_bytes) { f.planned_tasks = live_tasks; return true; }
        f.remember_cache(); f.clear_cache();
    }
    f.remember_cache();
    if (gran) {
        f.virtual_cache = true;
        // Shrink every bank before growing any: no transient second cache.
        for (size_t gi = 0; gi < f.groups.size(); ++gi) {
            auto & g = f.groups[gi];
            if (counts[gi] < g.slots.size()) {
                // Preserve valuable tail entries inside the retained prefix
                // with device copies before unmapping the discarded pages.
                const auto value = [&](const ggml_backend_sched_moe_ffn::slot & s) {
                    return s.layer < 0 ? 0.0 : f.score(s.layer, s.expert, sched->moe_resident_n_expert) * f.cpu_cost[s.layer];
                };
                for (size_t p = counts[gi]; p < g.slots.size() && counts[gi]; ++p) {
                    if (g.slots[p].layer < 0) continue;
                    size_t victim = 0;
                    for (size_t q = 1; q < counts[gi]; ++q)
                        if (value(g.slots[q]) < value(g.slots[victim])) victim = q;
                    if (value(g.slots[p]) <= value(g.slots[victim])) continue;
                    const auto & old = g.slots[victim];
                    if (old.layer >= 0) g.slot_index.erase((uint64_t(old.layer) << 32) | uint32_t(old.expert));
                    for (int b = 0; b < 3; ++b) if (g.storage[b]) {
                        ggml_backend_cuda_moe_cache_move(gpu, g.storage[b], victim * g.strides[b], p * g.strides[b], g.strides[b]);
                        f.d2d_bytes += g.strides[b];
                    }
                    g.slots[victim] = g.slots[p];
                    g.slot_index[(uint64_t(g.slots[p].layer) << 32) | uint32_t(g.slots[p].expert)] = int(victim);
                    g.slots[p] = {};
                }
                ggml_backend_synchronize(gpu);
                for (size_t p = counts[gi]; p < g.slots.size(); ++p) {
                    const auto & s = g.slots[p];
                    if (s.layer >= 0) g.slot_index.erase((uint64_t(s.layer) << 32) | uint32_t(s.expert));
                }
                g.slots.resize(counts[gi]);
                for (int b = 0; b < 3; ++b) if (g.storage[b] &&
                        !ggml_backend_cuda_moe_cache_resize(g.storage[b], (counts[gi] + ring) * g.strides[b])) {
                    f.clear_cache(); f.disabled = true; return false;
                }
            }
        }
        size_t physical = 0;
        for (size_t gi = 0; gi < f.groups.size(); ++gi) {
            auto & g = f.groups[gi];
            const size_t capacity = size_t(sched->moe_resident_layers) * sched->moe_resident_n_expert + ring;
            for (int b = 0; b < 3; ++b) if (g.strides[b]) {
                if (!g.storage[b]) g.storage[b] = ggml_backend_cuda_moe_cache_new(device, capacity * g.strides[b]);
                if (!g.storage[b] || !ggml_backend_cuda_moe_cache_resize(g.storage[b], (counts[gi] + ring) * g.strides[b])) {
                    f.clear_cache(); f.disabled = true; return false;
                }
                physical += ggml_backend_buffer_get_size(g.storage[b]);
                if (!f.cache) f.cache = g.storage[b]; // non-owning presence marker
            }
            g.slots.resize(counts[gi]);
        }
        f.allocation_bytes = physical;
    } else {
        size_t offset = 0;
        for (size_t gi = 0; gi < f.groups.size(); ++gi) {
            auto & g = f.groups[gi];
            g.slots.resize(counts[gi]);
            for (int b = 0; b < 3; ++b) { g.offsets[b] = offset; offset += (counts[gi] + ring) * g.strides[b]; }
        }
        f.cache = ggml_backend_buft_alloc_buffer(sched->bufts[0], offset);
        if (!f.cache) { f.clear_cache(); f.disabled = true; return false; }
        ggml_backend_buffer_clear(f.cache, 0);
        f.allocation_bytes = offset;
    }
    if (ring) {
        f.rings.resize(f.groups.size());
        for (auto & r : f.rings) r.resize(ring);
    }
    // Fill vacant slots from bounded conversation history. Existing slots and
    // their physical bytes survive a VMM resize; only genuinely new slots copy.
    for (const auto & c : candidates) {
        auto & g = f.groups[c.group];
        const uint64_t key = (uint64_t(c.layer) << 32) | uint32_t(c.expert);
        if (g.slot_index.count(key)) continue;
        const auto vacant = std::find_if(g.slots.begin(), g.slots.end(), [](const ggml_backend_sched_moe_ffn::slot & s) {
            return s.layer < 0;
        });
        if (vacant == g.slots.end()) continue;
        auto & t = *f.tasks[c.layer];
        const size_t p = size_t(vacant - g.slots.begin());
        resident_upload(f, t, c.expert, p, gpu);
        vacant->layer = c.layer; vacant->expert = c.expert; vacant->used = ++f.clock;
        for (int b = 0; b < 3; ++b) { vacant->banks[b] = t.banks[b]; vacant->data[b] = t.banks[b] ? t.banks[b]->data : nullptr; }
        g.slot_index[key] = int(p); ++f.fills;
    }
    f.remembered.clear();
    for (auto & g : f.groups) { g.victims.clear(); g.victim_epoch = UINT64_MAX; }
    resident_mapping(sched, gpu);
    ggml_backend_synchronize(gpu);
    f.planned_tasks = live_tasks;
    fprintf(stderr, "ggml_backend_sched: moe-resident new cache=%zu MiB slots=%zu layers=%zu budget=%zu MiB stable=%s history=%zu\n",
            f.allocation_bytes / 1048576, f.slot_count(), live_tasks, f.budget_bytes / 1048576,
            f.virtual_cache ? "yes" : "no", candidates.size());
    GGML_UNUSED(total);
    return true;
}

static ggml_status resident_gpu_ffn(ggml_backend_sched_moe_ffn & f,
        ggml_backend_sched_moe_ffn::task & t, ggml_backend_t gpu) {
    for (int b = 0; b < 3; ++b) if (t.banks[b]) {
        t.device_banks[b] = *t.banks[b];
        resident_bank(f, t.group, b, t.device_banks[b]);
    }
    t.device_up = *t.up; t.device_down = *t.down;
    // The CUDA prefill histogram consumes original expert IDs. These custom
    // nodes consume physical cache IDs and must not enter that histogram.
    t.device_up.op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0x52534e50;
    t.device_up.src[0] = &t.device_banks[0];
    t.device_up.src[1] = t.banks[1] ? &t.device_banks[1] : nullptr;
    t.device_up.src[3] = &t.device_ids;
    t.device_down.src[0] = &t.device_banks[2];
    t.device_down.src[1] = &t.device_up; t.device_down.src[2] = &t.device_ids;
    t.device_nodes[0] = &t.device_up; t.device_nodes[1] = &t.device_down;
    t.device_graph = {}; t.device_graph.nodes = t.device_nodes; t.device_graph.n_nodes = 2;
    ggml_backend_cuda_moe_copy_rows(gpu, &t.device_up, nullptr, &t.device_ids);
    return ggml_backend_graph_compute_async(gpu, &t.device_graph);
}

enum class resident_admission { rejected, deferred, submitted };

// Lowest-value slots, refreshed at most once per model pass.
template <typename value_fn>
static void resident_refresh_victims(ggml_backend_sched_moe_ffn & f, ggml_backend_sched_moe_ffn::group & g,
        uint64_t epoch, const value_fn & value) {
    if (!g.victims.empty() && g.victim_epoch == epoch) return;
    g.victims.clear();
    for (size_t pos = 0; pos < g.slots.size(); ++pos) if (g.slots[pos].layer != -2)
        g.victims.emplace_back(value(g.slots[pos].layer, g.slots[pos].expert), pos);
    const size_t count = std::min<size_t>(64, g.victims.size());
    std::partial_sort(g.victims.begin(), g.victims.begin() + count, g.victims.end());
    g.victims.resize(count);
    g.victim_epoch = epoch;
    GGML_UNUSED(f);
}

static bool resident_pinned(const ggml_tensor * bank) {
    return bank && bank->buffer && ggml_backend_buffer_get_type(bank->buffer) == ggml_backend_cuda_host_buffer_type();
}

static resident_admission resident_promote(ggml_backend_sched_t sched, ggml_backend_sched_moe_ffn::task & t,
        const std::vector<int> & missing, const std::vector<bool> & protected_slots, ggml_backend_t gpu,
        bool prediction = false, bool early_fence = false, double window_us = -1) {
    auto & f = *sched->moe_ffn;
    if (!f.cache || missing.empty()) return resident_admission::rejected;
    if (f.upload.pending) return resident_admission::deferred;
    const int64_t admission_started = ggml_time_us();
    auto & g = f.groups[t.group];
    const int experts = sched->moe_resident_n_expert;
    size_t bytes = 0; for (size_t stride : g.strides) bytes += stride;
    if (!bytes) return resident_admission::deferred;
    const uint64_t epoch = f.policy_clock / f.history_layers;
    if (f.traffic_epoch != epoch) {
        f.traffic_epoch = epoch;
        // At most 1/128 cache turnover per model pass, capped at 16 MiB.
        // Tiny caches still permit one expert. No credits accumulate across
        // passes, widths or the active/post-join/between-invocation call sites.
        f.traffic_remaining = std::max(bytes, std::min<size_t>(16 * 1024 * 1024, f.allocation_bytes / 128));
    }
    const size_t batch_limit = f.upload_limit(bytes, window_us);
    if (!batch_limit) {
        if (f.traffic_remaining < bytes) ++f.traffic_deferred;
        else ++f.window_deferred;
        return resident_admission::deferred;
    }
    const auto value = [&](int layer, int expert) { return f.reuse_value(layer, expert, experts, bytes); };
    auto & candidates = f.admission_candidates;
    candidates.clear();
    for (const int e : missing) if (e >= 0 && e < experts &&
            !g.slot_index.count((uint64_t(t.layer) << 32) | uint32_t(e)) &&
            std::none_of(candidates.begin(), candidates.end(), [e](const auto & c) { return c.second == e; }))
        candidates.emplace_back(f.reuse_value(t.layer, e, experts, bytes, prediction), e);
    if (candidates.empty()) return resident_admission::rejected;
    // Cheap absolute rejection avoids a cache-wide ranking for one-off hints.
    const double minimum = 1.25 * f.upload_cost(bytes) / bytes;
    if (std::none_of(candidates.begin(), candidates.end(), [minimum](const auto & c) { return c.first > minimum; })) {
        ++f.admission_rejected; return resident_admission::rejected;
    }
    resident_refresh_victims(f, g, epoch, value);
    std::vector<ggml_backend_sched_moe_ffn::promotion::item> admitted;
    while (!candidates.empty() && admitted.size() < batch_limit) {
        const auto best = std::max_element(candidates.begin(), candidates.end());
        const int expert = best->second;
        const double candidate_value = best->first;
        const double candidate_history = f.session_score(t.layer, expert, experts);
        candidates.erase(best);
        size_t victim = g.slots.size();
        // Snapshot ranks are at most one pass old. Validate identity, protection
        // and current value only for this bounded shortlist, without repeated
        // exponentials in a sort comparator or a full slot scan per layer.
        for (const auto & ranked : g.victims) {
            const size_t pos = ranked.second;
            if (pos >= g.slots.size() || protected_slots[pos] || g.slots[pos].layer == -2 ||
                    std::any_of(admitted.begin(), admitted.end(), [&](const auto & item) { return item.position == pos; })) continue;
            const auto & slot = g.slots[pos];
            if (f.slot_protected(slot)) continue;
            if (slot.layer >= 0) {
                const size_t id = size_t(slot.layer) * experts + slot.expert;
                if (id < f.forecast_probability.size() && f.forecast_probability[id] >= 0.75) continue;
            }
            if (f.admission_allows(candidate_value, candidate_history, slot, experts, bytes, prediction)) { victim = pos; break; }
        }
        if (victim == g.slots.size()) { ++f.admission_rejected; continue; }
        ggml_backend_sched_moe_ffn::promotion::item item;
        item.group = t.group; item.position = victim;
        item.entry.layer = t.layer; item.entry.expert = expert; item.entry.predicted = prediction;
        item.entry.admitted_at = f.policy_clock;
        item.entry.admission_cost_us = f.upload_cost(bytes);
        for (int bank = 0; bank < 3; ++bank) {
            item.entry.banks[bank] = t.banks[bank];
            item.entry.data[bank] = t.banks[bank] ? t.banks[bank]->data : nullptr;
        }
        admitted.push_back(item);
    }
    if (admitted.empty()) return resident_admission::rejected;
    size_t staging_bytes = 0;
    for (auto * bank : t.banks) if (bank && !resident_pinned(bank)) staging_bytes += bank->nb[2];
    const bool setup = !f.uploader || !f.upload_worker.joinable() || !f.upload_staging ||
            ggml_backend_buffer_get_size(f.upload_staging) < staging_bytes * admitted.size();
    const int64_t setup_started = setup ? ggml_time_us() : 0;
    if (!f.uploader) {
        f.uploader = ggml_backend_cuda_init(ggml_backend_cuda_get_device(gpu), nullptr, &f);
        if (!f.uploader) return resident_admission::deferred;
        f.upload_started = ggml_backend_cuda_moe_timed_event(f.uploader);
        f.upload_ready = ggml_backend_cuda_moe_timed_event(f.uploader);
        f.upload_fence = ggml_backend_event_new(gpu);
    }
    if (!f.upload_started || !f.upload_ready || !f.upload_fence) return resident_admission::deferred;
    if (!ggml_backend_sched_moe_decode_buffer(f.upload_staging,
            ggml_backend_cuda_host_buffer_type(), std::max<size_t>(1, staging_bytes * admitted.size()))) return resident_admission::deferred;
    f.ensure_upload_worker();
    const int64_t setup_us = setup ? ggml_time_us() - setup_started : 0;
    f.upload_setup_us += setup_us;
    for (auto & item : admitted) item.entry.admission_cost_us += double(setup_us) / admitted.size();
    f.upload = {}; f.upload.entries = std::move(admitted);
    f.upload_copies.clear();
    for (auto & item : f.upload.entries) {
        auto & slot = g.slots[item.position];
        if (slot.layer >= 0) {
            g.slot_index.erase((uint64_t(slot.layer) << 32) | uint32_t(slot.expert));
            ++f.evictions;
        }
        item.entry.used = ++f.clock;
        for (int bank = 0; bank < 3; ++bank) if (t.banks[bank]) {
            ggml_backend_sched_moe_ffn::upload_copy copy;
            copy.destination = *t.banks[bank]; resident_bank(f, t.group, bank, copy.destination);
            copy.bank = t.banks[bank];
            copy.source = static_cast<const uint8_t *>(copy.bank->data) + size_t(item.entry.expert) * copy.bank->nb[2];
            copy.bytes = copy.bank->nb[2]; copy.offset = item.position * g.strides[bank];
            copy.direct = resident_pinned(copy.bank); // pinned weights need no staging copy
            f.upload.bytes += copy.bytes; f.upload_copies.push_back(copy);
        }
        slot = {}; slot.layer = -2;
    }
    // One reservation phase and one fence cover the whole bounded batch.
    // Old slot readers precede this fence; the uploader cannot overwrite them
    // until it passes, even when admission overlaps the current CPU job.
    resident_mapping(sched, gpu, true);
    if (!early_fence) ggml_backend_event_record(f.upload_fence);
    f.host_bytes += f.upload.bytes;
    f.traffic_remaining -= bytes * f.upload.entries.size();
    ++f.upload_batches; f.batched_experts += f.upload.entries.size();
    if (prediction) f.prediction_uploads += f.upload.entries.size();
    f.upload_queued.store(false, std::memory_order_release);
    f.upload.pending = true;
    // Charge one-time setup to these entries, not to every future upload.
    f.upload.policy_us = ggml_time_us() - admission_started - setup_us;
    {
        std::lock_guard<std::mutex> lock(f.upload_mutex);
        f.upload_job = true;
    }
    f.upload_wake.notify_one();
    return resident_admission::submitted;
}

static void resident_prefetch_prediction(ggml_backend_sched_t sched, ggml_backend_t gpu, int completed_layer = -1,
        const ggml_backend_sched_moe_ffn::task * active = nullptr,
        const std::vector<bool> * active_slots = nullptr, int64_t cpu_started = 0) {
    auto & f = *sched->moe_ffn;
    // Forecast protection expires when the actual route takes over; an active
    // invocation supplies its own slot protection until its reader fence.
    // Upload-queue consumption must not erase the independent accuracy plan.
    for (const auto & c : f.forecasts) if (c.layer <= completed_layer) {
        const size_t i = size_t(c.layer) * sched->moe_resident_n_expert + c.expert;
        if (i < f.forecast_probability.size()) f.forecast_probability[i] = 0;
    }
    if (!f.cache || f.upload.pending || f.in_flight.owner) return;
    int attempts = 0;
    for (auto it = f.forecast_queue.begin(); it != f.forecast_queue.end();) {
        const auto c = *it;
        if (c.layer <= completed_layer || size_t(c.layer) >= f.tasks.size() || !f.tasks[c.layer] ||
                !f.live_task(*f.tasks[c.layer])) { it = f.forecast_queue.erase(it); continue; }
        auto & t = *f.tasks[c.layer];
        auto & g = f.groups[t.group];
        const auto found = g.slot_index.find((uint64_t(c.layer) << 32) | uint32_t(c.expert));
        if (found != g.slot_index.end() && found->second >= 0 && size_t(found->second) < g.slots.size() &&
                ggml_backend_sched_moe_ffn_match(g.slots[found->second], t, c.expert)) {
            it = f.forecast_queue.erase(it); continue;
        }
        // The uploader owns copied descriptors and fences every old reader
        // before writing. Active invocation slots remain protected as well.
        // No new arena, blocking weight upload, or router substitution.
        auto & protected_slots = f.prediction_protected_slots;
        protected_slots.assign(g.slots.size(), false);
        if (active && active_slots && active->group == t.group) protected_slots = *active_slots;
        auto & experts = f.prediction_experts;
        experts.clear();
        for (auto q = it; q != f.forecast_queue.end() && experts.size() < 2; ++q) {
            if (q->layer != c.layer) continue;
            const auto key = (uint64_t(q->layer) << 32) | uint32_t(q->expert);
            if (!g.slot_index.count(key)) experts.push_back(q->expert);
        }
        const auto uploads = f.prediction_uploads;
        double window_us = -1;
        if (active) {
            const size_t cpu_rows = std::count_if(f.cpu_ids.begin(), f.cpu_ids.end(), [](int e) { return e >= 0; });
            // CPU running is overlap opportunity, not evidence of idle RAM or
            // PCIe. Reserve half the estimated remainder for required work;
            // the shared per-pass byte budget independently limits contention.
            window_us = active->upload_fenced ? 0.5 * std::max(0.0,
                    f.cpu_cost[active->layer] * cpu_rows - double(ggml_time_us() - cpu_started)) : 0;
        }
        const auto result = resident_promote(sched, t, experts, protected_slots, gpu, true,
                active && active->upload_fenced, window_us);
        if (result == resident_admission::deferred) return; // keep hints for a later window
        f.forecast_queue.erase(std::remove_if(f.forecast_queue.begin(), f.forecast_queue.end(), [&](const auto & hint) {
            if (hint.layer != c.layer) return false;
            if (result == resident_admission::submitted) return std::any_of(f.upload.entries.begin(), f.upload.entries.end(),
                    [&](const auto & item) { return item.entry.layer == hint.layer && item.entry.expert == hint.expert; });
            return std::find(experts.begin(), experts.end(), hint.expert) != experts.end();
        }), f.forecast_queue.end());
        it = f.forecast_queue.begin(); // rejected hints cannot stall later layers
        if (f.prediction_uploads != uploads || ++attempts >= 2) return;
    }
}

static void resident_record_token_routes(ggml_backend_sched_t sched,
        const ggml_backend_sched_moe_ffn::task & t, const int32_t * ids, int tokens, int k) {
    auto & f = *sched->moe_ffn;
    if (f.route_tokens.size() != size_t(tokens)) return;
    const int experts = sched->moe_resident_n_expert;
    if (experts <= 0 || experts > 1024) return;
    std::array<bool, 1024> selected{};
    for (int i = 0; i < k * tokens; ++i) if (ids[i] >= 0 && ids[i] < experts) selected[ids[i]] = true;
    for (int e = 0; e < experts; ++e) {
        const size_t i = size_t(t.layer) * experts + e;
        if (i < f.forecast_probability.size() && f.forecast_probability[i] >= 0.75) {
            ++f.prediction_selected; f.prediction_correct += selected[e];
        }
    }
    for (int token = 0; token < tokens; ++token) f.observe_token_route(f.route_tokens[token], t.layer,
            ids + token * k, k, sched->moe_resident_layers, experts);
}

// Opt-in next-layer prefetch ring. Ordering contract (one compute stream, host
// thread only): a retired entry's last readers (its layer's GPU FFN and any
// promotion copy) are queued before the ring_fence record that every forecast
// upload waits on before overwriting an entry. An installed entry makes the
// compute stream wait on its upload event, so a late upload delays only the
// GPU half of its layer, which runs alongside that layer's CPU experts.
static bool resident_position_valid(const ggml_backend_sched_moe_ffn & f, size_t group,
        const ggml_backend_sched_moe_ffn::task & t, int expert, int position) {
    const auto & g = f.groups[group];
    if (size_t(position) < g.slots.size()) return ggml_backend_sched_moe_ffn_match(g.slots[position], t, expert);
    const size_t i = size_t(position) - g.slots.size();
    if (group >= f.rings.size() || i >= f.rings[group].size()) return false;
    const auto & r = f.rings[group][i];
    if (!r.installed || r.layer != t.layer || r.expert != expert) return false;
    for (int b = 0; b < 3; ++b) if (r.data[b] != (t.banks[b] ? t.banks[b]->data : nullptr)) return false;
    return true;
}

// A used forecast becomes a cache entry by a device copy: no PCIe or host RAM traffic.
static void resident_ring_promote(ggml_backend_sched_t sched, size_t group, size_t index, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    auto & g = f.groups[group];
    const auto & r = f.rings[group][index];
    if (size_t(r.layer) >= f.tasks.size() || !f.tasks[r.layer] || f.tasks[r.layer]->group != group) return;
    const auto & t = *f.tasks[r.layer];
    const uint64_t key = (uint64_t(r.layer) << 32) | uint32_t(r.expert);
    if (g.slot_index.count(key)) return;
    for (int b = 0; b < 3; ++b) if (r.data[b] != (t.banks[b] ? t.banks[b]->data : nullptr)) return;
    const int experts = sched->moe_resident_n_expert;
    size_t bytes = 0; for (size_t stride : g.strides) bytes += stride;
    if (!bytes || g.slots.empty()) return;
    const auto value = [&](int layer, int expert) { return f.reuse_value(layer, expert, experts, bytes); };
    resident_refresh_victims(f, g, f.policy_clock / f.history_layers, value);
    const double candidate = value(r.layer, r.expert);
    size_t victim = g.slots.size();
    for (auto it = g.victims.begin(); it != g.victims.end(); ++it) {
        const size_t pos = it->second;
        if (pos >= g.slots.size()) continue;
        const auto & s = g.slots[pos];
        if (s.layer == -2 || f.slot_protected(s)) continue;
        if (s.layer >= 0) {
            const size_t id = size_t(s.layer) * experts + s.expert;
            if (id < f.forecast_probability.size() && f.forecast_probability[id] >= 0.75) continue;
            if (candidate <= 1.25 * value(s.layer, s.expert)) break; // shortlist is ascending
        }
        victim = pos;
        g.victims.erase(it);
        break;
    }
    if (victim == g.slots.size()) return;
    const size_t source = g.slots.size() + index;
    for (int b = 0; b < 3; ++b) if (g.strides[b] && t.banks[b]) {
        auto * buffer = f.virtual_cache ? g.storage[b] : f.cache;
        const size_t base = f.virtual_cache ? 0 : g.offsets[b];
        ggml_backend_cuda_moe_cache_move(gpu, buffer, base + victim * g.strides[b], base + source * g.strides[b], g.strides[b]);
        f.d2d_bytes += g.strides[b];
    }
    auto & s = g.slots[victim];
    if (s.layer >= 0) { g.slot_index.erase((uint64_t(s.layer) << 32) | uint32_t(s.expert)); ++f.evictions; }
    s = {};
    s.layer = r.layer; s.expert = r.expert; s.used = ++f.clock;
    s.admitted_at = f.policy_clock;
    s.admission_cost_us = double(bytes) / 3e5; // device copy at roughly 300 GB/s
    for (int b = 0; b < 3; ++b) { s.banks[b] = t.banks[b]; s.data[b] = r.data[b]; }
    g.slot_index[key] = int(victim);
    ++f.ring_promotions; ++f.promotions;
    f.mapping_dirty = true;
}

// At each layer: install this layer's completed forecasts, retire all others.
static void resident_ring_service(ggml_backend_sched_t sched, int layer, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    for (size_t gi = 0; gi < f.rings.size(); ++gi) for (size_t i = 0; i < f.rings[gi].size(); ++i) {
        auto & r = f.rings[gi][i];
        if (r.layer < 0) continue;
        if (r.layer == layer) {
            if (!r.installed && r.ready) {
                if (!ggml_backend_cuda_moe_event_ready(r.ready)) { ggml_backend_event_wait(gpu, r.ready); ++f.ring_late; }
                r.installed = true; ++f.ring_installed; f.mapping_dirty = true;
            }
            continue;
        }
        if (r.installed && r.used) resident_ring_promote(sched, gi, i, gpu);
        else ++f.ring_unused;
        if (r.installed) f.mapping_dirty = true;
        r.layer = r.expert = -1; r.installed = r.used = false;
    }
}

// Upload up to N predicted, non-resident experts of the next layer from pinned
// host weights. Runs on its own stream during this layer's CPU work.
static void resident_ring_prefetch(ggml_backend_sched_t sched, const ggml_backend_sched_moe_ffn::task & t,
        ggml_backend_t gpu, const int32_t * forecast, int count) {
    auto & f = *sched->moe_ffn;
    const int next = t.layer + 1;
    if (f.rings.empty() || next >= int(f.tasks.size()) || !f.tasks[next] || !f.live_task(*f.tasks[next])) return;
    const auto & nt = *f.tasks[next];
    for (auto * bank : nt.banks) if (bank && !resident_pinned(bank)) { ++f.ring_skipped; return; }
    auto & g = f.groups[nt.group];
    auto & ring = f.rings[nt.group];
    if (!f.prefetcher) {
        f.prefetcher = ggml_backend_cuda_init(ggml_backend_cuda_get_device(gpu), nullptr, &f);
        if (!f.prefetcher) return;
    }
    if (!f.ring_fence && !(f.ring_fence = ggml_backend_event_new(gpu))) return;
    const int experts = sched->moe_resident_n_expert;
    bool fenced = false;
    int issued = 0;
    for (int i = 0; i < count && issued < sched->moe_resident_prefetch; ++i) {
        const int e = forecast[i];
        if (e < 0 || e >= experts || f.mapping[size_t(next) * experts + e] >= 0) continue;
        if (std::any_of(ring.begin(), ring.end(), [&](const auto & r) { return r.layer == next && r.expert == e; })) continue;
        const auto entry = std::find_if(ring.begin(), ring.end(), [](const auto & r) { return r.layer < 0; });
        if (entry == ring.end()) break;
        if (!entry->ready && !(entry->ready = ggml_backend_event_new(f.prefetcher))) break;
        if (!fenced) {
            if (!f.ring_fence_ready) ggml_backend_event_record(f.ring_fence);
            ggml_backend_event_wait(f.prefetcher, f.ring_fence);
            fenced = true;
        }
        const size_t position = g.slots.size() + size_t(entry - ring.begin());
        for (int b = 0; b < 3; ++b) {
            auto * bank = nt.banks[b];
            entry->data[b] = bank ? bank->data : nullptr;
            if (!bank) continue;
            auto destination = *bank;
            resident_bank(f, nt.group, b, destination);
            ggml_backend_tensor_set_async(f.prefetcher, &destination,
                    static_cast<const uint8_t *>(bank->data) + size_t(e) * bank->nb[2], position * g.strides[b], bank->nb[2]);
            f.ring_bytes += bank->nb[2];
        }
        ggml_backend_event_record(entry->ready);
        entry->layer = next; entry->expert = e; entry->installed = entry->used = false;
        ++issued; ++f.ring_uploads;
    }
    f.ring_fence_ready = false;
}

// Device join: one pinned word pair per scheduler, probed once with a gate that
// passes at once. False keeps the host join (unsupported GPU or allocation failure).
static bool resident_gate_ready(ggml_backend_sched_t sched, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    if (!sched->moe_resident_device_join) return false;
    if (f.gate_words) return true;
    if (f.gate_buffer) return false; // probe failed earlier
    f.gate_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 64);
    if (!f.gate_buffer) { sched->moe_resident_device_join = false; return false; }
    auto * words = static_cast<uint32_t *>(ggml_backend_buffer_get_base(f.gate_buffer));
    words[0] = words[1] = 0;
    if (!ggml_backend_cuda_moe_gate(gpu, &words[0], 0, &words[1])) {
        fprintf(stderr, "ggml_backend_sched: moe-resident device join unsupported on this GPU; using host join\n");
        return false;
    }
    f.gate_words = words; f.gate_value = 0;
    return true;
}

// Device join publication: queue this layer's CPU-result upload behind a GPU
// wait on the worker's release. The host continues with later splits. The job
// is settled (joined, measured, admitted) at the next point needing the CPU.
static void resident_publish_gated(ggml_backend_sched_moe_ffn & f, ggml_backend_sched_moe_ffn::task & t,
        ggml_backend_t gpu, size_t gpu_count) {
    auto & call = f.in_flight;
    ggml_backend_cuda_moe_gate(gpu, &f.gate_words[0], t.gate_value, &f.gate_words[1]);
    if (gpu_count) {
        const size_t bytes = t.pack_rows.size() * t.host_down.nb[1];
        t.device_cpu = *t.down;
        ggml_backend_sched_moe_ffn_bind(t.device_cpu, f.device_cpu);
        ggml_backend_tensor_set_async(gpu, &t.device_cpu, t.pack_dst, 0, bytes);
        ggml_backend_cuda_moe_scatter(gpu, t.down, &t.device_cpu, &t.device_ids);
        f.result_bytes += bytes; f.packed_result_bytes += bytes;
    } else {
        ggml_backend_tensor_set_async(gpu, t.down, t.host_down.data, 0, ggml_nbytes(&t.host_down));
        f.result_bytes += ggml_nbytes(&t.host_down);
    }
    // Retired ring entries' readers are all queued now. A forecast upload issued
    // at settle time must not wait for the next layer's attention as well.
    if (call.cache && call.forecast && !f.rings.empty() &&
            (f.ring_fence || (f.ring_fence = ggml_backend_event_new(gpu)))) {
        ggml_backend_event_record(f.ring_fence);
        f.ring_fence_ready = true;
    }
    call.published = true; call.published_us = ggml_time_us();
    ++f.gated_layers;
}

static ggml_status resident_finish_ffn(ggml_backend_sched_t sched) {
    auto & f = *sched->moe_ffn;
    auto & call = f.in_flight;
    if (!call.owner) return GGML_STATUS_SUCCESS;
    auto & t = *call.owner;
    auto * gpu = call.gpu;
    const int tokens = call.tokens, k = call.k;
    const size_t cpu_count = call.cpu_count, gpu_count = call.gpu_count;
    const int64_t cpu_started = call.cpu_started, host_started = call.host_started;
    auto status = call.status;
    const bool cache = call.cache;
    const auto & missing = call.missing;
    const auto & protected_slots = call.protected_slots;
    if (call.gated && !call.published) {
        resident_publish_gated(f, t, gpu, gpu_count);
        return GGML_STATUS_SUCCESS;
    }
    if (cpu_count || !gpu_count) {
        const int64_t join_started = ggml_time_us();
        const auto cpu_status = f.finish();
        // Device join: the GPU, not the host, waited from publication to CPU completion.
        const int64_t exposed = std::max<int64_t>(1, call.gated
                ? f.cpu_finished_us - call.published_us : ggml_time_us() - join_started);
        if (t.allow_defer) f.deferred_join_us += exposed;
        // Independent GPU work may outlast this job. Admission must use CPU
        // execution time, not the later time at which its consumer joins it.
        const int64_t elapsed = std::max<int64_t>(1, f.cpu_finished_us - cpu_started);
        f.cpu_wait_us += elapsed;
        f.observe_cpu_cost(t.layer, tokens, cpu_count, double(elapsed), double(exposed),
                sched->moe_resident_n_expert_used);
        if (status == GGML_STATUS_SUCCESS) status = cpu_status;
        if (call.gated) {
            if (status == GGML_STATUS_SUCCESS && __atomic_load_n(&f.gate_words[1], __ATOMIC_ACQUIRE)) {
                fprintf(stderr, "ggml_backend_sched: moe-resident device join timed out\n");
                status = GGML_STATUS_FAILED;
            }
        } else if (status == GGML_STATUS_SUCCESS && gpu_count) {
            if (!ggml_backend_sched_moe_decode_buffer(f.packed_output,
                    ggml_backend_cuda_host_buffer_type(), ggml_nbytes(&t.host_down))) {
                ggml_backend_synchronize(gpu); call.owner = nullptr; return GGML_STATUS_ALLOC_FAILED;
            }
            auto * packed = static_cast<uint8_t *>(ggml_backend_buffer_get_base(f.packed_output));
            size_t bytes = 0;
            for (int row = 0; row < k; ++row) if (f.gpu_ids[row] < 0) {
                std::memcpy(packed + bytes, static_cast<const uint8_t *>(t.host_down.data) + row * t.host_down.nb[1],
                        t.host_down.nb[1]);
                bytes += t.host_down.nb[1];
            }
            t.device_cpu = *t.down;
            ggml_backend_sched_moe_ffn_bind(t.device_cpu, f.device_cpu);
            ggml_backend_tensor_set_async(gpu, &t.device_cpu, packed, 0, bytes);
            ggml_backend_cuda_moe_scatter(gpu, t.down, &t.device_cpu, &t.device_ids);
            f.result_bytes += bytes; f.packed_result_bytes += bytes;
        } else if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_set_async(gpu, t.down, t.host_down.data, 0, ggml_nbytes(&t.host_down));
            f.result_bytes += ggml_nbytes(&t.host_down);
        }
    } else {
        ++f.all_hit_layers;
    }
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_synchronize(gpu); call.owner = nullptr; return status;
    }
    call.owner = nullptr; // result is queued; future uploads use the stream fence
    // Start next-layer forecast DMA after the CPU join: it then overlaps the GPU
    // attention gap instead of competing with CPU experts for host DRAM.
    if (cache && call.forecast)
        resident_ring_prefetch(sched, t, gpu, f.prediction_experts.data(), int(f.prediction_experts.size()));
    f.ring_fence_ready = false; // a publication fence covers only its own layer
    const int64_t admission_started = ggml_time_us();
    if (cache) {
        if (f.rings.empty()) resident_prefetch_prediction(sched, gpu, t.layer);
        resident_promote(sched, t, missing, protected_slots, gpu);
    }
    f.admission_us += ggml_time_us() - admission_started;
    // Covers both result DMA and any lookup DMA queued by admission. Recording
    // before admission would allow its reusable host map to be overwritten.
    ggml_backend_event_record(f.completion); f.pending = true;
    ++f.layers; f.cpu_refs += cpu_count; f.gpu_refs += gpu_count;
    f.token_layers += tokens;
    f.batch2_layers += tokens == 2;
    f.batch3_layers += tokens == 3;
    f.batch4_layers += tokens == 4;
    f.host_layer_us += ggml_time_us() - host_started;
    if (f.layers % (64ull * sched->moe_resident_layers) == 0) f.report_new("periodic");
    call.owner = nullptr;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_sched_moe_new_compute(ggml_backend_sched_t sched,
        ggml_backend_sched_moe_ffn::task & t, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    GGML_ASSERT(!f.in_flight.owner);
    const int64_t host_started = ggml_time_us();
    t.upload_fenced = false;
    const bool cache = ggml_backend_sched_moe_new_cache(sched, gpu);
    auto & g = f.groups[t.group];
    const int tokens = int(t.up->src[3]->ne[1]);
    const int k = int(ggml_nelements(t.up->src[3])), experts = sched->moe_resident_n_expert;
    resident_history(sched);
    // Every eligible layer advances age, including GPU-only complete layers.
    // Changing cache coverage must not change the history's decay rate.
    f.policy_clock += tokens;
    if (!f.router_ready) f.router_ready = ggml_backend_event_new(gpu);
    if (!f.router_ready) return GGML_STATUS_FAILED;
    if (f.upload.pending && f.upload_queued.load(std::memory_order_acquire) &&
            ggml_backend_cuda_moe_event_ready(f.upload_ready)) {
        f.publish_upload();
    }
    if (cache && !f.rings.empty()) resident_ring_service(sched, t.layer, gpu);
    if (cache && f.mapping_dirty) {
        const int64_t started = ggml_time_us();
        f.wait_io(); // mapping is a reusable asynchronous DMA source
        f.mapping_wait_us += ggml_time_us() - started;
        resident_mapping(sched, gpu, true);
    }
    // A rebuilt graph can contain different bank descriptors. Never reuse a
    // mapping whose source weight identity no longer matches the current task.
    bool complete = cache, has_resident = false;
    std::vector<int> stale;
    for (int e = 0; e < experts; ++e) {
        const int position = f.mapping[size_t(t.layer) * experts + e];
        if (position >= 0 && !resident_position_valid(f, t.group, t, e, position)) {
            stale.push_back(e);
            complete = false;
            continue;
        }
        complete &= position >= 0;
        has_resident |= cache && position >= 0;
    }
    if (!stale.empty() && cache) {
        // Do not mutate mapping's host DMA source until the previous upload
        // is finished. The common matching case adds no completion wait.
        const int64_t started = ggml_time_us();
        f.wait_io();
        f.mapping_wait_us += ggml_time_us() - started;
        for (int e : stale) {
            const int position = f.mapping[size_t(t.layer) * experts + e];
            if (size_t(position) < g.slots.size()) {
                g.slot_index.erase((uint64_t(t.layer) << 32) | uint32_t(e));
                g.slots[position] = {};
            } else if (t.group < f.rings.size() && size_t(position) - g.slots.size() < f.rings[t.group].size()) {
                auto & r = f.rings[t.group][size_t(position) - g.slots.size()];
                r.layer = r.expert = -1; r.installed = r.used = false;
            }
        }
        resident_mapping(sched, gpu, true);
    }
    t.device_ids = *t.up->src[3];
    ggml_backend_sched_moe_ffn_bind(t.device_ids, f.device_ids);
    const auto * device_map = cache ? static_cast<const int32_t *>(ggml_backend_buffer_get_base(f.slot_map)) +
            size_t(t.layer) * experts : nullptr;
    if (complete) {
        // A fully resident layer needs neither host routing nor activation I/O.
        ggml_backend_cuda_moe_route(gpu, t.up->src[3], device_map, &t.device_ids, experts);
        const int64_t started = ggml_time_us();
        const auto status = resident_gpu_ffn(f, t, gpu);
        f.gpu_enqueue_us += ggml_time_us() - started;
        if (status != GGML_STATUS_SUCCESS) { ggml_backend_synchronize(gpu); return status; }
        if (f.histogram) {
            auto * counts = static_cast<uint32_t *>(ggml_backend_buffer_get_base(f.histogram));
            ggml_backend_cuda_moe_histogram(gpu, t.up->src[3], counts + size_t(t.layer) * experts, experts);
            f.histogram_dirty = true;
        }
        for (auto & slot : g.slots) if (slot.layer == t.layer) slot.used = ++f.clock;
        ggml_backend_event_record(f.completion); f.pending = true;
        ++f.layers; ++f.all_hit_layers; f.hits += k; f.gpu_refs += k;
        f.token_layers += tokens;
        f.batch2_layers += tokens == 2;
        f.batch3_layers += tokens == 3;
        f.batch4_layers += tokens == 4;
        t.route_gpu.assign(size_t(k), 0); // all rows use the GPU reference path
        ++f.mapped_routes;
        f.host_layer_us += ggml_time_us() - host_started;
        if (f.layers % (64ull * sched->moe_resident_layers) == 0) f.report_new("periodic");
        return GGML_STATUS_SUCCESS;
    }
    t.host_act = *t.up->src[2]; t.host_ids = *t.up->src[3];
    t.host_up = *t.up; t.host_down = *t.down;
    ggml_backend_sched_moe_ffn_bind(t.host_act, f.activation);
    ggml_backend_sched_moe_ffn_bind(t.host_ids, f.ids);
    ggml_backend_sched_moe_ffn_bind(t.host_up, f.intermediate);
    ggml_backend_sched_moe_ffn_bind(t.host_down, f.output);
    auto * ids = static_cast<int32_t *>(t.host_ids.data);
    // Partial layers need the route on CPU regardless. Let the copy engine
    // transfer the small pinned inputs, rather than a kernel writing host RAM.
    ggml_backend_tensor_get_async(gpu, t.up->src[3], ids, 0, ggml_nbytes(&t.host_ids));
    ggml_backend_tensor_get_async(gpu, t.up->src[2], t.host_act.data, 0, ggml_nbytes(&t.host_act));
    const bool forecast = cache && !f.rings.empty() && t.pregate && t.pregate->buffer &&
            !ggml_backend_buffer_is_host(t.pregate->buffer) &&
            ggml_backend_sched_moe_decode_buffer(f.pregate_host, ggml_backend_cuda_host_buffer_type(), ggml_nbytes(t.pregate));
    if (forecast) ggml_backend_tensor_get_async(gpu, t.pregate, ggml_backend_buffer_get_base(f.pregate_host), 0, ggml_nbytes(t.pregate));
    ++f.fallback_routes;
    ggml_backend_event_record(f.router_ready);
    if (!f.upload.pending && f.upload_fence) {
        // All previous readers precede input readiness. Current GPU hits
        // are protected explicitly, so optional writes to other slots need
        // not wait behind the entire current GPU FFN. Never re-record a
        // fence still owned by a pending uploader job.
        ggml_backend_event_record(f.upload_fence);
        t.upload_fenced = true;
    }
    // The readback event deliberately precedes cached-expert execution. Queue
    // GPU hits using the device lookup before blocking the host: CPU input
    // readiness must not wait for those hits to finish. Neither the lookup nor
    // its slots can be replaced until resident_finish_ffn() fences this work.
    auto status = GGML_STATUS_SUCCESS;
    if (has_resident) {
        const int64_t started = ggml_time_us();
        ggml_backend_cuda_moe_route(gpu, t.up->src[3], device_map, &t.device_ids, experts);
        status = resident_gpu_ffn(f, t, gpu);
        f.gpu_enqueue_us += ggml_time_us() - started;
        if (status != GGML_STATUS_SUCCESS) { ggml_backend_synchronize(gpu); return status; }
        ++f.early_gpu_layers;
    }
    if (!has_resident) ++f.empty_gpu_layers;
    const int64_t router_started = ggml_time_us();
    ggml_backend_event_synchronize(f.router_ready);
    f.router_wait_us += ggml_time_us() - router_started;
    f.pending = false; // the prior result's DMA precedes this routing event
    t.route_experts.assign(ids, ids + k);
    f.cpu_ids.assign(ids, ids + k); f.gpu_ids.assign(size_t(k), -1);
    std::vector<bool> protected_slots(g.slots.size(), false);
    std::vector<int> missing;
    size_t cpu_count = 0, gpu_count = 0;
    for (int row = 0; row < k; ++row) {
        const int e = ids[row];
        if (e < 0 || e >= experts) { f.cpu_ids[row] = -1; continue; }
        const size_t i = size_t(t.layer) * experts + e;
        const int p = cache ? f.mapping[i] : -1;
        if (p >= 0 && size_t(p) >= g.slots.size()) {
            // Forecast hit in the ring; the entry is retired after this layer.
            f.gpu_ids[row] = p; f.cpu_ids[row] = -1;
            f.rings[t.group][size_t(p) - g.slots.size()].used = true;
            ++gpu_count; ++f.hits; ++f.ring_hits;
        } else if (p >= 0) {
            f.gpu_ids[row] = p; f.cpu_ids[row] = -1; protected_slots[p] = true;
            if (g.slots[p].predicted) { ++f.prediction_used; g.slots[p].predicted = false; }
            g.slots[p].used = ++f.clock;
            g.slots[p].saved_us += f.cpu_marginal_cost[t.layer];
            ++gpu_count; ++f.hits;
        } else {
            ++cpu_count;
            if (std::find(missing.begin(), missing.end(), e) == missing.end()) missing.push_back(e);
        }
    }
    t.route_gpu = f.gpu_ids;
    f.activation_bytes += ggml_nbytes(&t.host_act);
    if (!gpu_count) ++f.cpu_only_layers;
    if (has_resident && !gpu_count) ++f.early_gpu_empty_layers;
    const bool cpu_work = cpu_count || !gpu_count;
    const int64_t cpu_started = ggml_time_us();
    if (cpu_work) {
        std::memcpy(ids, f.cpu_ids.data(), size_t(k) * sizeof(int32_t));
        t.host_up.src[0] = t.banks[0]; t.host_up.src[1] = t.banks[1];
        t.host_up.src[2] = &t.host_act; t.host_up.src[3] = &t.host_ids;
        t.host_down.src[0] = t.banks[2]; t.host_down.src[1] = &t.host_up; t.host_down.src[2] = &t.host_ids;
        // One exact active-expert list for the whole MTP verification batch.
        // Both projections share it; no forecast ID participates in CPU math.
        f.cpu_experts.clear();
        for (const int e : f.cpu_ids) if (e >= 0) f.cpu_experts.push_back(e);
        std::sort(f.cpu_experts.begin(), f.cpu_experts.end());
        f.cpu_experts.erase(std::unique(f.cpu_experts.begin(), f.cpu_experts.end()), f.cpu_experts.end());
        t.host_experts = {};
        t.host_experts.type = GGML_TYPE_I32;
        t.host_experts.ne[0] = int64_t(f.cpu_experts.size());
        for (int d = 1; d < GGML_MAX_DIMS; ++d) t.host_experts.ne[d] = 1;
        t.host_experts.nb[0] = sizeof(int32_t);
        for (int d = 1; d < GGML_MAX_DIMS; ++d)
            t.host_experts.nb[d] = t.host_experts.nb[d - 1] * t.host_experts.ne[d - 1];
        t.host_experts.data = f.cpu_experts.data();
        t.host_up.src[6] = &t.host_experts; t.host_down.src[3] = &t.host_experts;
        t.host_up.op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_MOE_RESIDENT_CPU_ROUTES;
        t.host_down.op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = GGML_MOE_RESIDENT_CPU_ROUTES;
        f.cpu_batch_rows += cpu_count; f.cpu_batch_experts += f.cpu_experts.size();
        f.cpu_masked_quant_rows += k - cpu_count;
        t.host_nodes[0] = &t.host_up; t.host_nodes[1] = &t.host_down;
        t.host_graph = {}; t.host_graph.nodes = t.host_nodes; t.host_graph.n_nodes = 2;
        f.cpu = sched->backends[1];
        t.gate_value = 0; t.pack_rows.clear(); t.pack_dst = nullptr;
        if (status == GGML_STATUS_SUCCESS && resident_gate_ready(sched, gpu) &&
                (!gpu_count || ggml_backend_sched_moe_decode_buffer(f.packed_output,
                        ggml_backend_cuda_host_buffer_type(), ggml_nbytes(&t.host_down)))) {
            // All-CPU layers upload host_down (pinned) as is; partial layers upload packed CPU rows.
            if (gpu_count) {
                for (int row = 0; row < k; ++row) if (f.gpu_ids[row] < 0) t.pack_rows.push_back(row);
                t.pack_dst = static_cast<uint8_t *>(ggml_backend_buffer_get_base(f.packed_output));
            }
            if (++f.gate_value == 0) ++f.gate_value; // 0 means host join
            t.gate_value = f.gate_value;
        }
        f.submit(t);
    }
    // Policy learning is not an input to this invocation's arithmetic. Run it
    // after submitting misses so its scans/allocations overlap the CPU job.
    // ids now contains the CPU mask; the task retains the actual original IDs.
    resident_record_token_routes(sched, t, t.route_experts.data(), tokens, k / tokens);
    for (int row = 0; row < k; ++row) {
        const int e = t.route_experts[row];
        if (e < 0 || e >= experts) continue;
        const size_t i = size_t(t.layer) * experts + e;
        f.popularity[i] = f.session_score(t.layer, e, experts) + 1;
        f.recent_popularity[i] = f.recent_score(t.layer, e, experts) + 1;
        f.observed_at[i] = f.policy_clock;
    }
    // Submit only optional future-layer replacement here. Its old-reader
    // fence precedes current GPU hits, which remain protected; DMA may overlap this CPU job
    // and subsequently queued shared work. Current hits remain protected.
    if (forecast) {
        // Rank-major across verification tokens: every token's best guess first.
        const auto * predicted = static_cast<const int32_t *>(ggml_backend_buffer_get_base(f.pregate_host));
        const int ranks = int(t.pregate->ne[0]), rows = int(t.pregate->ne[1]);
        f.prediction_experts.clear();
        for (int r = 0; r < ranks; ++r) for (int j = 0; j < rows; ++j) f.prediction_experts.push_back(predicted[j * ranks + r]);
    } else if (cpu_work && cache) {
        resident_prefetch_prediction(sched, gpu, t.layer, &t, &protected_slots, cpu_started);
    }
    auto & call = f.in_flight;
    call.owner = &t; call.gpu = gpu; call.tokens = tokens; call.k = k;
    call.cpu_count = cpu_count; call.gpu_count = gpu_count;
    call.cpu_started = cpu_started; call.host_started = host_started; call.status = status;
    call.cache = cache; call.forecast = forecast;
    call.gated = cpu_work && t.gate_value != 0; call.published = false;
    call.missing = std::move(missing); call.protected_slots = std::move(protected_slots);
    if (status == GGML_STATUS_SUCCESS && cpu_work && t.allow_defer) {
        ++f.deferred_layers;
        return GGML_STATUS_SUCCESS; // publish at the marked first dependent split
    }
    return resident_finish_ffn(sched);
}
#else
static ggml_status resident_finish_ffn(ggml_backend_sched_t) { return GGML_STATUS_SUCCESS; }
static void ggml_backend_sched_moe_new_collect(ggml_backend_sched_t) {}
static void ggml_backend_sched_moe_new_observe(ggml_backend_sched_t, const ggml_tensor *, ggml_backend_t) {}
static bool ggml_backend_sched_moe_new_cache(ggml_backend_sched_t, ggml_backend_t, bool) { return false; }
static ggml_status ggml_backend_sched_moe_new_compute(ggml_backend_sched_t,
        ggml_backend_sched_moe_ffn::task &, ggml_backend_t) { return GGML_STATUS_FAILED; }
#endif
