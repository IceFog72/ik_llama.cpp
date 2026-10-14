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
        f.observed_at.assign(n, f.policy_clock);
        f.mapping.assign(n, -1);
        f.cpu_cost.assign(size_t(sched->moe_resident_layers), 100.0);
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
            f.popularity[i] = f.score(layer, e, experts) * 0.5 + counts[i] * scale;
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
            f.popularity[i] = f.score(found->second, values[j], experts) + scale;
            f.observed_at[i] = f.policy_clock;
        }
    }
}

static void resident_bank(ggml_backend_sched_moe_ffn & f, size_t group, int bank,
        ggml_tensor & tensor) {
    auto & g = f.groups[group];
    ggml_backend_sched_moe_ffn_bind(tensor, f.virtual_cache ? g.storage[bank] : f.cache);
    if (!f.virtual_cache) tensor.data = static_cast<uint8_t *>(tensor.data) + g.offsets[bank];
    tensor.ne[2] = int64_t(g.slots.size());
    tensor.nb[2] = g.strides[bank]; tensor.nb[3] = tensor.nb[2] * g.slots.size();
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

static void resident_mapping(ggml_backend_sched_t sched, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    std::fill(f.mapping.begin(), f.mapping.end(), -1);
    for (auto & g : f.groups) for (size_t p = 0; p < g.slots.size(); ++p) {
        const auto & s = g.slots[p];
        if (s.layer >= 0 && s.expert >= 0)
            f.mapping[size_t(s.layer) * sched->moe_resident_n_expert + s.expert] = int32_t(p);
    }
    auto map = resident_vector(f.slot_map, GGML_TYPE_I32, f.mapping.size());
    ggml_backend_tensor_set_async(gpu, &map, f.mapping.data(), 0, f.mapping.size() * sizeof(int32_t));
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
            const double score = f.score(t->layer, e, sched->moe_resident_n_expert);
            if (score > 0) candidates.push_back({t->layer, e, t->group, score * f.cpu_cost[t->layer] / bytes});
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
    if (f.cache && !resize) return true;
    if (f.disabled) return false;
    f.finish(); f.wait_io(); f.finish_upload();
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
    if (!ggml_backend_sched_moe_decode_buffer(f.slot_map, sched->bufts[0], f.mapping.size() * sizeof(int32_t))) return false;
    size_t free_bytes = 0, total = 0;
    ggml_backend_cuda_get_device_memory(device, &free_bytes, &total);
    f.budget_bytes = ggml_backend_sched_moe_ffn_budget(sched, free_bytes + f.allocation_bytes, f.reserve_bytes);
    const size_t gran = ggml_backend_cuda_moe_cache_granularity(device);
    size_t padding = ggml_backend_buffer_get_size(f.slot_map) +
            (f.histogram ? ggml_backend_buffer_get_size(f.histogram) : 0);
    for (const auto & g : f.groups) for (size_t stride : g.strides) if (stride) padding += gran;
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
        if (f.allocation_bytes <= f.budget_bytes) return true;
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
                        !ggml_backend_cuda_moe_cache_resize(g.storage[b], counts[gi] * g.strides[b])) {
                    f.clear_cache(); f.disabled = true; return false;
                }
            }
        }
        size_t physical = 0;
        for (size_t gi = 0; gi < f.groups.size(); ++gi) {
            auto & g = f.groups[gi];
            const size_t capacity = size_t(sched->moe_resident_layers) * sched->moe_resident_n_expert;
            for (int b = 0; b < 3; ++b) if (g.strides[b]) {
                if (!g.storage[b]) g.storage[b] = ggml_backend_cuda_moe_cache_new(device, capacity * g.strides[b]);
                if (!g.storage[b] || !ggml_backend_cuda_moe_cache_resize(g.storage[b], counts[gi] * g.strides[b])) {
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
            for (int b = 0; b < 3; ++b) { g.offsets[b] = offset; offset += counts[gi] * g.strides[b]; }
        }
        f.cache = ggml_backend_buft_alloc_buffer(sched->bufts[0], logical_bytes);
        if (!f.cache) { f.clear_cache(); f.disabled = true; return false; }
        ggml_backend_buffer_clear(f.cache, 0);
        f.allocation_bytes = logical_bytes;
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
    fprintf(stderr, "ggml_backend_sched: moe-resident new cache=%zu MiB slots=%zu stable=%s history=%zu\n",
            f.allocation_bytes / 1048576, f.slot_count(), f.virtual_cache ? "yes" : "no", candidates.size());
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

static void resident_promote(ggml_backend_sched_t sched, ggml_backend_sched_moe_ffn::task & t,
        const std::vector<int> & missing, const std::vector<bool> & protected_slots, ggml_backend_t gpu,
        bool prediction = false) {
    auto & f = *sched->moe_ffn;
    if (!f.cache || f.upload.pending || missing.empty()) return;
    auto & g = f.groups[t.group];
    const int n_expert = sched->moe_resident_n_expert;
    size_t bytes = 0; for (size_t stride : g.strides) bytes += stride;
    const auto value = [&](int layer, int expert) {
        if (layer < 0) return 0.0;
        const size_t i = size_t(layer) * n_expert + expert;
        const double forecast = i < f.forecast_probability.size() ? f.forecast_probability[i] : 0;
        return (f.score(layer, expert, n_expert) + 32 * forecast) * f.cpu_cost[layer] / bytes;
    };
    const int expert = *std::max_element(missing.begin(), missing.end(), [&](int a, int b) {
        return value(t.layer, a) < value(t.layer, b);
    });
    size_t victim = g.slots.size();
    double weakest = std::numeric_limits<double>::infinity();
    // Rank once per approximate model sweep, not an O(cache-size) exponential
    // scan for every layer. Recheck a bounded shortlist against current use.
    const uint64_t epoch = f.policy_clock / f.history_layers;
    if (g.victims.empty() || g.victim_epoch != epoch) {
        g.victims.clear();
        for (size_t p = 0; p < g.slots.size(); ++p) if (g.slots[p].layer != -2)
            g.victims.emplace_back(value(g.slots[p].layer, g.slots[p].expert), p);
        std::sort(g.victims.begin(), g.victims.end());
        g.victim_epoch = epoch;
    }
    const size_t shortlist = std::min<size_t>(64, g.victims.size());
    for (size_t i = 0; i < shortlist; ++i) {
        const size_t p = g.victims[i].second;
        if (p >= g.slots.size() || protected_slots[p] || g.slots[p].layer == -2) continue;
        if (g.slots[p].layer >= 0) {
            const size_t i = size_t(g.slots[p].layer) * n_expert + g.slots[p].expert;
            if (i < f.forecast_probability.size() && f.forecast_probability[i] >= 0.75) continue;
        }
        const double score = value(g.slots[p].layer, g.slots[p].expert);
        if (score < weakest) { weakest = score; victim = p; }
    }
    // Require repeated use and amortize a conservative upload estimate before
    // replacing a live expert. Empty slots do not evict a useful working set.
    const bool history_allows = prediction || victim == g.slots.size() || g.slots[victim].layer < 0 ||
            f.score(t.layer, expert, n_expert) >=
                    2 * f.score(g.slots[victim].layer, g.slots[victim].expert, n_expert) + 2;
    if (victim == g.slots.size() || (g.slots[victim].layer >= 0 &&
            (!history_allows || value(t.layer, expert) <=
                    weakest * (prediction ? 1.1 : 2.0) + f.upload_us_per_byte))) {
        ++f.admission_rejected; return;
    }
    if (!f.uploader) {
        // Distinct registry identity: do not replace the model's compute context.
        f.uploader = ggml_backend_cuda_init(ggml_backend_cuda_get_device(gpu), nullptr, &f);
        if (!f.uploader) return;
        f.upload_started = ggml_backend_cuda_moe_timed_event(f.uploader);
        f.upload_ready = ggml_backend_cuda_moe_timed_event(f.uploader);
        f.upload_fence = ggml_backend_event_new(gpu);
        if (!f.upload_ready || !f.upload_fence) return;
    }
    if (!f.upload_ready || !f.upload_fence) return;
    size_t staging_bytes = 0;
    for (auto * bank : t.banks) if (bank) staging_bytes += bank->nb[2];
    if (!ggml_backend_sched_moe_decode_buffer(f.upload_staging,
            ggml_backend_cuda_host_buffer_type(), staging_bytes)) return;
    f.ensure_upload_worker();
    auto & slot = g.slots[victim];
    if (slot.layer >= 0) {
        g.slot_index.erase((uint64_t(slot.layer) << 32) | uint32_t(slot.expert));
        ++f.evictions;
    }
    f.upload = {};
    f.upload.group = t.group; f.upload.position = victim;
    f.upload.entry.layer = t.layer; f.upload.entry.expert = expert; f.upload.entry.used = ++f.clock;
    f.upload.entry.predicted = prediction;
    for (int b = 0; b < 3; ++b) {
        f.upload.entry.banks[b] = t.banks[b];
        f.upload.entry.data[b] = t.banks[b] ? t.banks[b]->data : nullptr;
        auto & copy = f.upload_copies[b];
        copy = {};
        if (t.banks[b]) {
            copy.destination = *t.banks[b];
            resident_bank(f, t.group, b, copy.destination);
            copy.bank = t.banks[b];
            copy.source = static_cast<const uint8_t *>(t.banks[b]->data) + size_t(expert) * t.banks[b]->nb[2];
            copy.bytes = t.banks[b]->nb[2];
            copy.offset = victim * g.strides[b];
            f.upload.bytes += copy.bytes;
        }
    }
    slot = {}; slot.layer = -2; // unavailable on both host and device until published
    resident_mapping(sched, gpu);
    ggml_backend_event_record(f.upload_fence);
    f.host_bytes += f.upload.bytes;
    f.upload_queued.store(false, std::memory_order_release);
    f.upload.pending = true;
    if (prediction) ++f.prediction_uploads;
    {
        std::lock_guard<std::mutex> lock(f.upload_mutex);
        f.upload_job = true;
    }
    f.upload_wake.notify_one();
}

static void resident_prefetch_prediction(ggml_backend_sched_t sched, ggml_backend_t gpu, int completed_layer = -1) {
    auto & f = *sched->moe_ffn;
    // Forecast protection expires when that layer's actual result is queued.
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
        // Current execution has retired. The existing background uploader
        // owns copied descriptors and fences its victim before writing it.
        // No new arena, blocking weight upload, or router substitution.
        std::vector<bool> protected_slots(g.slots.size(), false);
        const std::vector<int> expert{c.expert};
        const auto uploads = f.prediction_uploads;
        resident_promote(sched, t, expert, protected_slots, gpu, true);
        it = f.forecast_queue.erase(it); // a rejected forecast cannot stall later layers
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
    if (cpu_count || !gpu_count) {
        const int64_t join_started = ggml_time_us();
        const auto cpu_status = f.finish();
        if (t.allow_defer) f.deferred_join_us += ggml_time_us() - join_started;
        // Independent GPU work may outlast this job. Admission must use CPU
        // execution time, not the later time at which its consumer joins it.
        const int64_t elapsed = std::max<int64_t>(1, f.cpu_finished_us - cpu_started);
        f.cpu_wait_us += elapsed;
        if (cpu_count) f.cpu_cost[t.layer] = 0.8 * f.cpu_cost[t.layer] +
                0.2 * double(std::max<int64_t>(1, elapsed)) / cpu_count;
        if (status == GGML_STATUS_SUCCESS) status = cpu_status;
        if (status == GGML_STATUS_SUCCESS && gpu_count) {
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
    const int64_t admission_started = ggml_time_us();
    if (cache) {
        resident_prefetch_prediction(sched, gpu, t.layer);
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
    if (cache && f.mapping_dirty) {
        const int64_t started = ggml_time_us();
        f.wait_io(); // mapping is a reusable asynchronous DMA source
        f.mapping_wait_us += ggml_time_us() - started;
        resident_mapping(sched, gpu);
    }
    // A rebuilt graph can contain different bank descriptors. Never reuse a
    // mapping whose source weight identity no longer matches the current task.
    bool complete = cache, has_resident = false;
    std::vector<int> stale;
    for (int e = 0; e < experts; ++e) {
        const int position = f.mapping[size_t(t.layer) * experts + e];
        if (position >= 0 && (size_t(position) >= g.slots.size() ||
                !ggml_backend_sched_moe_ffn_match(g.slots[position], t, e))) {
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
            g.slot_index.erase((uint64_t(t.layer) << 32) | uint32_t(e));
            if (size_t(position) < g.slots.size()) g.slots[position] = {};
        }
        resident_mapping(sched, gpu);
    }
    t.device_ids = *t.up->src[3];
    ggml_backend_sched_moe_ffn_bind(t.device_ids, f.device_ids);
    const auto * device_map = cache ? static_cast<const int32_t *>(ggml_backend_buffer_get_base(f.slot_map)) +
            size_t(t.layer) * experts : nullptr;
    if (complete) {
        // A fully resident layer needs neither host routing nor activation I/O.
        if (!ggml_backend_cuda_moe_route(gpu, t.up->src[3], device_map, &t.device_ids,
                t.up->src[2], nullptr, nullptr, experts)) return GGML_STATUS_FAILED;
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
    ++f.fallback_routes;
    ggml_backend_event_record(f.router_ready);
    if (!has_resident) ++f.empty_gpu_layers;
    const int64_t router_started = ggml_time_us();
    ggml_backend_event_synchronize(f.router_ready);
    f.router_wait_us += ggml_time_us() - router_started;
    f.pending = false; // the prior result's DMA precedes this routing event
    f.cpu_ids.assign(ids, ids + k); f.gpu_ids.assign(size_t(k), -1);
    resident_record_token_routes(sched, t, ids, tokens, k / tokens);
    std::vector<bool> protected_slots(g.slots.size(), false);
    std::vector<int> missing;
    size_t cpu_count = 0, gpu_count = 0;
    for (int row = 0; row < k; ++row) {
        const int e = ids[row];
        if (e < 0 || e >= experts) { f.cpu_ids[row] = -1; continue; }
        const size_t i = size_t(t.layer) * experts + e;
        f.popularity[i] = f.score(t.layer, e, experts) + 1;
        f.observed_at[i] = f.policy_clock;
        const int p = cache ? f.mapping[i] : -1;
        if (p >= 0) {
            f.gpu_ids[row] = p; f.cpu_ids[row] = -1; protected_slots[p] = true;
            if (g.slots[p].predicted) { ++f.prediction_used; g.slots[p].predicted = false; }
            g.slots[p].used = ++f.clock; ++gpu_count; ++f.hits;
        } else {
            ++cpu_count;
            if (std::find(missing.begin(), missing.end(), e) == missing.end()) missing.push_back(e);
        }
    }
    t.route_gpu = f.gpu_ids;
    f.activation_bytes += ggml_nbytes(&t.host_act);
    if (!gpu_count) ++f.cpu_only_layers;
    auto status = GGML_STATUS_SUCCESS;
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
        f.submit(t);
    }
    // CPU misses start first. Submit only the actual GPU hits, concurrently
    // with that CPU job; an all-miss route launches no masked GPU FFN.
    if (gpu_count) {
        auto * pinned_gpu_ids = ids + k;
        std::memcpy(pinned_gpu_ids, f.gpu_ids.data(), size_t(k) * sizeof(int32_t));
        const int64_t started = ggml_time_us();
        ggml_backend_tensor_set_async(gpu, &t.device_ids, pinned_gpu_ids, 0, ggml_nbytes(&t.device_ids));
        status = resident_gpu_ffn(f, t, gpu);
        f.gpu_enqueue_us += ggml_time_us() - started;
    }
    auto & call = f.in_flight;
    call.owner = &t; call.gpu = gpu; call.tokens = tokens; call.k = k;
    call.cpu_count = cpu_count; call.gpu_count = gpu_count;
    call.cpu_started = cpu_started; call.host_started = host_started; call.status = status;
    call.cache = cache;
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
