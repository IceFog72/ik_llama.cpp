// Hybrid expert residency and CPU/GPU execution inspired by FreeToken:
// https://arxiv.org/abs/2608.16157

// Scheduler-private complete-expert decode. Included after the resident-pool
// helpers so off/fixed and unsupported graphs keep their existing path.
static bool ggml_backend_sched_moe_ffn_node(ggml_backend_sched_t sched, const ggml_tensor * node) {
    return sched->moe_ffn &&
            sched->moe_ffn->graph_tasks.find(node) != sched->moe_ffn->graph_tasks.end();
}

static void ggml_backend_sched_moe_ffn_bind(ggml_tensor & t, ggml_backend_buffer_t buffer) {
    t.buffer = buffer;
    t.data = ggml_backend_buffer_get_base(buffer);
    t.view_src = nullptr;
    t.view_offs = 0;
    t.extra = nullptr;
}

static bool ggml_backend_sched_moe_ffn_scratch(ggml_backend_sched_t sched,
        const ggml_tensor * up, const ggml_tensor * down) {
#ifdef GGML_USE_CUDA
    auto & f = *sched->moe_ffn;
    f.wait_io();
    const auto host = ggml_backend_cuda_host_buffer_type();
    const auto device = sched->bufts[0];
    if (!f.completion) f.completion = ggml_backend_event_new(sched->backends[0]);
    return f.completion &&
        ggml_backend_sched_moe_decode_buffer(f.activation, host, ggml_nbytes(up->src[2])) &&
        ggml_backend_sched_moe_decode_buffer(f.intermediate, host, ggml_nbytes(up)) &&
        ggml_backend_sched_moe_decode_buffer(f.output, host, ggml_nbytes(down)) &&
        ggml_backend_sched_moe_decode_buffer(f.ids, host, 2 * ggml_nbytes(up->src[3])) &&
        ggml_backend_sched_moe_decode_buffer(f.device_ids, device, ggml_nbytes(up->src[3])) &&
        ggml_backend_sched_moe_decode_buffer(f.device_cpu, device, ggml_nbytes(down));
#else
    GGML_UNUSED(sched); GGML_UNUSED(up); GGML_UNUSED(down);
    return false;
#endif
}

static void ggml_backend_sched_moe_ffn_scan(ggml_backend_sched_t sched, const ggml_cgraph * graph) {
    if (sched->moe_ffn) {
        sched->moe_ffn->finish_upload();
        sched->moe_ffn->wait_io();
        sched->moe_ffn->graph_tasks.clear();
        sched->moe_ffn->observations.clear();
    }
auto note = [sched](int why) { ++sched->moe_ffn_reject_n[why]; };

#ifdef GGML_USE_CUDA
    if (sched->moe_resident_slots >= 0 || sched->n_backends != 2 || sched->n_copies != 1 ||
            sched->callback_eval || sched->split_mode_graph || !ggml_backend_is_cuda(sched->backends[0]) ||
            !ggml_backend_is_cpu(sched->backends[1]) ||
            !ggml_backend_sched_moe_resident_requested(sched, sched->backends[0]) ||
            (sched->moe_ffn && sched->moe_ffn->disabled)) {
        note(sched->MOE_FFN_R_GUARD);
        return;
    }
    std::unordered_map<const ggml_tensor *, int> uses;
    const bool new_policy = sched->moe_resident_zero_fill && sched->moe_resident_n_expert_used <= 1024;
    std::set<int> repeated_layers;
    std::set<int> seen_layers;
    for (int i = 0; i < graph->n_nodes; ++i)
        for (auto * src : graph->nodes[i]->src) if (src) ++uses[src];
    // Opt-in next-layer forecasts, keyed by the layer that produces them.
    std::unordered_map<int, ggml_tensor *> pregates;
    if (new_policy && sched->moe_resident_prefetch > 0) {
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * node = graph->nodes[i];
            if (node->op == GGML_OP_CONT && node->type == GGML_TYPE_I32 && ggml_is_contiguous(node) &&
                    node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == GGML_MOE_RESIDENT_PREGATE) {
                pregates[node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 2]] = node;
            }
        }
    }
    for (int i = 0; i + 1 < graph->n_nodes; ++i) {
        auto * up = graph->nodes[i], * down = graph->nodes[i + 1];
        if (up->op != GGML_OP_MOE_FUSED_UP_GATE) continue;
        if (new_policy && up->src[0] && up->src[3] &&
                up->src[3]->type == GGML_TYPE_I32 && ggml_is_contiguous(up->src[3])) {
            int observed_layer = -1;
            if (ggml_backend_sched_moe_parse_layer_id(up->src[0], sched->moe_resident_layers, observed_layer)) {
                if (!sched->moe_ffn) {
                    sched->moe_ffn = new ggml_backend_sched_moe_ffn;
                    sched->moe_ffn->ring_size = 2 * sched->moe_resident_prefetch;
                }
                sched->moe_ffn->new_policy = true;
                sched->moe_ffn->observations[up] = observed_layer;
            }
        }
        ++sched->moe_ffn_candidates;
        if (down->op != GGML_OP_MUL_MAT_ID || down->src[1] != up || down->src[2] != up->src[3]) {
            note(sched->MOE_FFN_R_OP_NOT_PAIR); continue;
        }
        if (uses[up] != 1 || (up->flags & GGML_TENSOR_FLAG_OUTPUT) || up->src[4] || up->src[5]) {
            note(sched->MOE_FFN_R_OP_WIRING); continue;
        }
        if (up->type != GGML_TYPE_F32 || down->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(up) || !ggml_is_contiguous(down)) {
            note(sched->MOE_FFN_R_OP_LAYOUT); continue;
        }
        if ((tensor_backend_id(up) >= 0 && tensor_backend_id(up) != 0) ||
                (tensor_backend_id(down) >= 0 && tensor_backend_id(down) != 0)) { note(sched->MOE_FFN_R_PLACEMENT); continue; }
        auto * act = up->src[2], * ids = up->src[3];
        if (!act || !ids) { note(sched->MOE_FFN_R_ACT_IDS_NULL); continue; }
        if (act->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
            note(sched->MOE_FFN_R_ACT_IDS_DTYPE); continue;
        }
        // Up to three MTP drafts verify alongside the current token. Reuse
        // the existing batched kernels for this bounded verification window.
        const int64_t max_tokens = new_policy ? 4 : 1;
        if (act->ne[2] < 1 || act->ne[2] > max_tokens || act->ne[3] != 1) {
            note(sched->MOE_FFN_R_ACT_BATCH); continue;
        }
        if (ids->ne[1] != act->ne[2] || (new_policy && ggml_nelements(ids) > 1024)) {
            note(sched->MOE_FFN_R_IDS_BATCH); continue;
        }
        if (ids->ne[0] != sched->moe_resident_n_expert_used || ids->ne[0] <= 0) {
            note(sched->MOE_FFN_R_IDS_WIDTH); continue;
        }
        if (!ggml_is_contiguous(act) || !ggml_is_contiguous(ids)) {
            note(sched->MOE_FFN_R_ACT_IDS_LAYOUT); continue;
        }
        ggml_tensor * banks[] = {up->src[0], up->src[1], down->src[0]};
        int layer = -1;
        bool supported = true;
        for (int b = 0; b < 3; ++b) {
            auto * bank = banks[b];
            if (!bank && b == 1) continue;
            int bank_layer = -1;
            if (!bank) { note(sched->MOE_FFN_R_BANK_NULL); supported = false; break; }
            if (!bank->buffer || !ggml_backend_buffer_is_host(bank->buffer)) {
                note(sched->MOE_FFN_R_BANK_NOT_HOST); supported = false; break;
            }
            if (ggml_backend_buffer_get_usage(bank->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                note(sched->MOE_FFN_R_BANK_USAGE); supported = false; break;
            }
            if (!ggml_is_quantized(bank->type)) { note(sched->MOE_FFN_R_BANK_UNQUANT); supported = false; break; }
            if (!ggml_is_contiguous(bank)) { note(sched->MOE_FFN_R_BANK_LAYOUT); supported = false; break; }
            if (bank->ne[2] != sched->moe_resident_n_expert || bank->ne[3] != 1) {
                note(sched->MOE_FFN_R_BANK_WIDTH); supported = false; break;
            }
            if (!ggml_backend_sched_moe_parse_layer_id(bank, sched->moe_resident_layers, bank_layer) ||
                    (layer >= 0 && layer != bank_layer)) {
                note(sched->MOE_FFN_R_BANK_LAYER_ID); supported = false; break;
            }
            layer = bank_layer;
        }
        if (!supported) continue;
        if (!ggml_backend_supports_op(sched->backends[0], up) ||
                !ggml_backend_supports_op(sched->backends[0], down) ||
                !ggml_backend_supports_op(sched->backends[1], up) ||
                !ggml_backend_supports_op(sched->backends[1], down)) {
            note(sched->MOE_FFN_R_UNSUPPORTED); continue;
        }
        ++sched->moe_ffn_qualified_events;
        if (sched->moe_ffn_qualified_layer.empty()) {
            // Only indexable layers exist; a dense layer has no expert bank and
            // must not be reported as missing.
            sched->moe_ffn_qualified_layer.resize(size_t(sched->moe_resident_layers), 0);
        }
        if (layer >= 0 && size_t(layer) < sched->moe_ffn_qualified_layer.size()) {
            if (!sched->moe_ffn_qualified_layer[size_t(layer)]++) {
                ++sched->moe_ffn_qualified_layers;
            }
        }
        if (!sched->moe_ffn) {
            sched->moe_ffn = new ggml_backend_sched_moe_ffn;
            sched->moe_ffn->ring_size = new_policy ? 2 * sched->moe_resident_prefetch : 0;
        }
        auto & f = *sched->moe_ffn;
        f.new_policy = new_policy;
        if (repeated_layers.count(layer)) { note(sched->MOE_FFN_R_REPEATED); continue; }
        if (!seen_layers.insert(layer).second) {
            f.graph_tasks.erase(f.tasks[layer]->up); f.graph_tasks.erase(f.tasks[layer]->down);
            repeated_layers.insert(layer); note(sched->MOE_FFN_R_REPEATED); continue;
        }
        ggml_backend_sched_moe_ffn::group layout;
        for (int b = 0; b < 3; ++b) {
            if (!banks[b]) continue;
            const size_t bytes = banks[b]->nb[2];
            if (bytes > SIZE_MAX - 1024) { supported = false; break; }
            layout.banks[b] = {true, banks[b]->type, banks[b]->ne[0], banks[b]->ne[1], bytes};
            layout.strides[b] = (bytes + 1023) / 512 * 512;
        }
        if (!supported) continue;
        // The fused CUDA operator uses a common expert stride for UP/GATE.
        if (banks[1]) layout.strides[0] = layout.strides[1] = std::max(layout.strides[0], layout.strides[1]);
        size_t group = 0;
        if (sched->moe_resident_grouping) {
            for (; group < f.groups.size(); ++group) {
                bool same = true;
                for (int b = 0; b < 3; ++b) {
                    const auto & a = layout.banks[b]; const auto & old = f.groups[group].banks[b];
                    same = same && a.present == old.present && a.type == old.type &&
                            a.columns == old.columns && a.rows == old.rows && a.bytes == old.bytes;
                }
                if (same) break;
            }
            const bool new_group = group == f.groups.size();
            const bool changed = new_group || size_t(layer) >= f.tasks.size() || !f.tasks[layer] || f.tasks[layer]->group != group;
            if (changed && f.cache) {
                ggml_backend_sched_synchronize(sched);
                f.wait_io();
                f.remember_cache();
                f.clear_cache();
            }
            if (new_group) f.groups.push_back(std::move(layout));
        } else {
            // Preserve the original shared pool: each projection retains its
            // largest observed stride, with one mapping across all layers.
            if (f.groups.empty()) f.groups.emplace_back();
            auto & shared = f.groups[0];
            bool changed = false;
            for (int b = 0; b < 3; ++b) {
                if (layout.strides[b] > shared.strides[b]) {
                    shared.strides[b] = layout.strides[b]; changed = true;
                }
            }
            if (banks[1]) {
                const size_t stride = std::max(shared.strides[0], shared.strides[1]);
                changed = changed || shared.strides[0] != stride || shared.strides[1] != stride;
                shared.strides[0] = shared.strides[1] = stride;
            }
            if (changed && f.cache) {
                ggml_backend_sched_synchronize(sched);
                f.wait_io();
                f.remember_cache();
                f.clear_cache();
            }
        }
        if (!ggml_backend_sched_moe_ffn_scratch(sched, up, down)) { note(sched->MOE_FFN_R_SCRATCH); continue; }
        if (size_t(layer) >= f.tasks.size()) f.tasks.resize(size_t(layer) + 1);
        if (!f.tasks[layer]) f.tasks[layer] = std::make_unique<ggml_backend_sched_moe_ffn::task>();
        auto & t = *f.tasks[layer];
        t.layer = layer; t.group = group; t.up = up; t.down = down;
        t.join = nullptr; t.allow_defer = false;
        const auto pregate = pregates.find(layer);
        t.pregate = pregate != pregates.end() && pregate->second->ne[1] == act->ne[2] ? pregate->second : nullptr;
        if (new_policy && down->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == GGML_MOE_RESIDENT_DEFERRED_DOWN) {
            for (int j = i + 2; j < graph->n_nodes; ++j) {
                auto * consumer = graph->nodes[j];
                if (consumer->src[0] == down &&
                        consumer->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == GGML_MOE_RESIDENT_JOIN) {
                    t.join = consumer;
                    break;
                }
            }
        }
        for (int b = 0; b < 3; ++b) t.banks[b] = banks[b];
        f.graph_tasks[up] = &t; f.graph_tasks[down] = &t;
    }
#else
    GGML_UNUSED(sched); GGML_UNUSED(graph);
#endif
}

static size_t ggml_backend_sched_moe_ffn_budget(ggml_backend_sched_t sched, size_t available, size_t & reserve) {
    reserve = sched->moe_resident_zero_fill ? 256ull * 1024 * 1024
            : std::max<size_t>((sched->moe_resident_adaptive ? 512ull : 256ull) * 1024 * 1024, available / 8);
    return ggml_backend_sched_moe_resident_cap_budget(sched, available > reserve ? available - reserve : 0);
}

static bool ggml_backend_sched_moe_new_cache(ggml_backend_sched_t sched, ggml_backend_t gpu, bool resize = false);
static void ggml_backend_sched_moe_new_collect(ggml_backend_sched_t sched);

// Called after the real graph arena has been allocated, before any new GPU work.
// begin_generation() has already waited and invalidated old CUDA captures.
static void ggml_backend_sched_moe_ffn_refresh(ggml_backend_sched_t sched) {
    if (!sched->moe_resident_refresh_pending) return;
    sched->moe_resident_refresh_pending = false;
#ifdef GGML_USE_CUDA
    if (!sched->moe_ffn || !sched->moe_ffn->cache) return; // lazy first allocation uses current free memory
    auto & f = *sched->moe_ffn;
    f.finish(); f.wait_io();
    if (f.new_policy) {
        f.finish_upload();
        ggml_backend_sched_moe_new_cache(sched, sched->backends[0], true);
        return;
    }
    ggml_backend_synchronize(sched->backends[0]);
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_cuda_get_device_memory(ggml_backend_cuda_get_device(sched->backends[0]), &free_bytes, &total_bytes);
    // Include our own live cache when calculating its replacement budget.
    const size_t available = free_bytes + f.allocation_bytes;
    f.budget_bytes = ggml_backend_sched_moe_ffn_budget(sched, available, f.reserve_bytes);
    std::vector<size_t> counts;
    size_t bytes = 0;
    const bool planned = f.plan_cache(f.budget_bytes, sched->moe_resident_n_expert_used,
            sched->moe_resident_n_expert, counts, bytes);
    // Always shrink to respect the cap/headroom. Avoid reuploading the working
    // set merely to gain a handful of slots after a small driver fluctuation.
    const bool keep = planned && f.allocation_bytes <= f.budget_bytes && bytes >= f.allocation_bytes &&
            bytes - f.allocation_bytes < 64ull * 1024 * 1024;
    fprintf(stderr, "ggml_backend_sched: moe-resident generation refresh available=%zu MiB budget=%zu MiB reserve=%zu MiB cache=%zu MiB %s\n",
            available / 1048576, f.budget_bytes / 1048576, f.reserve_bytes / 1048576,
            f.allocation_bytes / 1048576, keep ? "retained" : "resize");
    if (!keep) {
        f.remember_cache();
        f.clear_cache();
        f.disabled = false;
    }
    GGML_UNUSED(total_bytes);
#endif
}

static bool ggml_backend_sched_moe_ffn_cache(ggml_backend_sched_t sched, ggml_backend_t gpu) {
#ifdef GGML_USE_CUDA
    auto & f = *sched->moe_ffn;
    if (f.new_policy) return ggml_backend_sched_moe_new_cache(sched, gpu);
    if (f.cache) return true;
    if (f.disabled) return false;
    ggml_backend_synchronize(gpu);
    // Release legacy bank caches at a safe point before measuring actual free
    // memory. Their staging allocations belong to gallocr and remain valid.
    auto & pools = sched->moe_resident[0];
    for (int p = 0; p < pools.n_pools; ++p)
        ggml_backend_sched_moe_resident_clear_storage(&pools.pools[p], &sched->moe_resident_budget[0]);
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_cuda_get_device_memory(ggml_backend_cuda_get_device(gpu), &free_bytes, &total_bytes);
    f.budget_bytes = ggml_backend_sched_moe_ffn_budget(sched, free_bytes, f.reserve_bytes);
    std::vector<size_t> counts;
    size_t bytes = 0;
    f.disabled = !f.plan_cache(f.budget_bytes, sched->moe_resident_n_expert_used,
            sched->moe_resident_n_expert, counts, bytes);
    if (!f.disabled) {
        size_t offset = 0;
        for (size_t i = 0; i < f.groups.size(); ++i) {
            auto & g = f.groups[i];
            g.slots.resize(counts[i]);
            for (int b = 0; b < 3; ++b) { g.offsets[b] = offset; offset += counts[i] * g.strides[b]; }
        }
        f.allocation_bytes = bytes;
        f.cache = ggml_backend_buft_alloc_buffer(sched->bufts[0], f.allocation_bytes);
        if (f.cache) ggml_backend_buffer_clear(f.cache, 0);
        else f.disabled = true;
    }
    if (f.disabled) f.clear_cache();
    if (f.cache && !f.remembered.empty()) {
        // Refill the previous working set from authoritative host weights,
        // hottest first. Old and new VRAM caches never coexist.
        std::vector<size_t> next(f.groups.size(), 0);
        size_t restored = 0;
        for (const auto & saved : f.remembered) {
            const auto & old = saved.entry;
            if (saved.group >= f.groups.size() || old.layer < 0 || old.expert < 0 ||
                    old.expert >= sched->moe_resident_n_expert || size_t(old.layer) >= f.tasks.size() ||
                    !f.tasks[old.layer]) continue;
            auto & t = *f.tasks[old.layer];
            auto & g = f.groups[saved.group];
            if (t.group != saved.group || next[saved.group] >= g.slots.size()) continue;
            bool same = true;
            for (int b = 0; b < 3; ++b) {
                auto * bank = t.banks[b];
                same = same && old.banks[b] == bank && old.data[b] == (bank ? bank->data : nullptr) &&
                        (!bank || (bank->buffer && ggml_backend_buffer_is_host(bank->buffer) && bank->nb[2] <= g.strides[b]));
            }
            const uint64_t key = (uint64_t(old.layer) << 32) | uint32_t(old.expert);
            if (!same || g.slot_index.count(key)) continue;
            const size_t index = next[saved.group]++;
            for (int b = 0; b < 3; ++b) {
                auto * bank = t.banks[b];
                if (!bank) continue;
                ggml_moe_prefetch_wait(bank);
                ggml_tensor dst = *bank;
                dst.ne[2] = int64_t(g.slots.size());
                dst.nb[2] = g.strides[b]; dst.nb[3] = g.strides[b] * g.slots.size();
                ggml_backend_sched_moe_ffn_bind(dst, f.cache);
                dst.data = static_cast<uint8_t *>(dst.data) + g.offsets[b];
                ggml_backend_tensor_set_async(gpu, &dst,
                        static_cast<const uint8_t *>(bank->data) + size_t(old.expert) * bank->nb[2],
                        index * g.strides[b], bank->nb[2]);
                f.host_bytes += bank->nb[2];
            }
            g.slots[index] = old;
            g.slot_index[key] = int(index);
            ++restored;
        }
        ggml_backend_synchronize(gpu); // every restored slot is ready before dispatch
        fprintf(stderr, "ggml_backend_sched: moe-resident restored=%zu/%zu experts after cache resize\n",
                restored, f.remembered.size());
        f.remembered.clear();
    }
    for (size_t i = 0; i < f.groups.size(); ++i) if (!f.groups[i].slots.empty()) {
        const auto & g = f.groups[i];
        fprintf(stderr, "ggml_backend_sched: moe-resident layout=%zu expert_bytes=%zu slots=%zu\n",
                i, g.strides[0] + g.strides[1] + g.strides[2], g.slots.size());
    }
    fprintf(stderr, "ggml_backend_sched: moe-resident complete cache free_after_reserve=%zu MiB budget=%zu MiB reserve=%zu MiB slots=%zu allocated=%zu MiB%s\n",
            free_bytes / (1024 * 1024), f.budget_bytes / (1024 * 1024), f.reserve_bytes / (1024 * 1024),
            f.slot_count(), f.allocation_bytes / (1024 * 1024), f.disabled ? " CPU fallback" : "");
    if (f.disabled) {
        // The allocated pair remains valid because the host banks are still its
        // source of truth. Compute it fully on CPU, then rebuild fresh graphs.
        sched->moe_resident_replan_required = true;
        for (int l = 0; l < sched->moe_resident_hybrid_layers; ++l)
            ggml_backend_sched_moe_resident_hybrid_set_layer(sched, l, false);
    }
    GGML_UNUSED(total_bytes);
    return f.cache != nullptr;
#else
    GGML_UNUSED(sched); GGML_UNUSED(gpu);
    return false;
#endif
}

static bool ggml_backend_sched_moe_ffn_match(const ggml_backend_sched_moe_ffn::slot & slot,
        const ggml_backend_sched_moe_ffn::task & t, int expert) {
    if (slot.layer != t.layer || slot.expert != expert) return false;
    for (int b = 0; b < 3; ++b)
        if (slot.banks[b] != t.banks[b] || slot.data[b] != (t.banks[b] ? t.banks[b]->data : nullptr)) return false;
    return true;
}

static ggml_backend_sched_moe_ffn::timing & ggml_backend_sched_moe_ffn_timing(
        ggml_backend_sched_moe_ffn & f, const ggml_backend_sched_moe_ffn::task & t, int top_k) {
    for (auto & m : f.timings) {
        bool same = m.columns == t.banks[0]->ne[0] && m.rows == t.banks[0]->ne[1] &&
                std::memcmp(m.params, t.up->op_params, sizeof(m.params)) == 0;
        for (int b = 0; b < 3; ++b)
            same = same && m.types[b] == (t.banks[b] ? t.banks[b]->type : GGML_TYPE_COUNT);
        if (same && m.misses.size() == size_t(top_k + 1)) return m;
    }
    f.timings.emplace_back();
    auto & m = f.timings.back();
    m.columns = t.banks[0]->ne[0]; m.rows = t.banks[0]->ne[1];
    for (int b = 0; b < 3; ++b) m.types[b] = t.banks[b] ? t.banks[b]->type : GGML_TYPE_COUNT;
    std::memcpy(m.params, t.up->op_params, sizeof(m.params));
    m.misses.resize(size_t(top_k + 1));
    for (int misses = 0; misses <= top_k; ++misses) m.misses[misses].resize(size_t(misses + 1));
    return m;
}

#include "ggml-moe-resident-new.inc"

static ggml_status ggml_backend_sched_moe_ffn_compute(ggml_backend_sched_t sched,
        ggml_backend_sched_moe_ffn::task & t, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    if (f.new_policy) return ggml_backend_sched_moe_new_compute(sched, t, gpu);
    const bool cache = ggml_backend_sched_moe_ffn_cache(sched, gpu);
    auto & g = f.groups[t.group];
    const int k = int(t.up->src[3]->ne[0]);
    t.host_act = *t.up->src[2]; t.host_ids = *t.up->src[3];
    t.host_up = *t.up; t.host_down = *t.down;
    ggml_backend_sched_moe_ffn_bind(t.host_act, f.activation);
    ggml_backend_sched_moe_ffn_bind(t.host_ids, f.ids);
    ggml_backend_sched_moe_ffn_bind(t.host_up, f.intermediate);
    ggml_backend_sched_moe_ffn_bind(t.host_down, f.output);
    t.host_up.src[0] = t.banks[0]; t.host_up.src[1] = t.banks[1];
    t.host_up.src[2] = &t.host_act; t.host_up.src[3] = &t.host_ids;
    t.host_down.src[0] = t.banks[2]; t.host_down.src[1] = &t.host_up; t.host_down.src[2] = &t.host_ids;
    t.host_nodes[0] = &t.host_up; t.host_nodes[1] = &t.host_down;
    t.host_graph = {}; t.host_graph.nodes = t.host_nodes; t.host_graph.n_nodes = 2;
    // Routing and activations share one device wait. There is no intermediate
    // CPU→GPU→CPU transfer between the two expert projections.
    ggml_backend_tensor_get_async(gpu, t.up->src[3], t.host_ids.data, 0, ggml_nbytes(&t.host_ids));
    ggml_backend_tensor_get_async(gpu, t.up->src[2], t.host_act.data, 0, ggml_nbytes(&t.host_act));
    ggml_backend_synchronize(gpu);
    f.pending = false; // This wait also completed the prior result upload.
    auto * original_ids = static_cast<int32_t *>(t.host_ids.data);
    f.cpu_ids.assign(original_ids, original_ids + k);
    f.gpu_ids.assign(size_t(k), -1);
    std::vector<int> experts, positions, missing;
    for (int row = 0; row < k; ++row) {
        const int expert = original_ids[row];
        if (expert < 0 || expert >= sched->moe_resident_n_expert) {
            f.cpu_ids[row] = -1; continue;
        }
        if (std::find(experts.begin(), experts.end(), expert) != experts.end()) continue;
        experts.push_back(expert);
        int position = -1;
        if (cache) {
            const uint64_t key = (uint64_t(t.layer) << 32) | uint32_t(expert);
            const auto found = g.slot_index.find(key);
            if (found != g.slot_index.end()) {
                auto & slot = g.slots[found->second];
                if (ggml_backend_sched_moe_ffn_match(slot, t, expert)) position = found->second;
                else { slot = {}; g.slot_index.erase(found); }
            }
        }
        positions.push_back(position);
        if (position < 0) missing.push_back(int(experts.size() - 1));
        else ++f.hits;
    }
    auto & timing = ggml_backend_sched_moe_ffn_timing(f, t, k);
    const int m = int(missing.size());
    const int first = sched->moe_resident_zero_fill ? 0 : 1;
    int q = cache && m ? first : 0;
    bool probe = false;
    if (cache && m) {
        auto & samples = timing.misses[m];
        // Profile the real concurrent, complete-FFN path. Three observations per
        // candidate reject single hot-kernel/transfer measurements. Later probes
        // revisit the policy under changed host load without changing outputs.
        // Zero fills keeps existing GPU hits and computes every miss on CPU.
        for (int candidate = first; candidate <= m; ++candidate) {
            if (samples[candidate].count < 3) { q = candidate; probe = true; break; }
        }
        if (!probe) {
            for (int candidate = first + 1; candidate <= m; ++candidate)
                if (samples[candidate].us < samples[q].us) q = candidate;
            if (++timing.visits % 256 == 0) {
                q = sched->moe_resident_zero_fill
                        ? int((timing.visits / 256 - 1) % size_t(m + 1))
                        : 1 + int((timing.visits / 256) % size_t(m));
                probe = true;
            }
        }
    }
    // Protect every current hit and selected fill before choosing any victim.
    std::vector<bool> protected_slots(g.slots.size(), false);
    for (int position : positions) if (position >= 0) protected_slots[position] = true;
    std::vector<int> filled, victims;
    if (q) {
        for (size_t s = 0; s < g.slots.size(); ++s) if (!protected_slots[s]) victims.push_back(int(s));
        GGML_ASSERT(victims.size() >= size_t(q));
        std::partial_sort(victims.begin(), victims.begin() + q, victims.end(), [&](int a, int b) {
            const auto & sa = g.slots[a]; const auto & sb = g.slots[b];
            if ((sa.layer < 0) != (sb.layer < 0)) return sa.layer < 0;
            return sa.used == sb.used ? a < b : sa.used < sb.used;
        });
        for (int j = 0; j < q; ++j) {
            positions[missing[j]] = victims[j]; filled.push_back(missing[j]);
        }
    }
    size_t cpu_count = 0, gpu_count = 0;
    for (int row = 0; row < k; ++row) {
        const auto found = std::find(experts.begin(), experts.end(), original_ids[row]);
        const int position = found == experts.end() ? -1 : positions[size_t(found - experts.begin())];
        if (position >= 0) { f.gpu_ids[row] = position; f.cpu_ids[row] = -1; ++gpu_count; }
        else if (f.cpu_ids[row] >= 0) ++cpu_count;
    }
    t.route_gpu = f.gpu_ids;
    // The second half of pinned ID scratch is the stable DMA source for GPU
    // IDs. The first half belongs exclusively to the CPU worker.
    auto * pinned_gpu_ids = original_ids + k;
    std::memcpy(original_ids, f.cpu_ids.data(), size_t(k) * sizeof(int32_t));
    std::memcpy(pinned_gpu_ids, f.gpu_ids.data(), size_t(k) * sizeof(int32_t));
    f.cpu = sched->backends[1];
    const int64_t started = ggml_time_us();
    if (cpu_count || !gpu_count) f.submit(t);
    // Start the CPU worker before DMA reads compete for host-memory bandwidth.
    for (int index : filled) {
        auto & slot = g.slots[positions[index]];
        if (slot.layer >= 0) {
            ++f.evictions;
            g.slot_index.erase((uint64_t(slot.layer) << 32) | uint32_t(slot.expert));
        }
        for (int b = 0; b < 3; ++b) {
            auto * bank = t.banks[b];
            if (!bank) { slot.banks[b] = nullptr; slot.data[b] = nullptr; continue; }
            ggml_moe_prefetch_wait(bank);
            ggml_tensor destination = *bank;
            destination.ne[2] = int64_t(g.slots.size());
            destination.nb[2] = g.strides[b]; destination.nb[3] = destination.nb[2] * g.slots.size();
            ggml_backend_sched_moe_ffn_bind(destination, f.cache);
            destination.data = static_cast<uint8_t *>(destination.data) + g.offsets[b];
            const size_t offset = size_t(positions[index]) * g.strides[b];
            ggml_backend_tensor_set_async(gpu, &destination,
                    static_cast<const uint8_t *>(bank->data) + size_t(experts[index]) * bank->nb[2], offset, bank->nb[2]);
            f.host_bytes += bank->nb[2];
            slot.banks[b] = bank; slot.data[b] = bank->data;
        }
        slot.layer = t.layer; slot.expert = experts[index]; ++f.fills;
        g.slot_index[(uint64_t(t.layer) << 32) | uint32_t(slot.expert)] = positions[index];
    }
    if (++f.clock == 0) { for (auto & group : f.groups) for (auto & slot : group.slots) slot.used = 0; f.clock = 1; }
    for (int position : positions) if (position >= 0) g.slots[position].used = f.clock;
    ggml_status status = GGML_STATUS_SUCCESS;
    if (gpu_count) {
        t.device_ids = *t.up->src[3];
        ggml_backend_sched_moe_ffn_bind(t.device_ids, f.device_ids);
        ggml_backend_tensor_set_async(gpu, &t.device_ids, pinned_gpu_ids, 0, ggml_nbytes(&t.device_ids));
        for (int b = 0; b < 3; ++b) {
            if (!t.banks[b]) continue;
            t.device_banks[b] = *t.banks[b];
            t.device_banks[b].ne[2] = int64_t(g.slots.size());
            t.device_banks[b].nb[2] = g.strides[b];
            t.device_banks[b].nb[3] = g.strides[b] * g.slots.size();
            ggml_backend_sched_moe_ffn_bind(t.device_banks[b], f.cache);
            t.device_banks[b].data = static_cast<uint8_t *>(t.device_banks[b].data) + g.offsets[b];
        }
        t.device_up = *t.up; t.device_down = *t.down;
        t.device_up.src[0] = &t.device_banks[0];
        t.device_up.src[1] = t.banks[1] ? &t.device_banks[1] : nullptr;
        t.device_up.src[3] = &t.device_ids;
        t.device_down.src[0] = &t.device_banks[2];
        t.device_down.src[1] = &t.device_up; t.device_down.src[2] = &t.device_ids;
        t.device_nodes[0] = &t.device_up; t.device_nodes[1] = &t.device_down;
        t.device_graph = {}; t.device_graph.nodes = t.device_nodes; t.device_graph.n_nodes = 2;
#ifdef GGML_USE_CUDA
        // Masked vector kernels can leave skipped rows untouched. Clear those
        // intermediates before DOWN quantizes them, without uploading zeros.
        ggml_backend_cuda_moe_copy_rows(gpu, &t.device_up, nullptr, &t.device_ids);
#endif
        status = ggml_backend_graph_compute_async(gpu, &t.device_graph);
    }
    if (cpu_count || !gpu_count) {
        const auto cpu_status = f.finish();
        if (status == GGML_STATUS_SUCCESS) status = cpu_status;
        if (status == GGML_STATUS_SUCCESS) {
            // One final upload and one masked overwrite preserve original row
            // order and the existing weighting/reduction. CUDA rows stay intact.
            if (gpu_count) {
                t.device_cpu = *t.down;
                ggml_backend_sched_moe_ffn_bind(t.device_cpu, f.device_cpu);
                ggml_backend_tensor_set_async(gpu, &t.device_cpu, t.host_down.data, 0, ggml_nbytes(&t.host_down));
#ifdef GGML_USE_CUDA
                ggml_backend_cuda_moe_copy_rows(gpu, t.down, &t.device_cpu, &t.device_ids);
#endif
            } else {
                ggml_backend_tensor_set_async(gpu, t.down, t.host_down.data, 0, ggml_nbytes(&t.host_down));
            }
            f.result_bytes += ggml_nbytes(&t.host_down);
        }
    }
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_synchronize(gpu); return status;
    }
    ggml_backend_event_record(f.completion); f.pending = true;
    if (probe) {
        f.wait_io();
        const double elapsed = double(std::max<int64_t>(1, ggml_time_us() - started));
        auto & sample = timing.misses[m][q];
        sample.us = sample.count ? sample.us * 0.75 + elapsed * 0.25 : elapsed;
        ++sample.count;
        if (sample.count == 3) fprintf(stderr,
                "ggml_backend_sched: moe-resident FFN latency type=%s misses=%d fills=%d concurrent_us=%.1f samples=%d\n",
                ggml_type_name(t.banks[0]->type), m, q, sample.us, sample.count);
    }
    ++f.layers; f.cpu_refs += cpu_count; f.gpu_refs += gpu_count;
    return GGML_STATUS_SUCCESS;
}
