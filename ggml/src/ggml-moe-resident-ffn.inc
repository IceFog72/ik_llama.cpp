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
        sched->moe_ffn->wait_io();
        sched->moe_ffn->graph_tasks.clear();
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
    std::set<int> repeated_layers;
    for (int i = 0; i < graph->n_nodes; ++i)
        for (auto * src : graph->nodes[i]->src) if (src) ++uses[src];
    for (int i = 0; i + 1 < graph->n_nodes; ++i) {
        auto * up = graph->nodes[i], * down = graph->nodes[i + 1];
        if (up->op != GGML_OP_MOE_FUSED_UP_GATE) continue;
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
        if (act->ne[2] != 1 || act->ne[3] != 1) { note(sched->MOE_FFN_R_ACT_BATCH); continue; }
        if (ids->ne[1] != 1) { note(sched->MOE_FFN_R_IDS_BATCH); continue; }
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
        if (!sched->moe_ffn) sched->moe_ffn = new ggml_backend_sched_moe_ffn;
        auto & f = *sched->moe_ffn;
        if (repeated_layers.count(layer)) { note(sched->MOE_FFN_R_REPEATED); continue; }
        if (size_t(layer) < f.tasks.size() && f.tasks[layer] && f.graph_tasks.count(f.tasks[layer]->up)) {
            f.graph_tasks.erase(f.tasks[layer]->up); f.graph_tasks.erase(f.tasks[layer]->down);
            repeated_layers.insert(layer); note(sched->MOE_FFN_R_REPEATED); continue;
        }
        bool changed = false;
        for (int b = 0; b < 3; ++b) {
            if (!banks[b]) continue;
            const size_t bytes = banks[b]->nb[2];
            if (bytes > SIZE_MAX - 1024) { supported = false; break; }
            const size_t stride = (bytes + 1023) / 512 * 512;
            if (stride > f.strides[b]) { f.strides[b] = stride; changed = true; }
        }
        if (!supported) continue;
        // The fused CUDA operator uses a common expert stride for UP/GATE.
        if (banks[1]) {
            const size_t stride = std::max(f.strides[0], f.strides[1]);
            changed = changed || f.strides[0] != stride || f.strides[1] != stride;
            f.strides[0] = f.strides[1] = stride;
        }
        if (changed && f.cache) {
            ggml_backend_sched_synchronize(sched);
            f.wait_io();
            ggml_backend_buffer_free(f.cache);
            f.cache = nullptr; f.slots.clear(); f.slot_index.clear(); f.allocation_bytes = 0;
        }
        if (!ggml_backend_sched_moe_ffn_scratch(sched, up, down)) { note(sched->MOE_FFN_R_SCRATCH); continue; }
        if (size_t(layer) >= f.tasks.size()) f.tasks.resize(size_t(layer) + 1);
        if (!f.tasks[layer]) f.tasks[layer] = std::make_unique<ggml_backend_sched_moe_ffn::task>();
        auto & t = *f.tasks[layer];
        t.layer = layer; t.up = up; t.down = down;
        for (int b = 0; b < 3; ++b) t.banks[b] = banks[b];
        f.graph_tasks[up] = &t; f.graph_tasks[down] = &t;
    }
#else
    GGML_UNUSED(sched); GGML_UNUSED(graph);
#endif
}

static bool ggml_backend_sched_moe_ffn_cache(ggml_backend_sched_t sched, ggml_backend_t gpu) {
#ifdef GGML_USE_CUDA
    auto & f = *sched->moe_ffn;
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
    f.reserve_bytes = std::max<size_t>(256 * 1024 * 1024, free_bytes / 8);
    f.budget_bytes = free_bytes > f.reserve_bytes ? free_bytes - f.reserve_bytes : 0;
    size_t expert_bytes = 0;
    for (size_t stride : f.strides) {
        if (stride > SIZE_MAX - expert_bytes) { f.disabled = true; return false; }
        expert_bytes += stride;
    }
    if (!expert_bytes) { f.disabled = true; return false; }
    // Two full active working sets over the entire model bound the useful
    // allocation. Capacity is global, with no per-layer reservation.
    const size_t target = size_t(sched->moe_resident_layers) * size_t(sched->moe_resident_n_expert_used) * 2;
    const size_t slots = std::min({f.budget_bytes / expert_bytes, target, size_t(INT_MAX)});
    if (slots < size_t(sched->moe_resident_n_expert_used)) {
        f.disabled = true;
    } else {
        f.allocation_bytes = slots * expert_bytes;
        f.cache = ggml_backend_buft_alloc_buffer(sched->bufts[0], f.allocation_bytes);
        if (!f.cache) { f.disabled = true; f.allocation_bytes = 0; }
        else {
            ggml_backend_buffer_clear(f.cache, 0);
            f.slots.resize(slots);
            size_t offset = 0;
            for (int b = 0; b < 3; ++b) { f.offsets[b] = offset; offset += slots * f.strides[b]; }
        }
    }
    fprintf(stderr, "ggml_backend_sched: moe-resident complete cache free_after_reserve=%zu MiB budget=%zu MiB reserve=%zu MiB slots=%zu allocated=%zu MiB%s\n",
            free_bytes / (1024 * 1024), f.budget_bytes / (1024 * 1024), f.reserve_bytes / (1024 * 1024),
            f.slots.size(), f.allocation_bytes / (1024 * 1024), f.disabled ? " CPU fallback" : "");
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

static ggml_status ggml_backend_sched_moe_ffn_compute(ggml_backend_sched_t sched,
        ggml_backend_sched_moe_ffn::task & t, ggml_backend_t gpu) {
    auto & f = *sched->moe_ffn;
    const bool cache = ggml_backend_sched_moe_ffn_cache(sched, gpu);
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
            const auto found = f.slot_index.find(key);
            if (found != f.slot_index.end()) {
                auto & slot = f.slots[found->second];
                if (ggml_backend_sched_moe_ffn_match(slot, t, expert)) position = found->second;
                else { slot = {}; f.slot_index.erase(found); }
            }
        }
        positions.push_back(position);
        if (position < 0) missing.push_back(int(experts.size() - 1));
        else ++f.hits;
    }
    auto & timing = ggml_backend_sched_moe_ffn_timing(f, t, k);
    const int m = int(missing.size());
    int q = cache && m ? 1 : 0;
    bool probe = false;
    if (cache && m) {
        auto & samples = timing.misses[m];
        // Profile the real concurrent, complete-FFN path. Three observations per
        // candidate reject single hot-kernel/transfer measurements. Later probes
        // revisit the policy under changed host load without changing outputs.
        for (int candidate = 1; candidate <= m; ++candidate) {
            if (samples[candidate].count < 3) { q = candidate; probe = true; break; }
        }
        if (!probe) {
            for (int candidate = 2; candidate <= m; ++candidate)
                if (samples[candidate].us < samples[q].us) q = candidate;
            if (++timing.visits % 256 == 0) {
                q = 1 + int((timing.visits / 256) % size_t(m)); probe = true;
            }
        }
    }
    // Protect every current hit and selected fill before choosing any victim.
    std::vector<bool> protected_slots(f.slots.size(), false);
    for (int position : positions) if (position >= 0) protected_slots[position] = true;
    std::vector<int> filled, victims;
    if (q) {
        for (size_t s = 0; s < f.slots.size(); ++s) if (!protected_slots[s]) victims.push_back(int(s));
        GGML_ASSERT(victims.size() >= size_t(q));
        std::partial_sort(victims.begin(), victims.begin() + q, victims.end(), [&](int a, int b) {
            const auto & sa = f.slots[a]; const auto & sb = f.slots[b];
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
        auto & slot = f.slots[positions[index]];
        if (slot.layer >= 0) {
            ++f.evictions;
            f.slot_index.erase((uint64_t(slot.layer) << 32) | uint32_t(slot.expert));
        }
        for (int b = 0; b < 3; ++b) {
            auto * bank = t.banks[b];
            if (!bank) { slot.banks[b] = nullptr; slot.data[b] = nullptr; continue; }
            ggml_moe_prefetch_wait(bank);
            ggml_tensor destination = *bank;
            destination.ne[2] = int64_t(f.slots.size());
            destination.nb[2] = f.strides[b]; destination.nb[3] = destination.nb[2] * f.slots.size();
            ggml_backend_sched_moe_ffn_bind(destination, f.cache);
            destination.data = static_cast<uint8_t *>(destination.data) + f.offsets[b];
            const size_t offset = size_t(positions[index]) * f.strides[b];
            ggml_backend_tensor_set_async(gpu, &destination,
                    static_cast<const uint8_t *>(bank->data) + size_t(experts[index]) * bank->nb[2], offset, bank->nb[2]);
            f.host_bytes += bank->nb[2];
            slot.banks[b] = bank; slot.data[b] = bank->data;
        }
        slot.layer = t.layer; slot.expert = experts[index]; ++f.fills;
        f.slot_index[(uint64_t(t.layer) << 32) | uint32_t(slot.expert)] = positions[index];
    }
    if (++f.clock == 0) { for (auto & slot : f.slots) slot.used = 0; f.clock = 1; }
    for (int position : positions) if (position >= 0) f.slots[position].used = f.clock;
    ggml_status status = GGML_STATUS_SUCCESS;
    if (gpu_count) {
        t.device_ids = *t.up->src[3];
        ggml_backend_sched_moe_ffn_bind(t.device_ids, f.device_ids);
        ggml_backend_tensor_set_async(gpu, &t.device_ids, pinned_gpu_ids, 0, ggml_nbytes(&t.device_ids));
        for (int b = 0; b < 3; ++b) {
            if (!t.banks[b]) continue;
            t.device_banks[b] = *t.banks[b];
            t.device_banks[b].ne[2] = int64_t(f.slots.size());
            t.device_banks[b].nb[2] = f.strides[b];
            t.device_banks[b].nb[3] = f.strides[b] * f.slots.size();
            ggml_backend_sched_moe_ffn_bind(t.device_banks[b], f.cache);
            t.device_banks[b].data = static_cast<uint8_t *>(t.device_banks[b].data) + f.offsets[b];
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
