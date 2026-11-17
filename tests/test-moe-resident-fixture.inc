// Tiny real CUDA graph shared by history and delayed-upload checks.
#ifdef GGML_USE_CUDA
struct resident_cuda_fixture {
    static constexpr int width = 256, experts = 4, top_k = 2, layer = 1;
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * sources = nullptr, * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * banks[3]{}, * activation = nullptr, * ids = nullptr, * output = nullptr;
    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<float> input;
    std::array<int32_t, top_k> route{0, 1};

    resident_cuda_fixture(int slots, int ring_entries = 0) {
        gpu = ggml_backend_cuda_init(0, nullptr, nullptr);
        cpu = ggml_backend_cpu_init();
        GGML_ASSERT(gpu && cpu);
        ggml_backend_cpu_set_n_threads(cpu, 2);
        ggml_backend_t backends[]{gpu, cpu};
        sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false);
        ggml_backend_sched_set_moe_resident_model_info(sched, layer + 1, experts, top_k);
        ggml_backend_sched_set_moe_resident_zero_fill(sched, true);
        ggml_backend_sched_set_moe_resident(sched, -1);
        if (ring_entries) ggml_backend_sched_set_moe_resident_prefetch(sched, ring_entries / 2);
        sources = ggml_init({2 * 1024 * 1024, nullptr, true});
        auto allocate = [&](ggml_tensor * tensor) {
            const auto type = ggml_backend_cuda_host_buffer_type();
            auto * buffer = ggml_backend_buft_alloc_buffer(type, ggml_backend_buft_get_alloc_size(type, tensor));
            GGML_ASSERT(buffer);
            ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
            buffers.push_back(buffer);
        };
        for (int b = 0; b < 3; ++b) {
            banks[b] = ggml_new_tensor_3d(sources, GGML_TYPE_Q4_0, width, width, experts);
            ggml_format_name(banks[b], "blk.%d.ffn_%s_exps.weight", layer, b == 0 ? "up" : b == 1 ? "gate" : "down");
            allocate(banks[b]);
            ggml_backend_buffer_set_usage(banks[b]->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> values(ggml_nelements(banks[b]));
            for (size_t i = 0; i < values.size(); ++i) values[i] = 0.01f * (int((i * 17 + b * 3) % 23) - 11);
            std::vector<uint8_t> packed(ggml_nbytes(banks[b]));
            ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), packed.data(), 0,
                    ggml_nrows(banks[b]), width, nullptr, nullptr);
            ggml_backend_tensor_set(banks[b], packed.data(), 0, packed.size());
        }
        activation = ggml_new_tensor_3d(sources, GGML_TYPE_F32, width, 1, 1);
        ids = ggml_new_tensor_2d(sources, GGML_TYPE_I32, top_k, 1);
        allocate(activation); allocate(ids);
        ggml_set_input(activation); ggml_set_input(ids);
        input.resize(width);
        for (int i = 0; i < width; ++i) input[i] = 0.03f * (i % 13 - 6);
        ggml_backend_tensor_set(activation, input.data(), 0, ggml_nbytes(activation));
        set_route(0, 1);
        context = ggml_init({2 * 1024 * 1024, nullptr, true});
        graph = ggml_new_graph_custom(context, 128, false);
        auto * up = ggml_moe_up_gate(context, banks[0], banks[1], activation, ids, GGML_UNARY_OP_SILU);
        output = ggml_mul_mat_id(context, banks[2], up, ids);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        auto & f = state();
        GGML_ASSERT(f.new_policy && f.graph_tasks.size() == 2 && f.groups.size() == 1);
        f.clear_cache();
        resident_history(sched);
        f.planned_tasks = 1;
        f.ring_size = ring_entries;
        f.rings.resize(1); f.rings[0].resize(ring_entries);
        auto & group = f.groups[0];
        size_t bytes = 0;
        for (int b = 0; b < 3; ++b) {
            group.offsets[b] = bytes;
            bytes += size_t(slots + ring_entries) * group.strides[b];
        }
        group.slots.resize(slots);
        f.cache = ggml_backend_buft_alloc_buffer(sched->bufts[0], bytes);
        GGML_ASSERT(f.cache);
        f.allocation_bytes = bytes;
        GGML_ASSERT(ggml_backend_sched_moe_decode_buffer(f.slot_map, sched->bufts[0], f.mapping.size() * sizeof(int32_t)));
        auto & task = *f.tasks[layer];
        for (int e = 0; e < slots; ++e) {
            resident_upload(f, task, e, e, gpu);
            auto & slot = group.slots[e];
            slot.layer = layer; slot.expert = e;
            for (int b = 0; b < 3; ++b) { slot.banks[b] = task.banks[b]; slot.data[b] = task.banks[b]->data; }
            group.slot_index[(uint64_t(layer) << 32) | uint32_t(e)] = e;
        }
        resident_mapping(sched, gpu);
        ggml_backend_synchronize(gpu);
        f.upload_policy_us = 1e6; // keep optional admissions out of these ownership checks
    }

    ggml_backend_sched_moe_ffn & state() { GGML_ASSERT(sched->moe_ffn); return *sched->moe_ffn; }

    void set_route(int first, int second) {
        route = {first, second};
        ggml_backend_tensor_set(ids, route.data(), 0, ggml_nbytes(ids));
    }

    std::vector<float> reference(ggml_backend_t backend) {
        auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        ggml_tensor * weights[3];
        for (int b = 0; b < 3; ++b) weights[b] = ggml_dup_tensor(ctx, banks[b]);
        auto * act = ggml_dup_tensor(ctx, activation), * selected = ggml_dup_tensor(ctx, ids);
        auto * up = ggml_moe_up_gate(ctx, weights[0], weights[1], act, selected, GGML_UNARY_OP_SILU);
        auto * down = ggml_mul_mat_id(ctx, weights[2], up, selected);
        auto * native = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(native, down);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        for (int b = 0; b < 3; ++b) ggml_backend_tensor_set(weights[b], banks[b]->data, 0, ggml_nbytes(banks[b]));
        ggml_backend_tensor_set(act, input.data(), 0, ggml_nbytes(act));
        ggml_backend_tensor_set(selected, route.data(), 0, ggml_nbytes(selected));
        check(ggml_backend_graph_compute(backend, native) == GGML_STATUS_SUCCESS, "native reference succeeds");
        std::vector<float> result(ggml_nelements(down));
        ggml_backend_tensor_get(down, result.data(), 0, ggml_nbytes(down));
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        return result;
    }

    ~resident_cuda_fixture() {
        ggml_backend_sched_free(sched);
        ggml_free(context);
        for (auto * buffer : buffers) ggml_backend_buffer_free(buffer);
        ggml_free(sources);
        ggml_backend_free(cpu); ggml_backend_free(gpu);
    }
};

static void test_fully_resident_history() {
    resident_cuda_fixture fixture(resident_cuda_fixture::experts);
    auto & f = fixture.state();
    const auto expected = fixture.reference(fixture.gpu);
    for (int pass = 0; pass < 24; ++pass)
        check(ggml_backend_sched_graph_compute(fixture.sched, fixture.graph) == GGML_STATUS_SUCCESS,
                "fully resident decode succeeds");
    check(f.all_hit_layers == 24 && f.cpu_refs == 0, "history check exercises GPU-only layers");
    for (int e = 0; e < resident_cuda_fixture::experts; ++e) {
        const auto i = size_t(resident_cuda_fixture::layer) * resident_cuda_fixture::experts + e;
        check(e < 2 ? f.popularity[i] > 1 && f.recent_popularity[i] > 1 : f.popularity[i] == 0,
                "host history refreshes active experts without inventing unused routes");
    }
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(fixture.output, actual.data(), 0, ggml_nbytes(fixture.output));
    for (size_t i = 0; i < actual.size(); ++i)
        check(std::fabs(actual[i] - expected[i]) <= 1e-5f + 1e-5f * std::fabs(expected[i]),
                "history snapshots preserve native GPU output");
    resident_history_poll(fixture.sched, true);
    auto * signal = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 2 * sizeof(uint32_t));
    GGML_ASSERT(signal);
    auto * words = static_cast<uint32_t *>(ggml_backend_buffer_get_base(signal));
    words[0] = words[1] = 0;
    std::atomic<int64_t> released{0};
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        released.store(ggml_time_us());
        __atomic_store_n(words, 1, __ATOMIC_RELEASE);
    });
    const bool gate = ggml_backend_cuda_moe_gate(fixture.gpu, words, 1, words + 1);
    check(gate, "history test queues a real delayed GPU gate");
    auto * counts = static_cast<uint32_t *>(ggml_backend_buffer_get_base(f.histogram));
    ggml_backend_cuda_moe_histogram(fixture.gpu, f.tasks[fixture.layer]->up->src[3],
            counts + fixture.layer * fixture.experts, fixture.experts);
    f.histogram_dirty = true;
    resident_history_snapshot(fixture.sched);
    resident_history_poll(fixture.sched);
    const auto returned = ggml_time_us();
    check(f.histogram_pending, "a delayed snapshot retains its pinned buffer ownership");
    release.join();
    check(returned < released.load(), "ordinary history polling never waits for the GPU");
    ggml_backend_synchronize(fixture.gpu);
    resident_history_poll(fixture.sched);
    check(!f.histogram_pending && words[1] == 0, "a completed snapshot retires without a gate timeout");
    ggml_backend_buffer_free(signal);
    check(ggml_backend_sched_graph_compute_async(fixture.sched, fixture.graph) == GGML_STATUS_SUCCESS,
            "teardown can own a queued asynchronous history snapshot");
    if (!failures) fprintf(stderr, "VERIFIED resident-history\n");
}

static void test_late_ring_cpu_overlap() {
    resident_cuda_fixture fixture(1, 2);
    auto & f = fixture.state();
    fixture.set_route(0, 2);
    check(ggml_backend_sched_graph_compute(fixture.sched, fixture.graph) == GGML_STATUS_SUCCESS,
            "partial-layer warmup succeeds before delaying the upload");
    fixture.set_route(1, 2);
    const auto expected_cpu = fixture.reference(fixture.cpu);
    const auto expected_gpu = fixture.reference(fixture.gpu);
    const auto previous_cpu = f.cpu_finished_us;
    const auto old_cpu_refs = f.cpu_refs, old_gpu_refs = f.gpu_refs;
    f.prefetcher = ggml_backend_cuda_init(0, nullptr, &f);
    GGML_ASSERT(f.prefetcher);
    auto * signal = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 2 * sizeof(uint32_t));
    GGML_ASSERT(signal);
    fixture.buffers.push_back(signal);
    auto * words = static_cast<uint32_t *>(ggml_backend_buffer_get_base(signal));
    words[0] = words[1] = 0;
    auto & ring = f.rings[0][0];
    ring.layer = fixture.layer; ring.expert = 1;
    ring.ready = ggml_backend_event_new(f.prefetcher);
    GGML_ASSERT(ring.ready);
    std::atomic<int64_t> released{0};
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        released.store(ggml_time_us());
        __atomic_store_n(words, 1, __ATOMIC_RELEASE);
    });
    check(ggml_backend_cuda_moe_gate(f.prefetcher, words, 1, words + 1),
            "ring test queues a real delayed upload");
    auto & task = *f.tasks[fixture.layer];
    const auto & group = f.groups[task.group];
    for (int b = 0; b < 3; ++b) {
        auto destination = *task.banks[b];
        resident_bank(f, task.group, b, destination);
        ggml_backend_tensor_set_async(f.prefetcher, &destination,
                static_cast<const uint8_t *>(task.banks[b]->data) + task.banks[b]->nb[2],
                group.slots.size() * group.strides[b], task.banks[b]->nb[2]);
        ring.data[b] = task.banks[b]->data;
    }
    ggml_backend_event_record(ring.ready);
    check(ggml_backend_sched_graph_compute(fixture.sched, fixture.graph) == GGML_STATUS_SUCCESS,
            "partial layer with a late ring upload succeeds");
    release.join();
    check(f.cpu_finished_us > previous_cpu && f.cpu_finished_us < released.load(),
            "CPU misses finish before the late ring upload is released");
    check(f.ring_late > 0 && f.ring_hits > 0 && words[1] == 0,
            "the delayed upload is consumed without a gate timeout");
    check(f.cpu_refs - old_cpu_refs == 1 && f.gpu_refs - old_gpu_refs == 1,
            "the late-upload layer executes one CPU row and one GPU row");
    fprintf(stderr, "ring overlap CPU finished minus release=%lld us\n",
            (long long)(f.cpu_finished_us - released.load()));
    std::vector<float> actual(expected_cpu.size());
    ggml_backend_tensor_get(fixture.output, actual.data(), 0, ggml_nbytes(fixture.output));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float expected = i < fixture.width ? expected_gpu[i] : expected_cpu[i];
        check(std::fabs(actual[i] - expected) <= 1e-5f + 1e-5f * std::fabs(expected),
                "late ring rows match their native GPU and CPU references");
    }
    if (!failures) fprintf(stderr, "VERIFIED ring-order\n");
}
#endif
