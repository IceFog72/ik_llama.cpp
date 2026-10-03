// White-box regression tests for scheduler-owned policy. Compile the actual
// implementation, rather than reimplementing admission in a policy simulator.
// Like other private-implementation tests, this target is not built on Windows.
#include "../ggml/src/ggml-backend.cpp"

static int failures = 0;

static void check(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

struct history_fixture {
    ggml_backend_sched sched{};
    ggml_tensor ids_tensor{};

    explicit history_fixture(int layers = 40) {
        sched.moe_resident_layers = layers;
    }

    ~history_fixture() {
        free(sched.moe_resident_route_use_counts);
        free(sched.moe_resident_route_last_seen);
        free(sched.moe_resident_route_layer_epochs);
        free(sched.moe_resident_hybrid_gpu_layers);
    }

    ggml_backend_sched_moe_route_state route(std::initializer_list<int32_t> experts, uint32_t frequency = 1) {
        ggml_backend_sched_moe_route_state result;
        result.n_expert = 8;
        result.last_ids_tensor = &ids_tensor;
        result.ordered_unique_ids = experts;
        result.frequencies.assign(8, 0);
        result.unique_ids.assign(1, 0);
        for (const auto expert : experts) {
            if (expert >= 0 && expert < result.n_expert) {
                result.frequencies[expert] = frequency;
                result.unique_ids[0] |= 1u << expert;
            }
        }
        return result;
    }

    void record(ggml_backend_sched_moe_route_state & route, int layer) {
        ggml_backend_sched_moe_resident_record_route(&sched, route, layer);
    }

    bool reused(int layer, int32_t expert) {
        return ggml_backend_sched_moe_resident_expert_is_reused(&sched, layer, expert);
    }
};

static void test_history() {
    history_fixture fixture;
    for (int step = 0; step < 2; ++step) {
        auto route = fixture.route({3});
        for (int layer = 0; layer < 40; ++layer) {
            fixture.record(route, layer);
            check(fixture.reused(layer, 3) == (step == 1), "consecutive layer visits establish reuse");
        }
    }
    auto route = fixture.route({5}, 100);
    fixture.record(route, 39);
    check(!fixture.reused(39, 5), "many tokens in one route do not establish temporal reuse");
    fixture.record(route, 39);
    check(!fixture.reused(39, 5), "up/gate/down copies of one route count only once");
    auto next_route = fixture.route({5});
    fixture.record(next_route, 39);
    check(fixture.reused(39, 5), "next visit to same layer establishes reuse");

    for (int i = 0; i < 9; ++i) {
        auto cold = fixture.route({7});
        fixture.record(cold, 39);
    }
    check(!fixture.reused(39, 5), "reuse expires after eight visits to that layer");
    auto expired = fixture.route({5});
    fixture.record(expired, 39);
    check(!fixture.reused(39, 5), "returning after expiry starts a fresh admission history");
    auto hot = fixture.route({5});
    fixture.record(hot, 39);
    check(fixture.reused(39, 5), "hot expert may be admitted again after expiry");
    check(!fixture.reused(-1, 5) && !fixture.reused(40, 5) && !fixture.reused(0, -1) && !fixture.reused(0, 8),
            "invalid history lookups are rejected");

    history_fixture rotating(1);
    for (int step = 0; step < 24; ++step) {
        auto item = rotating.route({step % 3});
        rotating.record(item, 0);
        check(rotating.reused(0, step % 3) == (step >= 3), "rotating working set establishes bounded reuse");
    }
}

static void test_history_boundaries() {
    history_fixture fixture(2);
    auto first = fixture.route({1});
    fixture.record(first, 0);
    fixture.record(first, 1);
    fixture.sched.moe_resident_route_layer_epochs[0] = std::numeric_limits<uint64_t>::max();
    auto rollover = fixture.route({1});
    fixture.record(rollover, 0);
    check(fixture.sched.moe_resident_route_layer_epochs[0] == 1 && !fixture.reused(0, 1),
            "layer epoch rollover resets that layer's admission history");
    fixture.record(rollover, 1);
    check(fixture.reused(1, 1), "epoch rollover preserves other layers' history");

    ggml_tensor next_ids{};
    rollover.last_ids_tensor = &next_ids;
    fixture.record(rollover, 0);
    check(fixture.reused(0, 1), "different route tensor is recorded even in one compute pass");
    fixture.sched.moe_resident_route_use_counts[1] = std::numeric_limits<uint32_t>::max();
    auto saturated = fixture.route({1});
    fixture.record(saturated, 0);
    check(fixture.sched.moe_resident_route_use_counts[1] == 2, "history count is bounded without overflow");
    check(ggml_backend_sched_moe_resident_ensure_route_use_counts(&fixture.sched, 2, 9),
            "changed expert geometry reallocates history");
    check(!fixture.reused(0, 1), "new geometry clears stale admission state");
    check(!ggml_backend_sched_moe_resident_ensure_route_use_counts(&fixture.sched, 0, 8) &&
            !ggml_backend_sched_moe_resident_ensure_route_use_counts(&fixture.sched, 2, -1),
            "invalid geometry is rejected");
}

static void test_identity() {
    ggml_tensor tensor{};
    int layer = -1;
    ggml_set_name(&tensor, "blk.39.ffn_down_exps.weight");
    check(ggml_backend_sched_moe_parse_layer_id(&tensor, 40, layer) && layer == 39,
            "layer identity comes from the tensor name");
    check(ggml_backend_sched_moe_bank_role_from_tensor(&tensor) == GGML_MOE_BANK_DOWN,
            "down bank role is recognized");
    for (const auto name : {"blk.-1.ffn_down_exps.weight", "blk.40.ffn_down_exps.weight",
            "blk.9999999999999999999.ffn_down_exps.weight", "other.1.ffn_down_exps.weight"}) {
        ggml_set_name(&tensor, name);
        check(!ggml_backend_sched_moe_parse_layer_id(&tensor, 40, layer), "invalid layer identity is rejected");
    }
    ggml_set_name(&tensor, "blk.1.ffn_unknown_exps.weight");
    check(ggml_backend_sched_moe_bank_role_from_tensor(&tensor) == GGML_MOE_BANK_UNKNOWN,
            "unknown bank role bypasses residency");
}

static void test_hybrid_placement() {
    // B4 hybrid gate: empty plan and CPU-marked layers keep baseline
    // placement; only GPU-marked layers force CUDA. Determines determinism
    // for -cmoe (CPU MoE compute) mixed with hot-layer GPU staging.
    history_fixture fixture(4);
    ggml_tensor up{};
    ggml_tensor down{};
    ggml_tensor bad{};
    ggml_set_name(&up, "blk.2.ffn_up_exps.weight");
    ggml_set_name(&down, "blk.1.ffn_down_exps.weight");
    ggml_set_name(&bad, "other.0.ffn_up_exps.weight");
    up.op = GGML_OP_MUL_MAT_ID;
    down.op = GGML_OP_MOE_FUSED_UP_GATE;
    bad.op = GGML_OP_MUL_MAT_ID;
    auto * backend = reinterpret_cast<ggml_backend_t>(0x1);
    fixture.sched.moe_resident_slots = -1;
#ifdef GGML_USE_CUDA
    // Without the test CUDA backend pointer there is no residency request;
    // exercise the planner state machine directly instead.
    check(!ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, 5, true),
            "planner rejects layers outside model geometry");
#endif
    check(ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, 2, true),
            "hot layer is marked GPU-bound");
    check(ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, 1, false),
            "cold layer stays CPU-bound");
    check(fixture.sched.moe_resident_hybrid_layers == 4 &&
            fixture.sched.moe_resident_hybrid_gpu_layers[2] == 1 &&
            fixture.sched.moe_resident_hybrid_gpu_layers[1] == 0 &&
            fixture.sched.moe_resident_hybrid_gpu_layers[0] == 0,
            "planner records per-layer GPU/CPU intent without touching pools");
    check(!ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, -1, true),
            "planner rejects negative layers");
    // A runtime demotion on an allocated graph must request a fresh caller
    // graph. An idle scheduler has no allocated graph to invalidate.
    fixture.sched.is_alloc = true;
    fixture.sched.moe_resident_hybrid_plan_ready = true;
    check(!fixture.sched.moe_resident_replan_required,
            "allocated graph starts with no pending replan");
    check(ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, 2, false),
            "runtime failure demotes a preplanned GPU layer");
    check(fixture.sched.moe_resident_replan_required,
            "demotion on an allocated graph requests a fresh graph");
    fixture.sched.is_alloc = false;
    fixture.sched.moe_resident_replan_required = false;
    check(ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, 1, true),
            "idle scheduler accepts a promotion");
    check(!fixture.sched.moe_resident_replan_required,
            "demotion without an allocated graph needs no re-split");
    fixture.sched.is_alloc = false;
    fixture.sched.moe_resident_hybrid_plan_ready = false;
    fixture.sched.moe_resident_replan_required = false;
    (void) backend;
}

static void test_cpu_repeated_routes() {
    auto cpu = ggml_backend_cpu_init();
    auto sched = ggml_backend_sched_new(&cpu, nullptr, 1, 64, false);
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    auto bank = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 4);
    auto act = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 4, 1);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
    ggml_tensor * inputs[] = {bank, act, ids};
    ggml_backend_buffer_t buffers[3];
    for (int i = 0; i < 3; ++i) {
        buffers[i] = ggml_backend_alloc_buffer(cpu, ggml_nbytes(inputs[i]));
        GGML_ASSERT(buffers[i]);
        ggml_backend_tensor_alloc(buffers[i], inputs[i], ggml_backend_buffer_get_base(buffers[i]));
    }
    std::vector<float> weights(16 * 16 * 4), activations(16 * 4);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(i / (16 * 16) + 1);
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = float(i / 16 + 1);
    ggml_backend_tensor_set(bank, weights.data(), 0, ggml_nbytes(bank));
    ggml_backend_tensor_set(act, activations.data(), 0, ggml_nbytes(act));
    for (const std::array<int32_t, 4> route : {std::array<int32_t, 4>{0, 0, 1, 2},
            std::array<int32_t, 4>{3, 3, 3, 3}, std::array<int32_t, 4>{-1, 0, 0, 2}}) {
        ggml_backend_tensor_set(ids, route.data(), 0, ggml_nbytes(ids));
        auto out = ggml_mul_mat_id(ctx, bank, act, ids);
        auto graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, out);
        check(ggml_backend_sched_alloc_graph(sched, graph), "CPU repeated-route graph allocates");
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "CPU repeated-route graph computes");
        std::vector<float> values(ggml_nelements(out));
        ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
        for (size_t row = 0; row < route.size(); ++row) {
            const float expected = route[row] < 0 ? 0.0f : 16.0f * (route[row] + 1) * (row + 1);
            for (size_t col = 0; col < 16; ++col) check(values[row * 16 + col] == expected,
                    "CPU repeated and masked routes match independent dot-product oracle");
        }
        ggml_backend_sched_reset(sched);
    }
    ggml_backend_sched_free(sched);
    for (auto buffer : buffers) ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

static void test_cpu_quantized_masked_rows() {
    constexpr int width = 256, rows = 16, experts = 4, top_k = 4;
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        for (const int tokens : {1, 2}) for (const int input_rows : {1, 2, top_k}) {
            auto ctx = ggml_init({1024 * 1024, nullptr, true});
            auto bank = ggml_new_tensor_3d(ctx, type, width, rows, experts);
            auto act = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, input_rows, tokens);
            auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, tokens);
            auto out = ggml_mul_mat_id(ctx, bank, act, ids);
            auto graph = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(graph, out);
            auto reference_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, tokens);
            auto reference_out = ggml_mul_mat_id(ctx, bank, act, reference_ids);
            auto reference_graph = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(reference_graph, reference_out);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            GGML_ASSERT(buffer);
            std::vector<float> weights(ggml_nelements(bank));
            for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(i / (width * rows) + 1);
            std::vector<uint8_t> quantized(ggml_nbytes(bank));
            ggml_quantize_chunk(type, weights.data(), quantized.data(), 0,
                    ggml_nrows(bank), width, nullptr, nullptr);
            ggml_backend_tensor_set(bank, quantized.data(), 0, quantized.size());
            const int32_t routes[][top_k] = {
                {0, 1, 2, 3}, {-1, 2, experts, 2}, {-1, experts, -1, experts}, {3, 3, 1, 0},
            };
            std::vector<int32_t> route(top_k * tokens);
            std::vector<float> activation(ggml_nelements(act)), values(ggml_nelements(out)), reference(values.size());
            for (int pass = 0; pass < 4; ++pass) {
                for (int token = 0; token < tokens; ++token) {
                    std::copy(routes[pass], routes[pass] + top_k, route.begin() + token * top_k);
                    for (int row = 0; row < input_rows; ++row)
                        std::fill_n(activation.begin() + (token * input_rows + row) * width,
                                width, float(1 + pass + token * input_rows + row));
                }
                ggml_backend_tensor_set(act, activation.data(), 0, ggml_nbytes(act));
                ggml_backend_tensor_set(ids, route.data(), 0, ggml_nbytes(ids));
                check(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS,
                        "quantized masked rows compute on a reused graph");
                ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
                // Quantization rounds even constant rows. Compare against the
                // native unmasked path, replacing invalid experts with zero.
                // Run it after the masked graph so it cannot prime that graph's
                // scratch with the current activations and hide a reuse bug.
                auto valid_route = route;
                for (auto & expert : valid_route) if (expert < 0 || expert >= experts) expert = 0;
                ggml_backend_tensor_set(reference_ids, valid_route.data(), 0, ggml_nbytes(reference_ids));
                check(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS,
                        "native unmasked quantized reference computes");
                ggml_backend_tensor_get(reference_out, reference.data(), 0, ggml_nbytes(reference_out));
                bool correct = true;
                for (int token = 0; token < tokens; ++token) for (int row = 0; row < top_k; ++row) {
                    const int expert = route[token * top_k + row];
                    for (int col = 0; col < rows; ++col) {
                        const size_t index = (token * top_k + row) * rows + col;
                        const float expected = expert < 0 || expert >= experts ? 0.0f : reference[index];
                        correct = correct && std::abs(values[index] - expected)
                                <= 1e-5f + 1e-5f * std::abs(expected);
                    }
                }
                if (!correct) fprintf(stderr, "quantized mask oracle type=%s tokens=%d input_rows=%d pass=%d first=%.9g\n",
                        ggml_type_name(type), tokens, input_rows, pass, values[0]);
                check(correct, "quantized masked, repeated, shared and batched rows match native unmasked oracle");
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(cpu);
}

#ifdef GGML_USE_CUDA
static void test_cuda_rebuild(ggml_backend_t cuda, bool allocation_failure) {
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {cuda, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false);
    ggml_backend_sched_set_moe_resident_model_info(sched, 1, 8, 1);
    sched->moe_resident_slots = -1;
    sched->moe_resident_budget[0].initialized = true;
    sched->moe_resident_budget[0].budget_bytes = 1024 * 1024;
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    auto input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 1, 1);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_set_input(input);
    ggml_set_input(ids);
    ggml_tensor * banks[3];
    ggml_tensor * tensors[5] = {input, ids};
    ggml_backend_buffer_t buffers[5];
    const char * names[] = {"blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"};
    for (int i = 0; i < 3; ++i) {
        banks[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
        ggml_set_name(banks[i], names[i]);
        tensors[i + 2] = banks[i];
    }
    for (int i = 0; i < 5; ++i) {
        buffers[i] = ggml_backend_alloc_buffer(cpu, ggml_nbytes(tensors[i]));
        GGML_ASSERT(buffers[i] != nullptr);
        ggml_backend_tensor_alloc(buffers[i], tensors[i], ggml_backend_buffer_get_base(buffers[i]));
        ggml_backend_buffer_clear(buffers[i], 0);
    }
    float values[16];
    std::fill_n(values, 16, 1.0f);
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    for (int i = 0; i < 3; ++i) {
        ggml_backend_buffer_set_usage(buffers[i + 2], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> weights(16 * 16 * 8, float(i + 1));
        ggml_backend_tensor_set(banks[i], weights.data(), 0, weights.size() * sizeof(float));
    }
    auto build = [&](ggml_tensor ** outputs) {
        auto graph = ggml_new_graph_custom(ctx, 128, false);
        for (int i = 0; i < 3; ++i) {
            outputs[i] = ggml_mul_mat_id(ctx, banks[i], input, ids);
            ggml_set_output(outputs[i]);
            ggml_build_forward_expand(graph, outputs[i]);
        }
        return graph;
    };
    auto verify = [&](ggml_tensor ** outputs, ggml_backend_t expected_backend) {
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_get(outputs[i], values, 0, sizeof(values));
            check(std::all_of(values, values + 16, [i](float value) { return value == 16 * (i + 1); }),
                    "rebuild preserves all expert outputs without source aliasing");
            check(ggml_backend_sched_get_tensor_backend(sched, outputs[i]) == expected_backend,
                    "residency fallback uses the expected graph placement");
        }
    };
    ggml_tensor * outputs[3];
    auto graph = build(outputs);
    check(ggml_backend_sched_alloc_graph(sched, graph), "rebuild fixture allocates the original CUDA graph");
    auto buft = ggml_backend_get_default_buffer_type(cuda);
    auto saved_alloc = buft->iface.alloc_buffer;
    if (allocation_failure) {
        buft->iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
    }
    const auto status = ggml_backend_sched_graph_compute(sched, graph);
    buft->iface.alloc_buffer = saved_alloc;
    check(status == GGML_STATUS_SUCCESS, "cache allocation failure preserves the current compute");
    verify(outputs, cuda);
    if (!allocation_failure) {
        check(!ggml_backend_sched_moe_resident_needs_rebuild(sched), "normal compute does not request a rebuild");
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "normal graph reuse succeeds");
        verify(outputs, cuda);
        ggml_backend_sched_moe_resident_hybrid_set_layer(sched, 0, false);
    }
    check(ggml_backend_sched_moe_resident_needs_rebuild(sched), "runtime demotion requests a caller graph rebuild");
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "old graph remains safe until caller rebuilds");
    verify(outputs, cuda);
    check(ggml_backend_sched_moe_resident_needs_rebuild(sched), "old graph compute cannot consume the pending rebuild");
    ggml_backend_sched_reset(sched);
    check(ggml_backend_sched_moe_resident_needs_rebuild(sched), "reset preserves the pending rebuild");
    graph = build(outputs);
    check(ggml_backend_sched_alloc_graph(sched, graph), "fresh graph allocates after demotion");
    check(!ggml_backend_sched_moe_resident_needs_rebuild(sched), "fresh graph allocation completes the rebuild");
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "fresh CPU graph computes successfully");
    verify(outputs, cpu);
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "fresh CPU graph can be reused");
    verify(outputs, cpu);
    ggml_backend_sched_free(sched);
    for (auto buffer : buffers) ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

enum class decode_case { ordinary, fused, fused_tail, combined, scratch_failure, cache_failure };

static void test_cuda_mixed_decode(ggml_backend_t cuda, decode_case kind = decode_case::ordinary, int64_t top_k = 8) {
    const bool combined = kind == decode_case::combined;
    const bool fused = kind == decode_case::fused || kind == decode_case::fused_tail || combined;
    const bool tail = kind == decode_case::fused_tail;
    constexpr int64_t input_width = 256;
    constexpr int64_t expert_width = 512;
    const int64_t n_expert = 2 * top_k;
    constexpr int n_passes = 3;
    const char * bank_names[] = {
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_down_exps.weight",
    };

    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {cuda, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false);
    ggml_backend_sched_set_moe_resident_model_info(sched, 1, n_expert, top_k);
    ggml_backend_sched_set_moe_resident(sched, -1);
    // Fix only the working-set budget for this small fixture. It reserves
    // enough space for all three native Q4_0 banks without depending on free VRAM.
    sched->moe_resident_budget[0].initialized = true;
    sched->moe_resident_budget[0].budget_bytes = 32 * 1024 * 1024;
    sched->moe_decode = new ggml_backend_sched_moe_decode;
    // Exercise the deterministic mixed branch with a supplied measurement.
    // This deliberately does not claim that calibration selected this ratio.
    sched->moe_decode->measurements.push_back(
            {GGML_TYPE_Q4_0, GGML_OP_MUL_MAT_ID, input_width, expert_width, 0.2});
    sched->moe_decode->measurements.push_back(
            {GGML_TYPE_Q4_0, GGML_OP_MUL_MAT_ID, expert_width, input_width, 0.2});

    sched->moe_decode->measurements.push_back(
            {GGML_TYPE_Q4_0, GGML_OP_MOE_FUSED_UP_GATE, input_width, combined ? 2 * expert_width : expert_width, 0.2});
    auto ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    ggml_tensor * banks[3] = {
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, input_width, combined ? 2 * expert_width : expert_width, n_expert),
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, input_width, expert_width, n_expert),
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, expert_width, input_width, n_expert),
    };
    for (int i = 0; i < 3; ++i) ggml_set_name(banks[i], bank_names[i]);
    if (combined) ggml_set_name(banks[0], "blk.0.ffn_gate_up_exps.weight");

    ggml_tensor * activations[3] = {
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_width, fused ? 1 : top_k, 1),
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_width, fused ? 1 : top_k, 1),
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, expert_width, top_k, 1),
    };
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ggml_set_input(ids);
    std::vector<int32_t> route_ids(size_t(top_k), 0);
    const int32_t pattern[] = {0, 1, 2, 3, 0, 1, 4, 4};
    for (size_t i = 0; i < route_ids.size(); ++i) route_ids[i] = pattern[i % 8];
    ggml_backend_buffer_t source_buffers[7]{};
    for (int i = 0; i < 3; ++i) {
        source_buffers[i] = ggml_backend_alloc_buffer(cpu, ggml_nbytes(banks[i]));
        GGML_ASSERT(source_buffers[i] != nullptr);
        ggml_backend_tensor_alloc(source_buffers[i], banks[i], ggml_backend_buffer_get_base(source_buffers[i]));
        ggml_backend_buffer_set_usage(source_buffers[i], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<float> values(ggml_nelements(banks[i]));
        for (size_t j = 0; j < values.size(); ++j) {
            const int centered = static_cast<int>((j * 17 + size_t(i) * 13) % 29) - 14;
            values[j] = 0.04f * centered;
        }
        std::vector<uint8_t> quantized(ggml_nbytes(banks[i]));
        ggml_quantize_chunk(banks[i]->type, values.data(), quantized.data(), 0,
                ggml_nrows(banks[i]), banks[i]->ne[0], nullptr, nullptr);
        ggml_backend_tensor_set(banks[i], quantized.data(), 0, quantized.size());
    }
    for (int i = 0; i < 3; ++i) {
        source_buffers[i + 3] = ggml_backend_alloc_buffer(cpu, ggml_nbytes(activations[i]));
        GGML_ASSERT(source_buffers[i + 3] != nullptr);
        ggml_backend_tensor_alloc(source_buffers[i + 3], activations[i],
                ggml_backend_buffer_get_base(source_buffers[i + 3]));
        std::vector<float> values(ggml_nelements(activations[i]));
        for (size_t j = 0; j < values.size(); ++j) {
            values[j] = 0.1f * (static_cast<int>((j * 11 + size_t(i) * 7) % 17) - 8);
        }
        ggml_backend_tensor_set(activations[i], values.data(), 0, values.size() * sizeof(float));
    }
    source_buffers[6] = ggml_backend_alloc_buffer(cpu, ggml_nbytes(ids));
    GGML_ASSERT(source_buffers[6] != nullptr);
    ggml_backend_tensor_alloc(source_buffers[6], ids, ggml_backend_buffer_get_base(source_buffers[6]));
    ggml_backend_tensor_set(ids, route_ids.data(), 0, route_ids.size() * sizeof(int32_t));

    auto make_output = [&](ggml_context * graph_ctx, int bank) {
        auto * output = fused && bank == 0
                ? ggml_moe_up_gate(graph_ctx, banks[0], combined ? nullptr : banks[1], activations[0], ids, GGML_UNARY_OP_SILU)
                : ggml_mul_mat_id(graph_ctx, banks[bank], activations[bank], ids);
        if (tail) output = ggml_sqr(graph_ctx, output);
        ggml_set_output(output);
        return output;
    };
    auto make_graph = [&](ggml_context * graph_ctx, ggml_tensor * output) {
        auto * graph = ggml_new_graph_custom(graph_ctx, 128, false);
        ggml_build_forward_expand(graph, output);
        return graph;
    };

    // Reserve the complete layer together so the planner sees UP, GATE, and
    // DOWN geometry, while each real matmul gets its own mixed-decode split.
    auto plan_ctx = ggml_init({1024 * 1024, nullptr, true});
    auto empty_graph = ggml_new_graph_custom(plan_ctx, 128, false);
    ggml_backend_sched_moe_resident_hybrid_plan_graph(sched, empty_graph);
    check(!sched->moe_resident_hybrid_plan_ready, "empty graph does not seal an auto placement plan");
    auto plan_graph = ggml_new_graph_custom(plan_ctx, 128, false);
    for (int i = 0; i < 3; ++i) if (!combined || i != 1) ggml_build_forward_expand(plan_graph, make_output(ctx, i));
    check(ggml_backend_sched_reserve(sched, plan_graph), "mixed decode fixture reserves all three bank roles");
    check(sched->moe_resident_hybrid_plan_ready && sched->moe_resident_hybrid_gpu_layers[0] != 0,
            "complete quantized layer is planned for GPU staging");

    auto cpu_sched = ggml_backend_sched_new(&cpu, nullptr, 1, 128, false);
    std::vector<std::vector<float>> expected(3);
    for (int i = 0; i < 3; ++i) {
        if (combined && i == 1) continue;
        auto * output = make_output(ctx, i);
        auto * graph = make_graph(ctx, output);
        check(ggml_backend_sched_alloc_graph(cpu_sched, graph), "full CPU reference graph allocates");
        check(ggml_backend_sched_graph_compute(cpu_sched, graph) == GGML_STATUS_SUCCESS,
                "full CPU reference graph computes");
        expected[i].resize(ggml_nelements(output));
        ggml_backend_tensor_get(output, expected[i].data(), 0, ggml_nbytes(output));
        ggml_backend_sched_reset(cpu_sched);
    }

    auto gpu_sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false);
    ggml_backend_sched_set_only_active_experts(gpu_sched, true);
    std::vector<std::vector<float>> gpu_expected(3);
    for (int i = 0; i < 3; ++i) {
        if (combined && i == 1) continue;
        auto * output = make_output(ctx, i);
        auto * graph = make_graph(ctx, output);
        ggml_backend_sched_set_tensor_backend(gpu_sched, output, cuda);
        if (tail) ggml_backend_sched_set_tensor_backend(gpu_sched, output->src[0], cuda);
        check(ggml_backend_sched_alloc_graph(gpu_sched, graph), "full CUDA reference graph allocates");
        check(ggml_backend_sched_graph_compute(gpu_sched, graph) == GGML_STATUS_SUCCESS,
                "full CUDA reference graph computes");
        gpu_expected[i].resize(ggml_nelements(output));
        ggml_backend_tensor_get(output, gpu_expected[i].data(), 0, ggml_nbytes(output));
        ggml_backend_sched_reset(gpu_sched);
    }

    auto verify = [&](ggml_tensor * output, int bank, int pass, bool mixed) {
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        size_t bad = 0;
        float max_error = 0.0f;
        size_t worst = 0;
        std::vector<float> row_max(static_cast<size_t>(top_k), 0.0f);
        for (size_t j = 0; j < actual.size(); ++j) {
            const size_t row = j / size_t(output->ne[0]);
            const bool gpu_row = ggml_backend_sched_get_tensor_backend(sched, output) == cuda &&
                    (!mixed || sched->moe_decode->gpu_ids[row] >= 0);
            const auto & oracle = gpu_row ? gpu_expected[bank] : expected[bank];
            const float error = std::fabs(actual[j] - oracle[j]);
            const float tolerance = 1e-5f + 1e-5f * std::fabs(oracle[j]);
            if (error > max_error) { max_error = error; worst = j; }
            row_max[j / static_cast<size_t>(output->ne[0])] = std::max(
                    row_max[j / static_cast<size_t>(output->ne[0])], error);
            if (!(error <= tolerance)) ++bad;
        }
        if (bad) {
            fprintf(stderr, "mixed Q4_0 pass=%d bank=%d mismatches=%zu/%zu worst[%zu] got=%g expected=%g max_error=%g row_errors:",
                    pass, bank, bad, actual.size(), worst, actual[worst], expected[bank][worst], max_error);
            for (int row = 0; row < top_k; ++row) fprintf(stderr, " id%d=%.4g", route_ids[row], row_max[row]);
            fprintf(stderr, "\n");
        }
        check(bad == 0, "mixed CPU/GPU Q4_0 rows match their native backend reference");
        if (kind != decode_case::cache_failure) check(ggml_backend_sched_get_tensor_backend(sched, output) == cuda,
                "mixed decode retains CUDA output placement");
    };

    for (int pass = 0; pass < n_passes; ++pass) {
        for (int i = 0; i < 3; ++i) {
            if (combined && i == 1) continue;
            auto * output = make_output(ctx, i); // a fresh caller graph for every allocation
            auto * graph = make_graph(ctx, output);
            ggml_backend_sched_reset(sched);
            check(ggml_backend_sched_alloc_graph(sched, graph), "mixed decode graph allocates");
            auto buft = sched->bufts[0];
            auto original_alloc = buft->iface.alloc_buffer;
            auto failing_buft = *buft;
            const auto operators_before = sched->moe_decode->operators;
            if (kind == decode_case::scratch_failure) {
                failing_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
                sched->bufts[0] = &failing_buft;
            }
            static decltype(original_alloc) saved_alloc;
            if (kind == decode_case::cache_failure && pass == 0 && i == 0) {
                saved_alloc = original_alloc;
                buft->iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t bytes) -> ggml_backend_buffer_t {
                    return bytes > 1024 ? nullptr : saved_alloc(type, bytes);
                };
            }
            check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    "mixed decode graph computes");
            sched->bufts[0] = buft;
            buft->iface.alloc_buffer = original_alloc;
            const bool mixed = operators_before != sched->moe_decode->operators;
            verify(output, i, pass, mixed);
            if (kind == decode_case::scratch_failure) check(!mixed && !sched->moe_decode->prepared_split &&
                    sched->moe_decode->activation && !sched->moe_decode->device_ids,
                    "partial decode scratch allocation failure uses ordinary CUDA compute");
            if (kind == decode_case::cache_failure && pass == 0 && i == 0) check(mixed &&
                    ggml_backend_sched_moe_resident_needs_rebuild(sched),
                    "mixed cache allocation failure retains correct current rows and requests fresh graph");
        }
    }
    if (kind != decode_case::scratch_failure) {
        check(sched->moe_decode->cpu_refs > 0, "mixed decode executes references on CPU");
        check(sched->moe_decode->gpu_refs > 0, "mixed decode executes references on GPU");
        check(sched->moe_decode->avoided_bytes > 0, "mixed decode avoids complete-bank H2D transfers");
    }
    uint64_t resident_hits = 0;
    for (int p = 0; p < sched->moe_resident[0].n_pools; ++p) {
        auto & pool = sched->moe_resident[0].pools[p];
        resident_hits += pool.hits;
        if (pool.buffer) check(pool.layer_slots[0] == 2 * top_k,
                "auto pool honors planned per-layer capacity, including top-k above sixteen");
    }
    if (kind != decode_case::cache_failure) check(resident_hits > 0, "repeated mixed decode reuses resident expert rows");

    ggml_backend_sched_free(sched);
    ggml_backend_sched_free(cpu_sched);
    ggml_backend_sched_free(gpu_sched);
    for (auto buffer : source_buffers) ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(plan_ctx);
    ggml_free(ctx);
}

static void test_route_ingestion(ggml_backend_t cuda) {
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {cuda, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false);
    ggml_backend_sched_set_only_active_experts(sched, true);
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    auto bank = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 8);
    auto input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 6, 1);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, 1);
    ggml_set_name(bank, "blk.0.ffn_up_exps.weight");
    ggml_set_input(input);
    ggml_set_input(ids);
    ggml_backend_buffer_t buffers[3]{};
    ggml_tensor * tensors[] = {bank, input, ids};
    bool allocated = true;
    for (int i = 0; i < 3; ++i) {
        buffers[i] = ggml_backend_alloc_buffer(cpu, ggml_nbytes(tensors[i]));
        allocated = allocated && buffers[i] != nullptr;
        if (buffers[i] != nullptr) {
            ggml_backend_tensor_alloc(buffers[i], tensors[i], ggml_backend_buffer_get_base(buffers[i]));
            ggml_backend_buffer_clear(buffers[i], 0);
        }
    }
    check(allocated, "route fixture allocates host inputs");
    if (allocated) {
        ggml_backend_buffer_set_usage(buffers[0], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto output = ggml_mul_mat_id(ctx, bank, input, ids);
        auto graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(sched, output, cuda);
        allocated = ggml_backend_sched_alloc_graph(sched, graph);
        check(allocated, "route fixture allocates real cross-backend scheduler splits");
        if (allocated) {
            for (const int32_t first : {3, -1, 8, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()}) {
                const int32_t raw[] = {first, 0, 3, 7, 0, 7};
                ggml_backend_tensor_set(ids, raw, 0, sizeof(raw));
                ggml_backend_sched_moe_route_state route;
                std::array<bool, GGML_SCHED_MAX_BACKENDS> sync{};
                // Exercise route ingestion and shared-tensor reuse, but never
                // execute an expert kernel with intentionally invalid IDs.
                for (int pass = 0; pass < 2; ++pass) {
                    for (int split = 0; split < sched->n_splits; ++split) {
                        ggml_backend_sched_copy_inputs(sched, &sched->splits[split], sync, route);
                    }
                    ggml_backend_sched_synchronize(sched);
                }
                check(route.tensor_reads == 1 && route.tensor_syncs == 1 && route.tensor_reuses == 1,
                        "shared route is read/synchronized once and reused on the next split call");
                check(route.valid == (first == 3) && route.invalid_ids == (first == 3 ? 0 : 1),
                        "signed/sentinel/out-of-range IDs are rejected before bitset indexing");
                if (first == 3) {
                    check(route.ordered_unique_ids == std::vector<int32_t>{3, 0, 7}, "decoded IDs preserve first-occurrence order");
                    check(route.unique_ids.size() == 1 && route.unique_ids[0] == ((1u << 3) | 1u | (1u << 7)),
                            "decoded bitset and ordered IDs agree");
                    check(route.frequencies[0] == 2 && route.frequencies[3] == 2 && route.frequencies[7] == 2,
                            "decoded frequencies count raw route occurrences");
                }
            }
        }
    }
    ggml_backend_sched_free(sched);
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

static void test_cuda_copy() {
    auto backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    check(backend != nullptr, "CUDA backend initializes");
    if (backend == nullptr) {
        return;
    }
    test_route_ingestion(backend);
    test_cuda_rebuild(backend, false);
    test_cuda_rebuild(backend, true);
    test_cuda_mixed_decode(backend);
    test_cuda_mixed_decode(backend, decode_case::fused);
    test_cuda_mixed_decode(backend, decode_case::ordinary, 16);
    test_cuda_mixed_decode(backend, decode_case::ordinary, 24);
    test_cuda_mixed_decode(backend, decode_case::fused_tail);
    test_cuda_mixed_decode(backend, decode_case::combined);
    test_cuda_mixed_decode(backend, decode_case::scratch_failure);
    test_cuda_mixed_decode(backend, decode_case::cache_failure);
    constexpr size_t expert_size = 1024;
    constexpr size_t bank_bytes = 8 * expert_size;
    auto buffer = ggml_backend_alloc_buffer(backend, bank_bytes);
    check(buffer != nullptr, "CUDA staging buffer allocates");
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        return;
    }
    std::vector<uint8_t> host(bank_bytes);
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i / expert_size + i % 251);
    }
    auto input = ggml_backend_sched_moe_resident_byte_tensor(nullptr, host.data(), bank_bytes);
    ggml_set_name(&input, "blk.0.ffn_up_exps.weight");
    auto destination = ggml_backend_sched_moe_resident_byte_tensor(buffer, ggml_backend_buffer_get_base(buffer), bank_bytes);
    history_fixture fixture(1);
    fixture.sched.moe_resident_slots = 2;
    fixture.sched.moe_resident_n_expert_used = 1;
    fixture.sched.n_backends = 1;
    fixture.sched.backends[0] = backend;

    const auto copy = [&](ggml_backend_sched_moe_route_state & route) {
        const bool handled = ggml_backend_sched_moe_resident_copy(&fixture.sched, backend, &input, &destination,
                buffer, route, expert_size, GGML_MOE_BANK_UP);
        ggml_backend_synchronize(backend);
        return handled;
    };
    const auto verify_bytes = [&](const ggml_backend_sched_moe_route_state & route) {
        std::vector<uint8_t> actual(bank_bytes);
        ggml_backend_tensor_get(&destination, actual.data(), 0, bank_bytes);
        for (const auto expert : route.ordered_unique_ids) {
            const size_t offset = static_cast<size_t>(expert) * expert_size;
            const size_t bytes = expert < 7 ? expert_size + 512 : expert_size;
            check(std::equal(host.begin() + offset, host.begin() + offset + bytes, actual.begin() + offset),
                    "active expert and padding bytes equal source for mixed hit/miss copy");
        }
    };

    auto first = fixture.route({6, 1});
    const bool first_handled = copy(first);
    check(first_handled, "initial resident copy is handled");
    if (!first_handled) {
        for (int i = 0; i < fixture.sched.moe_resident[0].n_pools; ++i) {
            ggml_backend_sched_moe_resident_clear_storage(&fixture.sched.moe_resident[0].pools[i]);
        }
        free(fixture.sched.moe_resident[0].pools);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return;
    }
    verify_bytes(first);
    auto & pool = fixture.sched.moe_resident[0].pools[0];
    check(pool.entries[0].expert == 6 && pool.entries[1].expert == 1, "admissions preserve route order");
    for (int slot = 0; slot < pool.slots; ++slot) {
        pool.last_used[slot] = pool.entries[slot].expert == 6 ? 1 : 2;
    }
    pool.clock = 2;

    // Make expert 2 hot, then miss on it before touching the older active 6.
    // Evicting 6 would turn that later hit into an unnecessary H2D transfer.
    auto prime = fixture.route({2});
    fixture.record(prime, 0);
    auto mixed = fixture.route({2, 6});
    const auto hits_before = pool.hits;
    check(copy(mixed), "mixed resident copy is handled");
    verify_bytes(mixed);
    check(pool.hits == hits_before + 1, "miss does not evict an expert active in the same route");
    check(pool.evictions == 1, "hot miss evicts only an inactive resident");

    auto cold = fixture.route({7, 6});
    const auto admissions_before = pool.admissions;
    const auto evictions_before = pool.evictions;
    check(copy(cold), "cold miss takes direct H2D with active resident hit");
    verify_bytes(cold);
    check(pool.admissions == admissions_before && pool.evictions == evictions_before,
            "cold miss does not churn full cache");

    for (const auto invalid_id : {-1, 8}) {
        auto invalid = fixture.route({invalid_id});
        const auto before_invalid = pool.lookups;
        check(!copy(invalid), "invalid ordered route ID bypasses residency");
        check(pool.lookups == before_invalid, "invalid ID cannot queue partial copies");
    }

    auto oversized = fixture.route({0, 1, 2});
    const auto lookups_before = pool.lookups;
    check(!copy(oversized), "oversized working set bypasses residency");
    check(pool.lookups == lookups_before, "oversized bypass does not partially mutate cache");

    // Exercise the actual allocator seam with tiny pools. Every bank consumes
    // the same backend budget, and deliberately failed allocation is bounded.
    auto buft = ggml_backend_buffer_get_type(buffer);
    ggml_backend_sched_moe_resident_budget budget{};
    budget.initialized = true;
    budget.budget_bytes = 2 * 4 * (expert_size + 512);
    ggml_backend_sched_moe_resident_pool banks[3]{};
    for (int bank = 0; bank < 2; ++bank) {
        check(ggml_backend_sched_moe_resident_prepare(&banks[bank], backend, buft,
                4, 3, 2, expert_size + 512, &budget), "two bank pools share one auto budget");
        check(banks[bank].layer_slots != nullptr && banks[bank].cached_layers == 2 && banks[bank].layer_slots[0] == 2 &&
                banks[bank].layer_slots[1] == 2 && banks[bank].layer_slots[2] == 0,
                "quota selects viable cached layers rather than undersized slots everywhere");
    }
    check(budget.allocated_bytes == budget.budget_bytes, "shared budget accounts for both bank allocations");
    check(!ggml_backend_sched_moe_resident_prepare(&banks[2], backend, buft,
            1, 3, 1, expert_size + 512, &budget), "third pool cannot multiply exhausted budget");
    check(budget.unavailable && banks[2].buffer == nullptr, "budget exhaustion disables auto safely");
    for (auto & bank : banks) {
        ggml_backend_sched_moe_resident_clear_storage(&bank, &budget);
    }
    check(budget.allocated_bytes == 0, "freeing pools releases shared budget accounting");

    ggml_backend_sched_moe_resident_pool failed_pool{};
    ggml_backend_sched_moe_resident_budget failed_budget{};
    failed_budget.initialized = true;
    failed_budget.budget_bytes = 4096;
    auto failed_buft = *buft;
    failed_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t {
        return nullptr;
    };
    check(!ggml_backend_sched_moe_resident_prepare(&failed_pool, backend, &failed_buft,
            1, 1, 1, expert_size + 512, &failed_budget), "injected device allocation failure bypasses cache");
    check(failed_budget.unavailable && failed_pool.disabled && failed_pool.buffer == nullptr &&
            failed_budget.allocated_bytes == 0, "failed allocation consumes no resident budget");
    fixture.sched.moe_resident_slots = -1;
    fixture.sched.moe_resident_budget[0] = failed_budget;
    check(!ggml_backend_sched_moe_resident_requested(&fixture.sched, backend),
            "unavailable auto does not request residency-only placement for a new graph");

    ggml_backend_sched_moe_resident_clear_storage(&pool);
    free(fixture.sched.moe_resident[0].pools);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}
#endif

#ifdef GGML_USE_CUDA
static void test_complete_ffn(bool combined, bool fail_cache, bool mixed_quant = false) {
    constexpr int width = 256, hidden = 512, experts = 16, top_k = 8, layers = 3;
    auto * cuda = ggml_backend_cuda_init(0, nullptr, nullptr);
    check(cuda != nullptr, "complete FFN CUDA backend initializes");
    if (!cuda) return;
    auto * cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_t backends[] = {cuda, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 512, false);
    ggml_backend_sched_set_moe_resident_model_info(sched, layers, experts, top_k);
    ggml_backend_sched_set_moe_resident(sched, -1);
    auto * sources = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor * banks[layers][3]{}, * gpu_banks[layers][3]{};
    std::vector<ggml_backend_buffer_t> buffers;
    auto allocate = [&](ggml_tensor * tensor, ggml_backend_buffer_type_t buft) {
        auto buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
        GGML_ASSERT(buffer);
        ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
        buffers.push_back(buffer);
    };
    auto make_bank = [&](int layer, int b, int bank_hidden, int generation) {
        const int columns = b == 2 ? bank_hidden : width;
        const int rows = b == 2 ? width : combined ? bank_hidden * 2 : bank_hidden;
        // Exercise heterogeneous DOWN types with the shared padded strides,
        // matching the Q4_K/Q5_K/Q6_K family used by the model replay.
        const ggml_type type = mixed_quant
                ? (b == 2 ? (layer == 1 ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K) : GGML_TYPE_Q4_K)
                : GGML_TYPE_Q4_0;
        banks[layer][b] = ggml_new_tensor_3d(sources, type, columns, rows, experts);
        const char * role = b == 2 ? "down" : b == 1 ? "gate" : combined ? "gate_up" : "up";
        ggml_format_name(banks[layer][b], "blk.%d.ffn_%s_exps.weight", layer, role);
        allocate(banks[layer][b], ggml_backend_cuda_host_buffer_type());
        ggml_backend_buffer_set_usage(banks[layer][b]->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(ggml_nelements(banks[layer][b]));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.015f * (int((i * 13 + size_t(b * 7 + layer * 3 + generation * 11)) % 31) - 15);
        std::vector<uint8_t> quantized(ggml_nbytes(banks[layer][b]));
        ggml_quantize_chunk(type, values.data(), quantized.data(), 0,
                ggml_nrows(banks[layer][b]), columns, nullptr, nullptr);
        ggml_backend_tensor_set(banks[layer][b], quantized.data(), 0, quantized.size());
        gpu_banks[layer][b] = ggml_new_tensor_3d(sources, type, columns, rows, experts);
        allocate(gpu_banks[layer][b], ggml_backend_cuda_buffer_type(0));
        ggml_backend_tensor_set(gpu_banks[layer][b], quantized.data(), 0, quantized.size());
    };
    for (int layer = 0; layer < layers; ++layer) for (int b = 0; b < 3; ++b)
        if (!combined || b != 1) make_bank(layer, b, hidden, 0);
    auto * act = ggml_new_tensor_3d(sources, GGML_TYPE_F32, width, 1, 1);
    auto * ids = ggml_new_tensor_2d(sources, GGML_TYPE_I32, top_k, 1);
    auto * weights = ggml_new_tensor_3d(sources, GGML_TYPE_F32, 1, top_k, 1);
    for (auto * tensor : {act, ids, weights}) allocate(tensor, ggml_backend_cuda_host_buffer_type());
    ggml_set_input(ids); ggml_set_input(act); ggml_set_input(weights);
    std::vector<float> activation(width), scaling(top_k);
    for (int i = 0; i < width; ++i) activation[i] = 0.07f * (i % 17 - 8);
    for (int i = 0; i < top_k; ++i) scaling[i] = float(i + 1) / 36;
    ggml_backend_tensor_set(act, activation.data(), 0, ggml_nbytes(act));
    ggml_backend_tensor_set(weights, scaling.data(), 0, ggml_nbytes(weights));
    int32_t routes[][top_k] = {
        {0, 1, 2, 3, 0, 1, 4, 4}, {5, 6, 7, 8, 9, 10, 11, 12},
        {0, 1, 2, 3, 0, 1, 4, 4}, {-1, 3, 3, 3, 8, 9, 9, 9},
        {14, 14, 14, 14, 14, 14, 14, 14}, {15, 14, 13, 12, 11, 10, 9, 8},
    };
    auto native = [&](ggml_backend_t backend, int layer, const int32_t * route) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 1);
        auto * r = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
        auto * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, top_k, 1);
        auto ** wb = backend == cuda ? gpu_banks[layer] : banks[layer];
        auto * up = ggml_moe_up_gate(ctx, wb[0], wb[1], a, r, GGML_UNARY_OP_SILU);
        auto * down = ggml_mul_mat_id(ctx, wb[2], up, r);
        auto * out = ggml_sqr(ctx, ggml_mul(ctx, down, w));
        auto * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, out);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        ggml_backend_tensor_set(a, activation.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(r, route, 0, ggml_nbytes(r));
        ggml_backend_tensor_set(w, scaling.data(), 0, ggml_nbytes(w));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "complete FFN native reference computes");
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        return result;
    };
    auto original_buft = sched->bufts[0];
    auto failed_buft = *original_buft;
    failed_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
    bool injected = false;
    for (int r = 0; r < 6; ++r) {
        if (!fail_cache && r == 3) {
            // Replace the DOWN bank while this layer has a live cached expert.
            // Repeating that expert makes stale slot reuse observable.
            int cached_expert = -1;
            for (const auto & slot : sched->moe_ffn->slots)
                if (slot.layer == layers - 1) { cached_expert = slot.expert; break; }
            check(cached_expert >= 0, "bank identity regression starts with a live resident expert");
            GGML_ASSERT(cached_expert >= 0);
            routes[r][1] = routes[r][2] = routes[r][3] = cached_expert;
            make_bank(layers - 1, 2, hidden, 1);
        }
        const size_t old_cache_bytes = sched->moe_ffn ? sched->moe_ffn->allocation_bytes : 0;
        if (!fail_cache && r == 5) {
            // Grow an intermediate dimension on the same scheduler. All shared
            // slot strides and persistent CPU/device scratch must follow it.
            for (int b = 0; b < 3; ++b)
                if (!combined || b != 1) make_bank(1, b, hidden * 2, 2);
        }
        std::vector<std::vector<float>> expected_cpu, expected_gpu;
        for (int layer = 0; layer < layers; ++layer) {
            expected_cpu.push_back(native(cpu, layer, routes[r]));
            expected_gpu.push_back(native(cuda, layer, routes[r]));
        }
        auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        auto * graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_tensor * outputs[layers]{};
        for (int layer = 0; layer < layers; ++layer) {
            auto * up = ggml_moe_up_gate(ctx, banks[layer][0], banks[layer][1], act, ids, GGML_UNARY_OP_SILU);
            auto * down = ggml_mul_mat_id(ctx, banks[layer][2], up, ids);
            outputs[layer] = ggml_sqr(ctx, ggml_mul(ctx, down, weights));
            ggml_set_output(outputs[layer]);
            ggml_build_forward_expand(graph, outputs[layer]);
        }
        ggml_backend_tensor_set(ids, routes[r], 0, ggml_nbytes(ids));
        check(ggml_backend_sched_alloc_graph(sched, graph), "complete FFN graph allocates without bank staging");
        if (!fail_cache && r == 5) {
            check(!sched->moe_ffn->cache && sched->moe_ffn->slots.empty(),
                    "larger geometry releases the old global cache before execution");
        }
        if (!injected) {
            check(sched->moe_ffn && sched->moe_ffn->graph_tasks.size() == 2 * layers,
                    "all layers enter complete FFN decode independently of prefix plan");
            auto & f = *sched->moe_ffn;
            if (fail_cache) sched->bufts[0] = &failed_buft;
            else {
                // A deliberately tiny global cache forces cross-layer eviction.
                const size_t capacity = top_k;
                size_t bytes = 0;
                for (int b = 0; b < 3; ++b) { f.offsets[b] = bytes; bytes += capacity * f.strides[b]; }
                f.cache = ggml_backend_buft_alloc_buffer(original_buft, bytes);
                GGML_ASSERT(f.cache);
                ggml_backend_buffer_clear(f.cache, 0);
                f.slots.resize(capacity); f.allocation_bytes = bytes;
                for (auto & task : f.tasks) if (task) {
                    auto & timing = ggml_backend_sched_moe_ffn_timing(f, *task, top_k);
                    for (size_t m = 1; m < timing.misses.size(); ++m) for (size_t q = 1; q <= m; ++q) {
                        timing.misses[m][q].count = 3;
                        timing.misses[m][q].us = q == std::min<size_t>(2, m) ? 1.0 : 10.0;
                    }
                }
            }
            injected = true;
        }
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                "complete FFN graph computes with masks and downstream weighting");
        sched->bufts[0] = original_buft;
        if (!fail_cache && r == 5) check(sched->moe_ffn->allocation_bytes > old_cache_bytes,
                "larger geometry allocates complete slots with the new strides");
        if (r == 0 && !fail_cache) {
            for (int layer = 0; layer < layers; ++layer) {
                expected_cpu[layer] = native(cpu, layer, routes[1]);
                expected_gpu[layer] = native(cuda, layer, routes[1]);
            }
            ggml_backend_tensor_set(ids, routes[1], 0, ggml_nbytes(ids));
            check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
                    "complete FFN reuses an allocated graph with changed logical and physical IDs");
        }
        for (int layer = 0; layer < layers; ++layer) {
            std::vector<float> actual(ggml_nelements(outputs[layer]));
            ggml_backend_tensor_get(outputs[layer], actual.data(), 0, ggml_nbytes(outputs[layer]));
            auto & f = *sched->moe_ffn;
            // Each task retains its physical route, even after a later layer
            // evicts the slots. Compare rows with their actual native branch.
            const auto & mask = f.tasks[layer]->route_gpu;
            size_t bad = 0;
            for (size_t j = 0; j < actual.size(); ++j) {
                const size_t row = j / width;
                const bool gpu_row = !fail_cache && mask.size() == top_k && mask[row] >= 0;
                const auto & expected = gpu_row ? expected_gpu[layer] : expected_cpu[layer];
                if (!(std::fabs(actual[j] - expected[j]) <= 1e-5f + 1e-5f * std::fabs(expected[j]))) ++bad;
            }
            if (bad) fprintf(stderr, "complete FFN combined=%d fail=%d mixed_quant=%d route=%d layer=%d bad=%zu\n", combined, fail_cache, mixed_quant, r, layer, bad);
            check(bad == 0, "complete FFN rows match their native CPU or CUDA branch");
        }
        if (fail_cache && r == 0) check(ggml_backend_sched_moe_resident_needs_rebuild(sched),
                "complete cache allocation failure requests a fresh graph");
        ggml_backend_sched_reset(sched); ggml_free(ctx);
    }
    if (!fail_cache) {
        check(sched->moe_ffn->evictions > 0, "global cache evicts across layers");
        check(sched->moe_ffn->cpu_refs > 0 && sched->moe_ffn->gpu_refs > 0, "complete FFN executes both branches");
        check(sched->moe_ffn->fills > 0 && sched->moe_ffn->hits > 0, "complete slots admit and reuse experts");
    }
    ggml_backend_sched_free(sched);
    for (auto buffer : buffers) ggml_backend_buffer_free(buffer);
    ggml_free(sources); ggml_backend_free(cpu); ggml_backend_free(cuda);
}
#endif

#ifdef GGML_USE_CUDA
static void test_cuda_budget_cap() {
    auto * cuda = ggml_backend_cuda_init(0, nullptr, nullptr);
    check(cuda != nullptr, "budget cap CUDA backend initializes");
    if (!cuda) return;
    auto * cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {cuda, cpu};
    const size_t mib = 1024 * 1024;
    for (size_t cap : {4 * mib, mib}) {
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false);
        ggml_backend_sched_set_moe_resident_model_info(sched, 4, 8, 1);
        ggml_backend_sched_set_moe_resident_budget(sched, cap);
        ggml_backend_sched_set_moe_resident(sched, -1);
        check(sched->moe_resident_budget[0].budget_bytes <= cap,
                "legacy auto budget respects the explicit cap");
        sched->moe_ffn = new ggml_backend_sched_moe_ffn;
        // Three one-MiB projections form one complete expert. A four-MiB
        // cap must allocate one slot, while one MiB must fall back to CPU.
        for (auto & stride : sched->moe_ffn->strides) stride = mib;
        const bool allocated = ggml_backend_sched_moe_ffn_cache(sched, cuda);
        check(allocated == (cap == 4 * mib), "cap controls real complete-cache allocation");
        check(sched->moe_ffn->allocation_bytes <= cap,
                "device expert-cache allocation stays within the cap");
        if (allocated) check(sched->moe_ffn->slots.size() == 1,
                "capped allocation rounds down to complete expert slots");
        else check(sched->moe_ffn->disabled && ggml_backend_sched_moe_resident_needs_rebuild(sched),
                "insufficient cap safely requests CPU fallback and a fresh graph");
        ggml_backend_sched_free(sched);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(cuda);
}
#endif

int main(int argc, char ** argv) {
    test_history();
    test_history_boundaries();
    test_identity();
    test_hybrid_placement();
    test_cpu_repeated_routes();
    test_cpu_quantized_masked_rows();
    if (argc > 1 && strcmp(argv[1], "--cuda") == 0) {
#ifdef GGML_USE_CUDA
        test_cuda_budget_cap();
        test_cuda_copy();
        test_complete_ffn(false, false);
        test_complete_ffn(true, false);
        test_complete_ffn(false, true);
        test_complete_ffn(false, false, true);
#else
        check(false, "CUDA test requires a CUDA build");
#endif
    }
    fprintf(stderr, "resident policy: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
