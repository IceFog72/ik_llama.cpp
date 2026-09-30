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
    // B3R-G5 half: a runtime demotion on an allocated graph must request a
    // re-split so CPU fallback takes effect; an idle scheduler must not.
    fixture.sched.is_alloc = true;
    fixture.sched.moe_resident_hybrid_plan_ready = true;
    check(!fixture.sched.moe_resident_replan_required,
            "allocated graph starts with no pending replan");
    check(ggml_backend_sched_moe_resident_hybrid_set_layer(&fixture.sched, 2, false),
            "runtime failure demotes a preplanned GPU layer");
    check(fixture.sched.moe_resident_replan_required,
            "demotion on an allocated graph requests a safe-point re-split");
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

#ifdef GGML_USE_CUDA
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

int main(int argc, char ** argv) {
    test_history();
    test_history_boundaries();
    test_identity();
    test_hybrid_placement();
    if (argc > 1 && strcmp(argv[1], "--cuda") == 0) {
#ifdef GGML_USE_CUDA
        test_cuda_copy();
#else
        check(false, "CUDA test requires a CUDA build");
#endif
    }
    fprintf(stderr, "resident policy: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
