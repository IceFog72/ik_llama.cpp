// ft-bench-bw: Slice A profiler. Measures on THIS box:
//   1. CPU DRAM read ceiling (STREAM-style, multithreaded)
//   2. PCIe H2D/D2H linear copy bandwidth (pinned <-> device, CUDA events)
//   3. CPU MoE GEMV: Qwen3.6 geometry (H=2048, I=512, E=256, top_k=8),
//      Q4_K expert bank x F32 activations, MUL_MAT_ID + down proj, decode shape
//   4. PCIe expert gather: H2D copy of top_k experts worth of Q4_K bytes
// Prints GB/s + the FreeToken-style recommendation (hybrid iff cpu > 2x pcie)
// and the overlapped fetch fraction q = pcie/cpu.
//
// Build (from ik_llama_ft root, needs built ggml):
//   g++ -O3 -std=c++17 -Iggml/include -Iggml/src -Icommon \
//     examples/ft-bench-bw/ft-bench-bw.cpp -o /tmp/ft-bench-bw \
//     -L/mnt/Kingstone_SSD/build-ik-llama/ggml/src -lggml \
//     -Wl,-rpath,/mnt/Kingstone_SSD/build-ik-llama/ggml/src -lpthread
// Simpler: drop into examples/bench-bw/ with a CMakeLists like llama-bench's.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

static double now_s() {
    return ggml_time_us() / 1e6;
}

// ---- 1. CPU DRAM read ceiling ----
static double bench_cpu_read(int n_threads, size_t bytes, int iters) {
    std::vector<float> buf(bytes / sizeof(float), 1.0f);
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
        std::vector<double> partial(n_threads, 0.0);
        auto t0 = now_s();
        std::vector<std::thread> th;
        size_t chunk = buf.size() / n_threads;
        for (int t = 0; t < n_threads; ++t) {
            th.emplace_back([&, t]() {
                size_t lo = t * chunk, hi = (t == n_threads - 1) ? buf.size() : lo + chunk;
                double s = 0;
                for (size_t i = lo; i < hi; ++i) s += buf[i];
                partial[t] = s;
            });
        }
        for (auto & t : th) t.join();
        double dt = now_s() - t0;
        volatile double sink = 0;
        for (auto s : partial) sink += s;
        (void) sink;
        best = std::min(best, dt);
    }
    return bytes / best / 1e9;
}

int main(int argc, char ** argv) {
    int n_threads = 16;
    if (argc > 1) n_threads = atoi(argv[1]);
    printf("ft-bench-bw (slice A profiler) threads=%d\n", n_threads);
    setvbuf(stdout, NULL, _IONBF, 0);

    // 1. CPU ceiling
    double cpu_gbs = bench_cpu_read(n_threads, 256u << 20, 5);
    printf("CPU DRAM read ceiling : %6.2f GB/s\n", cpu_gbs);

#ifdef GGML_USE_CUDA
    // 2. PCIe linear copy
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!cuda) { printf("no CUDA backend\n"); return 1; }
    {
        const size_t nbytes = 256u << 20;
        auto buft_dev  = ggml_backend_cuda_buffer_type(0);
        auto buft_host = ggml_backend_cuda_host_buffer_type();
        auto * buf_dev  = ggml_backend_buft_alloc_buffer(buft_dev, nbytes);
        auto * buf_host = ggml_backend_buft_alloc_buffer(buft_host, nbytes);
        void * dev_ptr  = ggml_backend_buffer_get_base(buf_dev);
        void * host_ptr = ggml_backend_buffer_get_base(buf_host);
        memset(host_ptr, 0xAB, nbytes);
        const int iters = 30;
        // Timed H2D/D2H via tensor copy between host and device buffers
        auto t0 = now_s();
        struct ggml_init_params p = { /*mem_size*/ nbytes * 2 + 4096, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * tdev  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, nbytes);
        struct ggml_tensor * thost = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, nbytes);
        ggml_backend_buffer_t bdev  = ggml_backend_alloc_buffer(cuda, ggml_backend_buft_get_alloc_size(buft_dev, tdev));
        ggml_backend_t cpu = ggml_backend_cpu_init();
        // host tensor must live in pinned (cuda host) buffer for async copy path
        size_t thost_sz = ggml_backend_buft_get_alloc_size(buft_host, thost);
        ggml_backend_buffer_t bhost = ggml_backend_buft_alloc_buffer(buft_host, thost_sz);
        ggml_backend_tensor_alloc(bdev, tdev, ggml_backend_buffer_get_base(bdev));
        ggml_backend_tensor_alloc(bhost, thost, ggml_backend_buffer_get_base(bhost));
        memset(ggml_backend_buffer_get_base(bhost), 0xAB, nbytes);
        (void) buf_dev; (void) buf_host; (void) dev_ptr; (void) host_ptr;
        // H2D/D2H via backend tensor_set/get (host-pinned <-> device).
        // Warmup then time N iters. tensor_set/get route through the right
        // copy path (no dst->buffer deref like tensor_copy's fast path).
        for (int i = 0; i < 3; ++i) ggml_backend_tensor_set(tdev, ggml_backend_buffer_get_base(bhost), 0, nbytes);
        ggml_backend_synchronize(cuda);
        t0 = now_s();
        for (int i = 0; i < iters; ++i) ggml_backend_tensor_set(tdev, ggml_backend_buffer_get_base(bhost), 0, nbytes);
        ggml_backend_synchronize(cuda);
        double h2d = (nbytes * (double) iters) / (now_s() - t0) / 1e9;
        // D2H: reuse same host staging (overwrite ok)
        for (int i = 0; i < 3; ++i) ggml_backend_tensor_get(tdev, ggml_backend_buffer_get_base(bhost), 0, nbytes);
        ggml_backend_synchronize(cuda);
        t0 = now_s();
        for (int i = 0; i < iters; ++i) ggml_backend_tensor_get(tdev, ggml_backend_buffer_get_base(bhost), 0, nbytes);
        ggml_backend_synchronize(cuda);
        double d2h = (nbytes * (double) iters) / (now_s() - t0) / 1e9;
        printf("PCIe linear H2D       : %6.2f GB/s\n", h2d);
        printf("PCIe linear D2H       : %6.2f GB/s\n", d2h);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(cuda);
    }
#else
    printf("CUDA disabled in this build; PCIe probe skipped\n");
#endif

    // 3. CPU MoE GEMV: Qwen3.6 geometry (H=2048, I=512, E=256, top_k=8),
    //    Q4_K expert bank, F32 activations, decode shape (tokens=1).
    //    One layer pass = up+gate MUL_MAT_ID, SILU mul, down MUL_MAT_ID.
    // 4. PCIe expert gather: H2D copy of top_k experts of Q4_K bytes.
    double cpu_moe_gbs = 0, gather_gbs = 0;
    {
        const int64_t H = 2048, I = 512, E = 256, K = 8, T = 1;
        const ggml_type QT = GGML_TYPE_Q4_K;
        size_t bank_elems_up = (size_t) 2 * I * H * E; // fused up+gate [2I, H, E]
        size_t bank_elems_dn = (size_t) H * I * E;     // down [H, I, E]
        size_t bank_bytes_up = ggml_row_size(QT, bank_elems_up / E) * E;
        size_t bank_bytes_dn = ggml_row_size(QT, bank_elems_dn / E) * E;
        size_t expert_bytes = bank_bytes_up / E + bank_bytes_dn / E;
        printf("Qwen3.6 expert Q4_K   : %.2f MB, top_%lld active = %.1f MB/token/layer\n",
            expert_bytes / 1e6, (long long) K, K * expert_bytes / 1e6);

        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        struct ggml_init_params mp = { /*mem_size*/ bank_bytes_up + bank_bytes_dn + 64 * 1024 * 1024,
                                        /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        struct ggml_context * mctx = ggml_init(mp);
        struct ggml_tensor * bank_up = ggml_new_tensor_3d(mctx, QT, H, 2 * I, E);
        struct ggml_tensor * bank_dn = ggml_new_tensor_3d(mctx, QT, I, H, E);
        struct ggml_tensor * act     = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, H, T);
        struct ggml_tensor * ids     = ggml_new_tensor_2d(mctx, GGML_TYPE_I32, K, T);
        ggml_backend_buffer_t bbank = ggml_backend_alloc_buffer(cpu, bank_bytes_up + bank_bytes_dn + 16 * 1024 * 1024);
        ggml_backend_tensor_alloc(bbank, bank_up, ggml_backend_buffer_get_base(bbank));
        ggml_backend_tensor_alloc(bbank, bank_dn, (char *) ggml_backend_buffer_get_base(bbank) + bank_bytes_up);
        // activation + ids in small heap tensor buf
        struct ggml_init_params ap = { /*mem_size*/ 16 * 1024 * 1024, nullptr, false };
        struct ggml_context * actx = ggml_init(ap);
        struct ggml_tensor * hact = ggml_new_tensor_2d(actx, GGML_TYPE_F32, H, T);
        struct ggml_tensor * hids = ggml_new_tensor_2d(actx, GGML_TYPE_I32, K, T);
        ggml_set_f32(hact, 0.01f);
        for (int k = 0; k < K; ++k) ((int32_t *) hids->data)[k] = (k * 37) % E;
        // quantize random experts into bank (content irrelevant for timing)
        {
            std::vector<float> tmp(2 * I * H);
            for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = 0.01f * (float) (i % 13);
            for (int e = 0; e < E; ++e) {
                size_t row = ggml_row_size(QT, 2 * I * H);
                ggml_quantize_chunk(QT, tmp.data(), (char *) bank_up->data + e * row, 0, (2 * I * H) / bank_up->ne[0], bank_up->ne[0], nullptr, nullptr);
            }
            tmp.assign((size_t) H * I, 0.01f);
            for (int e = 0; e < E; ++e) {
                size_t row = ggml_row_size(QT, H * I);
                ggml_quantize_chunk(QT, tmp.data(), (char *) bank_dn->data + e * row, 0, (H * I) / bank_dn->ne[0], bank_dn->ne[0], nullptr, nullptr);
            }
        }
        // graph: up = MUL_MAT_ID(bank_up, act, ids); gate part = same bank rows [I:2I]? use full then slice via view
        // simpler faithful-enough: two MUL_MAT_ID (up rows via strided view is complex) -> use unfused path like stock fallback:
        // up_full [2I,K,T] = MUL_MAT_ID(bank_up, act, ids); par = SILU(up_full[I:,...]) * up_full[:I,...]; out = MUL_MAT_ID(bank_dn, par, ids)
        struct ggml_init_params gp = { /*mem_size*/ 64 * 1024 * 1024, nullptr, false };
        struct ggml_context * gctx = ggml_init(gp);
        struct ggml_tensor * gact = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, H, K, T);
        struct ggml_tensor * gids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, K, T);
        for (int k = 0; k < K; ++k) memcpy((char *) gact->data + k * H * sizeof(float), hact->data, H * sizeof(float));
        memcpy(gids->data, hids->data, ggml_nbytes(gids));
        struct ggml_tensor * upf = ggml_mul_mat_id(gctx, bank_up, gact, gids); // [2I,K,T]
        struct ggml_tensor * up1 = ggml_view_3d(gctx, upf, I, K, T, upf->nb[1], upf->nb[2], 0);
        struct ggml_tensor * gt1 = ggml_view_3d(gctx, upf, I, K, T, upf->nb[1], upf->nb[2], I * upf->nb[0]);
        struct ggml_tensor * sact = ggml_silu(gctx, gt1);
        struct ggml_tensor * par = ggml_mul(gctx, sact, up1);
        struct ggml_tensor * parc = ggml_cont(gctx, par);
        struct ggml_tensor * out = ggml_mul_mat_id(gctx, bank_dn, parc, gids); // [H,K,T]
        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, out);
        // bind bank tensors (already have data) + graph temps via backend alloc
        ggml_backend_buffer_t gbuf = ggml_backend_alloc_buffer(cpu, 64 * 1024 * 1024);
        // manual: assign unbound tensors in gctx from gbuf bump pointer
        {
            char * base = (char *) ggml_backend_buffer_get_base(gbuf);
            size_t off = 0;
            for (struct ggml_tensor * t = ggml_get_first_tensor(gctx); t; t = ggml_get_next_tensor(gctx, t)) {
                if (t->data) continue;
                size_t sz = ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), t);
                off = (off + 31) & ~(size_t) 31;
                ggml_backend_tensor_alloc(gbuf, t, base + off);
                off += sz;
                if (off > 64 * 1024 * 1024) { printf("graph buf overflow\n"); return 1; }
            }
        }
        // bank_up/bank_dn/act/ids live outside gctx: rewire src pointers
        upf->src[0] = bank_up; upf->src[1] = gact; upf->src[2] = gids;
        out->src[0] = bank_dn; out->src[2] = gids;
        const int iters = 20;
        for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(cpu, gf); // warmup
        auto t0 = now_s();
        for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(cpu, gf);
        double dt = now_s() - t0;
        double bytes_per_iter = (double) (K * expert_bytes); // unique expert bytes touched
        cpu_moe_gbs = bytes_per_iter * iters / dt / 1e9;
        printf("CPU MoE GEMV (1 layer, top_%lld, %d iters): %6.2f GB/s (%.1f ms/iter)\n",
            (long long) K, iters, cpu_moe_gbs, dt * 1000 / iters);
        ggml_backend_buffer_free(gbuf);
        ggml_backend_buffer_free(bbank);
        ggml_free(gctx); ggml_free(mctx); ggml_free(actx);
        ggml_backend_free(cpu);

#ifdef GGML_USE_CUDA
        // 4. PCIe expert gather: H2D copy of top_k experts bytes, pinned staging
        {
            ggml_backend_t cu = ggml_backend_cuda_init(0, nullptr, nullptr);
            auto buft_dev  = ggml_backend_cuda_buffer_type(0);
            auto buft_host = ggml_backend_cuda_host_buffer_type();
            size_t gbytes = K * expert_bytes;
            ggml_backend_buffer_t hd = ggml_backend_buft_alloc_buffer(buft_dev, gbytes);
            ggml_backend_buffer_t hh = ggml_backend_buft_alloc_buffer(buft_host, gbytes);
            memset(ggml_backend_buffer_get_base(hh), 0xAB, gbytes);
            struct ggml_init_params zp = { 64 * 1024 * 1024, nullptr, true };
            struct ggml_context * zctx = ggml_init(zp);
            struct ggml_tensor * zd = ggml_new_tensor_1d(zctx, GGML_TYPE_I8, gbytes);
            struct ggml_tensor * zh = ggml_new_tensor_1d(zctx, GGML_TYPE_I8, gbytes);
            ggml_backend_tensor_alloc(hd, zd, ggml_backend_buffer_get_base(hd));
            ggml_backend_tensor_alloc(hh, zh, ggml_backend_buffer_get_base(hh));
            const int giters = 50;
            for (int i = 0; i < 5; ++i) ggml_backend_tensor_set(zd, ggml_backend_buffer_get_base(hh), 0, gbytes);
            ggml_backend_synchronize(cu);
            auto t1 = now_s();
            for (int i = 0; i < giters; ++i) ggml_backend_tensor_set(zd, ggml_backend_buffer_get_base(hh), 0, gbytes);
            ggml_backend_synchronize(cu);
            gather_gbs = (gbytes * (double) giters) / (now_s() - t1) / 1e9;
            printf("PCIe expert gather (top_%lld = %.1f MB): %6.2f GB/s\n", (long long) K, gbytes / 1e6, gather_gbs);
            ggml_backend_buffer_free(hd); ggml_backend_buffer_free(hh);
            ggml_free(zctx); ggml_backend_free(cu);
        }
#endif
    }
    printf("FreeToken rule: hybrid iff cpu_moe_bw > 2x pcie_gather_bw; fetch frac q = pcie/cpu\n");
    if (cpu_moe_gbs > 0 && gather_gbs > 0) {
        printf("RESULT cpu_moe=%.2f pcie_gather=%.2f ratio=%.2fx backend=%s q=%.3f\n",
            cpu_moe_gbs, gather_gbs, cpu_moe_gbs / gather_gbs,
            cpu_moe_gbs > 2.0 * gather_gbs ? "hybrid" : "offload",
            gather_gbs / cpu_moe_gbs);
    }
    return 0;
}
