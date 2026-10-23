// Compare the opt-in tiled KV loader against the existing conversion + MMA
// path on identical quantized inputs. Run manually on a Turing-or-newer GPU.
// The tiled path splits the KV range across more blocks, so its f16 partial
// sums combine in a different order. Where it is selected, both paths are
// measured against a double-precision reference and the tiled error must not
// exceed the existing path's error. Fallback shapes must match the old path.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static std::atomic<int> selected{0};

static void log_callback(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "fa-quant-tile active")) {
        ++selected;
        fputs(text, stderr);
    } else if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_t baseline = nullptr, tiled = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~fixture() {
        // Retire captures before releasing their tensor storage.
        if (baseline) ggml_backend_free(baseline);
        if (tiled) ggml_backend_free(tiled);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

static bool compare(int valid_tokens, ggml_type v_type, int queries = 1, int gqa = 8,
                    bool sinks = false, float bias = 0.0f, float softcap = 0.0f,
                    ggml_type k_type = GGML_TYPE_Q4_0, bool interleaved = true) {
    fixture f;
    f.baseline = ggml_backend_cuda_init(0, "fa-quant-tile=0", nullptr);
    f.tiled = ggml_backend_cuda_init(0, "fa-quant-tile=1", nullptr);
    if (!f.baseline || !f.tiled) return false;
    f.ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!f.ctx) return false;

    constexpr int d = 256, kv_heads = 4;
    const int kv = ((valid_tokens + 255) / 256) * 256;
    const int mask_rows = ((queries + GGML_KQ_MASK_PAD - 1) / GGML_KQ_MASK_PAD) * GGML_KQ_MASK_PAD;
    auto * q = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F32, d, queries, kv_heads*gqa, 1);
    // llama's cache interleaves heads within each token. Also cover ordinary
    // head-major storage: unequal K/V strides must work in either layout.
    auto * k_storage = ggml_new_tensor_4d(f.ctx, k_type, d, interleaved ? kv_heads : kv, interleaved ? kv : kv_heads, 1);
    auto * v_storage = ggml_new_tensor_4d(f.ctx, v_type, d, interleaved ? kv_heads : kv, interleaved ? kv : kv_heads, 1);
    auto * k = interleaved ? ggml_permute(f.ctx, k_storage, 0, 2, 1, 3) : k_storage;
    auto * v = interleaved ? ggml_permute(f.ctx, v_storage, 0, 2, 1, 3) : v_storage;
    auto * m = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F16, kv, mask_rows, 1, 1);
    auto * out = ggml_flash_attn_ext(f.ctx, q, k, v, m, 1.0f/std::sqrt(float(d)), bias, softcap);
    auto * sink = sinks ? ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, kv_heads*gqa) : nullptr;
    if (sink) ggml_flash_attn_ext_add_sinks(out, sink);
    auto * graph = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(graph, out);
    if (!ggml_backend_supports_op(f.baseline, out) || !ggml_backend_supports_op(f.tiled, out)) {
        fprintf(stderr, "FAIL: attention shape unsupported on this device\n");
        return false;
    }
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, f.baseline);
    if (!f.buffer) return false;

    std::mt19937 rng(1729);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    std::vector<float> k_values, v_values; // dequantized storage for the reference
    for (auto * tensor : {k_storage, v_storage}) {
        std::vector<float> data(ggml_nelements(tensor));
        for (auto & value : data) value = random(rng);
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        const auto written = ggml_quantize_chunk(tensor->type, data.data(), bytes.data(), 0,
                ggml_nrows(tensor), d, nullptr, nullptr);
        if (written != bytes.size()) return false;
        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
        auto & values = tensor == k_storage ? k_values : v_values;
        values.resize(data.size());
        ggml_internal_get_type_traits(tensor->type).to_float(bytes.data(), values.data(), int64_t(values.size()));
    }
    if (sink) {
        std::vector<float> values(kv_heads*gqa);
        for (auto & value : values) value = 2.0f*random(rng);
        ggml_backend_tensor_set(sink, values.data(), 0, ggml_nbytes(sink));
    }
    std::vector<float> input(ggml_nelements(q));
    std::vector<ggml_fp16_t> mask(ggml_nelements(m));
    std::vector<float> reference(ggml_nelements(out)), actual(reference.size());
    std::vector<float> sink_values;
    if (sink) {
        sink_values.resize(ggml_nelements(sink));
        ggml_backend_tensor_get(sink, sink_values.data(), 0, ggml_nbytes(sink));
    }
    // out[h + heads*j][d]: softmax(scale*q.k + mask) over visible keys, plus an optional sink logit.
    auto exact = [&](const std::vector<float> & qv, const std::vector<ggml_fp16_t> & mv) {
        const int heads = kv_heads*gqa;
        const float scale = 1.0f/std::sqrt(float(d));
        std::vector<double> result(size_t(heads)*queries*d), score(kv);
        auto at = [&](const std::vector<float> & x, int token, int head) {
            return x.data() + (interleaved ? (size_t(token)*kv_heads + head) : (size_t(head)*kv + token))*d;
        };
        for (int j = 0; j < queries; ++j) {
            for (int h = 0; h < heads; ++h) {
                const float * qh = qv.data() + (size_t(h)*queries + j)*d;
                double top = sink ? sink_values[h] : -INFINITY;
                for (int i = 0; i < kv; ++i) {
                    const float mask_value = ggml_fp16_to_fp32(mv[size_t(j)*kv + i]);
                    score[i] = -INFINITY;
                    if (mask_value == -INFINITY) continue;
                    const float * ki = at(k_values, i, h / gqa);
                    double dot = 0;
                    for (int e = 0; e < d; ++e) dot += double(qh[e])*ki[e];
                    score[i] = scale*dot + mask_value;
                    top = std::max(top, score[i]);
                }
                double sum = sink ? std::exp(sink_values[h] - top) : 0.0;
                double * o = result.data() + (size_t(j)*heads + h)*d;
                for (int i = 0; i < kv; ++i) {
                    if (score[i] == -INFINITY) continue;
                    const double w = std::exp(score[i] - top);
                    sum += w;
                    const float * vi = at(v_values, i, h / gqa);
                    for (int e = 0; e < d; ++e) o[e] += w*vi[e];
                }
                for (int e = 0; e < d; ++e) o[e] /= sum;
            }
        }
        return result;
    };
    const bool expect_tiled = queries <= 2 && gqa % 8 == 0 && bias == 0.0f && softcap == 0.0f &&
            k_type == GGML_TYPE_Q4_0 && (v_type == GGML_TYPE_Q4_0 || v_type == GGML_TYPE_Q8_0);
    selected = 0;
    double worst = 0;
    // Repeated graphs with changed inputs exercise both capture and replay.
    // The last pass heavily masks the prefix to expose accidental reads of
    // masked rows, while retaining at least one visible token per split tile.
    for (int pass = 0; pass < 4; ++pass) {
        for (auto & value : input) value = (pass == 2 ? 4.0f : 0.5f)*random(rng);
        for (int row = 0; row < mask_rows; ++row) {
            for (int i = 0; i < kv; ++i) {
                const bool hidden = i >= valid_tokens ||
                        (row < queries && i > valid_tokens - queries + row) ||
                        (pass == 3 && i % 32 != 0);
                mask[size_t(row)*kv + i] = ggml_fp32_to_fp16(hidden ? -INFINITY : 0.0f);
            }
        }
        ggml_backend_tensor_set(q, input.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));
        for (int method = 0; method < 2; ++method) {
            // A failed/missing output write must not inherit the reference.
            std::vector<float> poison(actual.size(), std::numeric_limits<float>::quiet_NaN());
            ggml_backend_tensor_set(out, poison.data(), 0, ggml_nbytes(out));
            auto * backend = method == 0 ? f.baseline : f.tiled;
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(out, method == 0 ? reference.data() : actual.data(), 0, ggml_nbytes(out));
        }
        if (expect_tiled) {
            const auto truth = exact(input, mask);
            double base_error = 0, tiled_error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
                    fprintf(stderr, "FAIL: ctx=%d non-finite output pass=%d i=%zu\n", valid_tokens, pass, i);
                    return false;
                }
                base_error  = std::max(base_error,  std::abs(reference[i] - truth[i]));
                tiled_error = std::max(tiled_error, std::abs(actual[i]    - truth[i]));
            }
            worst = std::max(worst, tiled_error);
            if (tiled_error > 1.25*base_error + 1e-6) {
                fprintf(stderr, "FAIL: ctx=%d K=%s V=%s queries=%d pass=%d tiled_error=%g base_error=%g\n",
                        valid_tokens, ggml_type_name(k_type), ggml_type_name(v_type), queries, pass, tiled_error, base_error);
                return false;
            }
            continue;
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            const double error = std::abs(double(actual[i]) - reference[i]);
            worst = std::max(worst, error);
            if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
                    error > 1e-5 + 1e-4*std::abs(reference[i])) {
                fprintf(stderr, "FAIL: ctx=%d K=%s V=%s pass=%d i=%zu ref=%g tiled=%g\n",
                        valid_tokens, ggml_type_name(k_type), ggml_type_name(v_type), pass, i, reference[i], actual[i]);
                return false;
            }
        }
    }
    if ((selected.load() != 0) != expect_tiled) {
        fprintf(stderr, "FAIL: tiled dispatch expected=%d observed=%d (requires Turing or newer)\n",
                expect_tiled, selected.load());
        return false;
    }
    printf("PASS ctx=%d K=%s V=%s queries=%d gqa=%d sinks=%d bias=%g softcap=%g interleaved=%d tiled=%d max_abs=%g\n",
            valid_tokens, ggml_type_name(k_type), ggml_type_name(v_type), queries, gqa, sinks, bias, softcap, interleaved, expect_tiled, worst);
    return true;
}

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        fprintf(stderr, "No CUDA device available\n");
        return 77;
    }
    ggml_backend_cuda_log_set_callback(log_callback, nullptr);
    for (int context : {256, 257, 2048, 20000, 50000, 65536}) {
        for (auto v : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            if (!compare(context, v)) return 1;
        }
    }
    if (!compare(2048, GGML_TYPE_Q8_0, 1, 8, true)) return 1;
    if (!compare(2048, GGML_TYPE_Q4_0, 1, 8, false, 0.0f, 0.0f, GGML_TYPE_Q4_0, false)) return 1;
    if (!compare(2048, GGML_TYPE_Q8_0, 1, 8, false, 0.0f, 0.0f, GGML_TYPE_Q4_0, false)) return 1;
    // Two queries is the MTP verification batch.
    for (int context : {257, 2048, 28000, 65536}) {
        for (auto v : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            if (!compare(context, v, 2)) return 1;
        }
    }
    // These must use the old path even when the option is enabled.
    if (!compare(2048, GGML_TYPE_Q4_0, 3)) return 1;
    if (!compare(2048, GGML_TYPE_Q4_0, 1, 4)) return 1;
    if (!compare(2048, GGML_TYPE_Q8_0, 1, 8, false, 1.0f)) return 1;
    if (!compare(2048, GGML_TYPE_Q8_0, 1, 8, false, 0.0f, 4.0f)) return 1;
    if (!compare(2048, GGML_TYPE_Q8_0, 1, 8, false, 0.0f, 0.0f, GGML_TYPE_Q8_0)) return 1;
    ggml_backend_cuda_log_set_callback(nullptr, nullptr);
    return 0;
}
