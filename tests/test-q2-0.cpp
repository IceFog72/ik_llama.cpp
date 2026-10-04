#include "ggml.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static void check(bool ok, const char * message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static void close_enough(float actual, float expected) {
    check(std::isfinite(actual) && fabsf(actual - expected) <= 1e-4f * (1.0f + fabsf(expected)), "numeric result");
}

// Packed bytes exercise every two-bit code, including +2, with different block scales.
static std::vector<block_q2_0> weights(int k, int rows) {
    std::vector<block_q2_0> data(k / QK2_0 * rows);
    for (size_t b = 0; b < data.size(); ++b) {
        data[b].d = ggml_fp32_to_fp16(0.125f * (1 + b % 7));
        for (int j = 0; j < QK2_0 / 4; ++j) {
            data[b].qs[j] = uint8_t(b * 29 + j * 17);
        }
    }
    return data;
}

static float weight(const block_q2_0 * row, int j) {
    const auto & block = row[j / QK2_0];
    const int pos = j % QK2_0;
    return ggml_fp16_to_fp32(block.d) * (int((block.qs[pos / 4] >> (2 * (pos % 4))) & 3) - 1);
}

static void test_format() {
    check(int(GGML_TYPE_Q2_0) == 42, "GGUF type ID");
    check(ggml_blck_size(GGML_TYPE_Q2_0) == 64, "block size");
    check(ggml_type_size(GGML_TYPE_Q2_0) == 18, "encoded block size");
    check(ggml_row_size(GGML_TYPE_Q2_0, 640) == 180, "640-wide row size");
    auto traits = ggml_internal_get_type_traits(GGML_TYPE_Q2_0);
    check(traits.to_float && traits.from_float && traits.vec_dot, "CPU traits");
    check(traits.vec_dot_type == GGML_TYPE_Q8_0, "activation type");
    auto data = weights(640, 1);
    std::vector<float> decoded(640);
    traits.to_float(data.data(), decoded.data(), 640);
    for (int j = 0; j < 640; ++j) {
        close_enough(decoded[j], weight(data.data(), j));
    }
    std::vector<block_q8_0> activation(640 / QK8_0);
    float expected_dot = 0;
    for (size_t b = 0; b < activation.size(); ++b) {
        activation[b].d = ggml_fp32_to_fp16(0.25f * (1 + b % 5));
        for (int j = 0; j < QK8_0; ++j) {
            activation[b].qs[j] = int8_t((b * 11 + j * 7) % 255 - 127);
            expected_dot += decoded[b * QK8_0 + j] * ggml_fp16_to_fp32(activation[b].d) * activation[b].qs[j];
        }
    }
    float actual_dot = 0;
    traits.vec_dot(640, &actual_dot, 0, data.data(), 0, activation.data(), 0, 1);
    close_enough(actual_dot, expected_dot);
    check(ggml_validate_row_data(GGML_TYPE_Q2_0, data.data(), data.size() * sizeof(block_q2_0)), "valid block data");
    data[0].d = 0x7c00;
    check(!ggml_validate_row_data(GGML_TYPE_Q2_0, data.data(), data.size() * sizeof(block_q2_0)), "reject infinite scale");

    std::vector<float> input(128);
    const float values[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    for (int j = 0; j < 64; ++j) {
        input[j] = values[j % 5];
    }
    std::vector<block_q2_0> encoded(4);
    memset(encoded.data(), 0xa5, encoded.size() * sizeof(block_q2_0));
    const auto before = encoded;
    const size_t size = ggml_quantize_chunk(GGML_TYPE_Q2_0, input.data(), encoded.data(), 64, 1, 64, nullptr, nullptr);
    check(size == 18, "chunk size");
    check(memcmp(encoded.data(), before.data(), 18) == 0, "chunk preserves prefix");
    check(memcmp(encoded.data() + 2, before.data() + 2, 36) == 0, "chunk preserves suffix");
    check(encoded[1].d == 0, "zero block scale");
    for (auto q : encoded[1].qs) {
        check(q == 0x55, "zero block encoding");
    }
    traits.from_float(input.data(), encoded.data(), 64);
    for (int j = 0; j < 64; ++j) {
        close_enough(weight(encoded.data(), j), roundf(input[j]));
    }
}

static void test_graph(int k, int tokens, int threads, bool moe) {
    fprintf(stderr, "%s k=%d tokens=%d threads=%d\n", moe ? "MUL_MAT_ID" : "MUL_MAT", k, tokens, threads);
    const int rows = 33;
    const int experts = moe ? 4 : 1;
    const int used = moe ? 2 : 1;
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    check(ctx != nullptr, "graph context");
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, k, rows, experts);
    ggml_tensor * b = moe ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, tokens)
                         : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    ggml_tensor * ids = moe ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens) : nullptr;
    auto data = weights(k, rows * experts);
    memcpy(a->data, data.data(), ggml_nbytes(a));
    auto * input = static_cast<float *>(b->data);
    for (int t = 0; t < tokens; ++t) {
        for (int u = 0; u < used; ++u) {
            if (moe) {
                static_cast<int32_t *>(ids->data)[t * used + u] = (t + 2 * u) % experts;
            }
            for (int j = 0; j < k; ++j) {
                input[(t * used + u) * k + j] = float((j * 13 + t * 7 + u) % 255 - 127);
                if (j % QK8_0 == QK8_0 - 1) {
                    input[(t * used + u) * k + j] = 127.0f;
                }
            }
        }
    }
    ggml_tensor * out = moe ? ggml_mul_mat_id(ctx, a, b, ids) : ggml_mul_mat(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    check(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "matrix graph compute");
    for (int t = 0; t < tokens; ++t) {
        for (int u = 0; u < used; ++u) {
            const int expert = moe ? static_cast<int32_t *>(ids->data)[t * used + u] : 0;
            for (int r = 0; r < rows; ++r) {
                float expected = 0;
                const auto * row = data.data() + (expert * rows + r) * (k / QK2_0);
                for (int j = 0; j < k; ++j) {
                    expected += weight(row, j) * input[(t * used + u) * k + j];
                }
                close_enough(static_cast<float *>(out->data)[(t * used + u) * rows + r], expected);
            }
        }
    }
    ggml_free(ctx);
}

static void test_gguf(const char * model = nullptr) {
    char filename[] = "/tmp/ik-q2-0-XXXXXX";
    if (!model) {
        int fd = mkstemp(filename);
        check(fd >= 0, "temporary GGUF");
        close(fd);
        auto * ctx = ggml_init({1024 * 1024, nullptr, false});
        auto * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, 640, 3, 2);
        ggml_set_name(tensor, "blk.1.ffn_down_exps.weight");
        auto data = weights(640, 6);
        memcpy(tensor->data, data.data(), ggml_nbytes(tensor));
        auto * file = gguf_init_empty();
        gguf_add_tensor(file, tensor);
        gguf_write_to_file(file, filename, false);
        gguf_free(file);
        ggml_free(ctx);
        model = filename;
    }
    ggml_context * ctx = nullptr;
    auto * file = gguf_init_from_file(model, {true, &ctx});
    check(file != nullptr && ctx != nullptr, "GGUF header and tensor allocation");
    int count = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
        if (gguf_get_tensor_type(file, i) == GGML_TYPE_Q2_0) {
            auto * tensor = ggml_get_tensor(ctx, gguf_get_tensor_name(file, i));
            check(tensor && tensor->ne[0] == 640, "GGUF Q2_0 tensor shape");
            ++count;
        }
    }
    check(count == (model == filename ? 1 : 9), "Q2_0 tensor count");
    gguf_free(file);
    ggml_free(ctx);
    if (model == filename) {
        unlink(filename);
    }
}

int main(int argc, char ** argv) {
    auto * init = ggml_init({1024, nullptr, false});
    check(init != nullptr, "initialize conversion tables");
    ggml_free(init);
    test_format();
    test_gguf();
    for (int k : {64, 640}) {
        for (int tokens : {1, 7}) {
            for (int threads : {1, 4}) {
                test_graph(k, tokens, threads, false);
                test_graph(k, tokens, threads, true);
            }
        }
    }
    if (argc == 2) {
        test_gguf(argv[1]);
    }
    puts("Q2_0 format, quantization, GGUF, MUL_MAT and MUL_MAT_ID: PASS");
}
