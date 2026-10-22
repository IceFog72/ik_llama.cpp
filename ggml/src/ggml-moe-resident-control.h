#pragma once

#include "ggml-backend.h"

// Private graph-construction contract. A marked down projection may finish
// after independent GPU nodes, at its marked direct consumer (split boundary).
enum {
    GGML_MOE_RESIDENT_DEFERRED_DOWN = 0x52534e44,
    GGML_MOE_RESIDENT_SHARED = 0x52534e53,
    GGML_MOE_RESIDENT_CPU_ROUTES = 0x52534e43,
    // Next-layer expert forecast (IDs only, never an arithmetic input). The
    // second-to-last op parameter holds the producing layer.
    GGML_MOE_RESIDENT_PREGATE = 0x52534e46,
    GGML_MOE_RESIDENT_JOIN = 0xff,
};

#ifdef __cplusplus
extern "C" {
#endif

// Reclaim only expendable expert storage before a companion context allocates.
// Keeps the graph arena; waits for all users and invalidates CUDA captures.
GGML_API void ggml_backend_sched_moe_resident_release_cache(ggml_backend_sched_t sched);

// Tokens are copied; these APIs retain no caller or predictor-context pointers.
GGML_API void ggml_backend_sched_moe_resident_set_tokens(ggml_backend_sched_t sched, const int32_t * tokens, int n);
GGML_API void ggml_backend_sched_moe_resident_predict(ggml_backend_sched_t sched, const int32_t * tokens, int n);

#ifdef __cplusplus
}
#endif
