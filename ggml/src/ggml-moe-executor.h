#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scheduler-private synchronous executor. Invoke worker(data, ith) exactly once
// per ith in [0, n_threads), concurrently, and join before returning.
typedef void (*ggml_compute_worker_fn)(void * data, int ith);
typedef void (*ggml_compute_executor_fn)(void * executor_data,
        ggml_compute_worker_fn worker, void * worker_data, int n_threads);
GGML_API enum ggml_status ggml_graph_compute_with_executor(struct ggml_cgraph * graph,
        struct ggml_cplan * plan, ggml_compute_executor_fn executor, void * executor_data);

#ifdef __cplusplus
}
#endif
