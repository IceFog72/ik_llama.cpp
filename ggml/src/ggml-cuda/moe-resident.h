#pragma once
#include "ggml-backend.h"

// Scheduler-private masked row merge. NULL src clears CPU-assigned rows.
// The launch is asynchronous on the backend stream. Tensors must be CUDA,
// contiguous F32 rows with contiguous I32 IDs and one token.
extern "C" GGML_API void ggml_backend_cuda_moe_copy_rows(ggml_backend_t backend,
        ggml_tensor * dst, const ggml_tensor * src, const ggml_tensor * ids);
