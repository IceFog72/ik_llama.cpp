#pragma once
#include "ggml-backend.h"

// Scheduler-private masked row merge. NULL src clears CPU-assigned rows.
// The launch is asynchronous on the backend stream. Tensors must be CUDA,
// contiguous F32 rows with contiguous I32 IDs across the token planes.
extern "C" GGML_API void ggml_backend_cuda_moe_copy_rows(ggml_backend_t backend,
        ggml_tensor * dst, const ggml_tensor * src, const ggml_tensor * ids);

// Private opt-in residency helpers. All launches use the backend stream.
extern "C" GGML_API void ggml_backend_cuda_moe_histogram(ggml_backend_t backend,
        const ggml_tensor * ids, uint32_t * histogram, int experts);
extern "C" GGML_API void ggml_backend_cuda_moe_profile(ggml_backend_t backend,
        uint32_t * histogram, int layers, int experts);
// mapped[i] = slots[ids[i]], or -1 for an invalid expert ID.
extern "C" GGML_API void ggml_backend_cuda_moe_route(ggml_backend_t backend,
        const ggml_tensor * ids, const int32_t * slots, ggml_tensor * mapped, int experts);
extern "C" GGML_API void ggml_backend_cuda_moe_scatter(ggml_backend_t backend,
        ggml_tensor * dst, const ggml_tensor * packed, const ggml_tensor * gpu_ids);
extern "C" GGML_API bool ggml_backend_cuda_moe_event_ready(ggml_backend_event_t event);
// The stream waits until a host thread stores a value >= `value` (wrapping
// compare) to `flag`, a word in pinned host memory. A one-thread kernel polls
// it, so later stream work needs no host launch. After about 10 s it sets
// *error (also pinned) to 1 and lets the stream continue. A hardware stream
// wait replaces the kernel when the driver enables stream memory operations.
// Returns false when unsupported (HIP, MUSA or compute capability below 7.0);
// nothing is queued then.
extern "C" GGML_API bool ggml_backend_cuda_moe_gate(ggml_backend_t backend,
        const uint32_t * flag, uint32_t value, uint32_t * error);
extern "C" GGML_API ggml_backend_event_t ggml_backend_cuda_moe_timed_event(ggml_backend_t backend);
extern "C" GGML_API double ggml_backend_cuda_moe_elapsed_us(ggml_backend_event_t start, ggml_backend_event_t end);
extern "C" GGML_API size_t ggml_backend_cuda_moe_cache_granularity(int device);
extern "C" GGML_API ggml_backend_buffer_t ggml_backend_cuda_moe_cache_new(
        int device, size_t capacity);
// Caller has fenced readers/writers. Physical pages change; base stays fixed.
extern "C" GGML_API bool ggml_backend_cuda_moe_cache_resize(
        ggml_backend_buffer_t buffer, size_t bytes);
extern "C" GGML_API void ggml_backend_cuda_moe_cache_move(ggml_backend_t backend,
        ggml_backend_buffer_t buffer, size_t dst, size_t src, size_t bytes);
