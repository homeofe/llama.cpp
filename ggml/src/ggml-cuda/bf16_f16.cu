#include "ggml.h"
#include "ggml-backend.h"

#include "ggml-cuda.h"
#include "common.cuh"
#include "bf16_f16.cuh"

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

// in place bf16 -> f16, round to nearest even, subnormals included: same element size, so the
// weight buffer can hold the converted data. The conversion is written as integer math because
// the hardware cvt path flushes f16 subnormal results on this architecture.
__global__ void ggml_cuda_bf16_to_f16_inplace_kernel(uint16_t * GGML_RESTRICT x, const int64_t n) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = blockIdx.x*(int64_t)blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint32_t f = (uint32_t) x[i] << 16;         // bf16 -> f32 is a plain shift
        const uint32_t sign = (f >> 16) & 0x8000u;
        int32_t exp = (int32_t) ((f >> 23) & 0xff) - 127 + 15;
        uint16_t out;
        if (exp >= 0x1f) {
            out = (uint16_t) (sign | 0x7c00u);            // overflow becomes inf; inf and nan map to f16 inf/nan
        } else if (exp > 0) {
            // bf16 carries 7 mantissa bits and f16 keeps 10, so the normal range converts exactly
            out = (uint16_t) (sign | ((uint32_t) exp << 10) | ((f >> 13) & 0x3ffu));
        } else {
            // below the f16 normal range: scale into the subnormal lattice, round to nearest even
            const uint32_t man = 0x400u | ((f >> 13) & 0x3ffu);
            const uint32_t shift = (uint32_t) (1 - exp);
            uint32_t kept = shift >= 12 ? 0u : man >> shift;
            const uint32_t rrem = shift >= 12 ? 0u : man & ((1u << shift) - 1);
            const uint32_t half = shift >= 12 ? 0u : 1u << (shift - 1);
            if (rrem > half || (rrem == half && (kept & 1))) {
                kept++;
            }
            out = (uint16_t) (sign | kept);
        }
        x[i] = out;
    }
}

void ggml_cuda_bf16_to_f16_inplace(uint16_t * x, const int64_t n, const cudaStream_t stream) {
    const int64_t blocks = (n + 255) / 256;
    ggml_cuda_bf16_to_f16_inplace_kernel<<<(int) blocks, 256, 0, stream>>>(x, n);
}

// data pointer -> nbytes of the converted area; guarded by g_mutex (set_tensor and compute may run on different threads)
static std::unordered_map<const void *, int64_t> g_converted;
static std::mutex g_mutex;

static bool env_switch_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_BF16_AS_F16");
        return e != nullptr && (e[0] == '1' || e[0] == 'y');
    }();
    return on;
}

bool ggml_cuda_bf16_as_f16_useful(const int cc) {
    if (!env_switch_on()) {
        return false;
    }
    // convert only where the fused matmul path takes F16 but not BF16 (e.g. Turing)
    return (turing_mma_available(cc) || volta_mma_available(cc)) && !ampere_mma_available(cc);
}

static bool buffer_is_cuda_weights(const ggml_tensor * t, const int device) {
    return t->buffer != nullptr
        && ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(device)
        && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

void ggml_cuda_bf16_as_f16_survey(ggml_backend_cuda_context & ctx, ggml_cgraph * graph) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_bf16_as_f16_useful(cc)) {
        return;
    }

    // a BF16 2D weight tensor is convertible when MUL_MAT is its only consumer in this graph
    std::unordered_map<const void *, const ggml_tensor *> candidates;
    std::unordered_set<const void *> vetoed;

    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT) {
            const ggml_tensor * src0 = node->src[0];
            if (src0 != nullptr && src0->type == GGML_TYPE_BF16 && src0->ne[2] == 1 && src0->ne[3] == 1
                    && ggml_is_contiguous(src0) && buffer_is_cuda_weights(src0, ctx.device)) {
                candidates.emplace(src0->data, src0);
            }
        } else {
            // every other op vetoes its sources, MUL_MAT_ID included
            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                vetoed.insert(node->src[s]->data);
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto & [ptr, t] : candidates) {
        if (vetoed.count(ptr) || g_converted.count(ptr)) {
            continue;
        }
        ggml_cuda_bf16_to_f16_inplace((uint16_t *) t->data, ggml_nbytes(t) / 2, ctx.stream());
        g_converted.emplace(ptr, ggml_nbytes(t));
    }
}

bool ggml_cuda_bf16_as_f16_registered(const ggml_tensor * t) {
    if (g_converted.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_converted.count(t->data) != 0;
}

const ggml_tensor * ggml_cuda_bf16_as_f16_view(const ggml_tensor * t) {
    static thread_local ggml_tensor view;
    view = *t;
    view.type = GGML_TYPE_F16;
    return &view;
}
