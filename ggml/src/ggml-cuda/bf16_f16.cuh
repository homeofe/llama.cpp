#pragma once

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_cuda_context;
#include <cstdint>
struct CUstream_st;
typedef struct CUstream_st * cudaStream_t;

// Opt-in (GGML_CUDA_BF16_AS_F16=1, off by default): on devices whose fused matmul path supports
// F16 but not BF16 (e.g. Turing), 2D BF16 weight tensors that every node of the graph consumes
// only via MUL_MAT are converted in place to F16 once, and mul_mat then reads them as F16.
// Both types have the same element size, so no extra VRAM is used. With the switch off, or when
// a tensor has any other consumer (get_rows, mul, add, mul_mat_id, ...), nothing changes.

// true when the environment switch is on and this device benefits (F16 fast path, no BF16 one)
bool ggml_cuda_bf16_as_f16_useful(const int cc);

// walks the graph, marks convertible tensors and converts them on the given stream
void ggml_cuda_bf16_as_f16_survey(struct ggml_backend_cuda_context & ctx, struct ggml_cgraph * graph);

// the conversion itself, exported for the standalone check program
void ggml_cuda_bf16_to_f16_inplace(uint16_t * x, const int64_t n, const cudaStream_t stream);

// true when this tensor's device memory has been converted to F16
bool ggml_cuda_bf16_as_f16_registered(const struct ggml_tensor * t);

// a scratch tensor copy with type F16, valid until the next call on this thread
const struct ggml_tensor * ggml_cuda_bf16_as_f16_view(const struct ggml_tensor * t);
