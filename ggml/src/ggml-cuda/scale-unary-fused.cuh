#pragma once

#include "common.cuh"

#define CUDA_SCALE_SILU_BLOCK_SIZE 1024

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);
void ggml_cuda_op_scale_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);
