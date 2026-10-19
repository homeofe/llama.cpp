#include "scale-unary-fused.cuh"
#include "unary.cuh"

// fused GGML_OP_SCALE + GGML_UNARY_OP_SILU: dst[i] = t * sigmoid(t), t = scale * x[i]
static __global__ void scale_silu_f32(const float * x, float * dst, const float scale, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_op_silu_single(scale * x[i]);
}

static void scale_silu_f32_cuda(const float * x, float * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_SILU_BLOCK_SIZE - 1) / CUDA_SCALE_SILU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SCALE_SILU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_silu_f32, launch_params, x, dst, scale, k);
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, (float *) src->op_params + 0, sizeof(float));

    scale_silu_f32_cuda(src0_d, dst_d, scale, ggml_nelements(src0), stream);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_SIGMOID + GGML_OP_SCALE: dst[i] = postscale * sigmoid(prescale * x[i])
static __global__ void scale_sigmoid_scale_f32(const float * x, float * dst, const float prescale, const float postscale, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = postscale * (1.0f / (1.0f + expf(-(prescale * x[i]))));
}

static void scale_sigmoid_scale_f32_cuda(const float * x, float * dst, const float prescale, const float postscale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_SILU_BLOCK_SIZE - 1) / CUDA_SCALE_SILU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SCALE_SILU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_sigmoid_scale_f32, launch_params, x, dst, prescale, postscale, k);
}

void ggml_cuda_op_scale_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float prescale;
    float postscale;
    memcpy(&prescale,  (float *) src->op_params + 0, sizeof(float));
    memcpy(&postscale, (float *) dst->op_params + 0, sizeof(float));

    scale_sigmoid_scale_f32_cuda(src0_d, dst_d, prescale, postscale, ggml_nelements(src0), stream);
}
