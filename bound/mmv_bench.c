// mmv_bench.c - standalone bound harness: ggml MUL_MAT on the qwen4exp dense shapes, CUDA backend.
// usage: mmv_bench <type> <m> <n> <k> <iters>
//   m = rows of the weight tensor, n = batch (tokens), k = columns; times ggml_backend_graph_compute
//   (the production CUDA graph path included) and reports us/iter and GB/s over the weight bytes.
#define _POSIX_C_SOURCE 200809L
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct ggml_context ggml_context;
typedef struct ggml_tensor ggml_tensor;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <type> <m> <n> <k> <iters>\n", argv[0]);
        return 2;
    }
    const char * type_name = argv[1];
    const int64_t m = atoll(argv[2]);
    const int64_t n = atoll(argv[3]);
    const int64_t k = atoll(argv[4]);
    const int iters = atoi(argv[5]);

    enum ggml_type type;
    if      (!strcmp(type_name, "F16"))     type = GGML_TYPE_F16;
    else if (!strcmp(type_name, "BF16"))    type = GGML_TYPE_BF16;
    else if (!strcmp(type_name, "Q8_0"))    type = GGML_TYPE_Q8_0;
    else if (!strcmp(type_name, "Q4_K"))    type = GGML_TYPE_Q4_K;
    else if (!strcmp(type_name, "Q6_K"))    type = GGML_TYPE_Q6_K;
    else if (!strcmp(type_name, "IQ4_NL")) type = GGML_TYPE_IQ4_NL;
    else if (!strcmp(type_name, "IQ4_XS")) type = GGML_TYPE_IQ4_XS;
    else if (!strcmp(type_name, "IQ2_S"))   type = GGML_TYPE_IQ2_S;
    else { fprintf(stderr, "unknown type %s\n", type_name); return 2; }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "cuda backend init failed\n");
        return 1;
    }

    struct ggml_init_params ip = {0};
    ip.mem_size = 64 * 1024 * 1024;
    ip.mem_buffer = NULL;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    // host data: random bytes are fine for perf (kernel timing is data independent)
    const size_t nb_a = ggml_nbytes(a);
    const size_t nb_b = ggml_nbytes(b);
    unsigned char * ha = malloc(nb_a);
    unsigned char * hb = malloc(nb_b);
    unsigned int seed = 12345;
    for (size_t i = 0; i < nb_a; i++) { seed = seed * 1664525u + 1013904223u; ha[i] = (unsigned char) (seed >> 24); }
    for (size_t i = 0; i < nb_b / sizeof(float); i++) { seed = seed * 1664525u + 1013904223u; ((float *) hb)[i] = ((seed >> 8) & 0xffff) / 65536.0f - 0.5f; }
    ggml_backend_tensor_set(a, ha, 0, nb_a);
    ggml_backend_tensor_set(b, hb, 0, nb_b);

    // warmup (also builds the CUDA graph on first compute)
    for (int i = 0; i < 8; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);

    const double t0 = now_s();
    for (int i = 0; i < iters; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);
    const double dt = now_s() - t0;

    const double us = dt / iters * 1e6;
    const double gbps = (double) nb_a / (dt / iters) / 1e9;
    printf("%s m=%lld n=%lld k=%lld: %8.2f us/iter  %6.1f GB/s over %7.2f MB weights\n",
           type_name, (long long) m, (long long) n, (long long) k, us, gbps, nb_a / 1e6);

    free(ha); free(hb);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
