// check_mul_mat_bf16.cu - CUDA mul_mat on BF16 weights against the CPU reference, on the
// qwen4exp hc shapes with realistic weight magnitudes. Run it twice: once with
// GGML_CUDA_BF16_AS_F16 unset (old path) and once with =1 (converted path); each run must match
// the CPU reference within nmse < 5e-4 (the same tolerance test-backend-ops uses).
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ggml_context ggml_context;
typedef struct ggml_tensor ggml_tensor;

static double max_abs_diff(const float * a, const float * b, int64_t n, double * nmse) {
    double mx = 0.0, se = 0.0, sd = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = fabs((double) a[i] - (double) b[i]);
        if (d > mx) mx = d;
        se += d * d;
        sd += (double) b[i] * (double) b[i];
    }
    *nmse = se / (sd > 0 ? sd : 1);
    return mx;
}

static float * run_on(ggml_backend_t backend, const uint16_t * w, const float * x, int64_t m, int64_t n, int64_t k) {
    struct ggml_init_params ip = {0};
    ip.mem_size = 64 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(a, w, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, x, 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const int64_t on = m * n;
    float * res = (float *) malloc(on * sizeof(float));
    ggml_backend_tensor_get(out, res, 0, on * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(void) {
    const int64_t shapes[][3] = {  // {m, n, k}: hc_attn_up, hc_attn_down, router
        {10240, 4, 320}, {320, 4, 10240}, {512, 4, 2560}, {640, 4, 2560},
    };
    int failures = 0;
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        const int64_t m = shapes[s][0], n = shapes[s][1], k = shapes[s][2];
        uint16_t * w = (uint16_t *) malloc(m * k * 2);
        float * x = (float *) malloc(k * n * sizeof(float));
        unsigned int seed = 42 + s;
        for (int64_t i = 0; i < m * k; ++i) {
            seed = seed * 1664525u + 1013904223u;
            const float v = ((seed >> 8) & 0xffff) / 65536.0f - 0.5f;   // +-0.5, weight scale
            uint32_t f;
            memcpy(&f, &v, 4);
            const uint16_t bits = (uint16_t) ((f >> 16) >> 16);         // truncate to bf16
            memcpy(w + i, &bits, 2);
        }
        for (int64_t i = 0; i < k * n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            x[i] = ((seed >> 8) & 0xffff) / 65536.0f - 0.5f;
        }
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_t cuda = ggml_backend_cuda_init(0);
        // run this binary once with the switch off and once with it on: each run's CUDA result
        // must match the CPU reference on its own
        float * ref = run_on(cpu, w, x, m, n, k);
        float * got = run_on(cuda, w, x, m, n, k);
        double nmse = 0;
        const double mx = max_abs_diff(ref, got, m * n, &nmse);
        const int bad = !(nmse < 5e-4);
        printf("shape %lldx%lldx%lld: nmse %.2e (max %.3e) -> %s\n",
               (long long) m, (long long) n, (long long) k, nmse, mx, bad ? "FAIL" : "ok");
        failures += bad;
        free(ref); free(got); free(w); free(x);
        ggml_backend_free(cpu);
        ggml_backend_free(cuda);
    }
    printf(failures ? "RESULT: FAIL %d\n" : "RESULT: PASS\n", failures);
    return failures ? 1 : 0;
}
