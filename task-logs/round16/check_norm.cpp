#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int check(ggml_backend_t backend, int tokens, bool global) {
    const int width = 2560, streams = 4, hidden = width*streams;
    const float epsilon = 1e-6f;
    ggml_init_params ip = {1024*1024, nullptr, true};
    auto * ctx = ggml_init(ip);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, streams);
    auto * norm = global
        ? ggml_mul(ctx, ggml_rms_norm(ctx, input, epsilon), ggml_reshape_1d(ctx, weight, hidden))
        : ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, input, width, streams, tokens), epsilon), weight);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, norm);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const float scales[] = {0.25f, 1.0f, 4.0f, 16.0f};
    std::vector<float> x(hidden*tokens), w(hidden), result(hidden*tokens);
    for (int k = 0; k < hidden; ++k) w[k] = 1.0f + (k%7)*0.03125f;
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < hidden; ++k) x[t*hidden+k] = ((k*13+t*7)%37-18)*0.125f*scales[k/width];
    ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(weight, w.data(), 0, w.size()*sizeof(float));
    int errors = ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS;
    ggml_backend_tensor_get(norm, result.data(), 0, result.size()*sizeof(float));
    int wrong_mode_errors = 0;
    double maximum = 0;
    for (int t = 0; t < tokens; ++t) {
        double sums[streams] = {}, total = 0;
        for (int k = 0; k < hidden; ++k) {
            const double square = double(x[t*hidden+k])*x[t*hidden+k];
            sums[k/width] += square;
            total += square;
        }
        for (int k = 0; k < hidden; ++k) {
            const double mean = global ? total/hidden : sums[k/width]/width;
            const double wrong_mean = global ? sums[k/width]/width : total/hidden;
            const double expected = x[t*hidden+k]*w[k]/std::sqrt(mean+epsilon);
            const double wrong = x[t*hidden+k]*w[k]/std::sqrt(wrong_mean+epsilon);
            const double error = std::abs(result[t*hidden+k]-expected);
            maximum = std::max(maximum,error);
            errors += error > 2e-5*(1+std::abs(expected));
            wrong_mode_errors += std::abs(result[t*hidden+k]-wrong) > 2e-5*(1+std::abs(wrong));
        }
    }
    std::printf("tokens=%d global=%d errors=%d wrong_mode_errors=%d max_abs_error=%.9g\n", tokens, global, errors, wrong_mode_errors, maximum);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return errors || !wrong_mode_errors;
}

int main(int argc, char ** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    auto backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) return 2;
    if (!cuda) ggml_backend_cpu_set_n_threads(backend, 1);
    for (int tokens : {1, 2, 4, 7})
        for (bool global : {false, true})
            if (check(backend, tokens, global)) return 1;
    ggml_backend_free(backend);
    std::puts("NORM_CHECK_OK");
}
