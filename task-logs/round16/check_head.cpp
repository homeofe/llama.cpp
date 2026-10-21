#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int check(ggml_backend_t backend, const std::vector<int64_t> & ids, int outputs, int cut) {
    const int width = 32, vocab = 17, rows = ids.size();
    ggml_init_params ip = { 1024*1024, nullptr, true };
    auto * ctx = ggml_init(ip);
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, outputs);
    auto * map = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);
    auto * small = ggml_mul_mat(ctx, w, x);
    auto * logits = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, vocab, outputs), -INFINITY);
    auto * scatter = ggml_set_rows(ctx, logits, ggml_reshape_3d(ctx, small, 1, rows, outputs),
                                  ggml_reshape_3d(ctx, map, rows, 1, 1));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, scatter);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> weights(width*rows), inputs(width*outputs), expected(vocab*outputs, -INFINITY), result(vocab*outputs);
    for (int t = 0; t < outputs; ++t) {
        for (int k = 0; k < width; ++k) inputs[t*width+k] = ((k+t)%5-2)/8.0f;
        for (int r = 0; r < rows; ++r) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                float value = ((ids[r]*3+k)%13-6)/16.0f;
                weights[r*width+k] = value;
                sum += value*inputs[t*width+k];
            }
            expected[t*vocab+ids[r]] = sum;
        }
    }
    auto actual_ids = ids;
    if (cut == 1) std::swap(actual_ids[0], actual_ids[1]);
    if (cut == 2) weights[0] += 1e-6f;
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size()*sizeof(float));
    ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size()*sizeof(float));
    ggml_backend_tensor_set(map, actual_ids.data(), 0, actual_ids.size()*sizeof(int64_t));
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(scatter, result.data(), 0, result.size()*sizeof(float));
    int errors = status != GGML_STATUS_SUCCESS;
    for (size_t i = 0; i < result.size(); ++i) errors += std::memcmp(&result[i], &expected[i], sizeof(float)) != 0;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return errors;
}

int main(int argc, char ** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    auto backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) return 2;
    if (!cuda) ggml_backend_cpu_set_n_threads(backend, 1);
    std::vector<int64_t> identity;
    for (int i = 0; i < 17; ++i) identity.push_back(i);
    int cases = 0;
    for (int outputs : {1, 2, 4, 7}) {
        for (const auto & ids : {identity, std::vector<int64_t>{16, 2, 11, 0, 7}}) {
            const int clean = check(backend, ids, outputs, 0);
            const int swapped = check(backend, ids, outputs, 1);
            const int tiny = check(backend, ids, outputs, 2);
            const int restored = check(backend, ids, outputs, 0);
            std::printf("outputs=%d rows=%zu clean=%d swapped=%d tiny=%d restored=%d\n", outputs, ids.size(), clean, swapped, tiny, restored);
            if (clean || restored || !swapped || !tiny) return 1;
            ++cases;
        }
    }
    ggml_backend_free(backend);
    std::printf("HEAD_CHECK_OK backend=%s cases=%d\n", cuda ? "CUDA" : "CPU", cases);
    return 0;
}
