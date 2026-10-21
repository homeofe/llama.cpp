#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Exact arithmetic oracle across changed shapes, source addresses, strides and scale parameters.
int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { return 2; }
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    constexpr int width = 32, max_rows = 8;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, max_rows);
    ggml_tensor * alternate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, max_rows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, max_rows);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_scale(ctx, sum, 1.0f);
    ggml_set_name(sum, "variant_sum");
    ggml_set_name(out, "variant_out");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { return 2; }
    void * original = a->data;
    std::vector<float> av(width * max_rows), alt(av.size()), bv(av.size()), got(av.size());
    int failures = 0, checks = 0;
    const int patterns[] = {0, 0, 1, 1, 2, 2, 0, 2, 1, 0, 1, 2, 3, 3, 4, 4, 0, 1, 2, 3, 4, 0};
    for (int phase = 0; phase < 4; ++phase) {
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (int variant : patterns) {
                auto shape = [&](ggml_tensor * t, int rows, int stride) {
                    t->ne[1] = rows;
                    t->nb[1] = sizeof(float) * stride;
                    t->nb[2] = t->nb[1] * rows;
                    t->nb[3] = t->nb[2];
                };
                a->data = original;
                shape(a, max_rows, width);
                shape(b, max_rows, width);
                for (size_t i = 0; i < av.size(); ++i) {
                    av[i] = float((i * 7 + checks) % 23 + 1);
                    alt[i] = float((i * 3 + checks) % 17 + 33);
                    bv[i] = float(int((i * 11 + checks) % 13) - 6);
                }
                ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
                ggml_backend_tensor_set(alternate, alt.data(), 0, alt.size() * sizeof(float));
                ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
                const int rows = phase == 0 ? variant + 1 : 3;
                const bool moved = phase == 1 && variant % 2;
                const int stride = phase == 2 && variant % 2 ? width * 2 : width;
                const float scales[] = {1.0f, 2.0f, -1.0f, 1.000001f, 0.5f};
                const float scale = phase == 3 ? scales[variant] : 1.0f;
                a->data = moved ? alternate->data : original;
                shape(a, rows, stride);
                for (auto * tensor : {b, sum, out}) { shape(tensor, rows, width); }
                memcpy(out->op_params, &scale, sizeof(scale));
                graph->uid = 0; // Force the same property checks as freshly rebuilt scheduler views.
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
                ggml_backend_tensor_get(out, got.data(), 0, rows * width * sizeof(float));
                bool bad = false;
                const auto & input = moved ? alt : av;
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < width; ++col) {
                        const float expected = (input[row * stride + col] + bv[row * width + col]) * scale;
                        if (memcmp(&got[row * width + col], &expected, sizeof(float))) { bad = true; }
                    }
                }
                ++checks;
                if (bad) {
                    ++failures;
                    if (failures <= 10) { printf("FAIL phase=%d variant=%d repeat=%d\n", phase, variant, repeat); }
                }
            }
        }
    }
    printf("checks=%d failures=%d\n", checks, failures);
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return failures ? 1 : 0;
}
