#include "speculative-confidence.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static int check(ggml_backend_t backend, int count, int valid, float probability) {
    const int winner = valid/2;
    std::vector<float> logits(count, -INFINITY);
    const float other = std::log((1.0f-probability)/(probability*(valid-1)));
    std::fill_n(logits.data(), valid, other);
    logits[winner] = 0;
    double denom = 0;
    for (float value : logits) denom += std::exp(double(value));
    const double expected = 1/denom;
    std::vector<llama_token_data> cpu(count);
    for (int i = 0; i < count; ++i) cpu[i] = {i, logits[i], 0};
    llama_token_data_array cur = {cpu.data(), cpu.size(), -1, false};
    common_mtp_confidence_apply(nullptr, &cur);
    int errors = cur.size != 1 || cur.data[0].id != winner || std::abs(cur.data[0].p-expected)>3e-7;
    ggml_init_params ip = {1024*1024, nullptr, true};
    auto * ctx = ggml_init(ip);
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    llama_sampler_data data = {input, nullptr, nullptr, nullptr};
    auto * graph = ggml_new_graph(ctx);
    common_mtp_confidence_graph(nullptr, ctx, graph, &data);
    ggml_build_forward_expand(graph, data.logits);
    ggml_build_forward_expand(graph, data.probs);
    ggml_build_forward_expand(graph, data.sampled);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(input, logits.data(), 0, logits.size()*sizeof(float));
    errors += ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS;
    int32_t id = -1;
    float p = 0;
    ggml_backend_tensor_get(data.sampled, &id, 0, sizeof(id));
    ggml_backend_tensor_get(data.probs, &p, 0, sizeof(p));
    errors += id != winner || std::abs(p-expected)>3e-7;
    const double top10 = 1/(1 + 9*std::exp(double(other)));
    const bool rejects_top10 = std::abs(top10-expected)>3e-7;
    const bool rejects_tiny_cut = std::abs((p+1e-6)-expected)>3e-7;
    errors += !rejects_top10 || !rejects_tiny_cut;
    errors += (p >= .7f) != (expected >= double(.7f));
    std::printf("count=%d valid=%d p=%.9g expected=%.9g errors=%d top10_cut=%d tiny_cut=%d\n",
                count,valid,p,expected,errors,rejects_top10,rejects_tiny_cut);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return errors;
}

int main(int argc, char ** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    auto backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) return 2;
    if (!cuda) ggml_backend_cpu_set_n_threads(backend,1);
    int errors = 0;
    for (int n : {17,106299,248320})
        for (float p : {.1f,.69999f,.70001f,.99f})
            errors += check(backend,n,std::min(n,106299),p);
    ggml_backend_free(backend);
    std::printf("CONFIDENCE_CHECK errors=%d\n",errors);
    return errors ? 1 : 0;
}
