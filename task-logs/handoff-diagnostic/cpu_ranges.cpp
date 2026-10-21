#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

extern "C" int nvtxRangePushA(const char *);
extern "C" int nvtxRangePop();

// Diagnostic preload only; the original operation is called exactly once.
extern "C" enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * graph) {
    using fn = enum ggml_status (*)(ggml_backend_t, struct ggml_cgraph *);
    static const auto compute = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_backend_graph_compute_async"));
    if (!compute) std::abort();
    if (std::strncmp(ggml_backend_name(backend), "CPU", 3) != 0) return compute(backend, graph);
    int64_t tokens = 0;
    const char * name = "none";
    const char * weight = "none";
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MUL_MAT_ID_MASKED) {
            tokens = node->src[2]->ne[1];
            name = node->name;
            weight = node->src[0]->name;
            break;
        }
    }
    char label[256];
    std::snprintf(label, sizeof(label), "handoff-cpu tokens=%lld node=%s weight=%s", (long long) tokens, name, weight);
    nvtxRangePushA(label);
    const auto result = compute(backend, graph);
    nvtxRangePop();
    return result;
}
