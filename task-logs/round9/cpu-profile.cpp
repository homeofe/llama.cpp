#include "ggml-backend.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static std::atomic<uint64_t> cpu_ns[10] = {};
static std::atomic<uint64_t> calls[10] = {};

extern "C" enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * graph) {
    using compute_fn = enum ggml_status (*)(ggml_backend_t, struct ggml_cgraph *);
    static const auto compute = reinterpret_cast<compute_fn>(dlsym(RTLD_NEXT, "ggml_backend_graph_compute_async"));
    if (!compute) {
        std::abort();
    }
    if (std::strncmp(ggml_backend_name(backend), "CPU", 3) != 0) {
        return compute(backend, graph);
    }
    int bucket = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MUL_MAT_ID_MASKED) {
            const int64_t tokens = node->src[2]->ne[1];
            const int current = tokens <= 8 ? (int) tokens : 9;
            if (current > bucket) {
                bucket = current;
            }
        }
    }
    const auto start = std::chrono::steady_clock::now();
    const auto result = compute(backend, graph);
    const auto duration = std::chrono::steady_clock::now() - start;
    cpu_ns[bucket].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count(), std::memory_order_relaxed);
    calls[bucket].fetch_add(1, std::memory_order_relaxed);
    return result;
}

__attribute__((destructor)) static void report() {
    for (int bucket = 0; bucket < 10; ++bucket) {
        std::fprintf(stderr, "round9_cpu_profile tokens=%d calls=%llu cpu_ms=%.3f\n", bucket,
            (unsigned long long) calls[bucket].load(), cpu_ns[bucket].load()/1e6);
    }
}
