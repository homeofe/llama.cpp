#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <utility>

struct state {
    std::mutex lock;
    std::map<llama_context *, std::pair<ggml_threadpool *, ggml_threadpool *>> contexts;
    decltype(ggml_threadpool_new) * create = nullptr;
    decltype(ggml_threadpool_free) * destroy = nullptr;
};

static state & registry() {
    static auto * value = new state;
    return *value;
}

static ggml_threadpool * create_pool(ggml_threadpool_params * params) {
    auto * pool = registry().create(params);
    std::lock_guard<std::mutex> lock(registry().lock);
    std::fprintf(stderr, "R10_POOL create %p threads=%d poll=%u strict=%d mask=", (void *) pool,
                 params->n_threads, (unsigned) params->poll, params->strict_cpu);
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        if (params->cpumask[i]) {
            std::fprintf(stderr, "%d,", i);
        }
    }
    std::fprintf(stderr, "\n");
    return pool;
}

static void free_pool(ggml_threadpool * pool) {
    if (pool) {
        std::lock_guard<std::mutex> lock(registry().lock);
        for (const auto & entry : registry().contexts) {
            if (entry.second.first == pool || entry.second.second == pool) {
                std::fprintf(stderr, "R10_POOL violation pool=%p live_context=%p\n", (void *) pool, (void *) entry.first);
                std::fflush(stderr);
                std::_Exit(86);
            }
        }
        std::fprintf(stderr, "R10_POOL free %p\n", (void *) pool);
    }
    registry().destroy(pool);
}

extern "C" void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    static auto * original = (decltype(ggml_backend_reg_get_proc_address) *) dlsym(RTLD_NEXT, "ggml_backend_reg_get_proc_address");
    if (!original) {
        std::abort();
    }
    auto * result = original(reg, name);
    if (std::strcmp(name, "ggml_threadpool_new") == 0 && result) {
        registry().create = (decltype(ggml_threadpool_new) *) result;
        return (void *) create_pool;
    }
    if (std::strcmp(name, "ggml_threadpool_free") == 0 && result) {
        registry().destroy = (decltype(ggml_threadpool_free) *) result;
        return (void *) free_pool;
    }
    return result;
}

extern "C" void llama_attach_threadpool(llama_context * ctx, ggml_threadpool * pool, ggml_threadpool * batch) {
    static auto * original = (decltype(llama_attach_threadpool) *) dlsym(RTLD_NEXT, "llama_attach_threadpool");
    if (!original) {
        std::abort();
    }
    original(ctx, pool, batch);
    std::lock_guard<std::mutex> lock(registry().lock);
    registry().contexts[ctx] = {pool, batch};
    std::fprintf(stderr, "R10_POOL attach ctx=%p pool=%p batch=%p\n", (void *) ctx, (void *) pool, (void *) batch);
}

extern "C" void llama_free(llama_context * ctx) {
    static auto * original = (decltype(llama_free) *) dlsym(RTLD_NEXT, "llama_free");
    if (!original) {
        std::abort();
    }
    original(ctx);
    std::lock_guard<std::mutex> lock(registry().lock);
    registry().contexts.erase(ctx);
    std::fprintf(stderr, "R10_POOL context_freed %p\n", (void *) ctx);
}
