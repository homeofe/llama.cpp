#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

size_t llama_moe_cache_env_size(const char * name, size_t fallback, size_t maximum = SIZE_MAX);

// Host policy shared by the hybrid cache and routing replay tests.
struct llama_moe_cache_lfu {
    struct fill {
        int32_t key;
        int32_t slot;
        int32_t evicted;
    };

    std::vector<double> counts;
    std::vector<int32_t> slot_of;
    std::vector<int32_t> key_of;
    std::vector<int32_t> candidates;
    std::vector<bool> pending;
    std::vector<bool> pinned;
    std::vector<double> prompt_counts;
    uint64_t decay = 512;
    uint64_t tokens = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;

    void init(size_t n_keys, int32_t n_slots, uint64_t decay_interval = 512);
    void seed(int32_t key, int32_t slot, double count, bool pin = false);
    void begin_token();
    void observe(int32_t key, bool use_cache = true, bool prompt = false);
    void clear_prompt();
    std::vector<fill> plan(size_t limit, bool prompt = false);
};

// keeps experts of host-resident MoE layers in device banks
// the default policy uploads misses; hybrid mode leaves misses on the CPU
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const char * profile = nullptr, bool adaptive = false, bool draft = false, int32_t layer_start = 0);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    bool is_hybrid() const;

    // Collect routes on the CPU branch, using the IDs already copied for the expert matmuls.
    ggml_tensor * build_observe(ggml_context * ctx, ggml_tensor * ids, int32_t il, bool use_cache) const;
    void begin_graph(int64_t n_tokens, const int8_t * output = nullptr, const int32_t * positions = nullptr,
                     const int32_t * n_seq_id = nullptr, int32_t * const * seq_id = nullptr, bool prefill = false, bool prefill_end = false);
    void end_graph(bool success);

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
