#include "../src/llama-moe-cache.h"
#include "../src/llama-model.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "llama-cpp.h"
#include "common.h"
#include "speculative.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

static void test_policy() {
    llama_moe_cache_lfu policy;
    policy.init(6, 2);
    policy.begin_token();
    policy.observe(0);
    policy.observe(1);
    GGML_ASSERT(policy.hits == 0 && policy.misses == 2);
    GGML_ASSERT(policy.slot_of[0] == -1 && policy.slot_of[1] == -1);
    auto fills = policy.plan(1);
    GGML_ASSERT(fills.size() == 1 && fills[0].key == 0 && fills[0].evicted == -1);

    policy.begin_token();
    policy.observe(1);
    fills = policy.plan(8);
    GGML_ASSERT(fills.size() == 1 && fills[0].key == 1);
    policy.begin_token();
    policy.observe(2);
    GGML_ASSERT(policy.plan(8).empty()); // Equal counts cannot evict a resident.
    policy.begin_token();
    policy.observe(2);
    fills = policy.plan(8);
    GGML_ASSERT(fills.size() == 1 && fills[0].key == 2 && fills[0].evicted == 0);
    GGML_ASSERT(policy.slot_of[0] == -1 && policy.slot_of[2] >= 0);

    const auto hits = policy.hits;
    policy.begin_token();
    policy.observe(2);
    policy.observe(2);
    GGML_ASSERT(policy.hits == hits + 2); // Repeated routes count separately.
    policy.observe(2, false);
    GGML_ASSERT(policy.hits == hits + 2); // Ordinary fallback does not use the cache.
    policy.plan(0);

    policy.init(4, 2);
    policy.seed(0, 0, 9);
    policy.seed(1, 1, 4);
    for (int t = 0; t < 512; ++t) {
        policy.begin_token();
    }
    GGML_ASSERT(policy.counts[0] == 9 && policy.counts[1] == 4);
    policy.begin_token();
    GGML_ASSERT(policy.counts[0] == 4.5 && policy.counts[1] == 2);
    policy.observe(2);
    policy.observe(2);
    GGML_ASSERT(policy.plan(8).empty());
    policy.observe(2);
    fills = policy.plan(8);
    GGML_ASSERT(fills.size() == 1 && fills[0].evicted == 1);

    policy.init(10, 3);
    policy.seed(0, 0, 1);
    policy.seed(1, 1, 2);
    policy.seed(2, 2, 3);
    for (int t = 0; t < 4; ++t) {
        policy.begin_token();
        policy.observe(3);
        policy.observe(4);
        policy.observe(5);
    }
    // One ubatch: all four tokens see the old residency, then at most two uploads.
    GGML_ASSERT(policy.hits == 0 && policy.misses == 12);
    fills = policy.plan(2);
    GGML_ASSERT(fills.size() == 2 && fills[0].evicted == 0 && fills[1].evicted == 1);
    GGML_ASSERT(policy.slot_of[2] >= 0 && policy.slot_of[5] < 0);
    GGML_ASSERT(policy.plan(8).empty());
    policy.init(6, 3, 64);
    policy.seed(0, 0, 1000, true);
    policy.seed(1, 1, 1000);
    policy.observe(2, false, true);
    policy.observe(2, false, true);
    policy.plan(0);
    policy.observe(3, false, true);
    policy.observe(4, false, true);
    fills = policy.plan(2, true);
    GGML_ASSERT(fills.size() == 2 && fills[0].key == 2 && fills[1].key == 3);
    GGML_ASSERT(policy.slot_of[0] == 0 && policy.slot_of[1] == -1 && policy.slot_of[4] == -1);
    policy.clear_prompt();
    GGML_ASSERT(policy.plan(2, true).empty());
    for (int i = 0; i < 64; ++i) {
        policy.begin_token();
    }
    GGML_ASSERT(policy.counts[0] == 1000);
    policy.begin_token();
    GGML_ASSERT(policy.counts[0] == 500);
    policy.observe(5, false, true);
    fills = policy.plan(10, true);
    GGML_ASSERT(fills.size() == 1 && fills[0].key == 5 && policy.slot_of[0] == 0);
    policy.init(2, 1);
    policy.seed(0, 0, 0, true);
    policy.observe(1, true, true);
    GGML_ASSERT(policy.plan(10).empty() && policy.plan(10, true).empty());
    puts("Prompt ranking, cross-ubatch candidates, static pinning and configurable decay passed");
    puts("LFU policy checks passed");
}

// Host storage with a deferred stream: async copies retain their source pointers until drain().
struct queued_backend {
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    ggml_backend_device device = *ggml_backend_get_device(cpu.get());
    ggml_backend backend = {};
    std::vector<std::function<void()>> queue;
    size_t syncs = 0;

    queued_backend() {
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "queued-host-test"; };
        backend.iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
            auto * self = static_cast<queued_backend *>(b->context);
            self->queue.push_back([=]() { memcpy((char *) t->data + offset, data, size); });
        };
        backend.iface.synchronize = [](ggml_backend_t b) {
            auto * self = static_cast<queued_backend *>(b->context);
            ++self->syncs;
            self->drain();
        };
    }

    void drain() {
        for (const auto & operation : queue) {
            operation();
        }
        queue.clear();
    }
};

struct test_env {
    std::string name;
    std::string previous;
    bool had_previous;

    static void set(const char * name, const char * value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    test_env(const char * name, const char * value) : name(name), had_previous(getenv(name) != nullptr) {
        if (had_previous) {
            previous = getenv(name);
        }
        set(name, value);
    }

    ~test_env() {
        set(name.c_str(), had_previous ? previous.c_str() : nullptr);
    }
};

static void test_config() {
    const char * name = "GGML_MOE_HYBRID_ADMIT_PREFILL";
    for (const char * value : {"", "-1", " 1", "1x", "184467440737095516160"}) {
        test_env setting(name, value);
        bool rejected = false;
        try {
            llama_moe_cache_env_size(name, 1024);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        GGML_ASSERT(rejected);
    }
    {
        test_env setting(name, nullptr);
        GGML_ASSERT(llama_moe_cache_env_size(name, 1024) == 1024);
    }
    {
        const std::string maximum = std::to_string(SIZE_MAX);
        test_env setting(name, maximum.c_str());
        GGML_ASSERT(llama_moe_cache_env_size(name, 0) == SIZE_MAX);
    }
    {
        test_env setting(name, "101");
        bool rejected = false;
        try {
            llama_moe_cache_env_size(name, 0, 100);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        GGML_ASSERT(rejected);
    }
}

static void test_ordering(const char * path, const char * profile, const char * trace_path) {
    queued_backend queued;
    ggml_backend_cpu_set_n_threads(queued.cpu.get(), 4);
    ggml_backend_dev_t devices[] = {&queued.device, nullptr};
    auto params = llama_model_default_params();
    params.devices = devices;
    params.n_gpu_layers = 99;
    params.use_extra_bufts = false;
    llama_model_ptr model(llama_model_load_from_file(path, params));
    GGML_ASSERT(model && model->layers.size() == 2 && model->hparams.n_expert == 2);
    const auto & layer = model->layers[0];
    std::vector<ggml_tensor *> weights = {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps};
    const auto buft = ggml_backend_cpu_buffer_type();
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    auto padded = [&](size_t n) { return (n + alignment - 1)/alignment*alignment; };
    size_t budget = 2*padded(2*sizeof(int32_t));
    for (const auto * w : weights) {
        GGML_ASSERT(w);
        budget += padded(3*w->nb[2]); // Two slots plus the bank's padding expert.
    }
    bool rejected = false;
    try {
        llama_moe_cache too_small(*model, {&queued.backend}, {buft}, 1, nullptr, true);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    GGML_ASSERT(rejected);
    llama_moe_cache cache(*model, {&queued.backend}, {buft}, budget, nullptr, true);
    auto map0 = cache.get_slot_map(0, 1, 1);
    auto map1 = cache.get_slot_map(1, 1, 1);
    GGML_ASSERT(map0 && map1);
    GGML_ASSERT(((int32_t *) map0->data)[0] == -1 && ((int32_t *) map1->data)[0] == -1);
    const size_t syncs = queued.syncs;

    auto collect = [&](llama_moe_cache & cache, int il, const std::vector<int32_t> & ids, bool success, const std::vector<int8_t> & output = {}, bool prefill = false, bool prefill_end = true) {
        const size_t nt = output.empty() ? ids.size() : output.size();
        std::vector<int32_t> positions(nt);
        for (size_t i = 0; i < nt; ++i) {
            positions[i] = 100 + i;
        }
        cache.begin_graph(nt, output.empty() ? nullptr : output.data(), positions.data(), nullptr, nullptr, prefill, prefill_end);
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, false}));
        // Use a strided view, as top-k IDs need not be contiguous.
        auto src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, ids.size());
        for (size_t t = 0; t < ids.size(); ++t) {
            ((int32_t *) src->data)[2*t] = ids[t];
            ((int32_t *) src->data)[2*t + 1] = -999;
        }
        auto view = ggml_view_2d(ctx.get(), src, 1, ids.size(), src->nb[1], 0);
        auto observed = cache.build_observe(ctx.get(), view, il, nt <= 8);
        auto graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, observed);
        GGML_ASSERT(ggml_backend_graph_compute(queued.cpu.get(), graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(memcmp(observed->data, ids.data(), ids.size()*sizeof(int32_t)) == 0);
        cache.end_graph(success);
    };
    auto expect_resident = [&](llama_moe_cache & subject, int il) {
        queued.queue.push_back([&, il]() {
            const auto * map = (const int32_t *) subject.get_slot_map(il, 1, 1)->data;
            const auto & l = model->layers[il];
            for (int e = 0; e < 2; ++e) {
                GGML_ASSERT(map[e] >= 0 && map[e] < 2);
                for (const auto * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps}) {
                    const auto * bank = subject.get_experts(w);
                    GGML_ASSERT(memcmp((const char *) bank->data + map[e]*bank->nb[2], (const char *) w->data + e*w->nb[2], w->nb[2]) == 0);
                }
            }
        });
    };

    collect(cache, 0, {0, 1}, true);
    GGML_ASSERT(!queued.queue.empty());
    expect_resident(cache, 0); // A submitted graph must see the old weights and maps before the eviction writes.
    collect(cache, 1, {0, 1, 0, 1}, true);
    expect_resident(cache, 1);
    queued.queue.push_back([&]() {
        GGML_ASSERT(((int32_t *) map0->data)[0] == -1 && ((int32_t *) map0->data)[1] == -1);
    });
    const size_t n_pending = queued.queue.size();
    collect(cache, 0, {0, 0, 0, 0}, false);
    GGML_ASSERT(queued.queue.size() == n_pending);
    GGML_ASSERT(queued.syncs == syncs); // Neither routing nor admission synchronizes the cache backend.
    queued.drain();
    // No stale routes from the aborted graph are counted or uploaded on the next evaluation.
    cache.begin_graph(0);
    cache.end_graph(true);
    GGML_ASSERT(queued.queue.empty());
    puts("Deferred backend ordering, async source lifetime, strided IDs and abort checks passed");

    if (profile) {
        llama_moe_cache frozen(*model, {&queued.backend}, {buft}, budget, profile, false);
        GGML_ASSERT(frozen.get_slot_map(0, 1, 1) && !frozen.get_slot_map(1, 1, 1));
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, false}));
        auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
        GGML_ASSERT(frozen.build_observe(ctx.get(), ids, 0, true) == ids);
        frozen.begin_graph(1);
        frozen.end_graph(true);
        GGML_ASSERT(queued.queue.empty());

        llama_moe_cache warm(*model, {&queued.backend}, {buft}, budget, profile, true);
        auto warm0 = warm.get_slot_map(0, 1, 1);
        auto warm1 = warm.get_slot_map(1, 1, 1);
        GGML_ASSERT(warm0 && warm1);
        GGML_ASSERT(memcmp(warm0->data, frozen.get_slot_map(0, 1, 1)->data, 2*sizeof(int32_t)) == 0);
        for (const auto * w : weights) {
            const auto * a = warm.get_experts(w);
            const auto * b = frozen.get_experts(w);
            GGML_ASSERT(memcmp(a->data, b->data, ggml_nbytes(a)) == 0);
        }
        collect(warm, 1, {0}, true);
        GGML_ASSERT(queued.queue.empty()); // Profile counts are one, so the new count ties.
        collect(warm, 1, {0}, true);
        GGML_ASSERT(!queued.queue.empty());
        queued.drain();
        GGML_ASSERT(((int32_t *) warm0->data)[0] == -1 && ((int32_t *) warm1->data)[0] >= 0);
        puts("Frozen mode and adaptive warm-start checks passed");

        llama_moe_cache cropped(*model, {&queued.backend}, {buft}, budget, profile, true);
        cropped.begin_graph(511);
        cropped.end_graph(true);
        collect(cropped, 1, {0}, true, {0, 1});
        GGML_ASSERT(!queued.queue.empty()); // Count after decay: 1 > 0.5. Counting before decay would tie at 0.5.
        queued.drain();
        GGML_ASSERT(((int32_t *) cropped.get_slot_map(1, 1, 1)->data)[0] >= 0);
        puts("Cropped output rows across the 512-token decay boundary passed");
    }
    {
        test_env prefill("GGML_MOE_HYBRID_PREFILL", "1");
        test_env cap("GGML_MOE_HYBRID_ADMIT", "1");
        test_env burst_cap("GGML_MOE_HYBRID_ADMIT_PREFILL", "2");
        llama_moe_cache burst(*model, {&queued.backend}, {buft}, budget, nullptr, true);
        const size_t before = queued.syncs;
        collect(burst, 0, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, true, std::vector<int8_t>(10, 0), true, false);
        GGML_ASSERT(queued.queue.empty());
        collect(burst, 1, {0}, true, {}, true, true);
        GGML_ASSERT(!queued.queue.empty());
        expect_resident(burst, 0);
        const size_t pending = queued.queue.size();
        collect(burst, 1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true, std::vector<int8_t>(10, 0), true, false);
        collect(burst, 1, {0}, false, {}, true, true);
        GGML_ASSERT(queued.queue.size() == pending);
        // The aborted prompt must not bias this prompt toward expert zero.
        collect(burst, 1, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}, true, {}, true, true);
        expect_resident(burst, 1);
        GGML_ASSERT(queued.syncs == before);
        queued.drain();
        GGML_ASSERT(((int32_t *) burst.get_slot_map(1, 1, 1)->data)[1] == 0);
        puts("Prefill burst ordering, small final tail, prompt ranking and abort reset passed");
    }
    if (profile) {
        test_env prefill("GGML_MOE_HYBRID_PREFILL", "1");
        test_env base("GGML_MOE_HYBRID_STATIC_PCT", "50");
        llama_moe_cache target(*model, {&queued.backend}, {buft}, budget, profile, true);
        const auto * target0 = (const int32_t *) target.get_slot_map(0, 1, 1)->data;
        GGML_ASSERT(target0[0] == 0 && target0[1] == -1);
        collect(target, 1, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, true, {}, true, true);
        queued.drain();
        GGML_ASSERT(target0[0] == 0 && target0[1] == -1);
        const auto * target1 = (const int32_t *) target.get_slot_map(1, 1, 1)->data;
        GGML_ASSERT(target1[0] == 1 && target1[1] == -1);
        // A shared-model MTP cache allocates only the head's layer range and ignores the target's base percentage.
        llama_moe_cache draft(*model, {&queued.backend}, {buft}, budget, nullptr, true, true, 1);
        GGML_ASSERT(!draft.get_slot_map(0, 1, 1));
        collect(draft, 1, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, true, {}, true, true);
        expect_resident(draft, 1);
        queued.drain();
        GGML_ASSERT(target0[0] == 0 && target0[1] == -1 && target1[0] == 1 && target1[1] == -1);
        puts("Pinned static base, adaptive remainder, MTP layer range and independent cache storage passed");
    }
    if (trace_path && profile) {
        const std::string ranked_path = std::string(trace_path) + ".profile";
        auto write_profile = [&](const char * body) {
            std::ofstream output(ranked_path);
            output << "llama-moe-profile-v1\n" << body;
        };
        test_env base("GGML_MOE_HYBRID_STATIC_PCT", "50");
        write_profile("0 1 1\n1 1 9\n0 0 5\n1 0 9\n");
        {
            llama_moe_cache ranked(*model, {&queued.backend}, {buft}, budget, ranked_path.c_str(), true);
            GGML_ASSERT(((int32_t *) ranked.get_slot_map(1, 1, 1)->data)[0] == 0);
            GGML_ASSERT(((int32_t *) ranked.get_slot_map(1, 1, 1)->data)[1] == -1);
            GGML_ASSERT(((int32_t *) ranked.get_slot_map(0, 1, 1)->data)[0] == -1);
        }
        for (const char * body : {"0 0 1\n0 0 2\n", ""}) {
            write_profile(body);
            bool rejected = false;
            try {
                llama_moe_cache invalid(*model, {&queued.backend}, {buft}, budget, ranked_path.c_str(), true);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            GGML_ASSERT(rejected);
        }
        puts("Oversized unsorted base profile, deterministic ranking, duplicate and missing-base checks passed");
    }
    if (trace_path && profile) {
        std::remove(trace_path);
        {
            test_env trace("GGML_MOE_HYBRID_TRACE", trace_path);
            test_env prefill("GGML_MOE_HYBRID_PREFILL", "1");
            llama_moe_cache traced(*model, {&queued.backend}, {buft}, budget, nullptr, true);
            const size_t before = queued.syncs;
            collect(traced, 0, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, true, std::vector<int8_t>(10, 0), true, false);
            collect(traced, 1, {1}, true, {0, 1}, false, true);
            collect(traced, 0, {0}, false);
            GGML_ASSERT(queued.syncs == before);
            queued.drain();
            llama_moe_cache frozen(*model, {&queued.backend}, {buft}, budget, profile, false);
            collect(frozen, 0, {1}, true);
            GGML_ASSERT(queued.queue.empty());
            llama_moe_cache trace_only(*model, {}, {}, 0);
            GGML_ASSERT(!trace_only.get_slot_map(0, 1, 1));
            collect(trace_only, 0, {0, 1}, true);
            GGML_ASSERT(queued.queue.empty());
        }
        std::ifstream input(trace_path);
        std::vector<nlohmann::json> records;
        std::string line;
        while (std::getline(input, line)) {
            records.push_back(nlohmann::json::parse(line));
        }
        GGML_ASSERT(records.size() == 4);
        GGML_ASSERT(records[0]["context"] == records[1]["context"] && records[1]["context"] != records[2]["context"]);
        GGML_ASSERT(records[0]["n_tokens"] == 10 && records[0]["prefill"] == true && records[0]["prefill_end"] == false);
        GGML_ASSERT(records[0]["layers"][0]["cache"] == false && records[0]["layers"][0]["ids"][1][0] == 1);
        GGML_ASSERT(records[1]["prefill"] == true && records[1]["prefill_end"] == true);
        GGML_ASSERT(records[1]["positions"][1] == 101 && records[1]["seq_ids"][1][0] == 0);
        GGML_ASSERT(records[1]["layers"][0]["rows"] == nlohmann::json::array({1}));
        GGML_ASSERT(records[1]["layers"][0]["ids"][0][0] == 1);
        GGML_ASSERT(records[2]["prefill"] == false);
        GGML_ASSERT(records[3]["trace_only"] == true && records[3]["context"] != records[2]["context"]);
        puts("JSONL append, independent context IDs, cropped rows, positions and failed-graph exclusion passed");
    }

}

static void test_contexts(const char * path, const char * trace_path) {
    common_params base;
    base.moe_cache_size = 1024;
    const auto draft = common_base_params_to_speculative(base);
    GGML_ASSERT(draft.moe_cache_draft && draft.moe_cache_size == 0);
    GGML_ASSERT(common_context_params_to_llama(draft).moe_cache_draft);
    GGML_ASSERT(!common_context_params_to_llama(base).moe_cache_draft && base.moe_cache_size == 1024);

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(path, mp));
    GGML_ASSERT(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 64;
    cp.n_ubatch = 16;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.no_perf = true;
    llama_context_ptr baseline(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(baseline);
    auto batch = llama_batch_init(20, 0, 1);
    batch.n_tokens = 20;
    for (int i = 0; i < 20; ++i) {
        batch.token[i] = 1;
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == 19;
    }
    GGML_ASSERT(llama_decode(baseline.get(), batch) == 0);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const auto * logits = llama_get_logits_ith(baseline.get(), -1);
    std::vector<float> expected(logits, logits + nv);
    std::remove(trace_path);
    {
        test_env trace("GGML_MOE_HYBRID_TRACE", trace_path);
        llama_context_ptr traced(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(traced && llama_decode(traced.get(), batch) == 0);
        GGML_ASSERT(memcmp(expected.data(), llama_get_logits_ith(traced.get(), -1), nv*sizeof(float)) == 0);
        batch.n_tokens = 1;
        batch.pos[0] = 20;
        batch.logits[0] = 1;
        GGML_ASSERT(llama_decode(traced.get(), batch) == 0);
    }
    llama_batch_free(batch);
    std::ifstream trace(trace_path);
    std::vector<nlohmann::json> records;
    std::string line;
    while (std::getline(trace, line)) {
        records.push_back(nlohmann::json::parse(line));
    }
    GGML_ASSERT(records.size() == 3);
    GGML_ASSERT(records[0]["trace_only"] == true && records[0]["n_tokens"] == 16 && records[0]["prefill_end"] == false);
    GGML_ASSERT(records[1]["n_tokens"] == 4 && records[1]["prefill"] == true && records[1]["prefill_end"] == true);
    GGML_ASSERT(records[1]["positions"][3] == 19);
    GGML_ASSERT(records[2]["n_tokens"] == 1 && records[2]["prefill"] == false);

    const std::string mismatch = std::string(trace_path) + ".profile";
    {
        std::ofstream profile(mismatch);
        profile << "llama-moe-profile-v1\n999 0 1\n";
    }
    test_env hybrid("GGML_MOE_HYBRID", "1");
    test_env adaptive("GGML_MOE_HYBRID_ADAPTIVE", "1");
    test_env budget("GGML_MOE_HYBRID_MIB", "invalid-target-budget");
    test_env target_profile("GGML_MOE_HYBRID_PROFILE", mismatch.c_str());
    test_env draft_budget("GGML_MOE_HYBRID_DRAFT_MIB", "0");
    cp.moe_cache_draft = true;
    {
        llama_context_ptr disabled(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(disabled);
    }
    {
        test_env budget("GGML_MOE_HYBRID_DRAFT_MIB", "1");
        test_env profile("GGML_MOE_HYBRID_DRAFT_PROFILE", mismatch.c_str());
        llama_context_ptr mismatch_draft(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(mismatch_draft);
    }
    {
        test_env budget("GGML_MOE_HYBRID_DRAFT_MIB", "1");
        test_env profile("GGML_MOE_HYBRID_DRAFT_PROFILE", nullptr);
        // Cold draft initialization reaches allocation, ignoring the mismatched target profile and its invalid budget.
        llama_context_ptr cold(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(!cold); // This test has no GPU backend for the requested cache.
    }
    {
        test_env budget("GGML_MOE_HYBRID_MIB", "1");
        cp.moe_cache_draft = false;
        llama_context_ptr mismatch_target(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(mismatch_target);
    }
    puts("CPU trace-only logits are bitwise equal; logical prefill boundaries, draft parameter plumbing and profile guards passed");
}

static void replay(const char * path, int slots, size_t admit, size_t ubatch) {
    using token_key = std::pair<int, int>;
    std::map<token_key, std::map<int, std::vector<int>>> rows;
    const std::vector<std::string> topics = {"de", "en", "code", "agent"};
    int n_expert = 0, n_layer = 0;
    std::ifstream input(path);
    GGML_ASSERT(input.good());
    std::string line;
    while (std::getline(input, line)) {
        auto r = nlohmann::json::parse(line);
        const auto topic = std::find(topics.begin(), topics.end(), r.at("topic").get<std::string>());
        GGML_ASSERT(topic != topics.end());
        const int il = r.at("layer");
        const int token = r.at("token");
        GGML_ASSERT(il >= 0 && token >= 0);
        auto ids = r.at("ids").get<std::vector<int>>();
        for (int id : ids) {
            GGML_ASSERT(id >= 0);
            n_expert = std::max(n_expert, id + 1);
        }
        n_layer = std::max(n_layer, il + 1);
        rows[{topic - topics.begin(), token}][il] = std::move(ids);
    }
    GGML_ASSERT(!rows.empty() && slots > 0 && ubatch > 0);
    llama_moe_cache_lfu policy;
    policy.init((size_t) n_layer*n_expert, slots);
    size_t uploads = 0, n = 0;
    for (const auto & token : rows) {
        policy.begin_token();
        for (const auto & layer : token.second) {
            for (int id : layer.second) {
                policy.observe(layer.first*n_expert + id);
            }
        }
        if (++n % ubatch == 0 || n == rows.size()) {
            uploads += policy.plan(admit).size();
        }
    }
    printf("slots=%d admit=%zu ubatch=%zu tokens=%llu hits=%llu misses=%llu hit_rate=%.4f%% uploads=%zu uploads/token=%.4f\n",
        slots, admit, ubatch, (unsigned long long) policy.tokens, (unsigned long long) policy.hits, (unsigned long long) policy.misses,
        100.0*policy.hits/(policy.hits + policy.misses), uploads, (double) uploads/policy.tokens);
}

struct replay_context {
    struct prompt_stats {
        int seq;
        uint64_t tokens = 0;
        uint64_t hits = 0;
        uint64_t routes = 0;
    };

    llama_moe_cache_lfu policy;
    int n_expert = 0;
    int n_layer = 0;
    int group = -1;
    uint64_t uploads = 0;
    uint64_t bytes = 0;
    uint64_t initial_uploads = 0;
    uint64_t initial_bytes = 0;
    uint64_t decode_hits = 0;
    uint64_t decode_routes = 0;
    uint64_t first_hits = 0;
    uint64_t first_routes = 0;
    std::map<int, int> first_left;
    std::map<int, bool> in_prefill;
    std::map<int, size_t> prompt_of;
    std::vector<prompt_stats> prompts;
    std::map<int, size_t> expert_bytes;
};

static void replay_log(const char * path, int slots, size_t admit, size_t prefill, int static_pct, uint64_t decay, const char * profile, uint64_t context_id) {
    GGML_ASSERT(slots > 0 && static_pct >= 0 && static_pct <= 100 && decay > 0);
    std::ifstream input(path);
    GGML_ASSERT(input.good());
    std::vector<nlohmann::json> batches;
    std::map<uint64_t, replay_context> contexts;
    std::string line;
    while (std::getline(input, line)) {
        auto b = nlohmann::json::parse(line);
        GGML_ASSERT(b.at("version") == 1);
        const uint64_t id = b.at("context");
        if (context_id != UINT64_MAX && id != context_id) {
            continue;
        }
        auto & c = contexts[id];
        const int nt = b.at("n_tokens");
        GGML_ASSERT(nt > 0 && b.at("positions").size() == (size_t) nt && b.at("seq_ids").size() == (size_t) nt);
        for (const auto & l : b.at("layers")) {
            const int group = l.value("group", 0);
            if (group >= 0) {
                if (c.group >= 0 && group != c.group) {
                    throw std::runtime_error("replay-log currently supports one cache group per context");
                }
                c.group = group;
            }
            if (group < 0 && !b.value("trace_only", false)) {
                continue;
            }
            const int n_expert = l.value("n_expert", 0);
            GGML_ASSERT(n_expert >= 0 && n_expert <= 65536);
            c.n_expert = std::max(c.n_expert, n_expert);
            const int il = l.at("layer");
            GGML_ASSERT(il >= 0 && il < 4096);
            c.n_layer = std::max(c.n_layer, il + 1);
            c.expert_bytes[il] = l.at("expert_bytes").get<size_t>();
            GGML_ASSERT(l.at("rows").size() == l.at("ids").size());
            for (int row : l.at("rows")) {
                GGML_ASSERT(row >= 0 && row < nt);
            }
            for (const auto & ids : l.at("ids")) {
                for (int id : ids) {
                    GGML_ASSERT(id >= 0 && id < 65536);
                    c.n_expert = std::max(c.n_expert, id + 1);
                }
            }
        }
        batches.push_back(std::move(b));
    }
    GGML_ASSERT(!batches.empty());
    for (auto & entry : contexts) {
        auto & c = entry.second;
        GGML_ASSERT(c.n_layer > 0 && c.n_expert > 0);
        c.policy.init((size_t) c.n_layer*c.n_expert, slots, decay);
        if (static_pct) {
            GGML_ASSERT(profile);
            std::ifstream in(profile);
            std::getline(in, line);
            GGML_ASSERT(line == "llama-moe-profile-v1");
            std::vector<std::pair<double, int>> ranked;
            int il, expert;
            double count;
            while (in >> il >> expert >> count) {
                GGML_ASSERT(il >= 0 && il < c.n_layer && expert >= 0 && expert < c.n_expert && count >= 0);
                ranked.push_back({count, il*c.n_expert + expert});
            }
            GGML_ASSERT(in.eof());
            std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
                return a.first == b.first ? a.second < b.second : a.first > b.first;
            });
            const size_t base = (size_t) slots*static_pct/100;
            GGML_ASSERT(ranked.size() >= base);
            for (size_t slot = 0; slot < base; ++slot) {
                const auto & r = ranked[slot];
                c.policy.seed(r.second, slot, r.first, true);
                ++c.initial_uploads;
                c.initial_bytes += c.expert_bytes.at(r.second/c.n_expert);
            }
        }
    }
    for (const auto & b : batches) {
        auto & c = contexts.at(b.at("context").get<uint64_t>());
        auto & policy = c.policy;
        const int nt = b.at("n_tokens");
        const bool prompt = b.at("prefill");
        const bool prompt_end = b.at("prefill_end");
        std::vector<std::vector<std::pair<int, bool>>> routes(nt);
        for (const auto & l : b.at("layers")) {
            if (l.value("group", 0) < 0 && !b.value("trace_only", false)) {
                continue;
            }
            const int il = l.at("layer");
            // Trace-only logs have no banks; estimate supported GPU windows when replaying them.
            const bool use_cache = b.value("trace_only", false) ? l.at("ids").size() <= 8 : l.at("cache").get<bool>();
            for (size_t i = 0; i < l.at("rows").size(); ++i) {
                const int row = l.at("rows")[i];
                for (int id : l.at("ids")[i]) {
                    routes[row].push_back({il*c.n_expert + id, use_cache});
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            policy.begin_token();
            std::vector<size_t> first_prompts;
            for (int seq : b.at("seq_ids")[t]) {
                if (prompt) {
                    if (!c.in_prefill[seq] || b.at("positions")[t] == 0) {
                        c.prompt_of[seq] = c.prompts.size();
                        c.prompts.push_back({seq});
                        c.in_prefill[seq] = true;
                    }
                    c.first_left[seq] = 32;
                } else if (c.first_left[seq] > 0) {
                    --c.first_left[seq];
                    const size_t ip = c.prompt_of.at(seq);
                    first_prompts.push_back(ip);
                    ++c.prompts[ip].tokens;
                }
            }
            const bool first = !first_prompts.empty();
            for (const auto & r : routes[t]) {
                const bool hit = r.second && policy.slot_of[r.first] >= 0;
                policy.observe(r.first, r.second, prompt && prefill > 0);
                if (!prompt) {
                    c.decode_hits += hit;
                    ++c.decode_routes;
                    c.first_hits += first && hit;
                    c.first_routes += first;
                    for (size_t ip : first_prompts) {
                        c.prompts[ip].hits += hit;
                        ++c.prompts[ip].routes;
                    }
                }
            }
        }
        // Warm once at the prompt boundary; retain candidates from earlier prefill ubatches.
        const auto fills = prompt && prefill ? (prompt_end ? policy.plan(prefill, true) : policy.plan(0)) : policy.plan(admit);
        for (const auto & f : fills) {
            ++c.uploads;
            c.bytes += c.expert_bytes.at(f.key/c.n_expert) + sizeof(int32_t)*(f.evicted < 0 ? 1 : 2);
        }
        if (prompt_end) {
            policy.clear_prompt();
            for (auto & seq : c.in_prefill) {
                seq.second = false;
            }
        }
    }
    auto pct = [](uint64_t hits, uint64_t routes) { return routes ? 100.0*hits/routes : 0.0; };
    for (const auto & entry : contexts) {
        const auto & c = entry.second;
        const auto & p = c.policy;
        printf("context=%llu slots=%d admit=%zu prefill=%zu static_pct=%d decay=%llu tokens=%llu hits=%llu misses=%llu hit_rate=%.4f%% decode_hit_rate=%.4f%% first32_hit_rate=%.4f%% first32_routes=%llu uploads=%llu uploads/token=%.4f MiB/token=%.4f initial_uploads=%llu initial_MiB=%.4f\n",
            (unsigned long long) entry.first, slots, admit, prefill, static_pct, (unsigned long long) decay,
            (unsigned long long) p.tokens, (unsigned long long) p.hits, (unsigned long long) p.misses,
            pct(p.hits, p.hits + p.misses), pct(c.decode_hits, c.decode_routes), pct(c.first_hits, c.first_routes),
            (unsigned long long) c.first_routes, (unsigned long long) c.uploads, (double) c.uploads/p.tokens,
            (double) c.bytes/(1024*1024)/p.tokens, (unsigned long long) c.initial_uploads, (double) c.initial_bytes/(1024*1024));
        for (size_t ip = 0; ip < c.prompts.size(); ++ip) {
            const auto & prompt = c.prompts[ip];
            printf("prompt context=%llu index=%zu seq=%d first32_tokens=%llu first32_hits=%llu first32_routes=%llu first32_hit_rate=%.4f%%\n",
                (unsigned long long) entry.first, ip, prompt.seq, (unsigned long long) prompt.tokens,
                (unsigned long long) prompt.hits, (unsigned long long) prompt.routes, pct(prompt.hits, prompt.routes));
        }
    }
}

int main(int argc, char ** argv) {
    test_policy();
    test_config();
    if (argc >= 3 && std::string(argv[1]) == "--ordering") {
        GGML_ASSERT(argc <= 5);
        test_ordering(argv[2], argc >= 4 ? argv[3] : nullptr, argc == 5 ? argv[4] : nullptr);
    } else if (argc == 4 && std::string(argv[1]) == "--contexts") {
        test_contexts(argv[2], argv[3]);
    } else if (argc >= 8 && std::string(argv[1]) == "--replay-log") {
        GGML_ASSERT(argc <= 10);
        replay_log(argv[2], std::stoi(argv[3]), std::stoull(argv[4]), std::stoull(argv[5]), std::stoi(argv[6]), std::stoull(argv[7]), argc >= 9 && std::string(argv[8]) != "-" ? argv[8] : nullptr, argc == 10 ? std::stoull(argv[9]) : UINT64_MAX);
    } else if (argc > 1) {
        GGML_ASSERT(argc >= 2 && argc <= 5);
        replay(argv[1], argc > 2 ? std::stoi(argv[2]) : 1805, argc > 3 ? std::stoull(argv[3]) : 8, argc > 4 ? std::stoull(argv[4]) : 1);
    }
    return 0;
}
