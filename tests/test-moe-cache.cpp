#include "../src/llama-moe-cache.h"
#include "../src/llama-model.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "llama-cpp.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
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

static void test_ordering(const char * path, const char * profile) {
    queued_backend queued;
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

    auto collect = [&](llama_moe_cache & cache, int il, const std::vector<int32_t> & ids, bool success, const std::vector<int8_t> & output = {}) {
        cache.begin_graph(output.empty() ? ids.size() : output.size(), output.empty() ? nullptr : output.data());
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, false}));
        // Use a strided view, as top-k IDs need not be contiguous.
        auto src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, ids.size());
        for (size_t t = 0; t < ids.size(); ++t) {
            ((int32_t *) src->data)[2*t] = ids[t];
            ((int32_t *) src->data)[2*t + 1] = -999;
        }
        auto view = ggml_view_2d(ctx.get(), src, 1, ids.size(), src->nb[1], 0);
        auto observed = cache.build_observe(ctx.get(), view, il, true);
        auto graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, observed);
        GGML_ASSERT(ggml_backend_graph_compute(queued.cpu.get(), graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(memcmp(observed->data, ids.data(), ids.size()*sizeof(int32_t)) == 0);
        cache.end_graph(success);
    };
    auto expect_resident = [&](int il) {
        queued.queue.push_back([&, il]() {
            const auto * map = (const int32_t *) cache.get_slot_map(il, 1, 1)->data;
            const auto & l = model->layers[il];
            for (int e = 0; e < 2; ++e) {
                GGML_ASSERT(map[e] >= 0 && map[e] < 2);
                for (const auto * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps}) {
                    const auto * bank = cache.get_experts(w);
                    GGML_ASSERT(memcmp((const char *) bank->data + map[e]*bank->nb[2], (const char *) w->data + e*w->nb[2], w->nb[2]) == 0);
                }
            }
        });
    };

    collect(cache, 0, {0, 1}, true);
    GGML_ASSERT(!queued.queue.empty());
    expect_resident(0); // A submitted graph must see the old weights and maps before the eviction writes.
    collect(cache, 1, {0, 1, 0, 1}, true);
    expect_resident(1);
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

int main(int argc, char ** argv) {
    test_policy();
    if (argc >= 3 && std::string(argv[1]) == "--ordering") {
        GGML_ASSERT(argc <= 4);
        test_ordering(argv[2], argc == 4 ? argv[3] : nullptr);
    } else if (argc > 1) {
        GGML_ASSERT(argc >= 2 && argc <= 5);
        replay(argv[1], argc > 2 ? std::stoi(argv[2]) : 1805, argc > 3 ? std::stoull(argv[3]) : 8, argc > 4 ? std::stoull(argv[4]) : 1);
    }
    return 0;
}
