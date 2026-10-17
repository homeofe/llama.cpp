#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"

#include "json.hpp"

#include <clocale>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::ordered_json;

struct probe_trace {
    std::ofstream out;
    std::ofstream hits;
    std::string topic;
    size_t token = 0;
    size_t calls = 0;
    bool record = true;
    bool cpu = false;
};

static bool trace_ids(ggml_tensor * t, bool ask, void * data) {
    auto & trace = *static_cast<probe_trace *>(data);
    const bool is_ids = !trace.cpu && std::strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_mask = !trace.cpu && std::strncmp(t->name, "ffn_moe_hit_mask-", 17) == 0;
    const bool is_cpu = trace.cpu && std::strncmp(t->name, "ffn_moe_down_cpu-", 17) == 0;
    if (!trace.record || (!is_ids && !is_mask && !is_cpu)) {
        return false;
    }
    if (ask) {
        return true;
    }
    const int layer = std::stoi(t->name + (is_ids ? 13 : 17));
    if (is_mask || is_cpu) {
        if (is_cpu) {
            GGML_ASSERT(t->op == GGML_OP_MUL_MAT_ID_MASKED);
        }
        const ggml_tensor * mask = is_cpu ? t->src[3] : t;
        GGML_ASSERT(mask->type == GGML_TYPE_F32 && ggml_is_contiguous(mask));
        std::vector<float> values(ggml_nelements(mask));
        ggml_backend_tensor_get(mask, values.data(), 0, values.size()*sizeof(float));
        for (int64_t token = 0; token < mask->ne[1]; ++token) {
            size_t count = 0;
            for (int64_t id = 0; id < mask->ne[0]; ++id) {
                const float v = values[token*mask->ne[0] + id];
                GGML_ASSERT(v == 0.0f || v == 1.0f);
                count += is_cpu ? v == 0.0f : v != 0.0f;
            }
            trace.hits << json({{"topic", trace.topic}, {"token", trace.token + token},
                               {"layer", layer}, {"hits", count}, {"selected", mask->ne[0]}}).dump() << '\n';
        }
        if (is_mask) {
            return true;
        }
        t = t->src[2];
    }
    GGML_ASSERT(t->type == GGML_TYPE_I32 && t->ne[2] == 1 && t->ne[3] == 1);
    GGML_ASSERT(t->nb[0] == sizeof(int32_t));
    std::vector<int32_t> ids(t->ne[0]);
    for (int64_t token = 0; token < t->ne[1]; ++token) {
        ggml_backend_tensor_get(t, ids.data(), token*t->nb[1], ids.size()*sizeof(int32_t));
        trace.out << json({{"topic", trace.topic}, {"token", trace.token + token},
                          {"layer", layer}, {"ids", ids}}).dump() << '\n';
        trace.calls++;
    }
    return true;
}

static json device_memory() {
    json result = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        result.push_back({{"name", ggml_backend_dev_name(dev)}, {"free", free}, {"total", total},
            {"alignment", ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev))}});
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        return 1;
    }
    std::setlocale(LC_NUMERIC, "C");
    const std::string prefix = argv[1];
    common_init();
    common_params params;
    if (!common_params_parse(argc - 1, argv + 1, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.n_predict <= 0 || params.n_predict > params.n_ctx) {
        throw std::runtime_error("set 0 < n_predict <= n_ctx");
    }
    probe_trace trace;
    const char * enabled = std::getenv("GGML_MOE_PROBE_TRACE");
    trace.cpu = enabled && std::strcmp(enabled, "2") == 0;
    if (trace.cpu || (enabled && std::strcmp(enabled, "1") == 0)) {
        trace.out.open(prefix + ".routes.jsonl");
        trace.hits.open(prefix + ".hits.jsonl");
        if (!trace.out || !trace.hits) {
            return 1;
        }
        params.cb_eval = trace_ids;
        params.cb_eval_user_data = &trace;
    }
    params.warmup = false;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    auto * model = init->model();
    if (!ctx || !model) {
        return 1;
    }
    json report = {{"trace", params.cb_eval != nullptr}, {"memory_initial", device_memory()},
                   {"layers", json::array()}, {"topics", json::array()}};
    for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
        const auto & layer = model->layers[il];
        json tensors = json::array();
        for (auto * t : {layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
            if (t) {
                tensors.push_back({{"name", t->name}, {"type", ggml_type_name(t->type)},
                    {"ne", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}, {"expert_bytes", t->nb[2]},
                    {"host", t->buffer && ggml_backend_buffer_is_host(t->buffer)}});
            }
        }
        if (!tensors.empty()) {
            report["layers"].push_back({{"layer", il}, {"tensors", tensors}});
        }
    }
    std::ofstream logits;
    const char * dump_logits = std::getenv("GGML_MOE_PROBE_LOGITS");
    if (dump_logits && std::strcmp(dump_logits, "1") == 0) {
        logits.open(prefix + ".logits.bin", std::ios::binary);
        if (!logits) {
            return 1;
        }
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    report["n_vocab"] = n_vocab;
    const char * prefix_value = std::getenv("GGML_MOE_PROBE_PREFIX");
    const int prefix_tokens = prefix_value ? std::stoi(prefix_value) : 0;
    if (prefix_tokens < 0 || prefix_tokens > int(params.n_ctx) - params.n_predict) {
        throw std::runtime_error("invalid GGML_MOE_PROBE_PREFIX");
    }
    report["prefix_tokens"] = prefix_tokens;
    const char * window_value = std::getenv("GGML_MOE_PROBE_WINDOW");
    const int window = window_value ? std::stoi(window_value) : 1;
    if (window < 1 || window > 8 || window > params.n_ubatch) {
        throw std::runtime_error("probe window must be 1..8 and <= n_ubatch");
    }
    report["window_tokens"] = window;
    for (const char * topic : {"de", "en", "code", "agent"}) {
        const std::string path = std::string("/opt/llm-gateway/data/moe/texts/") + topic + ".txt";
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open " + path);
        }
        const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        auto tokens = common_tokenize(ctx, text, true, true);
        if (tokens.size() < size_t(params.n_predict + prefix_tokens)) {
            throw std::runtime_error("not enough tokens in " + path);
        }
        tokens.resize(params.n_predict + prefix_tokens);
        llama_memory_clear(llama_get_memory(ctx), true);
        trace.topic = topic;
        trace.record = false;
        for (int i = 0; i < prefix_tokens;) {
            const int count = std::min(prefix_tokens - i, std::min(params.n_batch, params.n_ubatch));
            auto batch = common_batch_get_one(ctx, &tokens[i], count);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                return 1;
            }
            i += count;
        }
        llama_synchronize(ctx);
        trace.record = true;
        std::vector<llama_token> prefix_ids(tokens.begin(), tokens.begin() + prefix_tokens);
        tokens.erase(tokens.begin(), tokens.begin() + prefix_tokens);
        json times = json::array();
        for (size_t i = 0; i < tokens.size(); i += window) {
            trace.token = i;
            const int count = std::min<size_t>(window, tokens.size() - i);
            auto batch = common_batch_get_one(ctx, &tokens[i], count);
            for (int token = 0; token < count; ++token) {
                batch.set_output(token, true);
            }
            const int64_t start = ggml_time_us();
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                return 1;
            }
            llama_synchronize(ctx);
            times.push_back(ggml_time_us() - start);
            if (logits.is_open()) {
                for (int token = 0; token < count; ++token) {
                    logits.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx, token)), n_vocab*sizeof(float));
                }
                if (!logits) {
                    return 1;
                }
            }
            if (i < 32 && i + count >= 32) {
                report["memory_steady"] = device_memory();
            }
        }
        report["topics"].push_back({{"topic", topic}, {"prefix_ids", prefix_ids}, {"tokens", tokens}, {"times_us", times}});
        std::ofstream out(prefix + ".report.json");
        out << report.dump(2) << '\n';
        if (!out) {
            return 1;
        }
        if (params.cb_eval) {
            trace.out.flush();
            trace.hits.flush();
            if (!trace.out || !trace.hits) {
                return 1;
            }
        }
        std::cout << "TOPIC " << topic << " tokens " << tokens.size() << " callbacks " << trace.calls << std::endl;
    }
    if (params.cb_eval && trace.calls == 0) {
        return 1;
    }
    init.reset();
    llama_backend_free();
}
