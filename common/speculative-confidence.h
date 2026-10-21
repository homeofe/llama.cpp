#pragma once

#include "llama.h"
#include "ggml-backend.h"
#include <cmath>

// Measurement only: greedy MTP confidence over the full vocabulary, before truncation.
inline void common_mtp_confidence_apply(llama_sampler *, llama_token_data_array * cur) {
    GGML_ASSERT(cur->size > 0);
    size_t best = 0;
    for (size_t i = 1; i < cur->size; ++i) {
        if (cur->data[i].logit > cur->data[best].logit) {
            best = i;
        }
    }
    double sum = 0;
    for (size_t i = 0; i < cur->size; ++i) {
        sum += std::exp(double(cur->data[i].logit) - cur->data[best].logit);
    }
    cur->data[0] = cur->data[best];
    cur->data[0].p = float(1.0/sum);
    cur->size = 1;
    cur->selected = 0;
    cur->sorted = true;
}

inline void common_mtp_confidence_graph(llama_sampler *, ggml_context * ctx, ggml_cgraph *, llama_sampler_data * data) {
    auto * logits = ggml_reshape_1d(ctx, data->logits, ggml_nelements(data->logits));
    auto * best = ggml_argmax(ctx, logits);
    auto * probs = ggml_soft_max(ctx, logits);
    auto * ids = best;
    if (data->candidates) {
        ids = ggml_get_rows(ctx, ggml_reshape_2d(ctx, data->candidates, 1, ggml_nelements(logits)), best);
    }
    data->logits = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, ggml_nelements(logits)), best);
    data->probs = ggml_get_rows(ctx, ggml_reshape_2d(ctx, probs, 1, ggml_nelements(logits)), best);
    data->candidates = ids;
    data->sampled = ids;
}

inline bool common_mtp_confidence_backend(llama_sampler * sampler, ggml_backend_buffer_type_t buft, uint32_t) {
    auto * dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return true;
    }
    ggml_init_params ip = {32*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true};
    auto * ctx = ggml_init(ip);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    llama_sampler_data data = {ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 248320), nullptr, nullptr, nullptr};
    common_mtp_confidence_graph(sampler, ctx, graph, &data);
    ggml_build_forward_expand(graph, data.logits);
    ggml_build_forward_expand(graph, data.probs);
    ggml_build_forward_expand(graph, data.sampled);
    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        supported = supported && ggml_backend_dev_supports_op(dev, ggml_graph_node(graph, i));
    }
    ggml_free(ctx);
    return supported;
}

inline llama_sampler * common_mtp_confidence_init() {
    static llama_sampler_i iface = {
        /* .name              = */ [](const llama_sampler *) { return "mtp-full-confidence"; },
        /* .accept            = */ nullptr,
        /* .apply             = */ common_mtp_confidence_apply,
        /* .reset             = */ nullptr,
        /* .clone             = */ nullptr,
        /* .free              = */ nullptr,
        /* .backend_init      = */ common_mtp_confidence_backend,
        /* .backend_accept    = */ nullptr,
        /* .backend_apply     = */ common_mtp_confidence_graph,
        /* .backend_set_input = */ nullptr,
        /* .backend_reset     = */ nullptr,
        /* .copy_state        = */ nullptr,
    };
    return llama_sampler_init(&iface, nullptr);
}
