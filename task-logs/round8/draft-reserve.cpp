#include "llama-context.h"
#include "llama-model.h"
#include "llama-cpp.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.no_alloc = true;
    mp.load_mtp = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = false;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    GGML_ASSERT(model);
    for (uint32_t ub : {512, 256, 64, 32, 8}) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 8192;
        cp.n_batch = 2048;
        cp.n_ubatch = ub;
        cp.n_seq_max = 1;
        cp.n_outputs_max = 1;
        cp.n_outputs_max_per_seq = 1;
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(ctx);
        auto mctx = ctx->get_memory()->init_full();
        auto * graph = ctx->graph_reserve(ub, 1, 1, mctx.get(), true);
        GGML_ASSERT(graph);
        std::vector<ggml_tensor *> nodes;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!node->view_src) {
                nodes.push_back(node);
            }
        }
        std::sort(nodes.begin(), nodes.end(), [](auto * a, auto * b) { return ggml_nbytes(a) > ggml_nbytes(b); });
        printf("ubatch=%u CPU graph, no device allocation; largest owned intermediate tensors:\n", ub);
        for (size_t i = 0; i < std::min<size_t>(12, nodes.size()); ++i) {
            const auto * n = nodes[i];
            printf("%9.3f MiB %-30s %-16s [%lld,%lld,%lld,%lld]\n", ggml_nbytes(n)/1048576.0, n->name, ggml_op_name(n->op), (long long)n->ne[0], (long long)n->ne[1], (long long)n->ne[2], (long long)n->ne[3]);
        }
    }
}
