#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        return 1;
    }
    ggml_backend_load_all_from_path(argv[2]);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 2;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 32;
    cp.n_ubatch = 4;
    cp.n_threads = cp.n_threads_batch = 4;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 3;
    }
    FILE * out = fopen(argv[3], "wb");
    if (!out) {
        return 4;
    }
    auto batch = llama_batch_init(20, 0, 1);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int window = 0; window < 4; ++window) {
        const int n = window == 0 ? 20 : 4;
        batch.n_tokens = n;
        for (int t = 0; t < n; ++t) {
            batch.token[t] = 1 + (t + window) % 8;
            batch.pos[t] = window == 0 ? t : 20 + (window-1)*4+t;
            batch.n_seq_id[t] = 1;
            batch.seq_id[t][0] = 0;
            batch.logits[t] = true;
        }
        if (llama_decode(ctx, batch)) {
            return 5;
        }
        for (int t = 0; t < n; ++t) {
            if (fwrite(llama_get_logits_ith(ctx,t), sizeof(float), nv, out) != (size_t) nv) {
                return 6;
            }
        }
    }
    fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
}
