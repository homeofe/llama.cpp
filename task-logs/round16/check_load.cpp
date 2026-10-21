#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mtp = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 1;
    llama_model_free(model);
    llama_backend_free();
    std::puts("MODEL_LOAD_OK");
    return 0;
}
