#include "arg.h"
#include "common.h"
#include "server-context.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string model = "/opt/llm-gateway/models/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf";
    std::vector<std::string> a = {"audit", "-m", model, "-md", model, "--spec-type", "draft-simple", "-ngl", "0", "-ngld", "0", "-np", "1", "-c", "512", "-b", "128", "-ub", "64", "-t", "2", "-tb", "2", "-fit", "off", "--no-warmup"};
    std::vector<char *> av; for (auto & s : a) { av.push_back(s.data()); }
    common_params p;
    if (!common_params_parse(av.size(), av.data(), p, LLAMA_EXAMPLE_SERVER)) { return 2; }
    const int requested = p.speculative.draft.n_max;
    llama_backend_init();
    if (argc > 1) { common_spec_auto(p); } // mutation: restore the premature call removed from the server
    server_context ctx;
    if (!ctx.load_model(p)) { return 3; }
    std::printf("default draft limit before=%d after=%d\n", requested, p.speculative.draft.n_max);
    return p.speculative.draft.n_max != requested;
}
