#include "arg.h"
#include "common.h"
#include "llama.h"
#include "speculative.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static common_params parse(std::vector<std::string> args) {
    args.insert(args.begin(), "performance-check");
    std::vector<char *> argv;
    for (auto & s : args) { argv.push_back(s.data()); }
    common_params p;
    require(common_params_parse(argv.size(), argv.data(), p, LLAMA_EXAMPLE_SERVER), "parser rejected arguments");
    return p;
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 2, "case required");
        const std::string which = argv[1];
        if (which == "cache-zero") {
            for (auto a : std::vector<std::vector<std::string>> {
                    {"--moe", "cache=0", "-at", "on"}, {"-at", "on", "--moe-expert-cache", "0"},
                    {"--moe-expert-cache", "0", "-at", "on"}, {"--moe", "cache=0,autotune=1"}}) {
                require(!common_autotune_on(parse(a)), "cache=0 must disable tuning regardless of flag order");
            }
        } else if (which == "env-cli") {
            setenv("LLAMA_AUTOTUNE", "0", 1);
            require(!common_autotune_on(parse({})), "legacy environment off ignored");
            require(common_autotune_on(parse({"-at", "on"})), "explicit -at on must override legacy environment");
            require(common_autotune_on(parse({"--moe", "autotune=1"})), "explicit MoE tune flag must override legacy environment");
            setenv("LLAMA_ARG_AUTOTUNE", "off", 1);
            require(common_autotune_on(parse({"--autotune", "on"})), "CLI must override LLAMA_ARG_AUTOTUNE");
        } else if (which == "flags") {
            auto p = parse({"-t", "7", "-tb", "9", "--load-mode", "pin", "--moe", "cache=-1,prefetch-slots=3,inserts=4,window=40,predict=5,train=6,GATE=3,MARGIN_MB=501", "-ub", "256", "-fa", "1"});
            require(p.cpuparams.n_threads == 7 && !p.cpuparams.auto_threads && p.cpuparams_batch.n_threads == 9 && p.threads_batch_set, "explicit thread counts changed");
            require(p.load_mode == LLAMA_LOAD_MODE_NONE && p.n_moe_cache_slots == -1 && p.prefetch_experts_slots == 3 && p.n_moe_cache_inserts == 4 && p.n_moe_cache_window == 40 && p.n_moe_predict == 5 && p.n_moe_predict_train == 6, "MoE flag mapping incorrect");
            require(p.ubatch_user && p.n_ubatch == 256 && p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED, "batch/FA flag mapping incorrect");
            require(common_autotune_on(p), "enabled cache tuning control failed");
            setenv("LLAMA_ARG_MOE", "cache=0", 1);
            require(!common_autotune_on(parse({})), "environment cache=0 ignored");
        } else if (which == "stock-spec") {
            auto p = parse({"--moe", "cache=0", "-md", "draft.gguf", "--spec-type", "draft-mtp"});
            const auto types = p.speculative.types;
            const int depth = p.speculative.draft.n_max;
            common_spec_auto(p);
            require(p.speculative.types == types && p.speculative.draft.n_max == depth && p.speculative.has_dft(), "stock path changed explicit speculative configuration");
        } else if (which == "margin") {
            require(argc == 3, "model required");
            auto p = parse({"-m", argv[2], "-ngl", "0", "-c", "512", "-np", "1", "-t", "1", "-tb", "1", "--no-warmup", "-fit", "off", "--moe", "cache=-1,MARGIN_MB=501"});
            auto init = common_init_from_params(p);
            require(init->context() != nullptr, "margin fixture failed to load");
            std::cout << "options after init: " << p.moe_opts << '\n';
            require(p.moe_opts.find("margin_mb=") == std::string::npos, "automatic margin overwrote explicit uppercase MARGIN_MB");
        } else if (which == "dense-state" || which == "manual-state") {
            require(argc == 4, "model and private state file required");
            setenv("LLAMA_MOE_STATE", argv[3], 1);
            auto args = std::vector<std::string>{"-m", argv[2], "-ngl", "0", "-c", "512", "-np", "1", "-lv", "2", "--no-warmup", "-fit", "off"};
            if (which == "manual-state") { args.insert(args.end(), {"-t", "7", "-tb", "9"}); }
            auto p = parse(args);
            const int nt = p.cpuparams.n_threads, nb = p.cpuparams_batch.n_threads;
            llama_state_set_model("performance-check");
            require(llama_state_set("performance-check", "threads", (std::to_string(nt) + " 2").c_str()), "state fixture write failed");
            require(llama_state_set("performance-check", "threads_batch", (std::to_string(nb) + " 2").c_str()), "batch state fixture write failed");
            auto init = common_init_from_params(p);
            require(init->context() != nullptr, "real model failed to load");
            llama_tuning_info info{};
            llama_get_tuning_info(init->context(), &info);
            std::cout << "threads before=" << nt << '/' << nb << " after=" << info.n_threads << '/' << info.n_threads_batch << '\n';
            require(!info.moe_active, "dense CPU model enabled expert cache");
            require(info.n_threads == nt && info.n_threads_batch == nb, "stock path loaded fork thread tuning state");
        } else {
            throw std::runtime_error("unknown case");
        }
        std::cout << "PASS " << which << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
