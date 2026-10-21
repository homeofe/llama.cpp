#include "llama-moecache.cpp"
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    user_knobs().clear();
    knobs() = knobs_t{};
    moe_cache first, second;
    const bool off_first = std::string(argv[1]) == "off-on";
    g_autotune_off = off_first;
    self_tune(&first);
    g_autotune_off = false;
    if (!off_first) {
        llama_moe_set_options("MARGIN=123,autotune=1");
    }
    self_tune(&second);
    const bool ok = off_first ? second.lv_k >= 0 : knobs().margin == 123;
    std::printf("%s: first=%d second=%d margin=%g %s\n", argv[1], first.lv_k, second.lv_k, knobs().margin, ok ? "PASS" : "FAIL");
    return !ok;
}
