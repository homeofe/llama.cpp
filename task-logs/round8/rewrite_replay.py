from pathlib import Path
p=Path('tests/test-moe-cache.cpp')
s=p.read_text()
a=s.index('struct replay_context {')
b=s.index('\nint main(',a)
s=s[:a]+r'''struct replay_context {
    struct group {
        llama_moe_cache_lfu policy;
        int n_expert = 0;
        int slots = 0;
        size_t bytes = 0;
        std::set<int> layers;
    };
    struct prompt_stats {
        int seq;
        uint64_t tokens = 0, hits = 0, routes = 0;
    };
    struct vote {
        int group, key, pos;
        uint64_t token;
    };
    struct span {
        int pos, n_tokens;
        std::vector<vote> votes;
    };
    std::map<int, group> groups;
    std::map<int, int> layer_group;
    std::map<int, span> spans;
    int n_layer = 0;
    size_t next_group = 0;
    uint64_t tokens = 0, hits = 0, misses = 0;
    uint64_t uploads = 0, bytes = 0, initial_uploads = 0, initial_bytes = 0;
    uint64_t decode_hits = 0, decode_routes = 0, decode_hit_bytes = 0, decode_bytes = 0;
    uint64_t first_hits = 0, first_routes = 0, first_hit_bytes = 0, first_bytes = 0;
    uint64_t rejected_routes = 0;
    std::map<int, int> first_left;
    std::map<int, bool> in_prefill;
    std::map<int, size_t> prompt_of;
    std::vector<prompt_stats> prompts;

    void plan(size_t limit, bool prompt, bool byte_value) {
        std::vector<llama_moe_cache_lfu *> policies;
        std::vector<size_t> sizes;
        for (auto & entry : groups) {
            policies.push_back(&entry.second.policy);
            sizes.push_back(entry.second.bytes);
        }
        const auto plans = llama_moe_cache_plan(policies, sizes, limit, prompt, next_group, byte_value);
        for (size_t ig = 0; ig < plans.size(); ++ig) {
            for (const auto & f : plans[ig]) {
                ++uploads;
                bytes += sizes[ig] + sizeof(int32_t)*(f.evicted < 0 ? 1 : 2);
            }
        }
        next_group = (next_group + 1) % groups.size();
    }
};

static void replay_log(const char * path, int slots, size_t admit, size_t prefill, int static_pct, uint64_t decay, const char * profile, uint64_t context_id) {
    GGML_ASSERT(slots >= 0 && static_pct >= 0 && static_pct <= 100 && decay > 0);
    const bool byte_value = llama_moe_cache_env_size("GGML_MOE_HYBRID_BYTE_VALUE", 0, 1);
    const size_t soft_pct = llama_moe_cache_env_size("GGML_MOE_HYBRID_LAYER_SOFT_PCT", 0, 1000);
    const bool accepted_only = llama_moe_cache_env_size("GGML_MOE_REPLAY_ACCEPTED_ONLY", 0, 1);
    const bool warm_start = llama_moe_cache_env_size("GGML_MOE_REPLAY_WARM", 0, 1);
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
        if (!b.contains("event")) {
            const int nt = b.at("n_tokens");
            GGML_ASSERT(nt > 0 && nt <= 65536 && b.at("positions").size() == (size_t) nt && b.at("seq_ids").size() == (size_t) nt);
            for (const auto & l : b.at("layers")) {
                const int ig = l.value("group", 0);
                if (ig < 0 && !b.value("trace_only", false)) {
                    continue;
                }
                auto & g = c.groups[std::max(0, ig)];
                const int ne = l.at("n_expert");
                const int il = l.at("layer");
                const size_t bytes = l.at("expert_bytes");
                GGML_ASSERT(ne > 0 && ne <= 65536 && il >= 0 && il < 4096 && bytes > 0);
                GGML_ASSERT(!g.n_expert || (g.n_expert == ne && g.bytes == bytes));
                g.n_expert = ne;
                g.bytes = bytes;
                g.slots = l.value("slots", 0);
                g.layers.insert(il);
                c.n_layer = std::max(c.n_layer, il + 1);
                GGML_ASSERT(!c.layer_group.count(il) || c.layer_group.at(il) == std::max(0, ig));
                c.layer_group[il] = std::max(0, ig);
                GGML_ASSERT(l.at("rows").size() == l.at("ids").size());
                for (int row : l.at("rows")) {
                    GGML_ASSERT(row >= 0 && row < nt);
                }
                for (const auto & ids : l.at("ids")) {
                    for (int expert : ids) {
                        GGML_ASSERT(expert >= 0 && expert < ne);
                    }
                }
            }
        }
        batches.push_back(std::move(b));
    }
    GGML_ASSERT(!batches.empty());
    for (auto & entry : contexts) {
        auto & c = entry.second;
        GGML_ASSERT(!c.groups.empty());
        if (slots > 0 && c.groups.size() != 1) {
            throw std::runtime_error("multi-group replay requires recorded slot counts (slots argument 0)");
        }
        for (auto & ge : c.groups) {
            auto & g = ge.second;
            if (slots > 0) {
                g.slots = slots;
            }
            GGML_ASSERT(g.slots > 0);
            g.policy.init((size_t) c.n_layer*g.n_expert, g.slots, decay);
            g.policy.keys_per_layer = g.n_expert;
            g.policy.layer_soft_limit = (double) g.slots/g.layers.size()*soft_pct/100.0;
            if (static_pct || warm_start) {
                GGML_ASSERT(profile);
                std::ifstream in(profile);
                std::getline(in, line);
                GGML_ASSERT(line == "llama-moe-profile-v1");
                std::vector<std::pair<double, int>> ranked;
                int il, expert;
                double count;
                while (in >> il >> expert >> count) {
                    GGML_ASSERT(il >= 0 && il < c.n_layer && expert >= 0 && expert < g.n_expert && std::isfinite(count) && count >= 0);
                    if (g.layers.count(il)) {
                        ranked.push_back({count, il*g.n_expert + expert});
                    }
                }
                GGML_ASSERT(in.eof());
                if (static_pct) {
                    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
                        return a.first == b.first ? a.second < b.second : a.first > b.first;
                    });
                }
                const size_t base = static_pct ? (size_t) g.slots*static_pct/100 : ranked.size();
                GGML_ASSERT(ranked.size() >= base && base <= (size_t) g.slots);
                for (size_t slot = 0; slot < base; ++slot) {
                    g.policy.seed(ranked[slot].second, slot, ranked[slot].first, static_pct > 0);
                    ++c.initial_uploads;
                    c.initial_bytes += g.bytes;
                }
            }
        }
    }
    for (const auto & b : batches) {
        auto & c = contexts.at(b.at("context").get<uint64_t>());
        if (b.contains("event")) {
            if (!accepted_only) {
                continue;
            }
            const int seq = b.at("seq"), pos = b.at("pos"), nt = b.at("n_tokens");
            GGML_ASSERT(seq >= 0 && pos >= 0 && nt > 0 && nt <= 65536);
            if (b.at("event") == "spec_begin") {
                if (c.spans.count(seq)) {
                    throw std::runtime_error("unresolved speculative span");
                }
                c.spans[seq] = {pos, nt, {}};
            } else if (b.at("event") == "spec_accept") {
                auto it = c.spans.find(seq);
                if (it == c.spans.end()) {
                    continue; // A draft may have no evaluated speculative rows.
                }
                const int accepted = b.at("n_accepted");
                GGML_ASSERT(it->second.pos == pos && accepted >= 0 && accepted <= nt && nt <= it->second.n_tokens);
                for (const auto & v : it->second.votes) {
                    auto & p = c.groups.at(v.group).policy;
                    if ((int64_t) v.pos < (int64_t) pos + accepted) {
                        const uint64_t shifts = (p.tokens - 1)/decay - (v.token - 1)/decay;
                        p.learn(v.key, shifts > 1074 ? 0 : std::ldexp(1.0, -(int) shifts));
                    } else {
                        ++c.rejected_routes;
                    }
                }
                c.spans.erase(it);
                c.plan(admit, false, byte_value);
            } else {
                throw std::runtime_error("unknown trace event");
            }
            continue;
        }
        const int nt = b.at("n_tokens");
        const bool prompt = b.at("prefill"), prompt_end = b.at("prefill_end");
        struct route { int group, key; bool cache; };
        std::vector<std::vector<route>> routes(nt);
        for (const auto & l : b.at("layers")) {
            if (l.value("group", 0) < 0 && !b.value("trace_only", false)) {
                continue;
            }
            const int il = l.at("layer"), ig = c.layer_group.at(il);
            const auto & g = c.groups.at(ig);
            const bool use_cache = b.value("trace_only", false) ? l.at("ids").size() <= 8 : l.at("cache").get<bool>();
            for (size_t i = 0; i < l.at("rows").size(); ++i) {
                const int row = l.at("rows")[i];
                for (int expert : l.at("ids")[i]) {
                    routes[row].push_back({ig, il*g.n_expert + expert, use_cache});
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            ++c.tokens;
            for (auto & g : c.groups) {
                g.second.policy.begin_token();
            }
            std::vector<size_t> first_prompts;
            replay_context::span * deferred = nullptr;
            const int pos = b.at("positions")[t];
            for (int seq : b.at("seq_ids")[t]) {
                if (accepted_only && !prompt && c.spans.count(seq)) {
                    auto & span = c.spans.at(seq);
                    if (pos >= span.pos && (int64_t) pos < (int64_t) span.pos + span.n_tokens) {
                        GGML_ASSERT(b.at("seq_ids")[t].size() == 1);
                        deferred = &span;
                    }
                }
                if (prompt) {
                    if (!c.in_prefill[seq] || pos == 0) {
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
                auto & g = c.groups.at(r.group);
                auto & policy = g.policy;
                const bool hit = r.cache && policy.slot_of[r.key] >= 0;
                policy.observe(r.key, r.cache, prompt && prefill > 0, !deferred);
                c.hits += hit;
                c.misses += !hit;
                if (deferred) {
                    deferred->votes.push_back({r.group, r.key, pos, policy.tokens});
                }
                if (!prompt) {
                    c.decode_hits += hit;
                    ++c.decode_routes;
                    c.decode_hit_bytes += hit*g.bytes;
                    c.decode_bytes += g.bytes;
                    c.first_hits += first && hit;
                    c.first_routes += first;
                    c.first_hit_bytes += (first && hit)*g.bytes;
                    c.first_bytes += first*g.bytes;
                    for (size_t ip : first_prompts) {
                        c.prompts[ip].hits += hit;
                        ++c.prompts[ip].routes;
                    }
                }
            }
        }
        c.plan(prompt && prefill ? (prompt_end ? prefill : 0) : admit, prompt && prefill && prompt_end, byte_value);
        if (prompt_end) {
            for (auto & g : c.groups) {
                g.second.policy.clear_prompt();
            }
            for (auto & seq : c.in_prefill) {
                seq.second = false;
            }
        }
    }
    auto pct = [](uint64_t hits, uint64_t routes) { return routes ? 100.0*hits/routes : 0.0; };
    for (const auto & entry : contexts) {
        const auto & c = entry.second;
        int total_slots = 0;
        for (const auto & g : c.groups) {
            total_slots += g.second.slots;
        }
        printf("context=%llu slots=%d admit=%zu prefill=%zu static_pct=%d decay=%llu tokens=%llu hits=%llu misses=%llu hit_rate=%.4f%% decode_hit_rate=%.4f%% first32_hit_rate=%.4f%% first32_routes=%llu uploads=%llu uploads/token=%.4f MiB/token=%.4f initial_uploads=%llu initial_MiB=%.4f decode_byte_hit_rate=%.4f%% first32_byte_hit_rate=%.4f%% groups=%zu rejected_routes=%llu unresolved_spans=%zu\n",
            (unsigned long long) entry.first, total_slots, admit, prefill, static_pct, (unsigned long long) decay,
            (unsigned long long) c.tokens, (unsigned long long) c.hits, (unsigned long long) c.misses,
            pct(c.hits, c.hits + c.misses), pct(c.decode_hits, c.decode_routes), pct(c.first_hits, c.first_routes),
            (unsigned long long) c.first_routes, (unsigned long long) c.uploads, (double) c.uploads/c.tokens,
            (double) c.bytes/(1024*1024)/c.tokens, (unsigned long long) c.initial_uploads, (double) c.initial_bytes/(1024*1024),
            pct(c.decode_hit_bytes, c.decode_bytes), pct(c.first_hit_bytes, c.first_bytes), c.groups.size(), (unsigned long long) c.rejected_routes, c.spans.size());
        for (size_t ip = 0; ip < c.prompts.size(); ++ip) {
            const auto & p = c.prompts[ip];
            printf("prompt context=%llu index=%zu seq=%d first32_tokens=%llu first32_hits=%llu first32_routes=%llu first32_hit_rate=%.4f%%\n",
                (unsigned long long) entry.first, ip, p.seq, (unsigned long long) p.tokens,
                (unsigned long long) p.hits, (unsigned long long) p.routes, pct(p.hits, p.routes));
        }
    }
}
''' +s[b:]
s=s.replace('#include <algorithm>','#include <algorithm>\n#include <cmath>\n#include <set>')
p.write_text(s)
