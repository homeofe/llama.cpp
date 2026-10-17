#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

void llama_moe_cache_lfu::init(size_t n_keys, int32_t n_slots) {
    counts.assign(n_keys, 0.0);
    slot_of.assign(n_keys, -1);
    key_of.assign(n_slots, -1);
    pending.assign(n_keys, false);
    candidates.clear();
    tokens = hits = misses = 0;
}

void llama_moe_cache_lfu::seed(int32_t key, int32_t slot, double count) {
    GGML_ASSERT(key >= 0 && (size_t) key < counts.size());
    GGML_ASSERT(slot >= 0 && (size_t) slot < key_of.size());
    GGML_ASSERT(slot_of[key] == -1 && key_of[slot] == -1);
    counts[key] = count;
    slot_of[key] = slot;
    key_of[slot] = key;
}

void llama_moe_cache_lfu::begin_token() {
    if (tokens && tokens % 512 == 0) {
        for (double & count : counts) {
            count *= 0.5;
        }
    }
    ++tokens;
}

void llama_moe_cache_lfu::observe(int32_t key, bool use_cache) {
    GGML_ASSERT(key >= 0 && (size_t) key < counts.size());
    ++counts[key];
    if (use_cache && slot_of[key] >= 0) {
        ++hits;
    } else {
        ++misses;
    }
    if (slot_of[key] < 0 && !pending[key]) {
        candidates.push_back(key);
        pending[key] = true;
    }
}

std::vector<llama_moe_cache_lfu::fill> llama_moe_cache_lfu::plan(size_t limit) {
    std::vector<int32_t> cold;
    if (limit == 0 || candidates.empty()) {
        for (int32_t key : candidates) {
            pending[key] = false;
        }
        candidates.clear();
        return {};
    }
    for (size_t slot = 0; slot < key_of.size(); ++slot) {
        cold.push_back(slot);
    }
    std::sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) {
        const int32_t ka = key_of[a];
        const int32_t kb = key_of[b];
        if (ka < 0 || kb < 0) {
            return ka == kb ? a < b : ka < kb;
        }
        return counts[ka] == counts[kb] ? ka < kb : counts[ka] < counts[kb];
    });
    std::sort(candidates.begin(), candidates.end(), [&](int32_t a, int32_t b) {
        return counts[a] == counts[b] ? a < b : counts[a] > counts[b];
    });
    std::vector<fill> fills;
    for (int32_t key : candidates) {
        if (fills.size() >= limit || fills.size() >= cold.size()) {
            break;
        }
        const int32_t slot = cold[fills.size()];
        const int32_t evicted = key_of[slot];
        if (evicted >= 0 && counts[key] <= counts[evicted]) {
            break;
        }
        if (evicted >= 0) {
            slot_of[evicted] = -1;
        }
        slot_of[key] = slot;
        key_of[slot] = key;
        fills.push_back({key, slot, evicted});
    }
    for (int32_t key : candidates) {
        pending[key] = false;
    }
    candidates.clear();
    return fills;
}

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
        llama_moe_cache_lfu lfu;
        std::vector<int32_t> slot_values; // immutable sources for async device map writes: -1, 0, ..., n_slots-1
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        bool has_resident = false;
        bool use_cache = false;
        int64_t n_selected = 0;
        int64_t n_tokens = 0;
        std::vector<int32_t> routes;
        ggml_tensor * slot_map_device = nullptr;
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    bool hybrid;
    bool adaptive;
    size_t admit = 8;
    uint64_t tokens = 0;
    uint64_t uploads = 0;
    uint64_t upload_bytes = 0;
    size_t next_group = 0;
    int64_t graph_tokens = 0;
    int64_t graph_outputs = 0;
    std::vector<int64_t> output_rows;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const char * profile, bool adaptive) :
            n_expert_used(model.hparams.n_expert_used_max()), hybrid(profile != nullptr || adaptive), adaptive(adaptive), layers(model.layers.size()) {
        if (adaptive) {
            const char * value = getenv("GGML_MOE_HYBRID_ADMIT");
            if (value) {
                admit = 0;
                if (!*value) {
                    throw std::runtime_error("invalid GGML_MOE_HYBRID_ADMIT");
                }
                for (const char * p = value; *p; ++p) {
                    if (*p < '0' || *p > '9' || admit > (SIZE_MAX - (*p - '0')) / 10) {
                        throw std::runtime_error("invalid GGML_MOE_HYBRID_ADMIT");
                    }
                    admit = admit*10 + (*p - '0');
                }
            }
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = hybrid ? g.layers.size()*GGML_PAD(g.ref[0]->ne[2]*sizeof(int32_t), alignment) : 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            if (adaptive && model.layers.size() > (size_t) INT32_MAX / n_expert) {
                throw std::runtime_error("too many MoE hybrid cache keys");
            }
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (hybrid && g.n_slots >= (1 << 24)) {
                throw std::runtime_error("MoE hybrid slots must be exactly representable as F32");
            }
            const int64_t min_slots = adaptive ? std::min<int64_t>(8LL*n_expert_used, n_expert) : n_expert_used;
            // A group must hold at least one layer's largest supported routing window.
            if (g.n_slots < min_slots) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            if (adaptive) {
                g.lfu.init(model.layers.size()*n_expert, g.n_slots);
                for (int32_t slot = -1; slot < g.n_slots; ++slot) {
                    g.slot_values.push_back(slot);
                }
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + (hybrid ? g.layers.size() : 0);
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                if (hybrid) {
                    l.slot_map_device = ggml_dup_tensor(ctx, l.slot_map);
                    ggml_format_name(l.slot_map_device, "moe_cache.slot_map_device-%d", il);
                }
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        if (hybrid && !model.hparams.no_alloc) {
            if (profile) {
                load_profile(profile);
            } else {
                for (const layer & l : layers) {
                    if (l.slot_map_device) {
                        ggml_backend_tensor_set(l.slot_map_device, l.slot_map->data, 0, ggml_nbytes(l.slot_map));
                    }
                }
            }
        }

        if (adaptive) {
            size_t slots = 0;
            for (const group & g : groups) {
                slots += g.n_slots;
            }
            LLAMA_LOG_WARN("llama_moe_cache: adaptive LFU: %zu slots, admit <= %zu experts/ubatch, decay every 512 tokens\n", slots, admit);
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
    }

    void load_profile(const char * path) {
        std::ifstream input(path);
        char header[64];
        if (!input.getline(header, sizeof(header)) || std::string(header) != "llama-moe-profile-v1") {
            throw std::runtime_error("invalid MoE hybrid profile header");
        }
        std::vector<int32_t> next_slot(groups.size(), 0);
        size_t uploaded = 0;
        while (input >> std::ws && input.peek() != std::char_traits<char>::eof()) {
            int64_t il;
            int64_t expert;
            int64_t count;
            if (!(input >> il >> expert >> count) || il < 0 || il >= (int64_t) layers.size() || count < 0) {
                throw std::runtime_error("invalid MoE hybrid profile entry");
            }
            layer & l = layers[il];
            if (l.ig < 0 || l.slot_map == nullptr || expert < 0 || expert >= l.slot_map->ne[1]) {
                throw std::runtime_error("MoE hybrid profile references an uncached layer or invalid expert");
            }
            group & g = groups[l.ig];
            int32_t & slot = g.lru.slot_map[il][expert];
            if (slot >= 0 || next_slot[l.ig] >= g.n_slots) {
                throw std::runtime_error("MoE hybrid profile has duplicate experts or exceeds its group budget");
            }
            slot = next_slot[l.ig]++;
            l.has_resident = true;
            g.lru.key_of[slot] = il*g.lru.n_expert + expert;
            if (adaptive) {
                g.lfu.seed(g.lru.key_of[slot], slot, (double) count);
            }
        }
        if (!input.eof()) {
            throw std::runtime_error("invalid MoE hybrid profile data");
        }
        size_t resident = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            resident += next_slot[ig];
            for (int32_t slot = 0; slot < next_slot[ig]; ++slot) {
                const int32_t key = g.lru.key_of[slot];
                const layer & l = layers[key / g.lru.n_expert];
                const int32_t expert = key % g.lru.n_expert;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ggml_backend_tensor_set_async(devices[g.id].backend, g.banks[ip],
                        (const char *) w->data + expert*w->nb[2], slot*w->nb[2], w->nb[2]);
                    uploaded += w->nb[2];
                }
            }
        }
        for (const layer & l : layers) {
            if (l.slot_map_device) {
                ggml_backend_tensor_set(l.slot_map_device, l.slot_map->data, 0, ggml_nbytes(l.slot_map));
            }
        }
        for (const device & d : devices) {
            ggml_backend_synchronize(d.backend);
        }
        if (adaptive) {
            LLAMA_LOG_WARN("llama_moe_cache: adaptive warm start: %zu experts, %.2f MiB (excluded from runtime uploads)\n", resident, uploaded/1024.0/1024.0);
        } else {
            LLAMA_LOG_INFO("llama_moe_cache: frozen hybrid profile: %zu resident experts, %.2f MiB uploaded at initialization\n",
                resident, uploaded/1024.0/1024.0);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        if (hybrid) {
            return n_tokens >= 1 && n_tokens <= 8 && (adaptive || l.has_resident) ? l.slot_map_device : nullptr;
        }

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    template<bool use_cache>
    static void collect_routes(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
        GGML_ASSERT(ith == 0 && nth == 1 && src->type == GGML_TYPE_I32);
        layer & l = *static_cast<layer *>(userdata);
        l.use_cache = use_cache;
        l.n_selected = src->ne[0];
        l.n_tokens = src->ne[1];
        l.routes.resize(l.n_selected*l.n_tokens);
        for (int64_t t = 0; t < l.n_tokens; ++t) {
            for (int64_t e = 0; e < l.n_selected; ++e) {
                int32_t id;
                memcpy(&id, (const char *) src->data + t*src->nb[1] + e*src->nb[0], sizeof(id));
                l.routes[t*l.n_selected + e] = id;
                memcpy((char *) dst->data + t*dst->nb[1] + e*dst->nb[0], &id, sizeof(id));
            }
        }
    }

    ggml_tensor * build_observe(ggml_context * ctx, ggml_tensor * ids, int32_t il, bool use_cache) {
        if (!adaptive || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return ids;
        }
        layer & l = layers[il];
        ggml_tensor * result = ggml_map_custom1(ctx, ids, use_cache ? collect_routes<true> : collect_routes<false>, 1, &l);
        ggml_format_name(result, "ffn_moe_observe-%d", il);
        return result;
    }

    void begin_graph(int64_t n_tokens, const int8_t * output) {
        if (!adaptive) {
            return;
        }
        graph_tokens = n_tokens;
        graph_outputs = 0;
        output_rows.resize(n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            output_rows[t] = !output || output[t] ? graph_outputs++ : -1;
        }
        for (layer & l : layers) {
            l.n_tokens = 0;
        }
    }

    void end_graph(bool success) {
        if (!adaptive || !success || groups.empty()) {
            return;
        }
        if (graph_tokens == 0) {
            return;
        }
        for (int64_t t = 0; t < graph_tokens; ++t) {
            for (group & g : groups) {
                if (g.n_slots) {
                    g.lfu.begin_token();
                }
            }
            for (size_t il = 0; il < layers.size(); ++il) {
                const layer & l = layers[il];
                if (l.n_tokens == 0) {
                    continue;
                }
                // The final FFN may only evaluate the requested output rows.
                GGML_ASSERT(l.n_tokens == graph_tokens || l.n_tokens == graph_outputs);
                const int64_t row = l.n_tokens == graph_tokens ? t : output_rows[t];
                if (row < 0) {
                    continue;
                }
                group & g = groups[l.ig];
                for (int64_t e = 0; e < l.n_selected; ++e) {
                    const int32_t expert = l.routes[row*l.n_selected + e];
                    GGML_ASSERT(expert >= 0 && expert < g.lru.n_expert);
                    g.lfu.observe(il*g.lru.n_expert + expert, l.use_cache);
                }
            }
        }
        tokens += graph_tokens;
        size_t remaining = admit;
        for (size_t i = 0; i < groups.size(); ++i) {
            group & g = groups[(next_group + i) % groups.size()];
            if (!g.n_slots) {
                continue;
            }
            const auto admitted = g.lfu.plan(remaining);
            remaining -= admitted.size();
            const auto backend = devices[g.id].backend;
            for (const auto & f : admitted) {
                const int32_t il = f.key / g.lru.n_expert;
                const int32_t expert = f.key % g.lru.n_expert;
                const layer & l = layers[il];
                // All graph reads are already submitted. CUDA joins its worker streams to this stream before returning.
                // These writes and the next graph's reads use that same stream; no slot can change under a reader.
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ggml_backend_tensor_set_async(backend, g.banks[ip], (const char *) w->data + expert*w->nb[2], f.slot*w->nb[2], w->nb[2]);
                    upload_bytes += w->nb[2];
                }
                if (f.evicted >= 0) {
                    const int32_t old_layer = f.evicted / g.lru.n_expert;
                    const int32_t old_expert = f.evicted % g.lru.n_expert;
                    g.lru.slot_map[old_layer][old_expert] = -1;
                    ggml_backend_tensor_set_async(backend, layers[old_layer].slot_map_device, &g.slot_values[0], old_expert*sizeof(int32_t), sizeof(int32_t));
                    upload_bytes += sizeof(int32_t);
                }
                g.lru.key_of[f.slot] = f.key;
                g.lru.slot_map[il][expert] = f.slot;
                // The upload sources are immutable until context teardown, which synchronizes all backends.
                ggml_backend_tensor_set_async(backend, l.slot_map_device, &g.slot_values[f.slot + 1], expert*sizeof(int32_t), sizeof(int32_t));
                upload_bytes += sizeof(int32_t);
                ++uploads;
            }
        }
        next_group = (next_group + 1) % groups.size();
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        if (hybrid) {
            return 0;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        if (hybrid) {
            return false;
        }
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        if (adaptive) {
            uint64_t hits = 0, misses = 0;
            for (const group & g : groups) {
                hits += g.lfu.hits;
                misses += g.lfu.misses;
            }
            const double per_token = tokens ? 1.0/tokens : 0.0;
            LLAMA_LOG_WARN("llama_moe_cache: adaptive: tokens = %llu, hits = %llu, misses = %llu, hit rate = %.2f%%, uploads = %llu, uploaded = %.2f MiB, hits/token = %.2f, misses/token = %.2f, uploads/token = %.3f, MiB/token = %.3f\n",
                (unsigned long long) tokens, (unsigned long long) hits, (unsigned long long) misses, hits + misses ? 100.0*hits/(hits + misses) : 0.0,
                (unsigned long long) uploads, upload_bytes/1024.0/1024.0, hits*per_token, misses*per_token, uploads*per_token, upload_bytes/1024.0/1024.0*per_token);
            return;
        }
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const char * profile, bool adaptive) :
    pimpl(new impl(model, backends, bufts, size, profile, adaptive)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

bool llama_moe_cache::is_hybrid() const {
    return pimpl->hybrid;
}

ggml_tensor * llama_moe_cache::build_observe(ggml_context * ctx, ggml_tensor * ids, int32_t il, bool use_cache) const {
    return pimpl->build_observe(ctx, ids, il, use_cache);
}

void llama_moe_cache::begin_graph(int64_t n_tokens, const int8_t * output) {
    pimpl->begin_graph(n_tokens, output);
}

void llama_moe_cache::end_graph(bool success) {
    pimpl->end_graph(success);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
