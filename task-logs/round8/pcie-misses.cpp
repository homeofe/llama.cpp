#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

// Call the existing masked MMVQ kernel without a scheduler upload of host weights.
struct ggml_backend_cuda_context;
struct ggml_cuda_mm_fusion_args_host;
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *, const ggml_tensor *);

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct mapped_weights {
    void * host = nullptr;
    void * device = nullptr;
    ggml_backend_buffer_ptr buffer;
    explicit mapped_weights(size_t size) {
        check(cudaHostAlloc(&host, size, cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&device, host, 0));
        buffer.reset(ggml_backend_cpu_buffer_from_ptr(host, size));
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ~mapped_weights() {
        buffer.reset();
        if (host) {
            cudaFreeHost(host);
        }
    }
};

struct device_array {
    ggml_backend_buffer_ptr buffer;
    void * data = nullptr;
    explicit device_array(size_t size) : buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0), size)) {
        GGML_ASSERT(buffer);
        data = ggml_backend_buffer_get_base(buffer.get());
    }
};

struct operation {
    int layer;
    std::vector<int> original_ids;
    std::vector<int32_t> ids;
    std::set<int> misses;
    ggml_context_ptr ctx;
    std::unique_ptr<mapped_weights> weights;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * w;
    ggml_tensor * x;
    ggml_tensor * route;
    ggml_tensor * mask;
    ggml_tensor * out;
    ggml_cgraph * graph;
    ggml_tensor gpu_w, gpu_x, gpu_ids, gpu_mask, gpu_out;
    std::unique_ptr<device_array> dx, di, dm, dy;
    std::vector<float> cpu_mask, cuda_mask, result_gpu, reference;

    operation(const nlohmann::json & l, const nlohmann::json & tensor) : layer(l.at("layer")) {
        std::set<int> unique;
        for (const auto & row : l.at("ids")) {
            for (int id : row) {
                unique.insert(id);
            }
        }
        original_ids.assign(unique.begin(), unique.end());
        GGML_ASSERT(!original_ids.empty() && original_ids.size() <= 40);
        for (int id : l.at("missed")) {
            misses.insert(id);
        }
        for (const auto & row : l.at("ids")) {
            GGML_ASSERT(row.size() == 10);
            for (int id : row) {
                ids.push_back(std::lower_bound(original_ids.begin(), original_ids.end(), id) - original_ids.begin());
            }
        }
        GGML_ASSERT(ids.size() == 40);
        const int k = tensor.at("shape")[0], m = tensor.at("shape")[1], ne = tensor.at("shape")[2];
        GGML_ASSERT((k == 2560 && m == 640) || (k == 640 && m == 2560));
        GGML_ASSERT(ne == 512 && original_ids.back() < ne && original_ids.front() >= 0);
        const int type = tensor.at("type_id");
        GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);
        ctx.reset(ggml_init({ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false), nullptr, true}));
        w = ggml_new_tensor_3d(ctx.get(), (ggml_type) type, k, m, original_ids.size());
        const size_t slice = w->nb[2];
        GGML_ASSERT(slice == tensor.at("expert_bytes").get<size_t>());
        weights.reset(new mapped_weights(ggml_nbytes(w) + 4096));
        std::memset((char *) weights->host + ggml_nbytes(w), 0, 4096);
        GGML_ASSERT(ggml_backend_tensor_alloc(weights->buffer.get(), w, weights->host) == GGML_STATUS_SUCCESS);
        std::ifstream file(tensor.at("file").get<std::string>(), std::ios::binary);
        GGML_ASSERT(file.good());
        const uint64_t offset = tensor.at("offset");
        for (size_t i = 0; i < original_ids.size(); ++i) {
            file.seekg(offset + original_ids[i]*slice);
            file.read((char *) w->data + i*slice, slice);
            GGML_ASSERT(file.good());
        }
        const bool down = tensor.at("name").get<std::string>().find("ffn_down") != std::string::npos;
        x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, down ? 10 : 1, 4);
        route = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 10, 4);
        mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 10, 4);
        out = ggml_mul_mat_id_masked(ctx.get(), w, x, route, mask);
        graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, out);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
        for (int64_t i = 0; i < ggml_nelements(x); ++i) {
            ((float *) x->data)[i] = std::sin((float) (i % 1024)*0.13f)*0.2f;
        }
        memcpy(route->data, ids.data(), ggml_nbytes(route));
        dx.reset(new device_array(ggml_nbytes(x)));
        di.reset(new device_array(ggml_nbytes(route)));
        dm.reset(new device_array(ggml_nbytes(mask)));
        dy.reset(new device_array(ggml_nbytes(out)));
        gpu_w = *w; gpu_w.data = weights->device;
        gpu_x = *x; gpu_x.data = dx->data; gpu_x.buffer = dx->buffer.get();
        gpu_ids = *route; gpu_ids.data = di->data; gpu_ids.buffer = di->buffer.get();
        gpu_mask = *mask; gpu_mask.data = dm->data; gpu_mask.buffer = dm->buffer.get();
        gpu_out = *out; gpu_out.data = dy->data; gpu_out.buffer = dy->buffer.get();
        cpu_mask.resize(ids.size()); cuda_mask.resize(ids.size());
        result_gpu.resize(ggml_nelements(out));
        reference.resize(ggml_nelements(out));
    }

    void run(ggml_backend_t cpu, ggml_backend_t gpu, const std::set<std::pair<int, int>> & selected, bool baseline) {
        bool use_gpu = false;
        for (size_t i = 0; i < ids.size(); ++i) {
            const int expert = original_ids[ids[i]];
            const bool miss = misses.count(expert);
            const bool on_gpu = miss && selected.count({layer, expert});
            cpu_mask[i] = miss && !on_gpu;
            cuda_mask[i] = on_gpu;
            use_gpu |= on_gpu;
        }
        memcpy(mask->data, cpu_mask.data(), ggml_nbytes(mask));
        if (use_gpu) {
            ggml_backend_tensor_set_async(gpu, &gpu_x, x->data, 0, ggml_nbytes(x));
            ggml_backend_tensor_set_async(gpu, &gpu_ids, ids.data(), 0, ggml_nbytes(route));
            ggml_backend_tensor_set_async(gpu, &gpu_mask, cuda_mask.data(), 0, ggml_nbytes(mask));
            ggml_cuda_mul_mat_vec_q(*static_cast<ggml_backend_cuda_context *>(gpu->context), &gpu_w, &gpu_x, &gpu_ids, &gpu_out, nullptr, &gpu_mask);
        }
        GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
        if (use_gpu) {
            ggml_backend_tensor_get_async(gpu, &gpu_out, result_gpu.data(), 0, ggml_nbytes(out));
            ggml_backend_synchronize(gpu);
            for (size_t i = 0; i < result_gpu.size(); ++i) {
                ((float *) out->data)[i] += result_gpu[i];
            }
        }
        if (baseline) {
            memcpy(reference.data(), out->data, ggml_nbytes(out));
        }
    }

    double error() const {
        double error = 0, scale = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double value = ((float *) out->data)[i];
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite result");
            }
            error += (value-reference[i])*(value-reference[i]);
            scale += reference[i]*reference[i];
        }
        return error/std::max(scale, 1e-30);
    }
};

int main(int argc, char ** argv) try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--check-manifest")) {
        fprintf(stderr, "usage: %s pcie-manifest.json [--check-manifest]\n", argv[0]);
        return 1;
    }
    std::ifstream input(argv[1]);
    nlohmann::json manifest;
    input >> manifest;
    std::map<int, std::set<std::pair<int, int>>> selections;
    for (int pct : {0, 10, 20, 30}) {
        for (const auto & expert : manifest.at("splits").at(std::to_string(pct)).at("experts")) {
            selections[pct].emplace(expert.at(0).get<int>(), expert.at(1).get<int>());
        }
    }
    if (argc == 3) {
        printf("manifest: %zu layers; 10/20/30 percent subsets: %zu/%zu/%zu experts\n", manifest.at("layers").size(), selections[10].size(), selections[20].size(), selections[30].size());
        return 0;
    }
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory) {
        throw std::runtime_error("device cannot map host memory");
    }
    ggml_backend_ptr gpu(ggml_backend_cuda_init(0));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    GGML_ASSERT(gpu && cpu);
    ggml_backend_cpu_set_n_threads(cpu.get(), 24);
    std::vector<std::unique_ptr<operation>> ops;
    for (const auto & layer : manifest.at("layers")) {
        for (const auto & tensor : layer.at("tensors")) {
            ops.emplace_back(new operation(layer, tensor));
        }
    }
    for (auto & op : ops) {
        op->run(cpu.get(), gpu.get(), selections[0], true);
        op->run(cpu.get(), gpu.get(), selections[30], false);
        GGML_ASSERT(op->error() < 5e-4);
    }
    for (int round = 0; round < 3; ++round) {
        for (int index = 0; index < 6; ++index) {
            const int order[] = {0, 10, 0, 20, 0, 30};
            const int pct = order[(index + 2*round) % 6];
            double before[3], after[3];
            GGML_ASSERT(getloadavg(before, 3) == 3);
            if (before[0] >= 3) {
                return 75;
            }
            const auto start = std::chrono::steady_clock::now();
            for (auto & op : ops) {
                op->run(cpu.get(), gpu.get(), selections[pct], false);
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            GGML_ASSERT(getloadavg(after, 3) == 3);
            double worst = 0;
            for (auto & op : ops) {
                worst = std::max(worst, op->error());
            }
            GGML_ASSERT(worst < 5e-4);
            printf("{\"round\":%d,\"gpu_pct\":%d,\"gpu_byte_pct\":%.5f,\"ms_per_window\":%.6f,\"load_start\":%.3f,\"load_end\":%.3f,\"nmse\":%.8g,\"accepted\":%s}\n", round+1,pct,manifest["splits"][std::to_string(pct)]["percent"].get<double>(),ms,before[0],after[0],worst,after[0]<3?"true":"false");
            fflush(stdout);
            if (after[0] >= 3) {
                return 75;
            }
        }
    }
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
}
