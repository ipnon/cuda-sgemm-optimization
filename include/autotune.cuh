#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <vector>
#include <string>
#include <cstdio>

#include <kernels/v6_vectorized.cuh>

// Kernel function pointer type
using SgemmKernel = void(*)(float*, float*, float*, int);

// Configuration for a kernel variant
struct KernelConfig {
    std::string name;
    int bm, bn, bk, tm, tn;
    SgemmKernel kernel;

    int threads() const { return (bm / tm) * (bn / tn); }
    dim3 grid(int n) const { return dim3(n / bn, n / bm); }
    bool valid_for(int n) const { return n % bm == 0 && n % bn == 0; }
};

// Pre-instantiated kernel configurations
inline std::vector<KernelConfig> get_configs() {
    return {
        {"64x64_k8",   64,  64,  8, 8, 8, matmul_v6_vectorized<64, 64, 8, 8, 8>},
        {"128x64_k8",  128, 64,  8, 8, 8, matmul_v6_vectorized<128, 64, 8, 8, 8>},
        {"64x128_k8",  64,  128, 8, 8, 8, matmul_v6_vectorized<64, 128, 8, 8, 8>},
        {"128x128_k8", 128, 128, 8, 8, 8, matmul_v6_vectorized<128, 128, 8, 8, 8>},
        {"128x128_k16",128, 128, 16,8, 8, matmul_v6_vectorized<128, 128, 16, 8, 8>},
        {"256x64_k8",  256, 64,  8, 8, 8, matmul_v6_vectorized<256, 64, 8, 8, 8>},
        {"64x256_k8",  64,  256, 8, 8, 8, matmul_v6_vectorized<64, 256, 8, 8, 8>},
    };
}

// Benchmark a single kernel config
inline float benchmark_config(const KernelConfig& cfg, float* A, float* B, float* C,
                              int n, int warmup = 5, int iters = 10) {
    if (!cfg.valid_for(n)) return 0.0f;

    dim3 threads(cfg.threads());
    dim3 blocks = cfg.grid(n);

    for (int i = 0; i < warmup; i++)
        cfg.kernel<<<blocks, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++)
        cfg.kernel<<<blocks, threads>>>(A, B, C, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 2.0f * n * n * n * iters / (ms * 1e6f);  // GFLOPS
}

// Autotuner: finds best config for each matrix size
class Autotuner {
public:
    // Run autotuning for a specific size
    void tune(int n, float* A, float* B, float* C, bool verbose = true) {
        auto configs = get_configs();
        float best_gflops = 0;
        int best_idx = 0;

        if (verbose) printf("Autotuning for N=%d:\n", n);

        for (size_t i = 0; i < configs.size(); i++) {
            if (!configs[i].valid_for(n)) continue;

            float gflops = benchmark_config(configs[i], A, B, C, n);
            if (verbose) printf("  %-14s: %.0f GFLOPS\n", configs[i].name.c_str(), gflops);

            if (gflops > best_gflops) {
                best_gflops = gflops;
                best_idx = i;
            }
        }

        best_config_[n] = configs[best_idx];
        if (verbose) printf("  Best: %s (%.0f GFLOPS)\n\n",
                           configs[best_idx].name.c_str(), best_gflops);
    }

    // Get best config for a size (must call tune() first)
    const KernelConfig& get_best(int n) const {
        return best_config_.at(n);
    }

    // Run SGEMM with autotuned config
    void sgemm(float* A, float* B, float* C, int n) const {
        const auto& cfg = get_best(n);
        dim3 threads(cfg.threads());
        dim3 blocks = cfg.grid(n);
        cfg.kernel<<<blocks, threads>>>(A, B, C, n);
    }

    // Check if size has been tuned
    bool is_tuned(int n) const {
        return best_config_.find(n) != best_config_.end();
    }

private:
    std::map<int, KernelConfig> best_config_;
};

// Hardcoded best configs from benchmarking (skip autotuning at runtime)
inline KernelConfig get_hardcoded_best(int n) {
    auto configs = get_configs();
    // Based on A10 benchmark results:
    // 1024 -> 64x64 (V6/V7 style)
    // 2048 -> 64x64
    // 4096 -> 128x128_k16
    // 8192 -> 256x64
    if (n <= 1024) return configs[0];       // 64x64_k8
    if (n <= 2048) return configs[0];       // 64x64_k8
    if (n <= 4096) return configs[4];       // 128x128_k16
    return configs[5];                       // 256x64_k8
}
