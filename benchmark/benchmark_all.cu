#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <kernels/v0_naive.cuh>
#include <kernels/v1_coalesced.cuh>
#include <kernels/v2_tiled.cuh>
#include <kernels/v3_noconflict.cuh>
#include <kernels/v4_blocktile_1d.cuh>
#include <kernels/v5_blocktile_2d.cuh>
#include <kernels/v6_vectorized.cuh>
#include <kernels/v7_swizzle.cuh>

// Tile parameters
constexpr int kBlockSize = 16;
constexpr int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;

// Benchmark infrastructure
float benchmark_cublas(float* A, float* B, float* C, int n,
                       int warmup = 3, int iters = 10) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  for (int i = 0; i < warmup; i++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  return 2.0 * n * n * n * iters / (ms * 1e6);
}

template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, float* A, float* B, float* C,
                       int n, dim3 threads, dim3 blocks,
                       int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; i++)
    kernel<<<blocks, threads>>>(A, B, C, n);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    kernel<<<blocks, threads>>>(A, B, C, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 2.0 * n * n * n * iters / (ms * 1e6);
}

struct KernelConfig {
  std::string name;
  std::function<float(float*, float*, float*, int)> run;
};

std::vector<KernelConfig> make_kernels() {
  return {
    {"V0", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v0_naive<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V1", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v1_coalesced<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V2", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v2_tiled<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V3", [](float* A, float* B, float* C, int n) {
      dim3 threads(kBlockSize, kBlockSize);
      dim3 blocks((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
      return benchmark_kernel(matmul_v3_noconflict<kBlockSize>, A, B, C, n, threads, blocks);
    }},
    {"V4", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * BN);
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v4_blocktile_1d<BM, BN, BK, TM>, A, B, C, n, threads, blocks);
    }},
    {"V5", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v5_blocktile_2d<BM, BN, BK, TM, TN>, A, B, C, n, threads, blocks);
    }},
    {"V6", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v6_vectorized<BM, BN, BK, TM, TN>, A, B, C, n, threads, blocks);
    }},
    {"V7s1", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v7_swizzle<BM, BN, BK, TM, TN, 1>, A, B, C, n, threads, blocks);
    }},
    {"V7s2", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v7_swizzle<BM, BN, BK, TM, TN, 2>, A, B, C, n, threads, blocks);
    }},
    {"V7s4", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v7_swizzle<BM, BN, BK, TM, TN, 4>, A, B, C, n, threads, blocks);
    }},
    {"V7s8", [](float* A, float* B, float* C, int n) {
      dim3 threads((BM / TM) * (BN / TN));
      dim3 blocks(n / BN, n / BM);
      return benchmark_kernel(matmul_v7_swizzle<BM, BN, BK, TM, TN, 8>, A, B, C, n, threads, blocks);
    }},
  };
}

int main() {
  std::array<int, 4> sizes = {512, 1024, 2048, 4096};
  auto kernels = make_kernels();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  float peak_tflops = prop.clockRate * 1e-6f * prop.multiProcessorCount * 128 * 2 / 1000.0f;

  printf("\nCUDA SGEMM Benchmark Suite\n");
  printf("==========================\n");
  printf("GPU: %s (%.1f TFLOPS FP32 peak)\n\n", prop.name, peak_tflops);

  // Collect all results first
  std::vector<std::vector<float>> results(kernels.size() + 1);  // +1 for cuBLAS
  for (auto& row : results) row.resize(sizes.size());

  for (size_t si = 0; si < sizes.size(); si++) {
    int n = sizes[si];
    float *A, *B, *C;
    cudaMalloc(&A, n * n * sizeof(float));
    cudaMalloc(&B, n * n * sizeof(float));
    cudaMalloc(&C, n * n * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, A, n * n);
    curandGenerateUniform(gen, B, n * n);
    curandDestroyGenerator(gen);

    for (size_t ki = 0; ki < kernels.size(); ki++) {
      results[ki][si] = kernels[ki].run(A, B, C, n);
    }
    results[kernels.size()][si] = benchmark_cublas(A, B, C, n);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }

  // Print GFLOPS table (kernels as rows, sizes as columns)
  printf("GFLOPS:\n");
  printf("%-8s", "Kernel");
  for (int n : sizes) printf(" | %6d", n);
  printf("\n");

  printf("--------");
  for (size_t i = 0; i < sizes.size(); i++) printf("-|-------");
  printf("\n");

  for (size_t ki = 0; ki < kernels.size(); ki++) {
    printf("%-8s", kernels[ki].name.c_str());
    for (size_t si = 0; si < sizes.size(); si++) {
      printf(" | %6.0f", results[ki][si]);
    }
    printf("\n");
  }
  printf("%-8s", "cuBLAS");
  for (size_t si = 0; si < sizes.size(); si++) {
    printf(" | %6.0f", results[kernels.size()][si]);
  }
  printf("\n");

  // Print % of cuBLAS table
  printf("\n%% of cuBLAS:\n");
  printf("%-8s", "Kernel");
  for (int n : sizes) printf(" | %6d", n);
  printf("\n");

  printf("--------");
  for (size_t i = 0; i < sizes.size(); i++) printf("-|-------");
  printf("\n");

  for (size_t ki = 0; ki < kernels.size(); ki++) {
    printf("%-8s", kernels[ki].name.c_str());
    for (size_t si = 0; si < sizes.size(); si++) {
      float pct = 100.0f * results[ki][si] / results[kernels.size()][si];
      printf(" | %5.1f%%", pct);
    }
    printf("\n");
  }

  printf("\nTarget: >= 70%% of cuBLAS across all sizes\n");
  return 0;
}
