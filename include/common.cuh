#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>

// Benchmark cuBLAS SGEMM, return GFLOPS
inline float benchmark_cublas(float* A, float* B, float* C, int n,
                              int warmup = 3, int iters = 10) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;

  for (int i = 0; i < warmup; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n,
                &beta, C, n);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n,
                &beta, C, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * n * n * n * iters / (ms * 1e6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  return gflops;
}

// Benchmark a kernel, return GFLOPS
template <typename KernelFunc>
inline float benchmark_kernel(KernelFunc kernel, float* A, float* B, float* C,
                              int n, dim3 threads, dim3 blocks,
                              int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; i++) {
    kernel<<<blocks, threads>>>(A, B, C, n);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    kernel<<<blocks, threads>>>(A, B, C, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * n * n * n * iters / (ms * 1e6);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return gflops;
}

// Core benchmark runner - allocates memory, benchmarks kernel vs cuBLAS
template <typename KernelFunc>
inline void run_benchmark(KernelFunc kernel, size_t n, dim3 threads,
                          dim3 blocks) {
  const int n_squared = n * n;
  const int bytes = n_squared * sizeof(float);

  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, bytes);
  cudaMalloc(&B_d, bytes);
  cudaMalloc(&C_d, bytes);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandGenerateUniform(gen, A_d, n_squared);
  curandGenerateUniform(gen, B_d, n_squared);
  curandDestroyGenerator(gen);

  float kernel_gflops = benchmark_kernel(kernel, A_d, B_d, C_d, n, threads, blocks);
  float cublas_gflops = benchmark_cublas(A_d, B_d, C_d, n);

  printf("N=%zu: %.1f GFLOPS (%.1f%% of cuBLAS %.1f GFLOPS)\n",
         n, kernel_gflops, 100.0 * kernel_gflops / cublas_gflops, cublas_gflops);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

// For v0-v3: 2D thread blocks (kBlockSize x kBlockSize)
template <int kBlockSize, typename KernelFunc>
inline void run_matmul_benchmark(KernelFunc kernel, size_t n) {
  dim3 threads(kBlockSize, kBlockSize);
  dim3 blocks((n + kBlockSize - 1) / kBlockSize,
              (n + kBlockSize - 1) / kBlockSize);
  run_benchmark(kernel, n, threads, blocks);
}

// For v4: 1D blocktiling (each thread computes TM elements)
template <int BM, int BN, int BK, int TM, typename KernelFunc>
inline void run_matmul_benchmark_blocktiled(KernelFunc kernel, size_t n) {
  dim3 threads((BM / TM) * BN);
  dim3 blocks(n / BN, n / BM);
  run_benchmark(kernel, n, threads, blocks);
}

// For v5+: 2D blocktiling (each thread computes TM x TN elements)
template <int BM, int BN, int BK, int TM, int TN, typename KernelFunc>
inline void run_matmul_benchmark_blocktiled_2d(KernelFunc kernel, size_t n) {
  dim3 threads((BM / TM) * (BN / TN));
  dim3 blocks(n / BN, n / BM);
  run_benchmark(kernel, n, threads, blocks);
}
