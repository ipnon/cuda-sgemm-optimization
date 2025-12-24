#pragma once
#include <cuda_runtime.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>

// Core implementation - allocates memory, fills with random, launches kernel
template <typename KernelFunc>
inline void run_matmul_kernel(KernelFunc kernel, size_t n, dim3 threads, dim3 blocks) {
  const int n_squared = n * n;
  const int bytes = n_squared * sizeof(float);

  float* A_h = static_cast<float*>(malloc(bytes));
  float* B_h = static_cast<float*>(malloc(bytes));
  float* C_h = static_cast<float*>(malloc(bytes));

  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, bytes);
  cudaMalloc(&B_d, bytes);
  cudaMalloc(&C_d, bytes);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  curandGenerateUniform(gen, A_d, n_squared);
  curandGenerateUniform(gen, B_d, n_squared);

  kernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
  cudaDeviceSynchronize();

  cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

  printf("Kernel launched: %d blocks (%dx%d), %d threads/block\n",
         blocks.x * blocks.y, blocks.x, blocks.y,
         threads.x * threads.y * threads.z);

  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

// For v0-v3: 2D thread blocks (kBlockSize x kBlockSize)
template <int kBlockSize, typename KernelFunc>
inline void run_matmul_test(KernelFunc kernel, size_t n) {
  dim3 threads(kBlockSize, kBlockSize);
  dim3 blocks((n + kBlockSize - 1) / kBlockSize,
              (n + kBlockSize - 1) / kBlockSize);
  run_matmul_kernel(kernel, n, threads, blocks);
}

// For v4: 1D blocktiling (each thread computes TM elements)
template <int BM, int BN, int BK, int TM, typename KernelFunc>
inline void run_matmul_test_blocktiled(KernelFunc kernel, size_t n) {
  dim3 threads((BM / TM) * BN);
  dim3 blocks(n / BN, n / BM);
  run_matmul_kernel(kernel, n, threads, blocks);
}

// For v5+: 2D blocktiling (each thread computes TM x TN elements)
template <int BM, int BN, int BK, int TM, int TN, typename KernelFunc>
inline void run_matmul_test_blocktiled_2d(KernelFunc kernel, size_t n) {
  dim3 threads((BM / TM) * (BN / TN));
  dim3 blocks(n / BN, n / BM);
  run_matmul_kernel(kernel, n, threads, blocks);
}
