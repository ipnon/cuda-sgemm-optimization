#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

template <typename KernelFunc>
inline void run_matmul_test(KernelFunc kernel, size_t n, size_t block_size) {
  const int bytes = n * n * sizeof(float);

  float* A_h = static_cast<float*>(malloc(bytes));
  float* B_h = static_cast<float*>(malloc(bytes));
  float* C_h = static_cast<float*>(malloc(bytes));

  for (size_t i = 0; i < n * n; i++) {
    A_h[i] = 1.0f;
    B_h[i] = 1.0f;
  }

  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, bytes);
  cudaMalloc(&B_d, bytes);
  cudaMalloc(&C_d, bytes);

  cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

  int threads = block_size * block_size;
  dim3 blocks((n + block_size - 1) / block_size,
              (n + block_size - 1) / block_size);

  kernel<<<blocks, threads>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

  int errors = 0;
  for (size_t i = 0; i < n * n; i++) {
    if (C_h[i] != static_cast<float>(n)) errors++;
  }
  printf("Errors: %d\n", errors);

  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
