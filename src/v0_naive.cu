#include <cuda_runtime.h>

#include <common.cuh>
#include <cstdio>

template <int kBlockSize>
__global__ void matmul_v0_naive(float* A, float* B, float* C, int N) {
  // Calculate this thread's global row and column
  int row = blockIdx.y * kBlockSize + threadIdx.y;
  int col = blockIdx.x * kBlockSize + threadIdx.x;
  // Do not compute outside of the matrix bounds
  if (row < N && col < N) {
    // Initialize the accumulator for the dot product
    float sum = 0.0f;
    // Loop over the shared dimension
    for (int k = 0; k < N; k++) {
      // Accumulate the k-th column of A * the k-th row of B
      sum += A[row * N + k] * B[k * N + col];
    }
    // Write the result
    C[row * N + col] = sum;
  }
}

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v0_naive<kBlockSize>, 1024);
}
