#include <cuda_runtime.h>

#include <common.cuh>
#include <cstdio>

__global__ void matmul_v0_naive(float* A, float* B, float* C, int N) {
  // Calculate this thread's global row and column
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
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

int main() { run_matmul_test(matmul_v0_naive, 1024, 16); }
