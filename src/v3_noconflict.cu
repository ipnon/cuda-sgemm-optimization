#pragma once
#include <cuda_runtime.h>

#include <common.cuh>
#include <cstdio>

// Shared memory has 32 banks, each 4 bytes wide. Bank conflicts occur when
// multiple threads in a warp access different addresses in the same bank.
// We avoid conflicts on B_smem (column access) using XOR swizzling.
// A_smem uses linear indexing (row access = no conflicts).
template <int kBlockSize>
__global__ void matmul_v3_noconflict(float* A, float* B, float* C, int N) {
  __shared__ float A_smem[kBlockSize][kBlockSize];
  __shared__ float B_smem[kBlockSize][kBlockSize];
  float sum = 0;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = block_row * kBlockSize + ty;
  int col = block_col * kBlockSize + tx;
  for (size_t b_i = 0; b_i < N / kBlockSize; b_i++) {
    // A_smem: linear indexing (row access = no bank conflicts)
    // B_smem: XOR swizzle (column access would conflict without it)
    A_smem[ty][tx] = A[row * N + (b_i * kBlockSize + tx)];
    B_smem[ty][tx ^ ty] = B[(b_i * kBlockSize + ty) * N + col];
    __syncthreads();
    for (size_t k = 0; k < kBlockSize; k++) {
      sum += A_smem[ty][k] * B_smem[k][tx ^ k];
    }
    __syncthreads();
  }
  C[row * N + col] = sum;
}

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v3_noconflict<kBlockSize>, 1024);
}
