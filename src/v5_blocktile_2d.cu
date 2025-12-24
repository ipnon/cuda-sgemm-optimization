#pragma once
#include <cuda_runtime.h>

#include <common.cuh>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v5_blocktile_2d(float* A, float* B, float* C, int N) {
  __shared__ float A_tile[BM][BK];
  __shared__ float B_tile[BK][BN];
  float acc[TM * TN] = {0.0f};
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  constexpr int num_threads = (BM / TM) * (BN / TN);
  int tid = threadIdx.x;
  int thread_col = tid % (BN / TN);
  int thread_row = tid / (BN / TN);
  int global_row = block_row * BM + thread_row * TM;
  int global_col = block_col * BN + thread_col * TN;
  for (int tile_k = 0; tile_k < N / BK; tile_k++) {
    for (int i = tid; i < BM * BK; i += num_threads) {
      int load_row = i / BK;
      int load_col = i % BK;
      A_tile[load_row][load_col] =
          A[(block_row * BM + load_row) * N + (tile_k * BK + load_col)];
    }
    for (int i = tid; i < BK * BN; i += num_threads) {
      int load_row = i / BN;
      int load_col = i % BN;
      B_tile[load_row][load_col] =
          B[(tile_k * BK + load_row) * N + (block_col * BN + load_col)];
    }
    __syncthreads();
    for (int k = 0; k < BK; k++) {
      for (int m = 0; m < TM; m++) {
        float a_val = A_tile[thread_row * TM + m][k];
        for (int n = 0; n < TN; n++) {
          float b_val = B_tile[k][thread_col * TN + n];
          acc[m * TN + n] += a_val * b_val;
        }
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; m++) {
    for (int n = 0; n < TN; n++) {
      C[(global_row + m) * N + (global_col + n)] = acc[m * TN + n];
    }
  }
}

int main() {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  run_matmul_test_blocktiled_2d<BM, BN, BK, TM, TN>(
      matmul_v5_blocktile_2d<BM, BN, BK, TM, TN>, 1024);
}
