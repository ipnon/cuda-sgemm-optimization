#pragma once
#include <cuda_runtime.h>

#include <common.cuh>

template <int kBlockM, int kBlockN, int kBlockK, int kThreadM, int kThreadN>
__global__ void matmul_v5_blocktile_2d(float* A, float* B, float* C, int N) {
  __shared__ float smem_A[kBlockM][kBlockK];
  __shared__ float smem_B[kBlockK][kBlockN];
  float accum[kThreadM * kThreadN] = {0.0f};

  constexpr int kThreadCount = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  int bm = blockIdx.y;
  int bn = blockIdx.x;
  int tid = threadIdx.x;
  int thread_idx_n = tid % (kBlockN / kThreadN);
  int thread_idx_m = tid / (kBlockN / kThreadN);
  int gm = bm * kBlockM + thread_idx_m * kThreadM;
  int gn = bn * kBlockN + thread_idx_n * kThreadN;

  for (int k_tile = 0; k_tile < N / kBlockK; k_tile++) {
    for (int i = tid; i < kBlockM * kBlockK; i += kThreadCount) {
      int load_row = i / kBlockK;
      int load_col = i % kBlockK;
      smem_A[load_row][load_col] =
          A[(bm * kBlockM + load_row) * N + (k_tile * kBlockK + load_col)];
    }
    for (int i = tid; i < kBlockK * kBlockN; i += kThreadCount) {
      int load_row = i / kBlockN;
      int load_col = i % kBlockN;
      smem_B[load_row][load_col] =
          B[(k_tile * kBlockK + load_row) * N + (bn * kBlockN + load_col)];
    }
    __syncthreads();

    for (int k = 0; k < kBlockK; k++) {
      for (int m = 0; m < kThreadM; m++) {
        float a_val = smem_A[thread_idx_m * kThreadM + m][k];
        for (int n = 0; n < kThreadN; n++) {
          float b_val = smem_B[k][thread_idx_n * kThreadN + n];
          accum[m * kThreadN + n] += a_val * b_val;
        }
      }
    }
    __syncthreads();
  }

  for (int m = 0; m < kThreadM; m++) {
    for (int n = 0; n < kThreadN; n++) {
      C[(gm + m) * N + (gn + n)] = accum[m * kThreadN + n];
    }
  }
}

int main() {
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 64;
  constexpr int kBlockK = 8;
  constexpr int kThreadM = 8;
  constexpr int kThreadN = 8;
  run_matmul_benchmark_blocktiled_2d<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>(
      matmul_v5_blocktile_2d<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>,
      1024);
}
