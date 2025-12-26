#pragma once
#include <cuda_runtime.h>

#include <common.cuh>

template <int kBlockM, int kBlockN, int kBlockK, int kThreadM, int kThreadN>
__global__ void matmul_v6_vectorized(float* A, float* B, float* C, int N) {
  __shared__ float smem_A[kBlockM][kBlockK];
  __shared__ float smem_B[kBlockK][kBlockN];
  float accum[kThreadM * kThreadN] = {0.0f};
  int bm = blockIdx.y;
  int bn = blockIdx.x;
  int tid = threadIdx.x;
  int thread_idx_n = tid % (kBlockN / kThreadN);
  int thread_idx_m = tid / (kBlockN / kThreadN);
  int gm = bm * kBlockM + thread_idx_m * kThreadM;
  int gn = bn * kBlockN + thread_idx_n * kThreadN;
  constexpr int kThreadCount = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  for (int k_tile = 0; k_tile < N / kBlockK; k_tile++) {
    constexpr int kALoadsPerThread = (kBlockM * kBlockK) / (kThreadCount * 4);
    for (int i = 0; i < kALoadsPerThread; i++) {
      int float4_idx = tid + i * kThreadCount;
      int float_idx = float4_idx * 4;
      int load_row = float_idx / kBlockK;
      int load_col = float_idx % kBlockK;
      // The [0] dereferences the pointer and loads the float4 at that address.
      float4 tmp = reinterpret_cast<float4*>(
          &A[(bm * kBlockM + load_row) * N + (k_tile * kBlockK + load_col)])[0];
      smem_A[load_row][load_col + 0] = tmp.x;
      smem_A[load_row][load_col + 1] = tmp.y;
      smem_A[load_row][load_col + 2] = tmp.z;
      smem_A[load_row][load_col + 3] = tmp.w;
    }
    constexpr int kBLoadsPerThread = (kBlockK * kBlockN) / (kThreadCount * 4);
    for (int i = 0; i < kBLoadsPerThread; i++) {
      int float4_idx = tid + i * kThreadCount;
      int float_idx = float4_idx * 4;
      int load_row = float_idx / kBlockN;
      int load_col = float_idx % kBlockN;
      float4 tmp = reinterpret_cast<float4*>(
          &B[(k_tile * kBlockK + load_row) * N + (bn * kBlockN + load_col)])[0];
      smem_B[load_row][load_col + 0] = tmp.x;
      smem_B[load_row][load_col + 1] = tmp.y;
      smem_B[load_row][load_col + 2] = tmp.z;
      smem_B[load_row][load_col + 3] = tmp.w;
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
  run_matmul_test_blocktiled_2d<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>(
      matmul_v6_vectorized<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>,
      1024);
}
