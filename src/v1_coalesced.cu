#include <cuda_runtime.h>

#include <common.cuh>
#include <cstdio>

/**
 * The first optimization enables GMEM coalescing by ensuring threads
 * in a warp access adjacent memory addresses.
 *
 * For C[i][j] = dot(row i of A, column j of B):
 * - Row i of A is contiguous in memory (good)
 * - Column j of B is strided by N (bad - each element is N floats apart)
 *
 * When threads in a warp compute different columns of C, they all
 * access different columns of B simultaneously, causing strided access.
 * This optimization reorders thread-to-output mapping so that threads
 * in a warp access adjacent elements.
 */
template <int kBlockSize>
__global__ void matmul_v1_coalesced(float* A, float* B, float* C, int N) {
  // Thread mapping:
  // tid = 0..255  (assuming kBlockSize=16, so 256 threads per block)
  // row = blockIdx.y * 16 + (tid / 16)   → local row 0-15 (strided)
  // col = blockIdx.x * 16 + (tid % 16)   → local col 0-15 (coalesced)
  int tid = threadIdx.x;
  int row = blockIdx.y * kBlockSize + (tid / kBlockSize);
  int col = blockIdx.x * kBlockSize + (tid % kBlockSize);
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() {
  constexpr int kBlockSize = 16;
  run_matmul_benchmark<kBlockSize>(matmul_v1_coalesced<kBlockSize>, 1024);
}
