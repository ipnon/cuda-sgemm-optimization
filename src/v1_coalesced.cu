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
__global__ void matmul_v1_coalesced(float* A, float* B, float* C, int N) {
  // Which thread am I from 0 to 255?
  int tid = threadIdx.x;
  // Which row of the tile?
  int row = blockIdx.y * BLOCK_SIZE + (tid / BLOCK_SIZE);
  // Which column of the tile? (consecutive threads -> consecutive columns)
  int col = blockIdx.x * BLOCK_SIZE + (tid % BLOCK_SIZE);
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() { run_matmul_test(matmul_v1_coalesced, 1024, 16); }
