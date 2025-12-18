#pragma once
#include <cuda_runtime.h>

#include <common.cuh>
#include <cstdio>

// Let's say we have a 9x9 matrix and 3x3 blocks. To compute
// block (1,1) of C, we need all blocks in the middle column of B and all blocks
// in the middle row of A. Load the topmost block of B and leftmost block of A
// into shared memory. Then compute the partial sum based on these available
// values. Next load the middlemost blocks, and similarly do the partial sum
// computation and add to the existing result. The same is done for the last
// blocks. By adding the partial sums for each value in the block, we arrive at
// the correct values for the block (1,1) in C.
__global__ void matmul_v2_tiled(float* A, float* B, float* C, int N) {
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  float sum = 0;
  // Which block of C are we computing?
  int block_row = blockIdx.y;  // row of blocks
  int block_col = blockIdx.x;  // col of blocks
  // Thread position within the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Global row/col this thread contributes to
  int row = block_row * BLOCK_SIZE + ty;
  int col = block_col * BLOCK_SIZE + tx;
  // Iterate through blocks along K
  for (size_t b = 0; b < N / BLOCK_SIZE; b++) {
    // Load A: row stays fixed, column walks right with b
    As[ty][tx] = A[row * N + (b * BLOCK_SIZE + tx)];
    // Load B: row walks down with b, column stays fixed
    Bs[ty][tx] = B[(b * BLOCK_SIZE + ty) * N + col];
    // Wait for all partial vectors to load into shared memory
    __syncthreads();
    // Compute partial sum from this block pair
    for (size_t k = 0; k < BLOCK_SIZE; k++) sum += As[ty][k] * Bs[k][tx];
    // Prevent fast threads from starting next iteration and overwriting shared
    // memory while slow threads are still reading current iteration
    __syncthreads();
  }
  C[row * N + col] = sum;  // Write to the global position in 1D answer
}

int main() { run_matmul_test(matmul_v2_tiled, 1024, 16); }
