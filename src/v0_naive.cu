#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK_SIZE = 16;
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// In naive parallel matrix multiplication, each thread computes one element of
// C.
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

int main() {
  const int n = 1024;
  const int n_squared_float = n * n * sizeof(float);
  // Allocate host memory
  float* h_a = static_cast<float*>(malloc(n_squared_float));
  float* h_b = static_cast<float*>(malloc(n_squared_float));
  float* h_c = static_cast<float*>(malloc(n_squared_float));
  // Initialize matrices
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      h_a[i * n + j] = 1.0f;
      h_b[i * n + j] = 1.0f;
    }
  }
  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n_squared_float);
  cudaMalloc(&d_b, n_squared_float);
  cudaMalloc(&d_c, n_squared_float);
  // Copy to device
  cudaMemcpy(d_a, h_a, n_squared_float, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n_squared_float, cudaMemcpyHostToDevice);
  // Launch kernel with 16x16 thread blocks
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(ceil_div(n, BLOCK_SIZE), ceil_div(n, BLOCK_SIZE));
  matmul_v0_naive<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
  // Copy result back
  cudaMemcpy(h_c, d_c, n_squared_float, cudaMemcpyDeviceToHost);
  // Verify result
  int errors = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (h_c[i * n + j] != (float)n) errors++;
    }
  }
  printf("Errors: %d\n", errors);
  // Free memory
  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
