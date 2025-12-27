#include <common.cuh>
#include <kernels/v7_swizzle.cuh>

int main() {
  constexpr int kBlockM = 64, kBlockN = 64, kBlockK = 8;
  constexpr int kThreadM = 8, kThreadN = 8;
  run_matmul_benchmark_blocktiled_2d<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>(
      matmul_v7_swizzle<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>, 1024);
}