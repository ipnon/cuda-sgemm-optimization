# CUDA SGEMM Optimization

13 CUDA SGEMM kernels demonstrating progressive optimization from naive implementation to warp-level tiling

## Goal

Achieve **70%+ of cuBLAS SGEMM performance** through iterative optimization.

## Kernel progression

| Version | Optimization | Key Concept |
|---------|--------------|-------------|
| V0 | Naive | Baseline, each thread computes one element |
| V1 | GMEM Coalescing | Reorder thread indexing for coalesced access |
| V2 | Shared Memory Tiling | Cache tiles in SMEM, reduce GMEM traffic |
| V3 | 1D Blocktiling | Each thread computes TM elements (column) |
| V4 | 2D Blocktiling | Each thread computes TM×TN elements |
| V5 | Vectorized Loads | float4 for 128-bit memory transactions |
| V6 | Bank Conflict Avoidance | SMEM padding/swizzling |
| V7 | Double Buffering | Overlap compute with memory loads |
| V8 | Autotuning | Parameter search (BM, BN, BK, TM, TN) |
| V9 | Warptiling | Warp-level scheduling, register cache locality |
| V10 | Warp Shuffles | `__shfl_sync` to exchange registers within warp |
| V11 | K-Loop Reordering | CUTLASS-style outermost k-loop |
| V12 | 2-Tile Streaming | Stream one matrix from GMEM, reduce SMEM pressure |

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Running

```bash
# Individual kernels
./v0_naive
./v1_coalesced
# ...

# Full benchmark
./benchmark_all
```

## Lessons learned

### Memory & access patterns

1. Why is coalesced memory access important? What happens at the hardware level when it's not coalesced?
2. What's the difference between global, shared, and register memory? Latency and bandwidth of each?
3. Why does accessing `B[k * N + col]` in naive matmul hurt performance?

### Shared memory

4. What is a bank conflict? How do you detect one? How do you fix it?
5. Why do we need `__syncthreads()` and what happens if you forget one?
6. How much shared memory does your GPU have per SM? Per block?

### Performance analysis

7. What is arithmetic intensity? Why does it matter for the roofline model?
8. Is your kernel compute-bound or memory-bound? How do you tell?
9. What does Nsight Compute's "Memory Throughput" vs "Compute Throughput" tell you?

### Parallelism

10. What is a warp? Why does warp divergence hurt performance?
11. What is occupancy? Is higher always better?
12. How do you choose block dimensions? What are the tradeoffs?

### Optimization techniques

13. Why does computing more results per thread improve performance?
14. What is double buffering and why does it help?
15. Why are `float4` loads faster than four `float` loads?

### Systems thinking

16. Given a new kernel that's slow, what's your debugging process?
17. How would you estimate the theoretical peak performance of a matmul on a given GPU?
18. Why is cuBLAS still faster than your best kernel? What tricks might they use?

## References

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- Programming Massively Parallel Processors (Hwu, Kirk & Wen)
- NVIDIA CUDA C++ Programming Guide:
  - [Thread Hierarchy (§5.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
  - [Memory Hierarchy (§5.3)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
  - [SIMT Architecture (§7.1)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)
  - [Shared Memory (§6.2.4)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
  - [Device Memory Accesses (§8.3.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
