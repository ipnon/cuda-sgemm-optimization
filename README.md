# CUDA SGEMM Optimization

14 CUDA SGEMM kernels demonstrating progressive optimization following the CUTLASS hierarchical GEMM structure

## Goal

Achieve **70%+ of cuBLAS SGEMM performance** through iterative optimization.

## Kernel progression

| Phase | Version | Optimization | Key Concept |
|-------|---------|--------------|-------------|
| **Basics** | V0 | Naive | Baseline, each thread computes one element |
| | V1 | GMEM Coalescing | Reorder thread indexing for coalesced access |
| | V2 | Shared Memory Tiling | Cache tiles in SMEM, reduce GMEM traffic |
| | V3 | Bank Conflict Avoidance | SMEM padding to avoid bank conflicts |
| **Thread-level** | V4 | 1D Blocktiling | Each thread computes TM elements (column) |
| | V5 | 2D Blocktiling | Each thread computes TM×TN elements |
| | V6 | Vectorized Loads | float4 for 128-bit memory transactions |
| **Pipelining** | V7 | SMEM Double Buffering | Overlap GMEM loads with compute |
| | V8 | Register Prefetch | Warp fragment double buffering |
| **Warp-level** | V9 | Warptiling | Warp-level GEMM, register cache locality |
| | V10 | Warp Shuffles | `__shfl_sync` to exchange registers within warp |
| **Scheduling** | V11 | Threadblock Swizzle | L2 cache locality via CTA reordering |
| | V12 | Split-K | Parallelize reduction across threadblocks |
| **Tuning** | V13 | Autotuning | Parameter search (BM, BN, BK, TM, TN) |

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

## Roofline analysis

Plot each kernel version on the roofline to understand optimization progress:

| Kernel | Arithmetic Intensity (FLOP/byte) | Bound |
|--------|----------------------------------|-------|
| V0 | ~0.25 | Memory |
| V2 | ~2.0 | Memory |
| V5 | ~8.0 | Transitioning |
| V9+ | ~16+ | Compute |

### Deliverables

- [ ] Build roofline model for your GPU (compute FLOPS ceiling, memory bandwidth ceiling)
- [ ] Plot each kernel version on the roofline
- [ ] Calculate arithmetic intensity for each version
- [ ] Predict theoretical peak before implementing, compare to actual
- [ ] Create reusable script/spreadsheet for future kernel analysis

### References

- [Roofline: An Insightful Visual Performance Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- Nsight Compute roofline analysis documentation

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

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) ([HN discussion](https://news.ycombinator.com/item?id=34256392))
- [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [CUTLASS: Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- Programming Massively Parallel Processors (Hwu, Kirk & Wen)

### CUDA C++ Programming Guide — Reading order

Read these sections before implementing each kernel:

| Kernel | Read Before |
|--------|-------------|
| V0 | [Thread Hierarchy (§5.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy) |
| V1 | [Device Memory Accesses (§8.3.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) |
| V2 | [Shared Memory (§6.2.4)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory), [Memory Hierarchy (§5.3)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy) |
| V3 | [Shared Memory (§6.2.4)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) — bank conflicts |
| V4–V5 | [Maximize Utilization (§8.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-utilization) |
| V6 | [Device Memory Accesses (§8.3.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) — size and alignment |
| V7 | [Asynchronous Concurrent Execution (§5.5)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) |
| V8 | [Instruction Level Parallelism (§8.2.2.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#instruction-level-parallelism) |
| V9 | [SIMT Architecture (§7.1)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) |
| V10 | [Warp Shuffle Functions (§9.7.8)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions) |
| V11 | [Memory Hierarchy (§5.3)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy) — L2 cache |
| V12 | [Memory Fence Functions (§9.7.2)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions) |
| V13 | [Execution Configuration (§8.4)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration-optimizations) |
