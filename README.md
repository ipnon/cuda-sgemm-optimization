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

Plot each kernel on the roofline to visualize optimization progress:

| Phase | Kernel | Optimization | AI (FLOP/byte) | Bound |
|-------|--------|--------------|----------------|-------|
| Basics | V0 | Naive | ~0.25 | Memory |
| | V1 | Coalescing | ~0.25 | Memory |
| | V2 | SMEM tiling | ~2.0 | Memory |
| | V3 | Bank conflicts | ~2.0 | Memory |
| Thread | V4 | 1D blocktile | ~4.0 | Memory |
| | V5 | 2D blocktile | ~8.0 | Transitioning |
| | V6 | Vectorized | ~8.0 | Transitioning |
| Pipeline | V7 | SMEM double buffer | ~10.0 | Transitioning |
| | V8 | Register prefetch | ~12.0 | Transitioning |
| Warp | V9 | Warptiling | ~16.0 | Compute |
| | V10 | Warp shuffles | ~16.0 | Compute |
| Sched | V11 | Swizzle | ~16.0 | Compute |
| | V12 | Split-K | ~16.0 | Compute |
| Tune | V13 | Autotuning | ~16.0+ | Compute |

### Deliverables

- [ ] Build roofline model for your GPU (compute FLOPS ceiling, memory bandwidth ceiling)
- [ ] Plot each kernel version on the roofline
- [ ] Calculate arithmetic intensity for each version
- [ ] Predict theoretical peak before implementing, compare to actual
- [ ] Create reusable script/spreadsheet for future kernel analysis

## References

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) ([HN discussion](https://news.ycombinator.com/item?id=34256392))
- [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [CUTLASS: Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [Roofline: An Insightful Visual Performance Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- Nsight Compute roofline analysis documentation
- Programming Massively Parallel Processors (Hwu, Kirk & Wen)

### [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) — Reading order

Read these sections before implementing each kernel:

| Kernel | Read Before |
|--------|-------------|
| V0 | Thread Hierarchy (§2.2.2) |
| V1 | Coalesced Global Memory Access (§2.2.4.1) |
| V2 | Shared Memory (§2.2.3.2), GPU Memory (§1.2.3) |
| V3 | Shared Memory Access Patterns (§2.2.4.2) |
| V4–V5 | Kernel Launch and Occupancy (§2.2.7) |
| V6 | Coalesced Global Memory Access (§2.2.4.1) — size and alignment |
| V7 | Asynchronous Execution (§2.3) |
| V8 | Pipelines (§4.10) |
| V9 | SIMT Execution Model (§3.2.2.1) |
| V10 | Warp Shuffle Functions (§5.4.6.5) |
| V11 | L2 Cache Control (§4.13) |
| V12 | Memory Fence Functions (§5.4.4.3) |
| V13 | Kernel Launch and Occupancy (§2.2.7) |
