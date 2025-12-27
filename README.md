# CUDA SGEMM Optimization

CUDA SGEMM kernels demonstrating progressive optimization, ordered by **profiler-measured impact**.

## Goal

Achieve **70%+ of cuBLAS SGEMM performance** through iterative optimization.

## Kernel Progression

These kernel versions are ordered by relative complexity, but the returns are marginal, and for parameters even regressive. Benchmarking is crucial.

| Phase | Version | Optimization | Key Concept |
|-------|---------|--------------|-------------|
| **Basics** | V0 | Naive | Baseline, each thread computes one element |
| | V1 | GMEM Coalescing | Reorder thread indexing for coalesced access |
| | V2 | Shared Memory Tiling | Cache tiles in SMEM, reduce GMEM traffic |
| | V3 | Bank Conflict Avoidance | SMEM padding to avoid bank conflicts |
| **Thread-level** | V4 | 1D Blocktiling | Each thread computes TM elements (column) |
| | V5 | 2D Blocktiling | Each thread computes TM×TN elements |
| | V6 | Vectorized Loads | float4 for 128-bit memory transactions |
| **Scheduling** | V7 | Threadblock Swizzle | L2 cache locality via CTA reordering |
| **Pipelining** | V8 | SMEM Double Buffering | Overlap GMEM loads with compute |
| | V9 | Register Prefetch | Prefetch into registers during compute |
| **Warp-level** | V10 | Warptiling | Warp-cooperative GEMM, register locality |
| | V11 | Warp Shuffles | `__shfl_sync` to exchange data within warp |
| **Advanced** | V12 | Split-K | Parallelize K-reduction across threadblocks |
| | V13 | Autotuning | Parameter search (BM, BN, BK, TM, TN) |

### Why Swizzle Before Pipelining?

After implementing V6, Nsight Compute profiling showed:

```
Compute (SM) Throughput:  41%
Memory Throughput:        68%   ← Already decent
SMSP Workload Imbalance:  23% potential speedup  ← Biggest bottleneck!
```

V6 already achieves 68% memory throughput. Pipelining (double buffering) hides latency
but won't help much when memory is already flowing well. The **workload imbalance**—some
SMs finishing early while others are overloaded—is the real bottleneck.

Swizzling reorders threadblock execution so simultaneously-running blocks share L2 cache
data, balancing work across SMs. This directly addresses the measured bottleneck.

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

## Roofline Analysis

Use `ncu --set roofline` to profile each kernel and plot on the roofline.

### Arithmetic Intensity Formula

For blocktiled GEMM with tile dimensions BM×BN:
```
AI = (BM × BN) / (2 × (BM + BN))    FLOP/byte
```

### Measured Results (NVIDIA A10)

| Phase | Kernel | Optimization | AI (FLOP/byte) | Bound |
|-------|--------|--------------|----------------|-------|
| Basics | V0 | Naive | ~3* | Memory |
| | V1 | Coalescing | ~3* | Memory |
| | V2 | SMEM tiling | 4 | Memory |
| | V3 | Bank conflicts | 4 | Memory |
| Thread | V4 | 1D blocktile | 16 | Memory |
| | V5 | 2D blocktile | 16 | Memory |
| | V6 | Vectorized | 16 | Memory |
| Sched | V7 | Swizzle | 16 | Memory |
| Pipeline | V8 | Double buffer | 16 | Memory |
| | V9 | Register prefetch | 16 | Transitioning |
| Warp | V10 | Warptiling | 16+ | Compute |
| | V11 | Warp shuffles | 16+ | Compute |
| Advanced | V12 | Split-K | 16+ | Compute |
| | V13 | Autotuning | 16+ | Compute |

*V0-V1 theoretical AI is 0.25, but L2 cache provides ~3 effective AI.

**Note:** Pipelining and swizzling don't change arithmetic intensity—they improve
how efficiently you reach the ceiling at a given AI.

### Deliverables

- [x] Build roofline model for your GPU (compute FLOPS ceiling, memory bandwidth ceiling)
- [x] Plot each kernel version on the roofline (via Nsight Compute)
- [x] Calculate arithmetic intensity for each version
- [ ] Predict theoretical peak before implementing, compare to actual
- [ ] Document optimization decisions based on profiler data

## References

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) ([HN discussion](https://news.ycombinator.com/item?id=34256392))
- [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [CUTLASS: Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [Roofline: An Insightful Visual Performance Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- Nsight Compute roofline analysis documentation
- Programming Massively Parallel Processors (Hwu, Kirk & Wen)

### [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) — Reading Order

Read these sections before implementing each kernel:

| Kernel | Read Before |
|--------|-------------|
| V0 | Thread Hierarchy (§2.2.2) |
| V1 | Coalesced Global Memory Access (§2.2.4.1) |
| V2 | Shared Memory (§2.2.3.2), GPU Memory (§1.2.3) |
| V3 | Shared Memory Access Patterns (§2.2.4.2) |
| V4–V5 | Kernel Launch and Occupancy (§2.2.7) |
| V6 | Coalesced Global Memory Access (§2.2.4.1) — size and alignment |
| V7 | L2 Cache Control (§4.13) |
| V8 | Asynchronous Execution (§2.3) |
| V9 | Pipelines (§4.10) |
| V10 | SIMT Execution Model (§3.2.2.1) |
| V11 | Warp Shuffle Functions (§5.4.6.5) |
| V12 | Memory Fence Functions (§5.4.4.3) |
| V13 | Kernel Launch and Occupancy (§2.2.7) |
