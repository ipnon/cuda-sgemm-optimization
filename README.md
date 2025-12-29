# CUDA SGEMM Optimization

Progressive CUDA SGEMM optimization achieving **83% of cuBLAS** on NVIDIA A10.

## Results (NVIDIA A10)

**GPU:** 31.2 TFLOPS FP32 peak, 600 GB/s HBM. **Target:** ≥70% of cuBLAS.

| Kernel | 4096×4096 | % cuBLAS | Optimization |
|--------|-----------|----------|--------------|
| V0 | 1,273 GFLOPS | 10% | Baseline (1 element/thread) |
| V1 | 1,351 GFLOPS | 11% | Coalesced global memory access |
| V2 | 1,746 GFLOPS | 14% | SMEM tiling (32×32) |
| V3 | 1,713 GFLOPS | 14% | Bank conflict padding |
| V4 | 4,365 GFLOPS | 35% | 1D blocktile (TM=8) |
| V5 | 6,133 GFLOPS | 50% | 2D blocktile (TM=TN=8) |
| V6 | 9,080 GFLOPS | **73%** | Vectorized float4 loads |
| V7 | 9,052 GFLOPS | 73% | Threadblock swizzle |
| V8 | 9,281 GFLOPS | 75% | Double buffering |
| V8_256 | 10,235 GFLOPS | **83%** | BK=256 tile size |
| cuBLAS | 12,390 GFLOPS | 100% | Reference |

## Key Learnings

### 1. Vectorization was the biggest win

V5→V6 (adding `float4` loads) jumped from 50% to 73% of cuBLAS—the largest single improvement. Wide memory transactions matter more than clever scheduling.

### 2. Profiler suggestions don't always match reality

After V6, Nsight Compute showed:
```
Compute (SM) Throughput:  41%
Memory Throughput:        68%
SMSP Workload Imbalance:  23% potential speedup  ← Profiler's top suggestion
```

I implemented swizzling (V7) to address the 23% imbalance. **Result: no measurable gain.** The profiler identified a real inefficiency, but fixing it didn't translate to wall-clock improvement at this problem size.

### 3. Tile size tuning > adding techniques

V8 (double buffering) gave only 2% over V6. But V8_256 (same kernel, BK=256 instead of 64) hit 83%. Parameter tuning delivered more than the algorithmic improvement.

### 4. The 80/20 rule applies

V0→V6 covers the fundamentals and gets you to 73%. Everything after (swizzle, pipelining, warptiling) fights for the remaining ~10%. Know when to stop optimizing and when to reach for cuBLAS.

## Kernel Progression

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

### Arithmetic Intensity

For blocktiled GEMM with tile dimensions BM×BN:
```
AI = (BM × BN) / (2 × (BM + BN))    FLOP/byte
```

V0-V3 operate at ~4 FLOP/byte (memory bound). V4+ reach 16 FLOP/byte, transitioning toward compute bound.

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Running

```bash
./benchmark_all        # Full benchmark
./v0_naive             # Individual kernels
./v6_vectorized
```

## References

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [CUTLASS: Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [Roofline Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- Programming Massively Parallel Processors (Hwu, Kirk & Wen)

### CUDA Programming Guide — Reading Order

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
