# High-Performance Tiled Matrix Multiplication

A deep dive into GPU optimization: 10 progressively optimized CUDA kernels, from naive to warptiled.

Based on [Simon Boehm's excellent worklog](https://siboehm.com/articles/22/CUDA-MMM).

## Goal

Achieve **70%+ of cuBLAS SGEMM performance** through iterative optimization.

## Kernel progression

| Version | Optimization | Key Concept | Target |
|---------|--------------|-------------|--------|
| V0 | Naive | Baseline, each thread computes one element | ~1% |
| V1 | GMEM Coalescing | Reorder thread indexing for coalesced access | ~8% |
| V2 | Shared Memory Tiling | Cache tiles in SMEM, reduce GMEM traffic | ~13% |
| V3 | 1D Blocktiling | Each thread computes TM elements (column) | ~37% |
| V4 | 2D Blocktiling | Each thread computes TM×TN elements | ~69% |
| V5 | Vectorized Loads | float4 for 128-bit memory transactions | ~78% |
| V6 | Bank Conflict Avoidance | SMEM padding/swizzling | ~80% |
| V7 | Double Buffering | Overlap compute with memory loads | ~82% |
| V8 | Autotuning | Parameter search (BM, BN, BK, TM, TN) | ~85% |
| V9 | Warptiling | Warp-level scheduling, register cache locality | ~94% |

## Concepts

- **V0-V1**: Memory coalescing, warp-level access patterns
- **V2**: Shared memory as programmer-managed cache, `__syncthreads()`
- **V3-V4**: Arithmetic intensity, register reuse, thread-level blocking
- **V5**: Vectorized memory access, `float4`, `reinterpret_cast`
- **V6**: Shared memory banks, conflict avoidance via padding
- **V7**: Latency hiding, software pipelining
- **V8**: Empirical tuning, parameter search space
- **V9**: Warp scheduling, ILP, register cache locality

## Project Structure

```
high-performance-tiled-matmul/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── v0_naive.cu
│   ├── v1_coalesced.cu
│   ├── v2_tiled.cu
│   ├── v3_blocktile_1d.cu
│   ├── v4_blocktile_2d.cu
│   ├── v5_vectorized.cu
│   ├── v6_noconflict.cu
│   ├── v7_double_buffer.cu
│   ├── v8_autotuned.cu
│   └── v9_warptiled.cu
└── benchmark/
    └── benchmark_all.cu
```

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

## Profiling

```bash
# Occupancy and throughput
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second ./v2_tiled

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./v6_noconflict

# Full analysis
ncu --set full -o profile ./benchmark_all
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

