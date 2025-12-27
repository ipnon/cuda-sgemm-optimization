cmake_minimum_required(VERSION 3.18)
project(cuda-sgemm-optimization LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # A10

find_package(CUDAToolkit REQUIRED)

# Interface library for shared settings (include path + cuRAND)
add_library(common INTERFACE)
target_include_directories(common INTERFACE include)
target_link_libraries(common INTERFACE CUDA::curand CUDA::cublas)

# Kernel executables (add new kernels here as implemented)
set(KERNELS
    v0_naive
    v1_coalesced
    v2_tiled
    v3_noconflict
    v4_blocktile_1d
    v5_blocktile_2d
    v6_vectorized
    v7_swizzle
)

foreach(kernel ${KERNELS})
    add_executable(${kernel} src/${kernel}.cu)
    target_link_libraries(${kernel} PRIVATE common)
endforeach()

# Benchmark suite
add_executable(benchmark_all benchmark/benchmark_all.cu)
target_include_directories(benchmark_all PRIVATE include)
target_link_libraries(benchmark_all PRIVATE CUDA::curand CUDA::cublas)
