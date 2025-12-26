cmake_minimum_required(VERSION 3.18)
project(cuda-sgemm-optimization LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # A10

find_package(CUDAToolkit REQUIRED)

# Interface library for shared settings (include path + cuRAND)
add_library(common INTERFACE)
target_include_directories(common INTERFACE include)
target_link_libraries(common INTERFACE CUDA::curand)

# Kernel executables
set(KERNELS
    v0_naive
    v1_coalesced
    v2_tiled
    v3_noconflict
    v4_blocktile_1d
    v5_blocktile_2d
    v6_vectorized
    v7_double_buffer
    v8_register_prefetch
    v9_warptiled
    v10_warp_shuffle
    v11_swizzle
    v12_splitk
    v13_autotuned
)

foreach(kernel ${KERNELS})
    add_executable(${kernel} src/${kernel}.cu)
    target_link_libraries(${kernel} PRIVATE common)
endforeach()
