cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

# RTX 2060 对应 SM 7.5
set(CMAKE_CUDA_ARCHITECTURES 75)

project(MiniGPU LANGUAGES CXX CUDA)

add_executable(
    gemm_bench_unified
    gemm_bench_unified.cu
)

add_executable(gemm_bench_graph gemm_bench_graph.cu)

target_link_libraries(gemm_bench_graph
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)



target_link_libraries(
    gemm_bench_unified
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)
find_package(CUDAToolkit REQUIRED)

add_executable(gemm_compare gemm_compare.cu)

target_link_libraries(gemm_compare
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# V1：双层 MLP
add_executable(mlp mlp.cu)

# V1：多 Shape 正确性测试
add_executable(mlp_shape_test mlp_shape_test.cu)

add_executable(gemm_bench gemm_bench.cu)

add_executable(gemm_tiled gemm_tiled.cu)
add_executable(gemm_register_2_tile8 gemm_register_2_tile8.cu)
add_executable(gemm_register_2 gemm_register_2.cu)

add_executable(
    gemm_register_2_tile32
    gemm_register_2_tile32.cu
)
# RTX 2060：Compute Capability 7.5
set_target_properties(
    mlp
    mlp_shape_test
    PROPERTIES
    CUDA_ARCHITECTURES 75
)