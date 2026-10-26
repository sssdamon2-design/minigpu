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

# V4.1：Row-wise Softmax
add_executable(softmax_v4 softmax_v4.cu)

# V4.2：RMSNorm
add_executable(rmsnorm_v4 rmsnorm_v4.cu)

# V4.3：Attention Forward
add_executable(
    attention_v4
    attention_v4.cu
    softmax_v4.cu
)

target_compile_definitions(
    attention_v4 PRIVATE MINIGPU_SOFTMAX_NO_MAIN
)

target_link_libraries(
    attention_v4 PRIVATE CUDA::cublas CUDA::cudart
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


# V5.1：GPU Buffer RAII + 已有 Softmax
add_executable(
    runtime_v5
    runtime_v5.cu
    softmax_v4.cu
)

target_compile_definitions(
    runtime_v5 PRIVATE MINIGPU_SOFTMAX_NO_MAIN
)

target_link_libraries(
    runtime_v5 PRIVATE CUDA::cudart
)

# V5.2: Buffer Reuse + Stream/Event + Attention
add_executable(
    runtime_attention_v5
    runtime_attention_v5.cu
    attention_v4.cu
    softmax_v4.cu
)

target_compile_definitions(
    runtime_attention_v5 PRIVATE
    MINIGPU_ATTENTION_NO_MAIN
    MINIGPU_SOFTMAX_NO_MAIN
)

target_link_libraries(
    runtime_attention_v5 PRIVATE
    CUDA::cublas
    CUDA::cudart
)