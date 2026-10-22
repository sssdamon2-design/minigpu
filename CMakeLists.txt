cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

# RTX 2060 对应 SM 7.5
set(CMAKE_CUDA_ARCHITECTURES 75)

project(MiniGPU LANGUAGES CXX CUDA)




set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# V1：双层 MLP
add_executable(mlp mlp.cu)

# V1：多 Shape 正确性测试
add_executable(mlp_shape_test mlp_shape_test.cu)

# RTX 2060：Compute Capability 7.5
set_target_properties(
    mlp
    mlp_shape_test
    PROPERTIES
    CUDA_ARCHITECTURES 75
)