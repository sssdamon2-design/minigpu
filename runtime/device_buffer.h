
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

// V5.1：最小 GPU 显存 RAII 封装
template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
        : count_(count)
    {
        if (count == 0)
            throw std::invalid_argument("DeviceBuffer size is zero");

        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&ptr_),
            count_ * sizeof(T)
        );

        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    }

    ~DeviceBuffer() noexcept
    {
        if (ptr_ != nullptr)
            (void)cudaFree(ptr_);
    }

    // 禁止复制，避免两个对象拥有同一块显存
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return ptr_; }

    std::size_t size() const noexcept { return count_; }

    std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};
