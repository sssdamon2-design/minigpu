
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

void launch_gemm_tiled16(
    const float*, const float*, float*,
    int, int, int, cudaStream_t);

torch::Tensor gemm_forward(
    torch::Tensor A,
    torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),
                "Inputs must be CUDA tensors");

    TORCH_CHECK(A.device() == B.device(),
                "Inputs must be on the same device");

    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32,
                "Only FP32 is supported");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "Inputs must be 2D matrices");

    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "Inputs must be contiguous");

    TORCH_CHECK(A.size(1) == B.size(0),
                "GEMM dimensions do not match");

    c10::cuda::CUDAGuard device_guard(A.device());

    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));

    auto C = torch::empty({M, N}, A.options());

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            A.get_device()).stream();

    launch_gemm_tiled16(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N,
        stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_forward,
          "MiniGPU FP32 GEMM Forward");
}
