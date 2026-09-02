#include <orthon/ops/cuda/fusion_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

namespace orthon {

static constexpr int kTile = 16;

static void validate_cuda_fused_linear_relu(const Tensor& x,
                                            const Tensor& weight,
                                            const Tensor& bias) {
    if (!x.device().is_cuda() || !weight.device().is_cuda() || !bias.device().is_cuda())
        throw std::runtime_error("cuda_fused_linear_relu: all inputs must be on CUDA");
    if (x.device() != weight.device() || x.device() != bias.device())
        throw std::runtime_error("cuda_fused_linear_relu: device mismatch");
    if (x.dtype() != DType::Float32 ||
        weight.dtype() != DType::Float32 ||
        bias.dtype() != DType::Float32)
        throw std::runtime_error("cuda_fused_linear_relu: only float32 supported");
    if (!x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous())
        throw std::runtime_error("cuda_fused_linear_relu: inputs must be contiguous");
    if (x.ndim() != 2 || weight.ndim() != 2 || bias.ndim() != 1)
        throw std::runtime_error("cuda_fused_linear_relu: expected x[2D], weight[2D], bias[1D]");
    if (x.shape()[1] != weight.shape()[0] || weight.shape()[1] != bias.shape()[0])
        throw std::runtime_error("cuda_fused_linear_relu: shape mismatch");
}

__global__ void fused_linear_relu_kernel(const float* __restrict__ x,
                                         const float* __restrict__ weight,
                                         const float* __restrict__ bias,
                                         float* __restrict__ out,
                                         int M,
                                         int K,
                                         int N) {
    const int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    const int col = (int)blockIdx.x * kTile + (int)threadIdx.x;

    if (row < M && col < N) {
        float acc = bias[col];
        for (int p = 0; p < K; ++p)
            acc += x[row * K + p] * weight[p * N + col];
        out[row * N + col] = acc > 0.0f ? acc : 0.0f;
    }
}

Tensor cuda_fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(x.device());

    validate_cuda_fused_linear_relu(x, weight, bias);

    const int M = static_cast<int>(x.shape()[0]);
    const int K = static_cast<int>(x.shape()[1]);
    const int N = static_cast<int>(weight.shape()[1]);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, x.dtype(), x.device()));

    dim3 block(kTile, kTile);
    dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
              static_cast<unsigned>((M + kTile - 1) / kTile));

    fused_linear_relu_kernel<<<grid, block>>>(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(weight.data()),
        static_cast<const float*>(bias.data()),
        static_cast<float*>(out.data()),
        M, K, N);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

}  // namespace orthon
