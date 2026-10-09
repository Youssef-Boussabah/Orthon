#include "index_validation.cuh"
#include <orthon/planning/index_validation.h>
#include <orthon/backends/cuda/nll_loss_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/runtime.h>
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

static constexpr int kNLLBlock = 256;

__device__ int64_t read_target_device(const void* targets, int64_t row, int is_int64) {
    if (is_int64)
        return static_cast<const int64_t*>(targets)[row];
    return static_cast<int64_t>(static_cast<const int32_t*>(targets)[row]);
}

__global__ void nll_none_forward_kernel(const float* __restrict__ log_probs,
                                        const void* __restrict__ targets,
                                        float* __restrict__ out,
                                        int64_t N,
                                        int64_t C,
                                        int is_int64) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < N; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t target = read_target_device(targets, row, is_int64);
        if (target < 0 || target >= C) continue;
        out[row] = -log_probs[row * C + target];
    }
}

__global__ void nll_sum_forward_kernel(const float* __restrict__ log_probs,
                                       const void* __restrict__ targets,
                                       float* __restrict__ out,
                                       int64_t N,
                                       int64_t C,
                                       int is_int64,
                                       int reduce_mean) {
    __shared__ float smem[kNLLBlock];

    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int64_t row = tid; row < N; row += kNLLBlock) {
        const int64_t target = read_target_device(targets, row, is_int64);
        if (target < 0 || target >= C) continue;
        acc += -log_probs[row * C + target];
    }
    smem[tid] = acc;
    __syncthreads();

    for (int stride = kNLLBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        out[0] = reduce_mean ? smem[0] / static_cast<float>(N) : smem[0];
}

__global__ void nll_backward_kernel(const void* __restrict__ targets,
                                    const float* __restrict__ grad_output,
                                    float* __restrict__ grad,
                                    int64_t N,
                                    int64_t C,
                                    int is_int64,
                                    int reduction_kind) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < N; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t target = read_target_device(targets, row, is_int64);
        if (target < 0 || target >= C) continue;
        const float upstream = reduction_kind == 0 ? grad_output[row] : grad_output[0];
        const float scale = reduction_kind == 1 ? 1.0f / static_cast<float>(N) : 1.0f;
        grad[row * C + target] = -upstream * scale;
    }
}

static int target_is_int64(const Tensor& targets) {
    return targets.dtype() == DType::Int64 ? 1 : 0;
}

Tensor cuda_nll_loss_forward(const Tensor& log_probs,
                             const Tensor& targets,
                             const std::string& reduction) {
    validate_nll_inputs(log_probs, targets, reduction);
    if (!log_probs.device().is_cuda()) throw std::invalid_argument("cuda nll_loss: CUDA tensors required");
    DeviceGuard guard(log_probs.device());
    auto context = default_runtime().cuda(log_probs.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int64_t N = log_probs.shape()[0];
    const int64_t C = log_probs.shape()[1];
    detail::validate_cuda_index_domain(targets, C, *context, "nll_loss");
    const int is_int64 = target_is_int64(targets);

    if (reduction == "none") {
        Tensor out(std::make_shared<TensorImpl>(
            std::vector<int64_t>{N}, DType::Float32, log_probs.device()));
        const int blocks = static_cast<int>(std::min<int64_t>(1 + (N - 1) / kNLLBlock, 4096));
        nll_none_forward_kernel<<<blocks, kNLLBlock, 0, stream>>>(
            static_cast<const float*>(log_probs.data()),
            targets.data(),
            static_cast<float*>(out.data()),
            N,
            C,
            is_int64);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{1}, DType::Float32, log_probs.device()));
    nll_sum_forward_kernel<<<1, kNLLBlock, 0, stream>>>(
        static_cast<const float*>(log_probs.data()),
        targets.data(),
        static_cast<float*>(out.data()),
        N,
        C,
        is_int64,
        reduction == "mean" ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_nll_loss_backward(const Tensor& log_probs,
                              const Tensor& targets,
                              const Tensor& grad_output,
                              const std::string& reduction) {
    validate_nll_inputs(log_probs, targets, reduction, &grad_output);
    if (!log_probs.device().is_cuda()) throw std::invalid_argument("cuda nll_loss backward: CUDA tensors required");
    DeviceGuard guard(log_probs.device());
    auto context = default_runtime().cuda(log_probs.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int64_t N = log_probs.shape()[0];
    const int64_t C = log_probs.shape()[1];
    detail::validate_cuda_index_domain(targets, C, *context, "nll_loss");
    const int is_int64 = target_is_int64(targets);

    Tensor grad(std::make_shared<TensorImpl>(
        log_probs.shape(), DType::Float32, log_probs.device()));
    context->memset_async(grad.data(), 0, static_cast<size_t>(grad.numel()) * sizeof(float));

    const int reduction_kind = reduction == "none" ? 0 : (reduction == "mean" ? 1 : 2);
    const int blocks = static_cast<int>(std::min<int64_t>(1 + (N - 1) / kNLLBlock, 4096));
    nll_backward_kernel<<<blocks, kNLLBlock, 0, stream>>>(
        targets.data(),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad.data()),
        N,
        C,
        is_int64,
        reduction_kind);
    CUDA_CHECK(cudaGetLastError());
    return grad;
}

}  // namespace orthon
