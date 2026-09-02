#include <orthon/ops/cuda/embedding_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

namespace orthon {

static constexpr int kThreads = 256;

static int cuda_blocks(int64_t n, int threads = kThreads) {
    return static_cast<int>((n + threads - 1) / threads);
}

template <typename IndexT>
__global__ void embedding_forward_kernel(const IndexT* __restrict__ indices,
                                         const float* __restrict__ weight,
                                         float* __restrict__ out,
                                         int64_t index_count,
                                         int64_t embedding_dim) {
    const int64_t total = index_count * embedding_dim;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t token_pos = i / embedding_dim;
        const int64_t dim = i - token_pos * embedding_dim;
        const int64_t token = static_cast<int64_t>(indices[token_pos]);
        out[i] = weight[token * embedding_dim + dim];
    }
}

template <typename IndexT>
__global__ void embedding_backward_kernel(const IndexT* __restrict__ indices,
                                          const float* __restrict__ grad_output,
                                          float* __restrict__ grad_weight,
                                          int64_t index_count,
                                          int64_t embedding_dim,
                                          int64_t padding_idx) {
    const int64_t total = index_count * embedding_dim;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t token_pos = i / embedding_dim;
        const int64_t dim = i - token_pos * embedding_dim;
        const int64_t token = static_cast<int64_t>(indices[token_pos]);
        if (token == padding_idx) continue;
        atomicAdd(grad_weight + token * embedding_dim + dim, grad_output[i]);
    }
}

Tensor cuda_embedding_forward(const Tensor& indices,
                              const Tensor& weight,
                              std::optional<int64_t> padding_idx) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(weight.device());

    (void)padding_idx;
    const int64_t index_count = indices.numel();
    const int64_t embedding_dim = weight.shape()[1];
    std::vector<int64_t> out_shape = indices.shape();
    out_shape.push_back(embedding_dim);
    Tensor out(std::make_shared<TensorImpl>(out_shape, DType::Float32, weight.device()));

    const int64_t total = index_count * embedding_dim;
    if (indices.dtype() == DType::Int64) {
        embedding_forward_kernel<<<cuda_blocks(total), kThreads>>>(
            static_cast<const int64_t*>(indices.data()),
            static_cast<const float*>(weight.data()),
            static_cast<float*>(out.data()),
            index_count,
            embedding_dim);
    } else if (indices.dtype() == DType::Int32) {
        embedding_forward_kernel<<<cuda_blocks(total), kThreads>>>(
            static_cast<const int32_t*>(indices.data()),
            static_cast<const float*>(weight.data()),
            static_cast<float*>(out.data()),
            index_count,
            embedding_dim);
    } else {
        throw std::runtime_error("cuda embedding: indices must be int32 or int64");
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_embedding_backward(const Tensor& indices,
                               const Tensor& grad_output,
                               const std::vector<int64_t>& weight_shape,
                               std::optional<int64_t> padding_idx) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(grad_output.device());

    const int64_t index_count = indices.numel();
    const int64_t embedding_dim = weight_shape[1];
    const int64_t total = index_count * embedding_dim;
    const int64_t pad = padding_idx.value_or(-1);
    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight_shape, DType::Float32, grad_output.device()));
    CUDA_CHECK(cudaMemset(grad_weight.data(), 0,
                          static_cast<size_t>(grad_weight.numel()) * sizeof(float)));

    if (indices.dtype() == DType::Int64) {
        embedding_backward_kernel<<<cuda_blocks(total), kThreads>>>(
            static_cast<const int64_t*>(indices.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(grad_weight.data()),
            index_count,
            embedding_dim,
            pad);
    } else if (indices.dtype() == DType::Int32) {
        embedding_backward_kernel<<<cuda_blocks(total), kThreads>>>(
            static_cast<const int32_t*>(indices.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(grad_weight.data()),
            index_count,
            embedding_dim,
            pad);
    } else {
        throw std::runtime_error("cuda embedding backward: indices must be int32 or int64");
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

}  // namespace orthon
