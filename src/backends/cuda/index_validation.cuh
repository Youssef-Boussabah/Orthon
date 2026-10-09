#pragma once
#include <orthon/core/tensor.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/memory/allocator.h>
#include <orthon/runtime/cuda_context.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace orthon { namespace detail {

template <typename Index>
__global__ void check_index_domain(const Index* indices, int64_t count,
                                   int64_t bound, int* invalid) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t index = static_cast<int64_t>(indices[i]);
        if (index < 0 || index >= bound) atomicExch(invalid, 1);
    }
}

inline void validate_cuda_index_domain(const Tensor& indices, int64_t bound,
                                       CudaContext& context, const char* operation) {
    if (!indices.device().is_cuda() || indices.device() != context.device() ||
        !indices.is_contiguous() ||
        (indices.dtype() != DType::Int32 && indices.dtype() != DType::Int64))
        throw std::invalid_argument(std::string(operation) + ": invalid CUDA index metadata");
    const auto n = indices.numel();
    if (n == 0) return;
    auto allocation = context.allocator()->allocate(sizeof(int));
    auto* invalid = static_cast<int*>(allocation.data());
    context.memset_async(invalid, 0, sizeof(int));
    const auto stream = static_cast<cudaStream_t>(context.compute_stream().native_handle());
    const int blocks = static_cast<int>(std::min<int64_t>(1 + (n - 1) / 256, 4096));
    if (indices.dtype() == DType::Int64)
        check_index_domain<<<blocks, 256, 0, stream>>>(
            static_cast<const int64_t*>(indices.data()), n, bound, invalid);
    else
        check_index_domain<<<blocks, 256, 0, stream>>>(
            static_cast<const int32_t*>(indices.data()), n, bound, invalid);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    // Managed rejection precedes the main gather/scatter. This wait also keeps
    // the temporary flag alive until every preflight reader has completed.
    int host_invalid = 0;
    context.copy_d2h_blocking(&host_invalid, invalid, sizeof(int));
    if (host_invalid) throw std::out_of_range(std::string(operation) + ": index out of range");
}

} } // namespace orthon::detail
