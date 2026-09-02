#include <orthon/ops/cuda/cast_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

static constexpr int kCastThreads = 256;

static int cast_blocks(int64_t n) {
    return static_cast<int>((n + kCastThreads - 1) / kCastThreads);
}

// float32 to float16 conversion.

__global__ void castop_f32_to_f16(const float* __restrict__ src,
                                   __half*       __restrict__ dst,
                                   int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(src[i]);
}

// float16 to float32 conversion.

__global__ void castop_f16_to_f32(const __half* __restrict__ src,
                                   float*         __restrict__ dst,
                                   int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __half2float(src[i]);
}

// float32 to bfloat16 conversion.

__global__ void castop_f32_to_bf16(const float*   __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst,
                                    int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(src[i]);
}

// bfloat16 to float32 conversion.

__global__ void castop_bf16_to_f32(const __nv_bfloat16* __restrict__ src,
                                    float*                __restrict__ dst,
                                    int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __bfloat162float(src[i]);
}

// float16 to bfloat16 conversion.

__global__ void castop_f16_to_bf16(const __half*  __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst,
                                    int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(__half2float(src[i]));
}

// bfloat16 to float16 conversion.

__global__ void castop_bf16_to_f16(const __nv_bfloat16* __restrict__ src,
                                    __half*               __restrict__ dst,
                                    int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(__bfloat162float(src[i]));
}

// Host-side cast dispatch.

Tensor cuda_cast(const Tensor& src, DType dst_dtype) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(src.device());

    if (!src.is_contiguous())
        throw std::runtime_error("cuda_cast: input must be contiguous");

    // Same dtype: allocate and memcpy.
    if (src.dtype() == dst_dtype) {
        Tensor dst(std::make_shared<TensorImpl>(src.shape(), dst_dtype, src.device()));
        const size_t nbytes = static_cast<size_t>(src.numel()) * dtype_size(dst_dtype);
        CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), nbytes, cudaMemcpyDeviceToDevice));
        return dst;
    }

    // Dtype validation must precede the empty-tensor early-out so an
    // unsupported pair is rejected identically for empty and non-empty
    // tensors.
    const auto lowp_ok = [](DType d) {
        return d == DType::Float32 || d == DType::Float16 || d == DType::BFloat16;
    };
    if (!lowp_ok(src.dtype()) || !lowp_ok(dst_dtype))
        throw std::runtime_error(
            "cuda_cast: unsupported dtype pair (only float32/float16/bfloat16 conversions)");

    Tensor dst(std::make_shared<TensorImpl>(src.shape(), dst_dtype, src.device()));
    const int64_t n = src.numel();
    // A zero-block grid is an invalid launch configuration; an empty cast
    // has nothing to convert.
    if (n == 0) return dst;
    const int blocks = cast_blocks(n);

    if (src.dtype() == DType::Float32 && dst_dtype == DType::Float16) {
        castop_f32_to_f16<<<blocks, kCastThreads>>>(
            static_cast<const float*>(src.data()),
            static_cast<__half*>(dst.data()), n);
    } else if (src.dtype() == DType::Float16 && dst_dtype == DType::Float32) {
        castop_f16_to_f32<<<blocks, kCastThreads>>>(
            static_cast<const __half*>(src.data()),
            static_cast<float*>(dst.data()), n);
    } else if (src.dtype() == DType::Float32 && dst_dtype == DType::BFloat16) {
        castop_f32_to_bf16<<<blocks, kCastThreads>>>(
            static_cast<const float*>(src.data()),
            static_cast<__nv_bfloat16*>(dst.data()), n);
    } else if (src.dtype() == DType::BFloat16 && dst_dtype == DType::Float32) {
        castop_bf16_to_f32<<<blocks, kCastThreads>>>(
            static_cast<const __nv_bfloat16*>(src.data()),
            static_cast<float*>(dst.data()), n);
    } else if (src.dtype() == DType::Float16 && dst_dtype == DType::BFloat16) {
        castop_f16_to_bf16<<<blocks, kCastThreads>>>(
            static_cast<const __half*>(src.data()),
            static_cast<__nv_bfloat16*>(dst.data()), n);
    } else if (src.dtype() == DType::BFloat16 && dst_dtype == DType::Float16) {
        castop_bf16_to_f16<<<blocks, kCastThreads>>>(
            static_cast<const __nv_bfloat16*>(src.data()),
            static_cast<__half*>(dst.data()), n);
    } else {
        throw std::runtime_error(
            "cuda_cast: unsupported dtype pair (only float32/float16/bfloat16 conversions)");
    }

    CUDA_CHECK(cudaGetLastError());
    return dst;
}

void cuda_copy_cast_float32_to_(Tensor& dst, const Tensor& src) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(dst.device());

    if (!src.device().is_cuda() || !dst.device().is_cuda())
        throw std::runtime_error("cuda_copy_cast_float32_to_: CUDA tensors required");
    if (src.device() != dst.device())
        throw std::runtime_error("cuda_copy_cast_float32_to_: device mismatch");
    if (src.dtype() != DType::Float32)
        throw std::runtime_error("cuda_copy_cast_float32_to_: source must be float32");
    if (dst.dtype() != DType::Float16 &&
        dst.dtype() != DType::BFloat16 &&
        dst.dtype() != DType::Float32)
        throw std::runtime_error(
            "cuda_copy_cast_float32_to_: destination must be float32/float16/bfloat16");
    if (!src.is_contiguous() || !dst.is_contiguous())
        throw std::runtime_error("cuda_copy_cast_float32_to_: tensors must be contiguous");
    if (src.numel() != dst.numel())
        throw std::runtime_error("cuda_copy_cast_float32_to_: numel mismatch");

    const int64_t n = src.numel();
    if (n == 0) return;

    if (dst.dtype() == DType::Float16) {
        castop_f32_to_f16<<<cast_blocks(n), kCastThreads>>>(
            static_cast<const float*>(src.data()),
            static_cast<__half*>(dst.data()), n);
    } else if (dst.dtype() == DType::BFloat16) {
        castop_f32_to_bf16<<<cast_blocks(n), kCastThreads>>>(
            static_cast<const float*>(src.data()),
            static_cast<__nv_bfloat16*>(dst.data()), n);
    } else {
        CUDA_CHECK(cudaMemcpy(dst.data(), src.data(),
                              static_cast<size_t>(n) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        return;
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace orthon
