#include <orthon/ops/cuda/unary_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/memory_pool.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

namespace orthon {

static constexpr int kThreads = 256;
static constexpr int kMaxDims = 8;

// cuda_blocks(0) is 0, and a zero-block grid is an invalid launch
// configuration. Every wrapper below early-returns its (empty) result
// before launching when n == 0, so valid zero-element tensors behave
// the same on CUDA as on CPU.
static int cuda_blocks(int64_t n, int threads = kThreads) {
    return static_cast<int>((n + threads - 1) / threads);
}

// Float32 forward kernels.

__global__ void neg_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = -src[i];
}

__global__ void relu_kernel(const float* __restrict__ src,
                              float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = src[i];
        dst[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void sigmoid_kernel(const float* __restrict__ src,
                               float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
}

__global__ void tanh_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = tanhf(src[i]);
}

__global__ void leaky_relu_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int64_t n, float negative_slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = src[i];
        dst[i] = v > 0.0f ? v : negative_slope * v;
    }
}

__global__ void gelu_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, int64_t n) {
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = src[i];
        dst[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}

__global__ void exp_kernel(const float* __restrict__ src,
                           float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = expf(src[i]);
}

__global__ void sqrt_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = sqrtf(src[i]);
}

__global__ void reciprocal_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = 1.0f / src[i];
}

// Float16 forward kernels use float32 math and store float16 results.

__global__ void neg_kernel_half(const __half* __restrict__ src,
                                 __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(-__half2float(src[i]));
}

__global__ void relu_kernel_half(const __half* __restrict__ src,
                                  __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(src[i]);
        dst[i] = __float2half(v > 0.0f ? v : 0.0f);
    }
}

__global__ void sigmoid_kernel_half(const __half* __restrict__ src,
                                    __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __half2float(src[i]);
        dst[i] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void tanh_kernel_half(const __half* __restrict__ src,
                                  __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(tanhf(__half2float(src[i])));
}

__global__ void leaky_relu_kernel_half(const __half* __restrict__ src,
                                       __half* __restrict__ dst,
                                       int64_t n, float negative_slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(src[i]);
        dst[i] = __float2half(v > 0.0f ? v : negative_slope * v);
    }
}

__global__ void gelu_kernel_half(const __half* __restrict__ src,
                                  __half* __restrict__ dst, int64_t n) {
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __half2float(src[i]);
        dst[i] = __float2half(0.5f * x * (1.0f + erff(x * inv_sqrt2)));
    }
}

__global__ void exp_kernel_half(const __half* __restrict__ src,
                                 __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(expf(__half2float(src[i])));
}

__global__ void sqrt_kernel_half(const __half* __restrict__ src,
                                  __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(sqrtf(__half2float(src[i])));
}

__global__ void reciprocal_kernel_half(const __half* __restrict__ src,
                                       __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(1.0f / __half2float(src[i]));
}

// BFloat16 forward kernels use float32 math and store bfloat16 results.

__global__ void neg_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                     __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(-__bfloat162float(src[i]));
}

__global__ void relu_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __bfloat162float(src[i]);
        dst[i] = __float2bfloat16(v > 0.0f ? v : 0.0f);
    }
}

__global__ void sigmoid_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __bfloat162float(src[i]);
        dst[i] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void tanh_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(tanhf(__bfloat162float(src[i])));
}

__global__ void leaky_relu_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                            __nv_bfloat16* __restrict__ dst,
                                            int64_t n, float negative_slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __bfloat162float(src[i]);
        dst[i] = __float2bfloat16(v > 0.0f ? v : negative_slope * v);
    }
}

__global__ void gelu_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst, int64_t n) {
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __bfloat162float(src[i]);
        dst[i] = __float2bfloat16(0.5f * x * (1.0f + erff(x * inv_sqrt2)));
    }
}

__global__ void exp_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                     __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(expf(__bfloat162float(src[i])));
}

__global__ void sqrt_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(sqrtf(__bfloat162float(src[i])));
}

__global__ void reciprocal_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                            __nv_bfloat16* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(1.0f / __bfloat162float(src[i]));
}

// Float32 dropout mask generation.

__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ float uniform01_from_u64(uint64_t x) {
    const uint32_t top24 = static_cast<uint32_t>(x >> 40);
    return static_cast<float>(top24) * (1.0f / 16777216.0f);
}

__global__ void dropout_mask_kernel(float* __restrict__ mask,
                                    int64_t n, float p, uint64_t seed) {
    const float keep = 1.0f - p;
    const float scale = 1.0f / keep;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const uint64_t bits = splitmix64(seed + static_cast<uint64_t>(i));
        const float r = uniform01_from_u64(bits);
        mask[i] = r >= p ? scale : 0.0f;
    }
}

// Host-side forward wrappers with dtype dispatch.

Tensor cuda_neg(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        neg_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        neg_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        neg_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_relu(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        relu_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        relu_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        relu_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_sigmoid(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        sigmoid_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        sigmoid_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        sigmoid_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_tanh(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        tanh_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        tanh_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        tanh_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_leaky_relu(const Tensor& t, float negative_slope) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        leaky_relu_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n, negative_slope);
    } else if (t.dtype() == DType::BFloat16) {
        leaky_relu_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n, negative_slope);
    } else {
        leaky_relu_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n, negative_slope);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_gelu(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        gelu_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        gelu_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        gelu_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_exp(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        exp_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        exp_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        exp_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_dropout_mask(const Tensor& t, float p, uint64_t seed) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    if (p < 0.0f || p >= 1.0f)
        throw std::runtime_error("cuda_dropout_mask: p must be in [0, 1)");
    const int64_t n = t.numel();
    Tensor mask(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return mask;
    dropout_mask_kernel<<<cuda_blocks(n), kThreads>>>(
        static_cast<float*>(mask.data()), n, p, seed);
    CUDA_CHECK(cudaGetLastError());
    return mask;
}

Tensor cuda_sqrt(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        sqrt_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        sqrt_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        sqrt_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_reciprocal(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;
    if (t.dtype() == DType::Float16) {
        reciprocal_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()), n);
    } else if (t.dtype() == DType::BFloat16) {
        reciprocal_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        reciprocal_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Float32 backward kernels.

__global__ void relu_backward_kernel(const float* __restrict__ input,
                                      const float* __restrict__ grad_output,
                                      float* __restrict__ grad_input,
                                      int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ input,
                                        const float* __restrict__ grad_output,
                                        float* __restrict__ grad_input,
                                        int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float s = 1.0f / (1.0f + expf(-input[i]));
        grad_input[i] = grad_output[i] * s * (1.0f - s);
    }
}

__global__ void tanh_backward_kernel(const float* __restrict__ input,
                                     const float* __restrict__ grad_output,
                                     float* __restrict__ grad_input,
                                     int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float y = tanhf(input[i]);
        grad_input[i] = grad_output[i] * (1.0f - y * y);
    }
}

__global__ void leaky_relu_backward_kernel(const float* __restrict__ input,
                                           const float* __restrict__ grad_output,
                                           float* __restrict__ grad_input,
                                           int64_t n, float negative_slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : negative_slope * grad_output[i];
}

__global__ void gelu_backward_kernel(const float* __restrict__ input,
                                     const float* __restrict__ grad_output,
                                     float* __restrict__ grad_input,
                                     int64_t n) {
    constexpr float inv_sqrt2   = 0.7071067811865475f;
    constexpr float inv_sqrt2pi = 0.3989422804014327f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x       = input[i];
        float cdf     = 0.5f * (1.0f + erff(x * inv_sqrt2));
        float pdf_term = x * inv_sqrt2pi * expf(-0.5f * x * x);
        grad_input[i] = grad_output[i] * (cdf + pdf_term);
    }
}

__global__ void sqrt_backward_kernel(const float* __restrict__ input,
                                     const float* __restrict__ grad_output,
                                     float* __restrict__ grad_input,
                                     int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = grad_output[i] * 0.5f / sqrtf(input[i]);
}

__global__ void reciprocal_backward_kernel(const float* __restrict__ input,
                                           const float* __restrict__ grad_output,
                                           float* __restrict__ grad_input,
                                           int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = input[i];
        grad_input[i] = -grad_output[i] / (x * x);
    }
}

__global__ void exp_backward_kernel(const float* __restrict__ input,
                                    const float* __restrict__ grad_output,
                                    float* __restrict__ grad_input,
                                    int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = grad_output[i] * expf(input[i]);
}

// Float16 backward kernels use float32 math.

__global__ void relu_backward_kernel_half(const __half* __restrict__ input,
                                           const __half* __restrict__ grad_output,
                                           __half* __restrict__ grad_input,
                                           int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __half2float(input[i]);
        float g = __half2float(grad_output[i]);
        grad_input[i] = __float2half(x > 0.0f ? g : 0.0f);
    }
}

__global__ void sigmoid_backward_kernel_half(const __half* __restrict__ input,
                                              const __half* __restrict__ grad_output,
                                              __half* __restrict__ grad_input,
                                              int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float s = 1.0f / (1.0f + expf(-__half2float(input[i])));
        grad_input[i] = __float2half(__half2float(grad_output[i]) * s * (1.0f - s));
    }
}

__global__ void tanh_backward_kernel_half(const __half* __restrict__ input,
                                          const __half* __restrict__ grad_output,
                                          __half* __restrict__ grad_input,
                                          int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float y = tanhf(__half2float(input[i]));
        grad_input[i] = __float2half(__half2float(grad_output[i]) * (1.0f - y * y));
    }
}

__global__ void leaky_relu_backward_kernel_half(const __half* __restrict__ input,
                                                 const __half* __restrict__ grad_output,
                                                 __half* __restrict__ grad_input,
                                                 int64_t n, float negative_slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __half2float(input[i]);
        float g = __half2float(grad_output[i]);
        grad_input[i] = __float2half(x > 0.0f ? g : negative_slope * g);
    }
}

__global__ void gelu_backward_kernel_half(const __half* __restrict__ input,
                                          const __half* __restrict__ grad_output,
                                          __half* __restrict__ grad_input,
                                          int64_t n) {
    constexpr float inv_sqrt2   = 0.7071067811865475f;
    constexpr float inv_sqrt2pi = 0.3989422804014327f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x        = __half2float(input[i]);
        float cdf      = 0.5f * (1.0f + erff(x * inv_sqrt2));
        float pdf_term = x * inv_sqrt2pi * expf(-0.5f * x * x);
        grad_input[i]  = __float2half(__half2float(grad_output[i]) * (cdf + pdf_term));
    }
}

__global__ void sqrt_backward_kernel_half(const __half* __restrict__ input,
                                          const __half* __restrict__ grad_output,
                                          __half* __restrict__ grad_input,
                                          int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = __float2half(
            __half2float(grad_output[i]) * 0.5f / sqrtf(__half2float(input[i])));
}

__global__ void reciprocal_backward_kernel_half(const __half* __restrict__ input,
                                                 const __half* __restrict__ grad_output,
                                                 __half* __restrict__ grad_input,
                                                 int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __half2float(input[i]);
        grad_input[i] = __float2half(-__half2float(grad_output[i]) / (x * x));
    }
}

__global__ void exp_backward_kernel_half(const __half* __restrict__ input,
                                         const __half* __restrict__ grad_output,
                                         __half* __restrict__ grad_input,
                                         int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = __float2half(
            __half2float(grad_output[i]) * expf(__half2float(input[i])));
}

// BFloat16 backward kernels use float32 math.

__global__ void relu_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                               const __nv_bfloat16* __restrict__ grad_output,
                                               __nv_bfloat16* __restrict__ grad_input,
                                               int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(grad_output[i]);
        grad_input[i] = __float2bfloat16(x > 0.0f ? g : 0.0f);
    }
}

__global__ void sigmoid_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                                  const __nv_bfloat16* __restrict__ grad_output,
                                                  __nv_bfloat16* __restrict__ grad_input,
                                                  int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float s = 1.0f / (1.0f + expf(-__bfloat162float(input[i])));
        grad_input[i] = __float2bfloat16(__bfloat162float(grad_output[i]) * s * (1.0f - s));
    }
}

__global__ void tanh_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                               const __nv_bfloat16* __restrict__ grad_output,
                                               __nv_bfloat16* __restrict__ grad_input,
                                               int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float y = tanhf(__bfloat162float(input[i]));
        grad_input[i] = __float2bfloat16(__bfloat162float(grad_output[i]) * (1.0f - y * y));
    }
}

__global__ void leaky_relu_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                                     const __nv_bfloat16* __restrict__ grad_output,
                                                     __nv_bfloat16* __restrict__ grad_input,
                                                     int64_t n, float negative_slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(grad_output[i]);
        grad_input[i] = __float2bfloat16(x > 0.0f ? g : negative_slope * g);
    }
}

__global__ void gelu_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                               const __nv_bfloat16* __restrict__ grad_output,
                                               __nv_bfloat16* __restrict__ grad_input,
                                               int64_t n) {
    constexpr float inv_sqrt2   = 0.7071067811865475f;
    constexpr float inv_sqrt2pi = 0.3989422804014327f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x        = __bfloat162float(input[i]);
        float cdf      = 0.5f * (1.0f + erff(x * inv_sqrt2));
        float pdf_term = x * inv_sqrt2pi * expf(-0.5f * x * x);
        grad_input[i]  = __float2bfloat16(__bfloat162float(grad_output[i]) * (cdf + pdf_term));
    }
}

__global__ void sqrt_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                               const __nv_bfloat16* __restrict__ grad_output,
                                               __nv_bfloat16* __restrict__ grad_input,
                                               int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = __float2bfloat16(
            __bfloat162float(grad_output[i]) * 0.5f / sqrtf(__bfloat162float(input[i])));
}

__global__ void reciprocal_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                                     const __nv_bfloat16* __restrict__ grad_output,
                                                     __nv_bfloat16* __restrict__ grad_input,
                                                     int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float x = __bfloat162float(input[i]);
        grad_input[i] = __float2bfloat16(-__bfloat162float(grad_output[i]) / (x * x));
    }
}

__global__ void exp_backward_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                              const __nv_bfloat16* __restrict__ grad_output,
                                              __nv_bfloat16* __restrict__ grad_input,
                                              int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        grad_input[i] = __float2bfloat16(
            __bfloat162float(grad_output[i]) * expf(__bfloat162float(input[i])));
}

// Host-side backward wrappers with dtype dispatch.

Tensor cuda_relu_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        relu_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        relu_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        relu_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_sigmoid_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        sigmoid_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        sigmoid_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        sigmoid_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_tanh_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        tanh_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        tanh_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        tanh_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_leaky_relu_backward(const Tensor& input, const Tensor& grad_output,
                                float negative_slope) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        leaky_relu_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n, negative_slope);
    } else if (input.dtype() == DType::BFloat16) {
        leaky_relu_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n, negative_slope);
    } else {
        leaky_relu_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n, negative_slope);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_gelu_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        gelu_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        gelu_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        gelu_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_sqrt_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        sqrt_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        sqrt_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        sqrt_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_reciprocal_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        reciprocal_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        reciprocal_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        reciprocal_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_exp_backward(const Tensor& input, const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int64_t n = input.numel();
    Tensor out(std::make_shared<TensorImpl>(input.shape(), input.dtype(), input.device()));
    if (n == 0) return out;
    if (input.dtype() == DType::Float16) {
        exp_backward_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(out.data()), n);
    } else if (input.dtype() == DType::BFloat16) {
        exp_backward_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(out.data()), n);
    } else {
        exp_backward_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(out.data()), n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Contiguous-copy and finite-check helpers.

struct ContiguousInfo {
    int64_t shape[kMaxDims];
    int64_t strides[kMaxDims];
    int     ndim;
};

__device__ int64_t logical_offset(int64_t linear, const ContiguousInfo& info) {
    int64_t off = 0;
    for (int d = info.ndim - 1; d >= 0; --d) {
        int64_t coord = linear % info.shape[d];
        linear /= info.shape[d];
        off += coord * info.strides[d];
    }
    return off;
}

__global__ void isfinite_kernel_float32(const float* __restrict__ src,
                                        int* __restrict__ all_finite,
                                        ContiguousInfo info,
                                        int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        if (!isfinite(src[off]))
            atomicExch(all_finite, 0);
    }
}

__global__ void isfinite_kernel_float64(const double* __restrict__ src,
                                        int* __restrict__ all_finite,
                                        ContiguousInfo info,
                                        int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        if (!isfinite(src[off]))
            atomicExch(all_finite, 0);
    }
}

__global__ void isfinite_kernel_float16(const __half* __restrict__ src,
                                        int* __restrict__ all_finite,
                                        ContiguousInfo info,
                                        int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        if (!isfinite(__half2float(src[off])))
            atomicExch(all_finite, 0);
    }
}

__global__ void isfinite_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                         int* __restrict__ all_finite,
                                         ContiguousInfo info,
                                         int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        if (!isfinite(__bfloat162float(src[off])))
            atomicExch(all_finite, 0);
    }
}

bool cuda_isfinite_all(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const DType dt = t.dtype();
    if (dt != DType::Float32 &&
        dt != DType::Float64 &&
        dt != DType::Float16 &&
        dt != DType::BFloat16) {
        return true;
    }
    const int64_t n = t.numel();
    if (n == 0) return true;
    const int nd = static_cast<int>(t.ndim());
    if (nd > kMaxDims)
        throw std::runtime_error(
            "cuda_isfinite_all: tensor rank " + std::to_string(nd) +
            " exceeds kMaxDims=" + std::to_string(kMaxDims));

    ContiguousInfo info{};
    info.ndim = nd;
    for (int d = 0; d < nd; ++d) {
        info.shape[d] = t.shape()[d];
        info.strides[d] = t.strides()[d];
    }

    int* device_flag = nullptr;
    int host_flag = 1;
    device_flag = static_cast<int*>(memory_pool_allocate(sizeof(int), t.device()));
    try {
        CUDA_CHECK(cudaMemcpy(device_flag, &host_flag, sizeof(int),
                              cudaMemcpyHostToDevice));

        const int blocks = std::min(cuda_blocks(n), 4096);
        if (dt == DType::Float32) {
            isfinite_kernel_float32<<<blocks, kThreads>>>(
                static_cast<const float*>(t.data()), device_flag, info, n);
        } else if (dt == DType::Float64) {
            isfinite_kernel_float64<<<blocks, kThreads>>>(
                static_cast<const double*>(t.data()), device_flag, info, n);
        } else if (dt == DType::Float16) {
            isfinite_kernel_float16<<<blocks, kThreads>>>(
                static_cast<const __half*>(t.data()), device_flag, info, n);
        } else {
            isfinite_kernel_bfloat16<<<blocks, kThreads>>>(
                static_cast<const __nv_bfloat16*>(t.data()), device_flag, info, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&host_flag, device_flag, sizeof(int),
                              cudaMemcpyDeviceToHost));
        memory_pool_deallocate(device_flag, t.device());
    } catch (...) {
        memory_pool_deallocate(device_flag, t.device());
        throw;
    }
    return host_flag != 0;
}

__global__ void unscale_check_kernel_float32(const float* __restrict__ src,
                                             float* __restrict__ dst,
                                             int* __restrict__ all_finite,
                                             ContiguousInfo info,
                                             int64_t n,
                                             float inv_scale) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        const float value = src[off] * inv_scale;
        dst[i] = value;
        if (!isfinite(value))
            atomicExch(all_finite, 0);
    }
}

__global__ void unscale_check_kernel_float16(const __half* __restrict__ src,
                                             float* __restrict__ dst,
                                             int* __restrict__ all_finite,
                                             ContiguousInfo info,
                                             int64_t n,
                                             float inv_scale) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        const float value = __half2float(src[off]) * inv_scale;
        dst[i] = value;
        if (!isfinite(value))
            atomicExch(all_finite, 0);
    }
}

__global__ void unscale_check_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                              float* __restrict__ dst,
                                              int* __restrict__ all_finite,
                                              ContiguousInfo info,
                                              int64_t n,
                                              float inv_scale) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        const int64_t off = logical_offset(i, info);
        const float value = __bfloat162float(src[off]) * inv_scale;
        dst[i] = value;
        if (!isfinite(value))
            atomicExch(all_finite, 0);
    }
}

std::pair<Tensor, bool> cuda_unscale_to_float32_and_check(const Tensor& src,
                                                          float inv_scale) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(src.device());

    const DType dt = src.dtype();
    if (dt != DType::Float32 &&
        dt != DType::Float16 &&
        dt != DType::BFloat16) {
        throw std::runtime_error(
            "cuda_unscale_to_float32_and_check: only float32/float16/bfloat16 supported");
    }
    const int64_t n = src.numel();
    const int nd = static_cast<int>(src.ndim());
    if (nd > kMaxDims)
        throw std::runtime_error(
            "cuda_unscale_to_float32_and_check: tensor rank " + std::to_string(nd) +
            " exceeds kMaxDims=" + std::to_string(kMaxDims));

    Tensor out(std::make_shared<TensorImpl>(src.shape(), DType::Float32, src.device()));
    if (n == 0) return {out, true};

    ContiguousInfo info{};
    info.ndim = nd;
    for (int d = 0; d < nd; ++d) {
        info.shape[d] = src.shape()[d];
        info.strides[d] = src.strides()[d];
    }

    int* device_flag = static_cast<int*>(memory_pool_allocate(sizeof(int), src.device()));
    int host_flag = 1;
    try {
        CUDA_CHECK(cudaMemcpy(device_flag, &host_flag, sizeof(int),
                              cudaMemcpyHostToDevice));

        const int blocks = std::min(cuda_blocks(n), 4096);
        if (dt == DType::Float32) {
            unscale_check_kernel_float32<<<blocks, kThreads>>>(
                static_cast<const float*>(src.data()),
                static_cast<float*>(out.data()),
                device_flag,
                info,
                n,
                inv_scale);
        } else if (dt == DType::Float16) {
            unscale_check_kernel_float16<<<blocks, kThreads>>>(
                static_cast<const __half*>(src.data()),
                static_cast<float*>(out.data()),
                device_flag,
                info,
                n,
                inv_scale);
        } else {
            unscale_check_kernel_bfloat16<<<blocks, kThreads>>>(
                static_cast<const __nv_bfloat16*>(src.data()),
                static_cast<float*>(out.data()),
                device_flag,
                info,
                n,
                inv_scale);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&host_flag, device_flag, sizeof(int),
                              cudaMemcpyDeviceToHost));
        memory_pool_deallocate(device_flag, src.device());
    } catch (...) {
        memory_pool_deallocate(device_flag, src.device());
        throw;
    }
    return {out, host_flag != 0};
}

__global__ void gather_contiguous_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst,
                                          ContiguousInfo info,
                                          int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t rem = i;
        int64_t off = 0;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem /= info.shape[d];
            off += coord * info.strides[d];
        }
        dst[i] = src[off];
    }
}

__global__ void gather_contiguous_kernel_half(const __half* __restrict__ src,
                                              __half* __restrict__ dst,
                                              ContiguousInfo info,
                                              int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t rem = i;
        int64_t off = 0;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem /= info.shape[d];
            off += coord * info.strides[d];
        }
        dst[i] = src[off];
    }
}

__global__ void gather_contiguous_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                                   __nv_bfloat16* __restrict__ dst,
                                                   ContiguousInfo info,
                                                   int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t rem = i;
        int64_t off = 0;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem /= info.shape[d];
            off += coord * info.strides[d];
        }
        dst[i] = src[off];
    }
}

Tensor cuda_make_contiguous(const Tensor& t) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const int64_t n = t.numel();
    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    if (n == 0) return out;

    if (t.is_contiguous()) {
        CUDA_CHECK(cudaMemcpy(out.data(), t.data(),
                              static_cast<size_t>(n) * dtype_size(t.dtype()),
                              cudaMemcpyDeviceToDevice));
        return out;
    }

    if (t.dtype() != DType::Float32 &&
        t.dtype() != DType::Float16 &&
        t.dtype() != DType::BFloat16)
        throw std::runtime_error(
            "cuda_make_contiguous: non-contiguous CUDA copies support only float32/float16/bfloat16");

    const int nd = static_cast<int>(t.ndim());
    if (nd > kMaxDims)
        throw std::runtime_error(
            "cuda_make_contiguous: tensor rank " + std::to_string(nd) +
            " exceeds kMaxDims=" + std::to_string(kMaxDims));

    ContiguousInfo info{};
    info.ndim = nd;
    for (int d = 0; d < nd; ++d) {
        info.shape[d]   = t.shape()[d];
        info.strides[d] = t.strides()[d];
    }

    if (t.dtype() == DType::Float16) {
        gather_contiguous_kernel_half<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()),
            info, n);
    } else if (t.dtype() == DType::BFloat16) {
        gather_contiguous_kernel_bfloat16<<<cuda_blocks(n), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            info, n);
    } else {
        gather_contiguous_kernel<<<cuda_blocks(n), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()),
            info, n);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

}  // namespace orthon
