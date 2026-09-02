#include <orthon/ops/cuda/binary_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

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

static int cuda_blocks(int64_t n, int threads = kThreads) {
    return static_cast<int>((n + threads - 1) / threads);
}

static void require_cuda_float32_contiguous(const Tensor& t,
                                            const char* op,
                                            const char* name) {
    if (!t.device().is_cuda())
        throw std::runtime_error(std::string(op) + ": input '" + name +
                                 "' must be on CUDA");
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(op) + ": input '" + name +
                                 "' must be float32");
    if (!t.is_contiguous())
        throw std::runtime_error(std::string(op) + ": input '" + name +
                                 "' is not contiguous; call .to(device) first");
}

// Like require_cuda_float32_contiguous but also accepts float16 and bfloat16.
static void require_cuda_fp_contiguous(const Tensor& t,
                                       const char* op,
                                       const char* name) {
    if (!t.device().is_cuda())
        throw std::runtime_error(std::string(op) + ": input '" + name +
                                 "' must be on CUDA");
    if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
        throw std::runtime_error(std::string(op) + ": input '" + name +
                                 "' must be float32, float16, or bfloat16");
    if (!t.is_contiguous())
        throw std::runtime_error(std::string(op) + ": input '" + name +
                                 "' is not contiguous; call .to(device) first");
}

static void require_same_cuda_device(const Tensor& a,
                                     const Tensor& b,
                                     const char* op) {
    if (a.device() != b.device())
        throw std::runtime_error(std::string(op) + ": device mismatch");
}

// Passed by value into the kernel to avoid extra cudaMalloc for shape/stride arrays.
struct BcastInfo {
    int64_t shape[kMaxDims];
    int64_t a_strides[kMaxDims];
    int64_t b_strides[kMaxDims];
    int     ndim;
};

// Grid-stride loop over output linear indices. For each index, unpacks to
// multi-dim coords and maps to input element offsets via broadcast strides.
// Broadcast dims have stride 0, so the same element is reused across that dim.
__global__ void add_broadcast_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out,
                                      BcastInfo info,
                                      int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t ia = 0, ib = 0;
        int64_t rem = i;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem           = rem / info.shape[d];
            ia += coord * info.a_strides[d];
            ib += coord * info.b_strides[d];
        }
        out[i] = a[ia] + b[ib];
    }
}

static std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) r += ", "; r += std::to_string(s[i]); }
    return r + "]";
}

static std::vector<int64_t> cuda_broadcast_shape(const std::vector<int64_t>& a,
                                                   const std::vector<int64_t>& b) {
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    const int nr = std::max(na, nb);
    std::vector<int64_t> out(nr);
    for (int i = 0; i < nr; ++i) {
        int64_t da = (i < nr - na) ? 1 : a[i - (nr - na)];
        int64_t db = (i < nr - nb) ? 1 : b[i - (nr - nb)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error(
                "cuda_add: cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b));
        // Not max(): a size-1 dim stretches to the OTHER size, including 0
        // ([0,3] + [3] is [0,3]); max(0, 1) would fabricate an element and
        // read out of bounds from the empty operand.
        out[i] = (da == 1) ? db : da;
    }
    return out;
}

// Expands strides to target_ndim with leading zeros; sets stride=0 for any dim
// where shape is 1 (the broadcast dim repeats the same element).
static std::vector<int64_t> cuda_to_broadcast_strides(const std::vector<int64_t>& shape,
                                                        const std::vector<int64_t>& strides,
                                                        int target_ndim) {
    const int n = static_cast<int>(shape.size());
    std::vector<int64_t> out(target_ndim, 0);
    for (int i = 0; i < n; ++i) {
        int out_i   = target_ndim - n + i;
        out[out_i]  = (shape[i] == 1) ? 0 : strides[i];
    }
    return out;
}

// Float16 broadcast add: upcast both inputs to float, add, downcast.
__global__ void add_broadcast_kernel_half(const __half* __restrict__ a,
                                          const __half* __restrict__ b,
                                          __half* __restrict__ out,
                                          BcastInfo info,
                                          int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t ia = 0, ib = 0;
        int64_t rem = i;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem           = rem / info.shape[d];
            ia += coord * info.a_strides[d];
            ib += coord * info.b_strides[d];
        }
        out[i] = __float2half(__half2float(a[ia]) + __half2float(b[ib]));
    }
}

// BFloat16 broadcast add: upcast both inputs to float, add, downcast.
__global__ void add_broadcast_kernel_bfloat16(const __nv_bfloat16* __restrict__ a,
                                               const __nv_bfloat16* __restrict__ b,
                                               __nv_bfloat16* __restrict__ out,
                                               BcastInfo info,
                                               int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t ia = 0, ib = 0;
        int64_t rem = i;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem           = rem / info.shape[d];
            ia += coord * info.a_strides[d];
            ib += coord * info.b_strides[d];
        }
        out[i] = __float2bfloat16(__bfloat162float(a[ia]) + __bfloat162float(b[ib]));
    }
}

Tensor cuda_add(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    require_cuda_fp_contiguous(a, "cuda_add", "a");
    require_cuda_fp_contiguous(b, "cuda_add", "b");
    require_same_cuda_device(a, b, "cuda_add");
    if (a.dtype() != b.dtype())
        throw std::runtime_error("cuda_add: dtype mismatch");

    const auto out_shape = cuda_broadcast_shape(a.shape(), b.shape());
    const int  nr        = static_cast<int>(out_shape.size());

    if (nr > kMaxDims)
        throw std::runtime_error(
            "cuda_add: tensors with more than " + std::to_string(kMaxDims) +
            " dimensions are not supported");

    const auto sa = cuda_to_broadcast_strides(a.shape(), a.strides(), nr);
    const auto sb = cuda_to_broadcast_strides(b.shape(), b.strides(), nr);

    BcastInfo info{};
    info.ndim = nr;
    for (int i = 0; i < nr; ++i) {
        info.shape[i]     = out_shape[i];
        info.a_strides[i] = sa[i];
        info.b_strides[i] = sb[i];
    }

    Tensor  out   = Tensor(std::make_shared<TensorImpl>(out_shape, a.dtype(), a.device()));
    int64_t n_out = out.numel();
    if (n_out == 0) return out;

    if (a.dtype() == DType::Float16) {
        add_broadcast_kernel_half<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(out.data()),
            info, n_out);
    } else if (a.dtype() == DType::BFloat16) {
        add_broadcast_kernel_bfloat16<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            info, n_out);
    } else {
        add_broadcast_kernel<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            info, n_out);
    }

    CUDA_CHECK(cudaGetLastError());
    return out;
}

// CUDA mul with broadcasting.

__global__ void mul_broadcast_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out,
                                      BcastInfo info,
                                      int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t ia = 0, ib = 0;
        int64_t rem = i;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem           = rem / info.shape[d];
            ia += coord * info.a_strides[d];
            ib += coord * info.b_strides[d];
        }
        out[i] = a[ia] * b[ib];
    }
}

// Float16 broadcast mul.
__global__ void mul_broadcast_kernel_half(const __half* __restrict__ a,
                                          const __half* __restrict__ b,
                                          __half* __restrict__ out,
                                          BcastInfo info,
                                          int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t ia = 0, ib = 0;
        int64_t rem = i;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem           = rem / info.shape[d];
            ia += coord * info.a_strides[d];
            ib += coord * info.b_strides[d];
        }
        out[i] = __float2half(__half2float(a[ia]) * __half2float(b[ib]));
    }
}

// BFloat16 broadcast mul.
__global__ void mul_broadcast_kernel_bfloat16(const __nv_bfloat16* __restrict__ a,
                                               const __nv_bfloat16* __restrict__ b,
                                               __nv_bfloat16* __restrict__ out,
                                               BcastInfo info,
                                               int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t ia = 0, ib = 0;
        int64_t rem = i;
        for (int d = info.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % info.shape[d];
            rem           = rem / info.shape[d];
            ia += coord * info.a_strides[d];
            ib += coord * info.b_strides[d];
        }
        out[i] = __float2bfloat16(__bfloat162float(a[ia]) * __bfloat162float(b[ib]));
    }
}

Tensor cuda_mul(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    require_cuda_fp_contiguous(a, "cuda_mul", "a");
    require_cuda_fp_contiguous(b, "cuda_mul", "b");
    require_same_cuda_device(a, b, "cuda_mul");
    if (a.dtype() != b.dtype())
        throw std::runtime_error("cuda_mul: dtype mismatch");

    const auto out_shape = cuda_broadcast_shape(a.shape(), b.shape());
    const int  nr        = static_cast<int>(out_shape.size());

    if (nr > kMaxDims)
        throw std::runtime_error(
            "cuda_mul: tensors with more than " + std::to_string(kMaxDims) +
            " dimensions are not supported");

    const auto sa = cuda_to_broadcast_strides(a.shape(), a.strides(), nr);
    const auto sb = cuda_to_broadcast_strides(b.shape(), b.strides(), nr);

    BcastInfo info{};
    info.ndim = nr;
    for (int i = 0; i < nr; ++i) {
        info.shape[i]     = out_shape[i];
        info.a_strides[i] = sa[i];
        info.b_strides[i] = sb[i];
    }

    Tensor  out   = Tensor(std::make_shared<TensorImpl>(out_shape, a.dtype(), a.device()));
    int64_t n_out = out.numel();
    if (n_out == 0) return out;

    if (a.dtype() == DType::Float16) {
        mul_broadcast_kernel_half<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(out.data()),
            info, n_out);
    } else if (a.dtype() == DType::BFloat16) {
        mul_broadcast_kernel_bfloat16<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            info, n_out);
    } else {
        mul_broadcast_kernel<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            info, n_out);
    }

    CUDA_CHECK(cudaGetLastError());
    return out;
}

// CUDA unbroadcast reduces a gradient to target_shape by summing.

struct UnbroadcastInfo {
    int64_t src_shape[kMaxDims];    // shape of the incoming grad tensor
    int64_t padded_shape[kMaxDims]; // target shape left-padded with 1s to match src rank
    int     ndim;
};

// Each thread owns one element of src and atomically adds it to the
// corresponding element of dst (which has the padded target shape).
// Broadcast dimensions (padded_shape[d]==1) collapse their coordinate to 0.
__global__ void unbroadcast_kernel(const float* __restrict__ src,
                                    float* __restrict__       dst,
                                    UnbroadcastInfo           info,
                                    int64_t                   n_src) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_src;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = i;
        int64_t coords[kMaxDims];
        for (int d = info.ndim - 1; d >= 0; --d) {
            coords[d] = tmp % info.src_shape[d];
            tmp      /= info.src_shape[d];
        }
        int64_t dst_flat = 0;
        for (int d = 0; d < info.ndim; ++d) {
            int64_t c = (info.padded_shape[d] == 1) ? 0 : coords[d];
            dst_flat  = dst_flat * info.padded_shape[d] + c;
        }
        atomicAdd(&dst[dst_flat], src[i]);
    }
}

// Float16 unbroadcast: upcast src to float32, atomicAdd into a float32 buffer.
// Using float32 atomicAdd avoids atomicAdd(__half*) which requires sm_70+.
__global__ void unbroadcast_kernel_half(const __half* __restrict__ src,
                                         float* __restrict__        dst,
                                         UnbroadcastInfo            info,
                                         int64_t                    n_src) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_src;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = i;
        int64_t coords[kMaxDims];
        for (int d = info.ndim - 1; d >= 0; --d) {
            coords[d] = tmp % info.src_shape[d];
            tmp      /= info.src_shape[d];
        }
        int64_t dst_flat = 0;
        for (int d = 0; d < info.ndim; ++d) {
            int64_t c = (info.padded_shape[d] == 1) ? 0 : coords[d];
            dst_flat  = dst_flat * info.padded_shape[d] + c;
        }
        atomicAdd(&dst[dst_flat], __half2float(src[i]));
    }
}

// Cast a float32 buffer to __half (separate output).
__global__ void cast_f32_to_f16_kernel(const float* __restrict__ src,
                                        __half* __restrict__      dst,
                                        int64_t                   n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2half(src[i]);
}

// BFloat16 unbroadcast: upcast src to float32, atomicAdd into float32 buffer.
// Using float32 atomicAdd avoids atomicAdd(__nv_bfloat16*) which requires sm_90+.
__global__ void unbroadcast_kernel_bfloat16(const __nv_bfloat16* __restrict__ src,
                                             float* __restrict__               dst,
                                             UnbroadcastInfo                   info,
                                             int64_t                           n_src) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_src;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = i;
        int64_t coords[kMaxDims];
        for (int d = info.ndim - 1; d >= 0; --d) {
            coords[d] = tmp % info.src_shape[d];
            tmp      /= info.src_shape[d];
        }
        int64_t dst_flat = 0;
        for (int d = 0; d < info.ndim; ++d) {
            int64_t c = (info.padded_shape[d] == 1) ? 0 : coords[d];
            dst_flat  = dst_flat * info.padded_shape[d] + c;
        }
        atomicAdd(&dst[dst_flat], __bfloat162float(src[i]));
    }
}

// Cast a float32 buffer to __nv_bfloat16 (separate output).
__global__ void cast_f32_to_bf16_kernel(const float* __restrict__    src,
                                         __nv_bfloat16* __restrict__  dst,
                                         int64_t                      n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        dst[i] = __float2bfloat16(src[i]);
}

Tensor cuda_unbroadcast(const Tensor& grad,
                        const std::vector<int64_t>& target_shape) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(grad.device());

    require_cuda_fp_contiguous(grad, "cuda_unbroadcast", "grad");

    if (grad.shape() == target_shape) return grad;

    const int grad_ndim = static_cast<int>(grad.shape().size());
    const int tgt_ndim  = static_cast<int>(target_shape.size());

    if (grad_ndim > kMaxDims)
        throw std::runtime_error("cuda_unbroadcast: tensor rank exceeds kMaxDims");
    if (tgt_ndim > grad_ndim)
        throw std::runtime_error(
            "cuda_unbroadcast: target rank cannot exceed grad rank");

    // An empty grad sums to zero everywhere; launching the accumulate kernel
    // with a zero-block grid would be an invalid configuration.
    if (grad.numel() == 0) {
        Tensor result(std::make_shared<TensorImpl>(target_shape, grad.dtype(), grad.device()));
        const size_t nbytes =
            static_cast<size_t>(result.numel()) * dtype_size(grad.dtype());
        if (nbytes > 0)
            CUDA_CHECK(cudaMemset(result.data(), 0, nbytes));
        return result;
    }

    // Left-pad target_shape with 1s to match grad rank.
    std::vector<int64_t> padded(grad_ndim, 1);
    for (int i = 0; i < tgt_ndim; ++i)
        padded[grad_ndim - tgt_ndim + i] = target_shape[i];

    int64_t n_out = 1;
    for (int64_t d : padded) n_out *= d;

    UnbroadcastInfo info{};
    info.ndim = grad_ndim;
    for (int i = 0; i < grad_ndim; ++i) {
        info.src_shape[i]    = grad.shape()[i];
        info.padded_shape[i] = padded[i];
    }

    if (grad.dtype() == DType::Float16) {
        // Accumulate into a float32 buffer (avoids atomicAdd(__half*) / sm_70 req).
        Tensor f32_buf(std::make_shared<TensorImpl>(padded, DType::Float32, grad.device()));
        CUDA_CHECK(cudaMemset(f32_buf.data(), 0, static_cast<size_t>(n_out) * sizeof(float)));

        unbroadcast_kernel_half<<<cuda_blocks(grad.numel()), kThreads>>>(
            static_cast<const __half*>(grad.data()),
            static_cast<float*>(f32_buf.data()),
            info, grad.numel());
        CUDA_CHECK(cudaGetLastError());

        // Cast float32 sums back to float16.
        Tensor result(std::make_shared<TensorImpl>(padded, DType::Float16, grad.device()));
        cast_f32_to_f16_kernel<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const float*>(f32_buf.data()),
            static_cast<__half*>(result.data()),
            n_out);
        CUDA_CHECK(cudaGetLastError());

        if (padded != target_shape)
            result = reshape(result, target_shape);
        return result;
    }

    if (grad.dtype() == DType::BFloat16) {
        // Accumulate into float32 (avoids atomicAdd(__nv_bfloat16*) / sm_90 req).
        Tensor f32_buf(std::make_shared<TensorImpl>(padded, DType::Float32, grad.device()));
        CUDA_CHECK(cudaMemset(f32_buf.data(), 0, static_cast<size_t>(n_out) * sizeof(float)));

        unbroadcast_kernel_bfloat16<<<cuda_blocks(grad.numel()), kThreads>>>(
            static_cast<const __nv_bfloat16*>(grad.data()),
            static_cast<float*>(f32_buf.data()),
            info, grad.numel());
        CUDA_CHECK(cudaGetLastError());

        // Cast float32 sums back to bfloat16.
        Tensor result(std::make_shared<TensorImpl>(padded, DType::BFloat16, grad.device()));
        cast_f32_to_bf16_kernel<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const float*>(f32_buf.data()),
            static_cast<__nv_bfloat16*>(result.data()),
            n_out);
        CUDA_CHECK(cudaGetLastError());

        if (padded != target_shape)
            result = reshape(result, target_shape);
        return result;
    }

    // Float32 path.
    Tensor result(std::make_shared<TensorImpl>(padded, grad.dtype(), grad.device()));
    CUDA_CHECK(cudaMemset(result.data(), 0, static_cast<size_t>(n_out) * sizeof(float)));

    unbroadcast_kernel<<<cuda_blocks(grad.numel()), kThreads>>>(
        static_cast<const float*>(grad.data()),
        static_cast<float*>(result.data()),
        info, grad.numel());
    CUDA_CHECK(cudaGetLastError());

    if (padded != target_shape)
        result = reshape(result, target_shape);
    return result;
}

// In-place ops.

__global__ void add_inplace_kernel(float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float alpha, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        a[i] += alpha * b[i];
}

void cuda_add_(Tensor& a, const Tensor& b, float alpha) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    require_cuda_float32_contiguous(a, "cuda_add_", "a");
    require_cuda_float32_contiguous(b, "cuda_add_", "b");
    require_same_cuda_device(a, b, "cuda_add_");
    if (a.shape() != b.shape())
        throw std::runtime_error("cuda_add_: shape mismatch");

    const int64_t n = a.numel();
    if (n == 0) return;
    add_inplace_kernel<<<cuda_blocks(n), kThreads>>>(
        static_cast<float*>(a.data()),
        static_cast<const float*>(b.data()),
        alpha, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mul_inplace_kernel(float* __restrict__ a, float alpha, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        a[i] *= alpha;
}

void cuda_mul_(Tensor& a, float alpha) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    require_cuda_float32_contiguous(a, "cuda_mul_", "a");

    const int64_t n = a.numel();
    if (n == 0) return;
    mul_inplace_kernel<<<cuda_blocks(n), kThreads>>>(
        static_cast<float*>(a.data()),
        alpha, n);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace orthon
