#include <orthon/backends/cuda/softmax_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/runtime.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <float.h>
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

static constexpr int kSoftmaxBlock = 256;

// Online softmax reduces pairs (m, s), where s = sum(exp(x - m)).
// Adding x uses m_new = max(m, x) and
// s_new = s * exp(m - m_new) + exp(x - m_new).
// Combining two pairs rescales both sums to their shared maximum.
// The identity is (-inf, 0); separate shared max/sum arrays prevent races.
__global__ void softmax_fused_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst,
                                      int N) {
    __shared__ float smem_m[kSoftmaxBlock];
    __shared__ float smem_d[kSoftmaxBlock];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Online single-pass reduction.
    float my_m = -FLT_MAX;
    float my_d = 0.0f;
    for (int j = tid; j < N; j += kSoftmaxBlock) {
        float x     = src[row * N + j];
        float new_m = fmaxf(my_m, x);
        my_d = my_d * expf(my_m - new_m) + expf(x - new_m);
        my_m = new_m;
    }
    smem_m[tid] = my_m;
    smem_d[tid] = my_d;
    __syncthreads();

    // Tree-reduce the (m, d) pairs.
    for (int stride = kSoftmaxBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = smem_m[tid],          d1 = smem_d[tid];
            float m2 = smem_m[tid + stride], d2 = smem_d[tid + stride];
            float m_new = fmaxf(m1, m2);
            smem_m[tid] = m_new;
            smem_d[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
        }
        __syncthreads();
    }

    const float rmax = smem_m[0];
    const float rsum = smem_d[0];

    // Normalize with a second coalesced global read of src.
    for (int j = tid; j < N; j += kSoftmaxBlock)
        dst[row * N + j] = expf(src[row * N + j] - rmax) / rsum;
}

Tensor cuda_softmax(const Tensor& t) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int M = static_cast<int>(t.shape()[0]);
    const int N = static_cast<int>(t.shape()[1]);

    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));

    softmax_fused_kernel<<<M, kSoftmaxBlock, 0, stream>>>(
        static_cast<const float*>(t.data()),
        static_cast<float*>(out.data()),
        N);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

static void dim_factors_cuda(const Tensor& t,
                             int64_t dim,
                             int64_t& outer,
                             int64_t& reduce,
                             int64_t& inner) {
    const auto& sh = t.shape();
    outer = 1;
    for (int64_t i = 0; i < dim; ++i)
        outer *= sh[static_cast<size_t>(i)];
    reduce = sh[static_cast<size_t>(dim)];
    inner = 1;
    for (int64_t i = dim + 1; i < t.ndim(); ++i)
        inner *= sh[static_cast<size_t>(i)];
}

__global__ void log_softmax_kernel(const float* __restrict__ src,
                                   float* __restrict__ dst,
                                   int outer,
                                   int reduce,
                                   int inner) {
    __shared__ float smem_m[kSoftmaxBlock];
    __shared__ float smem_d[kSoftmaxBlock];

    const int seg = blockIdx.x;
    const int tid = threadIdx.x;
    const int o = seg / inner;
    const int in = seg % inner;
    const int base = o * reduce * inner + in;

    float my_m = -FLT_MAX;
    float my_d = 0.0f;
    for (int r = tid; r < reduce; r += kSoftmaxBlock) {
        float x = src[base + r * inner];
        float new_m = fmaxf(my_m, x);
        my_d = my_d * expf(my_m - new_m) + expf(x - new_m);
        my_m = new_m;
    }
    smem_m[tid] = my_m;
    smem_d[tid] = my_d;
    __syncthreads();

    for (int stride = kSoftmaxBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = smem_m[tid], d1 = smem_d[tid];
            float m2 = smem_m[tid + stride], d2 = smem_d[tid + stride];
            float m_new = fmaxf(m1, m2);
            smem_m[tid] = m_new;
            smem_d[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
        }
        __syncthreads();
    }

    const float rmax = smem_m[0];
    const float log_sum = logf(smem_d[0]);
    for (int r = tid; r < reduce; r += kSoftmaxBlock) {
        const int idx = base + r * inner;
        dst[idx] = (src[idx] - rmax) - log_sum;
    }
}

__global__ void log_softmax_backward_kernel(const float* __restrict__ input,
                                            const float* __restrict__ grad_output,
                                            float* __restrict__ grad,
                                            int outer,
                                            int reduce,
                                            int inner) {
    __shared__ float smem_m[kSoftmaxBlock];
    __shared__ float smem_d[kSoftmaxBlock];
    __shared__ float smem_g[kSoftmaxBlock];

    const int seg = blockIdx.x;
    const int tid = threadIdx.x;
    const int o = seg / inner;
    const int in = seg % inner;
    const int base = o * reduce * inner + in;

    float my_m = -FLT_MAX;
    float my_d = 0.0f;
    float my_g = 0.0f;
    for (int r = tid; r < reduce; r += kSoftmaxBlock) {
        const int idx = base + r * inner;
        float x = input[idx];
        float new_m = fmaxf(my_m, x);
        my_d = my_d * expf(my_m - new_m) + expf(x - new_m);
        my_m = new_m;
        my_g += grad_output[idx];
    }
    smem_m[tid] = my_m;
    smem_d[tid] = my_d;
    smem_g[tid] = my_g;
    __syncthreads();

    for (int stride = kSoftmaxBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = smem_m[tid], d1 = smem_d[tid];
            float m2 = smem_m[tid + stride], d2 = smem_d[tid + stride];
            float m_new = fmaxf(m1, m2);
            smem_m[tid] = m_new;
            smem_d[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
            smem_g[tid] += smem_g[tid + stride];
        }
        __syncthreads();
    }

    const float rmax = smem_m[0];
    const float rsum = smem_d[0];
    const float grad_sum = smem_g[0];
    for (int r = tid; r < reduce; r += kSoftmaxBlock) {
        const int idx = base + r * inner;
        const float prob = expf(input[idx] - rmax) / rsum;
        grad[idx] = grad_output[idx] - prob * grad_sum;
    }
}

Tensor cuda_log_softmax(const Tensor& t, int64_t dim) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    int64_t outer64, reduce64, inner64;
    dim_factors_cuda(t, dim, outer64, reduce64, inner64);
    const int outer = static_cast<int>(outer64);
    const int reduce = static_cast<int>(reduce64);
    const int inner = static_cast<int>(inner64);

    Tensor out(std::make_shared<TensorImpl>(t.shape(), t.dtype(), t.device()));
    log_softmax_kernel<<<outer * inner, kSoftmaxBlock, 0, stream>>>(
        static_cast<const float*>(t.data()),
        static_cast<float*>(out.data()),
        outer,
        reduce,
        inner);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_log_softmax_backward(const Tensor& input,
                                 const Tensor& grad_output,
                                 int64_t dim) {
    DeviceGuard guard(input.device());
    auto context = default_runtime().cuda(input.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    int64_t outer64, reduce64, inner64;
    dim_factors_cuda(input, dim, outer64, reduce64, inner64);
    const int outer = static_cast<int>(outer64);
    const int reduce = static_cast<int>(reduce64);
    const int inner = static_cast<int>(inner64);

    Tensor grad(std::make_shared<TensorImpl>(
        input.shape(), input.dtype(), input.device()));
    log_softmax_backward_kernel<<<outer * inner, kSoftmaxBlock, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad.data()),
        outer,
        reduce,
        inner);
    CUDA_CHECK(cudaGetLastError());
    return grad;
}

}  // namespace orthon
