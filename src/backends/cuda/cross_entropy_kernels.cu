#include <orthon/backends/cuda/cross_entropy_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/runtime/runtime.h>
#include <orthon/memory/allocator.h>
#include <cuda_runtime.h>
#include <float.h>
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

static constexpr int kCEBlock = 256;

// One block reduces each row using online max/sum pairs. Combining pairs
// rescales both sums to the larger maximum; separate shared arrays prevent
// max/sum overwrite races. Loss is -target + row_max + log(row_sum).
__global__ void cross_entropy_fused_kernel(
        const float*   __restrict__ logits,
        const int64_t* __restrict__ targets,
        float*         __restrict__ losses,
        int C)
{
    __shared__ float smem_m[kCEBlock];
    __shared__ float smem_d[kCEBlock];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Single-pass online reduction for max and sum_exp.
    float my_m = -FLT_MAX;
    float my_d = 0.0f;
    for (int j = tid; j < C; j += kCEBlock) {
        float x     = logits[row * C + j];
        float new_m = fmaxf(my_m, x);
        my_d = my_d * expf(my_m - new_m) + expf(x - new_m);
        my_m = new_m;
    }
    smem_m[tid] = my_m;
    smem_d[tid] = my_d;
    __syncthreads();

    // Tree-reduce the (m, d) pairs.
    for (int stride = kCEBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = smem_m[tid],          d1 = smem_d[tid];
            float m2 = smem_m[tid + stride], d2 = smem_d[tid + stride];
            float m_new = fmaxf(m1, m2);
            smem_m[tid] = m_new;
            smem_d[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float   rmax = smem_m[0];
        const float   rsum = smem_d[0];
        const int64_t tgt  = targets[row];
        // Single scalar load; in L1 cache for C <= kCEBlock, one cache miss
        // for larger C, which is negligible versus the full-row read above.
        losses[row] = -logits[row * C + (int)tgt] + rmax + logf(rsum);
    }
}

__global__ void mean_loss_kernel(const float* __restrict__ losses,
                                 float* __restrict__ out,
                                 int B) {
    __shared__ float smem[kCEBlock];

    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < B; i += kCEBlock) {
        acc += losses[i];
    }
    smem[tid] = acc;
    __syncthreads();

    for (int stride = kCEBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = smem[0] / static_cast<float>(B);
    }
}

Tensor cuda_cross_entropy_loss(const Tensor& logits,
                               const std::vector<int64_t>& targets) {
    DeviceGuard guard(logits.device());
    auto context = default_runtime().cuda(logits.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int B = static_cast<int>(logits.shape()[0]);
    const int C = static_cast<int>(logits.shape()[1]);

    auto allocator = default_runtime().allocator(logits.device());
    auto target_allocation = allocator->allocate(static_cast<size_t>(B) * sizeof(int64_t));
    auto loss_allocation = allocator->allocate(static_cast<size_t>(B) * sizeof(float));
    auto* d_targets = static_cast<int64_t*>(target_allocation.data());
    auto* d_losses = static_cast<float*>(loss_allocation.data());
    context->copy_h2d_snapshot(d_targets, targets.data(), static_cast<size_t>(B) * sizeof(int64_t));

    cross_entropy_fused_kernel<<<B, kCEBlock, 0, stream>>>(
        static_cast<const float*>(logits.data()),
        d_targets, d_losses, C);
    CUDA_CHECK(cudaGetLastError());

    Tensor loss(std::make_shared<TensorImpl>(
        std::vector<int64_t>{1}, logits.dtype(), logits.device()));
    mean_loss_kernel<<<1, kCEBlock, 0, stream>>>(
        d_losses,
        static_cast<float*>(loss.data()),
        B);
    CUDA_CHECK(cudaGetLastError());

    return loss;
}

float cuda_cross_entropy(const Tensor& logits,
                         const std::vector<int64_t>& targets) {
    DeviceGuard guard(logits.device());
    auto context = default_runtime().cuda(logits.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    Tensor loss = cuda_cross_entropy_loss(logits, targets);
    float value = 0.0f;
    context->copy_d2h_blocking(&value, loss.data(), sizeof(float));
    return value;
}

// The derivative is (softmax - one_hot(target)) * grad_output / batch_size.
// Reuse the online row reduction without allocating a softmax tensor.
__global__ void cross_entropy_backward_kernel(
        const float*   __restrict__ logits,
        const int64_t* __restrict__ targets,
        const float*   __restrict__ grad_output,
        float*         __restrict__ grad,
        int B, int C)
{
    __shared__ float smem_m[kCEBlock];
    __shared__ float smem_d[kCEBlock];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Online (max, sum_exp) reduction across this row's columns.
    float my_m = -FLT_MAX, my_d = 0.0f;
    for (int j = tid; j < C; j += kCEBlock) {
        float x     = logits[row * C + j];
        float new_m = fmaxf(my_m, x);
        my_d = my_d * expf(my_m - new_m) + expf(x - new_m);
        my_m = new_m;
    }
    smem_m[tid] = my_m;
    smem_d[tid] = my_d;
    __syncthreads();

    for (int stride = kCEBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = smem_m[tid], d1 = smem_d[tid];
            float m2 = smem_m[tid + stride], d2 = smem_d[tid + stride];
            float m_new = fmaxf(m1, m2);
            smem_m[tid] = m_new;
            smem_d[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
        }
        __syncthreads();
    }

    const float   rmax = smem_m[0];
    const float   rsum = smem_d[0];
    const int64_t tgt  = targets[row];
    const float   scale = grad_output[0] / static_cast<float>(B);

    // Write (softmax[j] - one_hot[j]) * scale for each column j.
    for (int j = tid; j < C; j += kCEBlock) {
        float softmax_j = expf(logits[row * C + j] - rmax) / rsum;
        float delta     = softmax_j - ((j == (int)tgt) ? 1.0f : 0.0f);
        grad[row * C + j] = delta * scale;
    }
}

Tensor cuda_cross_entropy_backward(const Tensor& logits,
                                   const std::vector<int64_t>& targets,
                                   const Tensor& grad_output) {
    DeviceGuard guard(logits.device());
    auto context = default_runtime().cuda(logits.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int B = static_cast<int>(logits.shape()[0]);
    const int C = static_cast<int>(logits.shape()[1]);

    auto target_allocation = default_runtime().allocator(logits.device())->allocate(static_cast<size_t>(B) * sizeof(int64_t));
    auto* d_targets = static_cast<int64_t*>(target_allocation.data());
    context->copy_h2d_snapshot(d_targets, targets.data(), static_cast<size_t>(B) * sizeof(int64_t));

    Tensor grad(std::make_shared<TensorImpl>(
        logits.shape(), logits.dtype(), logits.device()));

    cross_entropy_backward_kernel<<<B, kCEBlock, 0, stream>>>(
        static_cast<const float*>(logits.data()),
        d_targets,
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad.data()),
        B, C);
    CUDA_CHECK(cudaGetLastError());

    return grad;
}

}  // namespace orthon
