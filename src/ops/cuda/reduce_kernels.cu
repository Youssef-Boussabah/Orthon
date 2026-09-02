#include <orthon/ops/cuda/reduce_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/memory_pool.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

namespace orthon {

static constexpr int kThreads = 256;
static constexpr int64_t kMaxBlocks = 1024;

static int cuda_blocks(int64_t n, int threads = kThreads) {
    return static_cast<int>((n + threads - 1) / threads);
}

__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                  float* __restrict__ partials,
                                  int64_t n) {
    extern __shared__ float smem[];

    float thread_sum = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        thread_sum += input[i];
    }

    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        partials[blockIdx.x] = smem[0];
}

__global__ void reduce_partials_kernel(const float* __restrict__ partials,
                                       float* __restrict__ out,
                                       int64_t n,
                                       float scale) {
    extern __shared__ float smem[];

    float thread_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        thread_sum += partials[i];

    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[0] = smem[0] * scale;
}

__global__ void reduce_dim_kernel(const float* __restrict__ input,
                                  float* __restrict__ out,
                                  int64_t outer,
                                  int64_t reduce,
                                  int64_t inner,
                                  int64_t n_out,
                                  float scale) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_out;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        int64_t base = outer_idx * reduce * inner + inner_idx;
        float acc = 0.0f;
        for (int64_t r = 0; r < reduce; ++r)
            acc += input[base + r * inner];
        out[idx] = acc * scale;
    }
}

// Float16 sum kernels accumulate in float32.
__global__ void reduce_sum_kernel_half(const __half* __restrict__ input,
                                       float* __restrict__ partials,
                                       int64_t n) {
    extern __shared__ float smem[];
    float thread_sum = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        thread_sum += __half2float(input[i]);
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];
}

__global__ void reduce_partials_kernel_half(const float* __restrict__ partials,
                                            __half* __restrict__ out,
                                            int64_t n, float scale) {
    extern __shared__ float smem[];
    float thread_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        thread_sum += partials[i];
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = __float2half(smem[0] * scale);
}

__global__ void reduce_dim_kernel_half(const __half* __restrict__ input,
                                       __half* __restrict__ out,
                                       int64_t outer,
                                       int64_t reduce,
                                       int64_t inner,
                                       int64_t n_out,
                                       float scale) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_out;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        int64_t base = outer_idx * reduce * inner + inner_idx;
        float acc = 0.0f;
        for (int64_t r = 0; r < reduce; ++r)
            acc += __half2float(input[base + r * inner]);
        out[idx] = __float2half(acc * scale);
    }
}

// BFloat16 sum kernels accumulate in float32.
__global__ void reduce_sum_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                            float* __restrict__ partials,
                                            int64_t n) {
    extern __shared__ float smem[];
    float thread_sum = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x)
        thread_sum += __bfloat162float(input[i]);
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];
}

__global__ void reduce_partials_kernel_bfloat16(const float* __restrict__ partials,
                                                 __nv_bfloat16* __restrict__ out,
                                                 int64_t n, float scale) {
    extern __shared__ float smem[];
    float thread_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        thread_sum += partials[i];
    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = __float2bfloat16(smem[0] * scale);
}

__global__ void reduce_dim_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                            __nv_bfloat16* __restrict__ out,
                                            int64_t outer,
                                            int64_t reduce,
                                            int64_t inner,
                                            int64_t n_out,
                                            float scale) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_out;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        int64_t base = outer_idx * reduce * inner + inner_idx;
        float acc = 0.0f;
        for (int64_t r = 0; r < reduce; ++r)
            acc += __bfloat162float(input[base + r * inner]);
        out[idx] = __float2bfloat16(acc * scale);
    }
}

static Tensor cuda_reduce_all(const Tensor& t,
                              const std::vector<int64_t>& out_shape,
                              float scale) {
    const int64_t n = t.numel();
    const int num_blocks = static_cast<int>(
        std::min((n + kThreads - 1) / kThreads, kMaxBlocks));
    const size_t partial_bytes = static_cast<size_t>(num_blocks) * sizeof(float);

    Tensor out(std::make_shared<TensorImpl>(out_shape, t.dtype(), t.device()));
    memory_pool_reset_scratch(t.device());
    float* partials = static_cast<float*>(
        memory_pool_scratch_allocate(partial_bytes, t.device()));

    if (t.dtype() == DType::Float16) {
        // Partial sums always in float32 for precision.
        reduce_sum_kernel_half<<<num_blocks, kThreads, kThreads * sizeof(float)>>>(
            static_cast<const __half*>(t.data()),
            partials, n);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials_kernel_half<<<1, kThreads, kThreads * sizeof(float)>>>(
            partials,
            static_cast<__half*>(out.data()), num_blocks, scale);
    } else if (t.dtype() == DType::BFloat16) {
        reduce_sum_kernel_bfloat16<<<num_blocks, kThreads, kThreads * sizeof(float)>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            partials, n);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials_kernel_bfloat16<<<1, kThreads, kThreads * sizeof(float)>>>(
            partials,
            static_cast<__nv_bfloat16*>(out.data()), num_blocks, scale);
    } else {
        reduce_sum_kernel<<<num_blocks, kThreads, kThreads * sizeof(float)>>>(
            static_cast<const float*>(t.data()),
            partials, n);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials_kernel<<<1, kThreads, kThreads * sizeof(float)>>>(
            partials,
            static_cast<float*>(out.data()), num_blocks, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

static Tensor cuda_reduce_dim(const Tensor& t,
                              const std::vector<int64_t>& out_shape,
                              int64_t dim,
                              float scale) {
    int64_t outer = 1;
    int64_t inner = 1;
    const auto& shape = t.shape();
    for (int64_t i = 0; i < dim; ++i)
        outer *= shape[static_cast<size_t>(i)];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        inner *= shape[static_cast<size_t>(i)];
    const int64_t reduce = shape[static_cast<size_t>(dim)];
    const int64_t n_out = outer * inner;

    Tensor out(std::make_shared<TensorImpl>(out_shape, t.dtype(), t.device()));
    if (t.dtype() == DType::Float16) {
        reduce_dim_kernel_half<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()),
            outer, reduce, inner, n_out, scale);
    } else if (t.dtype() == DType::BFloat16) {
        reduce_dim_kernel_bfloat16<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            outer, reduce, inner, n_out, scale);
    } else {
        reduce_dim_kernel<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()),
            outer, reduce, inner, n_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_sum(const Tensor& t,
                const std::vector<int64_t>& out_shape,
                int64_t dim,
                bool reduce_all) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    if (reduce_all)
        return cuda_reduce_all(t, out_shape, 1.0f);
    return cuda_reduce_dim(t, out_shape, dim, 1.0f);
}

Tensor cuda_mean(const Tensor& t,
                 const std::vector<int64_t>& out_shape,
                 int64_t dim,
                 bool reduce_all) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    const float factor = reduce_all
        ? static_cast<float>(t.numel())
        : static_cast<float>(t.shape()[static_cast<size_t>(dim)]);
    const float scale = 1.0f / factor;
    if (reduce_all)
        return cuda_reduce_all(t, out_shape, scale);
    return cuda_reduce_dim(t, out_shape, dim, scale);
}

// Max reduction kernels.

// Both the CPU and CUDA max reductions use -infinity as the identity, never
// a finite seed: a finite sentinel returns the wrong maximum (and index 0)
// for inputs whose true max lies below it, and for all--infinity inputs.
static constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct MaxPair {
    float   val;
    int64_t idx;
};

__global__ void reduce_max_kernel(const float* __restrict__ input,
                                  float*   __restrict__ partial_vals,
                                  int64_t* __restrict__ partial_idxs,
                                  int64_t n) {
    extern __shared__ char smem_raw[];
    MaxPair* smem = reinterpret_cast<MaxPair*>(smem_raw);

    MaxPair local{kNegInf, 0};
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        if (input[i] > local.val) { local.val = input[i]; local.idx = i; }
    }
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride].val > smem[threadIdx.x].val)
                smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_vals[blockIdx.x] = smem[0].val;
        partial_idxs[blockIdx.x] = smem[0].idx;
    }
}

__global__ void reduce_max_partials_kernel(const float*   __restrict__ partial_vals,
                                           const int64_t* __restrict__ partial_idxs,
                                           float*   __restrict__ out_val,
                                           int64_t* __restrict__ out_idx,
                                           int64_t n) {
    extern __shared__ char smem_raw[];
    MaxPair* smem = reinterpret_cast<MaxPair*>(smem_raw);

    MaxPair local{kNegInf, 0};
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (partial_vals[i] > local.val) {
            local.val = partial_vals[i];
            local.idx = partial_idxs[i];
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride].val > smem[threadIdx.x].val)
                smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[0] = smem[0].val;
        out_idx[0] = smem[0].idx;
    }
}

__global__ void reduce_max_dim_kernel(const float* __restrict__ input,
                                      float*   __restrict__ out_vals,
                                      int64_t* __restrict__ out_idxs,
                                      int64_t outer,
                                      int64_t reduce,
                                      int64_t inner,
                                      int64_t n_out) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_out;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        int64_t base = outer_idx * reduce * inner + inner_idx;
        float   best_val = kNegInf;
        int64_t best_r   = 0;
        for (int64_t r = 0; r < reduce; ++r) {
            float v = input[base + r * inner];
            if (v > best_val) { best_val = v; best_r = r; }
        }
        out_vals[idx] = best_val;
        out_idxs[idx] = best_r;
    }
}

__global__ void max_scatter_grad_kernel(const float*   __restrict__ grad_out,
                                        const int64_t* __restrict__ indices,
                                        float*         __restrict__ grad_in,
                                        int64_t outer,
                                        int64_t reduce,
                                        int64_t inner,
                                        int64_t n_out) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_out;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_i = i / inner;
        int64_t inner_i = i % inner;
        int64_t r = indices[i];
        grad_in[outer_i * reduce * inner + r * inner + inner_i] = grad_out[i];
    }
}

// Float16 max kernels compare in float32.

__global__ void reduce_max_kernel_half(const __half* __restrict__ input,
                                       float*   __restrict__ partial_vals,
                                       int64_t* __restrict__ partial_idxs,
                                       int64_t n) {
    extern __shared__ char smem_raw[];
    MaxPair* smem = reinterpret_cast<MaxPair*>(smem_raw);
    MaxPair local{kNegInf, 0};
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[i]);
        if (v > local.val) { local.val = v; local.idx = i; }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride].val > smem[threadIdx.x].val)
                smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_vals[blockIdx.x] = smem[0].val;
        partial_idxs[blockIdx.x] = smem[0].idx;
    }
}

// Writes the final max as __half.
__global__ void reduce_max_partials_kernel_half(const float*   __restrict__ partial_vals,
                                                const int64_t* __restrict__ partial_idxs,
                                                __half*  __restrict__ out_val,
                                                int64_t* __restrict__ out_idx,
                                                int64_t n) {
    extern __shared__ char smem_raw[];
    MaxPair* smem = reinterpret_cast<MaxPair*>(smem_raw);
    MaxPair local{kNegInf, 0};
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (partial_vals[i] > local.val) {
            local.val = partial_vals[i];
            local.idx = partial_idxs[i];
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride].val > smem[threadIdx.x].val)
                smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[0] = __float2half(smem[0].val);
        out_idx[0] = smem[0].idx;
    }
}

__global__ void reduce_max_dim_kernel_half(const __half* __restrict__ input,
                                           __half*   __restrict__ out_vals,
                                           int64_t*  __restrict__ out_idxs,
                                           int64_t outer,
                                           int64_t reduce,
                                           int64_t inner,
                                           int64_t n_out) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_out;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        int64_t base = outer_idx * reduce * inner + inner_idx;
        float   best_val = kNegInf;
        int64_t best_r   = 0;
        for (int64_t r = 0; r < reduce; ++r) {
            float v = __half2float(input[base + r * inner]);
            if (v > best_val) { best_val = v; best_r = r; }
        }
        out_vals[idx] = __float2half(best_val);
        out_idxs[idx] = best_r;
    }
}

// BFloat16 max kernels compare in float32.

__global__ void reduce_max_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                            float*   __restrict__ partial_vals,
                                            int64_t* __restrict__ partial_idxs,
                                            int64_t n) {
    extern __shared__ char smem_raw[];
    MaxPair* smem = reinterpret_cast<MaxPair*>(smem_raw);
    MaxPair local{kNegInf, 0};
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __bfloat162float(input[i]);
        if (v > local.val) { local.val = v; local.idx = i; }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride].val > smem[threadIdx.x].val)
                smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_vals[blockIdx.x] = smem[0].val;
        partial_idxs[blockIdx.x] = smem[0].idx;
    }
}

// Writes the final max as __nv_bfloat16.
__global__ void reduce_max_partials_kernel_bfloat16(const float*   __restrict__ partial_vals,
                                                     const int64_t* __restrict__ partial_idxs,
                                                     __nv_bfloat16* __restrict__ out_val,
                                                     int64_t* __restrict__ out_idx,
                                                     int64_t n) {
    extern __shared__ char smem_raw[];
    MaxPair* smem = reinterpret_cast<MaxPair*>(smem_raw);
    MaxPair local{kNegInf, 0};
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (partial_vals[i] > local.val) {
            local.val = partial_vals[i];
            local.idx = partial_idxs[i];
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem[threadIdx.x + stride].val > smem[threadIdx.x].val)
                smem[threadIdx.x] = smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[0] = __float2bfloat16(smem[0].val);
        out_idx[0] = smem[0].idx;
    }
}

__global__ void reduce_max_dim_kernel_bfloat16(const __nv_bfloat16* __restrict__ input,
                                                __nv_bfloat16* __restrict__ out_vals,
                                                int64_t* __restrict__ out_idxs,
                                                int64_t outer,
                                                int64_t reduce,
                                                int64_t inner,
                                                int64_t n_out) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_out;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        int64_t base = outer_idx * reduce * inner + inner_idx;
        float   best_val = kNegInf;
        int64_t best_r   = 0;
        for (int64_t r = 0; r < reduce; ++r) {
            float v = __bfloat162float(input[base + r * inner]);
            if (v > best_val) { best_val = v; best_r = r; }
        }
        out_vals[idx] = __float2bfloat16(best_val);
        out_idxs[idx] = best_r;
    }
}

static std::pair<Tensor, Tensor> cuda_max_all(const Tensor& t,
                                               const std::vector<int64_t>& out_shape) {
    const int64_t n = t.numel();
    const int num_blocks = static_cast<int>(
        std::min((n + kThreads - 1) / kThreads, kMaxBlocks));
    const size_t partial_vals_bytes = static_cast<size_t>(num_blocks) * sizeof(float);
    const size_t partial_idxs_bytes = static_cast<size_t>(num_blocks) * sizeof(int64_t);

    // Partial buffers always float32 for precision.
    memory_pool_reset_scratch(t.device());
    float* p_vals = static_cast<float*>(
        memory_pool_scratch_allocate(partial_vals_bytes, t.device()));
    int64_t* p_idxs = static_cast<int64_t*>(
        memory_pool_scratch_allocate(partial_idxs_bytes, t.device()));

    if (t.dtype() == DType::Float16) {
        Tensor out_val(std::make_shared<TensorImpl>(out_shape, DType::Float16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(out_shape, DType::Int64,   t.device()));

        reduce_max_kernel_half<<<num_blocks, kThreads, kThreads * sizeof(MaxPair)>>>(
            static_cast<const __half*>(t.data()),
            p_vals,
            p_idxs, n);
        CUDA_CHECK(cudaGetLastError());

        reduce_max_partials_kernel_half<<<1, kThreads, kThreads * sizeof(MaxPair)>>>(
            p_vals,
            p_idxs,
            static_cast<__half*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()), num_blocks);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    if (t.dtype() == DType::BFloat16) {
        Tensor out_val(std::make_shared<TensorImpl>(out_shape, DType::BFloat16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(out_shape, DType::Int64,    t.device()));

        reduce_max_kernel_bfloat16<<<num_blocks, kThreads, kThreads * sizeof(MaxPair)>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            p_vals,
            p_idxs, n);
        CUDA_CHECK(cudaGetLastError());

        reduce_max_partials_kernel_bfloat16<<<1, kThreads, kThreads * sizeof(MaxPair)>>>(
            p_vals,
            p_idxs,
            static_cast<__nv_bfloat16*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()), num_blocks);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    Tensor out_val(std::make_shared<TensorImpl>(out_shape, DType::Float32, t.device()));
    Tensor out_idx(std::make_shared<TensorImpl>(out_shape, DType::Int64,   t.device()));

    reduce_max_kernel<<<num_blocks, kThreads, kThreads * sizeof(MaxPair)>>>(
        static_cast<const float*>(t.data()),
        p_vals,
        p_idxs, n);
    CUDA_CHECK(cudaGetLastError());

    reduce_max_partials_kernel<<<1, kThreads, kThreads * sizeof(MaxPair)>>>(
        p_vals,
        p_idxs,
        static_cast<float*>(out_val.data()),
        static_cast<int64_t*>(out_idx.data()), num_blocks);
    CUDA_CHECK(cudaGetLastError());

    return {out_val, out_idx};
}

static std::pair<Tensor, Tensor> cuda_max_dim(const Tensor& t,
                                               const std::vector<int64_t>& out_shape,
                                               int64_t dim) {
    int64_t outer = 1, inner = 1;
    const auto& shape = t.shape();
    for (int64_t i = 0; i < dim; ++i)
        outer *= shape[static_cast<size_t>(i)];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        inner *= shape[static_cast<size_t>(i)];
    const int64_t reduce = shape[static_cast<size_t>(dim)];
    const int64_t n_out = outer * inner;

    if (t.dtype() == DType::Float16) {
        Tensor out_val(std::make_shared<TensorImpl>(out_shape, DType::Float16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(out_shape, DType::Int64,   t.device()));
        reduce_max_dim_kernel_half<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()),
            outer, reduce, inner, n_out);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    if (t.dtype() == DType::BFloat16) {
        Tensor out_val(std::make_shared<TensorImpl>(out_shape, DType::BFloat16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(out_shape, DType::Int64,    t.device()));
        reduce_max_dim_kernel_bfloat16<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()),
            outer, reduce, inner, n_out);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    Tensor out_val(std::make_shared<TensorImpl>(out_shape, DType::Float32, t.device()));
    Tensor out_idx(std::make_shared<TensorImpl>(out_shape, DType::Int64,   t.device()));

    reduce_max_dim_kernel<<<cuda_blocks(n_out), kThreads>>>(
        static_cast<const float*>(t.data()),
        static_cast<float*>(out_val.data()),
        static_cast<int64_t*>(out_idx.data()),
        outer, reduce, inner, n_out);
    CUDA_CHECK(cudaGetLastError());

    return {out_val, out_idx};
}

std::pair<Tensor, Tensor> cuda_max_with_indices(const Tensor& t,
                                                const std::vector<int64_t>& out_shape,
                                                int64_t dim,
                                                bool reduce_all) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(t.device());

    if (reduce_all)
        return cuda_max_all(t, out_shape);
    return cuda_max_dim(t, out_shape, dim);
}

__global__ void max_scatter_grad_kernel_half(const __half*  __restrict__ grad_out,
                                             const int64_t* __restrict__ indices,
                                             __half*        __restrict__ grad_in,
                                             int64_t outer,
                                             int64_t reduce,
                                             int64_t inner,
                                             int64_t n_out) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_out;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_i = i / inner;
        int64_t inner_i = i % inner;
        int64_t r = indices[i];
        grad_in[outer_i * reduce * inner + r * inner + inner_i] = grad_out[i];
    }
}

__global__ void max_scatter_grad_kernel_bfloat16(const __nv_bfloat16* __restrict__ grad_out,
                                                  const int64_t* __restrict__ indices,
                                                  __nv_bfloat16* __restrict__ grad_in,
                                                  int64_t outer,
                                                  int64_t reduce,
                                                  int64_t inner,
                                                  int64_t n_out) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_out;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_i = i / inner;
        int64_t inner_i = i % inner;
        int64_t r = indices[i];
        grad_in[outer_i * reduce * inner + r * inner + inner_i] = grad_out[i];
    }
}

void cuda_max_scatter_grad(const Tensor& grad_out,
                           const Tensor& indices,
                           Tensor& grad_in,
                           int64_t outer,
                           int64_t reduce,
                           int64_t inner) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(grad_in.device());

    const int64_t n_out = outer * inner;
    if (grad_in.dtype() == DType::Float16) {
        max_scatter_grad_kernel_half<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __half*>(grad_out.data()),
            static_cast<const int64_t*>(indices.data()),
            static_cast<__half*>(grad_in.data()),
            outer, reduce, inner, n_out);
    } else if (grad_in.dtype() == DType::BFloat16) {
        max_scatter_grad_kernel_bfloat16<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const __nv_bfloat16*>(grad_out.data()),
            static_cast<const int64_t*>(indices.data()),
            static_cast<__nv_bfloat16*>(grad_in.data()),
            outer, reduce, inner, n_out);
    } else {
        max_scatter_grad_kernel<<<cuda_blocks(n_out), kThreads>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<const int64_t*>(indices.data()),
            static_cast<float*>(grad_in.data()),
            outer, reduce, inner, n_out);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace orthon
