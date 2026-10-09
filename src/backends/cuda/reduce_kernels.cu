#include <orthon/backends/cuda/reduce_kernels.h>
#include <orthon/reduction/extrema.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/runtime.h>
#include <orthon/memory/scratch.h>
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

namespace {
// One operation owns one lease and partitions it before any kernel launch.
// Both passes use the context compute stream; lease release records completion.
class ReductionWorkspace {
    ScratchLease scratch;
public:
    float* values;
    int64_t* indices;
    ReductionWorkspace(const Tensor& t, int blocks, bool extrema)
        : scratch(default_runtime().scratch(t.device())->acquire(required(blocks, extrema))) {
        values = static_cast<float*>(scratch.allocate(static_cast<size_t>(blocks) * sizeof(float)));
        indices = extrema ? static_cast<int64_t*>(scratch.allocate(static_cast<size_t>(blocks) * sizeof(int64_t))) : nullptr;
    }
private:
    static size_t required(int blocks, bool extrema) {
        const size_t values = allocation_size_class(static_cast<size_t>(blocks) * sizeof(float));
        return extrema ? checked_allocation_add(values,
            allocation_size_class(static_cast<size_t>(blocks) * sizeof(int64_t))) : values;
    }
};
}

static Tensor cuda_reduce_all(const Tensor& t,
                              const ReductionPlan& plan,
                              float scale) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    const int64_t n = plan.input_numel;
    const int num_blocks = static_cast<int>(
        std::min((n - 1) / kThreads + 1, kMaxBlocks));
    Tensor out(std::make_shared<TensorImpl>(plan.output_shape, t.dtype(), t.device()));
    ReductionWorkspace workspace(t, num_blocks, false);
    float* partials = workspace.values;

    if (t.dtype() == DType::Float16) {
        // Partial sums always in float32 for precision.
        reduce_sum_kernel_half<<<num_blocks, kThreads, kThreads * sizeof(float), stream>>>(
            static_cast<const __half*>(t.data()),
            partials, n);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials_kernel_half<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            partials,
            static_cast<__half*>(out.data()), num_blocks, scale);
    } else if (t.dtype() == DType::BFloat16) {
        reduce_sum_kernel_bfloat16<<<num_blocks, kThreads, kThreads * sizeof(float), stream>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            partials, n);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials_kernel_bfloat16<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            partials,
            static_cast<__nv_bfloat16*>(out.data()), num_blocks, scale);
    } else {
        reduce_sum_kernel<<<num_blocks, kThreads, kThreads * sizeof(float), stream>>>(
            static_cast<const float*>(t.data()),
            partials, n);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials_kernel<<<1, kThreads, kThreads * sizeof(float), stream>>>(
            partials,
            static_cast<float*>(out.data()), num_blocks, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

static Tensor cuda_reduce_dim(const Tensor& t,
                              const ReductionPlan& plan,
                              float scale) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    const auto outer = plan.outer, reduce = plan.reduce, inner = plan.inner;
    const int64_t n_out = plan.output_numel;

    Tensor out(std::make_shared<TensorImpl>(plan.output_shape, t.dtype(), t.device()));
    if (t.dtype() == DType::Float16) {
        reduce_dim_kernel_half<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out.data()),
            outer, reduce, inner, n_out, scale);
    } else if (t.dtype() == DType::BFloat16) {
        reduce_dim_kernel_bfloat16<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            outer, reduce, inner, n_out, scale);
    } else {
        reduce_dim_kernel<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const float*>(t.data()),
            static_cast<float*>(out.data()),
            outer, reduce, inner, n_out, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_sum(const Tensor& t, const ReductionPlan& plan) {
    if (!t.device().is_cuda() || !t.is_contiguous())
        throw std::invalid_argument("reduction: CUDA execution requires contiguous CUDA operand");
    validate_reduction_operand(t, plan, ReductionKind::Sum);
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    if (plan.reduce_all) return cuda_reduce_all(t, plan, 1.0f);
    return cuda_reduce_dim(t, plan, 1.0f);
}
Tensor cuda_mean(const Tensor& t, const ReductionPlan& plan) {
    if (!t.device().is_cuda() || !t.is_contiguous())
        throw std::invalid_argument("reduction: CUDA execution requires contiguous CUDA operand");
    validate_reduction_operand(t, plan, ReductionKind::Mean);
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    const float scale = 1.0f / static_cast<float>(plan.reduce);
    if (plan.reduce_all) return cuda_reduce_all(t, plan, scale);
    return cuda_reduce_dim(t, plan, scale);
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

    MaxPair local{kNegInf, -1};
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        if (extrema_candidate_wins(input[i], i, local.val, local.idx)) { local.val = input[i]; local.idx = i; }
    }
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (extrema_candidate_wins(smem[threadIdx.x + stride].val, smem[threadIdx.x + stride].idx,
                                       smem[threadIdx.x].val, smem[threadIdx.x].idx))
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

    MaxPair local{kNegInf, -1};
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (extrema_candidate_wins(partial_vals[i], partial_idxs[i], local.val, local.idx)) {
            local.val = partial_vals[i];
            local.idx = partial_idxs[i];
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (extrema_candidate_wins(smem[threadIdx.x + stride].val, smem[threadIdx.x + stride].idx,
                                       smem[threadIdx.x].val, smem[threadIdx.x].idx))
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
        int64_t best_r   = -1;
        for (int64_t r = 0; r < reduce; ++r) {
            float v = input[base + r * inner];
            if (extrema_candidate_wins(v, r, best_val, best_r)) { best_val = v; best_r = r; }
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
    MaxPair local{kNegInf, -1};
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[i]);
        if (extrema_candidate_wins(v, i, local.val, local.idx)) { local.val = v; local.idx = i; }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (extrema_candidate_wins(smem[threadIdx.x + stride].val, smem[threadIdx.x + stride].idx,
                                       smem[threadIdx.x].val, smem[threadIdx.x].idx))
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
    MaxPair local{kNegInf, -1};
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (extrema_candidate_wins(partial_vals[i], partial_idxs[i], local.val, local.idx)) {
            local.val = partial_vals[i];
            local.idx = partial_idxs[i];
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (extrema_candidate_wins(smem[threadIdx.x + stride].val, smem[threadIdx.x + stride].idx,
                                       smem[threadIdx.x].val, smem[threadIdx.x].idx))
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
        int64_t best_r   = -1;
        for (int64_t r = 0; r < reduce; ++r) {
            float v = __half2float(input[base + r * inner]);
            if (extrema_candidate_wins(v, r, best_val, best_r)) { best_val = v; best_r = r; }
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
    MaxPair local{kNegInf, -1};
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)blockDim.x * gridDim.x) {
        float v = __bfloat162float(input[i]);
        if (extrema_candidate_wins(v, i, local.val, local.idx)) { local.val = v; local.idx = i; }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (extrema_candidate_wins(smem[threadIdx.x + stride].val, smem[threadIdx.x + stride].idx,
                                       smem[threadIdx.x].val, smem[threadIdx.x].idx))
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
    MaxPair local{kNegInf, -1};
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (extrema_candidate_wins(partial_vals[i], partial_idxs[i], local.val, local.idx)) {
            local.val = partial_vals[i];
            local.idx = partial_idxs[i];
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (extrema_candidate_wins(smem[threadIdx.x + stride].val, smem[threadIdx.x + stride].idx,
                                       smem[threadIdx.x].val, smem[threadIdx.x].idx))
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
        int64_t best_r   = -1;
        for (int64_t r = 0; r < reduce; ++r) {
            float v = __bfloat162float(input[base + r * inner]);
            if (extrema_candidate_wins(v, r, best_val, best_r)) { best_val = v; best_r = r; }
        }
        out_vals[idx] = __float2bfloat16(best_val);
        out_idxs[idx] = best_r;
    }
}

static std::pair<Tensor, Tensor> cuda_max_all(const Tensor& t,
                                               const ReductionPlan& plan) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    const int64_t n = plan.input_numel;
    const int num_blocks = static_cast<int>(
        std::min((n - 1) / kThreads + 1, kMaxBlocks));
    ReductionWorkspace workspace(t, num_blocks, true);
    float* p_vals = workspace.values;
    int64_t* p_idxs = workspace.indices;

    if (t.dtype() == DType::Float16) {
        Tensor out_val(std::make_shared<TensorImpl>(plan.output_shape, DType::Float16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64,   t.device()));

        reduce_max_kernel_half<<<num_blocks, kThreads, kThreads * sizeof(MaxPair), stream>>>(
            static_cast<const __half*>(t.data()),
            p_vals,
            p_idxs, n);
        CUDA_CHECK(cudaGetLastError());

        reduce_max_partials_kernel_half<<<1, kThreads, kThreads * sizeof(MaxPair), stream>>>(
            p_vals,
            p_idxs,
            static_cast<__half*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()), num_blocks);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    if (t.dtype() == DType::BFloat16) {
        Tensor out_val(std::make_shared<TensorImpl>(plan.output_shape, DType::BFloat16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64,    t.device()));

        reduce_max_kernel_bfloat16<<<num_blocks, kThreads, kThreads * sizeof(MaxPair), stream>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            p_vals,
            p_idxs, n);
        CUDA_CHECK(cudaGetLastError());

        reduce_max_partials_kernel_bfloat16<<<1, kThreads, kThreads * sizeof(MaxPair), stream>>>(
            p_vals,
            p_idxs,
            static_cast<__nv_bfloat16*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()), num_blocks);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    Tensor out_val(std::make_shared<TensorImpl>(plan.output_shape, DType::Float32, t.device()));
    Tensor out_idx(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64,   t.device()));

    reduce_max_kernel<<<num_blocks, kThreads, kThreads * sizeof(MaxPair), stream>>>(
        static_cast<const float*>(t.data()),
        p_vals,
        p_idxs, n);
    CUDA_CHECK(cudaGetLastError());

    reduce_max_partials_kernel<<<1, kThreads, kThreads * sizeof(MaxPair), stream>>>(
        p_vals,
        p_idxs,
        static_cast<float*>(out_val.data()),
        static_cast<int64_t*>(out_idx.data()), num_blocks);
    CUDA_CHECK(cudaGetLastError());

    return {out_val, out_idx};
}

static std::pair<Tensor, Tensor> cuda_max_dim(const Tensor& t,
                                               const ReductionPlan& plan) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    const auto outer = plan.outer, reduce = plan.reduce, inner = plan.inner;
    const int64_t n_out = plan.output_numel;

    if (t.dtype() == DType::Float16) {
        Tensor out_val(std::make_shared<TensorImpl>(plan.output_shape, DType::Float16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64,   t.device()));
        reduce_max_dim_kernel_half<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const __half*>(t.data()),
            static_cast<__half*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()),
            outer, reduce, inner, n_out);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    if (t.dtype() == DType::BFloat16) {
        Tensor out_val(std::make_shared<TensorImpl>(plan.output_shape, DType::BFloat16, t.device()));
        Tensor out_idx(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64,    t.device()));
        reduce_max_dim_kernel_bfloat16<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(t.data()),
            static_cast<__nv_bfloat16*>(out_val.data()),
            static_cast<int64_t*>(out_idx.data()),
            outer, reduce, inner, n_out);
        CUDA_CHECK(cudaGetLastError());
        return {out_val, out_idx};
    }

    Tensor out_val(std::make_shared<TensorImpl>(plan.output_shape, DType::Float32, t.device()));
    Tensor out_idx(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64,   t.device()));

    reduce_max_dim_kernel<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
        static_cast<const float*>(t.data()),
        static_cast<float*>(out_val.data()),
        static_cast<int64_t*>(out_idx.data()),
        outer, reduce, inner, n_out);
    CUDA_CHECK(cudaGetLastError());

    return {out_val, out_idx};
}

std::pair<Tensor, Tensor> cuda_max_with_indices(const Tensor& t, const ReductionPlan& plan) {
    if (!t.device().is_cuda() || !t.is_contiguous())
        throw std::invalid_argument("reduction: CUDA execution requires contiguous CUDA operand");
    validate_reduction_operand(t, plan, ReductionKind::Max);
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    if (plan.reduce_all) return cuda_max_all(t, plan);
    return cuda_max_dim(t, plan);
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
                           const ReductionPlan& plan) {
    DeviceGuard guard(grad_in.device());
    auto context = default_runtime().cuda(grad_in.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const auto outer = plan.outer, reduce = plan.reduce, inner = plan.inner;
    const int64_t n_out = plan.output_numel;
    if (grad_in.dtype() == DType::Float16) {
        max_scatter_grad_kernel_half<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const __half*>(grad_out.data()),
            static_cast<const int64_t*>(indices.data()),
            static_cast<__half*>(grad_in.data()),
            outer, reduce, inner, n_out);
    } else if (grad_in.dtype() == DType::BFloat16) {
        max_scatter_grad_kernel_bfloat16<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(grad_out.data()),
            static_cast<const int64_t*>(indices.data()),
            static_cast<__nv_bfloat16*>(grad_in.data()),
            outer, reduce, inner, n_out);
    } else {
        max_scatter_grad_kernel<<<cuda_blocks(n_out), kThreads, 0, stream>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<const int64_t*>(indices.data()),
            static_cast<float*>(grad_in.data()),
            outer, reduce, inner, n_out);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace {
ReductionPlan compatibility_plan(const Tensor& t, const std::vector<int64_t>& output,
    int64_t dim, bool all, ReductionKind kind) {
    // Legacy callers supplied the final shape; infer only keepdim, then let
    // normalization own every axis, shape and geometry decision.
    const auto plan = normalize_reduction(t, kind,
        all ? std::optional<int64_t>{} : std::optional<int64_t>{dim}, output.size() == t.shape().size());
    if (plan.output_shape != output)
        throw std::invalid_argument("reduction: compatibility output shape mismatch");
    return plan;
}
}
Tensor cuda_sum(const Tensor& t, const std::vector<int64_t>& output, int64_t dim, bool all) {
    return cuda_sum(t, compatibility_plan(t, output, dim, all, ReductionKind::Sum));
}
Tensor cuda_mean(const Tensor& t, const std::vector<int64_t>& output, int64_t dim, bool all) {
    return cuda_mean(t, compatibility_plan(t, output, dim, all, ReductionKind::Mean));
}
std::pair<Tensor, Tensor> cuda_max_with_indices(const Tensor& t,
    const std::vector<int64_t>& output, int64_t dim, bool all) {
    return cuda_max_with_indices(t, compatibility_plan(t, output, dim, all, ReductionKind::Max));
}
void cuda_max_scatter_grad(const Tensor& grad, const Tensor& indices, Tensor& input,
    int64_t outer, int64_t reduce, int64_t inner) {
    // The old native API supplies flattened geometry, rather than an axis.
    const auto plan = normalize_reduction({outer, reduce, inner}, input.device(),
        input.dtype(), ReductionKind::Max, 1, false);
    if (input.numel() != plan.input_numel || grad.numel() != plan.output_numel ||
        indices.numel() != plan.output_numel || grad.device() != plan.device ||
        indices.device() != plan.device || grad.dtype() != plan.dtype || indices.dtype() != DType::Int64)
        throw std::invalid_argument("reduction: compatibility scatter mismatch");
    cuda_max_scatter_grad(grad, indices, input, plan);
}

}  // namespace orthon
