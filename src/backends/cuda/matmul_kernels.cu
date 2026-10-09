#include <orthon/backends/cuda/matmul_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
#include <orthon/runtime/runtime.h>
#include <orthon/planning/matmul_planner.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t _status = (call); \
        if (_status != CUBLAS_STATUS_SUCCESS) \
            throw std::runtime_error( \
                std::string("cuBLAS error: ") + std::to_string(static_cast<int>(_status))); \
    } while (0)

namespace orthon {

// 16×16 thread block: threadIdx.x → column, threadIdx.y → row.
static constexpr int kTile = 16;
static cublasComputeType_t cublas_compute_type_for(DType dtype) {
#if CUBLAS_VERSION >= 11000
    if (dtype == DType::BFloat16)
        return CUBLAS_COMPUTE_32F_FAST_16BF;
#endif
    return CUBLAS_COMPUTE_32F;
}

static cudaDataType_t cublas_data_type_for(DType dtype) {
    if (dtype == DType::Float16)
        return CUDA_R_16F;
#if CUDART_VERSION >= 11000
    if (dtype == DType::BFloat16)
        return CUDA_R_16BF;
#endif
    throw std::runtime_error("cuBLAS matmul helper: unsupported dtype");
}

static void cublas_gemm_2d(const Tensor& a, const Tensor& b, Tensor& out) {
    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);
    const cudaDataType_t dtype = cublas_data_type_for(a.dtype());
    const cublasComputeType_t compute_type = cublas_compute_type_for(a.dtype());

    // The GEMM must be issued while the handle's device is current.
    DeviceGuard guard(a.device());
    auto context = default_runtime().cuda(a.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    auto lease = default_runtime().cuda(a.device().index)->acquire_cublas();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A(M,K) @ B(K,N) is equivalent to column-major
    // C^T(N,M) = B^T(N,K) @ A^T(K,M) over the same memory.
    CUBLAS_CHECK(cublasGemmEx(
        static_cast<cublasHandle_t>(lease.handle()),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        b.data(),
        dtype,
        N,
        a.data(),
        dtype,
        K,
        &beta,
        out.data(),
        dtype,
        N,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void cublas_gemm_batched(const Tensor& a, const Tensor& b, Tensor& out) {
    const int B = static_cast<int>(a.shape()[0]);
    const int M = static_cast<int>(a.shape()[1]);
    const int K = static_cast<int>(a.shape()[2]);
    const int N = static_cast<int>(b.shape()[2]);
    const cudaDataType_t dtype = cublas_data_type_for(a.dtype());
    const cublasComputeType_t compute_type = cublas_compute_type_for(a.dtype());

    // The GEMM must be issued while the handle's device is current.
    DeviceGuard guard(a.device());
    auto context = default_runtime().cuda(a.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());
    auto lease = default_runtime().cuda(a.device().index)->acquire_cublas();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const long long stride_a = static_cast<long long>(M) * K;
    const long long stride_b = static_cast<long long>(K) * N;
    const long long stride_c = static_cast<long long>(M) * N;

    // Row-major batched C = A(M,K) @ B(K,N) is equivalent to column-major
    // C^T(N,M) = B^T(N,K) @ A^T(K,M), with the same per-batch memory strides.
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        static_cast<cublasHandle_t>(lease.handle()),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        b.data(),
        dtype,
        N,
        stride_b,
        a.data(),
        dtype,
        K,
        stride_a,
        &beta,
        out.data(),
        dtype,
        N,
        stride_c,
        B,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Each thread computes exactly one C[row, col] by looping over K.
// Inputs are assumed row-major contiguous (stride = leading dimension).
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float*       __restrict__ C,
                                     int M, int K, int N) {
    int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    int col = (int)blockIdx.x * kTile + (int)threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int p = 0; p < K; ++p)
            acc += A[row * K + p] * B[p * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float*       __restrict__ C,
                                     int M, int K, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    int col = (int)blockIdx.x * kTile + (int)threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * kTile + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = t * kTile + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < kTile; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

__global__ void batched_matmul_tiled_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float*       __restrict__ C,
                                            int M, int K, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int batch = (int)blockIdx.z;
    const int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    const int col = (int)blockIdx.x * kTile + (int)threadIdx.x;
    const float* Ab = A + batch * M * K;
    const float* Bb = B + batch * K * N;
    float* Cb = C + batch * M * N;

    float acc = 0.0f;
    const int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;

        const int b_row = t * kTile + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < kTile; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        Cb[row * N + col] = acc;
}

// Float16 tiled matmul kernels with float32 accumulation.

__global__ void matmul_tiled_kernel_half(const __half* __restrict__ A,
                                          const __half* __restrict__ B,
                                          __half*       __restrict__ C,
                                          int M, int K, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    int col = (int)blockIdx.x * kTile + (int)threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * kTile + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? __half2float(A[row * K + a_col]) : 0.0f;

        int b_row = t * kTile + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? __half2float(B[b_row * N + col]) : 0.0f;

        __syncthreads();

        for (int k = 0; k < kTile; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = __float2half(acc);
}

__global__ void batched_matmul_tiled_kernel_half(const __half* __restrict__ A,
                                                  const __half* __restrict__ B,
                                                  __half*       __restrict__ C,
                                                  int M, int K, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int batch = (int)blockIdx.z;
    const int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    const int col = (int)blockIdx.x * kTile + (int)threadIdx.x;
    const __half* Ab = A + batch * M * K;
    const __half* Bb = B + batch * K * N;
    __half*       Cb = C + batch * M * N;

    float acc = 0.0f;
    const int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? __half2float(Ab[row * K + a_col]) : 0.0f;

        const int b_row = t * kTile + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? __half2float(Bb[b_row * N + col]) : 0.0f;

        __syncthreads();
        for (int k = 0; k < kTile; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        Cb[row * N + col] = __float2half(acc);
}

// BFloat16 tiled matmul kernels with float32 accumulation.

__global__ void matmul_tiled_kernel_bfloat16(const __nv_bfloat16* __restrict__ A,
                                               const __nv_bfloat16* __restrict__ B,
                                               __nv_bfloat16*       __restrict__ C,
                                               int M, int K, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    int col = (int)blockIdx.x * kTile + (int)threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * kTile + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? __bfloat162float(A[row * K + a_col]) : 0.0f;

        int b_row = t * kTile + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? __bfloat162float(B[b_row * N + col]) : 0.0f;

        __syncthreads();

        for (int k = 0; k < kTile; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = __float2bfloat16(acc);
}

__global__ void batched_matmul_tiled_kernel_bfloat16(const __nv_bfloat16* __restrict__ A,
                                                      const __nv_bfloat16* __restrict__ B,
                                                      __nv_bfloat16*       __restrict__ C,
                                                      int M, int K, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int batch = (int)blockIdx.z;
    const int row = (int)blockIdx.y * kTile + (int)threadIdx.y;
    const int col = (int)blockIdx.x * kTile + (int)threadIdx.x;
    const __nv_bfloat16* Ab = A + batch * M * K;
    const __nv_bfloat16* Bb = B + batch * K * N;
    __nv_bfloat16*       Cb = C + batch * M * N;

    float acc = 0.0f;
    const int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? __bfloat162float(Ab[row * K + a_col]) : 0.0f;

        const int b_row = t * kTile + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? __bfloat162float(Bb[b_row * N + col]) : 0.0f;

        __syncthreads();
        for (int k = 0; k < kTile; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        Cb[row * N + col] = __float2bfloat16(acc);
}

// Host-side CUDA matmul wrappers.

static Tensor execute_custom_2d(const Tensor& a, const Tensor& b) {
    DeviceGuard guard(a.device());
    auto context = default_runtime().cuda(a.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, a.dtype(), a.device()));

    dim3 block(kTile, kTile);
    dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
              static_cast<unsigned>((M + kTile - 1) / kTile));

    if (a.dtype() == DType::Float16) {
        matmul_tiled_kernel_half<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(out.data()),
            M, K, N);
    } else if (a.dtype() == DType::BFloat16) {
        matmul_tiled_kernel_bfloat16<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            M, K, N);
    } else {
        matmul_tiled_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, K, N);
    }

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_matmul_naive_execute(const Tensor& a, const Tensor& b) {
    if (!a.device().is_cuda() || a.ndim() != 2 || b.ndim() != 2 ||
        a.dtype() != DType::Float32 || b.dtype() != DType::Float32 ||
        a.device() != b.device() || a.shape()[1] != b.shape()[0] ||
        !a.is_contiguous() || !b.is_contiguous())
        throw std::invalid_argument("cuda naive matmul: invalid operands");

    DeviceGuard guard(a.device());
    auto context = default_runtime().cuda(a.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, a.dtype(), a.device()));

    dim3 block(kTile, kTile);
    dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
              static_cast<unsigned>((M + kTile - 1) / kTile));

    matmul_naive_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(out.data()),
        M, K, N);

    CUDA_CHECK(cudaGetLastError());
    return out;
}

static Tensor execute_custom_batched(const Tensor& a, const Tensor& b) {
    DeviceGuard guard(a.device());
    auto context = default_runtime().cuda(a.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int B = static_cast<int>(a.shape()[0]);
    const int M = static_cast<int>(a.shape()[1]);
    const int K = static_cast<int>(a.shape()[2]);
    const int N = static_cast<int>(b.shape()[2]);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{B, M, N}, a.dtype(), a.device()));

    dim3 block(kTile, kTile);
    dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
              static_cast<unsigned>((M + kTile - 1) / kTile),
              static_cast<unsigned>(B));

    if (a.dtype() == DType::Float16) {
        batched_matmul_tiled_kernel_half<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(out.data()),
            M, K, N);
    } else if (a.dtype() == DType::BFloat16) {
        batched_matmul_tiled_kernel_bfloat16<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            M, K, N);
    } else {
        batched_matmul_tiled_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, K, N);
    }

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_matmul_execute(const Tensor& a, const Tensor& b, const MatmulPlan& plan) {
    if (!a.device().is_cuda() || !a.is_contiguous() || !b.is_contiguous() ||
        (a.dtype() != DType::Float32 && a.dtype() != DType::Float16 && a.dtype() != DType::BFloat16) ||
        !matmul_plan_matches(a, b, plan))
        throw std::invalid_argument("cuda matmul: plan/operand mismatch");
    switch (plan.algorithm) {
    case MatmulAlgorithm::CudaCustomTiled:
        return plan.problem.rank == 2 ? execute_custom_2d(a, b)
                                      : execute_custom_batched(a, b);
    case MatmulAlgorithm::CudaCublas: {
        if (!plan.cublas_dtype_eligible || !plan.cublas_capability_eligible ||
            (a.dtype() != DType::Float16 && a.dtype() != DType::BFloat16))
            throw std::invalid_argument("cuda matmul: invalid cuBLAS plan");
        Tensor out(std::make_shared<TensorImpl>(
            plan.problem.rank == 2 ? std::vector<int64_t>{plan.problem.m, plan.problem.n}
                : std::vector<int64_t>{plan.problem.batch, plan.problem.m, plan.problem.n},
            a.dtype(), a.device()));
        if (plan.problem.rank == 2) cublas_gemm_2d(a, b, out);
        else cublas_gemm_batched(a, b, out);
        CUDA_CHECK(cudaGetLastError());
        return out;
    }
    default: throw std::invalid_argument("cuda matmul: invalid algorithm");
    }
}

}  // namespace orthon
