#include <orthon/ops/cuda/matmul_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <chrono>
#include <unordered_map>

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
static constexpr int64_t kCublasWorkThreshold = static_cast<int64_t>(16) << 20;

#if CUDART_VERSION >= 11000
static constexpr bool kBuildHasCudaBf16 = true;
#else
static constexpr bool kBuildHasCudaBf16 = false;
#endif

#if CUBLAS_VERSION >= 11000
static constexpr bool kBuildHasCublasFastBf16 = true;
#else
static constexpr bool kBuildHasCublasFastBf16 = false;
#endif

static cublasHandle_t cublas_handle(int device_index) {
    thread_local std::unordered_map<int, cublasHandle_t> handles;
    auto it = handles.find(device_index);
    if (it == handles.end()) {
        // A cuBLAS handle is bound to whichever device is current when it is
        // created, and this branch only runs on a cache MISS. The guard makes
        // creation self-contained rather than dependent on whichever device
        // the caller happened to leave current.
        DeviceGuard guard(Device(DeviceType::CUDA, device_index));
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        it = handles.emplace(device_index, handle).first;
    }
    return it->second;
}

struct Bfloat16CublasSupport {
    bool initialized = false;
    bool supported = false;
    int major = -1;
    int minor = -1;
    int cublas_version = -1;
};

static Bfloat16CublasSupport bfloat16_cublas_support(const Device& device) {
    thread_local std::unordered_map<int, Bfloat16CublasSupport> cache;
    Bfloat16CublasSupport& cached = cache[device.index];
    if (cached.initialized)
        return cached;

    cached.initialized = true;
#if CUDART_VERSION >= 11000 && CUBLAS_VERSION >= 11000
    DeviceGuard guard(device);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device.index));
    cached.major = prop.major;
    cached.minor = prop.minor;

    CUBLAS_CHECK(cublasGetVersion(cublas_handle(device.index), &cached.cublas_version));
    cached.supported = cached.major >= 8 && cached.cublas_version >= 11000;
#endif
    return cached;
}

static bool cublas_bfloat16_supported(const Device& device) {
    return bfloat16_cublas_support(device).supported;
}

static bool use_library_bfloat16_matmul(const Tensor& a, int64_t work_items) {
    return a.dtype() == DType::BFloat16 &&
           work_items >= kCublasWorkThreshold &&
           cublas_bfloat16_supported(a.device());
}

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

static bool valid_matmul_shape(const Tensor& a, const Tensor& b);

static int64_t matmul_work_items(const Tensor& a, const Tensor& b) {
    if (a.ndim() == 2)
        return a.shape()[0] * a.shape()[1] * b.shape()[1];
    return a.shape()[0] * a.shape()[1] * a.shape()[2] * b.shape()[2];
}

static bool use_library_matmul(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device() || !a.device().is_cuda())
        return false;
    if (a.dtype() != b.dtype())
        return false;
    if (!valid_matmul_shape(a, b))
        return false;
    if (!a.is_contiguous() || !b.is_contiguous())
        return false;

    const int64_t work_items = matmul_work_items(a, b);
    if (a.dtype() == DType::Float16) {
        if (a.ndim() == 2)
            return true;
        return work_items >= kCublasWorkThreshold;
    }
    return use_library_bfloat16_matmul(a, work_items);
}

static std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt8: return "uint8";
        case DType::Int8: return "int8";
        case DType::Bool: return "bool";
    }
    return "unknown";
}

static bool valid_matmul_shape(const Tensor& a, const Tensor& b) {
    if (a.ndim() == 2 && b.ndim() == 2)
        return a.shape()[1] == b.shape()[0];
    if (a.ndim() == 3 && b.ndim() == 3)
        return a.shape()[0] == b.shape()[0] && a.shape()[2] == b.shape()[1];
    return false;
}

static double elapsed_us(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static void cublas_gemm_2d(const Tensor& a, const Tensor& b, Tensor& out) {
    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);
    const cudaDataType_t dtype = cublas_data_type_for(a.dtype());
    const cublasComputeType_t compute_type = cublas_compute_type_for(a.dtype());

    // The GEMM must be issued while the handle's device is current.
    DeviceGuard guard(a.device());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A(M,K) @ B(K,N) is equivalent to column-major
    // C^T(N,M) = B^T(N,K) @ A^T(K,M) over the same memory.
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle(a.device().index),
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
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const long long stride_a = static_cast<long long>(M) * K;
    const long long stride_b = static_cast<long long>(K) * N;
    const long long stride_c = static_cast<long long>(M) * N;

    // Row-major batched C = A(M,K) @ B(K,N) is equivalent to column-major
    // C^T(N,M) = B^T(N,K) @ A^T(K,M), with the same per-batch memory strides.
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas_handle(a.device().index),
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

// The benchmark entry points below read shape()[0]/shape()[1] and launch
// directly, so they must reject anything that is not a well-formed 2-D
// CUDA GEMM before touching the shapes.
static void require_benchmark_gemm_operands(const Tensor& a, const Tensor& b,
                                            const char* name,
                                            bool float32_only) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error(std::string(name) + ": operands must be 2-D");
    if (!a.device().is_cuda() || !b.device().is_cuda())
        throw std::runtime_error(std::string(name) + ": operands must be CUDA tensors");
    if (a.device() != b.device())
        throw std::runtime_error(std::string(name) + ": operands must be on the same CUDA device");
    if (!a.is_contiguous() || !b.is_contiguous())
        throw std::runtime_error(std::string(name) + ": operands must be contiguous");
    if (a.dtype() != b.dtype())
        throw std::runtime_error(std::string(name) + ": dtype mismatch");
    if (float32_only) {
        if (a.dtype() != DType::Float32)
            throw std::runtime_error(std::string(name) + ": only float32 supported");
    } else if (a.dtype() != DType::Float32 &&
               a.dtype() != DType::Float16 &&
               a.dtype() != DType::BFloat16) {
        throw std::runtime_error(
            std::string(name) + ": only float32/float16/bfloat16 supported");
    }
    if (a.shape()[1] != b.shape()[0])
        throw std::runtime_error(
            std::string(name) + ": inner dimensions do not match");
}

Tensor cuda_matmul_tiled_custom(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, a.dtype(), a.device()));

    dim3 block(kTile, kTile);
    dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
              static_cast<unsigned>((M + kTile - 1) / kTile));

    if (a.dtype() == DType::Float16) {
        matmul_tiled_kernel_half<<<grid, block>>>(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(out.data()),
            M, K, N);
    } else if (a.dtype() == DType::BFloat16) {
        matmul_tiled_kernel_bfloat16<<<grid, block>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            M, K, N);
    } else {
        matmul_tiled_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, K, N);
    }

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_matmul_tiled(const Tensor& a, const Tensor& b) {
    require_benchmark_gemm_operands(a, b, "_matmul_tiled", /*float32_only=*/false);

    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);
    const int64_t work_items = static_cast<int64_t>(M) * K * N;

    if (a.dtype() != DType::Float16 &&
        !use_library_bfloat16_matmul(a, work_items))
        return cuda_matmul_tiled_custom(a, b);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, a.dtype(), a.device()));

    cublas_gemm_2d(a, b, out);

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_matmul_tiled_cublas(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    if (!use_library_matmul(a, b))
        throw std::runtime_error(
            "cuda_matmul_tiled_cublas: cuBLAS path is not enabled for this input; " +
            cuda_matmul_cublas_guard_report(a, b));
    const int M = static_cast<int>(a.shape()[0]);
    const int N = static_cast<int>(b.shape()[1]);
    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, a.dtype(), a.device()));
    cublas_gemm_2d(a, b, out);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_matmul_naive(const Tensor& a, const Tensor& b) {
    require_benchmark_gemm_operands(a, b, "_matmul_naive", /*float32_only=*/true);

    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[1]);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{M, N}, a.dtype(), a.device()));

    dim3 block(kTile, kTile);
    dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
              static_cast<unsigned>((M + kTile - 1) / kTile));

    matmul_naive_kernel<<<grid, block>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(out.data()),
        M, K, N);

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_batched_matmul_tiled_custom(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

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
        batched_matmul_tiled_kernel_half<<<grid, block>>>(
            static_cast<const __half*>(a.data()),
            static_cast<const __half*>(b.data()),
            static_cast<__half*>(out.data()),
            M, K, N);
    } else if (a.dtype() == DType::BFloat16) {
        batched_matmul_tiled_kernel_bfloat16<<<grid, block>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            M, K, N);
    } else {
        batched_matmul_tiled_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, K, N);
    }

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_batched_matmul_tiled(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    const int B = static_cast<int>(a.shape()[0]);
    const int M = static_cast<int>(a.shape()[1]);
    const int K = static_cast<int>(a.shape()[2]);
    const int N = static_cast<int>(b.shape()[2]);

    const int64_t work_items = static_cast<int64_t>(B) * M * N * K;

    if (a.dtype() != DType::Float16 &&
        !use_library_bfloat16_matmul(a, work_items))
        return cuda_batched_matmul_tiled_custom(a, b);

    if (work_items < kCublasWorkThreshold)
        return cuda_batched_matmul_tiled_custom(a, b);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{B, M, N}, a.dtype(), a.device()));

    cublas_gemm_batched(a, b, out);

    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_batched_matmul_tiled_cublas(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    if (!use_library_matmul(a, b))
        throw std::runtime_error(
            "cuda_batched_matmul_tiled_cublas: cuBLAS path is not enabled for this input; " +
            cuda_matmul_cublas_guard_report(a, b));
    const int B = static_cast<int>(a.shape()[0]);
    const int M = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[2]);
    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{B, M, N}, a.dtype(), a.device()));
    cublas_gemm_batched(a, b, out);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

bool cuda_matmul_uses_cublas(const Tensor& a, const Tensor& b) {
    return use_library_matmul(a, b);
}

std::string cuda_matmul_dispatch_path(const Tensor& a, const Tensor& b) {
    const char* dtype = "float32";
    if (a.dtype() == DType::Float16)
        dtype = "float16";
    else if (a.dtype() == DType::BFloat16)
        dtype = "bfloat16";

    if (use_library_matmul(a, b)) {
        if (a.dtype() == DType::BFloat16)
            return std::string("cublas-") + dtype + "-fast-bf16-tensor-op";
        return std::string("cublas-") + dtype + "-tensor-op";
    }
    return std::string("custom-tiled-") + dtype;
}

std::string cuda_matmul_cublas_guard_report(const Tensor& a, const Tensor& b) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    const int64_t work_items = valid_matmul_shape(a, b) ? matmul_work_items(a, b) : -1;
    int device_major = -1;
    int device_minor = -1;
    int cublas_version = -1;
    if (a.device().is_cuda()) {
        const Bfloat16CublasSupport support = bfloat16_cublas_support(a.device());
        device_major = support.major;
        device_minor = support.minor;
        cublas_version = support.cublas_version;
    }

    const bool same_dtype = a.dtype() == b.dtype();
    const bool supported_dtype = a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16;
    const bool bf16 = a.dtype() == DType::BFloat16;
    const bool same_device = a.device() == b.device();
    const bool cuda_device = a.device().is_cuda();
    const bool valid_ndim = (a.ndim() == 2 && b.ndim() == 2) || (a.ndim() == 3 && b.ndim() == 3);
    const bool shape_ok = valid_matmul_shape(a, b);
    const bool contiguous_ok = a.is_contiguous() && b.is_contiguous();
    const bool cc_ok = cuda_device && device_major >= 8;
    const bool cublas_version_ok = cublas_version >= 11000;
    const bool threshold_ok = work_items >= kCublasWorkThreshold;
    const bool bf16_build_ok = !bf16 || (kBuildHasCudaBf16 && kBuildHasCublasFastBf16);
    const bool dtype_guard_ok = same_dtype && supported_dtype;
    const bool bf16_runtime_ok = !bf16 || (cc_ok && cublas_version_ok);
    const bool enabled = use_library_matmul(a, b);

    std::ostringstream oss;
    oss << "enabled=" << (enabled ? "true" : "false")
        << "; a_dtype=" << dtype_name(a.dtype())
        << "; b_dtype=" << dtype_name(b.dtype())
        << "; output_dtype=" << dtype_name(a.dtype())
        << "; same_dtype=" << (same_dtype ? "true" : "false")
        << "; supported_dtype=" << (supported_dtype ? "true" : "false")
        << "; ndim=(" << a.ndim() << "," << b.ndim() << ")"
        << "; valid_ndim=" << (valid_ndim ? "true" : "false")
        << "; shape_ok=" << (shape_ok ? "true" : "false")
        << "; a_contiguous=" << (a.is_contiguous() ? "true" : "false")
        << "; b_contiguous=" << (b.is_contiguous() ? "true" : "false")
        << "; contiguity_ok=" << (contiguous_ok ? "true" : "false")
        << "; same_device=" << (same_device ? "true" : "false")
        << "; cuda_device=" << (cuda_device ? "true" : "false")
        << "; compute_capability=" << device_major << "." << device_minor
        << "; compute_capability_ok=" << (cc_ok ? "true" : "false")
        << "; CUDA_R_16BF_available=" << (kBuildHasCudaBf16 ? "true" : "false")
        << "; CUBLAS_COMPUTE_32F_FAST_16BF_available=" << (kBuildHasCublasFastBf16 ? "true" : "false")
        << "; cublas_version=" << cublas_version
        << "; cublas_version_ok=" << (cublas_version_ok ? "true" : "false")
        << "; work_items=" << work_items
        << "; work_threshold=" << kCublasWorkThreshold
        << "; work_threshold_ok=" << (threshold_ok ? "true" : "false")
        << "; dtype_guard_ok=" << (dtype_guard_ok ? "true" : "false")
        << "; bf16_build_ok=" << (bf16_build_ok ? "true" : "false")
        << "; bf16_runtime_ok=" << (bf16_runtime_ok ? "true" : "false");
    return oss.str();
}

double cuda_matmul_cublas_elapsed_ms(const Tensor& a, const Tensor& b, int reps) {
    if (reps <= 0)
        throw std::runtime_error("cuda_matmul_cublas_elapsed_ms: reps must be positive");
    if (!use_library_matmul(a, b))
        throw std::runtime_error(
            "cuda_matmul_cublas_elapsed_ms: cuBLAS path is not enabled for this input; " +
            cuda_matmul_cublas_guard_report(a, b));

    Tensor out = [&]() -> Tensor {
        if (a.ndim() == 2) {
            return Tensor(std::make_shared<TensorImpl>(
                std::vector<int64_t>{a.shape()[0], b.shape()[1]}, a.dtype(), a.device()));
        }
        return Tensor(std::make_shared<TensorImpl>(
            std::vector<int64_t>{a.shape()[0], a.shape()[1], b.shape()[2]},
            a.dtype(),
            a.device()));
    }();

    DeviceGuard guard(a.device());
    for (int i = 0; i < 5; ++i) {
        if (a.ndim() == 2)
            cublas_gemm_2d(a, b, out);
        else
            cublas_gemm_batched(a, b, out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        if (a.ndim() == 2)
            cublas_gemm_2d(a, b, out);
        else
            cublas_gemm_batched(a, b, out);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed) / static_cast<double>(reps);
}

std::string cuda_matmul_cublas_profile_report(const Tensor& a, const Tensor& b, int reps) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(a.device());

    if (reps <= 0)
        throw std::runtime_error("cuda_matmul_cublas_profile_report: reps must be positive");

    const auto guard_start = std::chrono::steady_clock::now();
    const bool enabled = use_library_matmul(a, b);
    const auto guard_end = std::chrono::steady_clock::now();
    if (!enabled)
        throw std::runtime_error(
            "cuda_matmul_cublas_profile_report: cuBLAS path is not enabled for this input; " +
            cuda_matmul_cublas_guard_report(a, b));

    const auto allocation_start = std::chrono::steady_clock::now();
    Tensor out = [&]() -> Tensor {
        if (a.ndim() == 2) {
            return Tensor(std::make_shared<TensorImpl>(
                std::vector<int64_t>{a.shape()[0], b.shape()[1]}, a.dtype(), a.device()));
        }
        return Tensor(std::make_shared<TensorImpl>(
            std::vector<int64_t>{a.shape()[0], a.shape()[1], b.shape()[2]},
            a.dtype(),
            a.device()));
    }();
    const auto allocation_end = std::chrono::steady_clock::now();

    const auto handle_start = std::chrono::steady_clock::now();
    (void)cublas_handle(a.device().index);
    const auto handle_end = std::chrono::steady_clock::now();

    const auto event_create_start = std::chrono::steady_clock::now();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const auto event_create_end = std::chrono::steady_clock::now();

    const auto pre_sync_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto pre_sync_end = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        if (a.ndim() == 2)
            cublas_gemm_2d(a, b, out);
        else
            cublas_gemm_batched(a, b, out);
    }
    CUDA_CHECK(cudaEventRecord(stop));

    const auto event_sync_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto event_sync_end = std::chrono::steady_clock::now();

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

    const auto event_destroy_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    const auto event_destroy_end = std::chrono::steady_clock::now();

    std::ostringstream oss;
    oss << "enabled=true"
        << "; reps=" << reps
        << "; guard_us=" << elapsed_us(guard_start, guard_end)
        << "; output_allocation_us=" << elapsed_us(allocation_start, allocation_end)
        << "; cublas_handle_acquire_us=" << elapsed_us(handle_start, handle_end)
        << "; cuda_event_create_us=" << elapsed_us(event_create_start, event_create_end)
        << "; cuda_device_synchronize_us=" << elapsed_us(pre_sync_start, pre_sync_end)
        << "; cublas_gemm_event_ms_per_call=" << (static_cast<double>(elapsed) / reps)
        << "; cuda_event_synchronize_us=" << elapsed_us(event_sync_start, event_sync_end)
        << "; cuda_event_destroy_us=" << elapsed_us(event_destroy_start, event_destroy_end);
    return oss.str();
}

}  // namespace orthon
