#pragma once
#include <orthon/core/tensor.h>
#include <string>

namespace orthon {

// Naive CUDA matmul: one thread per output element, global-memory K-loop.
// Preconditions (caller must verify): both tensors are contiguous float32 on
// the same CUDA device; A is (M,K), B is (K,N).
Tensor cuda_matmul_naive(const Tensor& a, const Tensor& b);

// Tiled CUDA matmul: 16×16 shared-memory tiles, coalesced global loads,
// zero-padding for non-multiple tile sizes.
// Preconditions as cuda_matmul_naive, but float16/bfloat16 are also accepted
// (dispatched to custom tiled kernels or cuBLAS).
Tensor cuda_matmul_tiled(const Tensor& a, const Tensor& b);

// Benchmark/regression helper for the custom tiled 2D kernel. Production
// dispatch may use library-backed implementations for selected dtypes.
Tensor cuda_matmul_tiled_custom(const Tensor& a, const Tensor& b);

// Benchmark/regression helper that forces the library-backed 2D GEMM path.
Tensor cuda_matmul_tiled_cublas(const Tensor& a, const Tensor& b);

// Batched tiled CUDA matmul for A=(B,M,K), B=(B,K,N), C=(B,M,N).
Tensor cuda_batched_matmul_tiled(const Tensor& a, const Tensor& b);

// Benchmark/regression helper for the custom tiled batched kernel. Production
// dispatch may use library-backed implementations for selected dtypes.
Tensor cuda_batched_matmul_tiled_custom(const Tensor& a, const Tensor& b);

// Benchmark/regression helper that forces the library-backed batched GEMM path.
Tensor cuda_batched_matmul_tiled_cublas(const Tensor& a, const Tensor& b);

// Debug helpers for benchmark labeling and CUDA-event timing around GEMM only.
bool cuda_matmul_uses_cublas(const Tensor& a, const Tensor& b);
std::string cuda_matmul_dispatch_path(const Tensor& a, const Tensor& b);
std::string cuda_matmul_cublas_guard_report(const Tensor& a, const Tensor& b);
double cuda_matmul_cublas_elapsed_ms(const Tensor& a, const Tensor& b, int reps);
std::string cuda_matmul_cublas_profile_report(const Tensor& a, const Tensor& b, int reps);

}  // namespace orthon
