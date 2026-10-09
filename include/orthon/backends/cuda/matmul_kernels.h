#pragma once
#include <orthon/core/tensor.h>
#include <orthon/planning/matmul_plan.h>
#include <string>

namespace orthon {

// Execution-only entry points. The selected plan is never recomputed here.
Tensor cuda_matmul_execute(const Tensor&, const Tensor&, const MatmulPlan&);
Tensor cuda_matmul_naive_execute(const Tensor&, const Tensor&);

// Private compatibility frontends validate metadata and pack views without graph
// attachment in src/diagnostics/matmul.cpp. Raw execution requires contiguous data.
// Naive CUDA matmul: one thread per output element, global-memory K-loop.
// Both tensors must be float32 on the same CUDA device; A=(M,K), B=(K,N).
Tensor cuda_matmul_naive(const Tensor& a, const Tensor& b);

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

// Private planner path and eligibility observations.
std::string cuda_matmul_dispatch_path(const Tensor& a, const Tensor& b);
std::string cuda_matmul_cublas_guard_report(const Tensor& a, const Tensor& b);

}  // namespace orthon
