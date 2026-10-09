#include <orthon/planning/matmul_planner.h>
#include <orthon/core/tensor.h>
#include <orthon/runtime/runtime.h>
#include <limits>
#include <stdexcept>

namespace orthon {
MatmulProblem matmul_problem(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device() || a.dtype() != b.dtype() ||
        a.ndim() != b.ndim() || (a.ndim() != 2 && a.ndim() != 3) ||
        a.shape().back() != b.shape()[b.ndim() - 2] ||
        (a.ndim() == 3 && a.shape()[0] != b.shape()[0]))
        throw std::invalid_argument("matmul planner: incompatible operands");
    return {a.device(), a.dtype(), a.ndim(), a.ndim() == 3 ? a.shape()[0] : 1,
            a.shape()[a.ndim() - 2], a.shape().back(), b.shape().back(),
            a.is_contiguous(), b.is_contiguous()};
}
bool matmul_plan_matches(const Tensor& a, const Tensor& b, const MatmulPlan& plan) {
    try {
        const auto p = matmul_problem(a, b);
        const auto& q = plan.problem;
        return p.device == q.device && p.dtype == q.dtype && p.rank == q.rank &&
               p.batch == q.batch && p.m == q.m && p.k == q.k && p.n == q.n;
    } catch (const std::invalid_argument&) { return false; }
}
MatmulPlan MatmulPlanner::plan(const MatmulProblem& p, Runtime& runtime,
                              MatmulPreference preference) const {
    if (preference != MatmulPreference::Automatic && preference != MatmulPreference::ForceCustom &&
        preference != MatmulPreference::ForceCublas)
        throw std::invalid_argument("matmul planner: invalid preference");
    if ((p.rank != 2 && p.rank != 3) || p.batch < 0 || p.m < 0 || p.k < 0 ||
        p.n < 0 || (p.rank == 2 && p.batch != 1))
        throw std::invalid_argument("matmul planner: invalid problem dimensions");
    Device(p.device.type, p.device.index);
    uint64_t work = 0;
    // A zero factor wins even if an earlier partial product would overflow.
    if (p.batch && p.m && p.k && p.n) {
        work = 1;
        for (auto factor : {p.batch, p.m, p.k, p.n}) {
            if (static_cast<uint64_t>(factor) > std::numeric_limits<uint64_t>::max() / work)
                throw std::overflow_error("matmul planner: work items overflow");
            work *= static_cast<uint64_t>(factor);
        }
    }
    if (p.device.is_cpu()) {
        if (p.dtype != DType::Float32 && p.dtype != DType::Float64)
            throw std::invalid_argument("matmul planner: unsupported CPU dtype");
        if (preference != MatmulPreference::Automatic)
            throw std::invalid_argument("matmul planner: forced CUDA algorithm on CPU");
        return {p, MatmulAlgorithm::CpuReference, preference, work, 0,
                false, false, true, false, false};
    }
    if (p.dtype != DType::Float32 && p.dtype != DType::Float16 && p.dtype != DType::BFloat16)
        throw std::invalid_argument("matmul planner: unsupported CUDA dtype");
    for (auto dimension : {p.batch, p.m, p.k, p.n})
        if (dimension > std::numeric_limits<int>::max())
            throw std::invalid_argument("matmul planner: CUDA dimension exceeds execution range");
    const bool dtype_ok = p.dtype != DType::Float32;
    const bool capability_ok = p.dtype != DType::BFloat16 ||
        runtime.cuda(p.device.index)->capabilities().cublas_bfloat16;
    const uint64_t threshold = p.dtype == DType::Float16
        ? (p.rank == 2 ? policy_.fp16_2d_cublas_threshold : policy_.fp16_batched_cublas_threshold)
        : (p.rank == 2 ? policy_.bf16_2d_cublas_threshold : policy_.bf16_batched_cublas_threshold);
    const bool threshold_ok = work >= threshold;
    const bool hard_ok = dtype_ok && capability_ok;
    if (preference == MatmulPreference::ForceCublas && !hard_ok)
        throw std::invalid_argument("matmul planner: forced cuBLAS requires eligible dtype and capability");
    const bool library = preference == MatmulPreference::ForceCublas ||
        (preference == MatmulPreference::Automatic && hard_ok && threshold_ok);
    return {p, library ? MatmulAlgorithm::CudaCublas : MatmulAlgorithm::CudaCustomTiled,
            preference, work, threshold, dtype_ok, capability_ok, threshold_ok,
            !p.lhs_contiguous, !p.rhs_contiguous};
}
const MatmulPlanner& default_matmul_planner() {
    static const MatmulPlanner planner;
    return planner;
}
}  // namespace orthon
