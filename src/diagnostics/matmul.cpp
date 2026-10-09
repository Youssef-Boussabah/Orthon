#ifdef WITH_CUDA
#include <orthon/backends/cuda/matmul_kernels.h>
#include <orthon/core/execution_copy.h>
#include <orthon/planning/matmul_planner.h>
#include <orthon/runtime/runtime.h>
#include <sstream>
#include <stdexcept>

namespace orthon {
namespace {
const char* dtype_name(DType dtype) {
    return dtype == DType::Float16 ? "float16" : dtype == DType::BFloat16 ? "bfloat16" : "float32";
}
MatmulPlan diagnostic_plan(const Tensor& a, const Tensor& b, MatmulPreference preference,
                           int rank = 0, const char* name = "matmul diagnostic") {
    if (rank && (a.ndim() != rank || b.ndim() != rank))
        throw std::runtime_error(std::string(name) + ": operands must be " + std::to_string(rank) + "-D");
    if (!a.device().is_cuda() || !b.device().is_cuda())
        throw std::runtime_error(std::string(name) + ": operands must be CUDA tensors");
    if (a.device() != b.device())
        throw std::runtime_error(std::string(name) + ": operands must be on the same CUDA device");
    if (a.dtype() != b.dtype())
        throw std::runtime_error(std::string(name) + ": dtype mismatch");
    if (rank == 2 && a.shape()[1] != b.shape()[0])
        throw std::runtime_error(std::string(name) + ": inner dimensions do not match");
    return default_matmul_planner().plan(matmul_problem(a, b), default_runtime(), preference);
}
Tensor execute_diagnostic(const Tensor& a, const Tensor& b, const MatmulPlan& plan) {
    return cuda_matmul_execute(execution_operand(a), execution_operand(b), plan);
}
}  // namespace

Tensor cuda_batched_matmul_tiled(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::Automatic, 3);
    return execute_diagnostic(a, b, plan);
}
Tensor cuda_matmul_tiled_custom(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::ForceCustom, 2);
    return execute_diagnostic(a, b, plan);
}
Tensor cuda_batched_matmul_tiled_custom(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::ForceCustom, 3);
    return execute_diagnostic(a, b, plan);
}
Tensor cuda_matmul_tiled_cublas(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::ForceCublas, 2);
    return execute_diagnostic(a, b, plan);
}
Tensor cuda_batched_matmul_tiled_cublas(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::ForceCublas, 3);
    return execute_diagnostic(a, b, plan);
}
Tensor cuda_matmul_naive(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::ForceCustom, 2, "_matmul_naive");
    if (plan.problem.dtype != DType::Float32)
        throw std::runtime_error("_matmul_naive: only float32 supported");
    return cuda_matmul_naive_execute(execution_operand(a), execution_operand(b));
}
std::string cuda_matmul_dispatch_path(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::Automatic);
    if (plan.algorithm == MatmulAlgorithm::CudaCublas)
        return plan.problem.dtype == DType::BFloat16 ? "cublas-bfloat16-fast-bf16-tensor-op"
                                                    : "cublas-float16-tensor-op";
    return std::string("custom-tiled-") + dtype_name(plan.problem.dtype);
}
std::string cuda_matmul_cublas_guard_report(const Tensor& a, const Tensor& b) {
    const auto plan = diagnostic_plan(a, b, MatmulPreference::Automatic);
    const auto& p = plan.problem;
    std::ostringstream out;
    out << std::boolalpha
        << "enabled=" << (plan.algorithm == MatmulAlgorithm::CudaCublas)
        << "; a_dtype=" << dtype_name(p.dtype) << "; b_dtype=" << dtype_name(b.dtype())
        << "; output_dtype=" << dtype_name(p.dtype)
        << "; dtype_guard_ok=" << plan.cublas_dtype_eligible
        << "; cublas_dtype_eligible=" << plan.cublas_dtype_eligible
        << "; cublas_capability_eligible=" << plan.cublas_capability_eligible
        << "; bf16_runtime_ok=" << plan.cublas_capability_eligible
        << "; rank=" << p.rank << "; batch=" << p.batch
        << "; M=" << p.m << "; K=" << p.k << "; N=" << p.n
        << "; device=" << device_to_string(p.device) << "; cuda_ordinal=" << p.device.index
        << "; a_contiguous=" << p.lhs_contiguous << "; b_contiguous=" << p.rhs_contiguous
        << "; packs_lhs=" << plan.packs_lhs << "; packs_rhs=" << plan.packs_rhs
        << "; execution_contiguous=true"
        << "; work_items=" << plan.work_items << "; work_threshold=" << plan.auto_threshold
        << "; work_threshold_ok=" << plan.threshold_satisfied;
    return out.str();
}
}  // namespace orthon

#endif
