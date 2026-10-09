#pragma once
#include <orthon/planning/matmul_plan.h>

namespace orthon {
class Runtime;
class Tensor;

struct MatmulPlannerPolicy {
    const uint64_t fp16_2d_cublas_threshold = 0;
    const uint64_t fp16_batched_cublas_threshold = uint64_t{16} << 20;
    const uint64_t bf16_2d_cublas_threshold = uint64_t{16} << 20;
    const uint64_t bf16_batched_cublas_threshold = uint64_t{16} << 20;
};
class MatmulPlanner {
public:
    explicit MatmulPlanner(MatmulPlannerPolicy policy = {}) : policy_(policy) {}
    MatmulPlan plan(const MatmulProblem&, Runtime&,
                    MatmulPreference = MatmulPreference::Automatic) const;
    const MatmulPlannerPolicy& policy() const noexcept { return policy_; }
private:
    const MatmulPlannerPolicy policy_;
};
const MatmulPlanner& default_matmul_planner();
MatmulProblem matmul_problem(const Tensor&, const Tensor&);
// Execution may have packed CUDA views; dimensions/dtype/exact device must agree.
bool matmul_plan_matches(const Tensor&, const Tensor&, const MatmulPlan&);

}  // namespace orthon
