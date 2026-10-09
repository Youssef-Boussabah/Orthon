#pragma once
#include <orthon/core/tensor.h>
#include <orthon/reduction/reduction_plan.h>

namespace orthon {
struct ReductionResult {
    Tensor values;
    std::optional<Tensor> indices;
};

// Preserve the legacy empty/dtype/axis error order before backend resolution.
ReductionPlan normalize_reduction(const Tensor&, ReductionKind,
    std::optional<int64_t> dim = {}, bool keepdim = false);
void validate_reduction_operand(const Tensor&, const ReductionPlan&, ReductionKind);
void validate_max_scatter(const Tensor&, const Tensor&, const Tensor&, const ReductionPlan&);
} // namespace orthon
