#pragma once
#include <orthon/core/dtype.h>
#include <orthon/device/device.h>
#include <cstdint>

namespace orthon {

// Logical metadata only; never retains operands or execution resources.
struct MatmulProblem {
    const Device device;
    const DType dtype;
    const int64_t rank, batch, m, k, n;
    const bool lhs_contiguous, rhs_contiguous;
};
enum class MatmulAlgorithm { CpuReference, CudaCustomTiled, CudaCublas };
enum class MatmulPreference { Automatic, ForceCustom, ForceCublas };
struct MatmulPlan {
    const MatmulProblem problem;
    const MatmulAlgorithm algorithm;
    const MatmulPreference preference;
    const uint64_t work_items, auto_threshold;
    const bool cublas_dtype_eligible, cublas_capability_eligible, threshold_satisfied;
    const bool packs_lhs, packs_rhs;
};

}  // namespace orthon
