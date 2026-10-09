#pragma once
#include <orthon/core/dtype.h>
#include <orthon/device/device.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace orthon {

enum class ReductionKind { Sum, Mean, Max, ArgMax };
enum class ExtremaPolicy { FirstLogicalMaximumFirstNaN };

// Immutable logical metadata. Packing, graph state and workspace are execution
// concerns and never belong to this value. Only normalization can construct it.
class ReductionPlan {
public:
    const ReductionKind kind;
    const Device device;
    const DType dtype;
    const std::vector<int64_t> input_shape, output_shape, broadcast_shape;
    const std::optional<int64_t> dim;
    const bool keepdim, reduce_all;
    const int64_t outer, reduce, inner, input_numel, output_numel;
    const ExtremaPolicy extrema_policy = ExtremaPolicy::FirstLogicalMaximumFirstNaN;
private:
    friend ReductionPlan normalize_reduction(const std::vector<int64_t>&, Device,
        DType, ReductionKind, std::optional<int64_t>, bool);
    ReductionPlan(ReductionKind kind, Device device, DType dtype,
        std::vector<int64_t> input, std::vector<int64_t> output,
        std::vector<int64_t> broadcast, std::optional<int64_t> dim, bool keepdim,
        int64_t outer, int64_t reduce, int64_t inner, int64_t input_numel,
        int64_t output_numel);
};

// Metadata-only normalization also works without allocating a Tensor or GPU.
ReductionPlan normalize_reduction(const std::vector<int64_t>& shape, Device device,
    DType dtype, ReductionKind kind, std::optional<int64_t> dim = {}, bool keepdim = false);

} // namespace orthon
