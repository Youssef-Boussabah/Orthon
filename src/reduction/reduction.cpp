#include <orthon/reduction/reduction.h>
#include <orthon/dispatch/dispatcher.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace orthon {
namespace {
const char* name(ReductionKind kind) {
    switch (kind) {
        case ReductionKind::Sum: return "sum";
        case ReductionKind::Mean: return "mean";
        case ReductionKind::Max: return "max";
        case ReductionKind::ArgMax: return "argmax";
    }
    throw std::invalid_argument("reduction: invalid kind");
}
dispatch::OperatorId operator_id(ReductionKind kind) {
    switch (kind) {
        case ReductionKind::Sum: return dispatch::OperatorId::Sum;
        case ReductionKind::Mean: return dispatch::OperatorId::Mean;
        case ReductionKind::Max: return dispatch::OperatorId::Max;
        case ReductionKind::ArgMax: return dispatch::OperatorId::ArgMax;
    }
    throw std::invalid_argument("reduction: invalid kind");
}
int64_t multiply(int64_t a, int64_t b) {
    if (b < 0) throw std::invalid_argument("reduction: negative dimension");
    if (b && a > (std::numeric_limits<int64_t>::max)() / b)
        throw std::overflow_error("reduction: geometry overflow");
    return a * b;
}
}
ReductionPlan::ReductionPlan(ReductionKind k, Device dev, DType dt,
    std::vector<int64_t> input, std::vector<int64_t> output,
    std::vector<int64_t> broadcast, std::optional<int64_t> axis, bool keep,
    int64_t o, int64_t r, int64_t i, int64_t n, int64_t out_n)
    : kind(k), device(dev), dtype(dt), input_shape(std::move(input)),
      output_shape(std::move(output)), broadcast_shape(std::move(broadcast)),
      dim(axis), keepdim(keep), reduce_all(!axis), outer(o), reduce(r), inner(i),
      input_numel(n), output_numel(out_n) {}

ReductionPlan normalize_reduction(const std::vector<int64_t>& shape, Device device,
    DType dtype, ReductionKind kind, std::optional<int64_t> dim, bool keepdim) {
    const std::string op = name(kind);
    if (std::find(shape.begin(), shape.end(), 0) != shape.end())
        throw std::runtime_error(op + ": cannot reduce empty tensor");
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (dim) {
        const int64_t original = *dim;
        if (*dim < 0) *dim += rank;
        if (*dim < 0 || *dim >= rank)
            throw std::out_of_range(op + ": dim " + std::to_string(original) +
                " out of range for tensor with " + std::to_string(rank) + " dimensions");
    }
    int64_t n = 1, outer = 1, inner = 1;
    for (auto extent : shape) n = multiply(n, extent);
    std::vector<int64_t> output, broadcast;
    int64_t reduce = n;
    if (dim) {
        for (int64_t d = 0; d < *dim; ++d) outer = multiply(outer, shape[d]);
        for (int64_t d = *dim + 1; d < rank; ++d) inner = multiply(inner, shape[d]);
        reduce = shape[*dim];
        broadcast = shape;
        broadcast[*dim] = 1;
        output = broadcast;
        if (!keepdim) output.erase(output.begin() + *dim);
    } else {
        broadcast.assign(shape.size(), 1);
        if (keepdim) output = broadcast;
    }
    const int64_t out_n = multiply(outer, inner);
    if (multiply(multiply(outer, reduce), inner) != n)
        throw std::logic_error("reduction: inconsistent geometry");
    return ReductionPlan(kind, device, dtype, shape, output, broadcast, dim,
                         keepdim, outer, reduce, inner, n, out_n);
}
ReductionPlan normalize_reduction(const Tensor& t, ReductionKind kind,
    std::optional<int64_t> dim, bool keepdim) {
    if (t.numel() == 0)
        throw std::runtime_error(std::string(name(kind)) + ": cannot reduce empty tensor");
    if (t.device().is_cpu())
        dispatch::default_dispatcher().check_dtype(dispatch::unary_request(operator_id(kind), t));
    return normalize_reduction(t.shape(), t.device(), t.dtype(), kind, dim, keepdim);
}
void validate_reduction_operand(const Tensor& t, const ReductionPlan& plan, ReductionKind kind) {
    const bool extrema = kind == ReductionKind::Max && plan.kind == ReductionKind::ArgMax;
    const bool supported = t.device().is_cpu()
        ? (t.dtype() == DType::Float32 || t.dtype() == DType::Float64)
        : (t.dtype() == DType::Float32 || t.dtype() == DType::Float16 || t.dtype() == DType::BFloat16);
    if ((!extrema && kind != plan.kind) || !supported || t.device() != plan.device ||
        t.dtype() != plan.dtype || t.shape() != plan.input_shape || t.numel() != plan.input_numel)
        throw std::invalid_argument("reduction: operand/plan mismatch");
}
void validate_max_scatter(const Tensor& input, const Tensor& grad, const Tensor& indices,
    const ReductionPlan& plan) {
    validate_reduction_operand(input, plan, ReductionKind::Max);
    if (grad.device() != plan.device || indices.device() != plan.device ||
        grad.dtype() != plan.dtype || indices.dtype() != DType::Int64 ||
        grad.shape() != plan.output_shape || indices.shape() != plan.output_shape ||
        !grad.is_contiguous() || !indices.is_contiguous())
        throw std::invalid_argument("reduction: scatter operand/plan mismatch");
}
} // namespace orthon
