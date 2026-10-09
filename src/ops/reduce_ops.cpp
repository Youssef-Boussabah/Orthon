#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/reduce_node.h>
#include <orthon/autograd/max_node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/ops/fill_ops.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace orthon {

static Tensor attach_reduce_grad(const Tensor& input, Tensor out, const ReductionPlan& plan) {
    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ReduceNode>(input, plan));
    }
    return out;
}

static Tensor reduce_impl(const Tensor& t, std::optional<int64_t> dim,
                          bool keepdim, bool is_mean) {
    using namespace dispatch;
    const auto id = is_mean ? OperatorId::Mean : OperatorId::Sum;
    const auto plan = normalize_reduction(t, is_mean ? ReductionKind::Mean : ReductionKind::Sum, dim, keepdim);
    const auto& kernel = default_dispatcher().resolve<ReductionKernel>(unary_request(id, t), default_runtime());
    auto result = kernel.call(t, plan);
    return attach_reduce_grad(t, result.values, plan);
}
Tensor sum(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    return reduce_impl(t, dim, keepdim, false);
}
Tensor mean(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    return reduce_impl(t, dim, keepdim, true);
}

// Max and argmax helpers.


Tensor reduce_max(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    using namespace dispatch;
    const auto plan = normalize_reduction(t, ReductionKind::Max, dim, keepdim);
    const auto& kernel = default_dispatcher().resolve<ReductionKernel>(
        unary_request(OperatorId::Max, t), default_runtime());
    auto result = kernel.call(t, plan);
    if (!result.indices) throw std::logic_error("max: internal kernel omitted indices");
    Tensor val_t = result.values;
    if (is_grad_enabled() && t.requires_grad()) {
        val_t.requires_grad_(true);
        val_t.set_is_leaf(false);
        val_t.set_grad_fn(std::make_shared<MaxNode>(
            t, *result.indices, plan));
    }
    return val_t;
}
Tensor argmax(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    using namespace dispatch;
    const auto plan = normalize_reduction(t, ReductionKind::ArgMax, dim, keepdim);
    const auto& kernel = default_dispatcher().resolve<ReductionKernel>(
        unary_request(OperatorId::ArgMax, t), default_runtime());
    auto result = kernel.call(t, plan);
    if (!result.indices) throw std::logic_error("argmax: internal kernel omitted indices");
    return *result.indices;
}

}  // namespace orthon
