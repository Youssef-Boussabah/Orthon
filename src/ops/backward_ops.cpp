#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/backward_ops.h>
namespace orthon {
Tensor unbroadcast_values(const Tensor& grad, const std::vector<int64_t>& target_shape) {
    if (grad.shape() == target_shape) return grad;
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ShapeKernel>(unary_request(OperatorId::Unbroadcast, grad), default_runtime());
    return kernel.call(grad, target_shape);
}
Tensor max_backward(const Tensor& input, const Tensor& grad, const Tensor& indices, const ReductionPlan& args) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<MaxScatterKernel>(unary_request(OperatorId::MaxBackward, input), default_runtime());
    return kernel.call(input, grad, indices, args);
}
}
