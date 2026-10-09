#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/maxpool2d_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/maxpool2d_node.h>
#include <orthon/core/dtype.h>
#include <orthon/planning/maxpool2d_validation.h>
#include <algorithm>
#include <limits>
#include <stdexcept>


namespace orthon {

Tensor maxpool2d(const Tensor& input,
                 int64_t kh, int64_t kw,
                 int64_t sh, int64_t sw,
                 int64_t ph, int64_t pw) {
    const dispatch::SpatialArgs args{sh, sw, ph, pw, kh, kw};
    validate_maxpool2d_input(input, args);

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<PoolKernel>(unary_request(OperatorId::MaxPool2d, input), default_runtime());
    Tensor out = kernel.call(input, args);
    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true); out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<MaxPool2DNode>(input, kh, kw, sh, sw, ph, pw));
    }
    return out;
}
Tensor maxpool2d_backward(const Tensor& input, const Tensor& grad, const dispatch::SpatialArgs& args) {
    validate_maxpool2d_gradient(input, grad, args);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<PoolBackwardKernel>(unary_request(OperatorId::MaxPool2dBackward, input), default_runtime());
    return kernel.call(input, grad, args);
}
} // namespace orthon
