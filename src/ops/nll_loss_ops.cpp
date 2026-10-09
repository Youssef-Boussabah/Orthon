#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/nll_loss_ops.h>
#include <orthon/planning/index_validation.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/nll_loss_node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <cstring>
#include <stdexcept>
#include <string>


namespace orthon {

Tensor nll_loss(const Tensor& log_probs,
                const Tensor& targets,
                const std::string& reduction) {
    validate_nll_inputs(log_probs, targets, reduction);

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TensorLossKernel>(unary_request(OperatorId::NllLoss, log_probs), default_runtime());
    Tensor out = kernel.call(log_probs, targets, reduction);
    if (is_grad_enabled() && log_probs.requires_grad()) {
        out.requires_grad_(true); out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<NllLossNode>(log_probs, targets, reduction));
    }
    return out;
}

Tensor nll_loss_backward(const Tensor& log_probs,
                         const Tensor& targets,
                         const Tensor& grad_output,
                         const std::string& reduction) {
    validate_nll_inputs(log_probs, targets, reduction, &grad_output);

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TensorLossBackwardKernel>(unary_request(OperatorId::NllLossBackward, log_probs), default_runtime());
    Tensor out = kernel.call(log_probs, targets, grad_output, reduction);
    return out;
}

}  // namespace orthon
