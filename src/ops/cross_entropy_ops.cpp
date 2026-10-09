#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/cross_entropy_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/autograd/cross_entropy_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/dtype.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>


namespace orthon {

static void validate_cross_entropy(const Tensor& logits,
                                    const std::vector<int64_t>& targets) {
    if (logits.ndim() != 2)
        throw std::runtime_error("cross_entropy: logits must be 2D");
    if (logits.dtype() != DType::Float32)
        throw std::runtime_error("cross_entropy: logits must be float32");

    const int64_t B = logits.shape()[0];
    const int64_t C = logits.shape()[1];

    if (B == 0)
        throw std::runtime_error("cross_entropy: empty batch");
    if (static_cast<int64_t>(targets.size()) != B)
        throw std::runtime_error(
            "cross_entropy: targets length (" +
            std::to_string(targets.size()) + ") does not match batch_size (" +
            std::to_string(B) + ")");

    for (int64_t i = 0; i < static_cast<int64_t>(targets.size()); ++i) {
        if (targets[i] < 0 || targets[i] >= C)
            throw std::runtime_error(
                "cross_entropy: target " + std::to_string(targets[i]) +
                " at index " + std::to_string(i) +
                " is out of range [0, " + std::to_string(C) + ")");
    }
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets) {
    validate_cross_entropy(logits, targets);
    if (!logits.is_contiguous()) throw std::runtime_error(logits.device().is_cuda() ? "cross_entropy: CUDA path requires contiguous logits" : "cross_entropy: CPU path requires contiguous logits");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TargetsLossKernel>(unary_request(OperatorId::CrossEntropy, logits), default_runtime());
    Tensor out = kernel.call(logits, targets);
    if (is_grad_enabled() && logits.requires_grad()) {
        out.requires_grad_(true); out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<CrossEntropyNode>(logits, targets));
    }
    return out;
}
Tensor cross_entropy_backward(const Tensor& logits, const std::vector<int64_t>& targets, const Tensor& grad) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TargetsBackwardKernel>(unary_request(OperatorId::CrossEntropyBackward, logits), default_runtime());
    return kernel.call(logits, targets, grad);
}
} // namespace orthon
