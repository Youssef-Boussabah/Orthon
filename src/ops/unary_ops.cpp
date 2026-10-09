#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/activation_node.h>
#include <orthon/autograd/relu_node.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstdint>


namespace orthon {

static Tensor attach_activation_grad(const Tensor& input,
                                     Tensor out,
                                     ActivationKind kind,
                                     float alpha = 0.0f) {
    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ActivationNode>(input, kind, alpha));
    }
    return out;
}
Tensor sqrt_t(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(unary_request(OperatorId::Sqrt, t), default_runtime());
    Tensor out = kernel.call(t);
    return attach_activation_grad(t, out, ActivationKind::Sqrt);
}
Tensor relu(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(unary_request(OperatorId::Relu, t), default_runtime());
    Tensor out = kernel.call(t);
    if (is_grad_enabled() && t.requires_grad()) {
        out.requires_grad_(true); out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ReluNode>(t));
    }
    return out;
}
Tensor sigmoid(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(unary_request(OperatorId::Sigmoid, t), default_runtime());
    Tensor out = kernel.call(t);
    return attach_activation_grad(t, out, ActivationKind::Sigmoid);
}
Tensor tanh_t(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(unary_request(OperatorId::Tanh, t), default_runtime());
    Tensor out = kernel.call(t);
    return attach_activation_grad(t, out, ActivationKind::Tanh);
}
Tensor gelu(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(unary_request(OperatorId::Gelu, t), default_runtime());
    Tensor out = kernel.call(t);
    return attach_activation_grad(t, out, ActivationKind::Gelu);
}
Tensor exp_t(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(unary_request(OperatorId::Exp, t), default_runtime());
    Tensor out = kernel.call(t);
    return attach_activation_grad(t, out, ActivationKind::Exp);
}
Tensor leaky_relu(const Tensor& t, float negative_slope) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ParamUnaryKernel>(unary_request(OperatorId::LeakyRelu, t), default_runtime());
    return attach_activation_grad(t, kernel.call(t, negative_slope, 0), ActivationKind::LeakyRelu, negative_slope);
}
Tensor neg(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(
        unary_request(OperatorId::Neg, t), default_runtime());
    return attach_activation_grad(t, kernel.call(t), ActivationKind::Neg);
}
Tensor reciprocal_t(const Tensor& t) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnaryKernel>(
        unary_request(OperatorId::Reciprocal, t), default_runtime());
    return attach_activation_grad(t, kernel.call(t), ActivationKind::Reciprocal);
}
Tensor dropout(const Tensor& t, float p, uint64_t seed) {
    if (p < 0.0f || p >= 1.0f) throw std::runtime_error("dropout: p must be in [0, 1)");
    if (p == 0.0f) return t;
    if (t.dtype() != DType::Float32) throw std::runtime_error("dropout: only float32 supported");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ParamUnaryKernel>(unary_request(OperatorId::DropoutMask, t), default_runtime());
    Tensor mask = kernel.call(t, p, seed);
    return mul(t, mask);
}
Tensor sigmoid_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::SigmoidBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("sigmoid_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::SigmoidBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor tanh_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::TanhBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("tanh_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::TanhBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor leaky_relu_backward(const Tensor& input,
                           const Tensor& grad_output,
                           float negative_slope) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::LeakyReluBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("leaky_relu_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::LeakyReluBackward, input), default_runtime());
    return kernel.call(input, grad_output, negative_slope);
}
Tensor gelu_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::GeluBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("gelu_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::GeluBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor exp_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::ExpBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("exp_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::ExpBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor neg_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::NegBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("neg_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::NegBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor sqrt_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::SqrtBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("sqrt_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::SqrtBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor reciprocal_backward(const Tensor& input, const Tensor& grad_output) {
    using namespace dispatch;
    default_dispatcher().check_dtype(unary_request(OperatorId::ReciprocalBackward, input));
    if (input.device().is_cpu() && !grad_output.is_contiguous())
        throw std::runtime_error("reciprocal_backward: CPU grad_output must be contiguous");
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::ReciprocalBackward, input), default_runtime());
    return kernel.call(input, grad_output, 0.0f);
}
Tensor relu_backward(const Tensor& input, const Tensor& grad) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<BackwardKernel>(unary_request(OperatorId::ReluBackward, input), default_runtime());
    return kernel.call(input, grad, 0.0f);
}

} // namespace orthon
