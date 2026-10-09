#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/fusion_ops.h>
#include <orthon/autograd/fused_linear_relu_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/core/dtype.h>
#include <stdexcept>
#include <string>


namespace orthon {

static void validate_fused_linear_relu_shapes(const Tensor& x,
                                              const Tensor& weight,
                                              const Tensor& bias) {
    if (x.ndim() != 2 || weight.ndim() != 2)
        throw std::runtime_error("fused_linear_relu: x and weight must be 2D");
    if (bias.ndim() != 1)
        throw std::runtime_error("fused_linear_relu: bias must be 1D");
    if (x.shape()[1] != weight.shape()[0])
        throw std::runtime_error(
            "fused_linear_relu: shape mismatch: x second dim must match weight first dim");
    if (weight.shape()[1] != bias.shape()[0])
        throw std::runtime_error(
            "fused_linear_relu: shape mismatch: bias length must match output dim");
}

static void validate_fused_linear_relu_common(const Tensor& x,
                                              const Tensor& weight,
                                              const Tensor& bias) {
    validate_fused_linear_relu_shapes(x, weight, bias);
    if (x.device() != weight.device() || x.device() != bias.device())
        throw std::runtime_error("fused_linear_relu: device mismatch");
    if (x.dtype() != DType::Float32 ||
        weight.dtype() != DType::Float32 ||
        bias.dtype() != DType::Float32)
        throw std::runtime_error("fused_linear_relu: only float32 supported");
}

static void attach_fused_linear_relu_grad_fn(Tensor& out,
                                             const Tensor& x,
                                             const Tensor& weight,
                                             const Tensor& bias) {
    const bool req = is_grad_enabled() &&
        (x.requires_grad() || weight.requires_grad() || bias.requires_grad());
    out.requires_grad_(req);
    if (req) {
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<FusedLinearReLUNode>(x, weight, bias));
    }
}

Tensor fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    validate_fused_linear_relu_common(x, weight, bias);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TernaryKernel>(unary_request(OperatorId::FusedLinearRelu, x), default_runtime());
    Tensor out = kernel.call(x, weight, bias);
    attach_fused_linear_relu_grad_fn(out, x, weight, bias);
    return out;
}
} // namespace orthon
