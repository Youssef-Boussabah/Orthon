#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/conv2d_ops.h>
#include <orthon/planning/conv2d_validation.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/autograd/conv2d_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>


namespace orthon {

static void attach_conv2d_grad_fn(Tensor& out,
                                  const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& bias,
                                  int64_t sh,
                                  int64_t sw,
                                  int64_t ph,
                                  int64_t pw) {
    const bool req = is_grad_enabled() &&
        (input.requires_grad() || weight.requires_grad() || bias.requires_grad());
    if (req) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<Conv2DNode>(input, weight, bias, sh, sw, ph, pw));
    }
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    validate_conv2d_forward(input, weight, bias, sh, sw, ph, pw);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ConvolutionKernel>(unary_request(OperatorId::Conv2d, input), default_runtime());
    Tensor out = kernel.call(input, weight, bias, {sh, sw, ph, pw});
    attach_conv2d_grad_fn(out, input, weight, bias, sh, sw, ph, pw);
    return out;
}
Tensor conv2d_direct(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    validate_conv2d_forward(input, weight, bias, sh, sw, ph, pw);
    if (!input.device().is_cpu()) throw std::runtime_error("conv2d_direct: only CPU supported");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ConvolutionKernel>(unary_request(OperatorId::Conv2dDirect, input), default_runtime());
    Tensor out = kernel.call(input, weight, bias, {sh, sw, ph, pw});
    attach_conv2d_grad_fn(out, input, weight, bias, sh, sw, ph, pw);
    return out;
}
Tensor conv2d_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    validate_conv2d_forward(input, weight, bias, sh, sw, ph, pw);
    if (!input.device().is_cpu()) throw std::runtime_error("conv2d_im2col: only CPU supported");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ConvolutionKernel>(unary_request(OperatorId::Conv2dIm2col, input), default_runtime());
    Tensor out = kernel.call(input, weight, bias, {sh, sw, ph, pw});
    attach_conv2d_grad_fn(out, input, weight, bias, sh, sw, ph, pw);
    return out;
}
std::vector<Tensor> conv2d_backward(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& grad, const dispatch::SpatialArgs& args) {
    validate_conv2d_forward(input, weight, bias, args.sh, args.sw, args.ph, args.pw);
    validate_conv2d_gradient(input, weight, grad, args.sh, args.sw, args.ph, args.pw);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ConvolutionBackwardKernel>(unary_request(OperatorId::Conv2dBackward, input), default_runtime());
    return kernel.call(input, weight, bias, grad, args);
}
} // namespace orthon
