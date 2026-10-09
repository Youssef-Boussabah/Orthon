#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/softmax_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/log_softmax_node.h>
#include <orthon/autograd/softmax_node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>


namespace orthon {

static int64_t resolve_dim(int64_t dim, int64_t ndim, const char* name) {
    const int64_t original = dim;
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::out_of_range(
            std::string(name) + ": dim " + std::to_string(original) +
            " out of range for tensor with " + std::to_string(ndim) +
            " dimensions");
    return dim;
}

static void validate_softmax_input(const Tensor& t, const char* name) {
    if (t.ndim() == 0)
        throw std::runtime_error(std::string(name) + ": scalar tensors are not supported");
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(name) + ": only float32 supported");
    if (t.numel() == 0)
        throw std::runtime_error(std::string(name) + ": empty tensor");
    if (!t.is_contiguous())
        throw std::runtime_error(std::string(name) + ": requires a contiguous tensor");
}


// Numerical forward only: never attaches a graph. `resolved` is already
// normalised and `t` already validated.
static Tensor softmax_forward_values(const Tensor& t, int64_t resolved) {
    if (t.ndim() == 2 && resolved == 1) {
        using namespace dispatch;
        const auto& kernel = default_dispatcher().resolve<DimensionKernel>(unary_request(OperatorId::SoftmaxRows, t), default_runtime());
        return kernel.call(t, resolved);
    }
    NoGradGuard guard;
    return exp_t(log_softmax(t, resolved));
}

Tensor softmax(const Tensor& t, int64_t dim) {
    validate_softmax_input(t, "softmax");
    const int64_t resolved = resolve_dim(dim, t.ndim(), "softmax");

    Tensor out = softmax_forward_values(t, resolved);

    if (is_grad_enabled() && t.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<SoftmaxNode>(t, resolved));
    }
    return out;
}

static Tensor attach_log_softmax_grad(const Tensor& input, Tensor out, int64_t dim) {
    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<LogSoftmaxNode>(input, dim));
    }
    return out;
}

Tensor log_softmax(const Tensor& t, int64_t dim) {
    validate_softmax_input(t, "log_softmax");
    const int64_t resolved = resolve_dim(dim, t.ndim(), "log_softmax");

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<DimensionKernel>(unary_request(OperatorId::LogSoftmax, t), default_runtime());
    return attach_log_softmax_grad(t, kernel.call(t, resolved), resolved);
}

Tensor log_softmax_backward(const Tensor& input,
                            const Tensor& grad_output,
                            int64_t dim) {
    validate_softmax_input(input, "log_softmax backward");
    if (grad_output.shape() != input.shape())
        throw std::runtime_error("log_softmax backward: grad_output shape mismatch");
    if (grad_output.dtype() != DType::Float32)
        throw std::runtime_error("log_softmax backward: grad_output must be float32");
    if (grad_output.device() != input.device())
        throw std::runtime_error("log_softmax backward: device mismatch");
    if (!grad_output.is_contiguous())
        throw std::runtime_error("log_softmax backward: grad_output must be contiguous");
    const int64_t resolved = resolve_dim(dim, input.ndim(), "log_softmax backward");

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<DimensionBackwardKernel>(unary_request(OperatorId::LogSoftmaxBackward, input), default_runtime());
    return kernel.call(input, grad_output, resolved);
}

}  // namespace orthon
