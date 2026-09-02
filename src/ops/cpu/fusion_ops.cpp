#include <orthon/ops/cpu/fusion_ops.h>
#include <orthon/autograd/fused_linear_relu_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/dtype.h>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/fusion_kernels.h>
#endif

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
        out.set_grad_fn(std::make_shared<FusedLinearReLUNode>(x, weight, bias));
    }
}

Tensor fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    validate_fused_linear_relu_common(x, weight, bias);

#ifdef WITH_CUDA
    if (x.device().is_cuda()) {
        // The CPU loop below reads x, weight and bias through their strides,
        // so it consumes views; the tiled CUDA kernel indexes row-major and
        // cannot. Normalising the kernel operands closes that mismatch without
        // touching the fused math or the kernel's contiguous-only contract.
        // FusedLinearReLUNode is attached to the ORIGINAL x/weight/bias, so
        // all three saved-version watches - the node is value-sensitive in
        // every input, bias included - observe the user tensors' storage.
        Tensor out = cuda_fused_linear_relu(
            kernel_operand(x), kernel_operand(weight), kernel_operand(bias));
        attach_fused_linear_relu_grad_fn(out, x, weight, bias);
        return out;
    }
#endif
    if (!x.device().is_cpu())
        throw std::runtime_error("fused_linear_relu: unsupported device");

    const int64_t m = x.shape()[0];
    const int64_t k = x.shape()[1];
    const int64_t n = weight.shape()[1];

    const int64_t sx0 = x.strides()[0], sx1 = x.strides()[1];
    const int64_t sw0 = weight.strides()[0], sw1 = weight.strides()[1];
    const int64_t sb0 = bias.strides()[0];

    const float* px = static_cast<const float*>(x.data());
    const float* pw = static_cast<const float*>(weight.data());
    const float* pb = static_cast<const float*>(bias.data());

    Tensor out = empty({m, n}, x.dtype(), x.device());
    float* po = static_cast<float*>(out.data());

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = pb[j * sb0];
            for (int64_t p = 0; p < k; ++p)
                acc += px[i * sx0 + p * sx1] * pw[p * sw0 + j * sw1];
            po[i * n + j] = acc > 0.0f ? acc : 0.0f;
        }
    }

    attach_fused_linear_relu_grad_fn(out, x, weight, bias);
    return out;
}

}  // namespace orthon
