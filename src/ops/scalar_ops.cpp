#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/scalar_ops.h>

#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/scalar_node.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/unary_ops.h>

#include <memory>

namespace orthon {

Tensor scalar_like(const Tensor& reference, double value) {
    // Nothing built here may join a graph: this is a constant, and the guard
    // makes that structural rather than a consequence of the caller's mode.
    // It also covers the CPU-scalar -> exact-CUDA-device hop below, which
    // would otherwise be a differentiable Tensor::to().
    NoGradGuard guard;

    // full() allocates on the CPU (fill ops are a host-side loop), so a CUDA
    // reference needs one 1-element transfer. The DEVICE OBJECT is passed
    // through rather than a "cuda" string: the Device parser resolves the bare
    // string to cuda:0, so a reference living on cuda:1 would get its constant
    // built on the wrong GPU. cuda:1 stays cuda:1.
    Tensor scalar = full({}, value, reference.dtype(), Device{});
    if (scalar.device() != reference.device())
        scalar = scalar.to(reference.device());
    return scalar;
}

// Share the binary forward under NoGradGuard, then attach only the scalar
// Node. Tensor validation remains in the binary operator.
static Tensor scalar_forward(const Tensor& t, double value, bool is_mul) {
    NoGradGuard guard;
    const Tensor scalar = scalar_like(t, value);
    return is_mul ? mul(t, scalar) : add(t, scalar);
}

Tensor scalar_mul_untracked(const Tensor& t, double value) {
    return scalar_forward(t, value, /*is_mul=*/true);
}

Tensor scalar_add(const Tensor& t, double value) {
    Tensor out = scalar_forward(t, value, /*is_mul=*/false);
    if (is_grad_enabled() && t.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ScalarAddNode>(t));
    }
    return out;
}

Tensor scalar_mul(const Tensor& t, double value) {
    Tensor out = scalar_mul_untracked(t, value);
    if (is_grad_enabled() && t.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ScalarMulNode>(t, value));
    }
    return out;
}

// The four remaining forms are pure composition; see scalar_node.h for why
// that composition is the CORRECT choice and not merely the cheap one.

Tensor scalar_sub(const Tensor& t, double value) {
    return scalar_add(t, -value);
}

Tensor scalar_rsub(double value, const Tensor& t) {
    // neg() first, then the constant: d(s - x)/dx = -1 needs none of x's
    // values, and ActivationNode(Neg) is already documented as the one
    // activation that does not watch its input.
    return scalar_add(neg(t), value);
}

Tensor scalar_div(const Tensor& t, double value) {
    // Reciprocal-then-multiply, matching how Tensor/Tensor div composes. The
    // reciprocal is taken in DOUBLE and narrowed once by scalar_like, so for a
    // float32 tensor the constant is bit-identical to computing 1.0f/value.
    //
    // Division by zero is not intercepted: 1.0/0.0 is +inf, and x * inf then
    // yields +inf / -inf / nan exactly as IEEE-754 division by zero would. The
    // floating-point semantics are preserved rather than replaced by an
    // invented Python exception.
    return scalar_mul(t, 1.0 / value);
}

Tensor scalar_rdiv(double value, const Tensor& t) {
    // The one asymmetric case: d(s/x)/dx = -s/x^2 genuinely depends on x, and
    // the Reciprocal ActivationNode already watches its input for exactly that
    // reason. Mutating x after this forward correctly fails backward.
    return scalar_mul(reciprocal_t(t), value);
}

}  // namespace orthon
