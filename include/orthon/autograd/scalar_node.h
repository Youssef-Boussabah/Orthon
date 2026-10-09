#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/scalar_ops.h>
#include <vector>

namespace orthon {

// Scalar operations differentiate only the Tensor operand. Dedicated nodes
// avoid watching values the derivative does not read and computing a
// discarded gradient for the constant. Subtraction and division compose
// these nodes with neg/reciprocal: x / s needs no watch, while s / x watches
// x through Reciprocal because its derivative depends on x.
class ScalarAddNode : public Node {
public:
    explicit ScalarAddNode(const Tensor& input) {
        inputs = {input};
        // No watch_version: d(x + s)/dx = 1 reads no payload.
    }

    const char* op_name() const override { return "ScalarAddNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // A rank-0 operand never changes the output shape, so there is nothing
        // to unbroadcast: the gradient already lives in the input's own
        // coordinate system (including when that input is a strided view).
        return {grad_output};
    }
};

class ScalarMulNode : public Node {
public:
    ScalarMulNode(const Tensor& input, double scale) : scale_(scale) {
        inputs = {input};
        // No watch_version: d(x * s)/dx = s, a value copied into this node at
        // construction. The input's payload is never read again.
    }

    const char* op_name() const override { return "ScalarMulNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Untracked by construction rather than by relying on the ambient
        // NoGradGuard that Tensor::backward happens to hold.
        return {scalar_mul_untracked(grad_output, scale_)};
    }

private:
    // A copy of the Python scalar, held at full double precision and narrowed
    // to the gradient's dtype only when the backward tensor is built.
    double scale_;
};

}  // namespace orthon
