#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/scalar_ops.h>
#include <vector>

namespace orthon {

// Autograd for Tensor-with-Python-scalar arithmetic.
//
// WHY THESE EXIST AT ALL
// ----------------------
// A scalar operand is a CONSTANT. It never receives a gradient, and - crucially
// - the derivative with respect to the TENSOR operand never reads the tensor's
// forward-time values:
//
//     d(x + s)/dx = 1        d(x * s)/dx = s        d(x - s)/dx = 1
//
// Routing these through the generic Tensor-Tensor nodes would get two things
// wrong:
//
//  1. SELECTIVITY - the principle that a node watches only the values its
//     backward actually reads.
//     MulNode watches BOTH operands, because for a genuine Tensor-Tensor
//     product each side's gradient needs the other side's payload. With a
//     hidden constant on the right that watch on the LEFT buys nothing, yet it
//     rejects a perfectly legal program:
//
//         y = x * 3 ; x.set_item([0], 99.0) ; sum(y).backward()
//
//     which fails with "tensor saved by MulNode as 'lhs' was modified
//     in-place after it was saved for backward". dy/dx is 3 - it does not
//     depend on x at all - so turning scalar convenience into a blanket
//     mutation rejection would be a strictly worse public contract.
//
//  2. WASTED WORK. The generic nodes compute a gradient for the constant
//     operand too: unbroadcast(grad, {}) reduces the WHOLE gradient down to
//     rank 0 - an O(numel) pass (an atomicAdd tree on CUDA) whose result the
//     engine immediately discards, since a constant does not require grad.
//
// Both nodes below therefore hold ONE input, watch NOTHING, and return exactly
// one gradient.
//
// WHAT IS DELIBERATELY *NOT* HERE
// -------------------------------
// There is no ScalarSub, ScalarRSub, ScalarDiv or ScalarRDiv node, because
// composition through existing, already-correct machinery gives precisely the
// right answer AND the right watch set:
//
//     x - s   ->  ScalarAdd(x, -s)                    no watch  (correct)
//     s - x   ->  ScalarAdd(neg(x), s)                no watch  (correct:
//                                                     d/dx = -1, and Neg is
//                                                     already documented as
//                                                     value-insensitive)
//     x / s   ->  ScalarMul(x, 1/s)                   no watch  (correct)
//     s / x   ->  ScalarMul(reciprocal(x), s)         WATCHES x via the
//                                                     Reciprocal ActivationNode
//                                                     - correct, because
//                                                     d(s/x)/dx = -s/x^2 really
//                                                     does need x's values.
//
// That asymmetry between `x / s` and `s / x` is exactly what selective
// watching is for, and it falls out of composition rather than being
// hand-coded.
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
