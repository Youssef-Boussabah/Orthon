#pragma once

#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/node.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/softmax_ops.h>
#include <orthon/ops/unary_ops.h>
#include <cstdint>
#include <vector>

namespace orthon {

// Backward of softmax along `dim_`:
//
//     grad_input = y * (g - sum(g * y, dim, keepdim=True))
//
// where y is the forward output. Built entirely from existing primitives, so
// no CUDA softmax-backward kernel is needed: sum(dim, keepdim), broadcasting
// add, neg and mul all already work on both devices for every dimension.
class SoftmaxNode : public Node {
public:
    int64_t dim_;

    SoftmaxNode(const Tensor& input, int64_t dim) : dim_(dim) {
        inputs = {input};
        // Backward recomputes softmax(input) from forward-time values.
        watch_version(input, "input");
    }

    const char* op_name() const override { return "SoftmaxNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);

        // Every operation below is a graph-attaching public op. Tensor::backward
        // already holds a structural NoGradGuard for the whole traversal; this
        // local guard keeps the invariant even if this backward is ever invoked
        // directly, where an unguarded softmax() call would attach a second
        // SoftmaxNode and build a graph on top of the returned gradient - i.e.
        // accidental higher-order autograd. The engine is first-order only.
        NoGradGuard guard;

        const Tensor probs    = softmax(inputs[0], dim_);
        const Tensor weighted = mul(grad, probs);
        const Tensor dot      = sum(weighted, dim_, /*keepdim=*/true);
        const Tensor centered = add(grad, neg(dot));
        return {mul(probs, centered)};
    }
};

}  // namespace orthon
