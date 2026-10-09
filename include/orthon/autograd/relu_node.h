#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/backward_ops.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/core/dtype.h>
#include <vector>
#include <cstdint>



namespace orthon {

class ReluNode : public Node {
public:
    explicit ReluNode(const Tensor& input) {
        inputs = {input};
        // Backward recomputes the input>0 mask from forward-time values.
        watch_version(input, "input");
    }

    const char* op_name() const override { return "ReluNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        // Same strided-gradient hazard as ActivationNode: cuda_relu_backward
        // indexes the gradient linearly, so a strided upstream gradient yields
        // misordered results. No-op when already contiguous, and it attaches no
        // node because a gradient does not require grad.
        const Tensor grad = contiguous(grad_output);

        return {relu_backward(input, grad)};
    }
};

}  // namespace orthon
