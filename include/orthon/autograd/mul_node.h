#pragma once
#include <orthon/autograd/node.h>
#include <orthon/autograd/grad_utils.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/shape_ops.h>

namespace orthon {

class MulNode : public Node {
public:
    MulNode(const Tensor& a, const Tensor& b) {
        inputs = {a, b};
        // Backward multiplies grad_output by the OTHER operand's forward-time
        // values, so both payloads must be unchanged at backward.
        watch_version(a, "lhs");
        watch_version(b, "rhs");
    }

    const char* op_name() const override { return "MulNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // cuda_mul requires contiguous operands; normalise once before both
        // multiplications. Broadcasting semantics are unchanged.
        const Tensor grad = contiguous(grad_output);
        return {
            unbroadcast(mul(grad, inputs[1]), inputs[0].shape()),
            unbroadcast(mul(grad, inputs[0]), inputs[1].shape())
        };
    }
};

}  // namespace orthon
