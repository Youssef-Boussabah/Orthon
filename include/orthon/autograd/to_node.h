#pragma once
#include <orthon/autograd/node.h>

namespace orthon {

// Backward of the public Tensor::to(device).
//
// Forward is a physical contiguous copy whose values equal the input in
// LOGICAL order - same shape, same dtype, possibly a different Device. The
// derivative is therefore identity in logical tensor coordinates, and
// backward only has to return the gradient to the source device:
//
//     grad_input = grad_output.to(input.device())
//
// Tensor::backward runs the whole traversal under NoGradGuard, so this
// recursive public to() call attaches no graph of its own. It also
// materialises a strided upstream gradient into logical contiguous order,
// which is exactly the coordinate system the input's own upstream nodes
// (e.g. TransposeNode) expect.
//
// Value-insensitive by construction: the derivative never reads the
// input's stored values, so no watch_version - mutating the source after
// forward stays legal unless some downstream value-sensitive node saved it.
class ToNode : public Node {
public:
    explicit ToNode(const Tensor& input) {
        inputs = {input};
    }

    const char* op_name() const override { return "ToNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        return {grad_output.to(inputs[0].device())};
    }
};

}  // namespace orthon
