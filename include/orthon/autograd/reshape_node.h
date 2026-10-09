#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/shape_ops.h>
#include <vector>
#include <cstdint>

namespace orthon {

// Backward of reshape: reshape grad_output back to the input's original shape.
// Works for both CPU and CUDA tensors because reshape is a pure metadata op
// that shares the underlying storage.
class ReshapeNode : public Node {
public:
    std::vector<int64_t> input_shape_;

    explicit ReshapeNode(const Tensor& input) : input_shape_(input.shape()) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        Tensor grad = grad_output.is_contiguous()
            ? grad_output
            : grad_output.to(grad_output.device());
        return {reshape(grad, input_shape_)};
    }
};

}  // namespace orthon
