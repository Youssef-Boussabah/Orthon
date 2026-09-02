#pragma once
#include <orthon/autograd/node.h>

namespace orthon {

// Backward of contiguous: the op is a memory-layout change only.
// Forward: input (possibly non-contiguous) → output (contiguous, same shape/values).
// Backward: grad_output (same shape as output) propagates unchanged back to input.
// The upstream node (e.g. TransposeNode) is responsible for un-transposing the grad.
class ContiguousNode : public Node {
public:
    explicit ContiguousNode(const Tensor& input) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        return {grad_output};
    }
};

}  // namespace orthon
