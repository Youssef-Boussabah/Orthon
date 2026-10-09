#pragma once
#include <orthon/autograd/node.h>

namespace orthon {

// Transfer backward returns the gradient to the exact source device in
// logical order. The public to() call runs under NoGradGuard and materializes
// strided gradients. Its derivative reads no saved values, so source
// mutation does not require a version watch.
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
