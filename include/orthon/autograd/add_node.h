#pragma once
#include <orthon/autograd/node.h>
#include <orthon/autograd/grad_utils.h>

namespace orthon {

class AddNode : public Node {
public:
    AddNode(const Tensor& a, const Tensor& b) {
        inputs = {a, b};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        return {
            unbroadcast(grad_output, inputs[0].shape()),
            unbroadcast(grad_output, inputs[1].shape())
        };
    }
};

}  // namespace orthon
