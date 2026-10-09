#pragma once

#include <orthon/autograd/node.h>
#include <orthon/ops/shape_ops.h>
#include <cstdint>
#include <vector>

namespace orthon {

class TransposeNode : public Node {
public:
    int64_t dim0_;
    int64_t dim1_;

    TransposeNode(const Tensor& input, int64_t dim0, int64_t dim1)
        : dim0_(dim0), dim1_(dim1) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        return {transpose(grad_output, dim0_, dim1_)};
    }
};

}  // namespace orthon
