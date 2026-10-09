#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/index_ops.h>
#include <cstdint>
#include <vector>

namespace orthon {

// Backward of the first-dimension integer index `input[index]`.
//
// Forward is a storage-sharing view that drops dimension 0, so the gradient of
// every element outside block `index` is exactly zero: backward scatters
// grad_output into a zero tensor shaped like the parent.
//
// The index stored here is already normalised by Tensor::operator[], so a
// negative subscript needs no second resolution at backward time.
class IndexNode : public Node {
public:
    int64_t index_;

    IndexNode(const Tensor& input, int64_t normalized_index)
        : index_(normalized_index) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        return {index_backward(grad_output, input.shape(), input.dtype(),
                               input.device(), index_)};
    }
};

}  // namespace orthon
