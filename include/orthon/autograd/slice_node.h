#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/slice_ops.h>
#include <cstdint>
#include <vector>

namespace orthon {

// Slice backward scatters into a zero tensor shaped like the parent.
// Store the dimension and bounds normalized by forward; backward must not
// reinterpret negative arguments.
class SliceNode : public Node {
public:
    int64_t dim_;
    int64_t start_;
    int64_t end_;

    SliceNode(const Tensor& input,
              int64_t normalized_dim,
              int64_t normalized_start,
              int64_t normalized_end)
        : dim_(normalized_dim)
        , start_(normalized_start)
        , end_(normalized_end) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        return {slice_backward(grad_output, input.shape(), input.dtype(),
                               input.device(), dim_, start_, end_)};
    }
};

}  // namespace orthon
