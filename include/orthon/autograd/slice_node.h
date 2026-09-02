#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/slice_ops.h>
#include <cstdint>
#include <vector>

namespace orthon {

// Backward of the range view `input.slice(dim, start, end)`.
//
// Forward is a storage-sharing view that narrows one dimension, so the gradient
// of every element outside [start, end) along `dim` is exactly zero: backward
// scatters grad_output into a zero tensor shaped like the parent.
//
// dim/start/end are stored as Tensor::slice resolved them, so a negative
// argument is normalised exactly once, in the forward. Backward never
// re-interprets the caller's spelling.
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
