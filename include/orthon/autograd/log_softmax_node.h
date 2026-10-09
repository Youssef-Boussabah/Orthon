#pragma once

#include <orthon/autograd/node.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/softmax_ops.h>
#include <vector>
#include <cstdint>

namespace orthon {

class LogSoftmaxNode : public Node {
public:
    int64_t dim_;

    LogSoftmaxNode(const Tensor& input, int64_t dim)
        : dim_(dim) {
        inputs = {input};
        // log_softmax_backward recomputes softmax from forward-time values.
        watch_version(input, "input");
    }

    const char* op_name() const override { return "LogSoftmaxNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        return {log_softmax_backward(inputs[0], grad, dim_)};
    }
};

}  // namespace orthon
