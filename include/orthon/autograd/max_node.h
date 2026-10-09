#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/backward_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/ops/fill_ops.h>
#include <cstdint>
#include <optional>
#include <vector>



namespace orthon {

class MaxNode : public Node {
public:
    const ReductionPlan plan_;
    Tensor argmax_indices_;

    MaxNode(const Tensor& input, const Tensor& argmax_indices, const ReductionPlan& plan)
        : plan_(plan), argmax_indices_(argmax_indices) {
        inputs = {input};
        // Backward scatters grad_output by the SAVED argmax buffer's VALUES;
        // the input payload is never re-read (only the plan, dtype and
        // device), so mutating the input after max() stays legal while the
        // saved index buffer itself is watched.
        watch_version(argmax_indices_, "argmax_indices");
    }

    const char* op_name() const override { return "MaxNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        const Tensor& input = inputs[0];

        return {max_backward(input, grad, argmax_indices_, plan_)};
    }
};

}  // namespace orthon
