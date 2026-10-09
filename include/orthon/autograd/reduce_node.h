#pragma once
#include <orthon/autograd/node.h>
#include <orthon/reduction/reduction_plan.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <optional>
#include <vector>

namespace orthon {

class ReduceNode : public Node {
public:
    const ReductionPlan plan_;

    ReduceNode(const Tensor& input, const ReductionPlan& plan) : plan_(plan) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        // reshape() below accepts only contiguous tensors, and an upstream view
        // node may legitimately hand back a strided gradient. No-op when the
        // gradient already is contiguous.
        Tensor grad = contiguous(grad_output);

        if (plan_.dim && !plan_.keepdim)
            grad = reshape(grad, plan_.broadcast_shape);

        Tensor ones_t = ones(plan_.input_shape, input.dtype(), Device{});
        if (input.device().is_cuda()) {
            ones_t = ones_t.to(input.device());
        }

        Tensor expanded = mul(ones_t, grad);
        if (plan_.kind == ReductionKind::Mean) {
            // mul_ excludes Float16, so use broadcast multiplication by a scalar.
            // Compute the reciprocal in double to preserve Float64 precision; full()
            // narrows once for other dtypes.
            Tensor scale = full({}, 1.0 / static_cast<double>(plan_.reduce),
                                input.dtype(), Device{});
            if (input.device().is_cuda()) scale = scale.to(input.device());
            expanded = mul(expanded, scale);
        }
        return {expanded};
    }
};

}  // namespace orthon
