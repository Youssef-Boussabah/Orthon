#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <optional>
#include <vector>

namespace orthon {

class ReduceNode : public Node {
public:
    std::optional<int64_t> dim_;
    bool keepdim_;
    bool is_mean_;
    int64_t factor_;
    std::vector<int64_t> input_shape_;

    ReduceNode(const Tensor& input,
               std::optional<int64_t> dim,
               bool keepdim,
               bool is_mean,
               int64_t factor)
        : dim_(dim),
          keepdim_(keepdim),
          is_mean_(is_mean),
          factor_(factor),
          input_shape_(input.shape()) {
        inputs = {input};
    }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        // reshape() below accepts only contiguous tensors, and an upstream view
        // node may legitimately hand back a strided gradient. No-op when the
        // gradient already is contiguous.
        Tensor grad = contiguous(grad_output);

        if (dim_.has_value() && !keepdim_) {
            std::vector<int64_t> grad_shape = grad.shape();
            grad_shape.insert(grad_shape.begin() + *dim_, 1);
            grad = reshape(grad, grad_shape);
        }

        Tensor ones_t = ones(input_shape_, input.dtype(), Device{});
        if (input.device().is_cuda()) {
            ones_t = ones_t.to(input.device());
        }

        Tensor expanded = mul(ones_t, grad);
        if (is_mean_) {
            // mul_ does not support Float16; use broadcast mul with a scalar tensor.
            //
            // The reciprocal is formed in DOUBLE. Forming it as
            // `1.0f / static_cast<float>(factor_)` would decide a float64
            // mean's backward scale by float32 arithmetic - 1/6 coming back as
            // 0.1666666716337204 instead of 0.16666666666666666. Because
            // full() also takes a double, the value survives intact to a
            // float64 destination and is narrowed exactly once, by full(), for
            // every other dtype. sum backward has no scale at all.
            Tensor scale = full({}, 1.0 / static_cast<double>(factor_),
                                input.dtype(), Device{});
            if (input.device().is_cuda()) scale = scale.to(input.device());
            expanded = mul(expanded, scale);
        }
        return {expanded};
    }
};

}  // namespace orthon
