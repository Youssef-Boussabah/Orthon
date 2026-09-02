#pragma once
#include <orthon/autograd/node.h>
#include <orthon/core/dtype.h>
#include <orthon/ops/cpu/cast_ops.h>
#include <orthon/ops/cpu/shape_ops.h>

namespace orthon {

// Backward of a floating->floating dtype conversion.
//
// Forward converts values elementwise into a fresh contiguous tensor of the
// destination dtype on the same device. The derivative treats the conversion
// as identity (the rounding step is not differentiated), so backward casts
// the gradient back to the ORIGINAL source dtype:
//
//     grad_input = cast(grad_output, source_dtype)
//
// Attached ONLY for real floating->floating conversions: a same-dtype cast
// keeps its identity/ContiguousNode behaviour, and any conversion involving
// an integer/bool dtype is intentionally non-differentiable - there is no
// straight-through estimator.
//
// Tensor::backward holds NoGradGuard, so the recursive cast()/contiguous()
// calls below attach nothing. cuda_cast demands a contiguous input, and an
// upstream view node may hand back a strided gradient, hence the one
// contiguous() normalisation (a no-op when already contiguous).
//
// Value-insensitive by construction: the derivative never reads the
// input's stored values, so no watch_version.
class CastNode : public Node {
public:
    DType src_dtype_;

    explicit CastNode(const Tensor& input) : src_dtype_(input.dtype()) {
        inputs = {input};
    }

    const char* op_name() const override { return "CastNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        return {cast(contiguous(grad_output), src_dtype_)};
    }
};

}  // namespace orthon
