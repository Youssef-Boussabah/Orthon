#pragma once
#include <orthon/autograd/node.h>
#include <orthon/core/dtype.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/ops/shape_ops.h>

namespace orthon {

// Floating casts differentiate as identity followed by a cast to the source
// dtype. Integer/bool conversions are non-differentiable; same-dtype casts
// use contiguous(). Backward runs under NoGradGuard and materializes strided
// gradients before CUDA casting. The derivative reads no saved values, so
// this node needs no version watch.
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
