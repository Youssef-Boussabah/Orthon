#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/backward_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>



namespace orthon {

class MaxPool2DNode : public Node {
public:
    int64_t kh_, kw_, sh_, sw_, ph_, pw_;

    MaxPool2DNode(const Tensor& input,
                  int64_t kernel_h, int64_t kernel_w,
                  int64_t stride_h, int64_t stride_w,
                  int64_t pad_h, int64_t pad_w)
        : kh_(kernel_h), kw_(kernel_w),
          sh_(stride_h), sw_(stride_w),
          ph_(pad_h), pw_(pad_w) {
        inputs = {input};
        // Backward re-runs the window max search over forward-time input
        // values (no indices were saved), so the payload must be intact.
        watch_version(input, "input");
    }

    const char* op_name() const override { return "MaxPool2DNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        const Tensor& input = inputs[0];

        return {maxpool2d_backward(input, grad, {sh_, sw_, ph_, pw_, kh_, kw_})};
    }
};

}  // namespace orthon
