#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/backward_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>



namespace orthon {

class Conv2DNode : public Node {
public:
    int64_t sh_, sw_, ph_, pw_;

    Conv2DNode(const Tensor& input, const Tensor& weight, const Tensor& bias,
               int64_t sh, int64_t sw, int64_t ph, int64_t pw)
        : sh_(sh), sw_(sw), ph_(ph), pw_(pw) {
        inputs = {input, weight, bias};
        // grad_input reads weight values, grad_weight reads input values.
        // grad_bias is a reduction of grad_output alone on both CPU (sums g)
        // and CUDA (cuda_conv2d_backward_bias touches only bias.shape/dtype/
        // device), so the bias payload is deliberately not watched.
        watch_version(input, "input");
        watch_version(weight, "weight");
    }

    const char* op_name() const override { return "Conv2DNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        const Tensor& inp = inputs[0];
        const Tensor& wgt = inputs[1];
        const Tensor& bia = inputs[2];

        return conv2d_backward(inp, wgt, bia, grad, {sh_, sw_, ph_, pw_});
    }
};

}  // namespace orthon
