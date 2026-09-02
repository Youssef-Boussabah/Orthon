#pragma once

#include <orthon/autograd/grad_utils.h>
#include <orthon/autograd/node.h>
#include <orthon/autograd/relu_node.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/matmul_ops.h>
#include <orthon/ops/cpu/shape_ops.h>

namespace orthon {

class FusedLinearReLUNode : public Node {
public:
    FusedLinearReLUNode(const Tensor& x, const Tensor& weight, const Tensor& bias) {
        inputs = {x, weight, bias};
        // Backward recomputes the preactivation add(matmul(x, weight), bias)
        // for the ReLU mask, so all three forward payloads - bias included -
        // are consumed by value.
        watch_version(x, "input");
        watch_version(weight, "weight");
        watch_version(bias, "bias");
    }

    const char* op_name() const override { return "FusedLinearReLUNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& x = inputs[0];
        const Tensor& weight = inputs[1];
        const Tensor& bias = inputs[2];

        Tensor preact = add(matmul(x, weight), bias);
        ReluNode relu_node(preact);
        Tensor grad_preact = relu_node.backward(grad_output)[0];

        Tensor weight_t = transpose(weight, 0, 1);
        Tensor x_t = transpose(x, 0, 1);

        if (x.device().is_cuda()) {
            if (!weight_t.is_contiguous()) weight_t = weight_t.to(weight_t.device());
            if (!x_t.is_contiguous()) x_t = x_t.to(x_t.device());
        }

        Tensor grad_x = matmul(grad_preact, weight_t);
        Tensor grad_weight = matmul(x_t, grad_preact);
        Tensor grad_bias = unbroadcast(grad_preact, bias.shape());
        return {grad_x, grad_weight, grad_bias};
    }
};

}  // namespace orthon
