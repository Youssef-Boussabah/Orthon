#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/unary_ops.h>
#include <stdexcept>
#include <vector>

namespace orthon {

enum class ActivationKind {
    Sigmoid,
    Tanh,
    LeakyRelu,
    Gelu,
    Exp,
    Sqrt,
    Reciprocal,
    Neg,
};

class ActivationNode : public Node {
public:
    ActivationKind kind_;
    float alpha_;

    ActivationNode(const Tensor& input, ActivationKind kind, float alpha = 0.0f)
        : kind_(kind), alpha_(alpha) {
        inputs = {input};
        // Every kind except Neg evaluates f'(input) from forward-time values.
        // Neg's backward is -grad_output: the input payload is never read
        // (the CPU dispatch lambda discards it; CUDA calls cuda_neg(grad)),
        // so mutating it after forward stays legal.
        if (kind != ActivationKind::Neg)
            watch_version(input, "input");
    }

    const char* op_name() const override { return "ActivationNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        // Elementwise backward kernels consume contiguous gradients. Materialize an
        // upstream strided view once for every activation kind; gradient tensors do
        // not require grad, so this introduces no higher-order graph.
        const Tensor grad = contiguous(grad_output);
        switch (kind_) {
            case ActivationKind::Sigmoid:
                return {sigmoid_backward(input, grad)};
            case ActivationKind::Tanh:
                return {tanh_backward(input, grad)};
            case ActivationKind::LeakyRelu:
                return {leaky_relu_backward(input, grad, alpha_)};
            case ActivationKind::Gelu:
                return {gelu_backward(input, grad)};
            case ActivationKind::Exp:
                return {exp_backward(input, grad)};
            case ActivationKind::Sqrt:
                return {sqrt_backward(input, grad)};
            case ActivationKind::Reciprocal:
                return {reciprocal_backward(input, grad)};
            case ActivationKind::Neg:
                return {neg_backward(input, grad)};
        }
        throw std::runtime_error("activation backward: unsupported activation kind");
    }
};

}  // namespace orthon
