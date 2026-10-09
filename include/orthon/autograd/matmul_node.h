#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/matmul_ops.h>
#include <orthon/ops/shape_ops.h>

namespace orthon {

class MatMulNode : public Node {
public:
    MatMulNode(const Tensor& a, const Tensor& b) {
        inputs = {a, b};
        // dA = grad @ B.T and dB = A.T @ grad read both forward payloads.
        watch_version(a, "lhs");
        watch_version(b, "rhs");
    }

    const char* op_name() const override { return "MatMulNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // dA = grad_output @ B.T,  dB = A.T @ grad_output
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        Tensor b_t = b.ndim() == 2 ? transpose(b, 0, 1) : transpose(b, -2, -1);
        Tensor a_t = a.ndim() == 2 ? transpose(a, 0, 1) : transpose(a, -2, -1);

        Tensor grad_a = matmul(grad_output, b_t);
        Tensor grad_b = matmul(a_t, grad_output);
        return {grad_a, grad_b};
    }
};

}  // namespace orthon
