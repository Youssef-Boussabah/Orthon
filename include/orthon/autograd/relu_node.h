#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/core/dtype.h>
#include <vector>
#include <cstdint>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/unary_kernels.h>
#endif

namespace orthon {

class ReluNode : public Node {
public:
    explicit ReluNode(const Tensor& input) {
        inputs = {input};
        // Backward recomputes the input>0 mask from forward-time values.
        watch_version(input, "input");
    }

    const char* op_name() const override { return "ReluNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& input = inputs[0];
        // Same strided-gradient hazard as ActivationNode: cuda_relu_backward
        // indexes the gradient linearly, so a strided upstream gradient yields
        // misordered results. No-op when already contiguous, and it attaches no
        // node because a gradient does not require grad.
        const Tensor grad = contiguous(grad_output);

#ifdef WITH_CUDA
        if (input.device().is_cuda()) {
            // relu() accepts a CUDA view, so the saved input can be
            // strided here. cuda_relu_backward indexes it linearly, so the
            // kernel operand is materialised the same way the forward does -
            // `input` itself stays the watched original.
            return {cuda_relu_backward(kernel_operand(input), grad)};
        }
#endif

        const int64_t n = input.numel();
        const DType dt  = input.dtype();
        Tensor mask     = empty(input.shape(), dt, input.device());

        auto fill_mask = [&](auto zero, auto one) {
            using T = decltype(zero);
            const T* src = static_cast<const T*>(input.data());
            T*       dst = static_cast<T*>(mask.data());
            if (input.is_contiguous()) {
                for (int64_t i = 0; i < n; ++i)
                    dst[i] = src[i] > zero ? one : zero;
            } else {
                const auto& sh  = input.shape();
                const auto& st  = input.strides();
                const int ndim  = static_cast<int>(sh.size());
                std::vector<int64_t> idx(ndim, 0);
                for (int64_t elem = 0; elem < n; ++elem) {
                    int64_t off = 0;
                    for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
                    dst[elem] = src[off] > zero ? one : zero;
                    for (int d = ndim - 1; d >= 0; --d) {
                        if (++idx[d] < sh[d]) break;
                        idx[d] = 0;
                    }
                }
            }
        };

        if (dt == DType::Float64)
            fill_mask(0.0, 1.0);
        else
            fill_mask(0.0f, 1.0f);

        return {mul(grad, mask)};
    }
};

}  // namespace orthon
