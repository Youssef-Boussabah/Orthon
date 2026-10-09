#pragma once
#include <orthon/core/tensor.h>

namespace orthon {

// Numerically stable softmax along `dim` (default: the last dimension).
// Supports contiguous float32 tensors on CPU and CUDA, rank >= 1, matching the
// log_softmax dimension contract. Attaches SoftmaxNode when the input requires
// grad and grad mode is enabled.
//
// The 2D last-dimension case keeps its dedicated row-wise implementation (the
// fused CUDA kernel, the direct CPU loop); other shapes are computed from
// log_softmax. Both are internal detail - the graph contract is the same.
Tensor softmax(const Tensor& t, int64_t dim = -1);

// Numerically stable log_softmax along `dim`.
// Supports contiguous float32 tensors on CPU and CUDA.
Tensor log_softmax(const Tensor& t, int64_t dim = -1);

// Backward for log_softmax:
// grad_input = grad_output - softmax(input) * sum(grad_output, dim, keepdim=True)
Tensor log_softmax_backward(const Tensor& input,
                            const Tensor& grad_output,
                            int64_t dim);

}  // namespace orthon
