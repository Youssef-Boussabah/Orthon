#pragma once
#include <orthon/core/tensor.h>

namespace orthon {

// Row-wise numerically stable softmax for 2D contiguous float32 CUDA tensors.
// Preconditions (caller must verify): tensor is contiguous float32 on CUDA,
// ndim == 2, numel > 0.
Tensor cuda_softmax(const Tensor& t);

Tensor cuda_log_softmax(const Tensor& t, int64_t dim);

Tensor cuda_log_softmax_backward(const Tensor& input,
                                 const Tensor& grad_output,
                                 int64_t dim);

}  // namespace orthon
