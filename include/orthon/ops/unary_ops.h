#pragma once
#include <orthon/core/tensor.h>
#include <cstdint>

namespace orthon {

// Element-wise unary ops. Each returns a new contiguous tensor with the same
// shape and dispatches to the CUDA kernels for CUDA tensors. Supports
// contiguous and non-contiguous inputs.
Tensor neg(const Tensor& t);
Tensor relu(const Tensor& t);
Tensor sigmoid(const Tensor& t);
Tensor tanh_t(const Tensor& t);
Tensor leaky_relu(const Tensor& t, float negative_slope = 0.01f);
Tensor gelu(const Tensor& t);
Tensor exp_t(const Tensor& t);
Tensor dropout(const Tensor& t, float p, uint64_t seed);
Tensor sqrt_t(const Tensor& t);
Tensor reciprocal_t(const Tensor& t);

Tensor sigmoid_backward(const Tensor& input, const Tensor& grad_output);
Tensor tanh_backward(const Tensor& input, const Tensor& grad_output);
Tensor leaky_relu_backward(const Tensor& input, const Tensor& grad_output,
                           float negative_slope = 0.01f);
Tensor gelu_backward(const Tensor& input, const Tensor& grad_output);
Tensor exp_backward(const Tensor& input, const Tensor& grad_output);
Tensor sqrt_backward(const Tensor& input, const Tensor& grad_output);
Tensor reciprocal_backward(const Tensor& input, const Tensor& grad_output);
Tensor neg_backward(const Tensor& input, const Tensor& grad_output);

}  // namespace orthon
