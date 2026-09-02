#pragma once
#include <orthon/core/tensor.h>
#include <utility>

namespace orthon {

// CUDA element-wise ops. Preconditions (caller must verify): tensor is a
// contiguous float32, float16, or bfloat16 tensor on CUDA.
Tensor cuda_neg(const Tensor& t);
Tensor cuda_relu(const Tensor& t);
Tensor cuda_sigmoid(const Tensor& t);
Tensor cuda_tanh(const Tensor& t);
Tensor cuda_leaky_relu(const Tensor& t, float negative_slope);
Tensor cuda_gelu(const Tensor& t);
Tensor cuda_exp(const Tensor& t);
Tensor cuda_dropout_mask(const Tensor& t, float p, uint64_t seed);

// CUDA relu backward: grad_input[i] = input[i] > 0 ? grad_output[i] : 0.
// Preconditions: both tensors are contiguous float32/float16/bfloat16 on the
// same CUDA device and have the same number of elements.
Tensor cuda_relu_backward(const Tensor& input, const Tensor& grad_output);
Tensor cuda_sigmoid_backward(const Tensor& input, const Tensor& grad_output);
Tensor cuda_tanh_backward(const Tensor& input, const Tensor& grad_output);
Tensor cuda_leaky_relu_backward(const Tensor& input, const Tensor& grad_output,
                                float negative_slope);
Tensor cuda_gelu_backward(const Tensor& input, const Tensor& grad_output);
Tensor cuda_exp_backward(const Tensor& input, const Tensor& grad_output);

Tensor cuda_reciprocal(const Tensor& t);
Tensor cuda_sqrt(const Tensor& t);
Tensor cuda_sqrt_backward(const Tensor& input, const Tensor& grad_output);
Tensor cuda_reciprocal_backward(const Tensor& input, const Tensor& grad_output);

// Return true if every floating-point element is finite. Non-floating dtypes
// are treated as finite by the Python binding before this helper is called.
bool cuda_isfinite_all(const Tensor& t);

// Return a contiguous float32 tensor containing src * inv_scale and a finite
// flag computed in the same CUDA pass. Supports CUDA float32/float16/bfloat16.
std::pair<Tensor, bool> cuda_unscale_to_float32_and_check(const Tensor& src,
                                                          float inv_scale);

// Return a contiguous copy of t on its current CUDA device.
// Works correctly for both contiguous and strided (e.g. transposed) tensors.
Tensor cuda_make_contiguous(const Tensor& t);

}  // namespace orthon
