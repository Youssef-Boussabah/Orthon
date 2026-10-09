#pragma once
#include <orthon/core/tensor.h>

namespace orthon {

// Prototype forward-only fused CUDA op:
//   relu(x @ weight + bias)
// Preconditions are validated by the wrapper.
Tensor cuda_fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias);

}  // namespace orthon
