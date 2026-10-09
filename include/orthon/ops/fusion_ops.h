#pragma once
#include <orthon/core/tensor.h>

namespace orthon {

// Fused op:
//   relu(x @ weight + bias)
// Only the forward is fused; the backward decomposes into the component
// gradients. Supports 2-D x/weight and 1-D bias.
Tensor fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias);

}  // namespace orthon
