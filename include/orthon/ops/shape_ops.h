#pragma once
#include <orthon/core/broadcast.h>
#include <cstdint>
#include <vector>

#include <orthon/core/tensor.h>

namespace orthon {

Tensor reshape    (const Tensor& input, const std::vector<int64_t>& new_shape);
Tensor transpose  (const Tensor& input, int64_t dim0, int64_t dim1);
Tensor contiguous (const Tensor& input);

// kernel_operand forwards to execution_operand: contiguous kernels consume
// a temporary made under NoGradGuard on the exact device. Public Nodes
// retain the original Tensor so view gradients and mutation watches survive.
Tensor kernel_operand(const Tensor& t);

}  // namespace orthon
