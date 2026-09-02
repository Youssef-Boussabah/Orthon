#pragma once
#include <vector>
#include <cstdint>
#include <orthon/core/tensor.h>

namespace orthon {

// Reduce `grad` back to `target_shape` by summing over any dimension that
// was broadcast during the forward pass.  If the shapes already match, grad
// is returned unchanged (zero-copy fast path).
Tensor unbroadcast(const Tensor& grad, const std::vector<int64_t>& target_shape);

}  // namespace orthon
