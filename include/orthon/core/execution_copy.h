#pragma once
#include <orthon/core/tensor.h>
namespace orthon {
// Execution-only materialization; contiguous inputs pass through.
Tensor execution_operand(const Tensor&);
Tensor execution_reshape(const Tensor&, const std::vector<int64_t>&);
} // namespace orthon
