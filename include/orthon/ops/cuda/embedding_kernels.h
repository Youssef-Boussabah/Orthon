#pragma once

#include <optional>
#include <vector>

#include <orthon/core/tensor.h>

namespace orthon {

Tensor cuda_embedding_forward(const Tensor& indices,
                              const Tensor& weight,
                              std::optional<int64_t> padding_idx);

Tensor cuda_embedding_backward(const Tensor& indices,
                               const Tensor& grad_output,
                               const std::vector<int64_t>& weight_shape,
                               std::optional<int64_t> padding_idx);

}  // namespace orthon
