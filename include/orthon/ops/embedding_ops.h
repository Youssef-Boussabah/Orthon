#pragma once

#include <optional>

#include <orthon/core/tensor.h>

namespace orthon {

Tensor embedding(const Tensor& indices,
                 const Tensor& weight,
                 std::optional<int64_t> padding_idx = std::nullopt);

Tensor embedding_backward(const Tensor& indices,
                          const Tensor& grad_output,
                          const std::vector<int64_t>& weight_shape,
                          std::optional<int64_t> padding_idx = std::nullopt);

}  // namespace orthon
