#pragma once
#include <orthon/core/tensor.h>
#include <optional>

namespace orthon {

// Tensor-valued reductions. dim=None reduces all elements to a scalar tensor
// with shape []; dim reductions return tensors on the same device.
Tensor sum(const Tensor& t,
           std::optional<int64_t> dim = std::nullopt,
           bool keepdim = false);
Tensor mean(const Tensor& t,
            std::optional<int64_t> dim = std::nullopt,
            bool keepdim = false);

// reduce_max: returns values tensor in the input dtype; attaches MaxNode when requires_grad.
// Named reduce_max to avoid conflict with std::max.
Tensor reduce_max(const Tensor& t,
                  std::optional<int64_t> dim = std::nullopt,
                  bool keepdim = false);

// argmax: returns Int64 indices tensor; no autograd.
Tensor argmax(const Tensor& t,
              std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false);

}  // namespace orthon
