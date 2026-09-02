#pragma once
#include <orthon/core/tensor.h>
#include <vector>
#include <cstdint>

namespace orthon {

// Row-wise cross-entropy loss. Returns the mean loss as a scalar Tensor
// (shape {1}).  When logits.requires_grad() the result carries a grad_fn so
// that loss.backward() propagates dL/dlogits = (softmax(logits) - one_hot) / B.
//   logits : float32, shape [batch_size, num_classes], contiguous
//   targets: class index per row, each in [0, num_classes)
// Throws std::runtime_error for shape/dtype/device/bounds violations.
Tensor cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets);

}  // namespace orthon
