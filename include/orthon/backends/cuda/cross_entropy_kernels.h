#pragma once
#include <orthon/core/tensor.h>
#include <vector>
#include <cstdint>

namespace orthon {

// Row-wise cross-entropy loss for a 2D contiguous float32 logits tensor.
// targets[i] is the ground-truth class index for row i (already validated by
// the caller to be in [0, num_classes)).
// Returns the mean loss across rows as a CUDA scalar tensor.
Tensor cuda_cross_entropy_loss(const Tensor& logits,
                               const std::vector<int64_t>& targets);

// Host-synchronized compatibility helper for benchmarks/tests that need a
// float directly.
float cuda_cross_entropy(const Tensor& logits,
                         const std::vector<int64_t>& targets);

// Fused cross-entropy backward: grad_logits[i,j] = (softmax(logits)[i,j] - one_hot[i,j]) * g / B
// logits must be contiguous float32 on CUDA.
// grad_output is the upstream scalar gradient tensor.
Tensor cuda_cross_entropy_backward(const Tensor& logits,
                                   const std::vector<int64_t>& targets,
                                   const Tensor& grad_output);

}  // namespace orthon
