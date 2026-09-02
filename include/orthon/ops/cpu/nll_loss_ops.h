#pragma once

#include <orthon/core/tensor.h>
#include <string>

namespace orthon {

// Negative log likelihood loss for 2D [N, C] log-probabilities and 1D int32
// or int64 class-index targets. Supported reductions: "mean", "sum", "none".
Tensor nll_loss(const Tensor& log_probs,
                const Tensor& targets,
                const std::string& reduction = "mean");

Tensor nll_loss_backward(const Tensor& log_probs,
                         const Tensor& targets,
                         const Tensor& grad_output,
                         const std::string& reduction);

}  // namespace orthon
