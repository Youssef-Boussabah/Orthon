#pragma once

#include <orthon/core/tensor.h>
#include <string>

namespace orthon {

Tensor cuda_nll_loss_forward(const Tensor& log_probs,
                             const Tensor& targets,
                             const std::string& reduction);

Tensor cuda_nll_loss_backward(const Tensor& log_probs,
                              const Tensor& targets,
                              const Tensor& grad_output,
                              const std::string& reduction);

}  // namespace orthon
