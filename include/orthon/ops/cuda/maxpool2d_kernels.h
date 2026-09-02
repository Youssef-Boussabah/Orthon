#pragma once
#include <orthon/core/tensor.h>
#include <cstdint>

namespace orthon {

// Naive CUDA MaxPool2D forward/backward.
// Preconditions (caller verifies): tensors are contiguous float32 CUDA tensors.
Tensor cuda_maxpool2d_forward(const Tensor& input,
                              int64_t kernel_h, int64_t kernel_w,
                              int64_t stride_h, int64_t stride_w,
                              int64_t pad_h, int64_t pad_w);

Tensor cuda_maxpool2d_backward(const Tensor& input,
                               const Tensor& grad_output,
                               int64_t kernel_h, int64_t kernel_w,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w);

}  // namespace orthon
