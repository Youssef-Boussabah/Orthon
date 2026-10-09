#pragma once
#include <orthon/core/tensor.h>
#include <cstdint>

namespace orthon {

// 2-D max pooling (contiguous tensors; dispatches to CUDA for CUDA inputs).
//   input : [N, C, H, W]
// Returns [N, C, H_out, W_out] where
//   H_out = floor((H + 2*pad_h - kernel_h) / stride_h) + 1
//   W_out = floor((W + 2*pad_w - kernel_w) / stride_w) + 1
// Negative numerators and unrepresentable geometry are rejected before dispatch.
// Zero spatial extents with legal padding-only windows remain supported.
Tensor maxpool2d(const Tensor& input,
                 int64_t kernel_h, int64_t kernel_w,
                 int64_t stride_h, int64_t stride_w,
                 int64_t pad_h = 0, int64_t pad_w = 0);

}  // namespace orthon
