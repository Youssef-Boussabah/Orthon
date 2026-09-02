#pragma once
#include <orthon/core/tensor.h>
#include <cstdint>

namespace orthon {

// 2-D convolution (contiguous tensors; dispatches to CUDA for CUDA inputs).
//   input : [N, C_in, H, W]
//   weight: [C_out, C_in, K_h, K_w]
//   bias  : [C_out]
// Returns [N, C_out, H_out, W_out] where
//   H_out = (H + 2*pad_h - K_h) / stride_h + 1
//   W_out = (W + 2*pad_w - K_w) / stride_w + 1
// Attaches Conv2DNode when any input requires grad.
Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              int64_t stride_h = 1, int64_t stride_w = 1,
              int64_t pad_h    = 0, int64_t pad_w    = 0);

// Internal CPU forward variants used for benchmarking and parity tests.
// Both attach the same Conv2DNode when gradients are required.
Tensor conv2d_direct(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     int64_t stride_h = 1, int64_t stride_w = 1,
                     int64_t pad_h    = 0, int64_t pad_w    = 0);

Tensor conv2d_im2col(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     int64_t stride_h = 1, int64_t stride_w = 1,
                     int64_t pad_h    = 0, int64_t pad_w    = 0);

}  // namespace orthon
