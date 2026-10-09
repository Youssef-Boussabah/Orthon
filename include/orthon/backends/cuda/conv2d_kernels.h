#pragma once
#include <orthon/core/tensor.h>
#include <cstdint>
#include <string>
#include <vector>

namespace orthon {

// Dispatching CUDA Conv2D forward. Defaults to optimized forward unless
// ORTHON_CUDA_CONV2D_FORWARD=naive is set.
Tensor cuda_conv2d_forward(const Tensor& input,
                           const Tensor& weight,
                           const Tensor& bias,
                           int64_t stride_h, int64_t stride_w,
                           int64_t pad_h, int64_t pad_w);

// Naive CUDA Conv2D forward: one thread per output element.
// Preconditions (caller verifies): input, weight, and bias are contiguous
// float32/float16/bfloat16 CUDA tensors on the same device.
Tensor cuda_conv2d_forward_naive(const Tensor& input,
                                 const Tensor& weight,
                                 const Tensor& bias,
                                 int64_t stride_h, int64_t stride_w,
                                 int64_t pad_h, int64_t pad_w);

// Optimized CUDA Conv2D forward: one block computes a spatial output tile for
// one (batch, output-channel) plane and caches that output channel's filter in
// shared memory.
Tensor cuda_conv2d_forward_optimized(const Tensor& input,
                                     const Tensor& weight,
                                     const Tensor& bias,
                                     int64_t stride_h, int64_t stride_w,
                                     int64_t pad_h, int64_t pad_w);

// CUDA Conv2D backward returns {grad_input, grad_weight, grad_bias}.
// ORTHON_CUDA_CONV2D_GRAD_INPUT/GRAD_WEIGHT/GRAD_BIAS=naive selects
// reference kernels. BIAS_GRAD=fused combines weight and bias reduction;
// small channel counts use it by default. GRAD_WEIGHT=grouped3x3 selects
// the grouped path, also the default for 3x3/stride-1/pad-1 Float32.
std::vector<Tensor> cuda_conv2d_backward(const Tensor& input,
                                         const Tensor& weight,
                                         const Tensor& bias,
                                         const Tensor& grad_output,
                                         int64_t stride_h, int64_t stride_w,
                                         int64_t pad_h, int64_t pad_w);

// Benchmark/profiling helpers for the backward component kernels.
Tensor cuda_conv2d_backward_input(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& grad_output,
                                  int64_t stride_h, int64_t stride_w,
                                  int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_input_naive(const Tensor& input,
                                        const Tensor& weight,
                                        const Tensor& grad_output,
                                        int64_t stride_h, int64_t stride_w,
                                        int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_input_optimized(const Tensor& input,
                                            const Tensor& weight,
                                            const Tensor& grad_output,
                                            int64_t stride_h, int64_t stride_w,
                                            int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_input_optimized_v2(const Tensor& input,
                                               const Tensor& weight,
                                               const Tensor& grad_output,
                                               int64_t stride_h, int64_t stride_w,
                                               int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_weight(const Tensor& input,
                                   const Tensor& weight,
                                   const Tensor& grad_output,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_weight_naive(const Tensor& input,
                                         const Tensor& weight,
                                         const Tensor& grad_output,
                                         int64_t stride_h, int64_t stride_w,
                                         int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_weight_optimized(const Tensor& input,
                                             const Tensor& weight,
                                             const Tensor& grad_output,
                                             int64_t stride_h, int64_t stride_w,
                                             int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_weight_optimized_v2(const Tensor& input,
                                                const Tensor& weight,
                                                const Tensor& grad_output,
                                                int64_t stride_h, int64_t stride_w,
                                                int64_t pad_h, int64_t pad_w);

std::vector<Tensor> cuda_conv2d_backward_weight_bias_optimized(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w);

Tensor cuda_conv2d_backward_bias(const Tensor& bias,
                                 const Tensor& grad_output);

Tensor cuda_conv2d_backward_bias_naive(const Tensor& bias,
                                       const Tensor& grad_output);

Tensor cuda_conv2d_backward_bias_optimized(const Tensor& bias,
                                           const Tensor& grad_output);


}  // namespace orthon
