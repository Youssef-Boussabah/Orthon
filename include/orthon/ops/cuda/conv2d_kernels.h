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

// CUDA Conv2D backward. Returns {grad_input, grad_weight, grad_bias}.
// grad_weight and grad_bias default to optimized block-reduction kernels unless
// ORTHON_CUDA_CONV2D_GRAD_INPUT=naive,
// ORTHON_CUDA_CONV2D_GRAD_WEIGHT=naive or
// ORTHON_CUDA_CONV2D_GRAD_BIAS=naive is set.
// ORTHON_CUDA_CONV2D_BIAS_GRAD=fused forces a combined weight+bias launch;
// small CIFAR-style channel counts use that fused path by default.
// ORTHON_CUDA_CONV2D_GRAD_WEIGHT=grouped3x3 selects the grouped 3x3
// grad-weight kernel; it is also the default for 3x3/stride-1/pad-1 float32.
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

void cuda_conv2d_profile_reset();
std::string cuda_conv2d_profile_summary();

}  // namespace orthon
