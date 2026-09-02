#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/maxpool2d_kernels.h>
#endif

namespace orthon {

class MaxPool2DNode : public Node {
public:
    int64_t kh_, kw_, sh_, sw_, ph_, pw_;

    MaxPool2DNode(const Tensor& input,
                  int64_t kernel_h, int64_t kernel_w,
                  int64_t stride_h, int64_t stride_w,
                  int64_t pad_h, int64_t pad_w)
        : kh_(kernel_h), kw_(kernel_w),
          sh_(stride_h), sw_(stride_w),
          ph_(pad_h), pw_(pad_w) {
        inputs = {input};
        // Backward re-runs the window max search over forward-time input
        // values (no indices were saved), so the payload must be intact.
        watch_version(input, "input");
    }

    const char* op_name() const override { return "MaxPool2DNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        const Tensor& input = inputs[0];

#ifdef WITH_CUDA
        if (input.device().is_cuda()) {
            return {cuda_maxpool2d_backward(input, grad, kh_, kw_, sh_, sw_, ph_, pw_)};
        }
#endif
        if (!input.device().is_cpu())
            throw std::runtime_error("maxpool2d backward: unsupported device");

        const int64_t N = input.shape()[0];
        const int64_t C = input.shape()[1];
        const int64_t H = input.shape()[2];
        const int64_t W = input.shape()[3];
        const int64_t H_out = grad.shape()[2];
        const int64_t W_out = grad.shape()[3];

        const float* ip = static_cast<const float*>(input.data());
        const float* go = static_cast<const float*>(grad.data());
        Tensor gi = zeros(input.shape(), DType::Float32, input.device());
        float* gi_p = static_cast<float*>(gi.data());

        const int64_t channel_stride = H * W;
        const int64_t image_stride = C * channel_stride;
        const int64_t out_channel_stride = H_out * W_out;
        const int64_t out_image_stride = C * out_channel_stride;

        for (int64_t n = 0; n < N; ++n) {
            const float* input_n = ip + n * image_stride;
            float* grad_input_n = gi_p + n * image_stride;
            const float* grad_n = go + n * out_image_stride;

            for (int64_t c = 0; c < C; ++c) {
                const float* input_c = input_n + c * channel_stride;
                float* grad_input_c = grad_input_n + c * channel_stride;
                const float* grad_c = grad_n + c * out_channel_stride;

                for (int64_t oh = 0; oh < H_out; ++oh) {
                    const int64_t h_base = oh * sh_ - ph_;
                    const int64_t h_begin = std::max<int64_t>(0, h_base);
                    const int64_t h_end = std::min<int64_t>(H, h_base + kh_);

                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        const int64_t w_base = ow * sw_ - pw_;
                        const int64_t w_begin = std::max<int64_t>(0, w_base);
                        const int64_t w_end = std::min<int64_t>(W, w_base + kw_);

                        float max_value = -std::numeric_limits<float>::infinity();
                        int64_t max_index = -1;
                        for (int64_t ih = h_begin; ih < h_end; ++ih) {
                            const float* input_row = input_c + ih * W;
                            for (int64_t iw = w_begin; iw < w_end; ++iw) {
                                const float value = input_row[iw];
                                if (value > max_value) {
                                    max_value = value;
                                    max_index = ih * W + iw;
                                }
                            }
                        }

                        if (max_index >= 0) {
                            grad_input_c[max_index] += grad_c[oh * W_out + ow];
                        }
                    }
                }
            }
        }

        return {gi};
    }
};

}  // namespace orthon
