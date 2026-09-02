#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/conv2d_kernels.h>
#endif

namespace orthon {

class Conv2DNode : public Node {
public:
    int64_t sh_, sw_, ph_, pw_;

    Conv2DNode(const Tensor& input, const Tensor& weight, const Tensor& bias,
               int64_t sh, int64_t sw, int64_t ph, int64_t pw)
        : sh_(sh), sw_(sw), ph_(ph), pw_(pw) {
        inputs = {input, weight, bias};
        // grad_input reads weight values, grad_weight reads input values.
        // grad_bias is a reduction of grad_output alone on both CPU (sums g)
        // and CUDA (cuda_conv2d_backward_bias touches only bias.shape/dtype/
        // device), so the bias PAYLOAD is deliberately not watched.
        watch_version(input, "input");
        watch_version(weight, "weight");
    }

    const char* op_name() const override { return "Conv2DNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        const Tensor& inp = inputs[0];
        const Tensor& wgt = inputs[1];
        const Tensor& bia = inputs[2];

#ifdef WITH_CUDA
        if (inp.device().is_cuda()) {
            return cuda_conv2d_backward(inp, wgt, bia, grad, sh_, sw_, ph_, pw_);
        }
#endif
        if (!inp.device().is_cpu())
            throw std::runtime_error("conv2d backward: unsupported device");

        const int64_t N    = inp.shape()[0];
        const int64_t C_in = inp.shape()[1];
        const int64_t H    = inp.shape()[2];
        const int64_t W    = inp.shape()[3];
        const int64_t C_out= wgt.shape()[0];
        const int64_t K_h  = wgt.shape()[2];
        const int64_t K_w  = wgt.shape()[3];
        const int64_t H_out= grad.shape()[2];
        const int64_t W_out= grad.shape()[3];

        const float* go  = static_cast<const float*>(grad.data());
        const float* ip  = static_cast<const float*>(inp.data());
        const float* wp  = static_cast<const float*>(wgt.data());

        Tensor gi = zeros(inp.shape(),  DType::Float32, inp.device());
        Tensor gw = zeros(wgt.shape(),  DType::Float32, wgt.device());
        Tensor gb = zeros(bia.shape(),  DType::Float32, bia.device());

        float* gi_p = static_cast<float*>(gi.data());
        float* gw_p = static_cast<float*>(gw.data());
        float* gb_p = static_cast<float*>(gb.data());

        const int64_t input_image_stride = C_in * H * W;
        const int64_t input_channel_stride = H * W;
        const int64_t output_image_stride = C_out * H_out * W_out;
        const int64_t output_channel_stride = H_out * W_out;
        const int64_t weight_filter_stride = C_in * K_h * K_w;
        const int64_t weight_channel_stride = K_h * K_w;

        for (int64_t n = 0; n < N; ++n) {
            const float* input_n = ip + n * input_image_stride;
            float* grad_input_n = gi_p + n * input_image_stride;
            const float* grad_n = go + n * output_image_stride;

            for (int64_t f = 0; f < C_out; ++f) {
                const float* weight_f = wp + f * weight_filter_stride;
                float* grad_weight_f = gw_p + f * weight_filter_stride;
                const float* grad_nf = grad_n + f * output_channel_stride;
                float bias_grad = 0.0f;

                for (int64_t h = 0; h < H_out; ++h) {
                    const int64_t h_base = h * sh_ - ph_;
                    const int64_t kh_begin = std::max<int64_t>(0, -h_base);
                    const int64_t kh_end = std::min<int64_t>(K_h, H - h_base);

                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        const float g = grad_nf[h * W_out + ow];
                        bias_grad += g;

                        const int64_t w_base = ow * sw_ - pw_;
                        const int64_t kw_begin = std::max<int64_t>(0, -w_base);
                        const int64_t kw_end = std::min<int64_t>(K_w, W - w_base);
                        if (kh_begin >= kh_end || kw_begin >= kw_end) {
                            continue;
                        }

                        for (int64_t c = 0; c < C_in; ++c) {
                            const float* input_c = input_n + c * input_channel_stride;
                            float* grad_input_c = grad_input_n + c * input_channel_stride;
                            const float* weight_fc = weight_f + c * weight_channel_stride;
                            float* grad_weight_fc = grad_weight_f + c * weight_channel_stride;

                            for (int64_t kh = kh_begin; kh < kh_end; ++kh) {
                                const float* input_row = input_c + (h_base + kh) * W;
                                float* grad_input_row = grad_input_c + (h_base + kh) * W;
                                const float* weight_row = weight_fc + kh * K_w;
                                float* grad_weight_row = grad_weight_fc + kh * K_w;

                                for (int64_t kw = kw_begin; kw < kw_end; ++kw) {
                                    const int64_t in_col = w_base + kw;
                                    grad_input_row[in_col] += g * weight_row[kw];
                                    grad_weight_row[kw] += g * input_row[in_col];
                                }
                            }
                        }
                    }
                }
                gb_p[f] += bias_grad;
            }
        }

        return {gi, gw, gb};
    }
};

}  // namespace orthon
