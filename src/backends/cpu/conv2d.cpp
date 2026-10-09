#include <orthon/planning/conv2d_validation.h>
#include <cstdlib>
#include <orthon/dispatch/operator.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/float_formats.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include <orthon/backends/cpu/kernels.h>

namespace orthon { namespace backend { namespace cpu {
using namespace dispatch;
static Tensor conv2d_direct_values(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    const auto geometry = validate_conv2d_forward(input, weight, bias, sh, sw, ph, pw);

    const int64_t N    = input.shape()[0];
    const int64_t C_in = input.shape()[1];
    const int64_t H    = input.shape()[2];
    const int64_t W    = input.shape()[3];
    const int64_t C_out= weight.shape()[0];
    const int64_t K_h  = weight.shape()[2];
    const int64_t K_w  = weight.shape()[3];

    const int64_t H_out = geometry.height;
    const int64_t W_out = geometry.width;

    Tensor out = empty({N, C_out, H_out, W_out});
    const float* ip = static_cast<const float*>(input.data());
    const float* wp = static_cast<const float*>(weight.data());
    const float* bp = static_cast<const float*>(bias.data());
    float*       op = static_cast<float*>(out.data());

    const int64_t input_image_stride = C_in * H * W;
    const int64_t input_channel_stride = H * W;
    const int64_t output_image_stride = C_out * H_out * W_out;
    const int64_t output_channel_stride = H_out * W_out;
    const int64_t weight_filter_stride = C_in * K_h * K_w;
    const int64_t weight_channel_stride = K_h * K_w;

    for (int64_t n = 0; n < N; ++n) {
        const float* input_n = ip + n * input_image_stride;
        float* output_n = op + n * output_image_stride;

        for (int64_t f = 0; f < C_out; ++f) {
            const float* weight_f = wp + f * weight_filter_stride;
            float* output_nf = output_n + f * output_channel_stride;

            for (int64_t oh = 0; oh < H_out; ++oh) {
                const int64_t h_base = oh * sh - ph;
                const int64_t kh_begin = std::max<int64_t>(0, -h_base);
                const int64_t kh_end = std::min<int64_t>(K_h, H - h_base);

                for (int64_t ow = 0; ow < W_out; ++ow) {
                    const int64_t w_base = ow * sw - pw;
                    const int64_t kw_begin = std::max<int64_t>(0, -w_base);
                    const int64_t kw_end = std::min<int64_t>(K_w, W - w_base);

                    float val = bp[f];
                    if (kh_begin < kh_end && kw_begin < kw_end) {
                        for (int64_t c = 0; c < C_in; ++c) {
                            const float* input_c = input_n + c * input_channel_stride;
                            const float* weight_fc = weight_f + c * weight_channel_stride;

                            for (int64_t kh = kh_begin; kh < kh_end; ++kh) {
                                const float* input_row = input_c + (h_base + kh) * W;
                                const float* weight_row = weight_fc + kh * K_w;

                                for (int64_t kw = kw_begin; kw < kw_end; ++kw) {
                                    val += input_row[w_base + kw] * weight_row[kw];
                                }
                            }
                        }
                    }
                    output_nf[oh * W_out + ow] = val;
                }
            }
        }
    }

    return out;
}
Tensor cpu_conv2d_direct(const Tensor& input, const Tensor& weight, const Tensor& bias, const SpatialArgs& a) {
    return conv2d_direct_values(input, weight, bias, a.sh, a.sw, a.ph, a.pw);
}
static Tensor conv2d_im2col_values(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    const auto geometry = validate_conv2d_forward(input, weight, bias, sh, sw, ph, pw);

    const int64_t N     = input.shape()[0];
    const int64_t C_in  = input.shape()[1];
    const int64_t H     = input.shape()[2];
    const int64_t W     = input.shape()[3];
    const int64_t C_out = weight.shape()[0];
    const int64_t K_h   = weight.shape()[2];
    const int64_t K_w   = weight.shape()[3];

    const int64_t H_out = geometry.height;
    const int64_t W_out = geometry.width;

    const int64_t patch_count = Layout({N, H_out, W_out}).numel();
    const int64_t kernel_size = C_in * K_h * K_w;

    Tensor columns = empty({patch_count, kernel_size});
    Tensor out = empty({N, C_out, H_out, W_out});

    const float* ip = static_cast<const float*>(input.data());
    const float* wp = static_cast<const float*>(weight.data());
    const float* bp = static_cast<const float*>(bias.data());
    float* col = static_cast<float*>(columns.data());
    float* op = static_cast<float*>(out.data());

    const int64_t input_image_stride = C_in * H * W;
    const int64_t input_channel_stride = H * W;
    const int64_t output_image_stride = C_out * H_out * W_out;
    const int64_t output_channel_stride = H_out * W_out;

    for (int64_t n = 0; n < N; ++n) {
        const float* input_n = ip + n * input_image_stride;
        for (int64_t oh = 0; oh < H_out; ++oh) {
            const int64_t h_base = oh * sh - ph;
            for (int64_t ow = 0; ow < W_out; ++ow) {
                const int64_t w_base = ow * sw - pw;
                const int64_t row = (n * H_out + oh) * W_out + ow;
                float* col_row = col + row * kernel_size;

                for (int64_t c = 0; c < C_in; ++c) {
                    const float* input_c = input_n + c * input_channel_stride;
                    for (int64_t kh = 0; kh < K_h; ++kh) {
                        const int64_t ih = h_base + kh;
                        for (int64_t kw = 0; kw < K_w; ++kw) {
                            const int64_t iw = w_base + kw;
                            const int64_t k = (c * K_h + kh) * K_w + kw;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                col_row[k] = 0.0f;
                            } else {
                                col_row[k] = input_c[ih * W + iw];
                            }
                        }
                    }
                }
            }
        }
    }

    for (int64_t n = 0; n < N; ++n) {
        float* output_n = op + n * output_image_stride;
        for (int64_t oh = 0; oh < H_out; ++oh) {
            for (int64_t ow = 0; ow < W_out; ++ow) {
                const int64_t row = (n * H_out + oh) * W_out + ow;
                const float* col_row = col + row * kernel_size;

                for (int64_t f = 0; f < C_out; ++f) {
                    const float* weight_f = wp + f * kernel_size;
                    float acc = bp[f];
                    for (int64_t k = 0; k < kernel_size; ++k) {
                        acc += col_row[k] * weight_f[k];
                    }
                    output_n[f * output_channel_stride + oh * W_out + ow] = acc;
                }
            }
        }
    }

    return out;
}
Tensor cpu_conv2d_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias, const SpatialArgs& a) {
    return conv2d_im2col_values(input, weight, bias, a.sh, a.sw, a.ph, a.pw);
}
Tensor cpu_conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, const SpatialArgs& a) {
    const char* algo = std::getenv("ORTHON_CONV2D_FORWARD");
    if (algo && std::string(algo) == "direct") return cpu_conv2d_direct(input, weight, bias, a);
    return cpu_conv2d_im2col(input, weight, bias, a);
}
std::vector<Tensor> cpu_conv2d_backward(const Tensor& inp, const Tensor& wgt, const Tensor& bia, const Tensor& grad, const SpatialArgs& args) {
    validate_conv2d_forward(inp, wgt, bia, args.sh, args.sw, args.ph, args.pw);
    validate_conv2d_gradient(inp, wgt, grad, args.sh, args.sw, args.ph, args.pw);
    const auto sh_ = args.sh, sw_ = args.sw, ph_ = args.ph, pw_ = args.pw;
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

} } }
