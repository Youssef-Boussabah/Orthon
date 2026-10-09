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
#include <orthon/planning/maxpool2d_validation.h>

namespace orthon { namespace backend { namespace cpu {
using namespace dispatch;
Tensor cpu_maxpool2d(const Tensor& input, const SpatialArgs& a) {
    const auto geometry = validate_maxpool2d_input(input, a);
    if (!input.device().is_cpu())
        throw std::invalid_argument("maxpool2d: CPU backend requires CPU input");
    const auto sh = a.sh, sw = a.sw, ph = a.ph, pw = a.pw, kh = a.kh, kw = a.kw;
    const auto N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
    const auto H_out = geometry.height, W_out = geometry.width;
    Tensor out = empty({N, C, H_out, W_out}, input.dtype(), input.device());
    const float* ip = static_cast<const float*>(input.data());
    float* op = static_cast<float*>(out.data());
    if (input.numel() == 0) {
        if (out.numel() > 0)
            std::fill_n(op, out.numel(), -std::numeric_limits<float>::infinity());
        return out;
    }

    const int64_t channel_stride = H * W;
    const int64_t image_stride = C * channel_stride;
    const int64_t out_channel_stride = H_out * W_out;
    const int64_t out_image_stride = C * out_channel_stride;

    for (int64_t n = 0; n < N; ++n) {
        const float* input_n = ip + n * image_stride;
        float* output_n = op + n * out_image_stride;

        for (int64_t c = 0; c < C; ++c) {
            const float* input_c = input_n + c * channel_stride;
            float* output_c = output_n + c * out_channel_stride;

            for (int64_t oh = 0; oh < H_out; ++oh) {
                const int64_t h_base = oh * sh - ph;
                const int64_t h_begin = std::max<int64_t>(0, h_base);
                const int64_t h_end = std::min<int64_t>(H, h_base + kh);

                for (int64_t ow = 0; ow < W_out; ++ow) {
                    const int64_t w_base = ow * sw - pw;
                    const int64_t w_begin = std::max<int64_t>(0, w_base);
                    const int64_t w_end = std::min<int64_t>(W, w_base + kw);

                    float max_value = -std::numeric_limits<float>::infinity();
                    bool has_value = false;
                    for (int64_t ih = h_begin; ih < h_end; ++ih) {
                        const float* input_row = input_c + ih * W;
                        for (int64_t iw = w_begin; iw < w_end; ++iw) {
                            const float value = input_row[iw];
                            if (!has_value || (!std::isnan(value) &&
                                (std::isnan(max_value) || value > max_value))) {
                                max_value = value;
                                has_value = true;
                            }
                        }
                    }
                    output_c[oh * W_out + ow] = max_value;
                }
            }
        }
    }

    return out;
}
Tensor cpu_maxpool2d_backward(const Tensor& input, const Tensor& grad, const SpatialArgs& a) {
    const auto geometry = validate_maxpool2d_gradient(input, grad, a);
    if (!input.device().is_cpu())
        throw std::invalid_argument("maxpool2d: CPU backend requires CPU input");
    const auto sh_ = a.sh, sw_ = a.sw, ph_ = a.ph, pw_ = a.pw, kh_ = a.kh, kw_ = a.kw;
        const int64_t N = input.shape()[0];
        const int64_t C = input.shape()[1];
        const int64_t H = input.shape()[2];
        const int64_t W = input.shape()[3];
        const int64_t H_out = geometry.height;
        const int64_t W_out = geometry.width;

        const float* ip = static_cast<const float*>(input.data());
        const float* go = static_cast<const float*>(grad.data());
        Tensor gi = zeros(input.shape(), DType::Float32, input.device());
        float* gi_p = static_cast<float*>(gi.data());
        if (gi.numel() == 0) return gi;

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
                                if (max_index < 0 || (!std::isnan(value) &&
                                    (std::isnan(max_value) || value > max_value))) {
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

    return gi;
}

} } }
