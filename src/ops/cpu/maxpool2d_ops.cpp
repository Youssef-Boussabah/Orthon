#include <orthon/ops/cpu/maxpool2d_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/maxpool2d_node.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/maxpool2d_kernels.h>
#endif

namespace orthon {

static void validate_maxpool2d(const Tensor& input,
                               int64_t kh, int64_t kw,
                               int64_t sh, int64_t sw,
                               int64_t ph, int64_t pw) {
    if (input.ndim() != 4)
        throw std::runtime_error("maxpool2d: input must be 4D [N,C,H,W]");
    if (input.dtype() != DType::Float32)
        throw std::runtime_error("maxpool2d: only float32 supported");
    if (!input.is_contiguous())
        throw std::runtime_error("maxpool2d: input must be contiguous");
    if (kh <= 0 || kw <= 0)
        throw std::runtime_error("maxpool2d: kernel_size must be positive");
    if (sh <= 0 || sw <= 0)
        throw std::runtime_error("maxpool2d: stride must be positive");
    if (ph < 0 || pw < 0)
        throw std::runtime_error("maxpool2d: padding must be non-negative");
    if (ph > kh / 2 || pw > kw / 2)
        throw std::runtime_error("maxpool2d: padding must be at most half the kernel size");
}

Tensor maxpool2d(const Tensor& input,
                 int64_t kh, int64_t kw,
                 int64_t sh, int64_t sw,
                 int64_t ph, int64_t pw) {
    validate_maxpool2d(input, kh, kw, sh, sw, ph, pw);

    const int64_t N = input.shape()[0];
    const int64_t C = input.shape()[1];
    const int64_t H = input.shape()[2];
    const int64_t W = input.shape()[3];

    const int64_t H_out = (H + 2 * ph - kh) / sh + 1;
    const int64_t W_out = (W + 2 * pw - kw) / sw + 1;
    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("maxpool2d: output spatial size is non-positive");

#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        Tensor out = cuda_maxpool2d_forward(input, kh, kw, sh, sw, ph, pw);
        if (is_grad_enabled() && input.requires_grad()) {
            out.requires_grad_(true);
            out.set_is_leaf(false);
            out.set_grad_fn(std::make_shared<MaxPool2DNode>(input, kh, kw, sh, sw, ph, pw));
        }
        return out;
    }
#else
    if (input.device().is_cuda())
        throw std::runtime_error("maxpool2d: CUDA is not available in this build");
#endif

    if (!input.device().is_cpu())
        throw std::runtime_error("maxpool2d: unsupported device");

    Tensor out = empty({N, C, H_out, W_out}, input.dtype(), input.device());
    const float* ip = static_cast<const float*>(input.data());
    float* op = static_cast<float*>(out.data());

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
                    for (int64_t ih = h_begin; ih < h_end; ++ih) {
                        const float* input_row = input_c + ih * W;
                        for (int64_t iw = w_begin; iw < w_end; ++iw) {
                            max_value = std::max(max_value, input_row[iw]);
                        }
                    }
                    output_c[oh * W_out + ow] = max_value;
                }
            }
        }
    }

    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<MaxPool2DNode>(input, kh, kw, sh, sw, ph, pw));
    }
    return out;
}

}  // namespace orthon
