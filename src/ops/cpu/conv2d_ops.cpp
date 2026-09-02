#include <orthon/ops/cpu/conv2d_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/autograd/conv2d_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/dtype.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/conv2d_kernels.h>
#endif

namespace orthon {

static void validate_conv2d(const Tensor& input,
                             const Tensor& weight,
                             const Tensor& bias) {
    if (input.ndim()  != 4) throw std::runtime_error("conv2d: input must be 4D [N,C,H,W]");
    if (weight.ndim() != 4) throw std::runtime_error("conv2d: weight must be 4D [Co,Ci,Kh,Kw]");
    if (bias.ndim()   != 1) throw std::runtime_error("conv2d: bias must be 1D [C_out]");
    const bool is_cuda = input.device().is_cuda();
    const bool all_float32 = input.dtype()  == DType::Float32 &&
                             weight.dtype() == DType::Float32 &&
                             bias.dtype()   == DType::Float32;
    const bool all_float16 = input.dtype()  == DType::Float16 &&
                             weight.dtype() == DType::Float16 &&
                             bias.dtype()   == DType::Float16;
    const bool all_bfloat16 = input.dtype()  == DType::BFloat16 &&
                              weight.dtype() == DType::BFloat16 &&
                              bias.dtype()   == DType::BFloat16;
    if (!all_float32 && !(is_cuda && all_float16) && !(is_cuda && all_bfloat16))
        throw std::runtime_error("conv2d: only float32 supported (float16/bfloat16 requires CUDA)");
    if (input.device() != weight.device() || input.device() != bias.device())
        throw std::runtime_error("conv2d: input, weight, and bias must be on the same device");
    if (!input.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous())
        throw std::runtime_error("conv2d: all tensors must be contiguous");
    if (input.shape()[1] != weight.shape()[1])
        throw std::runtime_error("conv2d: C_in mismatch between input and weight");
    if (bias.shape()[0]  != weight.shape()[0])
        throw std::runtime_error("conv2d: bias length must equal C_out");
}

static void attach_conv2d_grad_fn(Tensor& out,
                                  const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& bias,
                                  int64_t sh,
                                  int64_t sw,
                                  int64_t ph,
                                  int64_t pw) {
    const bool req = is_grad_enabled() &&
        (input.requires_grad() || weight.requires_grad() || bias.requires_grad());
    if (req) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<Conv2DNode>(input, weight, bias, sh, sw, ph, pw));
    }
}

Tensor conv2d_direct(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    validate_conv2d(input, weight, bias);
    if (!input.device().is_cpu())
        throw std::runtime_error("conv2d_direct: only CPU supported");

    const int64_t N    = input.shape()[0];
    const int64_t C_in = input.shape()[1];
    const int64_t H    = input.shape()[2];
    const int64_t W    = input.shape()[3];
    const int64_t C_out= weight.shape()[0];
    const int64_t K_h  = weight.shape()[2];
    const int64_t K_w  = weight.shape()[3];

    const int64_t H_out = (H + 2*ph - K_h) / sh + 1;
    const int64_t W_out = (W + 2*pw - K_w) / sw + 1;

    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("conv2d: output spatial size is non-positive; "
                                 "check kernel_size, stride, and padding");

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

    attach_conv2d_grad_fn(out, input, weight, bias, sh, sw, ph, pw);
    return out;
}

Tensor conv2d_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    validate_conv2d(input, weight, bias);
    if (!input.device().is_cpu())
        throw std::runtime_error("conv2d_im2col: only CPU supported");

    const int64_t N     = input.shape()[0];
    const int64_t C_in  = input.shape()[1];
    const int64_t H     = input.shape()[2];
    const int64_t W     = input.shape()[3];
    const int64_t C_out = weight.shape()[0];
    const int64_t K_h   = weight.shape()[2];
    const int64_t K_w   = weight.shape()[3];

    const int64_t H_out = (H + 2*ph - K_h) / sh + 1;
    const int64_t W_out = (W + 2*pw - K_w) / sw + 1;

    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("conv2d: output spatial size is non-positive; "
                                 "check kernel_size, stride, and padding");

    const int64_t patch_count = N * H_out * W_out;
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

    attach_conv2d_grad_fn(out, input, weight, bias, sh, sw, ph, pw);
    return out;
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    validate_conv2d(input, weight, bias);
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        const int64_t H = input.shape()[2];
        const int64_t W = input.shape()[3];
        const int64_t K_h = weight.shape()[2];
        const int64_t K_w = weight.shape()[3];
        const int64_t H_out = (H + 2*ph - K_h) / sh + 1;
        const int64_t W_out = (W + 2*pw - K_w) / sw + 1;
        if (H_out <= 0 || W_out <= 0)
            throw std::runtime_error("conv2d: output spatial size is non-positive; "
                                     "check kernel_size, stride, and padding");

        Tensor out = cuda_conv2d_forward(input, weight, bias, sh, sw, ph, pw);
        attach_conv2d_grad_fn(out, input, weight, bias, sh, sw, ph, pw);
        return out;
    }
#else
    if (input.device().is_cuda())
        throw std::runtime_error("conv2d: CUDA is not available in this build");
#endif

    const char* algo = std::getenv("ORTHON_CONV2D_FORWARD");
    if (algo != nullptr) {
        const std::string requested(algo);
        if (requested == "direct") {
            return conv2d_direct(input, weight, bias, sh, sw, ph, pw);
        }
        if (requested == "im2col") {
            return conv2d_im2col(input, weight, bias, sh, sw, ph, pw);
        }
    }
    return conv2d_im2col(input, weight, bias, sh, sw, ph, pw);
}

}  // namespace orthon
