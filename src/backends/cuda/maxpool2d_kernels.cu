#include <orthon/backends/cuda/maxpool2d_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/runtime.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/planning/maxpool2d_validation.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <float.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

namespace orthon {

static constexpr int kThreads = 256;

static int cuda_blocks(int64_t n) {
    return static_cast<int>(checked_maxpool2d_cuda_blocks(n));
}

__global__ void maxpool2d_forward_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int64_t total,
        int N, int C, int H, int W,
        int H_out, int W_out,
        int K_h, int K_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int ow = static_cast<int>(tmp % W_out);
        tmp /= W_out;
        const int oh = static_cast<int>(tmp % H_out);
        tmp /= H_out;
        const int c = static_cast<int>(tmp % C);
        tmp /= C;
        const int n = static_cast<int>(tmp);

        const int64_t h_base = static_cast<int64_t>(oh) * stride_h - pad_h;
        const int64_t w_base = static_cast<int64_t>(ow) * stride_w - pad_w;
        const int64_t h_begin = h_base > 0 ? h_base : 0;
        const int64_t h_end = h_base + K_h < H ? h_base + K_h : H;
        const int64_t w_begin = w_base > 0 ? w_base : 0;
        const int64_t w_end = w_base + K_w < W ? w_base + K_w : W;

        if (h_begin >= h_end || w_begin >= w_end) {
            output[linear] = -CUDART_INF_F;
            continue;
        }
        const float* input_c = input + ((int64_t)n * C + c) * H * W;
        float max_value = -CUDART_INF_F;
        bool has_value = false;
        for (int64_t ih = h_begin; ih < h_end; ++ih) {
            for (int64_t iw = w_begin; iw < w_end; ++iw) {
                const float value = input_c[ih * W + iw];
                if (!has_value || (!isnan(value) &&
                    (isnan(max_value) || value > max_value))) {
                    max_value = value;
                    has_value = true;
                }
            }
        }
        output[linear] = max_value;
    }
}

__global__ void maxpool2d_backward_kernel(
        const float* __restrict__ input,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_input,
        int64_t total,
        int N, int C, int H, int W,
        int H_out, int W_out,
        int K_h, int K_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int iw = static_cast<int>(tmp % W);
        tmp /= W;
        const int ih = static_cast<int>(tmp % H);
        tmp /= H;
        const int c = static_cast<int>(tmp % C);
        tmp /= C;
        const int n = static_cast<int>(tmp);

        const float* input_c = input + ((int64_t)n * C + c) * H * W;
        const float* grad_output_c = grad_output + ((int64_t)n * C + c) * H_out * W_out;
        float acc = 0.0f;

        for (int oh = 0; oh < H_out; ++oh) {
            const int64_t h_base = static_cast<int64_t>(oh) * stride_h - pad_h;
            const int64_t h_begin = h_base > 0 ? h_base : 0;
            const int64_t h_end = h_base + K_h < H ? h_base + K_h : H;
            if (ih < h_begin || ih >= h_end) {
                continue;
            }

            for (int ow = 0; ow < W_out; ++ow) {
                const int64_t w_base = static_cast<int64_t>(ow) * stride_w - pad_w;
                const int64_t w_begin = w_base > 0 ? w_base : 0;
                const int64_t w_end = w_base + K_w < W ? w_base + K_w : W;
                if (iw < w_begin || iw >= w_end) {
                    continue;
                }

                float max_value = -CUDART_INF_F;
                int64_t max_index = -1;
                for (int64_t kh = h_begin; kh < h_end; ++kh) {
                    for (int64_t kw = w_begin; kw < w_end; ++kw) {
                        const int64_t index = kh * W + kw;
                        const float value = input_c[index];
                        if (max_index < 0 || (!isnan(value) &&
                            (isnan(max_value) || value > max_value))) {
                            max_value = value;
                            max_index = index;
                        }
                    }
                }
                if (max_index == static_cast<int64_t>(ih) * W + iw) {
                    acc += grad_output_c[static_cast<int64_t>(oh) * W_out + ow];
                }
            }
        }
        grad_input[linear] = acc;
    }
}

Tensor cuda_maxpool2d_forward(const Tensor& input,
                              int64_t kernel_h, int64_t kernel_w,
                              int64_t stride_h, int64_t stride_w,
                              int64_t pad_h, int64_t pad_w) {
    const auto geometry = validate_maxpool2d_input(
        input, {stride_h, stride_w, pad_h, pad_w, kernel_h, kernel_w});
    if (!input.device().is_cuda())
        throw std::invalid_argument("maxpool2d: CUDA backend requires CUDA input");
    DeviceGuard guard(input.device());
    auto context = default_runtime().cuda(input.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int N = static_cast<int>(input.shape()[0]);
    const int C = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int H_out = static_cast<int>(geometry.height);
    const int W_out = static_cast<int>(geometry.width);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{N, C, H_out, W_out}, input.dtype(), input.device()));
    if (out.numel() == 0) return out;

    maxpool2d_forward_kernel<<<cuda_blocks(out.numel()), kThreads, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(out.data()),
        out.numel(),
        N, C, H, W,
        H_out, W_out,
        static_cast<int>(kernel_h), static_cast<int>(kernel_w),
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(pad_h), static_cast<int>(pad_w));
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_maxpool2d_backward(const Tensor& input,
                               const Tensor& grad_output,
                               int64_t kernel_h, int64_t kernel_w,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w) {
    const auto geometry = validate_maxpool2d_gradient(
        input, grad_output, {stride_h, stride_w, pad_h, pad_w, kernel_h, kernel_w});
    if (!input.device().is_cuda())
        throw std::invalid_argument("maxpool2d: CUDA backend requires CUDA input");
    DeviceGuard guard(input.device());
    auto context = default_runtime().cuda(input.device().index);
    const auto stream = static_cast<cudaStream_t>(context->compute_stream().native_handle());

    const int N = static_cast<int>(input.shape()[0]);
    const int C = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int H_out = static_cast<int>(geometry.height);
    const int W_out = static_cast<int>(geometry.width);

    Tensor grad_input(std::make_shared<TensorImpl>(
        input.shape(), input.dtype(), input.device()));
    if (grad_input.numel() == 0) return grad_input;

    maxpool2d_backward_kernel<<<cuda_blocks(grad_input.numel()), kThreads, 0, stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_input.data()),
        grad_input.numel(),
        N, C, H, W,
        H_out, W_out,
        static_cast<int>(kernel_h), static_cast<int>(kernel_w),
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(pad_h), static_cast<int>(pad_w));
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

}  // namespace orthon
