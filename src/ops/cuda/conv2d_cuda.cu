#include <orthon/ops/cuda/conv2d_kernels.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)

namespace orthon {

static constexpr int kThreads = 256;
static constexpr int kTileH = 8;
static constexpr int kTileW = 16;
static constexpr int kMaxSharedFilterFloats = 48 * 1024 / static_cast<int>(sizeof(float));
static constexpr int kMinSharedFilterFloats = 64;

static int cuda_blocks(int64_t n, int threads = kThreads) {
    return static_cast<int>((n + threads - 1) / threads);
}

struct Conv2DProfileStats {
    int64_t calls = 0;
    double grad_input_seconds = 0.0;
    double grad_weight_seconds = 0.0;
    double grad_bias_seconds = 0.0;
    double overhead_seconds = 0.0;
    double total_seconds = 0.0;
};

static std::mutex g_conv2d_profile_mutex;
static std::map<std::string, Conv2DProfileStats> g_conv2d_profile;

static bool conv2d_profile_enabled() {
    const char* value = std::getenv("ORTHON_PROFILE_CONV2D");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static std::chrono::high_resolution_clock::time_point conv2d_profile_now() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return std::chrono::high_resolution_clock::now();
}

static double conv2d_profile_elapsed(
        std::chrono::high_resolution_clock::time_point start) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

static std::string conv2d_profile_key(const Tensor& input,
                                      const Tensor& weight,
                                      const Tensor& grad_output,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t pad_h, int64_t pad_w,
                                      const std::string& input_path,
                                      const std::string& weight_path,
                                      const std::string& bias_path) {
    std::ostringstream os;
    os << "N=" << input.shape()[0]
       << " C=" << input.shape()[1]
       << " H=" << input.shape()[2]
       << " W=" << input.shape()[3]
       << " O=" << weight.shape()[0]
       << " K=" << weight.shape()[2] << "x" << weight.shape()[3]
       << " grad_out=[" << grad_output.shape()[0] << ","
       << grad_output.shape()[1] << ","
       << grad_output.shape()[2] << ","
       << grad_output.shape()[3] << "]"
       << " stride=" << stride_h << "x" << stride_w
       << " pad=" << pad_h << "x" << pad_w
       << " device=cuda"
       << " paths=input:" << input_path
       << ",weight:" << weight_path
       << ",bias:" << bias_path;
    return os.str();
}

static void conv2d_profile_record(const std::string& key,
                                  double grad_input_seconds,
                                  double grad_weight_seconds,
                                  double grad_bias_seconds,
                                  double overhead_seconds,
                                  double total_seconds) {
    std::lock_guard<std::mutex> lock(g_conv2d_profile_mutex);
    Conv2DProfileStats& stats = g_conv2d_profile[key];
    stats.calls += 1;
    stats.grad_input_seconds += grad_input_seconds;
    stats.grad_weight_seconds += grad_weight_seconds;
    stats.grad_bias_seconds += grad_bias_seconds;
    stats.overhead_seconds += overhead_seconds;
    stats.total_seconds += total_seconds;
}

void cuda_conv2d_profile_reset() {
    std::lock_guard<std::mutex> lock(g_conv2d_profile_mutex);
    g_conv2d_profile.clear();
}

std::string cuda_conv2d_profile_summary() {
    std::lock_guard<std::mutex> lock(g_conv2d_profile_mutex);
    if (g_conv2d_profile.empty()) {
        return "";
    }

    double total_conv_seconds = 0.0;
    for (const auto& item : g_conv2d_profile) {
        total_conv_seconds += item.second.total_seconds;
    }

    std::vector<std::pair<std::string, Conv2DProfileStats>> rows(
        g_conv2d_profile.begin(), g_conv2d_profile.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second.total_seconds > b.second.total_seconds;
    });

    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "Conv2D backward profile:\n";
    for (const auto& item : rows) {
        const Conv2DProfileStats& stats = item.second;
        const double avg_ms = stats.calls > 0
            ? (stats.total_seconds * 1000.0 / static_cast<double>(stats.calls))
            : 0.0;
        const double pct = total_conv_seconds > 0.0
            ? (stats.total_seconds / total_conv_seconds * 100.0)
            : 0.0;
        os << "  shape " << item.first << ":\n";
        os << "    calls=" << stats.calls
           << " total=" << stats.total_seconds << "s"
           << " avg=" << avg_ms << "ms/call"
           << " pct=" << pct << "%\n";
        os << "    grad_input=" << stats.grad_input_seconds << "s"
           << " grad_weight=" << stats.grad_weight_seconds << "s"
           << " grad_bias=" << stats.grad_bias_seconds << "s"
           << " overhead=" << stats.overhead_seconds << "s\n";
    }
    return os.str();
}

__device__ float warp_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;

    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    value = (threadIdx.x < warp_count) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_sum(value);
    }
    return value;
}

__global__ void conv2d_forward_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
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
        const int co = static_cast<int>(tmp % C_out);
        tmp /= C_out;
        const int n = static_cast<int>(tmp);

        float acc = bias[co];
        const int h_base = oh * stride_h - pad_h;
        const int w_base = ow * stride_w - pad_w;

        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int ih = h_base + kh;
                if (ih < 0 || ih >= H) {
                    continue;
                }
                for (int kw = 0; kw < K_w; ++kw) {
                    const int iw = w_base + kw;
                    if (iw < 0 || iw >= W) {
                        continue;
                    }
                    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
                    const int64_t wt_idx = ((int64_t)co * C_in + ci) * K_h * K_w + kh * K_w + kw;
                    acc += input[in_idx] * weight[wt_idx];
                }
            }
        }

        output[linear] = acc;
    }
}

static Tensor make_conv2d_output(const Tensor& input,
                                 const Tensor& weight,
                                 int64_t stride_h, int64_t stride_w,
                                 int64_t pad_h, int64_t pad_w) {
    const int N = static_cast<int>(input.shape()[0]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>((H + 2 * pad_h - K_h) / stride_h + 1);
    const int W_out = static_cast<int>((W + 2 * pad_w - K_w) / stride_w + 1);

    Tensor out(std::make_shared<TensorImpl>(
        std::vector<int64_t>{N, C_out, H_out, W_out},
        input.dtype(),
        input.device()));
    return out;
}

__global__ void conv2d_forward_kernel_half(
        const __half* __restrict__ input,
        const __half* __restrict__ weight,
        const __half* __restrict__ bias,
        __half* __restrict__ output,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int ow = static_cast<int>(tmp % W_out); tmp /= W_out;
        const int oh = static_cast<int>(tmp % H_out); tmp /= H_out;
        const int co = static_cast<int>(tmp % C_out); tmp /= C_out;
        const int n  = static_cast<int>(tmp);

        float acc = __half2float(bias[co]);
        const int h_base = oh * stride_h - pad_h;
        const int w_base = ow * stride_w - pad_w;

        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int ih = h_base + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < K_w; ++kw) {
                    const int iw = w_base + kw;
                    if (iw < 0 || iw >= W) continue;
                    const int64_t in_idx  = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
                    const int64_t wt_idx  = ((int64_t)co * C_in + ci) * K_h * K_w + kh * K_w + kw;
                    acc += __half2float(input[in_idx]) * __half2float(weight[wt_idx]);
                }
            }
        }
        output[linear] = __float2half(acc);
    }
}

__global__ void conv2d_forward_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ input,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ bias,
        __nv_bfloat16* __restrict__ output,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int ow = static_cast<int>(tmp % W_out); tmp /= W_out;
        const int oh = static_cast<int>(tmp % H_out); tmp /= H_out;
        const int co = static_cast<int>(tmp % C_out); tmp /= C_out;
        const int n  = static_cast<int>(tmp);

        float acc = __bfloat162float(bias[co]);
        const int h_base = oh * stride_h - pad_h;
        const int w_base = ow * stride_w - pad_w;

        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int ih = h_base + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < K_w; ++kw) {
                    const int iw = w_base + kw;
                    if (iw < 0 || iw >= W) continue;
                    const int64_t in_idx  = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
                    const int64_t wt_idx  = ((int64_t)co * C_in + ci) * K_h * K_w + kh * K_w + kw;
                    acc += __bfloat162float(input[in_idx]) * __bfloat162float(weight[wt_idx]);
                }
            }
        }
        output[linear] = __float2bfloat16(acc);
    }
}

Tensor cuda_conv2d_forward_naive(const Tensor& input,
                                 const Tensor& weight,
                                 const Tensor& bias,
                                 int64_t stride_h, int64_t stride_w,
                                 int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>((H + 2 * pad_h - K_h) / stride_h + 1);
    const int W_out = static_cast<int>((W + 2 * pad_w - K_w) / stride_w + 1);

    Tensor out = make_conv2d_output(input, weight, stride_h, stride_w, pad_h, pad_w);
    const int64_t total = out.numel();

    if (input.dtype() == DType::Float16) {
        conv2d_forward_kernel_half<<<cuda_blocks(total), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(weight.data()),
            static_cast<const __half*>(bias.data()),
            static_cast<__half*>(out.data()),
            total, N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else if (input.dtype() == DType::BFloat16) {
        conv2d_forward_kernel_bfloat16<<<cuda_blocks(total), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(weight.data()),
            static_cast<const __nv_bfloat16*>(bias.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            total, N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else {
        conv2d_forward_kernel<<<cuda_blocks(total), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(out.data()),
            total, N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

__global__ void conv2d_forward_tiled_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    extern __shared__ float filter[];

    const int co = static_cast<int>(blockIdx.z % C_out);
    const int n = static_cast<int>(blockIdx.z / C_out);
    const int filter_size = C_in * K_h * K_w;
    const int tid = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads) {
        filter[idx] = weight[(int64_t)co * filter_size + idx];
    }
    __syncthreads();

    const int oh = static_cast<int>(blockIdx.y) * kTileH + static_cast<int>(threadIdx.y);
    const int ow = static_cast<int>(blockIdx.x) * kTileW + static_cast<int>(threadIdx.x);
    if (n >= N || co >= C_out || oh >= H_out || ow >= W_out) {
        return;
    }

    float acc = bias[co];
    const int h_base = oh * stride_h - pad_h;
    const int w_base = ow * stride_w - pad_w;

    for (int ci = 0; ci < C_in; ++ci) {
        const int64_t input_c = ((int64_t)n * C_in + ci) * H * W;
        const int filter_c = ci * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int ih = h_base + kh;
            if (ih < 0 || ih >= H) {
                continue;
            }
            const int64_t input_row = input_c + (int64_t)ih * W;
            const int filter_row = filter_c + kh * K_w;
            for (int kw = 0; kw < K_w; ++kw) {
                const int iw = w_base + kw;
                if (iw < 0 || iw >= W) {
                    continue;
                }
                acc += input[input_row + iw] * filter[filter_row + kw];
            }
        }
    }

    const int64_t out_idx = ((int64_t)n * C_out + co) * H_out * W_out
                          + (int64_t)oh * W_out + ow;
    output[out_idx] = acc;
}

__global__ void conv2d_forward_tiled_kernel_half(
        const __half* __restrict__ input,
        const __half* __restrict__ weight,
        const __half* __restrict__ bias,
        __half* __restrict__ output,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    extern __shared__ float filter[];  // float smem for FP32 accumulation

    const int co = static_cast<int>(blockIdx.z % C_out);
    const int n  = static_cast<int>(blockIdx.z / C_out);
    const int filter_size = C_in * K_h * K_w;
    const int tid     = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads)
        filter[idx] = __half2float(weight[(int64_t)co * filter_size + idx]);
    __syncthreads();

    const int oh = static_cast<int>(blockIdx.y) * kTileH + static_cast<int>(threadIdx.y);
    const int ow = static_cast<int>(blockIdx.x) * kTileW + static_cast<int>(threadIdx.x);
    if (n >= N || co >= C_out || oh >= H_out || ow >= W_out) return;

    float acc = __half2float(bias[co]);
    const int h_base = oh * stride_h - pad_h;
    const int w_base = ow * stride_w - pad_w;

    for (int ci = 0; ci < C_in; ++ci) {
        const int64_t input_c  = ((int64_t)n * C_in + ci) * H * W;
        const int     filter_c = ci * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int ih = h_base + kh;
            if (ih < 0 || ih >= H) continue;
            const int64_t input_row  = input_c + (int64_t)ih * W;
            const int     filter_row = filter_c + kh * K_w;
            for (int kw = 0; kw < K_w; ++kw) {
                const int iw = w_base + kw;
                if (iw < 0 || iw >= W) continue;
                acc += __half2float(input[input_row + iw]) * filter[filter_row + kw];
            }
        }
    }
    const int64_t out_idx = ((int64_t)n * C_out + co) * H_out * W_out
                          + (int64_t)oh * W_out + ow;
    output[out_idx] = __float2half(acc);
}

__global__ void conv2d_forward_tiled_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ input,
        const __nv_bfloat16* __restrict__ weight,
        const __nv_bfloat16* __restrict__ bias,
        __nv_bfloat16* __restrict__ output,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    extern __shared__ float filter[];  // float smem for FP32 accumulation

    const int co = static_cast<int>(blockIdx.z % C_out);
    const int n  = static_cast<int>(blockIdx.z / C_out);
    const int filter_size = C_in * K_h * K_w;
    const int tid     = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads)
        filter[idx] = __bfloat162float(weight[(int64_t)co * filter_size + idx]);
    __syncthreads();

    const int oh = static_cast<int>(blockIdx.y) * kTileH + static_cast<int>(threadIdx.y);
    const int ow = static_cast<int>(blockIdx.x) * kTileW + static_cast<int>(threadIdx.x);
    if (n >= N || co >= C_out || oh >= H_out || ow >= W_out) return;

    float acc = __bfloat162float(bias[co]);
    const int h_base = oh * stride_h - pad_h;
    const int w_base = ow * stride_w - pad_w;

    for (int ci = 0; ci < C_in; ++ci) {
        const int64_t input_c  = ((int64_t)n * C_in + ci) * H * W;
        const int     filter_c = ci * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int ih = h_base + kh;
            if (ih < 0 || ih >= H) continue;
            const int64_t input_row  = input_c + (int64_t)ih * W;
            const int     filter_row = filter_c + kh * K_w;
            for (int kw = 0; kw < K_w; ++kw) {
                const int iw = w_base + kw;
                if (iw < 0 || iw >= W) continue;
                acc += __bfloat162float(input[input_row + iw]) * filter[filter_row + kw];
            }
        }
    }
    const int64_t out_idx = ((int64_t)n * C_out + co) * H_out * W_out
                          + (int64_t)oh * W_out + ow;
    output[out_idx] = __float2bfloat16(acc);
}

Tensor cuda_conv2d_forward_optimized(const Tensor& input,
                                     const Tensor& weight,
                                     const Tensor& bias,
                                     int64_t stride_h, int64_t stride_w,
                                     int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>((H + 2 * pad_h - K_h) / stride_h + 1);
    const int W_out = static_cast<int>((W + 2 * pad_w - K_w) / stride_w + 1);
    const int filter_size = C_in * K_h * K_w;

    if (filter_size < kMinSharedFilterFloats || filter_size > kMaxSharedFilterFloats) {
        return cuda_conv2d_forward_naive(input, weight, bias, stride_h, stride_w, pad_h, pad_w);
    }

    Tensor out = make_conv2d_output(input, weight, stride_h, stride_w, pad_h, pad_w);

    dim3 block(kTileW, kTileH);
    dim3 grid(static_cast<unsigned>((W_out + kTileW - 1) / kTileW),
              static_cast<unsigned>((H_out + kTileH - 1) / kTileH),
              static_cast<unsigned>(N * C_out));
    const size_t shared_bytes = static_cast<size_t>(filter_size) * sizeof(float);

    if (input.dtype() == DType::Float16) {
        conv2d_forward_tiled_kernel_half<<<grid, block, shared_bytes>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(weight.data()),
            static_cast<const __half*>(bias.data()),
            static_cast<__half*>(out.data()),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else if (input.dtype() == DType::BFloat16) {
        conv2d_forward_tiled_kernel_bfloat16<<<grid, block, shared_bytes>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(weight.data()),
            static_cast<const __nv_bfloat16*>(bias.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else {
        conv2d_forward_tiled_kernel<<<grid, block, shared_bytes>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(out.data()),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor cuda_conv2d_forward(const Tensor& input,
                           const Tensor& weight,
                           const Tensor& bias,
                           int64_t stride_h, int64_t stride_w,
                           int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const char* algo = std::getenv("ORTHON_CUDA_CONV2D_FORWARD");
    if (algo != nullptr && std::strcmp(algo, "naive") == 0) {
        return cuda_conv2d_forward_naive(input, weight, bias, stride_h, stride_w, pad_h, pad_w);
    }
    return cuda_conv2d_forward_optimized(input, weight, bias, stride_h, stride_w, pad_h, pad_w);
}

__global__ void conv2d_grad_input_kernel(
        const float* __restrict__ grad_output,
        const float* __restrict__ weight,
        float* __restrict__ grad_input,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
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
        const int ci = static_cast<int>(tmp % C_in);
        tmp /= C_in;
        const int n = static_cast<int>(tmp);

        float acc = 0.0f;
        for (int co = 0; co < C_out; ++co) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int oh_num = ih + pad_h - kh;
                if (oh_num % stride_h != 0) {
                    continue;
                }
                const int oh = oh_num / stride_h;
                if (oh < 0 || oh >= H_out) {
                    continue;
                }

                for (int kw = 0; kw < K_w; ++kw) {
                    const int ow_num = iw + pad_w - kw;
                    if (ow_num % stride_w != 0) {
                        continue;
                    }
                    const int ow = ow_num / stride_w;
                    if (ow < 0 || ow >= W_out) {
                        continue;
                    }

                    const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                                         + oh * W_out + ow;
                    const int64_t w_idx = ((int64_t)co * C_in + ci) * K_h * K_w
                                        + kh * K_w + kw;
                    acc += grad_output[go_idx] * weight[w_idx];
                }
            }
        }
        grad_input[linear] = acc;
    }
}

__global__ void conv2d_grad_weight_kernel(
        const float* __restrict__ input,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_weight,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int kw = static_cast<int>(tmp % K_w);
        tmp /= K_w;
        const int kh = static_cast<int>(tmp % K_h);
        tmp /= K_h;
        const int ci = static_cast<int>(tmp % C_in);
        tmp /= C_in;
        const int co = static_cast<int>(tmp);

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < H_out; ++oh) {
                const int ih = oh * stride_h + kh - pad_h;
                if (ih < 0 || ih >= H) {
                    continue;
                }
                for (int ow = 0; ow < W_out; ++ow) {
                    const int iw = ow * stride_w + kw - pad_w;
                    if (iw < 0 || iw >= W) {
                        continue;
                    }
                    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
                    const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                                         + oh * W_out + ow;
                    acc += input[in_idx] * grad_output[go_idx];
                }
            }
        }
        grad_weight[linear] = acc;
    }
}

__global__ void conv2d_grad_bias_kernel(
        const float* __restrict__ grad_output,
        float* __restrict__ grad_bias,
        int64_t total,
        int N, int C_out, int H_out, int W_out) {
    for (int64_t co = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         co < total;
         co += (int64_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* row = grad_output + ((int64_t)n * C_out + co) * H_out * W_out;
            for (int i = 0; i < H_out * W_out; ++i) {
                acc += row[i];
            }
        }
        grad_bias[co] = acc;
    }
}

__global__ void conv2d_grad_input_kernel_half(
        const __half* __restrict__ grad_output,
        const __half* __restrict__ weight,
        __half* __restrict__ grad_input,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int iw = static_cast<int>(tmp % W); tmp /= W;
        const int ih = static_cast<int>(tmp % H); tmp /= H;
        const int ci = static_cast<int>(tmp % C_in); tmp /= C_in;
        const int n  = static_cast<int>(tmp);

        float acc = 0.0f;
        for (int co = 0; co < C_out; ++co) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int oh_num = ih + pad_h - kh;
                if (oh_num % stride_h != 0) continue;
                const int oh = oh_num / stride_h;
                if (oh < 0 || oh >= H_out) continue;
                for (int kw = 0; kw < K_w; ++kw) {
                    const int ow_num = iw + pad_w - kw;
                    if (ow_num % stride_w != 0) continue;
                    const int ow = ow_num / stride_w;
                    if (ow < 0 || ow >= W_out) continue;
                    const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
                    const int64_t w_idx  = ((int64_t)co * C_in + ci) * K_h * K_w + kh * K_w + kw;
                    acc += __half2float(grad_output[go_idx]) * __half2float(weight[w_idx]);
                }
            }
        }
        grad_input[linear] = __float2half(acc);
    }
}

__global__ void conv2d_grad_input_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ grad_output,
        const __nv_bfloat16* __restrict__ weight,
        __nv_bfloat16* __restrict__ grad_input,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int iw = static_cast<int>(tmp % W); tmp /= W;
        const int ih = static_cast<int>(tmp % H); tmp /= H;
        const int ci = static_cast<int>(tmp % C_in); tmp /= C_in;
        const int n  = static_cast<int>(tmp);

        float acc = 0.0f;
        for (int co = 0; co < C_out; ++co) {
            for (int kh = 0; kh < K_h; ++kh) {
                const int oh_num = ih + pad_h - kh;
                if (oh_num % stride_h != 0) continue;
                const int oh = oh_num / stride_h;
                if (oh < 0 || oh >= H_out) continue;
                for (int kw = 0; kw < K_w; ++kw) {
                    const int ow_num = iw + pad_w - kw;
                    if (ow_num % stride_w != 0) continue;
                    const int ow = ow_num / stride_w;
                    if (ow < 0 || ow >= W_out) continue;
                    const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
                    const int64_t w_idx  = ((int64_t)co * C_in + ci) * K_h * K_w + kh * K_w + kw;
                    acc += __bfloat162float(grad_output[go_idx]) * __bfloat162float(weight[w_idx]);
                }
            }
        }
        grad_input[linear] = __float2bfloat16(acc);
    }
}

Tensor cuda_conv2d_backward_input_naive(const Tensor& input,
                                        const Tensor& weight,
                                        const Tensor& grad_output,
                                        int64_t stride_h, int64_t stride_w,
                                        int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_input(std::make_shared<TensorImpl>(
        input.shape(), input.dtype(), input.device()));
    if (input.dtype() == DType::Float16) {
        conv2d_grad_input_kernel_half<<<cuda_blocks(grad_input.numel()), kThreads>>>(
            static_cast<const __half*>(grad_output.data()),
            static_cast<const __half*>(weight.data()),
            static_cast<__half*>(grad_input.data()),
            grad_input.numel(),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else if (input.dtype() == DType::BFloat16) {
        conv2d_grad_input_kernel_bfloat16<<<cuda_blocks(grad_input.numel()), kThreads>>>(
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<const __nv_bfloat16*>(weight.data()),
            static_cast<__nv_bfloat16*>(grad_input.data()),
            grad_input.numel(),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else {
        conv2d_grad_input_kernel<<<cuda_blocks(grad_input.numel()), kThreads>>>(
            static_cast<const float*>(grad_output.data()),
            static_cast<const float*>(weight.data()),
            static_cast<float*>(grad_input.data()),
            grad_input.numel(),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

__global__ void conv2d_grad_input_tiled_kernel(
        const float* __restrict__ grad_output,
        const float* __restrict__ weight,
        float* __restrict__ grad_input,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    extern __shared__ float filter[];

    const int ci = static_cast<int>(blockIdx.z % C_in);
    const int n = static_cast<int>(blockIdx.z / C_in);
    const int filter_size = C_out * K_h * K_w;
    const int tid = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads) {
        const int kw_s = idx % K_w;
        int ftmp = idx / K_w;
        const int kh_s = ftmp % K_h;
        const int co_s = ftmp / K_h;
        filter[idx] = weight[((int64_t)co_s * C_in + ci) * K_h * K_w + kh_s * K_w + kw_s];
    }
    __syncthreads();

    const int ih = static_cast<int>(blockIdx.y) * kTileH + static_cast<int>(threadIdx.y);
    const int iw = static_cast<int>(blockIdx.x) * kTileW + static_cast<int>(threadIdx.x);
    if (n >= N || ci >= C_in || ih >= H || iw >= W) {
        return;
    }

    float acc = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        const int filter_co = co * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int oh_num = ih + pad_h - kh;
            if (oh_num % stride_h != 0) {
                continue;
            }
            const int oh = oh_num / stride_h;
            if (oh < 0 || oh >= H_out) {
                continue;
            }
            for (int kw = 0; kw < K_w; ++kw) {
                const int ow_num = iw + pad_w - kw;
                if (ow_num % stride_w != 0) {
                    continue;
                }
                const int ow = ow_num / stride_w;
                if (ow < 0 || ow >= W_out) {
                    continue;
                }

                const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                                     + oh * W_out + ow;
                acc += grad_output[go_idx] * filter[filter_co + kh * K_w + kw];
            }
        }
    }

    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
    grad_input[in_idx] = acc;
}

__global__ void conv2d_grad_input_tiled_kernel_half(
        const __half* __restrict__ grad_output,
        const __half* __restrict__ weight,
        __half* __restrict__ grad_input,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    extern __shared__ float filter[];  // float smem for FP32 accumulation

    const int ci = static_cast<int>(blockIdx.z % C_in);
    const int n  = static_cast<int>(blockIdx.z / C_in);
    const int filter_size = C_out * K_h * K_w;
    const int tid     = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads) {
        const int kw_s = idx % K_w;
        int ftmp  = idx / K_w;
        const int kh_s = ftmp % K_h;
        const int co_s = ftmp / K_h;
        filter[idx] = __half2float(weight[((int64_t)co_s * C_in + ci) * K_h * K_w + kh_s * K_w + kw_s]);
    }
    __syncthreads();

    const int ih = static_cast<int>(blockIdx.y) * kTileH + static_cast<int>(threadIdx.y);
    const int iw = static_cast<int>(blockIdx.x) * kTileW + static_cast<int>(threadIdx.x);
    if (n >= N || ci >= C_in || ih >= H || iw >= W) return;

    float acc = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        const int filter_co = co * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int oh_num = ih + pad_h - kh;
            if (oh_num % stride_h != 0) continue;
            const int oh = oh_num / stride_h;
            if (oh < 0 || oh >= H_out) continue;
            for (int kw = 0; kw < K_w; ++kw) {
                const int ow_num = iw + pad_w - kw;
                if (ow_num % stride_w != 0) continue;
                const int ow = ow_num / stride_w;
                if (ow < 0 || ow >= W_out) continue;
                const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
                acc += __half2float(grad_output[go_idx]) * filter[filter_co + kh * K_w + kw];
            }
        }
    }
    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
    grad_input[in_idx] = __float2half(acc);
}

__global__ void conv2d_grad_input_tiled_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ grad_output,
        const __nv_bfloat16* __restrict__ weight,
        __nv_bfloat16* __restrict__ grad_input,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    extern __shared__ float filter[];  // float smem for FP32 accumulation

    const int ci = static_cast<int>(blockIdx.z % C_in);
    const int n  = static_cast<int>(blockIdx.z / C_in);
    const int filter_size = C_out * K_h * K_w;
    const int tid     = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads) {
        const int kw_s = idx % K_w;
        int ftmp  = idx / K_w;
        const int kh_s = ftmp % K_h;
        const int co_s = ftmp / K_h;
        filter[idx] = __bfloat162float(weight[((int64_t)co_s * C_in + ci) * K_h * K_w + kh_s * K_w + kw_s]);
    }
    __syncthreads();

    const int ih = static_cast<int>(blockIdx.y) * kTileH + static_cast<int>(threadIdx.y);
    const int iw = static_cast<int>(blockIdx.x) * kTileW + static_cast<int>(threadIdx.x);
    if (n >= N || ci >= C_in || ih >= H || iw >= W) return;

    float acc = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        const int filter_co = co * K_h * K_w;
        for (int kh = 0; kh < K_h; ++kh) {
            const int oh_num = ih + pad_h - kh;
            if (oh_num % stride_h != 0) continue;
            const int oh = oh_num / stride_h;
            if (oh < 0 || oh >= H_out) continue;
            for (int kw = 0; kw < K_w; ++kw) {
                const int ow_num = iw + pad_w - kw;
                if (ow_num % stride_w != 0) continue;
                const int ow = ow_num / stride_w;
                if (ow < 0 || ow >= W_out) continue;
                const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
                acc += __bfloat162float(grad_output[go_idx]) * filter[filter_co + kh * K_w + kw];
            }
        }
    }
    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
    grad_input[in_idx] = __float2bfloat16(acc);
}

Tensor cuda_conv2d_backward_input_optimized(const Tensor& input,
                                            const Tensor& weight,
                                            const Tensor& grad_output,
                                            int64_t stride_h, int64_t stride_w,
                                            int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);
    const int filter_size = C_out * K_h * K_w;

    if (filter_size < kMinSharedFilterFloats || filter_size > kMaxSharedFilterFloats) {
        return cuda_conv2d_backward_input_naive(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }

    Tensor grad_input(std::make_shared<TensorImpl>(
        input.shape(), input.dtype(), input.device()));

    dim3 block(kTileW, kTileH);
    dim3 grid(static_cast<unsigned>((W + kTileW - 1) / kTileW),
              static_cast<unsigned>((H + kTileH - 1) / kTileH),
              static_cast<unsigned>(N * C_in));
    const size_t shared_bytes = static_cast<size_t>(filter_size) * sizeof(float);

    if (input.dtype() == DType::Float16) {
        conv2d_grad_input_tiled_kernel_half<<<grid, block, shared_bytes>>>(
            static_cast<const __half*>(grad_output.data()),
            static_cast<const __half*>(weight.data()),
            static_cast<__half*>(grad_input.data()),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else if (input.dtype() == DType::BFloat16) {
        conv2d_grad_input_tiled_kernel_bfloat16<<<grid, block, shared_bytes>>>(
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<const __nv_bfloat16*>(weight.data()),
            static_cast<__nv_bfloat16*>(grad_input.data()),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else {
        conv2d_grad_input_tiled_kernel<<<grid, block, shared_bytes>>>(
            static_cast<const float*>(grad_output.data()),
            static_cast<const float*>(weight.data()),
            static_cast<float*>(grad_input.data()),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

__global__ void conv2d_grad_input_3x3_pad1_kernel(
        const float* __restrict__ grad_output,
        const float* __restrict__ weight,
        float* __restrict__ grad_input,
        int N, int C_in, int H, int W,
        int C_out,
        int H_out, int W_out) {
    extern __shared__ float filter[];

    const int ci = static_cast<int>(blockIdx.z % C_in);
    const int n = static_cast<int>(blockIdx.z / C_in);
    const int filter_size = C_out * 9;
    const int tid = static_cast<int>(threadIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x * blockDim.y);

    for (int idx = tid; idx < filter_size; idx += threads) {
        const int k = idx % 9;
        const int co = idx / 9;
        filter[idx] = weight[((int64_t)co * C_in + ci) * 9 + k];
    }
    __syncthreads();

    const int ih = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y)
                 + static_cast<int>(threadIdx.y);
    const int iw = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                 + static_cast<int>(threadIdx.x);
    if (n >= N || ci >= C_in || ih >= H || iw >= W) {
        return;
    }

    float acc = 0.0f;
    for (int co = 0; co < C_out; ++co) {
        const float* go_base = grad_output + ((int64_t)n * C_out + co) * H_out * W_out;
        const float* f = filter + co * 9;

        if (ih > 0) {
            const float* go_row = go_base + (int64_t)(ih - 1) * W_out;
            if (iw > 0) {
                acc += go_row[iw - 1] * f[8];
            }
            acc += go_row[iw] * f[7];
            if (iw + 1 < W_out) {
                acc += go_row[iw + 1] * f[6];
            }
        }

        {
            const float* go_row = go_base + (int64_t)ih * W_out;
            if (iw > 0) {
                acc += go_row[iw - 1] * f[5];
            }
            acc += go_row[iw] * f[4];
            if (iw + 1 < W_out) {
                acc += go_row[iw + 1] * f[3];
            }
        }

        if (ih + 1 < H_out) {
            const float* go_row = go_base + (int64_t)(ih + 1) * W_out;
            if (iw > 0) {
                acc += go_row[iw - 1] * f[2];
            }
            acc += go_row[iw] * f[1];
            if (iw + 1 < W_out) {
                acc += go_row[iw + 1] * f[0];
            }
        }
    }

    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
    grad_input[in_idx] = acc;
}

Tensor cuda_conv2d_backward_input_optimized_v2(const Tensor& input,
                                               const Tensor& weight,
                                               const Tensor& grad_output,
                                               int64_t stride_h, int64_t stride_w,
                                               int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    if (K_h != 3 || K_w != 3 || stride_h != 1 || stride_w != 1 || pad_h != 1 || pad_w != 1 ||
        H_out != H || W_out != W) {
        return cuda_conv2d_backward_input_optimized(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }

    const int filter_size = C_out * 9;
    if (filter_size < kMinSharedFilterFloats || filter_size > kMaxSharedFilterFloats) {
        return cuda_conv2d_backward_input_naive(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }

    Tensor grad_input(std::make_shared<TensorImpl>(
        input.shape(), input.dtype(), input.device()));

    const int tile_w = (W <= 8) ? 8 : kTileW;
    const int tile_h = (H <= 8) ? 8 : kTileH;
    dim3 block(tile_w, tile_h);
    dim3 grid(static_cast<unsigned>((W + tile_w - 1) / tile_w),
              static_cast<unsigned>((H + tile_h - 1) / tile_h),
              static_cast<unsigned>(N * C_in));
    const size_t shared_bytes = static_cast<size_t>(filter_size) * sizeof(float);

    conv2d_grad_input_3x3_pad1_kernel<<<grid, block, shared_bytes>>>(
        static_cast<const float*>(grad_output.data()),
        static_cast<const float*>(weight.data()),
        static_cast<float*>(grad_input.data()),
        N, C_in, H, W,
        C_out,
        H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor cuda_conv2d_backward_input(const Tensor& input,
                                  const Tensor& weight,
                                  const Tensor& grad_output,
                                  int64_t stride_h, int64_t stride_w,
                                  int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const char* algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_INPUT");
    if (algo != nullptr && std::strcmp(algo, "naive") == 0) {
        return cuda_conv2d_backward_input_naive(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }
    const bool optimized_v2_eligible =
        weight.shape()[2] == 3 && weight.shape()[3] == 3 &&
        stride_h == 1 && stride_w == 1 &&
        pad_h == 1 && pad_w == 1 &&
        grad_output.shape()[2] == input.shape()[2] &&
        grad_output.shape()[3] == input.shape()[3] &&
        input.dtype() != DType::Float16 &&
        input.dtype() != DType::BFloat16;  // v2 kernel is float32-only
    if ((algo == nullptr && optimized_v2_eligible) ||
        (algo != nullptr && std::strcmp(algo, "optimized_v2") == 0)) {
        return cuda_conv2d_backward_input_optimized_v2(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }
    return cuda_conv2d_backward_input_optimized(
        input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
}

__global__ void conv2d_grad_weight_kernel_half(
        const __half* __restrict__ input,
        const __half* __restrict__ grad_output,
        __half* __restrict__ grad_weight,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int kw = static_cast<int>(tmp % K_w); tmp /= K_w;
        const int kh = static_cast<int>(tmp % K_h); tmp /= K_h;
        const int ci = static_cast<int>(tmp % C_in); tmp /= C_in;
        const int co = static_cast<int>(tmp);

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < H_out; ++oh) {
                const int ih = oh * stride_h + kh - pad_h;
                if (ih < 0 || ih >= H) continue;
                for (int ow = 0; ow < W_out; ++ow) {
                    const int iw = ow * stride_w + kw - pad_w;
                    if (iw < 0 || iw >= W) continue;
                    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
                    const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
                    acc += __half2float(input[in_idx]) * __half2float(grad_output[go_idx]);
                }
            }
        }
        grad_weight[linear] = __float2half(acc);
    }
}

__global__ void conv2d_grad_weight_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ input,
        const __nv_bfloat16* __restrict__ grad_output,
        __nv_bfloat16* __restrict__ grad_weight,
        int64_t total,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int64_t tmp = linear;
        const int kw = static_cast<int>(tmp % K_w); tmp /= K_w;
        const int kh = static_cast<int>(tmp % K_h); tmp /= K_h;
        const int ci = static_cast<int>(tmp % C_in); tmp /= C_in;
        const int co = static_cast<int>(tmp);

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < H_out; ++oh) {
                const int ih = oh * stride_h + kh - pad_h;
                if (ih < 0 || ih >= H) continue;
                for (int ow = 0; ow < W_out; ++ow) {
                    const int iw = ow * stride_w + kw - pad_w;
                    if (iw < 0 || iw >= W) continue;
                    const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
                    const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
                    acc += __bfloat162float(input[in_idx]) * __bfloat162float(grad_output[go_idx]);
                }
            }
        }
        grad_weight[linear] = __float2bfloat16(acc);
    }
}

Tensor cuda_conv2d_backward_weight_naive(const Tensor& input,
                                         const Tensor& weight,
                                         const Tensor& grad_output,
                                         int64_t stride_h, int64_t stride_w,
                                         int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight.shape(), weight.dtype(), weight.device()));
    if (input.dtype() == DType::Float16) {
        conv2d_grad_weight_kernel_half<<<cuda_blocks(grad_weight.numel()), kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(grad_weight.data()),
            grad_weight.numel(),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else if (input.dtype() == DType::BFloat16) {
        conv2d_grad_weight_kernel_bfloat16<<<cuda_blocks(grad_weight.numel()), kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(grad_weight.data()),
            grad_weight.numel(),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    } else {
        conv2d_grad_weight_kernel<<<cuda_blocks(grad_weight.numel()), kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(grad_weight.data()),
            grad_weight.numel(),
            N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

__global__ void conv2d_grad_weight_reduction_kernel(
        const float* __restrict__ input,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_weight,
        int total_weights,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    const int linear = static_cast<int>(blockIdx.x);
    if (linear >= total_weights) {
        return;
    }

    int tmp = linear;
    const int kw = tmp % K_w;
    tmp /= K_w;
    const int kh = tmp % K_h;
    tmp /= K_h;
    const int ci = tmp % C_in;
    tmp /= C_in;
    const int co = tmp;

    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int ow = pos_tmp % W_out;
        pos_tmp /= W_out;
        const int oh = pos_tmp % H_out;
        pos_tmp /= H_out;
        const int n = pos_tmp;

        const int ih = oh * stride_h + kh - pad_h;
        const int iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
            continue;
        }

        const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                             + oh * W_out + ow;
        acc += input[in_idx] * grad_output[go_idx];
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_weight[linear] = acc;
    }
}

__global__ void conv2d_grad_weight_3x3_kernel(
        const float* __restrict__ input,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_weight,
        int N, int C_in, int H, int W,
        int C_out,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    const int pair = static_cast<int>(blockIdx.x);
    const int ci = pair % C_in;
    const int co = pair / C_in;
    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int ow = pos_tmp % W_out;
        pos_tmp /= W_out;
        const int oh = pos_tmp % H_out;
        pos_tmp /= H_out;
        const int n = pos_tmp;

        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                             + oh * W_out + ow;
        const float go = grad_output[go_idx];
        const int base_h = oh * stride_h - pad_h;
        const int base_w = ow * stride_w - pad_w;
        const int64_t in_base = ((int64_t)n * C_in + ci) * H * W;

        const int ih0 = base_h;
        const int ih1 = base_h + 1;
        const int ih2 = base_h + 2;
        const int iw0 = base_w;
        const int iw1 = base_w + 1;
        const int iw2 = base_w + 2;

        if (ih0 >= 0 && ih0 < H) {
            const int64_t row = in_base + (int64_t)ih0 * W;
            if (iw0 >= 0 && iw0 < W) acc00 += input[row + iw0] * go;
            if (iw1 >= 0 && iw1 < W) acc01 += input[row + iw1] * go;
            if (iw2 >= 0 && iw2 < W) acc02 += input[row + iw2] * go;
        }
        if (ih1 >= 0 && ih1 < H) {
            const int64_t row = in_base + (int64_t)ih1 * W;
            if (iw0 >= 0 && iw0 < W) acc10 += input[row + iw0] * go;
            if (iw1 >= 0 && iw1 < W) acc11 += input[row + iw1] * go;
            if (iw2 >= 0 && iw2 < W) acc12 += input[row + iw2] * go;
        }
        if (ih2 >= 0 && ih2 < H) {
            const int64_t row = in_base + (int64_t)ih2 * W;
            if (iw0 >= 0 && iw0 < W) acc20 += input[row + iw0] * go;
            if (iw1 >= 0 && iw1 < W) acc21 += input[row + iw1] * go;
            if (iw2 >= 0 && iw2 < W) acc22 += input[row + iw2] * go;
        }
    }

    acc00 = block_sum(acc00);
    acc01 = block_sum(acc01);
    acc02 = block_sum(acc02);
    acc10 = block_sum(acc10);
    acc11 = block_sum(acc11);
    acc12 = block_sum(acc12);
    acc20 = block_sum(acc20);
    acc21 = block_sum(acc21);
    acc22 = block_sum(acc22);

    if (tid == 0) {
        float* dst = grad_weight + ((int64_t)co * C_in + ci) * 9;
        dst[0] = acc00;
        dst[1] = acc01;
        dst[2] = acc02;
        dst[3] = acc10;
        dst[4] = acc11;
        dst[5] = acc12;
        dst[6] = acc20;
        dst[7] = acc21;
        dst[8] = acc22;
    }
}

__global__ void conv2d_grad_weight_3x3_chunked_kernel(
        const float* __restrict__ input,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_weight,
        int N, int C_in, int H, int W,
        int C_out,
        int H_out, int W_out,
        int chunk_size) {
    const int pair = static_cast<int>(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    const int ci = pair % C_in;
    const int co = pair / C_in;
    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    const int start = chunk * chunk_size;
    const int end = min(start + chunk_size, positions);

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f;

    for (int pos = start + tid; pos < end; pos += blockDim.x) {
        int pos_tmp = pos;
        const int ow = pos_tmp % W_out;
        pos_tmp /= W_out;
        const int oh = pos_tmp % H_out;
        pos_tmp /= H_out;
        const int n = pos_tmp;

        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                             + oh * W_out + ow;
        const float go = grad_output[go_idx];
        const int base_h = oh - 1;
        const int base_w = ow - 1;
        const int64_t in_base = ((int64_t)n * C_in + ci) * H * W;

        const int ih0 = base_h;
        const int ih1 = base_h + 1;
        const int ih2 = base_h + 2;
        const int iw0 = base_w;
        const int iw1 = base_w + 1;
        const int iw2 = base_w + 2;

        if (ih0 >= 0 && ih0 < H) {
            const int64_t row = in_base + (int64_t)ih0 * W;
            if (iw0 >= 0 && iw0 < W) acc00 += input[row + iw0] * go;
            if (iw1 >= 0 && iw1 < W) acc01 += input[row + iw1] * go;
            if (iw2 >= 0 && iw2 < W) acc02 += input[row + iw2] * go;
        }
        if (ih1 >= 0 && ih1 < H) {
            const int64_t row = in_base + (int64_t)ih1 * W;
            if (iw0 >= 0 && iw0 < W) acc10 += input[row + iw0] * go;
            if (iw1 >= 0 && iw1 < W) acc11 += input[row + iw1] * go;
            if (iw2 >= 0 && iw2 < W) acc12 += input[row + iw2] * go;
        }
        if (ih2 >= 0 && ih2 < H) {
            const int64_t row = in_base + (int64_t)ih2 * W;
            if (iw0 >= 0 && iw0 < W) acc20 += input[row + iw0] * go;
            if (iw1 >= 0 && iw1 < W) acc21 += input[row + iw1] * go;
            if (iw2 >= 0 && iw2 < W) acc22 += input[row + iw2] * go;
        }
    }

    acc00 = block_sum(acc00);
    acc01 = block_sum(acc01);
    acc02 = block_sum(acc02);
    acc10 = block_sum(acc10);
    acc11 = block_sum(acc11);
    acc12 = block_sum(acc12);
    acc20 = block_sum(acc20);
    acc21 = block_sum(acc21);
    acc22 = block_sum(acc22);

    if (tid == 0) {
        float* dst = grad_weight + ((int64_t)co * C_in + ci) * 9;
        atomicAdd(dst + 0, acc00);
        atomicAdd(dst + 1, acc01);
        atomicAdd(dst + 2, acc02);
        atomicAdd(dst + 3, acc10);
        atomicAdd(dst + 4, acc11);
        atomicAdd(dst + 5, acc12);
        atomicAdd(dst + 6, acc20);
        atomicAdd(dst + 7, acc21);
        atomicAdd(dst + 8, acc22);
    }
}

__global__ void conv2d_grad_weight_reduction_kernel_half(
        const __half* __restrict__ input,
        const __half* __restrict__ grad_output,
        __half* __restrict__ grad_weight,
        int total_weights,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    const int linear = static_cast<int>(blockIdx.x);
    if (linear >= total_weights) {
        return;
    }

    int tmp = linear;
    const int kw = tmp % K_w;
    tmp /= K_w;
    const int kh = tmp % K_h;
    tmp /= K_h;
    const int ci = tmp % C_in;
    tmp /= C_in;
    const int co = tmp;

    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int ow = pos_tmp % W_out;
        pos_tmp /= W_out;
        const int oh = pos_tmp % H_out;
        pos_tmp /= H_out;
        const int n = pos_tmp;

        const int ih = oh * stride_h + kh - pad_h;
        const int iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
            continue;
        }

        const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                             + oh * W_out + ow;
        acc += __half2float(input[in_idx]) * __half2float(grad_output[go_idx]);
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_weight[linear] = __float2half(acc);
    }
}

__global__ void conv2d_grad_weight_reduction_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ input,
        const __nv_bfloat16* __restrict__ grad_output,
        __nv_bfloat16* __restrict__ grad_weight,
        int total_weights,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    const int linear = static_cast<int>(blockIdx.x);
    if (linear >= total_weights) return;

    int tmp = linear;
    const int kw = tmp % K_w; tmp /= K_w;
    const int kh = tmp % K_h; tmp /= K_h;
    const int ci = tmp % C_in; tmp /= C_in;
    const int co = tmp;

    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int ow = pos_tmp % W_out; pos_tmp /= W_out;
        const int oh = pos_tmp % H_out; pos_tmp /= H_out;
        const int n = pos_tmp;

        const int ih = oh * stride_h + kh - pad_h;
        const int iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

        const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + oh * W_out + ow;
        acc += __bfloat162float(input[in_idx]) * __bfloat162float(grad_output[go_idx]);
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_weight[linear] = __float2bfloat16(acc);
    }
}

Tensor cuda_conv2d_backward_weight_optimized_v2(const Tensor& input,
                                                const Tensor& weight,
                                                const Tensor& grad_output,
                                                int64_t stride_h, int64_t stride_w,
                                                int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    if (K_h != 3 || K_w != 3 || stride_h != 1 || stride_w != 1 || pad_h != 1 || pad_w != 1) {
        return cuda_conv2d_backward_weight_optimized(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }

    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight.shape(), weight.dtype(), weight.device()));
    CUDA_CHECK(cudaMemset(grad_weight.data(), 0, static_cast<size_t>(grad_weight.numel()) * sizeof(float)));

    const int positions = N * H_out * W_out;
    const int chunk_size = 4096;
    const int chunks = (positions + chunk_size - 1) / chunk_size;
    dim3 grid(static_cast<unsigned>(C_out * C_in), static_cast<unsigned>(chunks));
    conv2d_grad_weight_3x3_chunked_kernel<<<grid, kThreads>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_weight.data()),
        N, C_in, H, W,
        C_out,
        H_out, W_out,
        chunk_size);
    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

Tensor cuda_conv2d_backward_weight_optimized(const Tensor& input,
                                             const Tensor& weight,
                                             const Tensor& grad_output,
                                             int64_t stride_h, int64_t stride_w,
                                             int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    if (input.dtype() == DType::Float16) {
        Tensor grad_weight(std::make_shared<TensorImpl>(
            weight.shape(), weight.dtype(), weight.device()));
        const int total_weights = static_cast<int>(grad_weight.numel());
        conv2d_grad_weight_reduction_kernel_half<<<total_weights, kThreads>>>(
            static_cast<const __half*>(input.data()),
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(grad_weight.data()),
            total_weights, N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
        CUDA_CHECK(cudaGetLastError());
        return grad_weight;
    }
    if (input.dtype() == DType::BFloat16) {
        Tensor grad_weight(std::make_shared<TensorImpl>(
            weight.shape(), weight.dtype(), weight.device()));
        const int total_weights = static_cast<int>(grad_weight.numel());
        conv2d_grad_weight_reduction_kernel_bfloat16<<<total_weights, kThreads>>>(
            static_cast<const __nv_bfloat16*>(input.data()),
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(grad_weight.data()),
            total_weights, N, C_in, H, W, C_out, K_h, K_w, H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
        CUDA_CHECK(cudaGetLastError());
        return grad_weight;
    }
    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight.shape(), weight.dtype(), weight.device()));
    const char* algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_WEIGHT");
    const bool grouped_3x3_eligible =
        K_h == 3 && K_w == 3 &&
        stride_h == 1 && stride_w == 1 &&
        pad_h == 1 && pad_w == 1;
    const bool use_grouped_3x3 =
        grouped_3x3_eligible &&
        (algo == nullptr || std::strcmp(algo, "grouped3x3") == 0);
    const bool use_optimized_v2 =
        algo != nullptr && std::strcmp(algo, "optimized_v2") == 0;
    if (use_optimized_v2) {
        // Atomics-based experimental path. Keep it opt-in because grouped3x3
        // is simpler and slightly faster on the measured MiniVGG workloads.
        return cuda_conv2d_backward_weight_optimized_v2(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }
    if (use_grouped_3x3) {
        conv2d_grad_weight_3x3_kernel<<<C_out * C_in, kThreads>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(grad_weight.data()),
            N, C_in, H, W,
            C_out,
            H_out, W_out,
            static_cast<int>(stride_h), static_cast<int>(stride_w),
            static_cast<int>(pad_h), static_cast<int>(pad_w));
        CUDA_CHECK(cudaGetLastError());
        return grad_weight;
    }
    const int total_weights = static_cast<int>(grad_weight.numel());
    conv2d_grad_weight_reduction_kernel<<<total_weights, kThreads>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_weight.data()),
        total_weights,
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(pad_h), static_cast<int>(pad_w));
    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

Tensor cuda_conv2d_backward_weight(const Tensor& input,
                                   const Tensor& weight,
                                   const Tensor& grad_output,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const char* algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_WEIGHT");
    if (algo != nullptr && std::strcmp(algo, "naive") == 0) {
        return cuda_conv2d_backward_weight_naive(
            input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    }
    return cuda_conv2d_backward_weight_optimized(
        input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
}

__global__ void conv2d_grad_weight_bias_reduction_kernel(
        const float* __restrict__ input,
        const float* __restrict__ grad_output,
        float* __restrict__ grad_weight,
        float* __restrict__ grad_bias,
        int total_weights,
        int N, int C_in, int H, int W,
        int C_out, int K_h, int K_w,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    const int linear = static_cast<int>(blockIdx.x);
    if (linear >= total_weights) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;
    float bias_acc = 0.0f;

    int tmp = linear;
    const int kw = tmp % K_w;
    tmp /= K_w;
    const int kh = tmp % K_h;
    tmp /= K_h;
    const int ci = tmp % C_in;
    tmp /= C_in;
    const int co = tmp;
    const bool owns_bias = (ci == 0 && kh == 0 && kw == 0);

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int ow = pos_tmp % W_out;
        pos_tmp /= W_out;
        const int oh = pos_tmp % H_out;
        pos_tmp /= H_out;
        const int n = pos_tmp;

        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out
                             + oh * W_out + ow;
        const float go = grad_output[go_idx];
        if (owns_bias) {
            bias_acc += go;
        }

        const int ih = oh * stride_h + kh - pad_h;
        const int iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
            continue;
        }

        const int64_t in_idx = ((int64_t)n * C_in + ci) * H * W + ih * W + iw;
        acc += input[in_idx] * go;
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_weight[linear] = acc;
    }

    if (owns_bias) {
        bias_acc = block_sum(bias_acc);
        if (tid == 0) {
            grad_bias[co] = bias_acc;
        }
    }
}

std::vector<Tensor> cuda_conv2d_backward_weight_bias_optimized(
        const Tensor& input,
        const Tensor& weight,
        const Tensor& bias,
        const Tensor& grad_output,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const int N = static_cast<int>(input.shape()[0]);
    const int C_in = static_cast<int>(input.shape()[1]);
    const int H = static_cast<int>(input.shape()[2]);
    const int W = static_cast<int>(input.shape()[3]);
    const int C_out = static_cast<int>(weight.shape()[0]);
    const int K_h = static_cast<int>(weight.shape()[2]);
    const int K_w = static_cast<int>(weight.shape()[3]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight.shape(), weight.dtype(), weight.device()));
    Tensor grad_bias(std::make_shared<TensorImpl>(
        bias.shape(), bias.dtype(), bias.device()));
    const int total_weights = static_cast<int>(grad_weight.numel());
    conv2d_grad_weight_bias_reduction_kernel<<<total_weights, kThreads>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(grad_output.data()),
        static_cast<float*>(grad_weight.data()),
        static_cast<float*>(grad_bias.data()),
        total_weights,
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(pad_h), static_cast<int>(pad_w));
    CUDA_CHECK(cudaGetLastError());
    return {grad_weight, grad_bias};
}

__global__ void conv2d_grad_bias_kernel_half(
        const __half* __restrict__ grad_output,
        __half* __restrict__ grad_bias,
        int64_t total,
        int N, int C_out, int H_out, int W_out) {
    for (int64_t co = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         co < total;
         co += (int64_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const __half* row = grad_output + ((int64_t)n * C_out + co) * H_out * W_out;
            for (int i = 0; i < H_out * W_out; ++i) {
                acc += __half2float(row[i]);
            }
        }
        grad_bias[co] = __float2half(acc);
    }
}

__global__ void conv2d_grad_bias_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ grad_output,
        __nv_bfloat16* __restrict__ grad_bias,
        int64_t total,
        int N, int C_out, int H_out, int W_out) {
    for (int64_t co = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         co < total;
         co += (int64_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const __nv_bfloat16* row = grad_output + ((int64_t)n * C_out + co) * H_out * W_out;
            for (int i = 0; i < H_out * W_out; ++i) {
                acc += __bfloat162float(row[i]);
            }
        }
        grad_bias[co] = __float2bfloat16(acc);
    }
}

Tensor cuda_conv2d_backward_bias_naive(const Tensor& bias,
                                       const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(grad_output.device());

    const int N = static_cast<int>(grad_output.shape()[0]);
    const int C_out = static_cast<int>(grad_output.shape()[1]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_bias(std::make_shared<TensorImpl>(
        bias.shape(), bias.dtype(), bias.device()));
    if (grad_output.dtype() == DType::Float16) {
        conv2d_grad_bias_kernel_half<<<cuda_blocks(grad_bias.numel()), kThreads>>>(
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(grad_bias.data()),
            grad_bias.numel(),
            N, C_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        conv2d_grad_bias_kernel_bfloat16<<<cuda_blocks(grad_bias.numel()), kThreads>>>(
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(grad_bias.data()),
            grad_bias.numel(),
            N, C_out, H_out, W_out);
    } else {
        conv2d_grad_bias_kernel<<<cuda_blocks(grad_bias.numel()), kThreads>>>(
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(grad_bias.data()),
            grad_bias.numel(),
            N, C_out, H_out, W_out);
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_bias;
}

__global__ void conv2d_grad_bias_reduction_kernel_half(
        const __half* __restrict__ grad_output,
        __half* __restrict__ grad_bias,
        int C_out,
        int N, int H_out, int W_out) {
    const int co = static_cast<int>(blockIdx.x);
    if (co >= C_out) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int spatial = pos_tmp % (H_out * W_out);
        pos_tmp /= H_out * W_out;
        const int n = pos_tmp;
        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + spatial;
        acc += __half2float(grad_output[go_idx]);
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_bias[co] = __float2half(acc);
    }
}

__global__ void conv2d_grad_bias_reduction_kernel_bfloat16(
        const __nv_bfloat16* __restrict__ grad_output,
        __nv_bfloat16* __restrict__ grad_bias,
        int C_out,
        int N, int H_out, int W_out) {
    const int co = static_cast<int>(blockIdx.x);
    if (co >= C_out) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int spatial = pos_tmp % (H_out * W_out);
        pos_tmp /= H_out * W_out;
        const int n = pos_tmp;
        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + spatial;
        acc += __bfloat162float(grad_output[go_idx]);
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_bias[co] = __float2bfloat16(acc);
    }
}

__global__ void conv2d_grad_bias_reduction_kernel(
        const float* __restrict__ grad_output,
        float* __restrict__ grad_bias,
        int C_out,
        int N, int H_out, int W_out) {
    const int co = static_cast<int>(blockIdx.x);
    if (co >= C_out) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int positions = N * H_out * W_out;
    float acc = 0.0f;

    for (int pos = tid; pos < positions; pos += blockDim.x) {
        int pos_tmp = pos;
        const int spatial = pos_tmp % (H_out * W_out);
        pos_tmp /= H_out * W_out;
        const int n = pos_tmp;
        const int64_t go_idx = ((int64_t)n * C_out + co) * H_out * W_out + spatial;
        acc += grad_output[go_idx];
    }

    acc = block_sum(acc);
    if (tid == 0) {
        grad_bias[co] = acc;
    }
}

Tensor cuda_conv2d_backward_bias_optimized(const Tensor& bias,
                                           const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(grad_output.device());

    const int N = static_cast<int>(grad_output.shape()[0]);
    const int C_out = static_cast<int>(grad_output.shape()[1]);
    const int H_out = static_cast<int>(grad_output.shape()[2]);
    const int W_out = static_cast<int>(grad_output.shape()[3]);

    Tensor grad_bias(std::make_shared<TensorImpl>(
        bias.shape(), bias.dtype(), bias.device()));
    if (grad_output.dtype() == DType::Float16) {
        conv2d_grad_bias_reduction_kernel_half<<<C_out, kThreads>>>(
            static_cast<const __half*>(grad_output.data()),
            static_cast<__half*>(grad_bias.data()),
            C_out, N, H_out, W_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        conv2d_grad_bias_reduction_kernel_bfloat16<<<C_out, kThreads>>>(
            static_cast<const __nv_bfloat16*>(grad_output.data()),
            static_cast<__nv_bfloat16*>(grad_bias.data()),
            C_out, N, H_out, W_out);
    } else {
        conv2d_grad_bias_reduction_kernel<<<C_out, kThreads>>>(
            static_cast<const float*>(grad_output.data()),
            static_cast<float*>(grad_bias.data()),
            C_out, N, H_out, W_out);
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_bias;
}

Tensor cuda_conv2d_backward_bias(const Tensor& bias,
                                 const Tensor& grad_output) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(grad_output.device());

    const char* algo = std::getenv("ORTHON_CUDA_CONV2D_BIAS_GRAD");
    if (algo == nullptr) {
        algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_BIAS");
    }
    if (algo != nullptr && std::strcmp(algo, "naive") == 0) {
        return cuda_conv2d_backward_bias_naive(bias, grad_output);
    }
    return cuda_conv2d_backward_bias_optimized(bias, grad_output);
}

std::vector<Tensor> cuda_conv2d_backward(const Tensor& input,
                                         const Tensor& weight,
                                         const Tensor& bias,
                                         const Tensor& grad_output,
                                         int64_t stride_h, int64_t stride_w,
                                         int64_t pad_h, int64_t pad_w) {
    // Run all CUDA work on this op's device; restore the caller's device on exit.
    DeviceGuard guard(input.device());

    const bool profiling = conv2d_profile_enabled();
    const char* grad_weight_algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_WEIGHT");
    const char* grad_bias_algo = std::getenv("ORTHON_CUDA_CONV2D_BIAS_GRAD");
    const char* grad_input_algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_INPUT");
    if (grad_bias_algo == nullptr) {
        grad_bias_algo = std::getenv("ORTHON_CUDA_CONV2D_GRAD_BIAS");
    }
    const std::string input_path =
        (grad_input_algo != nullptr && std::strcmp(grad_input_algo, "naive") == 0)
            ? "naive"
            : (
                (grad_input_algo != nullptr && std::strcmp(grad_input_algo, "optimized_v2") == 0) ||
                (grad_input_algo == nullptr &&
                 weight.shape()[2] == 3 && weight.shape()[3] == 3 &&
                 stride_h == 1 && stride_w == 1 &&
                 pad_h == 1 && pad_w == 1 &&
                 grad_output.shape()[2] == input.shape()[2] &&
                 grad_output.shape()[3] == input.shape()[3])
                    ? "optimized_v2"
                    : "optimized");
    const bool weight_naive = grad_weight_algo != nullptr &&
        std::strcmp(grad_weight_algo, "naive") == 0;
    const bool weight_grouped3x3 = grad_weight_algo != nullptr &&
        std::strcmp(grad_weight_algo, "grouped3x3") == 0;
    const bool weight_optimized_v2 = grad_weight_algo != nullptr &&
        std::strcmp(grad_weight_algo, "optimized_v2") == 0;
    const bool weight_default_grouped3x3 =
        grad_weight_algo == nullptr &&
        weight.shape()[2] == 3 && weight.shape()[3] == 3 &&
        stride_h == 1 && stride_w == 1 &&
        pad_h == 1 && pad_w == 1;
    const bool bias_naive = grad_bias_algo != nullptr &&
        std::strcmp(grad_bias_algo, "naive") == 0;
    const bool bias_separate = grad_bias_algo != nullptr &&
        (std::strcmp(grad_bias_algo, "separate") == 0 ||
         std::strcmp(grad_bias_algo, "optimized") == 0);
    const bool bias_fused = grad_bias_algo != nullptr &&
         std::strcmp(grad_bias_algo, "fused") == 0;
    const bool prefer_fused_bias =
        bias_fused || (grad_bias_algo == nullptr && input.shape()[1] * weight.shape()[0] < 256);
    const bool use_fused_bias =
        !weight_naive && !bias_naive && !bias_separate && prefer_fused_bias &&
        input.dtype() != DType::Float16 &&
        input.dtype() != DType::BFloat16;
    const std::string weight_path = use_fused_bias
        ? "fused_weight_bias"
        : (weight_naive ? "naive" : (
            weight_optimized_v2 ? "optimized_v2" : (
                (weight_grouped3x3 || weight_default_grouped3x3)
                    ? "grouped3x3"
                    : "optimized")));
    const std::string bias_path = use_fused_bias
        ? "fused_weight_bias"
        : (bias_naive ? "naive" : "optimized");
    const auto total_start = profiling
        ? conv2d_profile_now()
        : std::chrono::high_resolution_clock::now();

    double grad_input_seconds = 0.0;
    double grad_weight_seconds = 0.0;
    double grad_bias_seconds = 0.0;

    auto component_start = profiling
        ? conv2d_profile_now()
        : std::chrono::high_resolution_clock::now();
    Tensor grad_input = cuda_conv2d_backward_input(
        input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    if (profiling) {
        grad_input_seconds = conv2d_profile_elapsed(component_start);
    }

    if (use_fused_bias) {
        component_start = profiling
            ? conv2d_profile_now()
            : std::chrono::high_resolution_clock::now();
        auto grads = cuda_conv2d_backward_weight_bias_optimized(
            input, weight, bias, grad_output, stride_h, stride_w, pad_h, pad_w);
        if (profiling) {
            grad_weight_seconds = conv2d_profile_elapsed(component_start);
            const double total_seconds = conv2d_profile_elapsed(total_start);
            const double overhead_seconds = total_seconds
                - grad_input_seconds - grad_weight_seconds - grad_bias_seconds;
            conv2d_profile_record(
                conv2d_profile_key(
                    input, weight, grad_output, stride_h, stride_w, pad_h, pad_w,
                    input_path, weight_path, bias_path),
                grad_input_seconds,
                grad_weight_seconds,
                grad_bias_seconds,
                overhead_seconds > 0.0 ? overhead_seconds : 0.0,
                total_seconds);
        }
        return {grad_input, grads[0], grads[1]};
    }

    component_start = profiling
        ? conv2d_profile_now()
        : std::chrono::high_resolution_clock::now();
    Tensor grad_weight = cuda_conv2d_backward_weight(
        input, weight, grad_output, stride_h, stride_w, pad_h, pad_w);
    if (profiling) {
        grad_weight_seconds = conv2d_profile_elapsed(component_start);
    }

    component_start = profiling
        ? conv2d_profile_now()
        : std::chrono::high_resolution_clock::now();
    Tensor grad_bias = cuda_conv2d_backward_bias(bias, grad_output);
    if (profiling) {
        grad_bias_seconds = conv2d_profile_elapsed(component_start);
        const double total_seconds = conv2d_profile_elapsed(total_start);
        const double overhead_seconds = total_seconds
            - grad_input_seconds - grad_weight_seconds - grad_bias_seconds;
        conv2d_profile_record(
            conv2d_profile_key(
                input, weight, grad_output, stride_h, stride_w, pad_h, pad_w,
                input_path, weight_path, bias_path),
            grad_input_seconds,
            grad_weight_seconds,
            grad_bias_seconds,
            overhead_seconds > 0.0 ? overhead_seconds : 0.0,
            total_seconds);
    }
    return {grad_input, grad_weight, grad_bias};
}

}  // namespace orthon
