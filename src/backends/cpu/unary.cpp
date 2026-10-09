#ifdef __AVX2__
#include <immintrin.h>
#endif
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
// Element-wise transform: typed, handles both contiguous and strided layouts.
template<typename T, typename Op>
static Tensor elementwise_typed(const Tensor& t, Op op) {
    const T* src  = static_cast<const T*>(t.data());
    const int64_t n = t.numel();
    Tensor out    = empty(t.shape(), t.dtype(), t.device());
    T* dst        = static_cast<T*>(out.data());

    if (t.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) dst[i] = op(src[i]);
        return out;
    }

    const auto& sh   = t.shape();
    const auto& st   = t.strides();
    const int   ndim = static_cast<int>(sh.size());
    std::vector<int64_t> idx(ndim, 0);
    for (int64_t elem = 0; elem < n; ++elem) {
        int64_t off = 0;
        for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
        dst[elem] = op(src[off]);
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}


#ifdef __AVX2__
static Tensor neg_contiguous_avx2_float(const Tensor& t) {
    const auto* src = static_cast<const float*>(t.data());
    Tensor out = empty(t.shape(), t.dtype(), t.device());
    auto* dst = static_cast<float*>(out.data());
    const int64_t n = t.numel();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_xor_ps(v, sign));
    }
    for (; i < n; ++i) dst[i] = -src[i];
    return out;
}

static Tensor neg_contiguous_avx2_double(const Tensor& t) {
    const auto* src = static_cast<const double*>(t.data());
    Tensor out = empty(t.shape(), t.dtype(), t.device());
    auto* dst = static_cast<double*>(out.data());
    const int64_t n = t.numel();
    const __m256d sign = _mm256_set1_pd(-0.0);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, _mm256_xor_pd(v, sign));
    }
    for (; i < n; ++i) dst[i] = -src[i];
    return out;
}

static Tensor relu_contiguous_avx2_float(const Tensor& t) {
    const auto* src = static_cast<const float*>(t.data());
    Tensor out = empty(t.shape(), t.dtype(), t.device());
    auto* dst = static_cast<float*>(out.data());
    const int64_t n = t.numel();
    const __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; ++i) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    return out;
}

static Tensor relu_contiguous_avx2_double(const Tensor& t) {
    const auto* src = static_cast<const double*>(t.data());
    Tensor out = empty(t.shape(), t.dtype(), t.device());
    auto* dst = static_cast<double*>(out.data());
    const int64_t n = t.numel();
    const __m256d zero = _mm256_setzero_pd();
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d v = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, _mm256_max_pd(v, zero));
    }
    for (; i < n; ++i) dst[i] = src[i] > 0.0 ? src[i] : 0.0;
    return out;
}

static Tensor leaky_relu_contiguous_avx2_float(const Tensor& t, float negative_slope) {
    const auto* src = static_cast<const float*>(t.data());
    Tensor out = empty(t.shape(), t.dtype(), t.device());
    auto* dst = static_cast<float*>(out.data());
    const int64_t n = t.numel();
    const __m256 zero = _mm256_setzero_ps();
    const __m256 slope = _mm256_set1_ps(negative_slope);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        const __m256 scaled = _mm256_mul_ps(v, slope);
        const __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        _mm256_storeu_ps(dst + i, _mm256_blendv_ps(scaled, v, mask));
    }
    for (; i < n; ++i) dst[i] = src[i] > 0.0f ? src[i] : negative_slope * src[i];
    return out;
}

static Tensor leaky_relu_contiguous_avx2_double(const Tensor& t, float negative_slope) {
    const auto* src = static_cast<const double*>(t.data());
    Tensor out = empty(t.shape(), t.dtype(), t.device());
    auto* dst = static_cast<double*>(out.data());
    const int64_t n = t.numel();
    const __m256d zero = _mm256_setzero_pd();
    const __m256d slope = _mm256_set1_pd(static_cast<double>(negative_slope));
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d v = _mm256_loadu_pd(src + i);
        const __m256d scaled = _mm256_mul_pd(v, slope);
        const __m256d mask = _mm256_cmp_pd(v, zero, _CMP_GT_OQ);
        _mm256_storeu_pd(dst + i, _mm256_blendv_pd(scaled, v, mask));
    }
    for (; i < n; ++i) {
        dst[i] = src[i] > 0.0 ? src[i] : static_cast<double>(negative_slope) * src[i];
    }
    return out;
}
#endif

// Backward helper: applies op(src_val, grad_val) element-wise.
// Handles both contiguous and strided input; grad_output must be contiguous.
template<typename T, typename Op>
static Tensor backward_elementwise(const Tensor& input,
                                    const Tensor& grad_output,
                                    Op op,
                                    const char* name) {
    Tensor grad = empty(input.shape(), input.dtype(), input.device());
    const T* src  = static_cast<const T*>(input.data());
    const T* gout = static_cast<const T*>(grad_output.data());
    T* dst        = static_cast<T*>(grad.data());
    const int64_t n = input.numel();
    if (input.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) dst[i] = op(src[i], gout[i]);
    } else {
        const auto& sh = input.shape();
        const auto& st = input.strides();
        const int ndim = static_cast<int>(sh.size());
        std::vector<int64_t> idx(ndim, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
            dst[elem] = op(src[off], gout[elem]);
            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    }
    return grad;
}

template<typename Op>
static Tensor backward_dispatch(const Tensor& input,
                                 const Tensor& grad_output,
                                 Op op,
                                 const char* name) {
    if (input.dtype() == DType::Float32)
        return backward_elementwise<float>(input, grad_output, op, name);
    if (input.dtype() == DType::Float64)
        return backward_elementwise<double>(input, grad_output, op, name);
    throw std::logic_error("CPU backward adapter received unsupported dtype");
}

Tensor cpu_neg(const Tensor& t) {
#ifdef __AVX2__
    if (t.is_contiguous()) {
        return t.dtype() == DType::Float32 ? neg_contiguous_avx2_float(t)
                                          : neg_contiguous_avx2_double(t);
    }
#endif
    const auto op = [](auto x) { return -x; };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op)
                                      : elementwise_typed<double>(t, op);
}
Tensor cpu_reciprocal(const Tensor& t) {
    const auto op = [](auto x) { using T = decltype(x); return T{1} / x; };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op)
                                      : elementwise_typed<double>(t, op);
}
Tensor cpu_sqrt(const Tensor& t) {
    const auto op = [](auto x) { return std::sqrt(x); };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
Tensor cpu_relu(const Tensor& t) {
#ifdef __AVX2__
    if (t.is_contiguous()) return t.dtype() == DType::Float32 ? relu_contiguous_avx2_float(t) : relu_contiguous_avx2_double(t);
#endif
    const auto op = [](auto x) { return x > decltype(x){0} ? x : decltype(x){0}; };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
Tensor cpu_sigmoid(const Tensor& t) {
    const auto op = [](auto x) { return decltype(x){1} / (decltype(x){1} + std::exp(-x)); };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
Tensor cpu_tanh(const Tensor& t) {
    const auto op = [](auto x) { return std::tanh(x); };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
Tensor cpu_gelu(const Tensor& t) {
    const auto op = [](auto x) { return decltype(x)(0.5) * x * (decltype(x){1} + std::erf(x * decltype(x)(0.7071067811865475))); };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
Tensor cpu_exp(const Tensor& t) {
    const auto op = [](auto x) { return std::exp(x); };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
Tensor cpu_leaky_relu(const Tensor& t, float negative_slope, uint64_t) {
#ifdef __AVX2__
    if (t.is_contiguous()) return t.dtype() == DType::Float32 ? leaky_relu_contiguous_avx2_float(t, negative_slope) : leaky_relu_contiguous_avx2_double(t, negative_slope);
#endif
    const auto op = [negative_slope](auto x) { return x > decltype(x){0} ? x : decltype(x)(negative_slope) * x; };
    return t.dtype() == DType::Float32 ? elementwise_typed<float>(t, op) : elementwise_typed<double>(t, op);
}
static uint64_t splitmix64_host(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static float uniform01_from_u64_host(uint64_t x) {
    const uint32_t top24 = static_cast<uint32_t>(x >> 40);
    return static_cast<float>(top24) * (1.0f / 16777216.0f);
}

Tensor cpu_dropout_mask(const Tensor& t, float p, uint64_t seed) {
    const float keep = 1.0f - p;
    const float scale = 1.0f / keep;
    Tensor mask = empty(t.shape(), DType::Float32, t.device());
    float* dst = static_cast<float*>(mask.data());
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        const float r = uniform01_from_u64_host(
            splitmix64_host(seed + static_cast<uint64_t>(i)));
        dst[i] = r >= p ? scale : 0.0f;
    }
    return mask;
}
Tensor cpu_sigmoid_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        using T = decltype(x);
        T s = T{1} / (T{1} + std::exp(-x));
        return g * s * (T{1} - s);
    }, "sigmoid_backward");
}
Tensor cpu_tanh_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        auto y = std::tanh(x);
        return g * (decltype(x){1} - y * y);
    }, "tanh_backward");
}
Tensor cpu_leaky_relu_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [negative_slope](auto x, auto g) {
        return x > decltype(x){0} ? g : decltype(x)(negative_slope) * g;
    }, "leaky_relu_backward");
}
Tensor cpu_gelu_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        using T = decltype(x);
        constexpr T inv_sqrt2   = T(0.7071067811865475);
        constexpr T inv_sqrt2pi = T(0.3989422804014327);
        T cdf      = T(0.5) * (T{1} + std::erf(x * inv_sqrt2));
        T pdf_term = x * inv_sqrt2pi * std::exp(T(-0.5) * x * x);
        return g * (cdf + pdf_term);
    }, "gelu_backward");
}
Tensor cpu_exp_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        return g * std::exp(x);
    }, "exp_backward");
}
Tensor cpu_neg_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        (void)x;
        return -g;
    }, "neg_backward");
}
Tensor cpu_sqrt_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        using T = decltype(x);
        return g * T(0.5) / std::sqrt(x);
    }, "sqrt_backward");
}
Tensor cpu_reciprocal_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        return -g / (x * x);
    }, "reciprocal_backward");
}
Tensor cpu_relu_backward(const Tensor& input, const Tensor& grad, float) {
        const int64_t n = input.numel();
        const DType dt  = input.dtype();
        Tensor mask     = empty(input.shape(), dt, input.device());

        auto fill_mask = [&](auto zero, auto one) {
            using T = decltype(zero);
            const T* src = static_cast<const T*>(input.data());
            T*       dst = static_cast<T*>(mask.data());
            if (input.is_contiguous()) {
                for (int64_t i = 0; i < n; ++i)
                    dst[i] = src[i] > zero ? one : zero;
            } else {
                const auto& sh  = input.shape();
                const auto& st  = input.strides();
                const int ndim  = static_cast<int>(sh.size());
                std::vector<int64_t> idx(ndim, 0);
                for (int64_t elem = 0; elem < n; ++elem) {
                    int64_t off = 0;
                    for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
                    dst[elem] = src[off] > zero ? one : zero;
                    for (int d = ndim - 1; d >= 0; --d) {
                        if (++idx[d] < sh[d]) break;
                        idx[d] = 0;
                    }
                }
            }
        };

        if (dt == DType::Float64)
            fill_mask(0.0, 1.0);
        else
            fill_mask(0.0f, 1.0f);

    return cpu_mul(grad, mask, input.shape());
}

} } }
