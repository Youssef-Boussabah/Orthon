#include <orthon/ops/cpu/unary_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/activation_node.h>
#include <orthon/autograd/relu_node.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstdint>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef WITH_CUDA
#include <orthon/ops/cuda/unary_kernels.h>
#endif

namespace orthon {

static void validate_unary(const Tensor& t, const char* name) {
    if (t.dtype() != DType::Float32 && t.dtype() != DType::Float64)
        throw std::runtime_error(
            std::string(name) + ": only float32/float64 supported on CPU");
    if (!t.device().is_cpu())
        throw std::runtime_error(
            std::string(name) + ": only CPU supported");
}

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

template<typename Op>
static Tensor elementwise(const Tensor& t, Op op, const char* name) {
    validate_unary(t, name);
    if (t.dtype() == DType::Float32) return elementwise_typed<float>(t, op);
    return elementwise_typed<double>(t, op);
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
    if (!grad_output.is_contiguous())
        throw std::runtime_error(std::string(name) + ": CPU grad_output must be contiguous");
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
    throw std::runtime_error(std::string(name) + ": only float32/float64 supported on CPU");
}

static Tensor attach_activation_grad(const Tensor& input,
                                     Tensor out,
                                     ActivationKind kind,
                                     float alpha = 0.0f) {
    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ActivationNode>(input, kind, alpha));
    }
    return out;
}

// Unary view normalisation.
//
// Every raw cuda_* unary kernel indexes its source LINEARLY (src[i]), so a
// strided operand would either be rejected or read in storage order. CPU
// accepts arbitrary strides for all of them, so the CUDA side is normalised to
// match by materialising only the KERNEL operand - the ActivationNode/ReluNode
// attached afterwards still saves and watches the ORIGINAL tensor, which is
// what keeps mutation detection and view backward routing pointed at the
// user's storage rather than at the temporary.
Tensor neg(const Tensor& t) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("neg: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(t, cuda_neg(kernel_operand(t)), ActivationKind::Neg);
    }
#endif
#ifdef __AVX2__
    if (t.device().is_cpu() && t.is_contiguous()) {
        validate_unary(t, "neg");
        if (t.dtype() == DType::Float32)
            return attach_activation_grad(t, neg_contiguous_avx2_float(t), ActivationKind::Neg);
        return attach_activation_grad(t, neg_contiguous_avx2_double(t), ActivationKind::Neg);
    }
#endif
    return attach_activation_grad(
        t, elementwise(t, [](auto x) { return -x; }, "neg"), ActivationKind::Neg);
}

Tensor sqrt_t(const Tensor& t) {
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        // The dtype guard is load-bearing, not defensive: cuda_sqrt's
        // dispatch falls through to the float32 kernel for anything that is
        // not float16/bfloat16, so without it a float64 CUDA tensor has its
        // bits reinterpreted as float32 and half its buffer written -
        // sqrt([1,4,9,16]) returning [0.061, 0.125, 0, 0], typed float64, with
        // no error. Layout is handled by normalising the kernel operand, since
        // the kernel indexes linearly.
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("sqrt: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(t, cuda_sqrt(kernel_operand(t)), ActivationKind::Sqrt);
#else
        throw std::runtime_error("sqrt: CUDA support not compiled");
#endif
    }
    Tensor out = elementwise(t, [](auto x) { return std::sqrt(x); }, "sqrt");
    return attach_activation_grad(t, out, ActivationKind::Sqrt);
}

Tensor reciprocal_t(const Tensor& t) {
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        // Same two guards as sqrt above, for the same reasons: an unchecked
        // dtype reinterprets float64 as float32, and an unchecked layout
        // reaches a linearly indexing kernel.
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("reciprocal: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(
            t, cuda_reciprocal(kernel_operand(t)), ActivationKind::Reciprocal);
#else
        throw std::runtime_error("reciprocal: CUDA support not compiled");
#endif
    }
    Tensor out = elementwise(t, [](auto x) {
        using T = decltype(x);
        return T{1} / x;
    }, "reciprocal");
    return attach_activation_grad(t, out, ActivationKind::Reciprocal);
}

Tensor relu(const Tensor& t) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("relu: CUDA path requires float32, float16, or bfloat16");
        Tensor out = cuda_relu(kernel_operand(t));
        if (is_grad_enabled() && t.requires_grad()) {
            out.requires_grad_(true);
            out.set_is_leaf(false);
            out.set_grad_fn(std::make_shared<ReluNode>(t));
        }
        return out;
    }
#endif
#ifdef __AVX2__
    if (t.device().is_cpu() && t.is_contiguous()) {
        validate_unary(t, "relu");
        Tensor out = (t.dtype() == DType::Float32)
            ? relu_contiguous_avx2_float(t)
            : relu_contiguous_avx2_double(t);
        if (is_grad_enabled() && t.requires_grad()) {
            out.requires_grad_(true);
            out.set_is_leaf(false);
            out.set_grad_fn(std::make_shared<ReluNode>(t));
        }
        return out;
    }
#endif
    Tensor out = elementwise(t, [](auto x) {
        return x > decltype(x){0} ? x : decltype(x){0};
    }, "relu");
    if (is_grad_enabled() && t.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ReluNode>(t));
    }
    return out;
}

Tensor sigmoid(const Tensor& t) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("sigmoid: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(
            t, cuda_sigmoid(kernel_operand(t)), ActivationKind::Sigmoid);
    }
#endif
    Tensor out = elementwise(t, [](auto x) {
        using T = decltype(x);
        return T{1} / (T{1} + std::exp(-x));
    }, "sigmoid");
    return attach_activation_grad(t, out, ActivationKind::Sigmoid);
}

Tensor tanh_t(const Tensor& t) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("tanh: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(
            t, cuda_tanh(kernel_operand(t)), ActivationKind::Tanh);
    }
#endif
    Tensor out = elementwise(t, [](auto x) { return std::tanh(x); }, "tanh");
    return attach_activation_grad(t, out, ActivationKind::Tanh);
}

Tensor leaky_relu(const Tensor& t, float negative_slope) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("leaky_relu: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(
            t, cuda_leaky_relu(kernel_operand(t), negative_slope),
            ActivationKind::LeakyRelu, negative_slope);
    }
#endif
#ifdef __AVX2__
    if (t.device().is_cpu() && t.is_contiguous()) {
        validate_unary(t, "leaky_relu");
        Tensor out = (t.dtype() == DType::Float32)
            ? leaky_relu_contiguous_avx2_float(t, negative_slope)
            : leaky_relu_contiguous_avx2_double(t, negative_slope);
        return attach_activation_grad(t, out, ActivationKind::LeakyRelu, negative_slope);
    }
#endif
    Tensor out = elementwise(t, [negative_slope](auto x) {
        return x > decltype(x){0} ? x : decltype(x)(negative_slope) * x;
    }, "leaky_relu");
    return attach_activation_grad(t, out, ActivationKind::LeakyRelu, negative_slope);
}

Tensor gelu(const Tensor& t) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("gelu: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(
            t, cuda_gelu(kernel_operand(t)), ActivationKind::Gelu);
    }
#endif
    Tensor out = elementwise(t, [](auto x) {
        using T = decltype(x);
        return T(0.5) * x * (T{1} + std::erf(x * T(0.7071067811865475)));
    }, "gelu");
    return attach_activation_grad(t, out, ActivationKind::Gelu);
}

Tensor exp_t(const Tensor& t) {
#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("exp: CUDA path requires float32, float16, or bfloat16");
        return attach_activation_grad(
            t, cuda_exp(kernel_operand(t)), ActivationKind::Exp);
    }
#endif
    Tensor out = elementwise(t, [](auto x) { return std::exp(x); }, "exp");
    return attach_activation_grad(t, out, ActivationKind::Exp);
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

Tensor dropout(const Tensor& t, float p, uint64_t seed) {
    if (p < 0.0f || p >= 1.0f)
        throw std::runtime_error("dropout: p must be in [0, 1)");
    if (p == 0.0f)
        return t;
    if (t.dtype() != DType::Float32)
        throw std::runtime_error("dropout: only float32 supported");

#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        // No view guard is needed here. The mask is a fresh CONTIGUOUS
        // tensor indexed by LOGICAL element position (seed + i, i in
        // [0, numel)), exactly like the CPU branch below, so element i of the
        // mask corresponds to logical element i of `t`; and CUDA mul accepts
        // strided operands. dropout(view) therefore works on the same RNG
        // stream as dropout of a contiguous tensor - same seed, same draw
        // count, same mask progression.
        Tensor mask = cuda_dropout_mask(t, p, seed);
        return mul(t, mask);
    }
#endif

    if (!t.device().is_cpu())
        throw std::runtime_error("dropout: unsupported device");

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
    return mul(t, mask);
}

// Backward functions.
//
// The saved `input` here is the ORIGINAL public operand, which - because the
// forwards above accept views - can legitimately be strided. The CPU backward
// helper walks strides; the cuda_*_backward kernels index input linearly, so
// each CUDA branch materialises the kernel operand the same way the forward
// does. grad_output is already normalised by ActivationNode / ReluNode, and
// kernel_operand is a no-op on it either way.
//
// The returned gradient is contiguous and in LOGICAL order, matching the
// forward output's shape - which is precisely the coordinate system the
// input's own upstream node (TransposeNode, SliceNode, ...) expects.

Tensor sigmoid_backward(const Tensor& input, const Tensor& grad_output) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_sigmoid_backward(kernel_operand(input), kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        using T = decltype(x);
        T s = T{1} / (T{1} + std::exp(-x));
        return g * s * (T{1} - s);
    }, "sigmoid_backward");
}

Tensor tanh_backward(const Tensor& input, const Tensor& grad_output) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_tanh_backward(kernel_operand(input), kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        auto y = std::tanh(x);
        return g * (decltype(x){1} - y * y);
    }, "tanh_backward");
}

Tensor leaky_relu_backward(const Tensor& input,
                           const Tensor& grad_output,
                           float negative_slope) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_leaky_relu_backward(kernel_operand(input), kernel_operand(grad_output),
                                        negative_slope);
    }
#endif
    return backward_dispatch(input, grad_output, [negative_slope](auto x, auto g) {
        return x > decltype(x){0} ? g : decltype(x)(negative_slope) * g;
    }, "leaky_relu_backward");
}

Tensor gelu_backward(const Tensor& input, const Tensor& grad_output) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_gelu_backward(kernel_operand(input), kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        using T = decltype(x);
        constexpr T inv_sqrt2   = T(0.7071067811865475);
        constexpr T inv_sqrt2pi = T(0.3989422804014327);
        T cdf      = T(0.5) * (T{1} + std::erf(x * inv_sqrt2));
        T pdf_term = x * inv_sqrt2pi * std::exp(T(-0.5) * x * x);
        return g * (cdf + pdf_term);
    }, "gelu_backward");
}

Tensor exp_backward(const Tensor& input, const Tensor& grad_output) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_exp_backward(kernel_operand(input), kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        return g * std::exp(x);
    }, "exp_backward");
}

Tensor neg_backward(const Tensor& input, const Tensor& grad_output) {
    // d(-x)/dx = -1, so grad_input = -grad_output. cuda_neg / backward_dispatch
    // are the raw kernels, so no autograd graph is built during the backward pass.
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_neg(kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        (void)x;
        return -g;
    }, "neg_backward");
}

Tensor sqrt_backward(const Tensor& input, const Tensor& grad_output) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_sqrt_backward(kernel_operand(input), kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        using T = decltype(x);
        return g * T(0.5) / std::sqrt(x);
    }, "sqrt_backward");
}

Tensor reciprocal_backward(const Tensor& input, const Tensor& grad_output) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return cuda_reciprocal_backward(kernel_operand(input), kernel_operand(grad_output));
    }
#endif
    return backward_dispatch(input, grad_output, [](auto x, auto g) {
        return -g / (x * x);
    }, "reciprocal_backward");
}

}  // namespace orthon
