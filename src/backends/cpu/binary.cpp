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
static std::vector<int64_t> to_broadcast_strides(const std::vector<int64_t>& shape,
                                                  const std::vector<int64_t>& strides,
                                                  int target_ndim) {
    const int n = static_cast<int>(shape.size());
    std::vector<int64_t> out(target_ndim, 0);
    for (int i = 0; i < n; ++i) {
        int out_i = target_ndim - n + i;
        out[out_i] = (shape[i] == 1) ? 0 : strides[i];
    }
    return out;
}

static int64_t flat_to_offset(int64_t idx,
                               const std::vector<int64_t>& out_shape,
                               const std::vector<int64_t>& strides) {
    int64_t offset = 0;
    for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
        offset += (idx % out_shape[i]) * strides[i];
        idx    /= out_shape[i];
    }
    return offset;
}

enum class BinaryFastOp {
    None,
    Add,
    Mul,
};

template<typename T>
static void apply_same_shape_scalar(const T* pa,
                                    const T* pb,
                                    T* po,
                                    int64_t n,
                                    BinaryFastOp fast_op) {
    if (fast_op == BinaryFastOp::Add) {
        for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
    } else {
        for (int64_t i = 0; i < n; ++i) po[i] = pa[i] * pb[i];
    }
}

#ifdef __AVX2__
static void add_same_shape_avx2(const float* pa, const float* pb, float* po, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 a = _mm256_loadu_ps(pa + i);
        const __m256 b = _mm256_loadu_ps(pb + i);
        _mm256_storeu_ps(po + i, _mm256_add_ps(a, b));
    }
    for (; i < n; ++i) po[i] = pa[i] + pb[i];
}

static void add_same_shape_avx2(const double* pa, const double* pb, double* po, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(pa + i);
        const __m256d b = _mm256_loadu_pd(pb + i);
        _mm256_storeu_pd(po + i, _mm256_add_pd(a, b));
    }
    for (; i < n; ++i) po[i] = pa[i] + pb[i];
}

static void mul_same_shape_avx2(const float* pa, const float* pb, float* po, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 a = _mm256_loadu_ps(pa + i);
        const __m256 b = _mm256_loadu_ps(pb + i);
        _mm256_storeu_ps(po + i, _mm256_mul_ps(a, b));
    }
    for (; i < n; ++i) po[i] = pa[i] * pb[i];
}

static void mul_same_shape_avx2(const double* pa, const double* pb, double* po, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(pa + i);
        const __m256d b = _mm256_loadu_pd(pb + i);
        _mm256_storeu_pd(po + i, _mm256_mul_pd(a, b));
    }
    for (; i < n; ++i) po[i] = pa[i] * pb[i];
}

static void mul_inplace_avx2(float* pa, float alpha, int64_t n) {
    const __m256 scale = _mm256_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 a = _mm256_loadu_ps(pa + i);
        _mm256_storeu_ps(pa + i, _mm256_mul_ps(a, scale));
    }
    for (; i < n; ++i) pa[i] *= alpha;
}

static void mul_inplace_avx2(double* pa, double alpha, int64_t n) {
    const __m256d scale = _mm256_set1_pd(alpha);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(pa + i);
        _mm256_storeu_pd(pa + i, _mm256_mul_pd(a, scale));
    }
    for (; i < n; ++i) pa[i] *= alpha;
}

static void add_inplace_avx2(float* pa, const float* pb, float alpha, int64_t n) {
    const __m256 scale = _mm256_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 a = _mm256_loadu_ps(pa + i);
        const __m256 b = _mm256_loadu_ps(pb + i);
#ifdef __FMA__
        _mm256_storeu_ps(pa + i, _mm256_fmadd_ps(b, scale, a));
#else
        _mm256_storeu_ps(pa + i, _mm256_add_ps(a, _mm256_mul_ps(b, scale)));
#endif
    }
    for (; i < n; ++i) pa[i] += alpha * pb[i];
}

static void add_inplace_avx2(double* pa, const double* pb, double alpha, int64_t n) {
    const __m256d scale = _mm256_set1_pd(alpha);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(pa + i);
        const __m256d b = _mm256_loadu_pd(pb + i);
#ifdef __FMA__
        _mm256_storeu_pd(pa + i, _mm256_fmadd_pd(b, scale, a));
#else
        _mm256_storeu_pd(pa + i, _mm256_add_pd(a, _mm256_mul_pd(b, scale)));
#endif
    }
    for (; i < n; ++i) pa[i] += alpha * pb[i];
}
#endif

template<typename T, typename Op>
static Tensor broadcast_apply_typed(const Tensor& a,
                                    const Tensor& b,
                                    Op op,
                                    BinaryFastOp fast_op,
                                    const std::vector<int64_t>& out_shape) {
    const int nr = static_cast<int>(out_shape.size());

    Tensor out = empty(out_shape, a.dtype(), a.device());
    const auto* pa = static_cast<const T*>(a.data());
    const auto* pb = static_cast<const T*>(b.data());
    auto*       po = static_cast<T*>(out.data());

    const int64_t n = out.numel();
    if (fast_op != BinaryFastOp::None && a.shape() == b.shape() &&
        a.is_contiguous() && b.is_contiguous()) {
#ifdef __AVX2__
        if (fast_op == BinaryFastOp::Add) {
            add_same_shape_avx2(pa, pb, po, n);
        } else {
            mul_same_shape_avx2(pa, pb, po, n);
        }
#else
        apply_same_shape_scalar(pa, pb, po, n, fast_op);
#endif
        return out;
    }

    const auto sa = to_broadcast_strides(a.shape(), a.strides(), nr);
    const auto sb = to_broadcast_strides(b.shape(), b.strides(), nr);

    for (int64_t i = 0; i < n; ++i) {
        int64_t oa = flat_to_offset(i, out_shape, sa);
        int64_t ob = flat_to_offset(i, out_shape, sb);
        po[i] = op(pa[oa], pb[ob]);
    }

    return out;
}

Tensor cpu_add(const Tensor& a, const Tensor& b,
                                  const std::vector<int64_t>& shape) {
    const auto op = [](auto x, auto y) { return x + y; };
    return a.dtype() == DType::Float32
        ? broadcast_apply_typed<float>(a, b, op, BinaryFastOp::Add, shape)
        : broadcast_apply_typed<double>(a, b, op, BinaryFastOp::Add, shape);
}
Tensor cpu_mul(const Tensor& a, const Tensor& b,
                                  const std::vector<int64_t>& shape) {
    const auto op = [](auto x, auto y) { return x * y; };
    return a.dtype() == DType::Float32
        ? broadcast_apply_typed<float>(a, b, op, BinaryFastOp::Mul, shape)
        : broadcast_apply_typed<double>(a, b, op, BinaryFastOp::Mul, shape);
}

void inplace_mul_(Tensor& a, float alpha) {    const int64_t n = a.numel();
    if (a.dtype() == DType::Float32) {
        float* pa = static_cast<float*>(a.data());
#ifdef __AVX2__
        mul_inplace_avx2(pa, alpha, n);
#else
        for (int64_t i = 0; i < n; ++i) pa[i] *= alpha;
#endif
    } else {
        double* pa = static_cast<double*>(a.data());
#ifdef __AVX2__
        mul_inplace_avx2(pa, static_cast<double>(alpha), n);
#else
        for (int64_t i = 0; i < n; ++i) pa[i] *= static_cast<double>(alpha);
#endif
    }

}
void inplace_add_(Tensor& a, const Tensor& b, float alpha) {    const int64_t n = a.numel();
    if (a.dtype() == DType::Float32) {
        float*       pa = static_cast<float*>(a.data());
        const float* pb = static_cast<const float*>(b.data());
#ifdef __AVX2__
        add_inplace_avx2(pa, pb, alpha, n);
#else
        for (int64_t i = 0; i < n; ++i) pa[i] += alpha * pb[i];
#endif
    } else {
        double*       pa = static_cast<double*>(a.data());
        const double* pb = static_cast<const double*>(b.data());
#ifdef __AVX2__
        add_inplace_avx2(pa, pb, static_cast<double>(alpha), n);
#else
        for (int64_t i = 0; i < n; ++i) pa[i] += static_cast<double>(alpha) * pb[i];
#endif
    }

}
} } }
