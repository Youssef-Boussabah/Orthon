#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/unary_ops.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/add_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/mul_node.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef WITH_CUDA
#include <orthon/ops/cuda/binary_kernels.h>
#endif

namespace orthon {

static std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) r += ", "; r += std::to_string(s[i]); }
    return r + "]";
}

void check_same_dtype(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype())
        throw std::runtime_error("dtype mismatch");
}

void check_same_device(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device())
        throw std::runtime_error("device mismatch");
}

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    const int nr = std::max(na, nb);
    std::vector<int64_t> out(nr);
    for (int i = 0; i < nr; ++i) {
        int64_t da = (i < nr - na) ? 1 : a[i - (nr - na)];
        int64_t db = (i < nr - nb) ? 1 : b[i - (nr - nb)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error(
                "cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b) +
                ": dimension " + std::to_string(i) + " has sizes " +
                std::to_string(da) + " and " + std::to_string(db));
        // Not max(): a size-1 dim stretches to the OTHER size, including 0
        // ([0,3] + [3] is [0,3]). max(0, 1) would fabricate an element and
        // read out of bounds from the empty operand.
        out[i] = (da == 1) ? db : da;
    }
    return out;
}

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

static void validate_binary_broadcast(const Tensor& a, const Tensor& b) {
    check_same_dtype(a, b);
    check_same_device(a, b);
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)
        throw std::runtime_error("binary ops: only float32/float64 supported on CPU");
    if (!a.device().is_cpu())
        throw std::runtime_error("binary ops: only CPU supported");
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
                                    BinaryFastOp fast_op) {
    const auto out_shape = broadcast_shape(a.shape(), b.shape());
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
        const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
        out.requires_grad_(req);
        out.set_is_leaf(!req);
        return out;
    }

    const auto sa = to_broadcast_strides(a.shape(), a.strides(), nr);
    const auto sb = to_broadcast_strides(b.shape(), b.strides(), nr);

    for (int64_t i = 0; i < n; ++i) {
        int64_t oa = flat_to_offset(i, out_shape, sa);
        int64_t ob = flat_to_offset(i, out_shape, sb);
        po[i] = op(pa[oa], pb[ob]);
    }

    const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    out.requires_grad_(req);
    out.set_is_leaf(!req);
    return out;
}

template<typename Op>
static Tensor broadcast_apply(const Tensor& a,
                              const Tensor& b,
                              Op op,
                              BinaryFastOp fast_op = BinaryFastOp::None) {
    validate_binary_broadcast(a, b);
    if (a.dtype() == DType::Float32)
        return broadcast_apply_typed<float>(a, b, op, fast_op);
    return broadcast_apply_typed<double>(a, b, op, fast_op);
}

Tensor add(const Tensor& a, const Tensor& b) {
#ifdef WITH_CUDA
    if (a.device().is_cuda() || b.device().is_cuda()) {
        check_same_device(a, b);
        check_same_dtype(a, b);
        if (a.dtype() != DType::Float32 && a.dtype() != DType::Float16 && a.dtype() != DType::BFloat16)
            throw std::runtime_error("add: CUDA path requires float32, float16, or bfloat16");
        // CPU add accepts arbitrary strides and broadcast strides, so CUDA add
        // accepts the logically equivalent views. Only the KERNEL operands are
        // materialised (a no-op for the contiguous case, which is every
        // operand in the hot path); the AddNode below stays attached to the
        // ORIGINAL a and b, so the public graph shape, view backward routing
        // and the saved-version watches all describe what the caller passed.
        // Broadcasting is unaffected: materialisation preserves shape exactly.
        Tensor out = cuda_add(kernel_operand(a), kernel_operand(b));
        const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
        out.requires_grad_(req);
        out.set_is_leaf(!req);
        if (req)
            out.set_grad_fn(std::make_shared<AddNode>(a, b));
        return out;
    }
#endif
    Tensor out = broadcast_apply(a, b, [](auto x, auto y) { return x + y; }, BinaryFastOp::Add);
    if (out.requires_grad())
        out.set_grad_fn(std::make_shared<AddNode>(a, b));
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
#ifdef WITH_CUDA
    if (a.device().is_cuda() || b.device().is_cuda()) {
        check_same_device(a, b);
        check_same_dtype(a, b);
        if (a.dtype() != DType::Float32 && a.dtype() != DType::Float16 && a.dtype() != DType::BFloat16)
            throw std::runtime_error("mul: CUDA path requires float32, float16, or bfloat16");
        // Same view normalisation as add. MulNode is value-SENSITIVE, and it
        // is constructed from the originals below, so its watch_version calls
        // record the user tensors' shared Storage - not the throwaway
        // contiguous copies made here. Mutating an original operand after
        // forward therefore fails backward, as it must.
        Tensor out = cuda_mul(kernel_operand(a), kernel_operand(b));
        const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
        out.requires_grad_(req);
        out.set_is_leaf(!req);
        if (req)
            out.set_grad_fn(std::make_shared<MulNode>(a, b));
        return out;
    }
#endif
    Tensor out = broadcast_apply(a, b, [](auto x, auto y) { return x * y; }, BinaryFastOp::Mul);
    if (out.requires_grad())
        out.set_grad_fn(std::make_shared<MulNode>(a, b));
    return out;
}

// Subtraction and division, by composition. See binary_ops.h for why no new
// kernel is introduced. Each inherits its validation from the op it is built
// on rather than restating it: neg()/reciprocal_t() run their own dtype and
// device dispatch first, then add()/mul() apply the shared broadcast, dtype
// and device checks, so a mismatched pair fails with the same message the
// underlying op produces.
Tensor sub(const Tensor& a, const Tensor& b) {
    return add(a, neg(b));
}

Tensor div(const Tensor& a, const Tensor& b) {
    return mul(a, reciprocal_t(b));
}

// The view normalisation above deliberately does NOT extend to the in-place
// ops below. An out-of-place op may materialise its operands because the
// materialised copy is only ever READ; an in-place op writes to its
// destination, and writing to a temporary contiguous copy would silently
// discard the mutation the caller asked for. Supporting a strided destination
// means a stride-aware in-place kernel, which is a kernel change, not a
// normalisation. The explicit contiguous-destination requirement therefore
// stands, and is pinned by test_runtime_consistency.py.
void mul_(Tensor& a, float alpha) {
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)
        throw std::runtime_error("mul_: only float32/float64 supported");
    if (!a.is_contiguous())
        throw std::runtime_error("mul_: tensor must be contiguous");
#ifdef WITH_CUDA
    if (a.device().is_cuda()) {
        cuda_mul_(a, alpha);
        // The version bump belongs to this native boundary, not the CUDA helper.
        a.impl()->storage()->bump_version();
        return;
    }
#endif
    if (!a.device().is_cpu())
        throw std::runtime_error("mul_: unsupported device");
    const int64_t n = a.numel();
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
    // Exactly one bump per successful logical mul_ (CPU path).
    a.impl()->storage()->bump_version();
}

void add_(Tensor& a, const Tensor& b, float alpha) {
    if (a.shape() != b.shape())
        throw std::runtime_error("add_: shape mismatch");
    if (a.dtype() != b.dtype())
        throw std::runtime_error("add_: dtype mismatch");
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)
        throw std::runtime_error("add_: only float32/float64 supported");
    if (a.device() != b.device())
        throw std::runtime_error("add_: device mismatch");
    if (!a.is_contiguous() || !b.is_contiguous())
        throw std::runtime_error("add_: tensors must be contiguous");
#ifdef WITH_CUDA
    if (a.device().is_cuda()) {
        cuda_add_(a, b, alpha);
        // The version bump belongs to this native boundary, not the CUDA helper.
        a.impl()->storage()->bump_version();
        return;
    }
#endif
    if (!a.device().is_cpu())
        throw std::runtime_error("add_: unsupported device");
    const int64_t n = a.numel();
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
    // Exactly one bump per successful logical add_ (CPU path).
    a.impl()->storage()->bump_version();
}

}  // namespace orthon
