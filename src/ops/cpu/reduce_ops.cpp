#include <orthon/ops/cpu/reduce_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/reduce_node.h>
#include <orthon/autograd/max_node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/reduce_kernels.h>
#include <orthon/ops/cuda/unary_kernels.h>
#endif

namespace orthon {

static void validate_reduce_common(const Tensor& t, const char* name) {
    if (t.numel() == 0)
        throw std::runtime_error(
            std::string(name) + ": cannot reduce empty tensor");
    // Float16 is allowed on CUDA; restrict dtype only for CPU paths.
    if (t.device().is_cpu() && t.dtype() != DType::Float32 && t.dtype() != DType::Float64)
        throw std::runtime_error(
            std::string(name) + ": only float32/float64 supported on CPU");
}

static int64_t resolve_dim(int64_t dim, int64_t ndim, const char* name) {
    const int64_t original = dim;
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::out_of_range(
            std::string(name) + ": dim " + std::to_string(original) +
            " out of range for tensor with " + std::to_string(ndim) +
            " dimensions");
    return dim;
}

static std::vector<int64_t> reduction_shape(const Tensor& t,
                                            std::optional<int64_t> dim,
                                            bool keepdim,
                                            const char* name) {
    if (!dim.has_value())
        return keepdim ? std::vector<int64_t>(t.ndim(), 1) : std::vector<int64_t>{};

    if (t.ndim() == 0)
        throw std::out_of_range(std::string(name) + ": dim out of range for scalar tensor");

    const int64_t d = resolve_dim(*dim, t.ndim(), name);
    std::vector<int64_t> out = t.shape();
    if (keepdim) {
        out[static_cast<size_t>(d)] = 1;
    } else {
        out.erase(out.begin() + d);
    }
    return out;
}

static int64_t reduction_factor(const Tensor& t, std::optional<int64_t> dim) {
    if (!dim.has_value())
        return t.numel();
    const int64_t d = resolve_dim(*dim, t.ndim(), "mean");
    return t.shape()[static_cast<size_t>(d)];
}

template<typename T>
static Tensor cpu_reduce_all(const Tensor& t,
                              const std::vector<int64_t>& out_shape,
                              T scale) {
    Tensor out = empty(out_shape, t.dtype(), t.device());
    T* dst = static_cast<T*>(out.data());
    const T* base = static_cast<const T*>(t.data());
    const int64_t n = t.numel();

    double acc = 0.0;
    if (t.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i)
            acc += static_cast<double>(base[i]);
    } else {
        const auto& sh = t.shape();
        const auto& st = t.strides();
        const int ndim = static_cast<int>(sh.size());
        std::vector<int64_t> idx(ndim, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d)
                off += idx[d] * st[d];
            acc += static_cast<double>(base[off]);
            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    }
    dst[0] = static_cast<T>(acc) * scale;
    return out;
}

template<typename T>
static Tensor cpu_reduce_dim(const Tensor& t,
                              const std::vector<int64_t>& out_shape,
                              int64_t dim,
                              bool keepdim,
                              T scale) {
    Tensor out = zeros(out_shape, t.dtype(), t.device());
    const T* src = static_cast<const T*>(t.data());
    T* dst = static_cast<T*>(out.data());
    const auto& sh = t.shape();
    const auto& st = t.strides();
    const int ndim = static_cast<int>(sh.size());
    const int64_t n = t.numel();

    std::vector<int64_t> idx(ndim, 0);
    for (int64_t elem = 0; elem < n; ++elem) {
        int64_t src_off = 0;
        int64_t dst_flat = 0;
        for (int d = 0; d < ndim; ++d) {
            src_off += idx[d] * st[d];
            if (d == dim) {
                // Under keepdim the reduced dim has extent 1, so it never
                // advances dst_flat.
                continue;
            }
            dst_flat = dst_flat * sh[d] + idx[d];
        }
        dst[dst_flat] += src[src_off] * scale;

        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}

static Tensor attach_reduce_grad(const Tensor& input,
                                 Tensor out,
                                 std::optional<int64_t> dim,
                                 bool keepdim,
                                 bool is_mean) {
    if (is_grad_enabled() && input.requires_grad()) {
        std::optional<int64_t> resolved = dim;
        if (resolved.has_value())
            resolved = resolve_dim(*resolved, input.ndim(), is_mean ? "mean" : "sum");
        const int64_t factor = is_mean ? reduction_factor(input, resolved) : 1;
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(
            std::make_shared<ReduceNode>(input, resolved, keepdim, is_mean, factor));
    }
    return out;
}

static Tensor reduce_impl(const Tensor& t,
                          std::optional<int64_t> dim,
                          bool keepdim,
                          bool is_mean,
                          const char* name) {
    validate_reduce_common(t, name);
    std::optional<int64_t> resolved = dim;
    if (resolved.has_value())
        resolved = resolve_dim(*resolved, t.ndim(), name);
    const std::vector<int64_t> out_shape = reduction_shape(t, resolved, keepdim, name);
    const bool reduce_all = !resolved.has_value();

#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error(std::string(name) + ": CUDA path requires float32, float16, or bfloat16");
        Tensor input = t.is_contiguous() ? t : cuda_make_contiguous(t);
        Tensor out = is_mean
            ? cuda_mean(input, out_shape, resolved.value_or(0), reduce_all)
            : cuda_sum(input, out_shape, resolved.value_or(0), reduce_all);
        return attach_reduce_grad(t, out, resolved, keepdim, is_mean);
    }
#endif

    if (!t.device().is_cpu())
        throw std::runtime_error(std::string(name) + ": unsupported device");

    Tensor out = [&]() -> Tensor {
        if (t.dtype() == DType::Float32) {
            const float scale = is_mean
                ? 1.0f / static_cast<float>(reduce_all ? t.numel() : t.shape()[static_cast<size_t>(*resolved)])
                : 1.0f;
            return reduce_all
                ? cpu_reduce_all<float>(t, out_shape, scale)
                : cpu_reduce_dim<float>(t, out_shape, *resolved, keepdim, scale);
        } else {
            const double scale = is_mean
                ? 1.0 / static_cast<double>(reduce_all ? t.numel() : t.shape()[static_cast<size_t>(*resolved)])
                : 1.0;
            return reduce_all
                ? cpu_reduce_all<double>(t, out_shape, scale)
                : cpu_reduce_dim<double>(t, out_shape, *resolved, keepdim, scale);
        }
    }();
    return attach_reduce_grad(t, out, resolved, keepdim, is_mean);
}

Tensor sum(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    return reduce_impl(t, dim, keepdim, false, "sum");
}

Tensor mean(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    return reduce_impl(t, dim, keepdim, true, "mean");
}

// Max and argmax helpers.

template<typename T>
static std::pair<Tensor, Tensor> cpu_max_all(const Tensor& t,
                                              const std::vector<int64_t>& out_shape) {
    const T* base = static_cast<const T*>(t.data());
    const int64_t n = t.numel();

    // -infinity, not lowest(): a finite seed misreports an all--infinity
    // input and steals index 0 when the true max equals lowest().
    T       best_val = -std::numeric_limits<T>::infinity();
    int64_t best_idx = 0;

    if (t.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) {
            if (base[i] > best_val) { best_val = base[i]; best_idx = i; }
        }
    } else {
        const auto& sh = t.shape();
        const auto& st = t.strides();
        const int ndim = static_cast<int>(sh.size());
        std::vector<int64_t> idx(ndim, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
            if (base[off] > best_val) { best_val = base[off]; best_idx = elem; }
            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    }

    Tensor val_t = empty(out_shape, t.dtype(), Device{});
    Tensor idx_t(std::make_shared<TensorImpl>(out_shape, DType::Int64, Device{}));
    *static_cast<T*>(val_t.data())        = best_val;
    *static_cast<int64_t*>(idx_t.data()) = best_idx;
    return {val_t, idx_t};
}

template<typename T>
static std::pair<Tensor, Tensor> cpu_max_dim(const Tensor& t,
                                              const std::vector<int64_t>& out_shape,
                                              int64_t dim,
                                              bool keepdim) {
    int64_t out_numel = 1;
    for (int64_t s : out_shape) out_numel *= s;

    Tensor val_t = empty(out_shape, t.dtype(), Device{});
    Tensor idx_t(std::make_shared<TensorImpl>(out_shape, DType::Int64, Device{}));
    T*       val_ptr = static_cast<T*>(val_t.data());
    int64_t* idx_ptr = static_cast<int64_t*>(idx_t.data());
    for (int64_t i = 0; i < out_numel; ++i) {
        // Same -infinity identity as the all-reduce above.
        val_ptr[i] = -std::numeric_limits<T>::infinity();
        idx_ptr[i] = 0;
    }

    const T* src     = static_cast<const T*>(t.data());
    const auto& sh   = t.shape();
    const auto& st   = t.strides();
    const int ndim   = static_cast<int>(sh.size());
    const int64_t n  = t.numel();

    std::vector<int64_t> coords(ndim, 0);
    for (int64_t elem = 0; elem < n; ++elem) {
        int64_t src_off  = 0;
        int64_t dst_flat = 0;
        for (int d = 0; d < ndim; ++d) src_off += coords[d] * st[d];

        for (int d = 0; d < ndim; ++d) {
            // Under keepdim the reduced dim has extent 1: it never advances dst_flat.
            if (d == dim) continue;
            dst_flat = dst_flat * sh[d] + coords[d];
        }

        T v = src[src_off];
        if (v > val_ptr[dst_flat]) {
            val_ptr[dst_flat] = v;
            idx_ptr[dst_flat] = coords[dim];
        }

        for (int d = ndim - 1; d >= 0; --d) {
            if (++coords[d] < sh[d]) break;
            coords[d] = 0;
        }
    }
    return {val_t, idx_t};
}

Tensor reduce_max(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    validate_reduce_common(t, "max");
    std::optional<int64_t> resolved = dim;
    if (resolved.has_value())
        resolved = resolve_dim(*resolved, t.ndim(), "max");
    const std::vector<int64_t> out_shape = reduction_shape(t, resolved, keepdim, "max");
    const bool reduce_all = !resolved.has_value();

    int64_t outer_sz = 1, inner_sz = 1, reduce_sz = 1;
    if (!reduce_all) {
        const auto& sh = t.shape();
        for (int64_t i = 0; i < *resolved; ++i) outer_sz *= sh[static_cast<size_t>(i)];
        for (int64_t i = *resolved + 1; i < static_cast<int64_t>(sh.size()); ++i)
            inner_sz *= sh[static_cast<size_t>(i)];
        reduce_sz = sh[static_cast<size_t>(*resolved)];
    }

#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("max: CUDA path requires float32, float16, or bfloat16");
        Tensor input = t.is_contiguous() ? t : cuda_make_contiguous(t);
        auto [val_t, idx_t] = cuda_max_with_indices(input, out_shape,
                                                     resolved.value_or(0), reduce_all);
        if (is_grad_enabled() && t.requires_grad()) {
            val_t.requires_grad_(true);
            val_t.set_is_leaf(false);
            val_t.set_grad_fn(std::make_shared<MaxNode>(
                t, idx_t, resolved, keepdim, outer_sz, inner_sz, reduce_sz));
        }
        return val_t;
    }
#endif

    if (!t.device().is_cpu())
        throw std::runtime_error("max: unsupported device");

    auto [val_t, idx_t] = [&]() -> std::pair<Tensor, Tensor> {
        if (t.dtype() == DType::Float32)
            return reduce_all
                ? cpu_max_all<float>(t, out_shape)
                : cpu_max_dim<float>(t, out_shape, *resolved, keepdim);
        return reduce_all
            ? cpu_max_all<double>(t, out_shape)
            : cpu_max_dim<double>(t, out_shape, *resolved, keepdim);
    }();

    if (is_grad_enabled() && t.requires_grad()) {
        val_t.requires_grad_(true);
        val_t.set_is_leaf(false);
        val_t.set_grad_fn(std::make_shared<MaxNode>(
            t, idx_t, resolved, keepdim, outer_sz, inner_sz, reduce_sz));
    }
    return val_t;
}

Tensor argmax(const Tensor& t, std::optional<int64_t> dim, bool keepdim) {
    validate_reduce_common(t, "argmax");
    std::optional<int64_t> resolved = dim;
    if (resolved.has_value())
        resolved = resolve_dim(*resolved, t.ndim(), "argmax");
    const std::vector<int64_t> out_shape = reduction_shape(t, resolved, keepdim, "argmax");
    const bool reduce_all = !resolved.has_value();

#ifdef WITH_CUDA
    if (t.device().is_cuda()) {
        if (t.dtype() != DType::Float32 && t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16)
            throw std::runtime_error("argmax: CUDA path requires float32, float16, or bfloat16");
        Tensor input = t.is_contiguous() ? t : cuda_make_contiguous(t);
        auto [val_t, idx_t] = cuda_max_with_indices(input, out_shape,
                                                     resolved.value_or(0), reduce_all);
        return idx_t;
    }
#endif

    if (!t.device().is_cpu())
        throw std::runtime_error("argmax: unsupported device");

    auto [val_t, idx_t] = [&]() -> std::pair<Tensor, Tensor> {
        if (t.dtype() == DType::Float32)
            return reduce_all
                ? cpu_max_all<float>(t, out_shape)
                : cpu_max_dim<float>(t, out_shape, *resolved, keepdim);
        return reduce_all
            ? cpu_max_all<double>(t, out_shape)
            : cpu_max_dim<double>(t, out_shape, *resolved, keepdim);
    }();
    return idx_t;
}

}  // namespace orthon
