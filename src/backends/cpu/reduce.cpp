#include <orthon/dispatch/operator.h>
#include <orthon/reduction/extrema.h>
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
template<typename T>
static Tensor cpu_reduce_all(const Tensor& t,
                              const ReductionPlan& plan,
                              T scale) {
    Tensor out = empty(plan.output_shape, t.dtype(), t.device());
    T* dst = static_cast<T*>(out.data());
    const T* base = static_cast<const T*>(t.data());
    const int64_t n = plan.input_numel;

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
                              const ReductionPlan& plan,
                              T scale) {
    const int64_t dim = *plan.dim;
    Tensor out = zeros(plan.output_shape, t.dtype(), t.device());
    const T* src = static_cast<const T*>(t.data());
    T* dst = static_cast<T*>(out.data());
    const auto& sh = t.shape();
    const auto& st = t.strides();
    const int ndim = static_cast<int>(sh.size());
    const int64_t n = plan.input_numel;

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


ReductionResult cpu_sum(const Tensor& t, const ReductionPlan& plan) {
    if (!t.device().is_cpu()) throw std::invalid_argument("reduction: CPU execution requires CPU operand");
    validate_reduction_operand(t, plan, ReductionKind::Sum);
    Tensor out = t.dtype() == DType::Float32
        ? (!plan.dim ? cpu_reduce_all<float>(t, plan, 1.0f)
                     : cpu_reduce_dim<float>(t, plan, 1.0f))
        : (!plan.dim ? cpu_reduce_all<double>(t, plan, 1.0)
                     : cpu_reduce_dim<double>(t, plan, 1.0));
    return {out, {}};
}
ReductionResult cpu_mean(const Tensor& t, const ReductionPlan& plan) {
    if (!t.device().is_cpu()) throw std::invalid_argument("reduction: CPU execution requires CPU operand");
    validate_reduction_operand(t, plan, ReductionKind::Mean);
    const int64_t factor = plan.reduce;
    if (t.dtype() == DType::Float32) {
        const float scale = 1.0f / static_cast<float>(factor);
        return {!plan.dim ? cpu_reduce_all<float>(t, plan, scale)
                          : cpu_reduce_dim<float>(t, plan, scale), {}};
    }
    const double scale = 1.0 / static_cast<double>(factor);
    return {!plan.dim ? cpu_reduce_all<double>(t, plan, scale)
                      : cpu_reduce_dim<double>(t, plan, scale), {}};
}

template<typename T>
static std::pair<Tensor, Tensor> cpu_max_all(const Tensor& t,
                                              const ReductionPlan& plan) {
    const T* base = static_cast<const T*>(t.data());
    const int64_t n = plan.input_numel;

    // -infinity, not lowest(): a finite seed misreports an all--infinity
    // input and steals index 0 when the true max equals lowest().
    T       best_val = -std::numeric_limits<T>::infinity();
    int64_t best_idx = -1;

    if (t.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) {
            if (extrema_candidate_wins(base[i], i, best_val, best_idx)) { best_val = base[i]; best_idx = i; }
        }
    } else {
        const auto& sh = t.shape();
        const auto& st = t.strides();
        const int ndim = static_cast<int>(sh.size());
        std::vector<int64_t> idx(ndim, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
            if (extrema_candidate_wins(base[off], elem, best_val, best_idx)) { best_val = base[off]; best_idx = elem; }
            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    }

    Tensor val_t = empty(plan.output_shape, t.dtype(), Device{});
    Tensor idx_t(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64, Device{}));
    *static_cast<T*>(val_t.data())        = best_val;
    *static_cast<int64_t*>(idx_t.data()) = best_idx;
    return {val_t, idx_t};
}

template<typename T>
static std::pair<Tensor, Tensor> cpu_max_dim(const Tensor& t,
                                              const ReductionPlan& plan) {

    const int64_t dim = *plan.dim, out_numel = plan.output_numel;
    Tensor val_t = empty(plan.output_shape, t.dtype(), Device{});
    Tensor idx_t(std::make_shared<TensorImpl>(plan.output_shape, DType::Int64, Device{}));
    T*       val_ptr = static_cast<T*>(val_t.data());
    int64_t* idx_ptr = static_cast<int64_t*>(idx_t.data());
    for (int64_t i = 0; i < out_numel; ++i) {
        // Same -infinity identity as the all-reduce above.
        val_ptr[i] = -std::numeric_limits<T>::infinity();
        idx_ptr[i] = -1;
    }

    const T* src     = static_cast<const T*>(t.data());
    const auto& sh   = t.shape();
    const auto& st   = t.strides();
    const int ndim   = static_cast<int>(sh.size());
    const int64_t n  = plan.input_numel;

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
        if (extrema_candidate_wins(v, coords[dim], val_ptr[dst_flat], idx_ptr[dst_flat])) {
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

ReductionResult cpu_max(const Tensor& t, const ReductionPlan& plan) {
    if (!t.device().is_cpu()) throw std::invalid_argument("reduction: CPU execution requires CPU operand");
    validate_reduction_operand(t, plan, ReductionKind::Max);
    auto result = t.dtype() == DType::Float32
        ? (!plan.dim ? cpu_max_all<float>(t, plan)
                     : cpu_max_dim<float>(t, plan))
        : (!plan.dim ? cpu_max_all<double>(t, plan)
                     : cpu_max_dim<double>(t, plan));
    return {result.first, result.second};
}

} } }
