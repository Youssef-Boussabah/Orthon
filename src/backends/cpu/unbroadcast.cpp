#include <orthon/core/execution_copy.h>
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
Tensor cpu_unbroadcast(const Tensor& grad, const std::vector<int64_t>& target_shape) {
    const int grad_ndim = static_cast<int>(grad.shape().size());
    const int tgt_ndim  = static_cast<int>(target_shape.size());

    // Left-pad target_shape with 1s to match grad rank.
    std::vector<int64_t> padded(grad_ndim, 1);
    for (int i = 0; i < tgt_ndim; ++i)
        padded[grad_ndim - tgt_ndim + i] = target_shape[i];

    // The accumulator follows the gradient dtype. In particular, a Float64
    // gradient must not be read or accumulated through Float32 pointers.
    const DType dt = grad.dtype();

    Tensor result = zeros(padded, dt, grad.device());

    const int64_t n  = grad.numel();
    const auto&   gs = grad.shape();
    const auto&   gt = grad.strides();

    auto accumulate = [&](auto* dst, const auto* src) {
        for (int64_t flat = 0; flat < n; ++flat) {
            // Decompose flat contiguous index into per-dim indices.
            int64_t remaining = flat;
            std::vector<int64_t> idx(grad_ndim);
            for (int d = grad_ndim - 1; d >= 0; --d) {
                idx[d]     = remaining % gs[d];
                remaining /= gs[d];
            }

            // Source offset respects non-contiguous strides.
            int64_t src_off = 0;
            for (int d = 0; d < grad_ndim; ++d)
                src_off += idx[d] * gt[d];

            // Collapse any broadcast dimension to index 0 in the result.
            int64_t res_flat = 0;
            for (int d = 0; d < grad_ndim; ++d) {
                int64_t ri = (padded[d] == 1) ? 0 : idx[d];
                res_flat   = res_flat * padded[d] + ri;
            }

            dst[res_flat] += src[src_off];
        }
    };

    if (dt == DType::Float32) {
        accumulate(static_cast<float*>(result.data()),
                   static_cast<const float*>(grad.data()));
    } else {
        accumulate(static_cast<double*>(result.data()),
                   static_cast<const double*>(grad.data()));
    }

    // Remove the leading 1s we inserted if target had fewer dims than grad.
    if (padded != target_shape)
        result = execution_reshape(result, target_shape);

    return result;
}
} } }
