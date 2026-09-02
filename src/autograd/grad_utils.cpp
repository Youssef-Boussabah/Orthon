#include <orthon/autograd/grad_utils.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/dtype.h>
#include <stdexcept>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/binary_kernels.h>
#endif

namespace orthon {

Tensor unbroadcast(const Tensor& grad, const std::vector<int64_t>& target_shape) {
    // Shapes already match: forward the gradient untouched, preserving a view
    // rather than forcing a copy. The CPU reduction path below is stride-aware.
    if (grad.shape() == target_shape) return grad;

#ifdef WITH_CUDA
    if (grad.device().is_cuda()) {
        // cuda_unbroadcast indexes its input linearly, so a strided gradient
        // must be materialised first.
        return cuda_unbroadcast(contiguous(grad), target_shape);
    }
#endif

    const int grad_ndim = static_cast<int>(grad.shape().size());
    const int tgt_ndim  = static_cast<int>(target_shape.size());

    // Left-pad target_shape with 1s to match grad rank.
    std::vector<int64_t> padded(grad_ndim, 1);
    for (int i = 0; i < tgt_ndim; ++i)
        padded[grad_ndim - tgt_ndim + i] = target_shape[i];

    // The accumulation is typed on the GRADIENT's dtype, not hard-coded to
    // Float32. Allocating a float32 result and reinterpreting the gradient's
    // storage as float* is right only for float32; for float64 it reads half
    // of each double's bits as a float and leaves the leaf carrying a float32
    // gradient:
    //
    //     a(1,3) + b(2,1), float64, backward
    //     a.grad -> [1.875, 1.875, 1.875] dtype float32   (expected [2,2,2])
    //     b.grad -> [1.875, 3.75]         dtype float32   (expected [3,3])
    //
    // Broadcasting backward is on the float64 parity path, so the reduction
    // dispatches on dtype below.
    const DType dt = grad.dtype();
    if (dt != DType::Float32 && dt != DType::Float64)
        throw std::runtime_error(
            "unbroadcast: only float32/float64 supported on CPU");

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
        result = reshape(result, target_shape);

    return result;
}

}  // namespace orthon
