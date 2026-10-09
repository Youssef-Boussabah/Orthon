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
Tensor cpu_max_backward(const Tensor& input, const Tensor& grad, const Tensor& argmax_indices_, const ReductionPlan& plan) {
    validate_max_scatter(input, grad, argmax_indices_, plan);
    const auto& input_shape_ = plan.input_shape;
    const auto outer_ = plan.outer, inner_ = plan.inner, reduce_ = plan.reduce;
    const auto dt = input.dtype();
        // CPU dim-reduction scatter.
        Tensor grad_in = zeros(input_shape_, dt, Device{});
        const int64_t* idx_ptr = static_cast<const int64_t*>(argmax_indices_.data());
        const int64_t  n_out   = plan.output_numel;
        if (dt == DType::Float64) {
            const double* go_ptr = static_cast<const double*>(grad.data());
            double*       gi_ptr = static_cast<double*>(grad_in.data());
            for (int64_t i = 0; i < n_out; ++i) {
                int64_t outer_i = i / inner_;
                int64_t inner_i = i % inner_;
                int64_t r       = idx_ptr[i];
                gi_ptr[outer_i * reduce_ * inner_ + r * inner_ + inner_i] = go_ptr[i];
            }
        } else {
            const float* go_ptr = static_cast<const float*>(grad.data());
            float*       gi_ptr = static_cast<float*>(grad_in.data());
            for (int64_t i = 0; i < n_out; ++i) {
                int64_t outer_i = i / inner_;
                int64_t inner_i = i % inner_;
                int64_t r       = idx_ptr[i];
                gi_ptr[outer_i * reduce_ * inner_ + r * inner_ + inner_i] = go_ptr[i];
            }
        }
    return grad_in;
}

} } }
