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
static void dim_factors(const Tensor& t,
                        int64_t dim,
                        int64_t& outer,
                        int64_t& reduce,
                        int64_t& inner) {
    const auto& sh = t.shape();
    outer = 1;
    for (int64_t i = 0; i < dim; ++i)
        outer *= sh[static_cast<size_t>(i)];
    reduce = sh[static_cast<size_t>(dim)];
    inner = 1;
    for (int64_t i = dim + 1; i < t.ndim(); ++i)
        inner *= sh[static_cast<size_t>(i)];
}
Tensor cpu_softmax(const Tensor& t, int64_t) {
        const int64_t M = t.shape()[0];
        const int64_t N = t.shape()[1];
        Tensor out = empty(t.shape(), t.dtype(), t.device());

        const float* src = static_cast<const float*>(t.data());
        float*       dst = static_cast<float*>(out.data());

        for (int64_t i = 0; i < M; ++i) {
            const float* row_src = src + i * N;
            float*       row_dst = dst + i * N;

            float rmax = *std::max_element(row_src, row_src + N);

            float rsum = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                row_dst[j] = std::exp(row_src[j] - rmax);
                rsum += row_dst[j];
            }
            for (int64_t j = 0; j < N; ++j)
                row_dst[j] /= rsum;
        }

        return out;

}
Tensor cpu_log_softmax(const Tensor& t, int64_t resolved) {
    int64_t outer, reduce, inner;
    dim_factors(t, resolved, outer, reduce, inner);
    Tensor out = empty(t.shape(), t.dtype(), t.device());

    const float* src = static_cast<const float*>(t.data());
    float* dst = static_cast<float*>(out.data());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            const int64_t base = o * reduce * inner + in;
            float vmax = -std::numeric_limits<float>::infinity();
            for (int64_t r = 0; r < reduce; ++r)
                vmax = std::max(vmax, src[base + r * inner]);

            double sum_exp = 0.0;
            for (int64_t r = 0; r < reduce; ++r)
                sum_exp += std::exp(static_cast<double>(src[base + r * inner] - vmax));
            const float log_sum = static_cast<float>(std::log(sum_exp));

            for (int64_t r = 0; r < reduce; ++r)
                dst[base + r * inner] = (src[base + r * inner] - vmax) - log_sum;
        }
    }

    return out;
}
Tensor cpu_log_softmax_backward(const Tensor& input, const Tensor& grad_output, int64_t resolved) {
    int64_t outer, reduce, inner;
    dim_factors(input, resolved, outer, reduce, inner);
    Tensor grad = empty(input.shape(), DType::Float32, input.device());

    const float* src = static_cast<const float*>(input.data());
    const float* gout = static_cast<const float*>(grad_output.data());
    float* dst = static_cast<float*>(grad.data());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            const int64_t base = o * reduce * inner + in;
            float vmax = -std::numeric_limits<float>::infinity();
            float grad_sum = 0.0f;
            for (int64_t r = 0; r < reduce; ++r) {
                vmax = std::max(vmax, src[base + r * inner]);
                grad_sum += gout[base + r * inner];
            }

            double sum_exp = 0.0;
            for (int64_t r = 0; r < reduce; ++r)
                sum_exp += std::exp(static_cast<double>(src[base + r * inner] - vmax));

            for (int64_t r = 0; r < reduce; ++r) {
                const int64_t idx = base + r * inner;
                const float prob = static_cast<float>(
                    std::exp(static_cast<double>(src[idx] - vmax)) / sum_exp);
                dst[idx] = gout[idx] - prob * grad_sum;
            }
        }
    }

    return grad;

}

} } }
