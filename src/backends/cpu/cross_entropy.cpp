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
Tensor cpu_cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets) {
    float loss_val;
        const int64_t B   = logits.shape()[0];
        const int64_t C   = logits.shape()[1];
        const float*  src = static_cast<const float*>(logits.data());

        double total = 0.0;
        for (int64_t i = 0; i < B; ++i) {
            const float* row  = src + i * C;
            float        rmax = *std::max_element(row, row + C);

            float sum_exp = 0.0f;
            for (int64_t j = 0; j < C; ++j)
                sum_exp += std::exp(row[j] - rmax);

            total += -static_cast<double>(row[targets[i]]) + rmax +
                     std::log(static_cast<double>(sum_exp));
        }
        loss_val = static_cast<float>(total / B);
    return full({1}, loss_val);
}
Tensor cpu_cross_entropy_backward(const Tensor& logits, const std::vector<int64_t>& targets_, const Tensor& grad_output) {
        const float g = static_cast<const float*>(grad_output.data())[0];

        const int64_t B   = logits.shape()[0];
        const int64_t C   = logits.shape()[1];
        const float inv_B = 1.0f / static_cast<float>(B);

        // Guard direct backward calls too: gradient construction and mutation must
        // not create higher-order autograd nodes.
        Tensor probs = cpu_softmax(logits, 1);
        float* p = static_cast<float*>(probs.data());

        // grad[i,j] = g * (softmax[i,j] - one_hot[i,j]) / B
        for (int64_t i = 0; i < B; ++i) {
            for (int64_t j = 0; j < C; ++j) {
                const float oh = (j == targets_[i]) ? 1.0f : 0.0f;
                p[i * C + j] = g * (p[i * C + j] - oh) * inv_B;
            }
        }

    return probs;
}

} } }
