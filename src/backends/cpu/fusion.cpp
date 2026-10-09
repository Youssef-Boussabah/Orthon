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
Tensor cpu_fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    const int64_t m = x.shape()[0];
    const int64_t k = x.shape()[1];
    const int64_t n = weight.shape()[1];

    const int64_t sx0 = x.strides()[0], sx1 = x.strides()[1];
    const int64_t sw0 = weight.strides()[0], sw1 = weight.strides()[1];
    const int64_t sb0 = bias.strides()[0];

    const float* px = static_cast<const float*>(x.data());
    const float* pw = static_cast<const float*>(weight.data());
    const float* pb = static_cast<const float*>(bias.data());

    Tensor out = empty({m, n}, x.dtype(), x.device());
    float* po = static_cast<float*>(out.data());

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = pb[j * sb0];
            for (int64_t p = 0; p < k; ++p)
                acc += px[i * sx0 + p * sx1] * pw[p * sw0 + j * sw1];
            po[i * n + j] = acc > 0.0f ? acc : 0.0f;
        }
    }

    return out;
}

} } }
