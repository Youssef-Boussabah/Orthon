#include <orthon/dispatch/operator.h>
#include <orthon/planning/matmul_planner.h>
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
static Tensor matmul_values(const Tensor& a, const Tensor& b) {
    const T* pa = static_cast<const T*>(a.data());
    const T* pb = static_cast<const T*>(b.data());

    std::vector<int64_t> out_shape;
    if (a.ndim() == 2) {
        out_shape = {a.shape()[0], b.shape()[1]};
    } else {
        out_shape = {a.shape()[0], a.shape()[1], b.shape()[2]};
    }
    Tensor out = empty(out_shape, a.dtype(), a.device());
    T* dst = static_cast<T*>(out.data());

    if (a.ndim() == 2) {
        const int64_t m   = a.shape()[0];
        const int64_t k   = a.shape()[1];
        const int64_t n   = b.shape()[1];
        const int64_t sa0 = a.strides()[0], sa1 = a.strides()[1];
        const int64_t sb0 = b.strides()[0], sb1 = b.strides()[1];
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j) {
                T acc = T{0};
                for (int64_t p = 0; p < k; ++p)
                    acc += pa[i * sa0 + p * sa1] * pb[p * sb0 + j * sb1];
                dst[i * n + j] = acc;
            }
    } else {
        const int64_t batch = a.shape()[0];
        const int64_t m     = a.shape()[1];
        const int64_t k     = a.shape()[2];
        const int64_t n     = b.shape()[2];
        const int64_t sa0 = a.strides()[0], sa1 = a.strides()[1], sa2 = a.strides()[2];
        const int64_t sb0 = b.strides()[0], sb1 = b.strides()[1], sb2 = b.strides()[2];
        for (int64_t ba = 0; ba < batch; ++ba)
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j) {
                    T acc = T{0};
                    for (int64_t p = 0; p < k; ++p)
                        acc += pa[ba * sa0 + i * sa1 + p * sa2] *
                               pb[ba * sb0 + p * sb1 + j * sb2];
                    dst[ba * m * n + i * n + j] = acc;
                }
    }
    return out;
}

Tensor cpu_matmul(const Tensor& a, const Tensor& b, const MatmulPlan& plan) {
    if (plan.algorithm != MatmulAlgorithm::CpuReference || !a.device().is_cpu() ||
        (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) ||
        !matmul_plan_matches(a, b, plan))
        throw std::invalid_argument("cpu matmul: plan/operand mismatch");
    return a.dtype() == DType::Float32 ? matmul_values<float>(a, b)
                                      : matmul_values<double>(a, b);
}

} } }
