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
void cpu_full_fill(Tensor& t, double value) {
    const auto dtype = t.dtype();
    const int64_t n = t.numel();
    switch (dtype) {
        case DType::Float32: {
            auto* p = static_cast<float*>(t.data());
            const float v = static_cast<float>(value);
            for (int64_t i = 0; i < n; ++i) p[i] = v;
            break;
        }
        case DType::Float64: {
            // The whole point of the widening: no float32 round-trip.
            auto* p = static_cast<double*>(t.data());
            for (int64_t i = 0; i < n; ++i) p[i] = value;
            break;
        }
        case DType::Float16: {
            auto* p    = static_cast<uint16_t*>(t.data());
            const uint16_t bits = f32_to_f16_bits(static_cast<float>(value));
            for (int64_t i = 0; i < n; ++i) p[i] = bits;
            break;
        }
        case DType::BFloat16: {
            auto* p    = static_cast<uint16_t*>(t.data());
            const uint16_t bits = f32_to_bf16_bits(static_cast<float>(value));
            for (int64_t i = 0; i < n; ++i) p[i] = bits;
            break;
        }
        case DType::Int64: {
            auto* p = static_cast<int64_t*>(t.data());
            for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int64_t>(value);
            break;
        }
        case DType::Int32: {
            auto* p = static_cast<int32_t*>(t.data());
            for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int32_t>(value);
            break;
        }
        case DType::UInt8: {
            auto* p = static_cast<uint8_t*>(t.data());
            for (int64_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>(value);
            break;
        }
        case DType::Int8: {
            auto* p = static_cast<int8_t*>(t.data());
            for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int8_t>(value);
            break;
        }
        case DType::Bool: {
            auto* p = static_cast<uint8_t*>(t.data());
            const uint8_t bv = (value != 0.0) ? 1u : 0u;
            for (int64_t i = 0; i < n; ++i) p[i] = bv;
            break;
        }
    }
}
void cpu_zero_fill(Tensor& t, double) {
    if (t.numel() == 0) return;
    std::memset(t.data(), 0, static_cast<size_t>(t.numel()) * dtype_size(t.dtype()));
}
void cpu_fill(Tensor& t, double fill_value) {
    const float value = static_cast<float>(fill_value);
    float* base = static_cast<float*>(t.data());
    const int64_t n = t.numel();

    if (t.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) base[i] = value;
    } else if (t.ndim() == 0) {
        *base = value;
    } else {
        const auto& shape   = t.shape();
        const auto& strides = t.strides();
        const int ndim = static_cast<int>(shape.size());

        std::vector<int64_t> idx(ndim, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) offset += idx[d] * strides[d];
            base[offset] = value;

            for (int d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
    }

}

} } }
