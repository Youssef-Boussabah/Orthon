#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/core/float_formats.h>
#include <orthon/core/tensor_impl.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace orthon {

static void require_cpu(const Device& device) {
    if (!device.is_cpu())
        throw std::runtime_error("fill ops: only CPU supported");
}

// The float16/bfloat16 conversions come from <orthon/core/float_formats.h>,
// which is the one place the rounding rule is defined. A local copy here is
// what let the fill path disagree with the cast path on subnormals and on
// whether normals round or truncate.

Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    require_cpu(device);
    return Tensor(std::make_shared<TensorImpl>(shape, dtype, device));
}

Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t = empty(shape, dtype, device);
    std::memset(t.data(), 0, static_cast<size_t>(t.numel()) * dtype_size(dtype));
    return t;
}

Tensor ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    return full(shape, 1.0, dtype, device);
}

// `value` arrives as a double and is narrowed ONCE, at the point where the
// destination dtype is known. Narrowing earlier would round every value
// through float32 before it reached a float64 destination. See the header for
// the per-dtype contract.
Tensor full(const std::vector<int64_t>& shape, double value, DType dtype, Device device) {
    Tensor t = empty(shape, dtype, device);
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
    return t;
}

void fill_(Tensor& t, float value) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error("fill_: only float32 supported");
    if (!t.device().is_cpu())
        throw std::runtime_error("fill_: only CPU supported");

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

    // Exactly one bump per successful logical fill. This is the only
    // bump on the fill path - partial __setitem__ routes through fill_ and
    // must not double-count. A zero-element fill still bumps: the counter
    // tracks mutation operations, not bytes written.
    t.impl()->storage()->bump_version();
}

}  // namespace orthon
