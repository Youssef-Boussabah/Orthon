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
bool cpu_isfinite_all(const Tensor& t) {
    const auto dt = t.dtype();
    Tensor materialized = t;
    auto finite_at = [&](int64_t off) -> bool {
        if (dt == DType::Float32)
            return std::isfinite(static_cast<const float*>(materialized.data())[off]);
        if (dt == DType::Float64)
            return std::isfinite(static_cast<const double*>(materialized.data())[off]);
        if (dt == DType::Float16)
            return std::isfinite(f16_bits_to_f32(
                static_cast<const uint16_t*>(materialized.data())[off]));
        return std::isfinite(bf16_bits_to_f32(
            static_cast<const uint16_t*>(materialized.data())[off]));
    };

    const int64_t n = materialized.numel();
    if (materialized.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) {
            if (!finite_at(i)) return false;
        }
        return true;
    }

    const auto& sh = materialized.shape();
    const auto& st = materialized.strides();
    const int ndim = static_cast<int>(sh.size());
    if (ndim == 0) return finite_at(0);
    std::vector<int64_t> idx(ndim, 0);
    for (int64_t elem = 0; elem < n; ++elem) {
        int64_t off = 0;
        for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
        if (!finite_at(off)) return false;
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
    return true;
}
void cpu_copy_range(Tensor& dst, const Tensor& src, const CopyArgs& a) {
    if (a.count > 0) {
        const auto esz = dtype_size(dst.dtype());
        std::memcpy(static_cast<char*>(dst.data()) + a.dst_offset * esz, static_cast<const char*>(src.data()) + a.src_offset * esz, a.count * esz);
    }
}
void cpu_amp_copy(Tensor& dst, const Tensor& src, const CopyArgs&) {
    const auto* values = static_cast<const float*>(src.data());
    if (dst.dtype() == DType::Float32) {
        // Empty tensors need no memory access; identical storage is a no-op.
        if (dst.numel() > 0)
            std::memmove(dst.data(), src.data(), dst.numel() * sizeof(float));
        return;
    }
    auto* bits = static_cast<uint16_t*>(dst.data());
    for (int64_t i = 0; i < dst.numel(); ++i)
        bits[i] = dst.dtype() == DType::Float16 ? f32_to_f16_bits(values[i])
                                               : f32_to_bf16_bits(values[i]);
}
float cpu_scalar_read(const Tensor& t, int64_t offset, const char*) { return static_cast<const float*>(t.data())[offset]; }
void cpu_scalar_write(const Tensor& t, int64_t offset, float value, const char*) { static_cast<float*>(t.data())[offset] = value; }

} } }
