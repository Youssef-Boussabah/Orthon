#include <orthon/core/host_copy.h>
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
static double read_elem(const void* data, DType dt, int64_t i) {
    switch (dt) {
        case DType::Float32:  return static_cast<double>(static_cast<const float*>   (data)[i]);
        case DType::Float64:  return                     static_cast<const double*>  (data)[i];
        case DType::Float16:  return static_cast<double>(f16_bits_to_f32(static_cast<const uint16_t*>(data)[i]));
        case DType::BFloat16: return static_cast<double>(bf16_bits_to_f32(static_cast<const uint16_t*>(data)[i]));
        case DType::Int32:    return static_cast<double>(static_cast<const int32_t*> (data)[i]);
        case DType::Int64:    return static_cast<double>(static_cast<const int64_t*> (data)[i]);
        case DType::UInt8:    return static_cast<double>(static_cast<const uint8_t*> (data)[i]);
        case DType::Int8:     return static_cast<double>(static_cast<const int8_t*>  (data)[i]);
        case DType::Bool:     return static_cast<double>(static_cast<const uint8_t*> (data)[i] ? 1 : 0);
    }
    return 0.0;
}

static void write_elem(void* data, DType dt, int64_t i, double v) {
    switch (dt) {
        case DType::Float32:  static_cast<float*>   (data)[i] = static_cast<float>(v);           break;
        case DType::Float64:  static_cast<double*>  (data)[i] = v;                               break;
        case DType::Float16:  static_cast<uint16_t*>(data)[i] = f32_to_f16_bits(static_cast<float>(v)); break;
        case DType::BFloat16: static_cast<uint16_t*>(data)[i] = f32_to_bf16_bits(static_cast<float>(v)); break;
        case DType::Int32:    static_cast<int32_t*> (data)[i] = static_cast<int32_t>(v);         break;
        case DType::Int64:    static_cast<int64_t*> (data)[i] = static_cast<int64_t>(v);         break;
        case DType::UInt8:    static_cast<uint8_t*> (data)[i] = static_cast<uint8_t>(v);         break;
        case DType::Int8:     static_cast<int8_t*>  (data)[i] = static_cast<int8_t>(v);          break;
        case DType::Bool:     static_cast<uint8_t*> (data)[i] = (v != 0.0) ? 1u : 0u;            break;
    }
}

Tensor cpu_cast(const Tensor& src, DType dst_dtype) {
    const Tensor s = src.is_contiguous() ? src : make_contiguous_copy(src);
    const int64_t n = s.numel();
    Tensor out(std::make_shared<TensorImpl>(s.shape(), dst_dtype, s.device()));
    const void* src_data = s.data();
    void* dst_data = out.data();
    for (int64_t i = 0; i < n; ++i)
        write_elem(dst_data, dst_dtype, i, read_elem(src_data, s.dtype(), i));
    return out;
}

} } }
