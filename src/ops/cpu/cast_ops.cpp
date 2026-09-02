#include <orthon/ops/cpu/cast_ops.h>
#include <orthon/core/float_formats.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/autograd/cast_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/tensor_impl.h>
#include <memory>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/cast_kernels.h>
#endif

namespace orthon {

// The four bit-level conversion helpers come from
// <orthon/core/float_formats.h>. Defining them per translation unit is what
// let the three low-precision entry points - fill, cast and from_numpy -
// disagree on rounding and subnormals; there is now exactly one definition.

// Universal element read/write helpers for contiguous buffers.

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

// Public cast operation.

Tensor cast(const Tensor& src, DType dst_dtype) {
    // Same-dtype cast is a FRAMEWORK-WIDE no-op / contiguity normalisation,
    // decided BEFORE any CPU/CUDA dispatch so the two devices cannot drift
    // into separate contracts for an operation that converts nothing.
    //
    // It never allocates for an already-contiguous tensor, never drops graph
    // metadata, and never attaches a CastNode. Callers that want an
    // independent physical copy have the public API for it - x.to(x.device()),
    // which is documented to copy and is differentiable.
    //
    // No runtime call site depends on same-dtype cast allocating.
    // Module.cast/half/bfloat16 and load_state_dict get their independence
    // from an unconditional to(), the AMP master-weight and GradScaler paths
    // guard every cast with an explicit dtype-differs check, and cast() has no
    // internal native caller at all.
    if (src.dtype() == dst_dtype)
        return contiguous(src);

    Tensor dst = [&]() -> Tensor {
        if (src.device().is_cuda()) {
#ifdef WITH_CUDA
            // cuda_cast stays contiguous-only by design; the
            // strided case is handled by materialising the kernel operand,
            // never by teaching the kernel about strides. CastNode below
            // attaches to `src`, so a transpose/slice view's gradient still
            // routes back through its own view node in logical coordinates.
            return cuda_cast(kernel_operand(src), dst_dtype);
#else
            throw std::runtime_error("cast: CUDA tensor requires a CUDA build");
#endif
        }

        // Ensure source is contiguous before reading linearly. Explicit
        // no-grad: this is an internal materialisation, not part of the
        // public graph - CastNode below attaches to `src` itself, and the
        // conversion already produces the gradient in logical order.
        const Tensor s = kernel_operand(src);
        const int64_t n = s.numel();

        Tensor out(std::make_shared<TensorImpl>(s.shape(), dst_dtype, s.device()));
        const void* src_data = s.data();
        void*       dst_data = out.data();

        for (int64_t i = 0; i < n; ++i)
            write_elem(dst_data, dst_dtype, i, read_elem(src_data, s.dtype(), i));

        return out;
    }();

    // A REAL floating->floating conversion joins the graph. Anything touching
    // an integer/bool dtype is intentionally non-differentiable (no
    // straight-through estimator). The src.dtype() != dst_dtype term is
    // redundant - same-dtype returned above on both devices - and is kept as
    // an explicit statement that CastNode is only ever a CONVERSION node.
    if (src.dtype() != dst_dtype &&
        dtype_is_floating(src.dtype()) && dtype_is_floating(dst_dtype) &&
        is_grad_enabled() && src.requires_grad()) {
        dst.requires_grad_(true);
        dst.set_is_leaf(false);
        dst.set_grad_fn(std::make_shared<CastNode>(src));
    }
    return dst;
}

}  // namespace orthon
