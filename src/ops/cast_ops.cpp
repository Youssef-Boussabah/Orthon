#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/core/float_formats.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/autograd/cast_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/tensor_impl.h>
#include <memory>
#include <cstdint>
#include <cstring>
#include <stdexcept>


namespace orthon {

// The four bit-level conversion helpers come from
// <orthon/core/float_formats.h>. Defining them per translation unit is what
// let the three low-precision entry points - fill, cast and from_numpy -
// disagree on rounding and subnormals; there is now exactly one definition.

// Universal element read/write helpers for contiguous buffers.


// Public cast operation.

Tensor cast(const Tensor& src, DType dst_dtype) {
    // Same-dtype cast reuses a contiguous Tensor or materializes a view, keeping
    // its graph without a CastNode. Call to() for an independent physical copy.
    if (src.dtype() == dst_dtype)
        return contiguous(src);

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<CastKernel>(
        cast_request(src, dst_dtype), default_runtime());
    Tensor dst = kernel.call(src, dst_dtype);

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
