#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/fill_ops.h>
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
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<FillKernel>(unary_request(OperatorId::ZeroFill, t), default_runtime());
    kernel.call(t, 0.0);
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
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<FillKernel>(unary_request(OperatorId::FullFill, t), default_runtime());
    kernel.call(t, value);
    return t;
}

void fill_(Tensor& t, float value) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error("fill_: only float32 supported");
    if (!t.device().is_cpu())
        throw std::runtime_error("fill_: only CPU supported");

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<FillKernel>(unary_request(OperatorId::Fill, t), default_runtime());
    kernel.call(t, value);
    t.impl()->buffer()->bump_version();
}

}  // namespace orthon
