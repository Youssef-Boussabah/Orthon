#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/add_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/mul_node.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace orthon {

static std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) r += ", "; r += std::to_string(s[i]); }
    return r + "]";
}

void check_same_dtype(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype())
        throw std::runtime_error("dtype mismatch");
}

void check_same_device(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device())
        throw std::runtime_error("device mismatch");
}


// Preserve the legacy relationship-check order, including mixed-device
// errors. This selects validation order only; it never selects execution.
static void validate_binary_relationships(const Tensor& a, const Tensor& b) {
#ifdef WITH_CUDA
    if (a.device().is_cuda() || b.device().is_cuda()) {
        check_same_device(a, b);
        check_same_dtype(a, b);
        return;
    }
#endif
    check_same_dtype(a, b);
    check_same_device(a, b);
}

static std::vector<int64_t> validated_binary_shape(const Tensor& a, const Tensor& b) {
    try {
        return broadcast_shape(a.shape(), b.shape());
    } catch (const std::runtime_error&) {
#ifdef WITH_CUDA
        // Preserve the legacy CUDA broadcast diagnostic, including mul's
        // cuda_add compatibility prefix. This changes formatting, not dispatch.
        if (a.device().is_cuda())
            throw std::runtime_error("cuda_add: cannot broadcast shapes " +
                                     shape_str(a.shape()) + " and " + shape_str(b.shape()));
#endif
        throw;
    }
}

Tensor dispatch::binary_semantic(OperatorId id, const Tensor& a, const Tensor& b,
                                const Dispatcher& dispatcher, Runtime& runtime) {
    validate_binary_relationships(a, b);
    const auto& kernel = dispatcher.resolve<BinaryKernel>(binary_request(id, a, b), runtime);
    // Reject unsupported backend/dtype before validating broadcast geometry.
    const auto shape = validated_binary_shape(a, b);
    Tensor out = kernel.call(a, b, shape);
    const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    out.requires_grad_(req);
    out.set_is_leaf(!req);
    if (req) {
        if (id == OperatorId::Add) out.set_grad_fn(std::make_shared<AddNode>(a, b));
        else out.set_grad_fn(std::make_shared<MulNode>(a, b));
    }
    return out;
}
Tensor add(const Tensor& a, const Tensor& b) {
    return dispatch::binary_semantic(dispatch::OperatorId::Add, a, b,
                                    dispatch::default_dispatcher(), default_runtime());
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return dispatch::binary_semantic(dispatch::OperatorId::Mul, a, b,
                                    dispatch::default_dispatcher(), default_runtime());
}

// Subtraction and division, by composition. See binary_ops.h for why no new
// kernel is introduced. Each inherits its validation from the op it is built
// on rather than restating it: neg()/reciprocal_t() run their own dtype and
// device dispatch first, then add()/mul() apply the shared broadcast, dtype
// and device checks, so a mismatched pair fails with the same message the
// underlying op produces.
Tensor sub(const Tensor& a, const Tensor& b) {
    dispatch::default_dispatcher().schema(dispatch::OperatorId::Sub);
    return add(a, neg(b));
}

Tensor div(const Tensor& a, const Tensor& b) {
    dispatch::default_dispatcher().schema(dispatch::OperatorId::Div);
    return mul(a, reciprocal_t(b));
}

// The view normalisation above deliberately does not extend to the in-place
// ops below. An out-of-place op may materialise its operands because the
// materialised copy is only ever READ; an in-place op writes to its
// destination, and writing to a temporary contiguous copy would silently
// discard the mutation the caller asked for. Supporting a strided destination
// means a stride-aware in-place kernel, which is a kernel change, not a
// normalisation. The explicit contiguous-destination requirement therefore
// stands, and is pinned by test_runtime_consistency.py.
void mul_(Tensor& a, float alpha) {
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)
        throw std::runtime_error("mul_: only float32/float64 supported");
    if (!a.is_contiguous())
        throw std::runtime_error("mul_: tensor must be contiguous");

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ScaleKernel>(unary_request(OperatorId::MulInplace, a), default_runtime());
    kernel.call(a, alpha);
    a.impl()->buffer()->bump_version();
}

void add_(Tensor& a, const Tensor& b, float alpha) {
    if (a.shape() != b.shape())
        throw std::runtime_error("add_: shape mismatch");
    if (a.dtype() != b.dtype())
        throw std::runtime_error("add_: dtype mismatch");
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)
        throw std::runtime_error("add_: only float32/float64 supported");
    if (a.device() != b.device())
        throw std::runtime_error("add_: device mismatch");
    if (!a.is_contiguous() || !b.is_contiguous())
        throw std::runtime_error("add_: tensors must be contiguous");

    if (a.numel() != 0 && a.impl()->buffer() == b.impl()->buffer()) {
        const auto dst = a.impl()->layout().storage_range();
        const auto src = b.impl()->layout().storage_range();
        if (dst != src && dst.first < src.second && src.first < dst.second)
            throw std::runtime_error("add_: partially overlapping tensor ranges are not supported");
    }

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<InplaceKernel>(unary_request(OperatorId::AddInplace, a), default_runtime());
    kernel.call(a, b, alpha);
    a.impl()->buffer()->bump_version();
}

}  // namespace orthon
