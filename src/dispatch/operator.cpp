#include <orthon/dispatch/operator.h>
#include <algorithm>
#include <stdexcept>

namespace orthon::dispatch {
DTypeMask dtype_bit(DType dtype) {
    auto n = static_cast<unsigned>(dtype);
    if (n >= 9) throw std::invalid_argument("dispatch: invalid dtype");
    return static_cast<DTypeMask>(1u << n);
}
DispatchRequest unary_request(OperatorId id, const Tensor& input) {
    return {id, input.device(), input.dtype(), input.numel() == 0 ? LayoutClass::Empty :
        input.is_contiguous() ? LayoutClass::Contiguous : LayoutClass::Strided, {}, {}, input.ndim()};
}
DispatchRequest binary_request(OperatorId id, const Tensor& a, const Tensor& b) {
    auto request = unary_request(id, a);
    if (b.numel() && !b.is_contiguous()) request.layout = LayoutClass::Strided;
    request.rank = std::max(a.ndim(), b.ndim());
    return request;
}
DispatchRequest cast_request(const Tensor& input, DType target) {
    auto request = unary_request(OperatorId::Cast, input);
    request.target_dtype = target;
    return request;
}
DispatchRequest transfer_request(const Tensor& input, Device target) {
    auto request = unary_request(OperatorId::To, input);
    request.target = target;
    return request;
}
Signature signature_of(const Kernel& kernel) {
    return std::visit([](const auto& k) { return k.signature; }, kernel);
}
unsigned signature_arity(Signature s) {
    switch (s) {
    case Signature::Binary: case Signature::Matmul: case Signature::Backward:
    case Signature::DimensionBackward: case Signature::PoolBackward:
    case Signature::TensorLoss: case Signature::Embedding: case Signature::EmbeddingBackward:
    case Signature::Inplace: case Signature::Copy: case Signature::TargetsBackward: return 2;
    case Signature::Ternary: case Signature::Convolution: case Signature::TensorLossBackward:
    case Signature::MaxScatter: return 3;
    case Signature::ConvolutionBackward: return 4;
    default: return 1;
    }
}
}  // namespace orthon::dispatch
