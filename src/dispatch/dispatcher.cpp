#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/runtime/cuda_context.h>

namespace orthon::dispatch {
namespace {
bool dtype_matches(const Constraints& c, const DispatchRequest& r) {
    return (c.dtypes & dtype_bit(r.dtype)) &&
           (!r.target_dtype || (c.target_dtypes & dtype_bit(*r.target_dtype)));
}
bool route_matches(const Constraints& c, const DispatchRequest& r) {
    return c.backend == r.source.type && c.target_backend ==
        (r.target ? std::optional<DeviceType>(r.target->type) : std::nullopt);
}
void validate_request(const Schema& s, const DispatchRequest& r) {
    // Defensive native metadata validation, never tensor shape semantics.
    Device(r.source.type, r.source.index);
    if (r.target) Device(r.target->type, r.target->index);
    dtype_bit(r.dtype);
    if (r.target_dtype) dtype_bit(*r.target_dtype);
    if ((s.signature == Signature::Transfer) != r.target.has_value() ||
        (s.signature == Signature::Cast) != r.target_dtype.has_value() ||
        r.rank < 0 || (r.layout != LayoutClass::Contiguous && r.layout != LayoutClass::Strided &&
                       r.layout != LayoutClass::Empty))
        throw std::logic_error("dispatch: malformed metadata request for " + s.name);
    if (s.composite) throw std::logic_error("dispatch: composite operator must execute its semantic composition: " + s.name);
}
}  // namespace

Dispatcher::Dispatcher(std::shared_ptr<const KernelRegistry> registry, Observer observer)
    : registry_(std::move(registry)), observer_(std::move(observer)) {
    if (!registry_ || !registry_->frozen())
        throw std::logic_error("dispatch: registry must be present and frozen");
}
void Dispatcher::check_dtype(const DispatchRequest& request) const {
    const auto& s = schema(request.op);
    validate_request(s, request);
    for (const auto& k : registry_->kernels_)
        if (k.op == request.op && route_matches(k.constraints, request) && dtype_matches(k.constraints, request))
            return;
    throw UnsupportedOperation(unsupported_diagnostic(s, request));
}
const KernelRegistration& Dispatcher::select(const DispatchRequest& request, Runtime& runtime) const {
    const auto& s = schema(request.op);
    validate_request(s, request);
    const KernelRegistration* selected = nullptr;
    bool dtype_supported = false;
    bool registered_operator = false;
    for (const auto& k : registry_->kernels_) {
        if (k.op == request.op) registered_operator = true;
        const auto& c = k.constraints;
        if (k.op != request.op || !route_matches(c, request) || !dtype_matches(c, request)) continue;
        dtype_supported = true;
        if (!(c.layouts & static_cast<unsigned>(request.layout))) continue;
        if (c.capability == Capability::CublasBFloat16 &&
            !runtime.cuda(request.source.index)->capabilities().cublas_bfloat16)
            throw UnsupportedOperation(s.name + ": CUDA backend lacks required cuBLAS bfloat16 capability on " +
                                       device_to_string(request.source));
        if (selected) throw std::logic_error("dispatch: corrupt ambiguous registry for " + s.name);
        selected = &k;
    }
    if (!selected) {
        if (!registered_operator && static_cast<unsigned>(s.id) < static_cast<unsigned>(OperatorId::Sqrt))
            throw UnsupportedOperation(s.name + ": unsupported backend/dtype/layout combination on " +
                                       device_to_string(request.source));
        if (dtype_supported && s.id != OperatorId::To && s.id != OperatorId::Cast)
            throw UnsupportedOperation(s.name + ": unsupported layout on " + device_to_string(request.source));
        throw UnsupportedOperation(unsupported_diagnostic(s, request));
    }
    return *selected;
}

std::string unsupported_diagnostic(const Schema& s, const DispatchRequest& r) {
    const bool cuda = r.source.is_cuda();
    if (s.id >= OperatorId::Sqrt) {
#ifndef WITH_CUDA
        if (cuda) {
            if (s.id == OperatorId::Conv2d || s.id == OperatorId::MaxPool2d) return s.name + ": CUDA is not available in this build";
            if (s.id == OperatorId::Sqrt) return "sqrt: CUDA support not compiled";
            if (s.id == OperatorId::Embedding || s.id == OperatorId::EmbeddingBackward) return s.name + ": CUDA support not compiled";
            if (s.id == OperatorId::AmpUnscale || s.id == OperatorId::AmpCopy || s.id == OperatorId::Finite || s.id == OperatorId::CopyRange) return s.name + ": CUDA is not available in this build";
            if (s.id == OperatorId::DropoutMask) return "dropout: unsupported device";
            if (s.id == OperatorId::Relu || s.id == OperatorId::Sigmoid || s.id == OperatorId::Tanh || s.id == OperatorId::Gelu || s.id == OperatorId::Exp || s.id == OperatorId::LeakyRelu) return s.name + ": only float32/float64 supported on CPU";
            return s.name + ": unsupported device";
        }
#endif
        if (s.id <= OperatorId::LeakyRelu || (s.id >= OperatorId::SigmoidBackward && s.id <= OperatorId::ReluBackward))
            return s.name + (cuda ? ": CUDA path requires float32, float16, or bfloat16" : ": only float32/float64 supported on CPU");
        if (s.id == OperatorId::Fill) return "fill_: only CPU supported";
        if (s.id == OperatorId::FullFill || s.id == OperatorId::ZeroFill) return "fill ops: only CPU supported";
        if (s.id == OperatorId::DropoutMask) return "dropout: unsupported device";
        if (s.id == OperatorId::Unbroadcast) return "unbroadcast: only float32/float64 supported on CPU";
        return s.name + ": unsupported device";
    }

#ifndef WITH_CUDA
    if (s.id == OperatorId::To && (cuda || (r.target && r.target->is_cuda())))
        return "to: CUDA is not available in this build (recompile with -DWITH_CUDA=ON)";
    if (cuda && s.id == OperatorId::Cast) return "cast: CUDA tensor requires a CUDA build";
    if (cuda && s.id == OperatorId::Reciprocal) return "reciprocal: CUDA support not compiled";
    if (cuda) return s.name + ": CUDA support not compiled";
#endif
    if (cuda && (s.id == OperatorId::To || s.id == OperatorId::Cast) &&
        r.layout == LayoutClass::Strided && r.dtype != DType::Float32 &&
        r.dtype != DType::Float16 && r.dtype != DType::BFloat16)
        return "cuda_make_contiguous: non-contiguous CUDA copies support only float32/float16/bfloat16";
    if (cuda && s.id == OperatorId::Cast && r.layout == LayoutClass::Strided && r.rank > 8)
        return "cuda_make_contiguous: tensor rank " + std::to_string(r.rank) + " exceeds kMaxDims=8";
    if (cuda && s.id == OperatorId::Cast)
        return "cuda_cast: unsupported dtype pair (only float32/float16/bfloat16 conversions)";
    if (s.id == OperatorId::Add || s.id == OperatorId::Mul)
        return cuda ? s.name + ": CUDA path requires float32, float16, or bfloat16"
                    : "binary ops: only float32/float64 supported on CPU";
    if (s.id == OperatorId::Neg || s.id == OperatorId::Reciprocal ||
        s.id == OperatorId::Sum || s.id == OperatorId::Mean ||
        s.id == OperatorId::Max || s.id == OperatorId::ArgMax || s.id == OperatorId::Matmul)
        return s.name + (cuda ? (s.id == OperatorId::Matmul
            ? ": CUDA path requires float32, float16, or bfloat16 (same dtype)"
            : ": CUDA path requires float32, float16, or bfloat16")
            : ": only float32/float64 supported on CPU");
    return s.name + ": unsupported backend/dtype/layout combination on " + device_to_string(r.source);
}
const Dispatcher& default_dispatcher() {
    static const Dispatcher dispatcher(build_builtin_registry());
    return dispatcher;
}
}  // namespace orthon::dispatch
