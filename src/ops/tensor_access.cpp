#include <orthon/ops/tensor_access.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <stdexcept>
namespace orthon {
bool isfinite_all_values(const Tensor& t) {
    const auto dt = t.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 && dt != DType::Float16 && dt != DType::BFloat16) return true;
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<FiniteKernel>(unary_request(OperatorId::Finite, t), default_runtime());
    return kernel.call(t);
}
std::pair<Tensor, bool> amp_unscale_values(const Tensor& t, float scale) {
    if (!t.device().is_cuda()) throw std::runtime_error("_amp_unscale_to_float32_and_check: CUDA tensor required");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<UnscaleKernel>(unary_request(OperatorId::AmpUnscale, t), default_runtime());
    return kernel.call(t, scale);
}
void amp_copy_values(Tensor& dst, const Tensor& src) {
    if (dst.device() != src.device())
        throw std::runtime_error("_amp_copy_master_to_model_: exact device mismatch");
    if (dst.shape() != src.shape())
        throw std::runtime_error("_amp_copy_master_to_model_: shape mismatch");
    if (src.dtype() != DType::Float32)
        throw std::runtime_error("_amp_copy_master_to_model_: source must be float32");
    if (dst.dtype() != DType::Float32 && dst.dtype() != DType::Float16 && dst.dtype() != DType::BFloat16)
        throw std::runtime_error("_amp_copy_master_to_model_: destination must be float32/float16/bfloat16");
    if (!dst.is_contiguous() || !src.is_contiguous())
        throw std::runtime_error("_amp_copy_master_to_model_: tensors must be contiguous");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<CopyKernel>(unary_request(OperatorId::AmpCopy, dst), default_runtime());
    kernel.call(dst, src, {});
    dst.impl()->buffer()->bump_version();
}
void copy_range_values(Tensor& dst, const Tensor& src, const dispatch::CopyArgs& args) {
    if (!dst.is_contiguous() || !src.is_contiguous() ||
        dst.dtype() != src.dtype() || dst.device() != src.device())
        throw std::invalid_argument("copy_range: tensors must have matching dtype/device and be contiguous");
    if (args.dst_offset < 0 || args.src_offset < 0 || args.count < 0 ||
        args.dst_offset > dst.numel() || args.src_offset > src.numel() ||
        args.count > dst.numel() - args.dst_offset ||
        args.count > src.numel() - args.src_offset)
        throw std::out_of_range("copy_range: range out of bounds");
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<CopyKernel>(unary_request(OperatorId::CopyRange, dst), default_runtime());
    kernel.call(dst, src, args);
}
float scalar_read_value(const Tensor& t, int64_t offset, const char* op) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ScalarReadKernel>(unary_request(OperatorId::ScalarRead, t), default_runtime());
    return kernel.call(t, offset, op);
}
void scalar_write_value(const Tensor& t, int64_t offset, float value, const char* op) {
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ScalarWriteKernel>(unary_request(OperatorId::ScalarWrite, t), default_runtime());
    kernel.call(t, offset, value, op);
}
}
