#include <orthon/backends/cuda/kernels.h>
#include <orthon/core/host_copy.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/runtime.h>
#include <orthon/core/execution_copy.h>
#include <orthon/core/tensor_impl.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <orthon/backends/cuda/binary_kernels.h>
#include <orthon/backends/cuda/cast_kernels.h>
#include <orthon/backends/cuda/conv2d_kernels.h>
#include <orthon/backends/cuda/cross_entropy_kernels.h>
#include <orthon/backends/cuda/embedding_kernels.h>
#include <orthon/backends/cuda/fusion_kernels.h>
#include <orthon/backends/cuda/matmul_kernels.h>
#include <orthon/backends/cuda/maxpool2d_kernels.h>
#include <orthon/backends/cuda/nll_loss_kernels.h>
#include <orthon/backends/cuda/reduce_kernels.h>
#include <orthon/backends/cuda/softmax_kernels.h>
#include <orthon/backends/cuda/unary_kernels.h>
namespace orthon { namespace backend { namespace cuda {
using namespace dispatch;
Tensor host_to_cuda(const Tensor& self, Device target) {
    Tensor src_c = self.is_contiguous() ? self : make_contiguous_copy(self);
    const size_t nbytes = static_cast<size_t>(src_c.numel()) * dtype_size(src_c.dtype());
    Tensor out(std::make_shared<TensorImpl>(src_c.shape(), src_c.dtype(), target));
    DeviceGuard guard(target);
    auto context = default_runtime().cuda(target.index);
    context->copy_h2d_snapshot(out.data(), src_c.data(), nbytes);
    return out;
}
Tensor cuda_to_host(const Tensor& self, Device target) {
    DeviceGuard guard(self.device());
    auto context = default_runtime().cuda(self.device().index);
    Tensor src_c = self.is_contiguous() ? self : cuda_make_contiguous(self);
    const size_t nbytes = static_cast<size_t>(src_c.numel()) * dtype_size(src_c.dtype());
    Tensor out(std::make_shared<TensorImpl>(src_c.shape(), src_c.dtype(), target));
    context->copy_d2h_blocking(out.data(), src_c.data(), nbytes);
    return out;
}
Tensor cuda_to_cuda(const Tensor& self, Device target) {
    // Exact-ordinal copy policy within an already selected CUDA/CUDA route.
    // Same-device to() always produces independent storage.
    if (self.device() == target) {
        DeviceGuard guard(self.device());
    auto context = default_runtime().cuda(self.device().index);
        return cuda_make_contiguous(self);
    }
    Tensor src_c = [&] {
        DeviceGuard guard(self.device());
    auto context = default_runtime().cuda(self.device().index);
        return self.is_contiguous() ? self : cuda_make_contiguous(self);
    }();
    const size_t nbytes = static_cast<size_t>(src_c.numel()) * dtype_size(src_c.dtype());
    Tensor out(std::make_shared<TensorImpl>(src_c.shape(), src_c.dtype(), target));
    DeviceGuard guard(target);
    auto context = default_runtime().cuda(target.index);
    context->peer_copy_blocking(out.data(), *default_runtime().cuda(src_c.device().index), src_c.data(), nbytes);
    return out;
}
Tensor cuda_add_adapter(const Tensor& a, const Tensor& b, const std::vector<int64_t>& shape) { return cuda_add(execution_operand(a), execution_operand(b), shape); }
Tensor cuda_mul_adapter(const Tensor& a, const Tensor& b, const std::vector<int64_t>& shape) { return cuda_mul(execution_operand(a), execution_operand(b), shape); }
Tensor cuda_cast_adapter(const Tensor& src, DType dst_dtype) { return cuda_cast(execution_operand(src), dst_dtype); }
Tensor cuda_matmul_adapter(const Tensor& a, const Tensor& b, const MatmulPlan& plan) {
    Tensor a_kernel = execution_operand(a);
    Tensor b_kernel = execution_operand(b);
    return cuda_matmul_execute(a_kernel, b_kernel, plan);
}
Tensor cuda_neg_adapter(const Tensor& t) { return cuda_neg(execution_operand(t)); }
Tensor cuda_reciprocal_adapter(const Tensor& t) { return cuda_reciprocal(execution_operand(t)); }
Tensor cuda_relu_adapter(const Tensor& t) { return cuda_relu(execution_operand(t)); }
Tensor cuda_sqrt_adapter(const Tensor& t) { return cuda_sqrt(execution_operand(t)); }
Tensor cuda_sigmoid_adapter(const Tensor& t) { return cuda_sigmoid(execution_operand(t)); }
Tensor cuda_tanh_adapter(const Tensor& t) { return cuda_tanh(execution_operand(t)); }
Tensor cuda_gelu_adapter(const Tensor& t) { return cuda_gelu(execution_operand(t)); }
Tensor cuda_exp_adapter(const Tensor& t) { return cuda_exp(execution_operand(t)); }
Tensor cuda_leaky_relu_adapter(const Tensor& t, float alpha, uint64_t) { return cuda_leaky_relu(execution_operand(t), alpha); }
Tensor cuda_dropout_mask_adapter(const Tensor& t, float p, uint64_t seed) { return cuda_dropout_mask(t, p, seed); }
Tensor cuda_sigmoid_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_sigmoid_backward(execution_operand(input), execution_operand(grad_output)); }
Tensor cuda_tanh_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_tanh_backward(execution_operand(input), execution_operand(grad_output)); }
Tensor cuda_leaky_relu_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_leaky_relu_backward(execution_operand(input), execution_operand(grad_output), alpha); }
Tensor cuda_gelu_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_gelu_backward(execution_operand(input), execution_operand(grad_output)); }
Tensor cuda_exp_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_exp_backward(execution_operand(input), execution_operand(grad_output)); }
Tensor cuda_neg_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_neg(execution_operand(grad_output)); }
Tensor cuda_sqrt_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_sqrt_backward(execution_operand(input), execution_operand(grad_output)); }
Tensor cuda_reciprocal_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_reciprocal_backward(execution_operand(input), execution_operand(grad_output)); }
Tensor cuda_relu_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha) { return cuda_relu_backward(execution_operand(input), execution_operand(grad_output)); }
ReductionResult cuda_sum_adapter(const Tensor& t, const ReductionPlan& plan) {
    Tensor input = execution_operand(t);
    return {cuda_sum(input, plan), {}};
}
ReductionResult cuda_mean_adapter(const Tensor& t, const ReductionPlan& plan) {
    Tensor input = execution_operand(t);
    return {cuda_mean(input, plan), {}};
}
ReductionResult cuda_max_adapter(const Tensor& t, const ReductionPlan& plan) {
    Tensor input = execution_operand(t);
    auto result = cuda_max_with_indices(input, plan);
    return {result.first, result.second};
}
Tensor cuda_softmax_adapter(const Tensor& t, int64_t) { return cuda_softmax(t); }
Tensor cuda_log_softmax_adapter(const Tensor& t, int64_t dim) { return cuda_log_softmax(t, dim); }
Tensor cuda_log_softmax_backward_adapter(const Tensor& t, const Tensor& grad, int64_t dim) { return cuda_log_softmax_backward(t, grad, dim); }
Tensor cuda_fused_linear_relu_adapter(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    return cuda_fused_linear_relu(
            execution_operand(x), execution_operand(weight), execution_operand(bias));
}
Tensor cuda_conv2d_adapter(const Tensor& x, const Tensor& w, const Tensor& b, const SpatialArgs& a) { return cuda_conv2d_forward(x, w, b, a.sh, a.sw, a.ph, a.pw); }
std::vector<Tensor> cuda_conv2d_backward_adapter(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor& g, const SpatialArgs& a) { return cuda_conv2d_backward(x, w, b, g, a.sh, a.sw, a.ph, a.pw); }
Tensor cuda_maxpool2d_adapter(const Tensor& x, const SpatialArgs& a) { return cuda_maxpool2d_forward(x, a.kh, a.kw, a.sh, a.sw, a.ph, a.pw); }
Tensor cuda_maxpool2d_backward_adapter(const Tensor& x, const Tensor& g, const SpatialArgs& a) { return cuda_maxpool2d_backward(x, g, a.kh, a.kw, a.sh, a.sw, a.ph, a.pw); }
Tensor cuda_cross_entropy_adapter(const Tensor& x, const std::vector<int64_t>& targets) { return cuda_cross_entropy_loss(x, targets); }
Tensor cuda_cross_entropy_backward_adapter(const Tensor& x, const std::vector<int64_t>& targets, const Tensor& grad) { return cuda_cross_entropy_backward(x, targets, grad); }
Tensor cuda_nll_loss_adapter(const Tensor& x, const Tensor& targets, const std::string& reduction) { return cuda_nll_loss_forward(x, targets, reduction); }
Tensor cuda_nll_loss_backward_adapter(const Tensor& x, const Tensor& targets, const Tensor& grad, const std::string& reduction) { return cuda_nll_loss_backward(x, targets, grad, reduction); }
Tensor cuda_embedding_adapter(const Tensor& indices, const Tensor& weight, std::optional<int64_t> pad) { return cuda_embedding_forward(indices, weight, pad); }
Tensor cuda_embedding_backward_adapter(const Tensor& indices, const Tensor& grad, const std::vector<int64_t>& shape, std::optional<int64_t> pad) { return cuda_embedding_backward(indices, grad, shape, pad); }
void cuda_inplace_mul_adapter(Tensor& a, float alpha) { cuda_mul_(a, alpha); }
void cuda_inplace_add_adapter(Tensor& a, const Tensor& b, float alpha) { cuda_add_(a, b, alpha); }
Tensor cuda_unbroadcast_adapter(const Tensor& grad, const std::vector<int64_t>& shape) { return cuda_unbroadcast(execution_operand(grad), shape); }
bool cuda_isfinite_all_adapter(const Tensor& t) { return cuda_isfinite_all(t); }
std::pair<Tensor, bool> cuda_unscale_adapter(const Tensor& t, float scale) { return cuda_unscale_to_float32_and_check(t, scale); }
void cuda_amp_copy_adapter(Tensor& dst, const Tensor& src, const CopyArgs&) { cuda_copy_cast_float32_to_(dst, src); }
void cuda_copy_range_adapter(Tensor& dst, const Tensor& src, const CopyArgs& a) {
    if (a.count > 0) {
        DeviceGuard guard(dst.device());
        auto context = default_runtime().cuda(dst.device().index);
        const auto esz = dtype_size(dst.dtype());
        context->copy_d2d_async(static_cast<char*>(dst.data()) + a.dst_offset * esz, static_cast<const char*>(src.data()) + a.src_offset * esz, a.count * esz);
    }
}
float cuda_scalar_read_adapter(const Tensor& t, int64_t offset, const char* op) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index); float value = 0;
    try { context->copy_d2h_blocking(&value, static_cast<const float*>(t.data()) + offset, sizeof(float)); }
    catch (const std::runtime_error& error) { throw std::runtime_error(std::string(op) + ": " + error.what()); }
    return value;
}
void cuda_scalar_write_adapter(const Tensor& t, int64_t offset, float value, const char* op) {
    DeviceGuard guard(t.device());
    auto context = default_runtime().cuda(t.device().index);
    try { context->copy_h2d_snapshot(static_cast<float*>(t.data()) + offset, &value, sizeof(float)); }
    catch (const std::runtime_error& error) { throw std::runtime_error(std::string(op) + ": " + error.what()); }
}
Tensor cuda_max_backward_adapter(const Tensor& input, const Tensor& grad, const Tensor& indices, const ReductionPlan& plan) {
    validate_max_scatter(input, grad, indices, plan);
    DeviceGuard guard(plan.device);
    auto context = default_runtime().cuda(plan.device.index);
    Tensor grad_in(std::make_shared<TensorImpl>(plan.input_shape, plan.dtype, plan.device));
    context->memset_async(grad_in.data(), 0, static_cast<size_t>(plan.input_numel) * dtype_size(plan.dtype));
    cuda_max_scatter_grad(grad, indices, grad_in, plan);
    return grad_in;
}
Tensor cuda_index_backward_adapter(const Tensor& grad, const ViewScatterArgs& args) {
    const auto& input_shape = args.shape;
    const auto input_dtype = args.dtype;
    const auto input_device = args.device;
    const auto index = args.index, dim = args.dim, start = args.start, end = args.end;
    const size_t esz   = dtype_size(input_dtype);
    const size_t block = static_cast<size_t>(grad.numel()) * esz;
    const size_t total = static_cast<size_t>(numel(input_shape)) * esz;
    // Secondary invariant: the exact-shape check above already implies this, so
    // a failure here means one of the two computations is wrong.
    if (block * static_cast<size_t>(input_shape[0]) != total)
        throw std::runtime_error(
            "index_backward: gradient holds " + std::to_string(grad.numel()) +
            " element(s) but one first-dimension block holds " +
            std::to_string(numel(input_shape) / input_shape[0]));

    Tensor grad_input(
        std::make_shared<TensorImpl>(input_shape, input_dtype, input_device));
    char* dst = static_cast<char*>(grad_input.data())
              + static_cast<size_t>(index) * block;

    DeviceGuard guard(input_device);
    auto context = default_runtime().cuda(input_device.index);

    // Stays on the device: no host round-trip for the gradient payload.
    context->memset_async(grad_input.data(), 0, total, "index_backward");
    context->copy_d2d_async(dst, grad.data(), block, "index_backward");
    return grad_input;

}
Tensor cuda_slice_backward_adapter(const Tensor& grad, const ViewScatterArgs& args) {
    const auto& input_shape = args.shape;
    const auto input_dtype = args.dtype;
    const auto input_device = args.device;
    const auto index = args.index, dim = args.dim, start = args.start, end = args.end;
    int64_t prefix = 1;
    for (int64_t d = 0; d < dim; ++d)
        prefix *= input_shape[static_cast<size_t>(d)];
    int64_t suffix = 1;
    for (size_t d = static_cast<size_t>(dim) + 1; d < input_shape.size(); ++d)
        suffix *= input_shape[d];

    const int64_t slice_len = end - start;
    const size_t  esz         = dtype_size(input_dtype);
    const size_t  width_bytes = static_cast<size_t>(slice_len) *
                                static_cast<size_t>(suffix) * esz;
    const size_t  dst_pitch   = static_cast<size_t>(input_shape[static_cast<size_t>(dim)]) *
                                static_cast<size_t>(suffix) * esz;
    const size_t  total       = static_cast<size_t>(numel(input_shape)) * esz;
    const size_t  start_bytes = static_cast<size_t>(start) *
                                static_cast<size_t>(suffix) * esz;

    Tensor grad_input(
        std::make_shared<TensorImpl>(input_shape, input_dtype, input_device));

    // A parent with zero total elements owns no storage at all (Storage skips
    // the allocation for 0 bytes), so data() is null. Return the correctly
    // shaped empty gradient rather than calling memset/cudaMemset on nullptr.
    if (total == 0)
        return grad_input;

    // An empty slice (start == end) is a legal forward - it yields a zero-length
    // dimension - and its gradient is simply the all-zero parent gradient. The
    // copies are skipped explicitly: grad.data() is null for a zero-element
    // gradient, and a 0-byte copy from a null source is not something to rely on.
    const bool has_payload = width_bytes > 0 && prefix > 0;

    DeviceGuard guard(input_device);
    auto context = default_runtime().cuda(input_device.index);

    // Stays on the device: no host round-trip for the gradient payload.
    context->memset_async(grad_input.data(), 0, total, "slice_backward");
    if (has_payload) {
        // The runs are evenly pitched, so the whole multi-run scatter is one
        // pitched 2-D copy: `prefix` rows of width_bytes, packed in the source
        // and dst_pitch apart in the destination. No custom kernel is needed.
        context->copy_d2d_2d_async(static_cast<char*>(grad_input.data()) + start_bytes, dst_pitch, grad.data(), width_bytes, width_bytes, static_cast<size_t>(prefix), "slice_backward");
    }
    return grad_input;

}

} } }
