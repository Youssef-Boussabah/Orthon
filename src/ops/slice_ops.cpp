#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/slice_ops.h>
#include <orthon/core/shape.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device_guard.h>
#include <orthon/ops/shape_ops.h>
#include <cstring>
#include <stdexcept>
#include <string>


namespace orthon {

static std::string slice_shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) r += ", ";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

Tensor slice_backward(const Tensor& grad_output,
                      const std::vector<int64_t>& input_shape,
                      DType   input_dtype,
                      Device  input_device,
                      int64_t dim,
                      int64_t start,
                      int64_t end) {
    // Tensor::slice refuses to slice a 0-d tensor (there is no dimension to
    // normalise), so a scalar parent reaching here is an engine defect.
    if (input_shape.empty())
        throw std::runtime_error("slice_backward: parent tensor has no dimensions");

    // SliceNode stores the values Tensor::slice already resolved. Re-checking
    // them is cheap and pins the contract: this function never normalises, so a
    // negative value arriving here means the caller skipped that step.
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size()))
        throw std::out_of_range(
            "slice_backward: normalised dim " + std::to_string(dim) +
            " out of range for parent shape " + slice_shape_str(input_shape));
    if (start < 0 || end < start || end > input_shape[static_cast<size_t>(dim)])
        throw std::out_of_range(
            "slice_backward: normalised range [" + std::to_string(start) + ", " +
            std::to_string(end) + ") out of range for dimension " +
            std::to_string(dim) + " with size " +
            std::to_string(input_shape[static_cast<size_t>(dim)]));

    // A gradient arriving on the wrong device is an engine bug, not something to
    // paper over with an implicit transfer: silently moving it would hide the
    // defect and hand a CUDA pointer to the CPU memcpy branch below. Checked
    // before the branch is chosen, and before contiguous() copies anything.
    if (grad_output.device() != input_device)
        throw std::runtime_error(
            "slice_backward: gradient device " + device_to_string(grad_output.device()) +
            " does not match sliced tensor device " + device_to_string(input_device));

    if (grad_output.dtype() != input_dtype)
        throw std::runtime_error(
            "slice_backward: gradient dtype (enum " +
            std::to_string(static_cast<int>(grad_output.dtype())) +
            ") does not match sliced tensor dtype (enum " +
            std::to_string(static_cast<int>(input_dtype)) + ")");

    // The slice keeps the parent's rank and every other dimension, so the
    // gradient must be shaped exactly input_shape with dim replaced by the slice
    // length. Element count alone is not enough: for parent [2, 5, 3] a slice of
    // [1:4] on dim 1 and one on dim 2 of [0:2] both hold 18 elements but scatter
    // to entirely different places.
    std::vector<int64_t> expected_shape = input_shape;
    expected_shape[static_cast<size_t>(dim)] = end - start;
    if (grad_output.shape() != expected_shape)
        throw std::runtime_error(
            "slice_backward: gradient shape " + slice_shape_str(grad_output.shape()) +
            " does not match the sliced view shape " + slice_shape_str(expected_shape) +
            " for parent shape " + slice_shape_str(input_shape));

    // A view node upstream may hand back a strided gradient - TransposeNode does
    // exactly that - and the run copies below read storage order, so materialise
    // first. No-op when the gradient is already contiguous.
    const Tensor grad = contiguous(grad_output);

    // Split the parent's LOGICAL shape around dim. The parent's physical strides
    // are irrelevant here: the gradient returned is a fresh contiguous tensor in
    // logical row-major order, which is what the next node up expects even when
    // the sliced parent was itself a strided view.
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ViewScatterKernel>(unary_request(OperatorId::SliceBackward, grad_output), default_runtime());
    return kernel.call(grad, {input_shape, input_dtype, input_device, 0, dim, start, end});
}

}  // namespace orthon
