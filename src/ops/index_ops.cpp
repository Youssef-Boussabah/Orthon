#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/index_ops.h>
#include <orthon/core/shape.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device_guard.h>
#include <orthon/ops/shape_ops.h>
#include <cstring>
#include <stdexcept>
#include <string>


namespace orthon {

static std::string index_shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) r += ", ";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

Tensor index_backward(const Tensor& grad_output,
                      const std::vector<int64_t>& input_shape,
                      DType   input_dtype,
                      Device  input_device,
                      int64_t index) {
    if (input_shape.empty())
        throw std::runtime_error("index_backward: parent tensor has no dimensions");
    if (index < 0 || index >= input_shape[0])
        throw std::out_of_range("index_backward: normalised index out of range");

    // A gradient arriving on the wrong device is an engine bug, not something to
    // paper over with an implicit transfer: silently moving it would hide the
    // defect and hand a CUDA pointer to the CPU memcpy branch below. Checked
    // before the branch is chosen, and before contiguous() copies anything.
    if (grad_output.device() != input_device)
        throw std::runtime_error(
            "index_backward: gradient device " + device_to_string(grad_output.device()) +
            " does not match indexed tensor device " + device_to_string(input_device));

    if (grad_output.dtype() != input_dtype)
        throw std::runtime_error(
            "index_backward: gradient dtype (enum " +
            std::to_string(static_cast<int>(grad_output.dtype())) +
            ") does not match indexed tensor dtype (enum " +
            std::to_string(static_cast<int>(input_dtype)) + ")");

    // input[index] drops dimension 0, so the gradient must be shaped exactly
    // input_shape[1:]. Element count alone is not enough: [2, 4] and [4, 2]
    // occupy the same bytes but scatter to different places.
    const std::vector<int64_t> expected_shape(input_shape.begin() + 1, input_shape.end());
    if (grad_output.shape() != expected_shape)
        throw std::runtime_error(
            "index_backward: gradient shape " + index_shape_str(grad_output.shape()) +
            " does not match the indexed block shape " + index_shape_str(expected_shape) +
            " for parent shape " + index_shape_str(input_shape));

    // A view node upstream may hand back a strided gradient; the block copy
    // below reads storage order, so materialise first. No-op when contiguous.
    const Tensor grad = contiguous(grad_output);

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<ViewScatterKernel>(unary_request(OperatorId::IndexBackward, grad_output), default_runtime());
    return kernel.call(grad, {input_shape, input_dtype, input_device, index});
}

}  // namespace orthon
