#include <orthon/ops/cpu/slice_ops.h>
#include <orthon/core/shape.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device_guard.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#define ORTHON_SLICE_CUDA_CHECK(call)                                       \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess)                                              \
            throw std::runtime_error(std::string("slice_backward: CUDA error: ") \
                                     + cudaGetErrorString(_err));             \
    } while (0)
#endif

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

    if (input_device.is_cpu()) {
        std::memset(grad_input.data(), 0, total);
        if (has_payload) {
            char*       dst = static_cast<char*>(grad_input.data()) + start_bytes;
            const char* src = static_cast<const char*>(grad.data());
            for (int64_t p = 0; p < prefix; ++p) {
                std::memcpy(dst + static_cast<size_t>(p) * dst_pitch,
                            src + static_cast<size_t>(p) * width_bytes,
                            width_bytes);
            }
        }
        return grad_input;
    }

#ifdef WITH_CUDA
    // The engine-wide guard, placed as index_backward places it: after
    // contiguous() and after the grad_input allocation, so nothing called later
    // can change which device is current before the transfers below.
    DeviceGuard guard(input_device);

    // Stays on the device: no host round-trip for the gradient payload.
    ORTHON_SLICE_CUDA_CHECK(cudaMemset(grad_input.data(), 0, total));
    if (has_payload) {
        // The runs are evenly pitched, so the whole multi-run scatter is one
        // pitched 2-D copy: `prefix` rows of width_bytes, packed in the source
        // and dst_pitch apart in the destination. No custom kernel is needed.
        ORTHON_SLICE_CUDA_CHECK(cudaMemcpy2D(
            static_cast<char*>(grad_input.data()) + start_bytes,
            dst_pitch,
            grad.data(),
            width_bytes,
            width_bytes,
            static_cast<size_t>(prefix),
            cudaMemcpyDeviceToDevice));
    }
    return grad_input;
#else
    throw std::runtime_error(
        "slice_backward: CUDA is not available in this build "
        "(recompile with -DWITH_CUDA=ON)");
#endif
}

}  // namespace orthon
