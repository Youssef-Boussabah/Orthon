#include <orthon/ops/cpu/index_ops.h>
#include <orthon/core/shape.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device_guard.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#define ORTHON_INDEX_CUDA_CHECK(call)                                       \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess)                                              \
            throw std::runtime_error(std::string("index_backward: CUDA error: ") \
                                     + cudaGetErrorString(_err));             \
    } while (0)
#endif

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

    if (input_device.is_cpu()) {
        std::memset(grad_input.data(), 0, total);
        std::memcpy(dst, grad.data(), block);
        return grad_input;
    }

#ifdef WITH_CUDA
    // The engine-wide guard selects the parent device and restores the
    // caller's on the way out. Placement is deliberate: after contiguous() and
    // after the grad_input allocation, so no later call can change what is
    // current before the copies below.
    DeviceGuard guard(input_device);

    // Stays on the device: no host round-trip for the gradient payload.
    ORTHON_INDEX_CUDA_CHECK(cudaMemset(grad_input.data(), 0, total));
    ORTHON_INDEX_CUDA_CHECK(
        cudaMemcpy(dst, grad.data(), block, cudaMemcpyDeviceToDevice));
    return grad_input;
#else
    throw std::runtime_error(
        "index_backward: CUDA is not available in this build "
        "(recompile with -DWITH_CUDA=ON)");
#endif
}

}  // namespace orthon
