#include <orthon/dispatch/operator.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/float_formats.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include <orthon/backends/cpu/kernels.h>

namespace orthon { namespace backend { namespace cpu {
using namespace dispatch;
Tensor cpu_slice_backward(const Tensor& grad, const ViewScatterArgs& args) {
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

    {
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


}

} } }
