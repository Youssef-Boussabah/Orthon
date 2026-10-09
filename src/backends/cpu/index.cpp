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
Tensor cpu_index_backward(const Tensor& grad, const ViewScatterArgs& args) {
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
    if (total == 0) return grad_input;
    char* dst = static_cast<char*>(grad_input.data())
              + static_cast<size_t>(index) * block;

    {
        std::memset(grad_input.data(), 0, total);
        std::memcpy(dst, grad.data(), block);
        return grad_input;
    }


}

} } }
