#pragma once
#include <orthon/core/tensor.h>
#include <orthon/core/layout.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace orthon {

struct Conv2dGeometry {
    int64_t height;
    int64_t width;
};

inline int64_t checked_conv2d_extent(int64_t input, int64_t kernel,
                                    int64_t stride, int64_t padding) {
    if (stride <= 0)
        throw std::invalid_argument("conv2d: stride must be positive");
    if (padding < 0)
        throw std::invalid_argument("conv2d: padding must be nonnegative");
    if (input <= 0 || kernel <= 0)
        throw std::invalid_argument("conv2d: input and kernel spatial dimensions must be positive");
    const auto limit = std::numeric_limits<int64_t>::max();
    if (padding > (limit - input) / 2)
        throw std::overflow_error("conv2d: padded spatial extent overflows int64");
    const int64_t padded = input + 2 * padding;
    if (kernel > padded)
        throw std::invalid_argument("conv2d: kernel exceeds padded input");
    // Nonnegative numerator and positive stride make integer division floor.
    return (padded - kernel) / stride + 1;
}

inline Conv2dGeometry validate_conv2d_operands(const Tensor& input, const Tensor& weight,
                                              int64_t sh, int64_t sw,
                                              int64_t ph, int64_t pw) {
    if (input.ndim() != 4 || weight.ndim() != 4)
        throw std::runtime_error("conv2d: input and weight must be 4D");
    if (input.device() != weight.device())
        throw std::runtime_error("conv2d: input and weight must be on the same device");
    const auto dtype = input.dtype();
    if (weight.dtype() != dtype ||
        (dtype != DType::Float32 && !(input.device().is_cuda() &&
          (dtype == DType::Float16 || dtype == DType::BFloat16))))
        throw std::runtime_error("conv2d: only float32 supported (float16/bfloat16 requires CUDA)");
    if (!input.is_contiguous() || !weight.is_contiguous())
        throw std::runtime_error("conv2d: all tensors must be contiguous");
    for (auto extent : input.shape())
        if (extent <= 0) throw std::invalid_argument("conv2d: input dimensions must be positive");
    for (auto extent : weight.shape())
        if (extent <= 0) throw std::invalid_argument("conv2d: weight dimensions must be positive");
    if (input.shape()[1] != weight.shape()[1])
        throw std::runtime_error("conv2d: C_in mismatch between input and weight");
    Conv2dGeometry geometry{
        checked_conv2d_extent(input.shape()[2], weight.shape()[2], sh, ph),
        checked_conv2d_extent(input.shape()[3], weight.shape()[3], sw, pw)};
    // Check output products and bytes before any backend allocates storage.
    Layout({input.shape()[0], weight.shape()[0], geometry.height, geometry.width})
        .payload_bytes(dtype_size(dtype));
    return geometry;
}

inline Conv2dGeometry validate_conv2d_forward(const Tensor& input, const Tensor& weight,
                                             const Tensor& bias, int64_t sh, int64_t sw,
                                             int64_t ph, int64_t pw) {
    const auto geometry = validate_conv2d_operands(input, weight, sh, sw, ph, pw);
    if (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])
        throw std::runtime_error("conv2d: bias length must equal C_out");
    if (bias.device() != input.device() || bias.dtype() != input.dtype())
        throw std::runtime_error("conv2d: bias device and dtype must match input");
    if (!bias.is_contiguous())
        throw std::runtime_error("conv2d: all tensors must be contiguous");
    return geometry;
}

inline void validate_conv2d_gradient(const Tensor& input, const Tensor& weight,
                                     const Tensor& grad, int64_t sh, int64_t sw,
                                     int64_t ph, int64_t pw) {
    const auto geometry = validate_conv2d_operands(input, weight, sh, sw, ph, pw);
    if (grad.shape() != std::vector<int64_t>{input.shape()[0], weight.shape()[0],
                                           geometry.height, geometry.width})
        throw std::invalid_argument("conv2d backward: grad_output shape mismatch");
    if (grad.device() != input.device() || grad.dtype() != input.dtype())
        throw std::invalid_argument("conv2d backward: grad_output device and dtype mismatch");
    if (!grad.is_contiguous())
        throw std::invalid_argument("conv2d backward: grad_output must be contiguous");
}

} // namespace orthon
