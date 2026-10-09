#pragma once
#include <orthon/core/tensor.h>
#include <orthon/core/layout.h>
#include <orthon/dispatch/operator.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace orthon {

struct MaxPool2dGeometry {
    int64_t height;
    int64_t width;
};

inline int64_t checked_maxpool2d_extent(int64_t input, int64_t kernel,
                                      int64_t stride, int64_t padding) {
    if (kernel <= 0)
        throw std::runtime_error("maxpool2d: kernel_size must be positive");
    if (stride <= 0)
        throw std::runtime_error("maxpool2d: stride must be positive");
    if (padding < 0)
        throw std::runtime_error("maxpool2d: padding must be non-negative");
    if (padding > kernel / 2)
        throw std::runtime_error("maxpool2d: padding must be at most half the kernel size");
    if (input < 0)
        throw std::invalid_argument("maxpool2d: input spatial size must be non-negative");
    const auto limit = (std::numeric_limits<int64_t>::max)();
    if (padding > (limit - input) / 2)
        throw std::overflow_error("maxpool2d: padded spatial extent overflows int64");
    const int64_t padded = input + 2 * padding;
    if (kernel > padded)
        throw std::runtime_error("maxpool2d: output spatial size is non-positive");
    // The numerator is nonnegative, so division is floor. Since kernel >= 1,
    // the quotient is at most INT64_MAX-1 and the final +1 also fits.
    // For every output o, o*stride <= padded-kernel. Thus base=o*stride-pad
    // and base+kernel are representable too, including padding-only windows.
    return (padded - kernel) / stride + 1;
}

inline int64_t checked_maxpool2d_cuda_blocks(int64_t total) {
    if (total < 0)
        throw std::invalid_argument("maxpool2d: negative launch size");
    // Matches the existing 256-thread kernels and CUDA's grid.x limit.
    const int64_t blocks = total == 0 ? 0 : 1 + (total - 1) / 256;
    if (blocks > (std::numeric_limits<int>::max)())
        throw std::overflow_error("maxpool2d: CUDA launch size exceeds grid.x range");
    return blocks;
}

// Metadata-only admission is also usable without allocating a Tensor or GPU.
inline MaxPool2dGeometry checked_maxpool2d_geometry(
        const std::vector<int64_t>& shape, const dispatch::SpatialArgs& a,
        bool cuda = false) {
    if (shape.size() != 4)
        throw std::runtime_error("maxpool2d: input must be 4D [N,C,H,W]");
    // Preserve the public argument-error precedence for unrelated invalid cases.
    if (a.kh <= 0 || a.kw <= 0)
        throw std::runtime_error("maxpool2d: kernel_size must be positive");
    if (a.sh <= 0 || a.sw <= 0)
        throw std::runtime_error("maxpool2d: stride must be positive");
    if (a.ph < 0 || a.pw < 0)
        throw std::runtime_error("maxpool2d: padding must be non-negative");
    if (a.ph > a.kh / 2 || a.pw > a.kw / 2)
        throw std::runtime_error("maxpool2d: padding must be at most half the kernel size");
    const Layout input_layout(shape);
    input_layout.payload_bytes(sizeof(float));
    const MaxPool2dGeometry geometry{
        checked_maxpool2d_extent(shape[2], a.kh, a.sh, a.ph),
        checked_maxpool2d_extent(shape[3], a.kw, a.sw, a.pw)};
    const Layout output_layout({shape[0], shape[1], geometry.height, geometry.width});
    output_layout.payload_bytes(sizeof(float));
    if (cuda) {
        const auto limit = (std::numeric_limits<int>::max)();
        for (int64_t value : {shape[0], shape[1], shape[2], shape[3],
                             geometry.height, geometry.width, a.kh, a.kw,
                             a.sh, a.sw, a.ph, a.pw}) {
            if (value > limit)
                throw std::overflow_error("maxpool2d: CUDA geometry exceeds int32 range");
        }
        checked_maxpool2d_cuda_blocks(input_layout.numel());
        checked_maxpool2d_cuda_blocks(output_layout.numel());
    }
    return geometry;
}

inline MaxPool2dGeometry validate_maxpool2d_input(
        const Tensor& input, const dispatch::SpatialArgs& a) {
    if (input.ndim() != 4)
        throw std::runtime_error("maxpool2d: input must be 4D [N,C,H,W]");
    if (input.dtype() != DType::Float32)
        throw std::runtime_error("maxpool2d: only float32 supported");
    if (!input.is_contiguous())
        throw std::runtime_error("maxpool2d: input must be contiguous");
    return checked_maxpool2d_geometry(input.shape(), a, input.device().is_cuda());
}

inline MaxPool2dGeometry validate_maxpool2d_gradient(
        const Tensor& input, const Tensor& grad, const dispatch::SpatialArgs& a) {
    const auto geometry = validate_maxpool2d_input(input, a);
    if (grad.shape() != std::vector<int64_t>{input.shape()[0], input.shape()[1],
                                           geometry.height, geometry.width})
        throw std::invalid_argument("maxpool2d backward: grad_output shape mismatch");
    if (grad.dtype() != input.dtype() || grad.device() != input.device())
        throw std::invalid_argument("maxpool2d backward: grad_output device and dtype mismatch");
    if (!grad.is_contiguous())
        throw std::invalid_argument("maxpool2d backward: grad_output must be contiguous");
    return geometry;
}

} // namespace orthon
