#pragma once
#include <cstdint>
#include <vector>

#include <orthon/device/device.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>

namespace orthon {

Tensor empty(const std::vector<int64_t>& shape,
             DType  dtype  = DType::Float32,
             Device device = Device{});

Tensor zeros(const std::vector<int64_t>& shape,
             DType  dtype  = DType::Float32,
             Device device = Device{});

Tensor ones(const std::vector<int64_t>& shape,
            DType  dtype  = DType::Float32,
            Device device = Device{});

// Double preserves Float64 source precision. Other destinations narrow once
// through the canonical host conversions, including ties-to-even rounding.
Tensor full(const std::vector<int64_t>& shape,
            double value,
            DType  dtype  = DType::Float32,
            Device device = Device{});

// In-place fill: writes `value` into every logical element of `t`.
// Supports both contiguous and non-contiguous (stride-based) tensors.
void fill_(Tensor& t, float value);

}  // namespace orthon
