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

// `value` is a DOUBLE, not a float.
//
// A float parameter would round every incoming numeric value to float32 BEFORE
// it reached the destination dtype, destroying the bottom 29 bits of a float64
// tensor: full({}, 1.0/3.0, Float64) storing 0.3333333432674408 rather than
// 0.3333333333333333. That single narrowing also caps float64 mean-backward
// precision, because ReduceNode builds its scale through here.
//
// Per destination dtype: Float64 keeps the incoming double exactly; Float32
// converts explicitly; Float16/BFloat16 go through the canonical float32
// bit-pattern helpers in <orthon/core/float_formats.h>; integer and bool use a
// truncating / nonzero conversion. A C++ caller passing a float is
// source-compatible through implicit widening.
Tensor full(const std::vector<int64_t>& shape,
            double value,
            DType  dtype  = DType::Float32,
            Device device = Device{});

// In-place fill: writes `value` into every logical element of `t`.
// Supports both contiguous and non-contiguous (stride-based) tensors.
void fill_(Tensor& t, float value);

}  // namespace orthon
