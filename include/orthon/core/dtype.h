#pragma once
#include <cstddef>

namespace orthon {

enum class DType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int32,
    Int64,
    UInt8,
    Int8,
    Bool,
};

size_t dtype_size(DType dtype);

// True for the four floating-point dtypes (Float32/Float64/Float16/BFloat16).
// Device/dtype conversion is differentiable only between floating dtypes;
// integer/bool conversions never join a graph.
bool dtype_is_floating(DType dtype);

}  // namespace orthon
