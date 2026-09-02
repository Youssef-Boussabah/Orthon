#pragma once
#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>

namespace orthon {

// Cast src to dst_dtype on CUDA.
// Supported: any pair among Float32/Float16/BFloat16, plus same-dtype (copy).
// Unsupported pairs throw std::runtime_error.
Tensor cuda_cast(const Tensor& src, DType dst_dtype);

// Copy src float32 values into an existing CUDA destination, casting in-place.
// Supported destinations: Float16, BFloat16, Float32.
void cuda_copy_cast_float32_to_(Tensor& dst, const Tensor& src);

}  // namespace orthon
