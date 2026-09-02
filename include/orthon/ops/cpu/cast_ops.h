#pragma once
#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>

namespace orthon {

// Return a new contiguous tensor with the same shape and device as `src` but
// with dtype `dst_dtype`.  Values are converted element-wise.
//
// CPU: all DType pairs supported.
// CUDA: any pair among Float32/Float16/BFloat16; same-dtype makes a plain copy.
Tensor cast(const Tensor& src, DType dst_dtype);

}  // namespace orthon
