#pragma once
#include <cstdint>
#include <vector>

#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>
#include <orthon/device/device.h>

namespace orthon {

// Backward of the first-dimension integer index `input[index]`.
//
// Returns a freshly allocated, zero-filled tensor shaped like the indexed
// parent, with `grad_output` scattered into first-dimension block `index`.
// Because the result is contiguous, that block is physically contiguous, so
// the scatter is a plain typed copy rather than a strided kernel.
//
// `index` must already be normalised (non-negative). `grad_output` may be
// strided; it is materialised internally.
Tensor index_backward(const Tensor& grad_output,
                      const std::vector<int64_t>& input_shape,
                      DType   input_dtype,
                      Device  input_device,
                      int64_t index);

}  // namespace orthon
