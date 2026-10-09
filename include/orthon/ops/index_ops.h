#pragma once
#include <cstdint>
#include <vector>

#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>
#include <orthon/device/device.h>

namespace orthon {

// Index backward scatters a contiguous upstream gradient into the selected
// first-dimension block of a fresh zero parent gradient. Forward normalizes
// the index; backward uses that resolved value.
Tensor index_backward(const Tensor& grad_output,
                      const std::vector<int64_t>& input_shape,
                      DType   input_dtype,
                      Device  input_device,
                      int64_t index);

}  // namespace orthon
