#pragma once
#include <cstdint>
#include <vector>

#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>
#include <orthon/device/device.h>

namespace orthon {

// Scatter slice gradients into prefix runs of slice_len * suffix elements.
// Destination runs are size(dim) * suffix elements apart, so one flat copy
// is insufficient. Forward normalizes all bounds; backward materializes a
// strided upstream gradient before copying into the zero parent tensor.
Tensor slice_backward(const Tensor& grad_output,
                      const std::vector<int64_t>& input_shape,
                      DType   input_dtype,
                      Device  input_device,
                      int64_t dim,
                      int64_t start,
                      int64_t end);

}  // namespace orthon
