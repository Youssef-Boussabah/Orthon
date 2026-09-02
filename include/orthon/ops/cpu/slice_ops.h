#pragma once
#include <cstdint>
#include <vector>

#include <orthon/core/dtype.h>
#include <orthon/core/tensor.h>
#include <orthon/device/device.h>

namespace orthon {

// Backward of the range view `input.slice(dim, start, end)`.
//
// Returns a freshly allocated, zero-filled tensor shaped like the sliced
// parent, with `grad_output` scattered into the logical region
// [.., start:end, ..] along `dim`.
//
// Unlike a first-dimension integer index, a general slice is NOT one
// contiguous block of the parent gradient. Splitting the parent's logical shape
// around `dim` gives
//
//     prefix    = product(input_shape[0     : dim])
//     suffix    = product(input_shape[dim+1 : ])
//     slice_len = end - start
//
// and the selected elements form `prefix` runs of `slice_len * suffix` elements
// each, landing `input_shape[dim] * suffix` elements apart in the destination.
// The scatter therefore copies run by run (CPU) or as one pitched 2-D transfer
// (CUDA), never as a single flat memcpy.
//
// `dim`, `start` and `end` must already be normalised (non-negative, in range):
// SliceNode stores the values Tensor::slice resolved, so no negative argument is
// ever re-interpreted here. `grad_output` may be strided; it is materialised
// internally.
Tensor slice_backward(const Tensor& grad_output,
                      const std::vector<int64_t>& input_shape,
                      DType   input_dtype,
                      Device  input_device,
                      int64_t dim,
                      int64_t start,
                      int64_t end);

}  // namespace orthon
