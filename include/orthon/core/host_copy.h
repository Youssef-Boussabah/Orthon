#pragma once
#include <orthon/core/tensor.h>
namespace orthon {
// Logical byte packing for host staging and CPU copy; no graph policy.
Tensor make_contiguous_copy(const Tensor& src) ;
// Zeroed host staging storage; byte initialization only, no operator policy.
Tensor make_zeroed_host_tensor(const std::vector<int64_t>& shape, DType dtype);
} // namespace orthon
