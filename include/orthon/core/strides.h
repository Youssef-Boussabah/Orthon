#pragma once
#include <cstdint>
#include <vector>

namespace orthon {

// Row-major (C-contiguous): strides[i] = product of shape[i+1 .. N-1].
std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape);

// Size-1 dimensions are ignored (any stride is valid there, matching PyTorch semantics).
bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides);

}  // namespace orthon
