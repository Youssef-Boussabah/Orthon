#pragma once
#include <cstdint>
#include <vector>

namespace orthon {

// Returns 1 for scalars (empty shape). Throws if any dimension is negative.
int64_t numel(const std::vector<int64_t>& shape);

void validate_shape(const std::vector<int64_t>& shape);

}  // namespace orthon
