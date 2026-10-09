#include <orthon/core/strides.h>
#include <orthon/core/layout.h>

namespace orthon {

std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape) {
    return Layout::contiguous_strides(shape);
}

bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    if (shape.size() != strides.size()) return false;
    return Layout(shape, strides).is_contiguous();
}

}  // namespace orthon
