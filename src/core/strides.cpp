#include <cstddef>
#include <orthon/core/strides.h>

namespace orthon {

std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape) {
    const size_t ndim = shape.size();
    std::vector<int64_t> strides(ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    const size_t ndim = shape.size();
    if (ndim != strides.size()) return false;
    int64_t expected = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        if (shape[i] == 1) continue;   // size-1 dim: stride doesn't affect layout
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

}  // namespace orthon
