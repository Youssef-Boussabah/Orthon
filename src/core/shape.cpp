#include <orthon/core/shape.h>
#include <orthon/core/layout.h>
#include <stdexcept>

namespace orthon {

void validate_shape(const std::vector<int64_t>& shape) {
    for (int64_t d : shape) {
        if (d < 0) {
            throw std::invalid_argument("Shape dimensions must be non-negative");
        }
    }
}

int64_t numel(const std::vector<int64_t>& shape) {
    return Layout::shape_numel(shape);
}

}  // namespace orthon
