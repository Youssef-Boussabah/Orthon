#include <orthon/core/shape.h>
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
    validate_shape(shape);
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

}  // namespace orthon
