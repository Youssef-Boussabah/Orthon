#include <orthon/core/layout.h>
#include <orthon/core/shape.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace orthon {

static int64_t checked_mul(int64_t a, int64_t b) {
    if (a < 0 || b < 0)
        throw std::invalid_argument("Layout: negative coordinate or stride");
    if (a != 0 && b > (std::numeric_limits<int64_t>::max)() / a)
        throw std::overflow_error("Layout: shape/stride product overflows");
    return a * b;
}

static int64_t checked_add(int64_t a, int64_t b) {
    if (a < 0 || b < 0)
        throw std::invalid_argument("Layout: negative coordinate or stride");
    if (b > (std::numeric_limits<int64_t>::max)() - a)
        throw std::overflow_error("Layout: storage span/offset overflows");
    return a + b;
}

static size_t checked_bytes(int64_t count, size_t element_size) {
    if (element_size == 0)
        throw std::invalid_argument("Layout: element size must be positive");
    // Keep pointer arithmetic representable as well as the allocation size.
    const auto limit = static_cast<uintmax_t>((std::numeric_limits<ptrdiff_t>::max)());
    if (static_cast<uintmax_t>(count) > limit / element_size)
        throw std::overflow_error("Layout: byte range overflows");
    return static_cast<size_t>(count) * element_size;
}

int64_t Layout::shape_numel(const std::vector<int64_t>& shape) {
    validate_shape(shape);
    // Zero-element shapes have a zero product regardless of axis order.
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
    int64_t n = 1;
    for (int64_t size : shape) n = checked_mul(n, size);
    return n;
}

std::vector<int64_t> Layout::contiguous_strides(const std::vector<int64_t>& shape) {
    validate_shape(shape);
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        if (i != 0) stride = checked_mul(stride, shape[i]);
    }
    return strides;
}

Layout::Layout(std::vector<int64_t> shape)
    : Layout(shape, contiguous_strides(shape), 0) {}

Layout::Layout(std::vector<int64_t> shape, std::vector<int64_t> strides,
               int64_t storage_offset)
    : shape_(std::move(shape)), strides_(std::move(strides)),
      storage_offset_(storage_offset), numel_(shape_numel(shape_)), span_(0) {
    if (shape_.size() != strides_.size())
        throw std::invalid_argument("Layout: shape/stride rank mismatch");
    if (storage_offset_ < 0)
        throw std::invalid_argument("Layout: negative storage offset");
    for (int64_t stride : strides_)
        if (stride < 0) throw std::invalid_argument("Layout: negative stride");
    if (numel_ != 0) {
        span_ = 1;  // includes the last element, including rank-zero scalars
        for (size_t i = 0; i < shape_.size(); ++i)
            span_ = checked_add(span_, checked_mul(shape_[i] - 1, strides_[i]));
        (void)checked_add(storage_offset_, span_);
    }
}

bool Layout::is_contiguous() const {
    int64_t expected = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        if (shape_[i] == 1) continue;
        if (strides_[i] != expected) return false;
        expected = checked_mul(expected, shape_[i]);
    }
    return true;
}

std::pair<int64_t, int64_t> Layout::storage_range() const {
    return {storage_offset_, checked_add(storage_offset_, span_)};
}

void Layout::validate_bounds(size_t capacity_bytes, size_t element_size) const {
    (void)offset_bytes(element_size);
    (void)payload_bytes(element_size);
    if (numel_ != 0 && checked_bytes(storage_range().second, element_size) > capacity_bytes)
        throw std::out_of_range("Layout: logical range exceeds Buffer capacity");
}

size_t Layout::payload_bytes(size_t element_size) const {
    return checked_bytes(numel_, element_size);
}
size_t Layout::offset_bytes(size_t element_size) const {
    return checked_bytes(storage_offset_, element_size);
}
int64_t Layout::shifted_offset(int64_t index, int64_t stride) const {
    return checked_add(storage_offset_, checked_mul(index, stride));
}

}  // namespace orthon
