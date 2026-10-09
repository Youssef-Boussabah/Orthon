#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace orthon {

// Immutable, backend-neutral element coordinates. Nonnegative strides match
// the existing view surface; negative-stride views are not introduced here.
class Layout {
public:
    explicit Layout(std::vector<int64_t> shape);
    Layout(std::vector<int64_t> shape, std::vector<int64_t> strides,
           int64_t storage_offset = 0);

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t storage_offset() const { return storage_offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    bool is_contiguous() const;

    // Half-open physical element range. Empty layouts access no elements;
    // their equal endpoints preserve the view's logical offset, even when
    // slicing an already-empty parent or an offset strided endpoint.
    std::pair<int64_t, int64_t> storage_range() const;
    void validate_bounds(size_t capacity_bytes, size_t element_size) const;
    size_t payload_bytes(size_t element_size) const;
    size_t offset_bytes(size_t element_size) const;
    int64_t shifted_offset(int64_t index, int64_t stride) const;

    // The legacy shape/stride helpers delegate here, so callers checking a
    // representation before construction use the same checked arithmetic.
    static int64_t shape_numel(const std::vector<int64_t>& shape);
    static std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t storage_offset_;
    int64_t numel_;
    int64_t span_;
};

}  // namespace orthon
