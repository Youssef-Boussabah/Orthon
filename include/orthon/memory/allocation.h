#pragma once
#include <cstddef>
#include <memory>
#include <orthon/device/device.h>

namespace orthon {

class Allocator;

// One checked-out lifetime; the releasing allocator outlives this record.
class Allocation {
public:
    Allocation() = default;
    ~Allocation();
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;
    void* data() const { return pointer_; }
    size_t nbytes() const { return requested_; }
    size_t capacity() const { return capacity_; }
    Device device() const { return device_; }
    const std::shared_ptr<Allocator>& allocator() const { return allocator_; }
    void reset() noexcept;
private:
    friend class Allocator;
    Allocation(void* pointer, size_t requested, size_t capacity,
               Device device, std::shared_ptr<Allocator> allocator) noexcept;
    void* pointer_ = nullptr;
    size_t requested_ = 0;
    size_t capacity_ = 0;
    Device device_;
    std::shared_ptr<Allocator> allocator_;
};

}  // namespace orthon
