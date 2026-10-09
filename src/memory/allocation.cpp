#include <orthon/memory/allocation.h>
#include <orthon/memory/allocator.h>
#include <utility>

namespace orthon {

Allocation::Allocation(void* pointer, size_t requested, size_t capacity,
                       Device device, std::shared_ptr<Allocator> allocator) noexcept
    : pointer_(pointer), requested_(requested), capacity_(capacity),
      device_(device), allocator_(std::move(allocator)) {}
Allocation::~Allocation() { reset(); }
Allocation::Allocation(Allocation&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      requested_(std::exchange(other.requested_, 0)),
      capacity_(std::exchange(other.capacity_, 0)), device_(other.device_),
      allocator_(std::move(other.allocator_)) {}
Allocation& Allocation::operator=(Allocation&& other) noexcept {
    if (this != &other) {
        reset();
        pointer_ = std::exchange(other.pointer_, nullptr);
        requested_ = std::exchange(other.requested_, 0);
        capacity_ = std::exchange(other.capacity_, 0);
        device_ = other.device_;
        allocator_ = std::move(other.allocator_);
    }
    return *this;
}
void Allocation::reset() noexcept {
    if (pointer_) allocator_->release_block(pointer_);
    pointer_ = nullptr;
    requested_ = capacity_ = 0;
    allocator_.reset();
}

}  // namespace orthon
