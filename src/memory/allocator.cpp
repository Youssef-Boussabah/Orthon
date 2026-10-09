#include <orthon/memory/allocator.h>
#include <orthon/memory/raw_allocator.h>
#include <orthon/runtime/runtime.h>
#include <stdexcept>

namespace orthon {

Allocation Allocator::allocate(size_t nbytes) {
    // Retain lifetime BEFORE acquisition; stack allocators cannot hand out
    // ownership records with dangling deleters.
    auto owner = shared_from_this();
    const Device allocation_device = device();
    if (nbytes == 0) return Allocation(nullptr, 0, 0, allocation_device, std::move(owner));
    const AllocationBlock block = allocate_block(nbytes);
    return Allocation(block.pointer, nbytes, block.capacity, allocation_device, std::move(owner));
}
std::shared_ptr<Allocator> allocator_for(const Device& device) {
    return default_runtime().allocator(device);
}
Allocator* get_allocator(const Device& device) { return default_runtime().allocator(device).get(); }

namespace {
class UncachedAllocator final : public Allocator {
public:
    UncachedAllocator(Device device, std::shared_ptr<RawAllocator> raw)
        : device_(device), raw_(std::move(raw)) {
        if (!raw_) throw std::invalid_argument("Allocator: null raw backend");
    }
    Device device() const override { return device_; }
protected:
    AllocationBlock allocate_block(size_t nbytes) override {
        const size_t capacity = allocation_size_class(nbytes);
        return {raw_->allocate(capacity, device_), capacity};
    }
    void release_block(void* pointer) noexcept override { raw_->deallocate(pointer, device_); }
private:
    Device device_;
    std::shared_ptr<RawAllocator> raw_;
};
}
std::shared_ptr<Allocator> uncached_allocator_for(const Device& device,
                                                std::shared_ptr<RawAllocator> raw) {
    return std::make_shared<UncachedAllocator>(device, std::move(raw));
}

}  // namespace orthon
