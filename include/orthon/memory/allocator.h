#pragma once
#include <memory>
#include <orthon/memory/allocation.h>

namespace orthon {

struct AllocationBlock { void* pointer; size_t capacity; };

class Allocator : public std::enable_shared_from_this<Allocator> {
public:
    virtual ~Allocator() = default;
    Allocation allocate(size_t n_bytes);
    virtual Device device() const = 0;
protected:
    virtual AllocationBlock allocate_block(size_t n_bytes) = 0;
    virtual void release_block(void* pointer) noexcept = 0;
private:
    friend class Allocation;
};

std::shared_ptr<Allocator> allocator_for(const Device& device);
// Non-owning compatibility lookup into the default Runtime context. allocate()
// returns an Allocation retaining shared ownership of that allocator.
Allocator* get_allocator(const Device& device);
class RawAllocator;
// Scratch bypasses Tensor caches and has separate accounting.
std::shared_ptr<Allocator> uncached_allocator_for(
    const Device& device, std::shared_ptr<RawAllocator> raw);

}  // namespace orthon
