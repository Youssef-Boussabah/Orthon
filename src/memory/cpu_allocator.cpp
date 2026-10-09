#include <orthon/memory/cpu_allocator.h>
#include <orthon/memory/pool.h>
#include <cstdlib>
#include <new>
#include <stdexcept>
#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace orthon {

CPUAllocator::CPUAllocator() : CPUAllocator(default_memory_pool()) {}
CPUAllocator::CPUAllocator(std::shared_ptr<MemoryPool> pool) : pool_(std::move(pool)) {
    if (!pool_) throw std::invalid_argument("CPUAllocator: null pool");
}
Device CPUAllocator::device() const { return Device(DeviceType::CPU, 0); }
AllocationBlock CPUAllocator::allocate_block(size_t nbytes) { return pool_->acquire(nbytes, device()); }
void CPUAllocator::release_block(void* pointer) noexcept { pool_->release(pointer); }

void* cpu_raw_allocate(size_t capacity) {
    if (!capacity) return nullptr;
#ifdef _MSC_VER
    void* pointer = _aligned_malloc(capacity, 64);
#else
    void* pointer = std::aligned_alloc(64, capacity);
#endif
    if (!pointer) throw std::bad_alloc();
    return pointer;
}
void cpu_raw_free(void* pointer) noexcept {
#ifdef _MSC_VER
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

}  // namespace orthon
