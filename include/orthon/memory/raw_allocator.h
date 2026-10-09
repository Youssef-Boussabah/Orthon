#pragma once
#include <cstddef>
#include <memory>
#include <orthon/device/device.h>

namespace orthon {

// Injectable low-level backend. Only allocation implementations consume
// these owning pointers; Tensor and operator code use Allocation records.
class RawAllocator {
public:
    virtual ~RawAllocator() = default;
    virtual void* allocate(size_t capacity, Device device) = 0;
    virtual void deallocate(void* pointer, Device device) noexcept = 0;
};
std::shared_ptr<RawAllocator> system_raw_allocator();
size_t allocation_size_class(size_t nbytes);
size_t checked_allocation_add(size_t a, size_t b);
void* cpu_raw_allocate(size_t capacity);
void cpu_raw_free(void* pointer) noexcept;
void* cuda_raw_allocate(size_t capacity, Device device);
void cuda_raw_free(void* pointer, Device device) noexcept;

}  // namespace orthon
