#pragma once
#include <cstddef>
#include <orthon/device/device.h>
#include <orthon/memory/allocator.h>

namespace orthon {

class MemoryPool;
class CPUAllocator : public Allocator {
public:
    CPUAllocator();
    explicit CPUAllocator(std::shared_ptr<MemoryPool> pool);
    Device device() const override;
protected:
    AllocationBlock allocate_block(size_t nbytes) override;
    void release_block(void* pointer) noexcept override;
private:
    std::shared_ptr<MemoryPool> pool_;
};

}  // namespace orthon
