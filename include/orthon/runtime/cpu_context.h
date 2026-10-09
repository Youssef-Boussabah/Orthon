#pragma once
#include <orthon/device/device.h>
#include <memory>

namespace orthon {

class Allocator;
class MemoryPool;
class RawAllocator;
class ScratchArena;

class CpuContext {
public:
    CpuContext(std::shared_ptr<MemoryPool> pool, std::shared_ptr<RawAllocator> raw);
    Device device() const noexcept { return Device(); }
    std::shared_ptr<Allocator> allocator() const { return allocator_; }
    std::shared_ptr<ScratchArena> scratch() const { return scratch_; }
private:
    std::shared_ptr<Allocator> allocator_;
    std::shared_ptr<ScratchArena> scratch_;
};

}  // namespace orthon
