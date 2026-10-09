#include <orthon/runtime/cpu_context.h>
#include <orthon/memory/cpu_allocator.h>
#include <orthon/memory/scratch.h>

namespace orthon {

CpuContext::CpuContext(std::shared_ptr<MemoryPool> pool, std::shared_ptr<RawAllocator> raw)
    : allocator_(std::make_shared<CPUAllocator>(std::move(pool))),
      scratch_(std::make_shared<ScratchArena>(Device(), std::move(raw))) {}

}  // namespace orthon
