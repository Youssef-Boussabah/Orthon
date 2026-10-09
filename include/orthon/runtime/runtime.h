#pragma once
#include <orthon/runtime/cpu_context.h>
#include <orthon/runtime/cuda_context.h>
#include <orthon/memory/raw_allocator.h>
#include <map>
#include <memory>
#include <mutex>

namespace orthon {

class Runtime {
public:
    explicit Runtime(std::shared_ptr<RawAllocator> raw = system_raw_allocator(),
                     CudaResourcesFactory cuda_factory = {});
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    std::shared_ptr<CpuContext> cpu();
    std::shared_ptr<CudaContext> cuda(int ordinal);
    std::shared_ptr<Allocator> allocator(Device device);
    std::shared_ptr<ScratchArena> scratch(Device device);
    std::shared_ptr<MemoryPool> memory_pool() const { return pool_; }
    // Compatibility policy: current physical device only, CPU no-op, CUDA
    // failures ignored just as the existing Python timing helper did.
    void synchronize_current_cuda() noexcept;
private:
    friend Runtime& default_runtime();
    static void shutdown_default_cuda() noexcept;
    bool default_process_ = false;
    std::shared_ptr<RawAllocator> raw_;
    std::shared_ptr<MemoryPool> pool_;
    CudaResourcesFactory cuda_factory_;
    std::mutex mutex_;
    std::shared_ptr<CpuContext> cpu_;
    std::map<int, std::shared_ptr<CudaContext>> cuda_;
};

Runtime& default_runtime();

}  // namespace orthon
