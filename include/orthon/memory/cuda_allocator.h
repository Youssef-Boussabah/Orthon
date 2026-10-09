#pragma once
#include <orthon/memory/allocator.h>
#include <orthon/runtime/cuda_stream.h>

namespace orthon {

class MemoryPool;
class CUDAAllocator : public Allocator {
  public:
    explicit CUDAAllocator(Device device);
    CUDAAllocator(Device device, std::shared_ptr<MemoryPool> pool);
    CUDAAllocator(Device device, std::shared_ptr<MemoryPool> pool, CudaStream queue);
    Device device() const override {
        return device_;
    }

  protected:
    AllocationBlock allocate_block(size_t nbytes) override;
    void release_block(void *pointer) noexcept override;

  private:
    Device device_;
    std::shared_ptr<MemoryPool> pool_;
    CudaStream queue_;
};

} // namespace orthon
