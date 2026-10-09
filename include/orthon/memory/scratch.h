#pragma once
#include <memory>
#include <mutex>
#include <orthon/core/memory_pool.h>
#include <orthon/memory/allocator.h>
#include <orthon/memory/raw_allocator.h>
#include <orthon/runtime/completion.h>
#include <vector>

namespace orthon {

class ScratchArena;

// Exclusive host lease. Release records queue completion; later checkout is
// stream ordered and raw reclamation waits for the last consumer.
class ScratchLease {
  public:
    ScratchLease() = default;
    ~ScratchLease();
    ScratchLease(const ScratchLease &) = delete;
    ScratchLease &operator=(const ScratchLease &) = delete;
    ScratchLease(ScratchLease &&other) noexcept;
    ScratchLease &operator=(ScratchLease &&other) noexcept;
    void *allocate(size_t nbytes);
    size_t capacity() const {
        return allocation_.capacity();
    }
    size_t used_bytes() const {
        return used_;
    }
    Device device() const {
        return allocation_.device();
    }

  private:
    friend class ScratchArena;
    ScratchLease(std::shared_ptr<ScratchArena> arena, Allocation allocation,
                 CudaStream queue) noexcept;
    void release() noexcept;
    std::shared_ptr<ScratchArena> arena_;
    Allocation allocation_;
    size_t used_ = 0;
    CudaStream queue_;
};

class ScratchArena : public std::enable_shared_from_this<ScratchArena> {
  public:
    explicit ScratchArena(Device device, std::shared_ptr<RawAllocator> raw = system_raw_allocator(),
                          CudaStream queue = {});
    ~ScratchArena();
    ScratchLease acquire(size_t required_bytes);
    ScratchLease acquire(size_t required_bytes, CudaStream queue);
    ScratchPoolStats stats() const;
    void *legacy_allocate(size_t nbytes);
    void legacy_reset();

  private:
    friend class ScratchLease;
    Allocation take_locked(size_t required, const CudaStream &queue);
    void return_locked(Allocation allocation, size_t used, CompletionToken completion) noexcept;
    void release(Allocation allocation, size_t used, const CudaStream &queue) noexcept;
    void *advance(ScratchLease &lease, size_t nbytes);
    std::shared_ptr<Allocator> allocator_;
    mutable std::mutex mutex_;
    struct IdleRegion {
        Allocation allocation;
        CompletionToken completion;
    };
    std::vector<IdleRegion> idle_;
    CudaStream queue_;
    Allocation legacy_;
    size_t legacy_used_ = 0;
    ScratchPoolStats stats_;
};

std::shared_ptr<ScratchArena> scratch_arena_for(Device device);

} // namespace orthon
