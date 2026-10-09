#pragma once
#include <map>
#include <mutex>
#include <orthon/core/memory_pool.h>
#include <orthon/memory/allocator.h>
#include <orthon/memory/raw_allocator.h>
#include <orthon/runtime/completion.h>
#include <unordered_map>
#include <vector>

namespace orthon {

// One process-wide budget with exact-device size-class caches. Live entries
// are non-owning accounting records; only free lists own reusable blocks.
class MemoryPool {
  public:
    explicit MemoryPool(std::shared_ptr<RawAllocator> raw = system_raw_allocator());
    ~MemoryPool();
    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;
    AllocationBlock acquire(size_t requested, Device device, CudaStream queue = {});
    void release(void *pointer) noexcept;
    void set_enabled(bool enabled);
    bool enabled() const;
    void set_max_cached_bytes(size_t bytes);
    size_t max_cached_bytes() const;
    MemoryPoolStats stats() const;
    void reset_stats();
    void clear_cache();

  private:
    struct Key {
        DeviceType type;
        int index;
        size_t capacity;
        bool operator<(const Key &other) const;
    };
    struct Live {
        size_t requested;
        Key key;
        CudaStream queue;
    };
    struct CachedBlock {
        void *pointer;
        CompletionToken completion;
    };
    void evict_locked(size_t limit) noexcept;
    std::shared_ptr<RawAllocator> raw_;
    mutable std::mutex mutex_;
    bool enabled_ = true;
    size_t max_cached_bytes_ = static_cast<size_t>(-1);
    MemoryPoolStats stats_;
    std::map<Key, std::vector<CachedBlock>> cache_;
    std::unordered_map<void *, Live> live_;
};
std::shared_ptr<MemoryPool> default_memory_pool();

} // namespace orthon
