#include <algorithm>
#include <orthon/memory/pool.h>
#include <orthon/runtime/runtime.h>
#include <stdexcept>
#include <tuple>

namespace orthon {

MemoryPool::MemoryPool(std::shared_ptr<RawAllocator> raw) : raw_(std::move(raw)) {
    if (!raw_)
        throw std::invalid_argument("MemoryPool: null raw backend");
}
MemoryPool::~MemoryPool() {
    clear_cache();
}
bool MemoryPool::Key::operator<(const Key &other) const {
    return std::tie(type, index, capacity) < std::tie(other.type, other.index, other.capacity);
}
AllocationBlock MemoryPool::acquire(size_t requested, Device device, CudaStream queue) {
    if (queue && queue.device() != device)
        throw std::invalid_argument("MemoryPool: queue device mismatch");
    if (!requested)
        return {nullptr, 0};
    if (device.is_cuda() && !queue)
        throw std::invalid_argument("MemoryPool: CUDA allocation requires an execution queue");
    const size_t capacity = allocation_size_class(requested);
    const Key key{device.type, device.index, capacity};
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t next_current = checked_allocation_add(stats_.current_bytes, requested);
    auto found = cache_.find(key);
    if (enabled_ && found != cache_.end() && !found->second.empty()) {
        auto &block = found->second.back();
        block.completion.enqueue_wait_on(queue);
        void *pointer = block.pointer;
        // Insertion can throw: keep the block cached until it succeeds.
        live_.emplace(pointer, Live{requested, key, queue});
        found->second.pop_back();
        stats_.cached_bytes -= capacity;
        ++stats_.reused_allocations;
        ++stats_.total_allocations;
        stats_.current_bytes = next_current;
        stats_.peak_bytes = std::max(stats_.peak_bytes, next_current);
        return {pointer, capacity};
    }
    const size_t next_reserved = checked_allocation_add(stats_.reserved_bytes, capacity);
    const auto release_raw = [raw = raw_, device](void *pointer) noexcept {
        raw->deallocate(pointer, device);
    };
    std::unique_ptr<void, decltype(release_raw)> pending(raw_->allocate(capacity, device),
                                                         release_raw);
    void *pointer = pending.get();
    live_.emplace(pointer, Live{requested, key, queue});
    pending.release();
    ++stats_.raw_allocations;
    ++stats_.total_allocations;
    stats_.reserved_bytes = next_reserved;
    stats_.current_bytes = next_current;
    stats_.peak_bytes = std::max(stats_.peak_bytes, next_current);
    return {pointer, capacity};
}
void MemoryPool::release(void *pointer) noexcept {
    if (!pointer)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = live_.find(pointer);
    if (found == live_.end())
        return;
    const Live info = found->second;
    auto completion = info.queue.record_completion_noexcept();
    live_.erase(found);
    stats_.current_bytes -= info.requested;
    if (enabled_) {
        try {
            cache_[info.key].push_back(CachedBlock{pointer, completion});
            stats_.cached_bytes += info.key.capacity;
            evict_locked(max_cached_bytes_);
            return;
        } catch (...) {
            // Release directly if allocating cache metadata fails.
        }
    }
    stats_.reserved_bytes -= info.key.capacity;
    completion.wait_host_noexcept();
    raw_->deallocate(pointer, Device(info.key.type, info.key.index));
}
void MemoryPool::evict_locked(size_t limit) noexcept {
    for (auto &item : cache_) {
        while (!item.second.empty() && stats_.cached_bytes > limit) {
            auto block = std::move(item.second.back());
            block.completion.wait_host_noexcept();
            void *pointer = block.pointer;
            item.second.pop_back();
            stats_.cached_bytes -= item.first.capacity;
            stats_.reserved_bytes -= item.first.capacity;
            raw_->deallocate(pointer, Device(item.first.type, item.first.index));
        }
    }
}
void MemoryPool::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}
bool MemoryPool::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}
void MemoryPool::set_max_cached_bytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_bytes_ = bytes;
    evict_locked(bytes);
}
size_t MemoryPool::max_cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_cached_bytes_;
}
MemoryPoolStats MemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}
void MemoryPool::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_allocations = stats_.raw_allocations = stats_.reused_allocations = 0;
    stats_.peak_bytes = stats_.current_bytes;
}
void MemoryPool::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    evict_locked(0);
    cache_.clear();
}
std::shared_ptr<MemoryPool> default_memory_pool() {
    return default_runtime().memory_pool();
}

} // namespace orthon
