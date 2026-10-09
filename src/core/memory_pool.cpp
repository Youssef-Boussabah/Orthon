#include <orthon/core/memory_pool.h>
#include <orthon/runtime/runtime.h>
#include <orthon/memory/pool.h>
#include <orthon/memory/scratch.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace orthon {

namespace {
// The pointer-returning compatibility API retains its actual RAII owners.
// Buffer and operator code never use this registry.
struct LegacyAllocations {
    std::mutex mutex;
    std::unordered_map<void*, Allocation> owners;
};
LegacyAllocations& legacy_allocations() { static LegacyAllocations owners; return owners; }
}
void* memory_pool_allocate(size_t nbytes, const Device& device) {
    Allocation allocation = default_runtime().allocator(device)->allocate(nbytes);
    void* pointer = allocation.data();
    if (!pointer) return nullptr;
    auto& legacy = legacy_allocations();
    std::lock_guard<std::mutex> lock(legacy.mutex);
    legacy.owners.emplace(pointer, std::move(allocation));
    return pointer;
}
void memory_pool_deallocate(void* pointer, const Device& device) {
    if (!pointer) return;
    (void)device;  // Release on the recorded allocation device, as before.
    Allocation owner;
    auto& legacy = legacy_allocations();
    {
        std::lock_guard<std::mutex> lock(legacy.mutex);
        auto found = legacy.owners.find(pointer);
        if (found == legacy.owners.end())
            throw std::invalid_argument("memory_pool_deallocate: pointer is not a compatibility allocation");
        owner = std::move(found->second);
        legacy.owners.erase(found);
    }
}
void memory_pool_set_enabled(bool enabled) { default_runtime().memory_pool()->set_enabled(enabled); }
bool memory_pool_enabled() { return default_runtime().memory_pool()->enabled(); }
void memory_pool_set_max_cached_bytes(size_t bytes) { default_runtime().memory_pool()->set_max_cached_bytes(bytes); }
size_t memory_pool_max_cached_bytes() { return default_runtime().memory_pool()->max_cached_bytes(); }
MemoryPoolStats memory_pool_stats() { return default_runtime().memory_pool()->stats(); }
void memory_pool_reset_stats() { default_runtime().memory_pool()->reset_stats(); }
void memory_pool_clear_cache() { default_runtime().memory_pool()->clear_cache(); }
void* memory_pool_scratch_allocate(size_t bytes, const Device& device) {
    return default_runtime().scratch(device)->legacy_allocate(bytes);
}
void memory_pool_reset_scratch(const Device& device) { default_runtime().scratch(device)->legacy_reset(); }
ScratchPoolStats memory_pool_scratch_stats(const Device& device) { return default_runtime().scratch(device)->stats(); }

}  // namespace orthon
