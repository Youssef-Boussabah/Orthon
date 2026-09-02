#pragma once

#include <cstddef>
#include <cstdint>
#include <orthon/device/device.h>

namespace orthon {

struct MemoryPoolStats {
    uint64_t total_allocations = 0;   // Allocation requests.
    uint64_t raw_allocations   = 0;   // Calls to malloc/cudaMalloc.
    uint64_t reused_allocations = 0;  // Blocks served from a free list.
    size_t current_bytes = 0;         // Bytes currently checked out.
    size_t peak_bytes = 0;            // Peak checked-out bytes.
    size_t cached_bytes = 0;          // Bytes currently held in free lists.
};

struct ScratchPoolStats {
    size_t reserved_bytes = 0;        // Bytes reserved in the reusable workspace.
    size_t used_bytes = 0;            // Bytes consumed since the last reset.
    size_t peak_used_bytes = 0;       // Peak bytes consumed in this workspace.
};

void* memory_pool_allocate(size_t n_bytes, const Device& device);
void  memory_pool_deallocate(void* ptr, const Device& device);

void* memory_pool_scratch_allocate(size_t n_bytes, const Device& device);
void memory_pool_reset_scratch(const Device& device);
ScratchPoolStats memory_pool_scratch_stats(const Device& device);

void memory_pool_set_enabled(bool enabled);
bool memory_pool_enabled();

void memory_pool_set_max_cached_bytes(size_t max_bytes);
size_t memory_pool_max_cached_bytes();

MemoryPoolStats memory_pool_stats();
void memory_pool_reset_stats();
void memory_pool_clear_cache();

}  // namespace orthon
