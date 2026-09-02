#include <orthon/core/memory_pool.h>
#include <orthon/device/device_guard.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace orthon {

namespace {

constexpr size_t kCPUAlignment = 64;
constexpr size_t kSizeClassAlignment = 64;
constexpr size_t kScratchInitialBytes = 1 << 20;

struct BlockInfo {
    size_t requested_bytes;
    size_t bin_bytes;
    Device device;
};

struct PoolKey {
    DeviceType type;
    int index;
    size_t n_bytes;

    bool operator==(const PoolKey& other) const {
        return type == other.type && index == other.index && n_bytes == other.n_bytes;
    }
};

struct PoolKeyHash {
    size_t operator()(const PoolKey& key) const {
        size_t h = std::hash<int>()(static_cast<int>(key.type));
        h ^= std::hash<int>()(key.index) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>()(key.n_bytes) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct DeviceKey {
    DeviceType type;
    int index;

    bool operator==(const DeviceKey& other) const {
        return type == other.type && index == other.index;
    }
};

struct DeviceKeyHash {
    size_t operator()(const DeviceKey& key) const {
        size_t h = std::hash<int>()(static_cast<int>(key.type));
        h ^= std::hash<int>()(key.index) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct ScratchWorkspace {
    void* ptr = nullptr;
    size_t reserved_bytes = 0;
    size_t used_bytes = 0;
    size_t peak_used_bytes = 0;
    Device device;
};

std::mutex g_mutex;
bool g_enabled = true;
size_t g_max_cached_bytes = std::numeric_limits<size_t>::max();
MemoryPoolStats g_stats;
std::unordered_map<PoolKey, std::vector<void*>, PoolKeyHash> g_free_lists;
std::unordered_map<void*, BlockInfo> g_live_blocks;
std::unordered_map<DeviceKey, ScratchWorkspace, DeviceKeyHash> g_scratch_workspaces;

PoolKey make_key(size_t n_bytes, const Device& device) {
    return PoolKey{device.type, device.index, n_bytes};
}

DeviceKey make_device_key(const Device& device) {
    return DeviceKey{device.type, device.index};
}

size_t size_class_for(size_t n_bytes) {
    if (n_bytes == 0) return 0;
    if (n_bytes > std::numeric_limits<size_t>::max() - (kSizeClassAlignment - 1)) {
        throw std::bad_alloc();
    }
    return (n_bytes + kSizeClassAlignment - 1) & ~(kSizeClassAlignment - 1);
}

size_t checked_add(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        throw std::bad_alloc();
    }
    return a + b;
}

size_t grow_scratch_capacity(size_t current, size_t required) {
    size_t next = current == 0
        ? std::max<size_t>(kScratchInitialBytes, required)
        : std::max<size_t>(current, kSizeClassAlignment);
    while (next < required) {
        if (next > std::numeric_limits<size_t>::max() / 2) {
            next = required;
            break;
        }
        next *= 2;
    }
    return size_class_for(next);
}

void* raw_cpu_allocate(size_t n_bytes) {
    if (n_bytes == 0) return nullptr;
#ifdef _MSC_VER
    void* ptr = _aligned_malloc(n_bytes, kCPUAlignment);
#else
    const size_t rounded = (n_bytes + kCPUAlignment - 1) & ~(kCPUAlignment - 1);
    void* ptr = std::aligned_alloc(kCPUAlignment, rounded);
#endif
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void raw_cpu_free(void* ptr) {
    if (!ptr) return;
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void* raw_cuda_allocate(size_t n_bytes) {
    if (n_bytes == 0) return nullptr;
#ifdef WITH_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, n_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    return ptr;
#else
    (void)n_bytes;
    throw std::runtime_error(
        "memory_pool: CUDA is not available in this build "
        "(recompile with -DWITH_CUDA=ON)");
#endif
}

void raw_cuda_free(void* ptr) {
    if (!ptr) return;
#ifdef WITH_CUDA
    cudaFree(ptr);
#else
    (void)ptr;
#endif
}

void* raw_allocate(size_t n_bytes, const Device& device) {
    if (device.is_cpu()) return raw_cpu_allocate(n_bytes);
    if (device.is_cuda()) {
        // The guard both selects the device for cudaMalloc and restores the
        // caller's device afterwards. Restoring is what keeps a cache MISS and
        // a cache HIT indistinguishable from outside: without it a miss would
        // reach this function and leave the allocation's device current, while
        // a hit returns earlier and leaves it untouched, so callers could
        // accidentally depend on allocator history for their ambient device.
        DeviceGuard guard(device);
        return raw_cuda_allocate(n_bytes);
    }
    throw std::runtime_error("memory_pool: unsupported device type");
}

void raw_free(void* ptr, const Device& device) {
    if (!ptr) return;
    if (device.is_cpu()) {
        raw_cpu_free(ptr);
        return;
    }
    if (device.is_cuda()) {
        DeviceGuard guard(device);
        raw_cuda_free(ptr);
        return;
    }
    throw std::runtime_error("memory_pool: unsupported device type");
}

void update_peak_locked() {
    g_stats.peak_bytes = std::max(g_stats.peak_bytes, g_stats.current_bytes);
}

void evict_cached_blocks_locked(std::vector<std::pair<void*, Device>>& to_free) {
    if (g_stats.cached_bytes <= g_max_cached_bytes) return;

    for (auto it = g_free_lists.begin();
         it != g_free_lists.end() && g_stats.cached_bytes > g_max_cached_bytes;) {
        const PoolKey key = it->first;
        auto& list = it->second;
        Device device(key.type, key.index);

        while (!list.empty() && g_stats.cached_bytes > g_max_cached_bytes) {
            void* ptr = list.back();
            list.pop_back();
            to_free.emplace_back(ptr, device);
            g_stats.cached_bytes -= key.n_bytes;
        }

        if (list.empty()) {
            it = g_free_lists.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace

void* memory_pool_allocate(size_t n_bytes, const Device& device) {
    if (n_bytes == 0) return nullptr;
    const size_t bin_bytes = size_class_for(n_bytes);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_stats.total_allocations += 1;
        if (g_enabled) {
            PoolKey key = make_key(bin_bytes, device);
            auto& list = g_free_lists[key];
            if (!list.empty()) {
                void* ptr = list.back();
                list.pop_back();
                g_stats.reused_allocations += 1;
                g_stats.current_bytes += n_bytes;
                g_stats.cached_bytes -= bin_bytes;
                g_live_blocks.emplace(ptr, BlockInfo{n_bytes, bin_bytes, device});
                update_peak_locked();
                return ptr;
            }
        }
    }

    void* ptr = raw_allocate(bin_bytes, device);

    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats.raw_allocations += 1;
    g_stats.current_bytes += n_bytes;
    g_live_blocks.emplace(ptr, BlockInfo{n_bytes, bin_bytes, device});
    update_peak_locked();
    return ptr;
}

void* memory_pool_scratch_allocate(size_t n_bytes, const Device& device) {
    if (n_bytes == 0) return nullptr;
    const size_t aligned_bytes = size_class_for(n_bytes);

    std::lock_guard<std::mutex> lock(g_mutex);
    ScratchWorkspace& workspace = g_scratch_workspaces[make_device_key(device)];
    if (!workspace.ptr) {
        workspace.device = device;
    }

    const size_t next_used = checked_add(workspace.used_bytes, aligned_bytes);
    if (next_used > workspace.reserved_bytes) {
        if (workspace.used_bytes != 0) {
            throw std::runtime_error(
                "memory_pool scratch: workspace exhausted; call reset_scratch before growing");
        }

        const size_t new_reserved = grow_scratch_capacity(workspace.reserved_bytes, aligned_bytes);
        void* new_ptr = raw_allocate(new_reserved, device);
        if (workspace.ptr) {
            raw_free(workspace.ptr, workspace.device);
        }
        workspace.ptr = new_ptr;
        workspace.reserved_bytes = new_reserved;
        workspace.device = device;
    }

    auto* base = static_cast<unsigned char*>(workspace.ptr);
    void* ptr = base + workspace.used_bytes;
    workspace.used_bytes = next_used;
    workspace.peak_used_bytes = std::max(workspace.peak_used_bytes, workspace.used_bytes);
    return ptr;
}

void memory_pool_reset_scratch(const Device& device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_scratch_workspaces.find(make_device_key(device));
    if (it != g_scratch_workspaces.end()) {
        it->second.used_bytes = 0;
    }
}

ScratchPoolStats memory_pool_scratch_stats(const Device& device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_scratch_workspaces.find(make_device_key(device));
    if (it == g_scratch_workspaces.end()) {
        return ScratchPoolStats{};
    }
    const ScratchWorkspace& workspace = it->second;
    return ScratchPoolStats{
        workspace.reserved_bytes,
        workspace.used_bytes,
        workspace.peak_used_bytes,
    };
}

void memory_pool_deallocate(void* ptr, const Device& device) {
    if (!ptr) return;

    BlockInfo info{0, 0, device};
    std::vector<std::pair<void*, Device>> to_free;
    bool tracked = false;
    bool cache = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_live_blocks.find(ptr);
        if (it != g_live_blocks.end()) {
            info = it->second;
            tracked = true;
            g_live_blocks.erase(it);
            g_stats.current_bytes -= info.requested_bytes;
            cache = g_enabled;
            if (cache) {
                g_free_lists[make_key(info.bin_bytes, info.device)].push_back(ptr);
                g_stats.cached_bytes += info.bin_bytes;
                evict_cached_blocks_locked(to_free);
            }
        }
    }

    if (!tracked) {
        raw_free(ptr, device);
        return;
    }
    if (!cache) {
        raw_free(ptr, info.device);
    }
    for (auto& item : to_free) {
        raw_free(item.first, item.second);
    }
}

void memory_pool_set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_enabled = enabled;
}

bool memory_pool_enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_enabled;
}

void memory_pool_set_max_cached_bytes(size_t max_bytes) {
    std::vector<std::pair<void*, Device>> to_free;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_max_cached_bytes = max_bytes;
        evict_cached_blocks_locked(to_free);
    }

    for (auto& item : to_free) {
        raw_free(item.first, item.second);
    }
}

size_t memory_pool_max_cached_bytes() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_max_cached_bytes;
}

MemoryPoolStats memory_pool_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_stats;
}

void memory_pool_reset_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats = MemoryPoolStats{};
}

void memory_pool_clear_cache() {
    std::vector<std::pair<void*, Device>> to_free;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto& kv : g_free_lists) {
            const PoolKey& key = kv.first;
            Device device(key.type, key.index);
            for (void* ptr : kv.second) {
                to_free.emplace_back(ptr, device);
            }
        }
        g_free_lists.clear();
        g_stats.cached_bytes = 0;
    }

    for (auto& item : to_free) {
        raw_free(item.first, item.second);
    }
}

}  // namespace orthon
