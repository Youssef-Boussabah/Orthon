#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <orthon/device/device.h>
#include <orthon/memory/allocation.h>

namespace orthon {

// Tensor-facing physical identity. All aliases retain this same object.
// Allocation directly owns checked-out bytes; queries never duplicate state.
class Buffer {
public:
    Buffer(size_t nbytes, Device device);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* data() const;
    size_t nbytes() const;
    size_t capacity() const;
    const Allocation& allocation() const { return allocation_; }
    Device device() const;

    // Successful logical writes count once, including same-value/empty writes.
    // Metadata-only operations do not count. Async collectives bump on enqueue.
    uint64_t version() const { return generation_.load(std::memory_order_relaxed); }
    void bump_version() { generation_.fetch_add(1, std::memory_order_relaxed); }

private:
    Allocation allocation_;
    std::atomic<uint64_t> generation_{0};
};

}  // namespace orthon
