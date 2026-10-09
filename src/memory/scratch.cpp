#include <algorithm>
#include <limits>
#include <orthon/memory/scratch.h>
#include <orthon/runtime/runtime.h>
#include <stdexcept>
#include <utility>

namespace orthon {

ScratchLease::ScratchLease(std::shared_ptr<ScratchArena> arena, Allocation allocation,
                           CudaStream queue) noexcept
    : arena_(std::move(arena)), allocation_(std::move(allocation)), queue_(std::move(queue)) {}
ScratchLease::~ScratchLease() {
    release();
}
ScratchLease::ScratchLease(ScratchLease &&other) noexcept
    : arena_(std::move(other.arena_)), allocation_(std::move(other.allocation_)),
      used_(std::exchange(other.used_, 0)), queue_(std::move(other.queue_)) {}
ScratchLease &ScratchLease::operator=(ScratchLease &&other) noexcept {
    if (this != &other) {
        release();
        arena_ = std::move(other.arena_);
        allocation_ = std::move(other.allocation_);
        used_ = std::exchange(other.used_, 0);
        queue_ = std::move(other.queue_);
    }
    return *this;
}
void ScratchLease::release() noexcept {
    if (arena_)
        arena_->release(std::move(allocation_), used_, queue_);
    used_ = 0;
    arena_.reset();
    queue_ = {};
}
void *ScratchLease::allocate(size_t nbytes) {
    if (!arena_)
        throw std::logic_error("ScratchLease: no active lease");
    return arena_->advance(*this, nbytes);
}
ScratchArena::ScratchArena(Device device, std::shared_ptr<RawAllocator> raw, CudaStream queue)
    : allocator_(uncached_allocator_for(device, std::move(raw))), queue_(std::move(queue)) {
    if (device.is_cuda() && !queue_)
        throw std::invalid_argument("ScratchArena: CUDA arena requires an execution queue");
    if (queue_ && queue_.device() != device)
        throw std::invalid_argument("ScratchArena: queue device mismatch");
}
ScratchArena::~ScratchArena() {
    // No checked-out lease remains (leases retain this arena). Wait before
    // uncached Allocation destructors raw-free idle/legacy regions.
    if (legacy_.data())
        queue_.synchronize_noexcept();
    for (auto &region : idle_)
        region.completion.wait_host_noexcept();
}
Allocation ScratchArena::take_locked(size_t required, const CudaStream &queue) {
    if (!required)
        return allocator_->allocate(0);
    const size_t aligned = allocation_size_class(required);
    // Last released suitable block first, preserving legacy reset/reuse.
    for (size_t i = idle_.size(); i-- > 0;) {
        if (idle_[i].allocation.capacity() >= aligned) {
            idle_[i].completion.enqueue_wait_on(queue);
            Allocation allocation = std::move(idle_[i].allocation);
            idle_.erase(idle_.begin() + i);
            return allocation;
        }
    }
    size_t capacity = 1 << 20;
    while (capacity < aligned) {
        if (capacity > (std::numeric_limits<size_t>::max)() / 2) {
            capacity = aligned;
            break;
        }
        capacity *= 2;
    }
    const size_t next_reserved = checked_allocation_add(stats_.reserved_bytes, capacity);
    Allocation allocation = allocator_->allocate(capacity);
    stats_.reserved_bytes = next_reserved;
    return allocation;
}
ScratchLease ScratchArena::acquire(size_t required_bytes) {
    return acquire(required_bytes, queue_);
}
ScratchLease ScratchArena::acquire(size_t required_bytes, CudaStream queue) {
    if (queue && queue.device() != allocator_->device())
        throw std::invalid_argument("ScratchArena: queue device mismatch");
    if (queue_ && !queue)
        throw std::invalid_argument("ScratchArena: missing CUDA queue");
    auto owner = shared_from_this();
    std::lock_guard<std::mutex> lock(mutex_);
    return ScratchLease(std::move(owner), take_locked(required_bytes, queue), queue);
}
void ScratchArena::return_locked(Allocation allocation, size_t used,
                                 CompletionToken completion) noexcept {
    stats_.used_bytes -= used;
    if (!allocation.capacity())
        return;
    const size_t capacity = allocation.capacity();
    try {
        idle_.reserve(idle_.size() + 1);
        idle_.push_back(IdleRegion{std::move(allocation), completion});
    } catch (...) {
        completion.wait_host_noexcept();
        stats_.reserved_bytes -= capacity;
    }
}
void ScratchArena::release(Allocation allocation, size_t used, const CudaStream &queue) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return_locked(std::move(allocation), used, queue.record_completion_noexcept());
}
void *ScratchArena::advance(ScratchLease &lease, size_t nbytes) {
    if (!nbytes)
        return nullptr;
    const size_t aligned = allocation_size_class(nbytes);
    const size_t next = checked_allocation_add(lease.used_, aligned);
    if (next > lease.capacity())
        throw std::runtime_error("ScratchLease: region exhausted");
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t next_total = checked_allocation_add(stats_.used_bytes, aligned);
    void *pointer = static_cast<unsigned char *>(lease.allocation_.data()) + lease.used_;
    lease.used_ = next;
    stats_.used_bytes = next_total;
    stats_.peak_used_bytes = std::max(stats_.peak_used_bytes, next_total);
    return pointer;
}
ScratchPoolStats ScratchArena::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}
void *ScratchArena::legacy_allocate(size_t nbytes) {
    if (!nbytes)
        return nullptr;
    const size_t aligned = allocation_size_class(nbytes);
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t next = checked_allocation_add(legacy_used_, aligned);
    if (next > legacy_.capacity()) {
        if (legacy_used_)
            throw std::runtime_error(
                "memory_pool scratch: workspace exhausted; call reset_scratch before growing");
        // Acquiring replacement can throw: keep all previous ownership intact.
        Allocation replacement = take_locked(aligned, queue_);
        return_locked(std::move(legacy_), 0, queue_.record_completion_noexcept());
        legacy_ = std::move(replacement);
    }
    const size_t next_total = checked_allocation_add(stats_.used_bytes, aligned);
    void *pointer = static_cast<unsigned char *>(legacy_.data()) + legacy_used_;
    legacy_used_ = next;
    stats_.used_bytes = next_total;
    stats_.peak_used_bytes = std::max(stats_.peak_used_bytes, next_total);
    return pointer;
}
void ScratchArena::legacy_reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    return_locked(std::move(legacy_), legacy_used_, queue_.record_completion_noexcept());
    legacy_used_ = 0;
}
std::shared_ptr<ScratchArena> scratch_arena_for(Device device) {
    return default_runtime().scratch(device);
}

} // namespace orthon
