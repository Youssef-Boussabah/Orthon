#pragma once
#include <cstddef>
#include <memory>
#include <mutex>
#include <orthon/runtime/cuda_resources.h>

namespace orthon {
class CompletionToken;
// Shared lower-level owner; it never retains a Runtime, context, allocator or arena.
class CudaResourceState {
  public:
    CudaResourceState(Device device, std::unique_ptr<CudaResources> resources);
    ~CudaResourceState() noexcept;
    const Device device;
    std::unique_ptr<CudaResources> resources;
    void *compute = nullptr;
    std::mutex cublas_mutex;
};

// Value handle to a permanently device-bound queue. Copies share one RAII owner.
class CudaStream {
    struct State;

  public:
    CudaStream() = default;
    static CudaStream canonical(std::shared_ptr<CudaResourceState> owner);
    CudaStream create_stream() const;
    explicit operator bool() const noexcept {
        return bool(state_);
    }
    Device device() const;
    void *native_handle() const;
    void synchronize() const;
    // Used only at noexcept retirement/reclamation boundaries. Stream failure
    // falls back to a checked device wait; unrecoverable failure fails closed.
    void synchronize_noexcept() const noexcept;
    CompletionToken record_completion(bool timing = false) const;
    CompletionToken record_completion_noexcept() const noexcept;
    const std::shared_ptr<CudaResourceState> &resource_owner() const;
    bool same_queue(const CudaStream &other) const noexcept;

  private:
    friend class CompletionToken;
    explicit CudaStream(std::shared_ptr<State> state) : state_(std::move(state)) {}
    std::shared_ptr<State> state_;
};
} // namespace orthon
