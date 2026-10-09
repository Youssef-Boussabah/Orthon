#include <orthon/runtime/completion.h>
#include <orthon/runtime/cuda_stream.h>
#include <orthon/utils/logging.h>
#include <stdexcept>

namespace orthon {
namespace {
void diagnose(const char *message) noexcept {
    try {
        orthon_log_device_fallback("CudaStream", message);
    } catch (...) {
    }
}
} // namespace
CudaResourceState::CudaResourceState(Device device_, std::unique_ptr<CudaResources> resources_)
    : device(device_), resources(std::move(resources_)) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("CudaStream: expected an exact CUDA device");
    if (!resources)
        throw std::invalid_argument("CudaContext: null CUDA resources");
    auto guard = resources->select_device();
    compute = resources->create_stream();
    if (!compute)
        throw std::runtime_error("CudaContext: null compute stream");
    try {
        resources->bind_compute_stream(compute);
    } catch (...) {
        // No enqueue has occurred. Unwind the unpublished handle/queue pair.
        resources->destroy_cublas();
        resources->destroy_stream(compute);
        compute = nullptr;
        throw;
    }
}
CudaResourceState::~CudaResourceState() noexcept {
    // All leases, streams, allocations and events have dropped their references.
    // Explicit order: complete work, destroy library handle, destroy compute queue.
    try {
        auto guard = resources->select_device();
        try {
            resources->synchronize_stream(compute);
        } catch (...) {
            resources->synchronize();
        }
        resources->destroy_cublas();
        resources->destroy_stream(compute);
    } catch (...) {
        diagnose("resource teardown could not establish completion");
        std::terminate();
    }
}
struct CudaStream::State {
    std::shared_ptr<CudaResourceState> owner;
    void *handle = nullptr;
    bool auxiliary = false;
    ~State() noexcept {
        if (!auxiliary)
            return;
        try {
            auto guard = owner->resources->select_device();
            try {
                owner->resources->synchronize_stream(handle);
            } catch (...) {
                owner->resources->synchronize();
            }
            owner->resources->destroy_stream(handle);
        } catch (...) {
            diagnose("auxiliary teardown could not establish completion");
            std::terminate();
        }
    }
};
CudaStream CudaStream::canonical(std::shared_ptr<CudaResourceState> owner) {
    if (!owner)
        throw std::invalid_argument("CudaStream: null resource owner");
    auto state = std::make_shared<State>();
    state->handle = owner->compute;
    state->owner = std::move(owner);
    return CudaStream(std::move(state));
}
CudaStream CudaStream::create_stream() const {
    auto owner = resource_owner();
    auto state = std::make_shared<State>();
    state->owner = owner;
    auto guard = owner->resources->select_device();
    state->handle = owner->resources->create_stream();
    if (!state->handle)
        throw std::runtime_error("CudaStream: null auxiliary stream");
    state->auxiliary = true;
    return CudaStream(std::move(state));
}
const std::shared_ptr<CudaResourceState> &CudaStream::resource_owner() const {
    if (!state_)
        throw std::logic_error("CudaStream: no queue");
    return state_->owner;
}
Device CudaStream::device() const {
    return resource_owner()->device;
}
void *CudaStream::native_handle() const {
    auto guard = resource_owner()->resources->select_device();
    return state_->handle;
}
void CudaStream::synchronize() const {
    auto guard = resource_owner()->resources->select_device();
    resource_owner()->resources->synchronize_stream(state_->handle);
}
void CudaStream::synchronize_noexcept() const noexcept {
    if (!state_)
        return;
    try {
        synchronize();
        return;
    } catch (...) {
        diagnose("stream wait failed; using device fallback");
    }
    try {
        auto guard = resource_owner()->resources->select_device();
        resource_owner()->resources->synchronize();
    } catch (...) {
        diagnose("completion cannot be established; refusing unsafe reclamation");
        std::terminate();
    }
}
bool CudaStream::same_queue(const CudaStream &other) const noexcept {
    return state_ && other.state_ && state_->owner == other.state_->owner &&
           state_->handle == other.state_->handle;
}
CompletionToken CudaStream::record_completion(bool timing) const {
    return CompletionToken(*this, timing);
}
CompletionToken CudaStream::record_completion_noexcept() const noexcept {
    if (!state_)
        return {};
    try {
        return record_completion();
    } catch (...) {
        diagnose("completion recording failed; waiting before retirement");
        synchronize_noexcept();
        return {};
    }
}
} // namespace orthon
