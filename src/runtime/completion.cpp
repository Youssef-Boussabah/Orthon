#include <orthon/runtime/completion.h>
#include <stdexcept>

namespace orthon {
struct CompletionToken::State {
    CudaStream stream;
    void *event = nullptr;
    bool timing;
    State(CudaStream queue, bool timed) : stream(std::move(queue)), timing(timed) {}
    ~State() noexcept {
        if (!event)
            return;
        // CUDA event destruction retires a pending event asynchronously. No
        // token observer remains; queued event waits retain CUDA's dependency.
        try {
            auto owner = stream.resource_owner();
            auto guard = owner->resources->select_device();
            owner->resources->destroy_event(event);
        } catch (...) {
            std::terminate();
        }
    }
};
CompletionToken::CompletionToken(const CudaStream &stream, bool timing)
    : state_(std::make_shared<State>(stream, timing)) {
    auto owner = stream.resource_owner();
    auto guard = owner->resources->select_device();
    state_->event = owner->resources->create_event(timing);
    if (!state_->event)
        throw std::runtime_error("CompletionToken: null event");
    owner->resources->record_event(state_->event, stream.native_handle());
}
bool CompletionToken::ready() const {
    if (!state_)
        return true;
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    return owner->resources->query_event(state_->event);
}
void CompletionToken::wait_host() const {
    if (!state_)
        return;
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    owner->resources->wait_event(state_->event);
}
void CompletionToken::wait_host_noexcept() const noexcept {
    if (!state_)
        return;
    try {
        wait_host();
    } catch (...) {
        state_->stream.synchronize_noexcept();
    }
}
bool CompletionToken::same_ordering_domain(const CudaStream &target) const noexcept {
    return !state_ || state_->stream.same_queue(target);
}
void CompletionToken::enqueue_wait_on(const CudaStream &target) const {
    if (!state_ || same_ordering_domain(target))
        return;
    if (!target) {
        wait_host();
        return;
    }
    if (target.device() != device())
        throw std::invalid_argument("CompletionToken: different CUDA devices");
    auto owner = target.resource_owner();
    auto guard = owner->resources->select_device();
    // Only queues in one injected/native resource domain may share events.
    if (owner != state_->stream.resource_owner()) {
        wait_host();
        return;
    }
    owner->resources->stream_wait_event(target.native_handle(), state_->event);
}
Device CompletionToken::device() const {
    return state_ ? state_->stream.device() : Device();
}
float CompletionToken::elapsed_since(const CompletionToken &start) const {
    if (!state_ || !start.state_ || !state_->timing || !start.state_->timing ||
        !state_->stream.same_queue(start.state_->stream))
        throw std::invalid_argument("CompletionToken: incompatible timing events");
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    return owner->resources->elapsed_events(start.state_->event, state_->event);
}
struct CudaTimingEvent::State {
    CudaStream stream;
    void *event = nullptr;
    bool recorded = false;
    explicit State(CudaStream queue) : stream(std::move(queue)) {}
    ~State() noexcept {
        if (!event)
            return;
        try {
            auto owner = stream.resource_owner();
            auto guard = owner->resources->select_device();
            owner->resources->destroy_event(event);
        } catch (...) {
            std::terminate();
        }
    }
};
CudaTimingEvent::CudaTimingEvent(CudaStream stream)
    : state_(std::make_unique<State>(std::move(stream))) {
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    state_->event = owner->resources->create_event(true);
    if (!state_->event)
        throw std::runtime_error("CudaTimingEvent: null event");
}
CudaTimingEvent::~CudaTimingEvent() = default;
CudaTimingEvent::CudaTimingEvent(CudaTimingEvent &&) noexcept = default;
CudaTimingEvent &CudaTimingEvent::operator=(CudaTimingEvent &&) noexcept = default;
void CudaTimingEvent::record() {
    if (!state_)
        throw std::logic_error("CudaTimingEvent: moved event");
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    owner->resources->record_event(state_->event, state_->stream.native_handle());
    state_->recorded = true;
}
void CudaTimingEvent::wait_host() const {
    if (!state_ || !state_->recorded)
        throw std::logic_error("CudaTimingEvent: unrecorded event");
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    owner->resources->wait_event(state_->event);
}
float CudaTimingEvent::elapsed_since(const CudaTimingEvent &start) const {
    if (!state_ || !start.state_ || !state_->recorded || !start.state_->recorded ||
        !state_->stream.same_queue(start.state_->stream))
        throw std::invalid_argument("CudaTimingEvent: incompatible events");
    auto owner = state_->stream.resource_owner();
    auto guard = owner->resources->select_device();
    return owner->resources->elapsed_events(start.state_->event, state_->event);
}
} // namespace orthon
