#pragma once
#include <orthon/runtime/cuda_stream.h>

namespace orthon {
// Empty is immediately complete (CPU or successfully synchronized fallback).
// Shared event state is immutable after recording; no observer sees rerecording.
class CompletionToken {
    struct State;

  public:
    CompletionToken() = default;
    bool ready() const;
    void wait_host() const;
    void wait_host_noexcept() const noexcept;
    void enqueue_wait_on(const CudaStream &target) const;
    bool same_ordering_domain(const CudaStream &target) const noexcept;
    Device device() const;
    float elapsed_since(const CompletionToken &start) const;

  private:
    friend class CudaStream;
    explicit CompletionToken(const CudaStream &stream, bool timing);
    std::shared_ptr<State> state_;
};
// Diagnostic timing owner; separate from immutable memory retirement tokens.
class CudaTimingEvent {
    struct State;

  public:
    explicit CudaTimingEvent(CudaStream stream);
    ~CudaTimingEvent();
    CudaTimingEvent(CudaTimingEvent &&) noexcept;
    CudaTimingEvent &operator=(CudaTimingEvent &&) noexcept;
    CudaTimingEvent(const CudaTimingEvent &) = delete;
    CudaTimingEvent &operator=(const CudaTimingEvent &) = delete;
    void record();
    void wait_host() const;
    float elapsed_since(const CudaTimingEvent &start) const;

  private:
    std::unique_ptr<State> state_;
};
} // namespace orthon
