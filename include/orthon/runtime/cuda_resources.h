#pragma once
#include <functional>
#include <memory>
#include <orthon/device/device_guard.h>

namespace orthon {
struct CudaCapabilities {
    int compute_major = -1;
    int compute_minor = -1;
    int cublas_version = -1;
    bool supports_bfloat16 = false;
    bool cublas_bfloat16 = false;
};

// Internal native injection boundary. The system implementation alone owns
// CUDA library resources; host tests can exercise lifetime/failure/map rules.
// Opaque handle/stream identities keep CPU consumers free of CUDA headers.
class CudaResources {
  public:
    virtual ~CudaResources() = default;
    virtual CudaCapabilities capabilities() const = 0;
    virtual std::unique_ptr<DeviceGuard> select_device() = 0;
    virtual void *cublas_handle() const noexcept = 0;
    virtual void bind_compute_stream(void *stream) = 0;
    virtual void synchronize() = 0;
    virtual void *create_stream() = 0;
    virtual void destroy_stream(void *stream) noexcept = 0;
    virtual void synchronize_stream(void *stream) = 0;
    virtual void *create_event(bool timing) = 0;
    virtual void destroy_event(void *event) noexcept = 0;
    virtual void record_event(void *event, void *stream) = 0;
    virtual bool query_event(void *event) = 0;
    virtual void wait_event(void *event) = 0;
    virtual void stream_wait_event(void *stream, void *event) = 0;
    virtual float elapsed_events(void *start, void *stop) = 0;
    virtual void destroy_cublas() noexcept = 0;
};
using CudaResourcesFactory = std::function<std::unique_ptr<CudaResources>(Device)>;

} // namespace orthon
