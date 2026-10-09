#pragma once
#include <orthon/runtime/cuda_stream.h>
// Host-only injectable defaults for metadata tests. No physical CUDA claim.
class TestCudaResources : public orthon::CudaResources {
  public:
    orthon::CudaCapabilities capabilities() const override {
        return {};
    }
    std::unique_ptr<orthon::DeviceGuard> select_device() override {
        return {};
    }
    void *cublas_handle() const noexcept override {
        return const_cast<TestCudaResources *>(this);
    }
    void bind_compute_stream(void *) override {}
    void synchronize() override {}
    void *create_stream() override {
        return new int(0);
    }
    void destroy_stream(void *stream) noexcept override {
        delete static_cast<int *>(stream);
    }
    void synchronize_stream(void *) override {}
    void *create_event(bool) override {
        return new int(0);
    }
    void destroy_event(void *event) noexcept override {
        delete static_cast<int *>(event);
    }
    void record_event(void *, void *) override {}
    bool query_event(void *) override {
        return true;
    }
    void wait_event(void *) override {}
    void stream_wait_event(void *, void *) override {}
    float elapsed_events(void *, void *) override {
        return 0;
    }
    void destroy_cublas() noexcept override {}
};
inline orthon::CudaStream host_test_stream(orthon::Device device) {
    return orthon::CudaStream::canonical(
        std::make_shared<orthon::CudaResourceState>(device, std::make_unique<TestCudaResources>()));
}
