#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/scratch.h>
#include <orthon/runtime/cuda_context.h>
#include <orthon/utils/logging.h>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace orthon {

namespace {
#ifdef WITH_CUDA
void check_cuda(cudaError_t status, const char *operation = nullptr) {
    if (status != cudaSuccess)
        throw std::runtime_error(
            (operation ? std::string(operation) + ": CUDA error: " : std::string("CUDA error: ")) +
            cudaGetErrorString(status));
}
void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(status)));
}
class SystemCudaResources final : public CudaResources {
  public:
    explicit SystemCudaResources(Device device) : device_(device) {
        DeviceGuard guard(device_);
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device_.index));
        capabilities_.compute_major = properties.major;
        capabilities_.compute_minor = properties.minor;
        capabilities_.supports_bfloat16 = properties.major >= 8;
        try {
            check_cublas(cublasCreate(&handle_));
            check_cublas(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
            check_cublas(cublasGetVersion(handle_, &capabilities_.cublas_version));
#if CUDART_VERSION >= 11000 && CUBLAS_VERSION >= 11000
            capabilities_.cublas_bfloat16 =
                capabilities_.supports_bfloat16 && capabilities_.cublas_version >= 11000;
#endif
        } catch (...) {
            destroy();
            throw;
        }
    }
    ~SystemCudaResources() override {
        destroy();
    }
    CudaCapabilities capabilities() const override {
        return capabilities_;
    }
    std::unique_ptr<DeviceGuard> select_device() override {
        return std::make_unique<DeviceGuard>(device_);
    }
    void *cublas_handle() const noexcept override {
        return handle_;
    }
    void bind_compute_stream(void *stream) override {
        check_cublas(cublasSetStream(handle_, static_cast<cudaStream_t>(stream)));
    }
    void synchronize() override {
        DeviceGuard guard(device_);
        check_cuda(cudaDeviceSynchronize());
    }
    void *create_stream() override {
        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return stream;
    }
    void destroy_stream(void *stream) noexcept override {
        if (cudaStreamDestroy(static_cast<cudaStream_t>(stream)) != cudaSuccess)
            try {
                orthon_log_device_fallback("CudaStream", "destroy failed");
            } catch (...) {
            }
    }
    void synchronize_stream(void *stream) override {
        check_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }
    void *create_event(bool timing) override {
        cudaEvent_t event = nullptr;
        check_cuda(
            cudaEventCreateWithFlags(&event, timing ? cudaEventDefault : cudaEventDisableTiming));
        return event;
    }
    void destroy_event(void *event) noexcept override {
        if (cudaEventDestroy(static_cast<cudaEvent_t>(event)) != cudaSuccess)
            try {
                orthon_log_device_fallback("CompletionToken", "destroy failed");
            } catch (...) {
            }
    }
    void record_event(void *event, void *stream) override {
        check_cuda(
            cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
    }
    bool query_event(void *event) override {
        const auto status = cudaEventQuery(static_cast<cudaEvent_t>(event));
        if (status == cudaErrorNotReady)
            return false;
        check_cuda(status);
        return true;
    }
    void wait_event(void *event) override {
        check_cuda(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
    }
    void stream_wait_event(void *stream, void *event) override {
        check_cuda(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                                       static_cast<cudaEvent_t>(event), 0));
    }
    float elapsed_events(void *start, void *stop) override {
        float ms = 0;
        check_cuda(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start),
                                        static_cast<cudaEvent_t>(stop)));
        return ms;
    }
    void destroy_cublas() noexcept override {
        auto handle = std::exchange(handle_, nullptr);
        if (!handle)
            return;
        try {
            DeviceGuard guard(device_);
            if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
                orthon_log_device_fallback("CudaContext", "cuBLAS destroy failed");
        } catch (...) {
            try {
                orthon_log_device_fallback("CudaContext", "cuBLAS teardown device unavailable");
            } catch (...) {
            }
        }
    }

  private:
    void destroy() noexcept {
        destroy_cublas();
    }
    Device device_;
    CudaCapabilities capabilities_;
    cublasHandle_t handle_ = nullptr;
};
#endif
std::unique_ptr<CudaResources> system_cuda_resources(Device device) {
#ifdef WITH_CUDA
    return std::make_unique<SystemCudaResources>(device);
#else
    (void)device;
    throw std::runtime_error(
        "CudaContext: CUDA is not available in this build (recompile with -DWITH_CUDA=ON)");
#endif
}
} // namespace

struct CudaContext::CublasState {
    explicit CublasState(CudaStream queue) : compute(std::move(queue)) {}
    CudaStream compute;
};
CudaContext::CudaContext(Device device, std::shared_ptr<MemoryPool> pool,
                         std::shared_ptr<RawAllocator> raw, CudaResourcesFactory factory)
    : device_(device) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("CudaContext: expected an exact CUDA device");
    auto resources = factory ? factory(device) : system_cuda_resources(device);
    if (!resources)
        throw std::invalid_argument("CudaContext: null CUDA resources");
    capabilities_ = resources->capabilities();
    auto owner = std::make_shared<CudaResourceState>(device, std::move(resources));
    compute_ = CudaStream::canonical(owner);
    cublas_ = std::make_shared<CublasState>(compute_);
    allocator_ = std::make_shared<CUDAAllocator>(device, std::move(pool), compute_);
    scratch_ = std::make_shared<ScratchArena>(device, std::move(raw), compute_);
}
CudaContext::~CudaContext() = default;
CudaContext::CublasLease::CublasLease(std::shared_ptr<CublasState> state)
    : state_(std::move(state)),
      guard_(state_->compute.resource_owner()->resources->select_device()),
      lock_(state_->compute.resource_owner()->cublas_mutex) {
    state_->compute.resource_owner()->resources->bind_compute_stream(
        state_->compute.native_handle());
}
CudaContext::CublasLease::~CublasLease() = default;
void *CudaContext::CublasLease::handle() const noexcept {
    return state_->compute.resource_owner()->resources->cublas_handle();
}
CudaContext::CublasLease CudaContext::acquire_cublas() {
    return CublasLease(cublas_);
}
CudaStream CudaContext::compute_stream() const {
    return compute_;
}
CudaStream CudaContext::create_stream() const {
    return compute_.create_stream();
}
void CudaContext::synchronize_compute() {
    compute_.synchronize();
}
void CudaContext::synchronize() {
    auto guard = compute_.resource_owner()->resources->select_device();
    compute_.resource_owner()->resources->synchronize();
}
void CudaContext::copy_h2d_snapshot(void *dst, const void *src, size_t bytes) {
    if (!bytes)
        return;
#ifdef WITH_CUDA
    DeviceGuard guard(device_);
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(compute_.native_handle())));
    synchronize_compute();
#else
    throw std::runtime_error("CUDA transfer: CUDA is not available in this build");
#endif
}
void CudaContext::copy_d2h_blocking(void *dst, const void *src, size_t bytes) {
    if (!bytes)
        return;
#ifdef WITH_CUDA
    DeviceGuard guard(device_);
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                               static_cast<cudaStream_t>(compute_.native_handle())));
    synchronize_compute();
#else
    throw std::runtime_error("CUDA transfer: CUDA is not available in this build");
#endif
}
void CudaContext::copy_d2d_async(void *dst, const void *src, size_t bytes, const char *operation) {
    if (!bytes)
        return;
#ifdef WITH_CUDA
    DeviceGuard guard(device_);
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                               static_cast<cudaStream_t>(compute_.native_handle())),
               operation);
#else
    throw std::runtime_error("CUDA transfer: CUDA is not available in this build");
#endif
}
void CudaContext::copy_d2d_2d_async(void *dst, size_t dst_pitch, const void *src, size_t src_pitch,
                                    size_t width, size_t height, const char *operation) {
    if (!width || !height)
        return;
#ifdef WITH_CUDA
    DeviceGuard guard(device_);
    check_cuda(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height,
                                 cudaMemcpyDeviceToDevice,
                                 static_cast<cudaStream_t>(compute_.native_handle())),
               operation);
#else
    throw std::runtime_error("CUDA transfer: CUDA is not available in this build");
#endif
}
void CudaContext::memset_async(void *dst, int value, size_t bytes, const char *operation) {
    if (!bytes)
        return;
#ifdef WITH_CUDA
    DeviceGuard guard(device_);
    check_cuda(
        cudaMemsetAsync(dst, value, bytes, static_cast<cudaStream_t>(compute_.native_handle())),
        operation);
#else
    throw std::runtime_error("CUDA transfer: CUDA is not available in this build");
#endif
}
void CudaContext::peer_copy_blocking(void *dst, CudaContext &source, const void *src,
                                     size_t bytes) {
    if (!bytes)
        return;
#ifdef WITH_CUDA
    source.synchronize_compute();
    DeviceGuard guard(device_);
    check_cuda(cudaMemcpyPeerAsync(dst, device_.index, src, source.device().index, bytes,
                                   static_cast<cudaStream_t>(compute_.native_handle())));
    synchronize_compute(); // Source storage cannot be released until the copy finishes.
#else
    throw std::runtime_error("CUDA transfer: CUDA is not available in this build");
#endif
}
} // namespace orthon
