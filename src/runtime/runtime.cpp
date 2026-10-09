#include <orthon/runtime/runtime.h>
#include <orthon/memory/pool.h>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace orthon {

Runtime::Runtime(std::shared_ptr<RawAllocator> raw, CudaResourcesFactory factory)
    : raw_(std::move(raw)), pool_(std::make_shared<MemoryPool>(raw_)),
      cuda_factory_(std::move(factory)) {}
Runtime::~Runtime() = default;

std::shared_ptr<CpuContext> Runtime::cpu() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cpu_) cpu_ = std::make_shared<CpuContext>(pool_, raw_);
    return cpu_;
}
std::shared_ptr<CudaContext> Runtime::cuda(int ordinal) {
    const Device device(DeviceType::CUDA, ordinal);
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = cuda_.find(ordinal);
    if (found != cuda_.end()) return found->second;
    // Publish only after every resource is initialized. Any failure leaves
    // the key absent and all acquired resources unwind; a later call retries.
    auto context = std::make_shared<CudaContext>(device, pool_, raw_, cuda_factory_);
    cuda_.emplace(ordinal, context);
    if (default_process_ && std::atexit(&Runtime::shutdown_default_cuda) != 0) {
        cuda_.erase(ordinal);
        throw std::runtime_error("Runtime: cannot register CUDA shutdown");
    }
    return context;
}
std::shared_ptr<Allocator> Runtime::allocator(Device device) {
    device = Device(device.type, device.index);
    if (device.is_cpu()) return cpu()->allocator();
    return cuda(device.index)->allocator();
}
std::shared_ptr<ScratchArena> Runtime::scratch(Device device) {
    device = Device(device.type, device.index);
    if (device.is_cpu()) return cpu()->scratch();
    return cuda(device.index)->scratch();
}
void Runtime::synchronize_current_cuda() noexcept {
#ifdef WITH_CUDA
    try {
        int ordinal = 0;
        if (cudaGetDevice(&ordinal) == cudaSuccess) cuda(ordinal)->synchronize();
    } catch (...) {
        // Preserve the existing best-effort, no-argument Python helper.
    }
#endif
}
Runtime& default_runtime() {
    // The ordinary owner still has explicit RAII. A CUDA shutdown callback is
    // registered AFTER each device's library initialization, so it precedes
    // CUDA's unloading callbacks regardless of initial CPU-only use.
    static auto runtime = [] {
        auto owner = std::make_unique<Runtime>();
        owner->default_process_ = true;
        return owner;
    }();
    return *runtime;
}
void Runtime::shutdown_default_cuda() noexcept {
    try {
        auto& runtime = default_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex_);
        runtime.cuda_.clear();
        // Release reusable bytes while CUDA is still available. Checked-out
        // records and externally held contexts keep their own shared owners.
        runtime.pool_->clear_cache();
    } catch (...) {
        // Process shutdown must never throw. Isolated Runtime teardown uses
        // the same context RAII directly and requires no process callback.
    }
}

}  // namespace orthon
