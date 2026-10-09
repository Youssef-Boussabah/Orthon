#include <orthon/device/device_guard.h>
#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/pool.h>
#include <orthon/runtime/runtime.h>
#include <orthon/utils/logging.h>
#include <stdexcept>
#include <string>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace orthon {

CUDAAllocator::CUDAAllocator(Device device) : CUDAAllocator(device, default_memory_pool()) {}
CUDAAllocator::CUDAAllocator(Device device, std::shared_ptr<MemoryPool> pool)
    : device_(device), pool_(std::move(pool)) {
    if (!device.is_cuda() || !pool_)
        throw std::invalid_argument("CUDAAllocator: invalid device/pool");
}
CUDAAllocator::CUDAAllocator(Device device, std::shared_ptr<MemoryPool> pool, CudaStream queue)
    : device_(device), pool_(std::move(pool)), queue_(std::move(queue)) {
    if (!device.is_cuda() || !pool_ || !queue_ || queue_.device() != device)
        throw std::invalid_argument("CUDAAllocator: invalid device/pool");
}
AllocationBlock CUDAAllocator::allocate_block(size_t nbytes) {
    // Native compatibility construction is lazy; context construction supplies
    // the queue directly. Pool live records retain the selected queue.
    auto queue = queue_ ? queue_ : default_runtime().cuda(device_.index)->compute_stream();
    return pool_->acquire(nbytes, device_, queue);
}
void CUDAAllocator::release_block(void *pointer) noexcept {
    pool_->release(pointer);
}

void *cuda_raw_allocate(size_t capacity, Device device) {
    if (!capacity)
        return nullptr;
#ifdef WITH_CUDA
    DeviceGuard guard(device);
    void *pointer = nullptr;
    const cudaError_t err = cudaMalloc(&pointer, capacity);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA allocation: ") + cudaGetErrorString(err));
    return pointer;
#else
    (void)device;
    throw std::runtime_error("CUDA allocation: CUDA is not available in this build");
#endif
}
void cuda_raw_free(void *pointer, Device device) noexcept {
    if (!pointer)
        return;
#ifdef WITH_CUDA
    try {
        DeviceGuard guard(device);
        const cudaError_t err = cudaFree(pointer);
        if (err != cudaSuccess)
            orthon_log_device_fallback("CUDAAllocator", cudaGetErrorString(err));
    } catch (const std::exception &error) {
        orthon_log_device_fallback("CUDAAllocator", error.what());
    }
#else
    (void)device;
#endif
}

} // namespace orthon
