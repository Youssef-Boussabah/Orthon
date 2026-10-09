#pragma once
#include <memory>
#include <mutex>
#include <orthon/device/device.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/cuda_stream.h>

namespace orthon {

class Allocator;
class MemoryPool;
class RawAllocator;
class ScratchArena;

class CudaContext {
    struct CublasState;

  public:
    // Keeps the resource owner and scoped device selection alive through the
    // whole configuration/enqueue sequence. One handle per context, serialized.
    class CublasLease {
      public:
        ~CublasLease();
        CublasLease(const CublasLease &) = delete;
        CublasLease &operator=(const CublasLease &) = delete;
        void *handle() const noexcept;

      private:
        friend class CudaContext;
        explicit CublasLease(std::shared_ptr<CublasState> state);
        std::shared_ptr<CublasState> state_;
        std::unique_ptr<DeviceGuard> guard_;
        std::unique_lock<std::mutex> lock_;
    };

    CudaContext(Device device, std::shared_ptr<MemoryPool> pool, std::shared_ptr<RawAllocator> raw,
                CudaResourcesFactory factory = {});
    ~CudaContext();
    CudaContext(const CudaContext &) = delete;
    CudaContext &operator=(const CudaContext &) = delete;
    Device device() const noexcept {
        return device_;
    }
    std::shared_ptr<Allocator> allocator() const {
        return allocator_;
    }
    std::shared_ptr<ScratchArena> scratch() const {
        return scratch_;
    }
    const CudaCapabilities &capabilities() const noexcept {
        return capabilities_;
    }
    CudaStream compute_stream() const;
    CudaStream create_stream() const;
    void synchronize_compute();
    void copy_h2d_snapshot(void *dst, const void *src, size_t bytes);
    void copy_d2h_blocking(void *dst, const void *src, size_t bytes);
    void copy_d2d_async(void *dst, const void *src, size_t bytes, const char *operation = nullptr);
    void copy_d2d_2d_async(void *dst, size_t dst_pitch, const void *src, size_t src_pitch,
                           size_t width, size_t height, const char *operation = nullptr);
    void memset_async(void *dst, int value, size_t bytes, const char *operation = nullptr);
    void peer_copy_blocking(void *dst, CudaContext &source, const void *src, size_t bytes);
    CublasLease acquire_cublas();
    void synchronize();

  private:
    Device device_;
    std::shared_ptr<Allocator> allocator_;
    std::shared_ptr<ScratchArena> scratch_;
    CudaCapabilities capabilities_;
    std::shared_ptr<CublasState> cublas_;
    CudaStream compute_;
};

} // namespace orthon
