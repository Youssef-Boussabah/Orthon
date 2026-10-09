#pragma once
#include <memory>
#include <string>
#include <orthon/core/dtype.h>
#include <orthon/device/device.h>
namespace orthon {
class CudaContext;
namespace distributed {
enum class CollectiveKind { Sum, Broadcast };
class NcclCommunicator {
public:
    virtual ~NcclCommunicator() = default;
    virtual void enqueue(CollectiveKind kind, void* data, size_t count, DType dtype,
                         int src, void* stream) = 0;
    virtual void check() = 0;
    virtual void shutdown() = 0;
    virtual void abort_failed() = 0;
};
class NcclTransport {
public:
    virtual ~NcclTransport() = default;
    virtual std::string unique_id() = 0;
    virtual std::unique_ptr<NcclCommunicator> create(const std::string& id, int rank,
                                                  int world_size, Device device) = 0;
    // The real adapter uses the context's ordinary transfer service; host
    // injection initializes the allocation without requiring a CUDA driver.
    virtual void initialize_barrier(CudaContext& context, void* data) = 0;
};
std::shared_ptr<NcclTransport> system_nccl_transport();
} // namespace distributed
} // namespace orthon
