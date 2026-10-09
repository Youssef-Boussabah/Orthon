#pragma once
#include <mutex>
#include <vector>
#include <orthon/core/buffer.h>
#include <orthon/distributed/nccl_transport.h>
#include <orthon/distributed/process_group.h>
#include <orthon/memory/allocation.h>
#include <orthon/runtime/cuda_context.h>
namespace orthon {
namespace distributed {
class NcclStatus;
// Native enqueue descriptor. Tensor adapters derive all fields from one
// TensorImpl; injected transports can exercise queue ownership without a GPU.
struct CollectiveBuffer {
    void* data;
    size_t count;
    DType dtype;
    Device device;
    bool contiguous;
    std::shared_ptr<Buffer> owner;
};
class NcclProcessGroup final : public ProcessGroup {
public:
    NcclProcessGroup(CudaContext& context, std::shared_ptr<NcclTransport> transport,
                     const std::string& id, int rank, int world_size);
    ~NcclProcessGroup() noexcept override;
    Device device() const { return device_; }
    std::shared_ptr<Work> all_reduce(Tensor& tensor) override;
    std::shared_ptr<Work> broadcast(Tensor& tensor, int src) override;
    std::shared_ptr<Work> barrier() override;
    std::shared_ptr<Work> enqueue(CollectiveBuffer buffer, CollectiveKind kind, int src = 0);
    void shutdown() override;
private:
    Device device_;
    std::mutex mutex_;
    CudaStream compute_, communication_;
    Allocation barrier_storage_;
    std::shared_ptr<NcclCommunicator> communicator_;
    std::shared_ptr<NcclStatus> status_;
    std::vector<std::weak_ptr<Work>> work_;
    std::shared_ptr<Work> enqueue_locked(CollectiveBuffer buffer, CollectiveKind kind, int src);
    void validate(const CollectiveBuffer& buffer, CollectiveKind kind, int src) const;
};
} // namespace distributed
} // namespace orthon
