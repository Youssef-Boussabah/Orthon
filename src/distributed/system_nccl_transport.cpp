#include <orthon/distributed/nccl_transport.h>
#include <orthon/device/device_guard.h>
#include <orthon/runtime/cuda_context.h>
#include <cstring>
#include <stdexcept>
#ifdef WITH_NCCL
#include <nccl.h>
#endif
namespace orthon {
namespace distributed {
#ifdef WITH_NCCL
namespace {
void checked(ncclResult_t result) {
    if (result != ncclSuccess)
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result));
}
ncclDataType_t native_dtype(DType dtype) {
    if (dtype == DType::Float32) return ncclFloat32;
    if (dtype == DType::Float16) return ncclFloat16;
    if (dtype == DType::BFloat16) return ncclBfloat16;
    throw std::runtime_error("unsupported NCCL dtype");
}
class SystemCommunicator final : public NcclCommunicator {
    ncclComm_t comm_ = nullptr;
    Device device_;
public:
    SystemCommunicator(const std::string& bytes, int rank, int world_size, Device device) : device_(device) {
        if (bytes.size() != sizeof(ncclUniqueId)) throw std::invalid_argument("invalid NCCL unique id size");
        ncclUniqueId id;
        std::memcpy(&id, bytes.data(), sizeof(id));
        DeviceGuard guard(device_);
        try { checked(ncclCommInitRank(&comm_, world_size, id, rank)); }
        catch (...) {
            if (comm_) { ncclCommAbort(comm_); comm_ = nullptr; }
            throw;
        }
    }
    ~SystemCommunicator() override { try { shutdown(); } catch (...) {} }
    void enqueue(CollectiveKind kind, void* data, size_t count, DType dtype, int src, void* stream) override {
        DeviceGuard guard(device_);
        if (kind == CollectiveKind::Sum)
            checked(ncclAllReduce(data, data, count, native_dtype(dtype), ncclSum, comm_,
                                  static_cast<cudaStream_t>(stream)));
        else checked(ncclBroadcast(data, data, count, native_dtype(dtype), src, comm_,
                                   static_cast<cudaStream_t>(stream)));
    }
    void check() override {
        if (!comm_) return;
        DeviceGuard guard(device_);
        ncclResult_t error;
        checked(ncclCommGetAsyncError(comm_, &error));
        checked(error);
    }
    void shutdown() override {
        if (!comm_) return;
        DeviceGuard guard(device_);
        auto comm = comm_;
        comm_ = nullptr;
        checked(ncclCommDestroy(comm));
    }
    void abort_failed() override {
        if (!comm_) return;
        DeviceGuard guard(device_);
        auto comm = comm_;
        comm_ = nullptr;
        checked(ncclCommAbort(comm));
    }
};
class SystemTransport final : public NcclTransport {
public:
    std::string unique_id() override {
        ncclUniqueId id;
        checked(ncclGetUniqueId(&id));
        return {reinterpret_cast<const char*>(&id), sizeof(id)};
    }
    std::unique_ptr<NcclCommunicator> create(const std::string& id, int rank, int size, Device device) override {
        return std::make_unique<SystemCommunicator>(id, rank, size, device);
    }
    void initialize_barrier(CudaContext& context, void* data) override {
        context.memset_async(data, 0, sizeof(float), "NCCL barrier initialization");
    }
};
}
#endif
std::shared_ptr<NcclTransport> system_nccl_transport() {
#ifdef WITH_NCCL
    return std::make_shared<SystemTransport>();
#else
    throw std::runtime_error("NCCL is not available in this build (recompile with -DWITH_CUDA=ON -DWITH_NCCL=ON)");
#endif
}
} // namespace distributed
} // namespace orthon
