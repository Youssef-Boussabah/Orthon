#include <orthon/distributed/nccl_process_group.h>
#include <orthon/core/buffer.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/memory/allocator.h>
#include <stdexcept>
#include <utility>
namespace orthon {
namespace distributed {
// Shared with surviving Work, without retaining a ProcessGroup or context.
// Serializing status and retirement makes abort exactly once even when a Work
// query races shutdown. Transport failure remains sticky after retirement.
class NcclStatus {
    std::mutex mutex_;
    std::shared_ptr<NcclCommunicator> comm_;
    std::shared_ptr<GroupLifetime> lifetime_;
    std::exception_ptr failure_;
    void retire_locked(std::exception_ptr failure) {
        if (failure_) return;
        failure_ = std::move(failure);
        lifetime_->failed = true;
        try { comm_->abort_failed(); } catch (...) {} // preserve original failure
    }
public:
    NcclStatus(std::shared_ptr<NcclCommunicator> comm, std::shared_ptr<GroupLifetime> lifetime)
        : comm_(std::move(comm)), lifetime_(std::move(lifetime)) {}
    void check() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        try { comm_->check(); }
        catch (...) { retire_locked(std::current_exception()); throw; }
    }
    void enqueue(CollectiveKind kind, void* data, size_t count, DType dtype, int src, void* stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        try { comm_->check(); comm_->enqueue(kind, data, count, dtype, src, stream); }
        catch (...) { retire_locked(std::current_exception()); throw; }
    }
    void retire(std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        retire_locked(std::move(failure));
    }
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        comm_->shutdown();
    }
};
namespace {
CollectiveBuffer view(Tensor& tensor) {
    return {tensor.data(), static_cast<size_t>(tensor.numel()), tensor.dtype(), tensor.device(),
            tensor.is_contiguous(), tensor.impl()->buffer()};
}
}
NcclProcessGroup::NcclProcessGroup(CudaContext& context, std::shared_ptr<NcclTransport> transport,
                                 const std::string& id, int rank, int world_size)
    : ProcessGroup(rank, world_size), device_(context.device()), compute_(context.compute_stream()),
      communication_(context.create_stream()), barrier_storage_(context.allocator()->allocate(sizeof(float))) {
    if (!transport) throw std::invalid_argument("NCCL transport is required");
    transport->initialize_barrier(context, barrier_storage_.data());
    communicator_ = transport->create(id, rank, world_size, device_);
    status_ = std::make_shared<NcclStatus>(communicator_, lifetime_);
}
NcclProcessGroup::~NcclProcessGroup() noexcept {
    try { shutdown(); }
    catch (...) {
        if (status_) status_->retire(std::current_exception());
        if (communication_) communication_.synchronize_noexcept();
    }
}
void NcclProcessGroup::validate(const CollectiveBuffer& b, CollectiveKind kind, int src) const {
    ensure_open();
    if (!b.device.is_cuda() || b.device != device_)
        throw std::runtime_error("nccl collective: tensor CUDA device does not match communicator device");
    if (!b.contiguous) throw std::runtime_error("nccl collective: tensor must be contiguous");
    if (b.dtype != DType::Float32 && b.dtype != DType::Float16 && b.dtype != DType::BFloat16)
        throw std::runtime_error("nccl collective: only float32, float16, and bfloat16 tensors are supported");
    if (kind == CollectiveKind::Broadcast && (src < 0 || src >= world_size()))
        throw std::invalid_argument("broadcast src must be in [0, world_size)");
}
std::shared_ptr<Work> NcclProcessGroup::all_reduce(Tensor& tensor) {
    return enqueue(view(tensor), CollectiveKind::Sum);
}
std::shared_ptr<Work> NcclProcessGroup::broadcast(Tensor& tensor, int src) {
    return enqueue(view(tensor), CollectiveKind::Broadcast, src);
}
std::shared_ptr<Work> NcclProcessGroup::enqueue(CollectiveBuffer b, CollectiveKind kind, int src) {
    std::lock_guard<std::mutex> lock(mutex_);
    validate(b, kind, src);
    return enqueue_locked(std::move(b), kind, src);
}
std::shared_ptr<Work> NcclProcessGroup::enqueue_locked(CollectiveBuffer b, CollectiveKind kind, int src) {
    try {
        compute_.record_completion().enqueue_wait_on(communication_);
        status_->enqueue(kind, b.data, b.count, b.dtype, src, communication_.native_handle());
        if (b.owner) b.owner->bump_version();
        auto completion = communication_.record_completion();
        completion.enqueue_wait_on(compute_);
        auto work = std::make_shared<Work>(std::move(completion), lifetime_, std::move(b.owner),
                                          [status = status_]() { status->check(); });
        // Completed/dropped work must not grow a long-running group's bookkeeping.
        for (auto it = work_.begin(); it != work_.end();) {
            if (it->expired()) it = work_.erase(it); else ++it;
        }
        work_.push_back(work);
        return work;
    } catch (...) {
        lifetime_->failed = true;
        status_->retire(std::current_exception());
        // Failed event setup after enqueue must not leave storage retirement
        // unordered. This is exceptional cleanup, never the normal enqueue path.
        communication_.synchronize_noexcept();
        throw;
    }
}
std::shared_ptr<Work> NcclProcessGroup::barrier() {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_open();
    return enqueue_locked({barrier_storage_.data(), 1, DType::Float32, device_, true, {}},
                          CollectiveKind::Sum, 0);
}
void NcclProcessGroup::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!communication_) return;
    lifetime_->closed = true;
    std::exception_ptr error;
    try {
        status_->check();
        Work drain(communication_.record_completion(), lifetime_, {},
                   [status = status_]() { status->check(); });
        drain.wait();
    } catch (...) {
        error = std::current_exception();
        lifetime_->failed = true;
        status_->retire(error);
    }
    if (!error) {
        try { status_->shutdown(); }
        catch (...) { error = std::current_exception(); status_->retire(error); }
    }
    communicator_.reset();
    for (auto& weak : work_) if (auto work = weak.lock()) work->complete_after_drain(error);
    work_.clear();
    status_.reset();
    barrier_storage_.reset();
    communication_ = {};
    compute_ = {};
    if (error) std::rethrow_exception(error);
}
} // namespace distributed
} // namespace orthon
