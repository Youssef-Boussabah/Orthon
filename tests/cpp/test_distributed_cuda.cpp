#include <gtest/gtest.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/distributed/nccl_process_group.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/scalar_ops.h>
#include <orthon/runtime/runtime.h>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <condition_variable>
#include <future>
#endif
using namespace orthon;
using namespace orthon::distributed;
#ifdef WITH_CUDA
namespace {
void cuda_checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Counts { int creates = 0, destroys = 0, enqueues = 0; void* stream = nullptr; };
class ZeroComm : public NcclCommunicator {
    std::shared_ptr<Counts> counts;
    bool closed = false;
public:
    explicit ZeroComm(std::shared_ptr<Counts> c) : counts(std::move(c)) { ++counts->creates; }
    ~ZeroComm() override { shutdown(); }
    void enqueue(CollectiveKind, void* data, size_t count, DType dtype, int, void* stream) override {
        if (counts->stream) EXPECT_EQ(stream, counts->stream);
        counts->stream = stream;
        ++counts->enqueues;
        cuda_checked(cudaMemsetAsync(data, 0, count * dtype_size(dtype), static_cast<cudaStream_t>(stream)));
    }
    void check() override {}
    void shutdown() override { if (!closed) { closed = true; ++counts->destroys; } }
    void abort_failed() override { shutdown(); }
};
class ZeroTransport : public NcclTransport {
public:
    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    std::string unique_id() override { return "one GPU test transport; not NCCL"; }
    std::unique_ptr<NcclCommunicator> create(const std::string&, int, int, Device) override {
        return std::make_unique<ZeroComm>(counts);
    }
    void initialize_barrier(CudaContext& context, void* data) override {
        context.memset_async(data, 0, sizeof(float));
    }
};
bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
struct Gate {
    explicit Gate(CudaStream stream) : queue(std::move(stream)) {}
    CudaStream queue;
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    ~Gate() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            release = true;
            cv.notify_all();
        }
        // Keep callback state alive through exceptional test unwinding too.
        queue.synchronize_noexcept();
    }
};
void CUDART_CB wait_gate(void* pointer) {
    auto& gate = *static_cast<Gate*>(pointer);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.cv.wait(lock, [&] { return gate.release; });
}
void release(Gate& gate) {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.release = true;
    gate.cv.notify_all();
}
}
#endif
TEST(DistributedCuda, RealComputeCommunicationOrderingAndOneStream) {
#ifdef WITH_CUDA
    if (!available()) GTEST_SKIP() << "usable CUDA device required";
    auto context = default_runtime().cuda(0);
    auto transport = std::make_shared<ZeroTransport>();
    NcclProcessGroup group(*context, transport, "injected", 0, 1);
    auto tensor = full({4096}, 3).to(context->device());
    auto five = full({4096}, 5).to(context->device());
    Gate gate(context->compute_stream());
    cuda_checked(cudaLaunchHostFunc(static_cast<cudaStream_t>(context->compute_stream().native_handle()), wait_gate, &gate));
    // This copy remains behind the gate. The communication queue must wait
    // for it; the later compute consumer must in turn wait for the zero write.
    context->copy_d2d_async(tensor.data(), five.data(), 4096 * sizeof(float));
    auto version = tensor.impl()->buffer()->version();
    auto work = group.all_reduce(tensor);
    EXPECT_EQ(tensor.impl()->buffer()->version(), version + 1);
    release(gate);
    auto consumer = scalar_add(tensor, 7);
    auto cpu = consumer.to(Device());
    EXPECT_FLOAT_EQ(static_cast<float*>(cpu.data())[0], 7);
    EXPECT_TRUE(work->wait()); EXPECT_TRUE(work->wait());
    EXPECT_EQ(tensor.impl()->buffer()->version(), version + 1);
    for (auto dtype : {DType::Float16, DType::BFloat16}) {
        auto value = full({16}, 2, dtype).to(context->device());
        auto before = value.impl()->buffer()->version();
        EXPECT_TRUE(group.broadcast(value, 0)->wait());
        EXPECT_EQ(value.impl()->buffer()->version(), before + 1);
    }
    group.barrier()->wait();
    group.shutdown(); group.shutdown();
    EXPECT_TRUE(work->is_completed());
    EXPECT_EQ(transport->counts->creates, 1);
    EXPECT_EQ(transport->counts->destroys, 1);
#else
    GTEST_SKIP() << "CUDA build required; no hardware evidence from host tests";
#endif
}
TEST(DistributedCuda, DroppedWorkAndTensorRetireWithoutBlockingOrReuseRace) {
#ifdef WITH_CUDA
    if (!available()) GTEST_SKIP() << "usable CUDA device required";
    auto context = default_runtime().cuda(0);
    auto transport = std::make_shared<ZeroTransport>();
    NcclProcessGroup group(*context, transport, "injected", 0, 1);
    auto tensor = full({8192}, 9).to(context->device());
    Gate gate(context->compute_stream());
    cuda_checked(cudaLaunchHostFunc(static_cast<cudaStream_t>(context->compute_stream().native_handle()), wait_gate, &gate));
    auto work = group.all_reduce(tensor);
    auto drop = std::async(std::launch::async, [work = std::move(work), tensor = std::move(tensor)]() mutable {
        work.reset(); tensor = Tensor(nullptr);
    });
    auto status = drop.wait_for(std::chrono::seconds(2));
    release(gate); // Always release, including failed assertions, to reap safely.
    EXPECT_EQ(status, std::future_status::ready);
    drop.get();
    auto reused = full({8192}, 11).to(context->device());
    group.shutdown();
    auto cpu = reused.to(Device());
    EXPECT_FLOAT_EQ(static_cast<float*>(cpu.data())[0], 11);
#else
    GTEST_SKIP() << "CUDA build required";
#endif
}
