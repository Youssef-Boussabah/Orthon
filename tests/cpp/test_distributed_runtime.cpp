#include "cuda_test_resources.h"
#include <gtest/gtest.h>
#include <orthon/core/buffer.h>
#include <orthon/distributed/nccl_process_group.h>
#include <orthon/memory/cpu_allocator.h>
#include <orthon/runtime/runtime.h>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <vector>
using namespace orthon;
using namespace orthon::distributed;
namespace {
struct Trace {
    int streams = 0, frees = 0, host_waits = 0, syncs = 0, creates = 0, destroys = 0;
    int checks = 0, fail_check = 0, queries = 0, ready_after = 2, aborts = 0, events = 0;
    bool fail_enqueue = false, fail_wait = false, ready = false;
    std::vector<std::string> order;
};
class Resources : public TestCudaResources {
    std::shared_ptr<Trace> t;
public:
    explicit Resources(std::shared_ptr<Trace> trace) : t(std::move(trace)) {}
    void* create_stream() override { ++t->streams; return TestCudaResources::create_stream(); }
    void destroy_stream(void* s) noexcept override { ++t->frees; TestCudaResources::destroy_stream(s); }
    void record_event(void*, void*) override { t->order.push_back("record"); }
    void stream_wait_event(void*, void*) override { t->order.push_back("queue wait"); }
    void* create_event(bool timing) override { ++t->events; return TestCudaResources::create_event(timing); }
    void destroy_event(void* event) noexcept override { --t->events; TestCudaResources::destroy_event(event); }
    bool query_event(void*) override {
        ++t->queries;
        if (t->fail_wait) throw std::runtime_error("checked completion failed");
        if (t->ready_after && t->queries >= t->ready_after) t->ready = true;
        return t->ready;
    }
    void wait_event(void*) override {
        ++t->host_waits;
        if (t->fail_wait) throw std::runtime_error("checked completion failed");
        t->ready = true;
    }
    void synchronize_stream(void*) override { ++t->syncs; t->order.push_back("drain"); t->ready = true; }
};
class Raw : public RawAllocator {
public:
    void* allocate(size_t bytes, Device) override { return cpu_raw_allocate(bytes); }
    void deallocate(void* p, Device) noexcept override { cpu_raw_free(p); }
};
class Comm : public NcclCommunicator {
    std::shared_ptr<Trace> t;
    bool closed = false;
public:
    explicit Comm(std::shared_ptr<Trace> trace) : t(std::move(trace)) { ++t->creates; }
    ~Comm() override { shutdown(); }
    void enqueue(CollectiveKind, void*, size_t, DType, int, void*) override {
        if (t->fail_enqueue) throw std::runtime_error("enqueue failed");
        t->order.push_back("collective");
    }
    void check() override {
        ++t->checks;
        if (t->fail_check && t->checks >= t->fail_check)
            throw std::runtime_error("late asynchronous communicator failure");
    }
    void shutdown() override {
        if (!closed) { ++t->destroys; t->order.push_back("communicator destroy"); closed = true; }
    }
    void abort_failed() override {
        if (!closed) { ++t->aborts; t->order.push_back("communicator abort"); closed = true; }
    }
};
class Transport : public NcclTransport {
    std::shared_ptr<Trace> t;
public:
    explicit Transport(std::shared_ptr<Trace> trace) : t(std::move(trace)) {}
    std::string unique_id() override { return "host ownership test"; }
    std::unique_ptr<NcclCommunicator> create(const std::string&, int, int, Device) override {
        return std::make_unique<Comm>(t);
    }
    void initialize_barrier(CudaContext&, void* p) override { std::memset(p, 0, sizeof(float)); }
};
struct Fixture {
    std::shared_ptr<Trace> trace = std::make_shared<Trace>();
    Runtime runtime{std::make_shared<Raw>(), [this](Device) { return std::make_unique<Resources>(trace); }};
    std::shared_ptr<CudaContext> context = runtime.cuda(0);
    std::shared_ptr<Transport> transport = std::make_shared<Transport>(trace);
    NcclProcessGroup group{*context, transport, transport->unique_id(), 0, 1};
    std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(16, Device());
    CollectiveBuffer view() { return {buffer->data(), 4, DType::Float32, Device(DeviceType::CUDA, 0), true, buffer}; }
};
}
TEST(DistributedRuntime, OneCommunicatorAndStreamOrderEveryEnqueue) {
    Fixture f;
    int streams = f.trace->streams;
    f.trace->order.clear();
    auto a = f.group.enqueue(f.view(), CollectiveKind::Sum);
    auto b = f.group.enqueue(f.view(), CollectiveKind::Broadcast);
    EXPECT_EQ(f.trace->streams, streams);
    EXPECT_EQ(f.trace->creates, 1);
    EXPECT_EQ(f.trace->host_waits, 0);
    EXPECT_EQ(f.trace->order, (std::vector<std::string>{"record", "queue wait", "collective", "record", "queue wait",
                                                      "record", "queue wait", "collective", "record", "queue wait"}));
    EXPECT_EQ(f.buffer->version(), 2);
    EXPECT_FALSE(a->is_completed());
    EXPECT_TRUE(a->wait()); EXPECT_TRUE(a->wait()); EXPECT_TRUE(a->is_completed());
    EXPECT_EQ(f.trace->host_waits, 0);
    f.group.shutdown(); f.group.shutdown();
    EXPECT_TRUE(b->wait()); EXPECT_TRUE(b->is_completed());
    EXPECT_EQ(f.trace->destroys, 1);
    EXPECT_EQ(f.buffer->version(), 2);
    EXPECT_THROW(f.group.enqueue(f.view(), CollectiveKind::Sum), std::runtime_error);
}
TEST(DistributedRuntime, DroppedWorkDoesNotHostWait) {
    Fixture f;
    int syncs = f.trace->syncs;
    f.group.enqueue(f.view(), CollectiveKind::Sum).reset();
    EXPECT_EQ(f.trace->host_waits, 0);
    EXPECT_EQ(f.trace->syncs, syncs);
    f.group.barrier().reset();
    EXPECT_EQ(f.trace->syncs, syncs);
}
TEST(DistributedRuntime, PreflightDoesNotMutateOrPoison) {
    Fixture f;
    auto bad = f.view(); bad.device = Device();
    EXPECT_THROW(f.group.enqueue(bad, CollectiveKind::Sum), std::runtime_error);
    bad = f.view(); bad.device.index = 1;
    EXPECT_THROW(f.group.enqueue(bad, CollectiveKind::Sum), std::runtime_error);
    bad = f.view(); bad.contiguous = false;
    EXPECT_THROW(f.group.enqueue(bad, CollectiveKind::Sum), std::runtime_error);
    for (auto dtype : {DType::Float64, DType::Int32, DType::Int64, DType::Bool}) {
        bad = f.view(); bad.dtype = dtype;
        EXPECT_THROW(f.group.enqueue(bad, CollectiveKind::Sum), std::runtime_error);
    }
    EXPECT_THROW(f.group.enqueue(f.view(), CollectiveKind::Broadcast, 1), std::invalid_argument);
    EXPECT_EQ(f.buffer->version(), 0);
    EXPECT_TRUE(f.group.enqueue(f.view(), CollectiveKind::Sum)->wait());
    EXPECT_EQ(f.buffer->version(), 1);
}
TEST(DistributedRuntime, EnqueueFailureRejectsLaterCallsWithoutVersionBump) {
    Fixture f;
    f.trace->fail_enqueue = true;
    EXPECT_THROW(f.group.enqueue(f.view(), CollectiveKind::Sum), std::runtime_error);
    EXPECT_EQ(f.buffer->version(), 0);
    f.trace->fail_enqueue = false;
    EXPECT_THROW(f.group.enqueue(f.view(), CollectiveKind::Sum), std::runtime_error);
    EXPECT_THROW(f.group.shutdown(), std::runtime_error);
    EXPECT_EQ(f.trace->aborts, 1);
}
TEST(DistributedRuntime, CheckedCompletionFailureMarksGroupFailed) {
    Fixture f;
    auto work = f.group.enqueue(f.view(), CollectiveKind::Sum);
    f.trace->fail_wait = true;
    EXPECT_THROW(work->wait(), std::runtime_error);
    EXPECT_THROW(f.group.barrier(), std::runtime_error);
    f.trace->fail_wait = false;
    f.group.shutdown();
    EXPECT_TRUE(work->wait());
}
TEST(DistributedRuntime, ContextCanDieBeforeGroupAndWork) {
    auto trace = std::make_shared<Trace>();
    auto transport = std::make_shared<Transport>(trace);
    std::unique_ptr<NcclProcessGroup> group;
    std::shared_ptr<Work> work;
    auto buffer = std::make_shared<Buffer>(4, Device());
    {
        Runtime runtime(std::make_shared<Raw>(), [trace](Device) { return std::make_unique<Resources>(trace); });
        auto context = runtime.cuda(2);
        group = std::make_unique<NcclProcessGroup>(*context, transport, "host", 0, 1);
        work = group->enqueue({buffer->data(), 1, DType::Float32, context->device(), true, buffer}, CollectiveKind::Sum);
    }
    group->shutdown(); group.reset();
    EXPECT_TRUE(work->wait());
    EXPECT_EQ(trace->destroys, 1);
}
TEST(DistributedRuntime, LateAsyncFailureDuringPollingAbortsAndStaysFailed) {
    auto trace = std::make_shared<Trace>();
    {
        Fixture f;
        trace = f.trace;
        auto work = f.group.enqueue(f.view(), CollectiveKind::Sum);
        auto peer = f.group.enqueue(f.view(), CollectiveKind::Sum);
        const int streams = trace->streams, checks = trace->checks;
        trace->ready_after = 0;
        trace->fail_check = checks + 2;
        auto start = std::chrono::steady_clock::now();
        EXPECT_THROW(work->wait(), std::runtime_error);
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
        EXPECT_EQ(trace->checks, checks + 2);
        EXPECT_EQ(trace->queries, 1);
        EXPECT_EQ(trace->aborts, 1);
        EXPECT_EQ(trace->host_waits, 0);
        EXPECT_EQ(trace->streams, streams);
        EXPECT_THROW(f.group.barrier(), std::runtime_error);
        EXPECT_THROW(f.group.shutdown(), std::runtime_error);
        f.group.shutdown();
        EXPECT_THROW(work->wait(), std::runtime_error);
        EXPECT_THROW(work->is_completed(), std::runtime_error);
        EXPECT_THROW(peer->wait(), std::runtime_error);
        EXPECT_THROW(peer->is_completed(), std::runtime_error);
        EXPECT_EQ(trace->aborts, 1);
        EXPECT_EQ(trace->destroys, 0);
        auto abort = std::find(trace->order.begin(), trace->order.end(), "communicator abort");
        auto drain = std::find(trace->order.begin(), trace->order.end(), "drain");
        EXPECT_LT(abort - trace->order.begin(), drain - trace->order.begin());
    }
    EXPECT_EQ(trace->streams, trace->frees);
    EXPECT_EQ(trace->events, 0);
}
TEST(DistributedRuntime, ReadyEventFinalStatusFailureNeverPublishesSuccess) {
    for (bool query : {false, true}) {
        Fixture f;
        auto work = f.group.enqueue(f.view(), CollectiveKind::Sum);
        f.trace->ready = true;
        f.trace->fail_check = f.trace->checks + 2;
        if (query) EXPECT_THROW(work->is_completed(), std::runtime_error);
        else EXPECT_THROW(work->wait(), std::runtime_error);
        EXPECT_THROW(f.group.barrier(), std::runtime_error);
        EXPECT_EQ(f.trace->queries, 1);
        EXPECT_EQ(f.trace->aborts, 1);
        EXPECT_THROW(f.group.shutdown(), std::runtime_error);
        EXPECT_THROW(work->wait(), std::runtime_error);
        EXPECT_THROW(work->is_completed(), std::runtime_error);
        EXPECT_EQ(f.trace->aborts, 1);
    }
}
TEST(DistributedRuntime, SuccessfulPollingChecksEveryQueryAndFinalReadiness) {
    Fixture f;
    auto work = f.group.enqueue(f.view(), CollectiveKind::Sum);
    const int checks = f.trace->checks, streams = f.trace->streams;
    f.trace->ready_after = 4;
    EXPECT_TRUE(work->wait());
    EXPECT_EQ(f.trace->queries, 4);
    EXPECT_EQ(f.trace->checks, checks + 5);
    EXPECT_EQ(f.trace->host_waits, 0);
    EXPECT_EQ(f.trace->streams, streams);
    EXPECT_TRUE(work->is_completed());
    f.group.shutdown();
    EXPECT_TRUE(work->wait());
    EXPECT_EQ(f.trace->destroys, 1);
    EXPECT_EQ(f.trace->aborts, 0);
}
TEST(DistributedRuntime, ShutdownChecksLateFailureBeforeRawStreamDrain) {
    Fixture f;
    auto work = f.group.enqueue(f.view(), CollectiveKind::Sum);
    f.trace->ready_after = 0;
    f.trace->fail_check = f.trace->checks + 3;
    EXPECT_THROW(f.group.shutdown(), std::runtime_error);
    EXPECT_EQ(f.trace->queries, 1);
    EXPECT_EQ(f.trace->aborts, 1);
    EXPECT_EQ(f.trace->destroys, 0);
    EXPECT_EQ(f.trace->host_waits, 0);
    EXPECT_THROW(work->wait(), std::runtime_error);
    auto abort = std::find(f.trace->order.begin(), f.trace->order.end(), "communicator abort");
    auto drain = std::find(f.trace->order.begin(), f.trace->order.end(), "drain");
    EXPECT_LT(abort - f.trace->order.begin(), drain - f.trace->order.begin());
}
