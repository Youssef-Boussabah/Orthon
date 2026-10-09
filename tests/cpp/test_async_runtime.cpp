#include "cuda_test_resources.h"
#include <gtest/gtest.h>
#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/pool.h>
#include <orthon/memory/scratch.h>
#include <orthon/runtime/completion.h>
#include <orthon/runtime/runtime.h>
#include <set>
#include <string>
#include <vector>
using namespace orthon;
namespace {
struct Counts {
    int owners = 0, destroyed = 0, streams = 0, stream_frees = 0;
    int events = 0, event_frees = 0, records = 0, queries = 0;
    int host_waits = 0, queue_waits = 0, stream_syncs = 0, device_syncs = 0;
    int binds = 0, cublas_frees = 0, selections = 0;
    bool fail_stream = false, fail_event = false, fail_record = false;
    bool fail_wait = false, fail_sync = false, fail_bind = false, fail_host_wait = false;
    std::vector<std::string> order;
};
class Resources final : public TestCudaResources {
    struct Event {
        void *stream = nullptr;
        bool ready = false;
    };
    std::shared_ptr<Counts> c;
    std::set<Event *> events;
    bool cublas_destroyed = false;

  public:
    Resources(std::shared_ptr<Counts> counts) : c(std::move(counts)) {
        ++c->owners;
    }
    ~Resources() override {
        EXPECT_TRUE(events.empty());
        destroy_cublas();
        ++c->destroyed;
    }
    std::unique_ptr<DeviceGuard> select_device() override {
        ++c->selections;
        return {};
    }
    void *create_stream() override {
        if (c->fail_stream)
            throw std::runtime_error("stream create failure");
        ++c->streams;
        return new int(0);
    }
    void destroy_stream(void *stream) noexcept override {
        ++c->stream_frees;
        c->order.push_back("stream destroy");
        delete static_cast<int *>(stream);
    }
    void synchronize_stream(void *stream) override {
        ++c->stream_syncs;
        if (c->fail_sync)
            throw std::runtime_error("stream sync failure");
        c->order.push_back("stream complete");
        for (auto e : events)
            if (e->stream == stream)
                e->ready = true;
    }
    void synchronize() override {
        ++c->device_syncs;
        for (auto e : events)
            e->ready = true;
    }
    void *create_event(bool) override {
        if (c->fail_event)
            throw std::runtime_error("event create failure");
        auto e = std::make_unique<Event>();
        events.insert(e.get());
        ++c->events;
        return e.release();
    }
    void destroy_event(void *event) noexcept override {
        auto e = static_cast<Event *>(event);
        events.erase(e);
        delete e;
        ++c->event_frees;
    }
    void record_event(void *event, void *stream) override {
        if (c->fail_record)
            throw std::runtime_error("event record failure");
        ++c->records;
        static_cast<Event *>(event)->stream = stream;
    }
    bool query_event(void *event) override {
        ++c->queries;
        return static_cast<Event *>(event)->ready;
    }
    void wait_event(void *event) override {
        ++c->host_waits;
        if (c->fail_host_wait)
            throw std::runtime_error("event wait failure");
        static_cast<Event *>(event)->ready = true;
    }
    void stream_wait_event(void *, void *) override {
        if (c->fail_wait)
            throw std::runtime_error("stream wait failure");
        ++c->queue_waits;
    }
    void bind_compute_stream(void *stream) override {
        EXPECT_NE(stream, nullptr);
        if (c->fail_bind)
            throw std::runtime_error("bind failure");
        ++c->binds;
    }
    void destroy_cublas() noexcept override {
        if (!cublas_destroyed) {
            cublas_destroyed = true;
            ++c->cublas_frees;
            c->order.push_back("cublas destroy");
        }
    }
};
struct Raw : RawAllocator {
    int allocations = 0, frees = 0;
    void *allocate(size_t n, Device) override {
        ++allocations;
        return cpu_raw_allocate(n);
    }
    void deallocate(void *p, Device) noexcept override {
        ++frees;
        cpu_raw_free(p);
    }
};
struct Fixture {
    std::shared_ptr<Counts> c = std::make_shared<Counts>();
    std::shared_ptr<Raw> raw = std::make_shared<Raw>();
    CudaResourcesFactory factory() {
        return [c = c](Device) { return std::make_unique<Resources>(c); };
    }
};
} // namespace
TEST(AsyncRuntime, ExactlyOneComputeAndOneAuxiliaryStream) {
    Fixture f;
    {
        Runtime r(f.raw, f.factory());
        auto context = r.cuda(3);
        EXPECT_NE(context->compute_stream().native_handle(), nullptr);
        EXPECT_EQ(context->compute_stream().device(), Device(DeviceType::CUDA, 3));
        {
            auto aux = context->create_stream();
            EXPECT_FALSE(aux.same_queue(context->compute_stream()));
            EXPECT_EQ(f.c->streams, 2);
        }
        EXPECT_EQ(f.c->stream_frees, 1);
    }
    EXPECT_EQ(f.c->stream_frees, 2);
    EXPECT_EQ(f.c->cublas_frees, 1);
    EXPECT_EQ(f.c->destroyed, 1);
    auto n = f.c->order.size();
    ASSERT_GE(n, 3u);
    EXPECT_EQ(f.c->order[n - 3], "stream complete");
    EXPECT_EQ(f.c->order[n - 2], "cublas destroy");
    EXPECT_EQ(f.c->order[n - 1], "stream destroy");
}
TEST(AsyncRuntime, FailedComputeCreationNeverPublishesAndRetries) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    f.c->fail_stream = true;
    EXPECT_THROW(r.cuda(0), std::runtime_error);
    EXPECT_EQ(f.c->destroyed, 1);
    EXPECT_EQ(f.c->streams, 0);
    f.c->fail_stream = false;
    EXPECT_EQ(r.cuda(0), r.cuda(0));
    EXPECT_EQ(f.c->owners, 2);
}
TEST(AsyncRuntime, AuxiliaryCreationFailureRetainsCanonicalQueue) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    f.c->fail_stream = true;
    EXPECT_THROW(context->create_stream(), std::runtime_error);
    EXPECT_EQ(f.c->stream_frees, 0);
    f.c->fail_stream = false;
    EXPECT_NE(context->create_stream().native_handle(), nullptr);
}
TEST(AsyncRuntime, TokenCopiesRetainEventAndResourcesBeyondContext) {
    Fixture f;
    CompletionToken first, second;
    {
        Runtime r(f.raw, f.factory());
        first = r.cuda(0)->compute_stream().record_completion();
        second = first;
    }
    EXPECT_EQ(f.c->destroyed, 0);
    EXPECT_FALSE(first.ready());
    first = {};
    EXPECT_EQ(f.c->event_frees, 0);
    second.wait_host();
    EXPECT_TRUE(second.ready());
    second = {};
    EXPECT_EQ(f.c->event_frees, 1);
    EXPECT_EQ(f.c->stream_frees, 1);
    EXPECT_EQ(f.c->destroyed, 1);
}
TEST(AsyncRuntime, NoTokenMeansImmediatelyComplete) {
    CompletionToken t;
    EXPECT_TRUE(t.ready());
    EXPECT_NO_THROW(t.wait_host());
    EXPECT_NO_THROW(t.enqueue_wait_on({}));
}
TEST(AsyncRuntime, TokenSameQueueNeedsNoWaitAndCrossQueueWaits) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    auto a = context->compute_stream(), b = context->create_stream();
    auto t = a.record_completion();
    t.enqueue_wait_on(a);
    EXPECT_EQ(f.c->queue_waits, 0);
    EXPECT_EQ(f.c->host_waits, 0);
    t.enqueue_wait_on(b);
    EXPECT_EQ(f.c->queue_waits, 1);
    EXPECT_EQ(f.c->host_waits, 0);
    EXPECT_FALSE(t.ready());
}
TEST(AsyncRuntime, CrossDeviceTokenWaitIsRejected) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto t = r.cuda(0)->compute_stream().record_completion();
    EXPECT_THROW(t.enqueue_wait_on(r.cuda(1)->compute_stream()), std::invalid_argument);
}
TEST(AsyncRuntime, EventCreationFailureUnwindsAndCanRetry) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto s = r.cuda(0)->compute_stream();
    f.c->fail_event = true;
    EXPECT_THROW(s.record_completion(), std::runtime_error);
    EXPECT_EQ(f.c->events, 0);
    f.c->fail_event = false;
    EXPECT_FALSE(s.record_completion().ready());
}
TEST(AsyncRuntime, EventRecordFailureDestroysPartialEventAndRetries) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto s = r.cuda(0)->compute_stream();
    f.c->fail_record = true;
    EXPECT_THROW(s.record_completion(), std::runtime_error);
    EXPECT_EQ(f.c->events, 1);
    EXPECT_EQ(f.c->event_frees, 1);
    f.c->fail_record = false;
    EXPECT_FALSE(s.record_completion().ready());
}
TEST(AsyncRuntime, PoolSameStreamReusesPendingBlockWithoutHostWait) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto a = r.cuda(0)->allocator();
    auto x = a->allocate(7);
    auto p = x.data();
    x.reset();
    EXPECT_EQ(f.c->records, 1);
    EXPECT_EQ(r.memory_pool()->stats().cached_bytes, 64u);
    auto y = a->allocate(7);
    EXPECT_EQ(y.data(), p);
    EXPECT_EQ(f.c->host_waits, 0);
    EXPECT_EQ(f.c->stream_syncs, 0);
    EXPECT_EQ(r.memory_pool()->stats().reused_allocations, 1u);
}
TEST(AsyncRuntime, PoolCrossStreamReusesAndCountsDependency) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    auto b = std::make_shared<CUDAAllocator>(context->device(), r.memory_pool(),
                                             context->create_stream());
    auto x = context->allocator()->allocate(7);
    auto p = x.data();
    x.reset();
    auto y = b->allocate(7);
    EXPECT_EQ(y.data(), p);
    EXPECT_EQ(f.c->queue_waits, 1);
    EXPECT_EQ(f.c->host_waits, 0);
    EXPECT_EQ(f.raw->allocations, 1);
}
TEST(AsyncRuntime, FailedPoolDependencyDoesNotPublishReusableBlock) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    auto b = std::make_shared<CUDAAllocator>(context->device(), r.memory_pool(),
                                             context->create_stream());
    auto x = context->allocator()->allocate(7);
    auto p = x.data();
    x.reset();
    f.c->fail_wait = true;
    EXPECT_THROW(b->allocate(7), std::runtime_error);
    EXPECT_EQ(r.memory_pool()->stats().current_bytes, 0u);
    EXPECT_EQ(r.memory_pool()->stats().cached_bytes, 64u);
    f.c->fail_wait = false;
    EXPECT_EQ(b->allocate(7).data(), p);
}
TEST(AsyncRuntime, FailedRetirementRecordWaitsBeforeCaching) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto a = r.cuda(0)->allocator();
    auto x = a->allocate(7);
    f.c->fail_record = true;
    EXPECT_NO_THROW(x.reset());
    EXPECT_EQ(f.c->stream_syncs, 1);
    EXPECT_EQ(f.c->event_frees, 1);
    EXPECT_EQ(r.memory_pool()->stats().cached_bytes, 64u);
    f.c->fail_record = false;
}
TEST(AsyncRuntime, FailedRetirementAndStreamWaitUseDeviceFallback) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto x = r.cuda(0)->allocator()->allocate(7);
    f.c->fail_event = true;
    f.c->fail_sync = true;
    x.reset();
    EXPECT_EQ(f.c->device_syncs, 1);
    EXPECT_EQ(r.memory_pool()->stats().cached_bytes, 64u);
    f.c->fail_sync = false;
}
TEST(AsyncRuntime, ClearAndEvictionWaitBeforeRawFree) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto a = r.cuda(0)->allocator();
    auto x = a->allocate(7);
    x.reset();
    r.memory_pool()->clear_cache();
    EXPECT_EQ(f.c->host_waits, 1);
    EXPECT_EQ(f.raw->frees, 1);
    x = a->allocate(7);
    x.reset();
    r.memory_pool()->set_max_cached_bytes(0);
    EXPECT_EQ(f.c->host_waits, 2);
    EXPECT_EQ(f.raw->frees, 2);
}
TEST(AsyncRuntime, CacheDisabledWaitsBeforeRawFree) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    r.memory_pool()->set_enabled(false);
    auto x = r.cuda(0)->allocator()->allocate(7);
    x.reset();
    EXPECT_EQ(f.c->host_waits, 1);
    EXPECT_EQ(f.raw->frees, 1);
    EXPECT_EQ(r.memory_pool()->stats().reserved_bytes, 0u);
}
TEST(AsyncRuntime, FailedEventHostWaitUsesStreamFallbackBeforeFree) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto x = r.cuda(0)->allocator()->allocate(7);
    x.reset();
    f.c->fail_host_wait = true;
    r.memory_pool()->clear_cache();
    EXPECT_EQ(f.c->stream_syncs, 1);
    EXPECT_EQ(f.raw->frees, 1);
    f.c->fail_host_wait = false;
}
TEST(AsyncRuntime, AllocationRetainsQueueAndPoolAfterContextDies) {
    Fixture f;
    Allocation x;
    std::weak_ptr<CudaContext> weak;
    {
        Runtime r(f.raw, f.factory());
        weak = r.cuda(0);
        x = r.cuda(0)->allocator()->allocate(7);
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(f.c->destroyed, 0);
    x.reset();
    EXPECT_EQ(f.c->records, 1);
    EXPECT_EQ(f.c->host_waits, 1);
    EXPECT_EQ(f.c->destroyed, 1);
    EXPECT_EQ(f.raw->frees, 1);
}
TEST(AsyncRuntime, ScratchCrossStreamReusesPendingRegionAndPreservesStats) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    auto arena = context->scratch();
    auto b = context->create_stream();
    void *p;
    {
        auto x = arena->acquire(7);
        p = x.allocate(7);
    }
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    EXPECT_EQ(arena->stats().reserved_bytes, 1u << 20);
    {
        auto y = arena->acquire(7, b);
        EXPECT_EQ(y.allocate(7), p);
        EXPECT_EQ(f.c->queue_waits, 1);
        EXPECT_EQ(f.c->host_waits, 0);
    }
    EXPECT_EQ(arena->stats().peak_used_bytes, 64u);
    EXPECT_EQ(f.raw->allocations, 1);
}
TEST(AsyncRuntime, FailedScratchWaitRetainsRegionForRetry) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    auto arena = context->scratch();
    auto b = context->create_stream();
    {
        auto x = arena->acquire(7);
        x.allocate(7);
    }
    f.c->fail_wait = true;
    EXPECT_THROW(arena->acquire(7, b), std::runtime_error);
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    EXPECT_EQ(f.raw->allocations, 1);
    f.c->fail_wait = false;
    EXPECT_NE(arena->acquire(7, b).allocate(7), nullptr);
}
TEST(AsyncRuntime, ScratchRetainsResourcesAfterContextAndExceptionScope) {
    Fixture f;
    ScratchLease x;
    {
        Runtime r(f.raw, f.factory());
        x = r.cuda(0)->scratch()->acquire(7);
        x.allocate(7);
    }
    EXPECT_EQ(f.c->destroyed, 0);
    x = {};
    EXPECT_EQ(f.raw->frees, 1);
    EXPECT_EQ(f.c->host_waits, 1);
    EXPECT_EQ(f.c->destroyed, 1);
    try {
        Runtime r(f.raw, f.factory());
        auto y = r.cuda(0)->scratch()->acquire(7);
        y.allocate(7);
        throw std::runtime_error("scope");
    } catch (const std::runtime_error &) {
    }
    EXPECT_EQ(f.c->destroyed, 2);
    EXPECT_EQ(f.c->events, f.c->event_frees);
    EXPECT_EQ(f.c->streams, f.c->stream_frees);
}
TEST(AsyncRuntime, ScratchRecordingFailureUsesSafeFallback) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto arena = r.cuda(0)->scratch();
    f.c->fail_record = true;
    {
        auto x = arena->acquire(7);
        x.allocate(7);
    }
    EXPECT_EQ(f.c->stream_syncs, 1);
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    f.c->fail_record = false;
    EXPECT_NE(arena->acquire(7).allocate(7), nullptr);
}
TEST(AsyncRuntime, CheckedSynchronizationAndCublasBindFailuresRetry) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto context = r.cuda(0);
    f.c->fail_sync = true;
    EXPECT_THROW(context->synchronize_compute(), std::runtime_error);
    f.c->fail_sync = false;
    EXPECT_NO_THROW(context->synchronize_compute());
    f.c->fail_bind = true;
    EXPECT_THROW(context->acquire_cublas(), std::runtime_error);
    f.c->fail_bind = false;
    EXPECT_NE(context->acquire_cublas().handle(), nullptr);
    EXPECT_EQ(f.c->binds, 2);
    EXPECT_GT(f.c->selections, 0);
}
TEST(AsyncRuntime, TimingEventsUseComputeQueueAndExceptionUnwinds) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto s = r.cuda(0)->compute_stream();
    {
        CudaTimingEvent start(s), stop(s);
        EXPECT_THROW(stop.wait_host(), std::logic_error);
        start.record();
        stop.record();
        stop.wait_host();
        EXPECT_EQ(stop.elapsed_since(start), 0);
    }
    EXPECT_EQ(f.c->events, 2);
    EXPECT_EQ(f.c->event_frees, 2);
    f.c->fail_record = true;
    EXPECT_THROW(
        {
            CudaTimingEvent event(s);
            event.record();
        },
        std::runtime_error);
    EXPECT_EQ(f.c->events, f.c->event_frees);
}
TEST(AsyncRuntime, CudaPoolAndScratchRejectMissingOrWrongDeviceQueue) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    auto s = r.cuda(0)->compute_stream();
    EXPECT_THROW(r.memory_pool()->acquire(7, Device(DeviceType::CUDA, 0)), std::invalid_argument);
    EXPECT_THROW(r.memory_pool()->acquire(7, Device(DeviceType::CUDA, 1), s),
                 std::invalid_argument);
    EXPECT_THROW(std::make_shared<ScratchArena>(Device(DeviceType::CUDA, 0), f.raw),
                 std::invalid_argument);
    EXPECT_THROW(r.cuda(0)->scratch()->acquire(7, r.cuda(1)->compute_stream()),
                 std::invalid_argument);
    EXPECT_EQ(f.raw->allocations, 0);
}

TEST(AsyncRuntime, InitialCublasBindFailureNeverPublishesAndReleasesQueue) {
    Fixture f;
    Runtime r(f.raw, f.factory());
    f.c->fail_bind = true;
    EXPECT_THROW(r.cuda(0), std::runtime_error);
    EXPECT_EQ(f.c->destroyed, 1);
    EXPECT_EQ(f.c->streams, 1);
    EXPECT_EQ(f.c->stream_frees, 1);
    EXPECT_EQ(f.c->cublas_frees, 1);
    f.c->fail_bind = false;
    EXPECT_EQ(r.cuda(0), r.cuda(0));
}
