#include <gtest/gtest.h>
#include "cuda_test_resources.h"
#include <orthon/runtime/runtime.h>
#include <orthon/memory/allocator.h>
#include <orthon/memory/cpu_allocator.h>
#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/pool.h>
#include <orthon/memory/scratch.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

using namespace orthon;

namespace {
// These resources model ownership/device keys on the host, never GPU success.
struct Counts {
    std::atomic<int> created{0}, destroyed{0}, bound{0}, synchronized{0};
    bool fail_capabilities = false;
    bool fail_bind = false;
};
class HostResources final : public TestCudaResources {
public:
    HostResources(Device device, std::shared_ptr<Counts> counts)
        : device_(device), counts_(std::move(counts)) { ++counts_->created; }
    ~HostResources() override { ++counts_->destroyed; }
    CudaCapabilities capabilities() const override {
        if (counts_->fail_capabilities) throw std::runtime_error("injected capability failure");
        CudaCapabilities result;
        result.compute_major = device_.index; // Device-key oracle, no hardware fact.
        return result;
    }
    std::unique_ptr<DeviceGuard> select_device() override { return {}; }
    void* cublas_handle() const noexcept override { return const_cast<HostResources*>(this); }
    void bind_compute_stream(void* stream) override {
        EXPECT_NE(stream, nullptr);
        if (counts_->fail_bind) throw std::runtime_error("injected configuration failure");
        ++counts_->bound;
    }
    void synchronize() override { ++counts_->synchronized; }
private:
    Device device_;
    std::shared_ptr<Counts> counts_;
};
class HostRaw final : public RawAllocator {
public:
    std::atomic<int> allocations{0}, frees{0};
    void* allocate(size_t bytes, Device) override { ++allocations; return cpu_raw_allocate(bytes); }
    void deallocate(void* pointer, Device) noexcept override { ++frees; cpu_raw_free(pointer); }
};
struct HostRuntime {
    std::shared_ptr<HostRaw> raw = std::make_shared<HostRaw>();
    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    CudaResourcesFactory factory() {
        return [counts = counts](Device device) { return std::make_unique<HostResources>(device, counts); };
    }
};
// Constructed before the function-local default Runtime. Its destruction
// therefore exercises actual late process teardown after Runtime destruction.
struct LateOwners {
    Allocation allocation;
    ScratchLease scratch;
    unsigned char* scratch_pointer = nullptr;
    ~LateOwners() {
        if (allocation.data() && static_cast<unsigned char*>(allocation.data())[0] != 0x71)
            std::abort();
        if (scratch_pointer && scratch_pointer[0] != 0x72) std::abort();
    }
} late_owners;
}

TEST(RuntimeContext, CpuIdentityAndAllocatorAreCanonical) {
    Runtime runtime;
    auto cpu = runtime.cpu();
    EXPECT_EQ(cpu, runtime.cpu());
    EXPECT_EQ(cpu->device(), Device());
    EXPECT_NE(dynamic_cast<CPUAllocator*>(cpu->allocator().get()), nullptr);
    EXPECT_EQ(cpu->allocator(), runtime.allocator(Device()));
    EXPECT_EQ(cpu->scratch(), runtime.scratch(Device()));
}
TEST(RuntimeContext, LateDefaultRuntimeTeardownKeepsOutstandingOwnersValid) {
    late_owners.allocation = default_runtime().allocator(Device())->allocate(8);
    static_cast<unsigned char*>(late_owners.allocation.data())[0] = 0x71;
    late_owners.scratch = default_runtime().scratch(Device())->acquire(8);
    late_owners.scratch_pointer = static_cast<unsigned char*>(late_owners.scratch.allocate(8));
    late_owners.scratch_pointer[0] = 0x72;
    EXPECT_EQ(static_cast<unsigned char*>(late_owners.allocation.data())[0], 0x71);
}
TEST(RuntimeContext, IsolatedRuntimesOwnDifferentContextsAndPools) {
    Runtime first, second;
    EXPECT_NE(first.cpu(), second.cpu());
    EXPECT_NE(first.memory_pool(), second.memory_pool());
    auto allocation = first.cpu()->allocator()->allocate(11);
    EXPECT_EQ(first.memory_pool()->stats().current_bytes, 11u);
    EXPECT_EQ(second.memory_pool()->stats().current_bytes, 0u);
}
TEST(RuntimeContext, RuntimeDestructionReleasesUnheldContexts) {
    std::weak_ptr<CpuContext> weak;
    std::weak_ptr<MemoryPool> pool;
    {
        Runtime runtime;
        weak = runtime.cpu();
        pool = runtime.memory_pool();
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_TRUE(pool.expired());
}
TEST(RuntimeContext, HeldCpuContextOutlivesRuntime) {
    std::shared_ptr<CpuContext> held;
    std::weak_ptr<CpuContext> weak;
    {
        Runtime runtime;
        held = runtime.cpu();
        weak = held;
    }
    EXPECT_FALSE(weak.expired());
    EXPECT_NE(held->allocator()->allocate(8).data(), nullptr);
    held.reset();
    EXPECT_TRUE(weak.expired());
}
TEST(RuntimeContext, AllocationOutlivesRuntimeAndReleasesPoolExactlyOnce) {
    auto raw = std::make_shared<HostRaw>();
    std::weak_ptr<CpuContext> context;
    std::weak_ptr<MemoryPool> pool;
    Allocation held;
    {
        Runtime runtime(raw);
        context = runtime.cpu();
        pool = runtime.memory_pool();
        held = runtime.cpu()->allocator()->allocate(19);
        std::memset(held.data(), 0x5a, 19);
    }
    EXPECT_TRUE(context.expired());
    EXPECT_FALSE(pool.expired());
    EXPECT_EQ(static_cast<unsigned char*>(held.data())[18], 0x5a);
    EXPECT_EQ(raw->frees, 0);
    held.reset();
    EXPECT_TRUE(pool.expired());
    EXPECT_EQ(raw->frees, 1);
}
TEST(RuntimeContext, ScratchLeaseOutlivesRuntimeAndReleasesArenaExactlyOnce) {
    auto raw = std::make_shared<HostRaw>();
    std::weak_ptr<ScratchArena> arena;
    ScratchLease held;
    unsigned char* pointer = nullptr;
    {
        Runtime runtime(raw);
        arena = runtime.cpu()->scratch();
        held = runtime.cpu()->scratch()->acquire(12);
        pointer = static_cast<unsigned char*>(held.allocate(12));
        pointer[11] = 0x6b;
    }
    EXPECT_FALSE(arena.expired());
    EXPECT_EQ(pointer[11], 0x6b);
    EXPECT_EQ(raw->frees, 0);
    held = ScratchLease();
    EXPECT_TRUE(arena.expired());
    EXPECT_EQ(raw->frees, 1);
}
TEST(RuntimeContext, MemoryPoolResetAndGlobalBudgetArePreserved) {
    HostRuntime f;
    Runtime runtime(f.raw, f.factory());
    auto pool = runtime.memory_pool();
    pool->set_max_cached_bytes(64);
    auto cpu = runtime.cpu()->allocator()->allocate(7);
    auto gpu = runtime.cuda(4)->allocator()->allocate(9);
    pool->reset_stats();
    EXPECT_EQ(pool->stats().current_bytes, 16u);
    EXPECT_EQ(pool->stats().reserved_bytes, 128u);
    cpu.reset();
    gpu.reset();
    EXPECT_EQ(pool->stats().current_bytes, 0u);
    EXPECT_EQ(pool->stats().cached_bytes, 64u);
    EXPECT_EQ(pool->stats().reserved_bytes, 64u);
    EXPECT_EQ(f.raw->frees, 1);
    pool->clear_cache();
    EXPECT_EQ(f.raw->frees, 2);
}
TEST(RuntimeContext, ConcurrentCpuLookupPublishesOneIdentity) {
    Runtime runtime;
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<std::shared_ptr<CpuContext>>> work;
    for (int i = 0; i < 16; ++i)
        work.push_back(std::async(std::launch::async, [&] { ready.wait(); return runtime.cpu(); }));
    start.set_value();
    auto first = work.front().get();
    for (size_t i = 1; i < work.size(); ++i) EXPECT_EQ(work[i].get(), first);
}
TEST(RuntimeContext, HostDeviceKeysAreDistinctAndStable) {
    HostRuntime f;
    Runtime runtime(f.raw, f.factory());
    auto a = runtime.cuda(2), b = runtime.cuda(7);
    EXPECT_EQ(a, runtime.cuda(2));
    EXPECT_NE(a, b);
    EXPECT_EQ(a->device(), Device(DeviceType::CUDA, 2));
    EXPECT_EQ(b->capabilities().compute_major, 7);
    EXPECT_NE(a->allocator(), b->allocator());
    EXPECT_NE(a->scratch(), b->scratch());
    EXPECT_NE(dynamic_cast<CUDAAllocator*>(a->allocator().get()), nullptr);
    EXPECT_EQ(a->allocator(), runtime.allocator(a->device()));
    EXPECT_EQ(a->scratch(), runtime.scratch(a->device()));
    EXPECT_NE(a->compute_stream().native_handle(), nullptr);
    EXPECT_EQ(f.counts->created, 2);
}
TEST(RuntimeContext, ConcurrentHostCudaLookupPublishesOneCompleteIdentity) {
    HostRuntime f;
    Runtime runtime(f.raw, f.factory());
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<std::shared_ptr<CudaContext>>> work;
    for (int i = 0; i < 16; ++i)
        work.push_back(std::async(std::launch::async, [&] {
            ready.wait();
            auto context = runtime.cuda(3);
            EXPECT_EQ(context->allocator(), runtime.allocator(context->device()));
            EXPECT_EQ(context->scratch(), runtime.scratch(context->device()));
            return context;
        }));
    start.set_value();
    auto first = work.front().get();
    for (size_t i = 1; i < work.size(); ++i) EXPECT_EQ(work[i].get(), first);
    EXPECT_EQ(f.counts->created, 1);
}
TEST(RuntimeContext, CapabilityFailureUnwindsAndLookupRetriesCleanly) {
    HostRuntime f;
    Runtime runtime(f.raw, f.factory());
    f.counts->fail_capabilities = true;
    EXPECT_THROW(runtime.cuda(3), std::runtime_error);
    EXPECT_EQ(f.counts->created, 1);
    EXPECT_EQ(f.counts->destroyed, 1);
    f.counts->fail_capabilities = false;
    auto context = runtime.cuda(3);
    EXPECT_EQ(context, runtime.cuda(3));
    EXPECT_EQ(f.counts->created, 2);
}
TEST(RuntimeContext, LibraryInitializationFailureUnwindsAndRetries) {
    HostRuntime f;
    int attempts = 0;
    Runtime runtime(f.raw, [&](Device device) -> std::unique_ptr<CudaResources> {
        auto partial = std::make_unique<HostResources>(device, f.counts);
        if (++attempts == 1) throw std::runtime_error("injected cuBLAS initialization failure");
        return partial;
    });
    EXPECT_THROW(runtime.cuda(3), std::runtime_error);
    EXPECT_EQ(f.counts->destroyed, 1);
    EXPECT_EQ(runtime.cuda(3), runtime.cuda(3));
    EXPECT_EQ(attempts, 2);
}
TEST(RuntimeContext, NullResourcesAndInvalidOrdinalsNeverPublish) {
    Runtime runtime(system_raw_allocator(), [](Device) { return std::unique_ptr<CudaResources>(); });
    EXPECT_THROW(runtime.cuda(1), std::invalid_argument);
    EXPECT_THROW(runtime.cuda(1), std::invalid_argument);
    EXPECT_THROW(runtime.cuda(-1), std::invalid_argument);
    EXPECT_NE(runtime.cpu(), nullptr);
}
TEST(RuntimeContext, HeldCudaContextDefersResourceDestructionAndCanBeRecreated) {
    HostRuntime f;
    std::shared_ptr<CudaContext> held;
    {
        Runtime runtime(f.raw, f.factory());
        held = runtime.cuda(5);
    }
    EXPECT_EQ(f.counts->destroyed, 0);
    held->synchronize();
    EXPECT_EQ(f.counts->synchronized, 1);
    held.reset();
    EXPECT_EQ(f.counts->destroyed, 1);
    {
        Runtime recreated(f.raw, f.factory());
        EXPECT_EQ(recreated.cuda(5)->device().index, 5);
    }
    EXPECT_EQ(f.counts->created, 2);
    EXPECT_EQ(f.counts->destroyed, 2);
}
TEST(RuntimeContext, CublasLeaseRetainsResourcesAfterContextAndRuntimeDestruction) {
    HostRuntime f;
    auto runtime = std::make_unique<Runtime>(f.raw, f.factory());
    auto context = runtime->cuda(1);
    {
        auto lease = context->acquire_cublas();
        EXPECT_NE(lease.handle(), nullptr);
        context.reset();
        runtime.reset();
        EXPECT_EQ(f.counts->destroyed, 0);
        EXPECT_NE(lease.handle(), nullptr);
    }
    EXPECT_EQ(f.counts->destroyed, 1);
}
TEST(RuntimeContext, CublasLeaseSerializesUseAndBindsEveryAcquisition) {
    HostRuntime f;
    Runtime runtime(f.raw, f.factory());
    auto context = runtime.cuda(1);
    std::atomic<int> inside{0}, maximum{0};
    std::vector<std::future<void>> work;
    for (int i = 0; i < 8; ++i) work.push_back(std::async(std::launch::async, [&] {
        auto lease = context->acquire_cublas();
        const int active = ++inside;
        maximum.store(std::max(maximum.load(), active));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        --inside;
    }));
    for (auto& task : work) task.get();
    EXPECT_EQ(maximum, 1);
    EXPECT_EQ(f.counts->bound, 9);
}
TEST(RuntimeContext, FailedLeaseConfigurationReleasesLockAndOwner) {
    HostRuntime f;
    Runtime runtime(f.raw, f.factory());
    auto context = runtime.cuda(1);
    f.counts->fail_bind = true;
    EXPECT_THROW(context->acquire_cublas(), std::runtime_error);
    f.counts->fail_bind = false;
    auto lease = context->acquire_cublas();
    EXPECT_NE(lease.handle(), nullptr);
    EXPECT_EQ(f.counts->bound, 2);
}
TEST(RuntimeContext, HostCudaAllocationAndScratchOutliveContextLookupOwner) {
    HostRuntime f;
    Allocation allocation;
    ScratchLease scratch;
    {
        Runtime runtime(f.raw, f.factory());
        auto context = runtime.cuda(6);
        allocation = context->allocator()->allocate(13);
        scratch = context->scratch()->acquire(15);
        std::memset(scratch.allocate(15), 0x55, 15);
    }
    EXPECT_EQ(f.counts->destroyed, 0);
    EXPECT_EQ(allocation.device().index, 6);
    EXPECT_EQ(scratch.device().index, 6);
    EXPECT_EQ(f.raw->frees, 0);
    scratch = ScratchLease();
    allocation.reset();
    EXPECT_EQ(f.raw->frees, 2);
    EXPECT_EQ(f.counts->destroyed, 1);
}
#ifndef WITH_CUDA
TEST(RuntimeContext, CpuBuildRejectsRealCudaContextAndSyncIsNoOp) {
    Runtime runtime;
    EXPECT_THROW(runtime.cuda(0), std::runtime_error);
    EXPECT_THROW(runtime.cuda(0), std::runtime_error);
    EXPECT_THROW(runtime.allocator(Device(DeviceType::CUDA, 0)), std::runtime_error);
    EXPECT_THROW(runtime.scratch(Device(DeviceType::CUDA, 0)), std::runtime_error);
    EXPECT_NO_THROW(runtime.synchronize_current_cuda());
    EXPECT_NE(runtime.cpu(), nullptr);
}
#else
namespace {
bool real_cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}
TEST(RuntimeContextCuda, ExactDeviceCapabilitiesAndAmbientRestoration) {
    if (!real_cuda_available()) GTEST_SKIP() << "No usable CUDA device";
    int ambient = -1, count = 0;
    ASSERT_EQ(cudaGetDevice(&ambient), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    const int ordinal = count > 1 ? (ambient + 1) % count : ambient;
    Runtime runtime;
    auto context = runtime.cuda(ordinal);
    EXPECT_EQ(context, runtime.cuda(ordinal));
    EXPECT_EQ(context->device().index, ordinal);
    int after = -1;
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, ambient);
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, ordinal), cudaSuccess);
    EXPECT_EQ(context->capabilities().compute_major, properties.major);
    EXPECT_EQ(context->capabilities().compute_minor, properties.minor);
    EXPECT_EQ(context->capabilities().supports_bfloat16, properties.major >= 8);
    EXPECT_EQ(context->allocator(), runtime.allocator(context->device()));
    EXPECT_EQ(context->scratch(), runtime.scratch(context->device()));
    cudaStream_t displaced = nullptr;
    {
        DeviceGuard guard(context->device());
        ASSERT_EQ(cudaStreamCreate(&displaced), cudaSuccess);
    }
    {
        auto lease = context->acquire_cublas();
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(1);
        auto handle = static_cast<cublasHandle_t>(lease.handle());
        ASSERT_EQ(cublasGetStream(handle, &stream), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(stream, context->compute_stream().native_handle());
        int version = -1;
        ASSERT_EQ(cublasGetVersion(handle, &version), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(context->capabilities().cublas_version, version);
        EXPECT_EQ(context->capabilities().cublas_bfloat16, properties.major >= 8 && version >= 11000);
        // A prior handle user cannot leave a different stream behind.
        ASSERT_EQ(cublasSetStream(handle, displaced), CUBLAS_STATUS_SUCCESS);
    }
    {
        auto lease = context->acquire_cublas();
        cudaStream_t stream = displaced;
        ASSERT_EQ(cublasGetStream(static_cast<cublasHandle_t>(lease.handle()), &stream), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(stream, context->compute_stream().native_handle());
    }
    {
        DeviceGuard guard(context->device());
        ASSERT_EQ(cudaStreamDestroy(displaced), cudaSuccess);
    }
    context->synchronize();
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, ambient);
}
TEST(RuntimeContextCuda, InvalidOrdinalFailureRestoresAndValidLookupStillWorks) {
    if (!real_cuda_available()) GTEST_SKIP() << "No usable CUDA device";
    int ambient = -1, count = 0, after = -1;
    ASSERT_EQ(cudaGetDevice(&ambient), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    Runtime runtime;
    EXPECT_THROW(runtime.cuda(count), std::runtime_error);
    EXPECT_THROW(runtime.cuda(count), std::runtime_error);
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, ambient);
    EXPECT_EQ(runtime.cuda(ambient)->device().index, ambient);
    (void)cudaGetLastError();
}
TEST(RuntimeContextCuda, ContextHandlePerformsRepeatedRealGemmAndTeardownRestores) {
    if (!real_cuda_available()) GTEST_SKIP() << "No usable CUDA device";
    int ambient = -1, count = 0;
    ASSERT_EQ(cudaGetDevice(&ambient), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    const int ordinal = count > 1 ? (ambient + 1) % count : ambient;
    const float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8};
    const float expected[] = {23, 34, 31, 46}; // column-major product
    for (int recreation = 0; recreation < 2; ++recreation) {
        Runtime runtime;
        auto context = runtime.cuda(ordinal);
        auto first = context->allocator()->allocate(sizeof(a));
        auto second = context->allocator()->allocate(sizeof(b));
        auto result = context->allocator()->allocate(sizeof(a));
        {
            DeviceGuard guard(context->device());
            ASSERT_EQ(cudaMemcpy(first.data(), a, sizeof(a), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(second.data(), b, sizeof(b), cudaMemcpyHostToDevice), cudaSuccess);
            cudaPointerAttributes attributes{};
            ASSERT_EQ(cudaPointerGetAttributes(&attributes, first.data()), cudaSuccess);
            EXPECT_EQ(attributes.device, ordinal);
        }
        for (int repeated = 0; repeated < 2; ++repeated) {
            {
                auto lease = context->acquire_cublas();
                int current = -1;
                ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
                EXPECT_EQ(current, ordinal);
                const float alpha = 1, beta = 0;
                ASSERT_EQ(cublasSgemm(static_cast<cublasHandle_t>(lease.handle()), CUBLAS_OP_N,
                    CUBLAS_OP_N, 2, 2, 2, &alpha, static_cast<float*>(first.data()), 2,
                    static_cast<float*>(second.data()), 2, &beta, static_cast<float*>(result.data()), 2),
                    CUBLAS_STATUS_SUCCESS);
            }
            context->synchronize();
            float values[4];
            {
                DeviceGuard guard(context->device());
                ASSERT_EQ(cudaMemcpy(values, result.data(), sizeof(values), cudaMemcpyDeviceToHost), cudaSuccess);
            }
            for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(values[i], expected[i]);
        }
    }
    int after = -1;
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, ambient);
}
TEST(RuntimeContextCuda, RealAllocationAndScratchSurviveRuntimeTeardown) {
    if (!real_cuda_available()) GTEST_SKIP() << "No usable CUDA device";
    int ordinal = -1;
    ASSERT_EQ(cudaGetDevice(&ordinal), cudaSuccess);
    Allocation allocation;
    ScratchLease scratch;
    void* pointer = nullptr;
    std::weak_ptr<CudaContext> context;
    {
        Runtime runtime;
        context = runtime.cuda(ordinal);
        allocation = runtime.cuda(ordinal)->allocator()->allocate(32);
        scratch = runtime.cuda(ordinal)->scratch()->acquire(32);
        pointer = scratch.allocate(32);
        DeviceGuard guard(allocation.device());
        ASSERT_EQ(cudaMemset(allocation.data(), 0x31, 32), cudaSuccess);
        ASSERT_EQ(cudaMemset(pointer, 0x42, 32), cudaSuccess);
    }
    EXPECT_TRUE(context.expired());
    DeviceGuard guard(allocation.device());
    unsigned char values[32];
    ASSERT_EQ(cudaMemcpy(values, allocation.data(), 32, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(values[31], 0x31);
    ASSERT_EQ(cudaMemcpy(values, pointer, 32, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(values[31], 0x42);
}
TEST(RuntimeContextCuda, ConcurrentRealLookupAndDefaultCompatibility) {
    if (!real_cuda_available()) GTEST_SKIP() << "No usable CUDA device";
    int ordinal = -1;
    ASSERT_EQ(cudaGetDevice(&ordinal), cudaSuccess);
    Runtime runtime;
    std::vector<std::future<std::shared_ptr<CudaContext>>> work;
    for (int i = 0; i < 8; ++i)
        work.push_back(std::async(std::launch::async, [&] { return runtime.cuda(ordinal); }));
    auto first = work.front().get();
    for (size_t i = 1; i < work.size(); ++i) EXPECT_EQ(first, work[i].get());
    auto context = default_runtime().cuda(ordinal);
    EXPECT_EQ(default_runtime().allocator(context->device()), context->allocator());
    EXPECT_EQ(default_runtime().scratch(context->device()), context->scratch());
    EXPECT_NO_THROW(default_runtime().synchronize_current_cuda());
}
#endif
