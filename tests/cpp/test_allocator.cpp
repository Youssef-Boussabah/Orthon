#include <gtest/gtest.h>
#include "cuda_test_resources.h"
#include <orthon/core/buffer.h>
#include <orthon/core/tensor.h>
#include <orthon/memory/allocation.h>
#include <orthon/memory/cpu_allocator.h>
#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/pool.h>
#include <orthon/memory/scratch.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace orthon;

namespace {
// Host-backed device-key/failure model, never a claimed CUDA execution.
class TestRaw final : public RawAllocator {
public:
    int attempts = 0;
    int fail_on = 0;
    int frees = 0;
    bool wrong_device = false;
    std::unordered_map<void*, Device> live;
    void* allocate(size_t bytes, Device device) override {
        if (++attempts == fail_on) throw std::bad_alloc();
        void* pointer = cpu_raw_allocate(bytes);
        try { live.emplace(pointer, device); }
        catch (...) { cpu_raw_free(pointer); throw; }
        return pointer;
    }
    void deallocate(void* pointer, Device device) noexcept override {
        auto found = live.find(pointer);
        if (found == live.end()) { wrong_device = true; return; }
        if (found->second != device) wrong_device = true;
        live.erase(found);
        ++frees;
        cpu_raw_free(pointer);
    }
};
struct Fixture {
    std::shared_ptr<TestRaw> raw = std::make_shared<TestRaw>();
    std::shared_ptr<MemoryPool> pool = std::make_shared<MemoryPool>(raw);
    std::shared_ptr<CPUAllocator> cpu = std::make_shared<CPUAllocator>(pool);
};
}

TEST(AllocationOwnership, MoveOnlyAndNoexceptMove) {
    EXPECT_FALSE(std::is_copy_constructible<Allocation>::value);
    EXPECT_FALSE(std::is_copy_assignable<Allocation>::value);
    EXPECT_TRUE(std::is_nothrow_move_constructible<Allocation>::value);
    EXPECT_TRUE(std::is_nothrow_move_assignable<Allocation>::value);
    EXPECT_FALSE(std::is_copy_constructible<ScratchLease>::value);
}
TEST(AllocationOwnership, MoveConstructionAndDestructionReleaseOnce) {
    Fixture f;
    f.pool->set_enabled(false);
    {
        Allocation first = f.cpu->allocate(7);
        void* pointer = first.data();
        Allocation moved(std::move(first));
        EXPECT_EQ(first.data(), nullptr);
        EXPECT_EQ(first.nbytes(), 0u);
        EXPECT_EQ(moved.data(), pointer);
        EXPECT_EQ(moved.nbytes(), 7u);
        EXPECT_EQ(moved.capacity(), 64u);
        EXPECT_EQ(f.raw->frees, 0);
    }
    EXPECT_EQ(f.raw->frees, 1);
    EXPECT_TRUE(f.raw->live.empty());
}
TEST(AllocationOwnership, MoveAssignmentReleasesPreviousAndSelfMoveIsSafe) {
    Fixture f;
    f.pool->set_enabled(false);
    Allocation first = f.cpu->allocate(7);
    Allocation second = f.cpu->allocate(9);
    void* pointer = first.data();
    second = std::move(first);
    EXPECT_EQ(f.raw->frees, 1);
    EXPECT_EQ(second.data(), pointer);
    second = std::move(second);
    EXPECT_EQ(second.data(), pointer);
    second.reset();
    second.reset();
    EXPECT_EQ(f.raw->frees, 2);
}
TEST(AllocationOwnership, ZeroByteRecordKeepsDeviceAndIsSafe) {
    Fixture f;
    Allocation zero = f.cpu->allocate(0);
    EXPECT_EQ(zero.data(), nullptr);
    EXPECT_EQ(zero.nbytes(), 0u);
    EXPECT_EQ(zero.capacity(), 0u);
    EXPECT_EQ(zero.device(), Device{});
    EXPECT_EQ(zero.allocator(), f.cpu);
    EXPECT_EQ(f.raw->attempts, 0);
    zero.reset();
    EXPECT_EQ(f.raw->frees, 0);
}
TEST(AllocationOwnership, RetainsAllocatorAndPoolAfterExternalOwnersDisappear) {
    auto raw = std::make_shared<TestRaw>();
    std::weak_ptr<Allocator> weak_allocator;
    std::weak_ptr<MemoryPool> weak_pool;
    Allocation allocation;
    {
        auto pool = std::make_shared<MemoryPool>(raw);
        auto cpu = std::make_shared<CPUAllocator>(pool);
        weak_pool = pool;
        weak_allocator = cpu;
        allocation = cpu->allocate(9);
    }
    EXPECT_FALSE(weak_allocator.expired());
    EXPECT_FALSE(weak_pool.expired());
    allocation.reset();
    EXPECT_TRUE(weak_allocator.expired());
    EXPECT_TRUE(weak_pool.expired());
    EXPECT_EQ(raw->frees, 1);
}
TEST(AllocationOwnership, StackAllocatorRejectsBeforeAcquiringResource) {
    Fixture f;
    CPUAllocator stack(f.pool);
    EXPECT_THROW(stack.allocate(64), std::bad_weak_ptr);
    EXPECT_EQ(f.raw->attempts, 0);
}
TEST(AllocationOwnership, BufferDirectRecordReportsRequestedAndCapacity) {
    Buffer buffer(17, Device{});
    EXPECT_EQ(buffer.data(), buffer.allocation().data());
    EXPECT_EQ(buffer.nbytes(), 17u);
    EXPECT_EQ(buffer.capacity(), 64u);
    EXPECT_NE(dynamic_cast<CPUAllocator*>(buffer.allocation().allocator().get()), nullptr);
    EXPECT_EQ(buffer.version(), 0u);
}
TEST(AllocationOwnership, CPUAlignmentIsAtLeast64Bytes) {
    auto cpu = default_runtime().cpu()->allocator();
    for (size_t bytes : {1u, 63u, 64u, 65u, 129u}) {
        auto allocation = cpu->allocate(bytes);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation.data()) % 64, 0u);
        EXPECT_EQ(allocation.capacity() % 64, 0u);
        EXPECT_GE(allocation.capacity(), bytes);
    }
}
TEST(AllocationOwnership, FailureUnwindsPreviouslyAcquiredRecords) {
    Fixture f;
    f.pool->set_enabled(false);
    f.raw->fail_on = 2;
    EXPECT_THROW({
        auto targets = f.cpu->allocate(32);
        auto losses = f.cpu->allocate(32);
        (void)targets; (void)losses;
    }, std::bad_alloc);
    EXPECT_EQ(f.raw->frees, 1);
    EXPECT_TRUE(f.raw->live.empty());
    EXPECT_EQ(f.pool->stats().current_bytes, 0u);
    EXPECT_EQ(f.pool->stats().reserved_bytes, 0u);
    EXPECT_EQ(f.pool->stats().total_allocations, 1u);
    auto usable = f.cpu->allocate(32);
    EXPECT_NE(usable.data(), nullptr);
}
TEST(AllocationCache, SameBinReusesCapacityWithNewRequestedSize) {
    Fixture f;
    auto first = f.cpu->allocate(65);
    void* pointer = first.data();
    first.reset();
    auto second = f.cpu->allocate(100);
    EXPECT_EQ(second.data(), pointer);
    EXPECT_EQ(second.nbytes(), 100u);
    EXPECT_EQ(second.capacity(), 128u);
    const auto stats = f.pool->stats();
    EXPECT_EQ(stats.current_bytes, 100u);
    EXPECT_EQ(stats.reserved_bytes, 128u);
    EXPECT_EQ(stats.cached_bytes, 0u);
    EXPECT_EQ(stats.raw_allocations, 1u);
    EXPECT_EQ(stats.reused_allocations, 1u);
}
TEST(AllocationCache, DeviceKeysIsolateCPUAndCUDAOrdinalsWithFakeBackend) {
    Fixture f;
    auto gpu0 = std::make_shared<CUDAAllocator>(Device(DeviceType::CUDA, 0), f.pool, host_test_stream(Device(DeviceType::CUDA, 0)));
    auto gpu1 = std::make_shared<CUDAAllocator>(Device(DeviceType::CUDA, 1), f.pool, host_test_stream(Device(DeviceType::CUDA, 1)));
    auto cpu = f.cpu->allocate(64);
    void* cpu_ptr = cpu.data(); cpu.reset();
    auto zero = gpu0->allocate(64);
    void* zero_ptr = zero.data(); zero.reset();
    auto one = gpu1->allocate(64);
    void* one_ptr = one.data(); one.reset();
    EXPECT_NE(cpu_ptr, zero_ptr);
    EXPECT_NE(zero_ptr, one_ptr);
    auto cpu_again = f.cpu->allocate(64);
    auto zero_again = gpu0->allocate(64);
    auto one_again = gpu1->allocate(64);
    EXPECT_EQ(cpu_again.data(), cpu_ptr);
    EXPECT_EQ(zero_again.data(), zero_ptr);
    EXPECT_EQ(one_again.data(), one_ptr);
    EXPECT_EQ(one_again.device(), Device(DeviceType::CUDA, 1));
    cpu_again.reset(); zero_again.reset(); one_again.reset();
    f.pool->clear_cache();
    EXPECT_FALSE(f.raw->wrong_device);
    EXPECT_TRUE(f.raw->live.empty());
}
TEST(AllocationCache, DisabledCacheAlwaysUsesRawAllocation) {
    Fixture f;
    f.pool->set_enabled(false);
    for (int i = 0; i < 3; ++i) { auto allocation = f.cpu->allocate(9); }
    EXPECT_EQ(f.raw->frees, 3);
    EXPECT_EQ(f.pool->stats().raw_allocations, 3u);
    EXPECT_EQ(f.pool->stats().reused_allocations, 0u);
    EXPECT_EQ(f.pool->stats().cached_bytes, 0u);
    EXPECT_EQ(f.pool->stats().reserved_bytes, 0u);
}
TEST(AllocationCache, DisableLeavesExistingCacheUntilClearButBypassesIt) {
    Fixture f;
    auto allocation = f.cpu->allocate(64); allocation.reset();
    f.pool->set_enabled(false);
    auto uncached = f.cpu->allocate(64);
    EXPECT_EQ(f.pool->stats().raw_allocations, 2u);
    EXPECT_EQ(f.pool->stats().cached_bytes, 64u);
    uncached.reset();
    EXPECT_EQ(f.pool->stats().reserved_bytes, 64u);
    f.pool->clear_cache();
    EXPECT_EQ(f.raw->frees, 2);
}
TEST(AllocationCache, ProcessBudgetCoversAllDevicesAndLoweringEvicts) {
    Fixture f;
    auto gpu = std::make_shared<CUDAAllocator>(Device(DeviceType::CUDA, 1), f.pool, host_test_stream(Device(DeviceType::CUDA, 1)));
    auto cpu = f.cpu->allocate(64); auto device = gpu->allocate(64);
    cpu.reset(); device.reset();
    EXPECT_EQ(f.pool->stats().cached_bytes, 128u);
    f.pool->set_max_cached_bytes(64);
    EXPECT_EQ(f.pool->stats().cached_bytes, 64u);
    EXPECT_EQ(f.pool->stats().reserved_bytes, 64u);
    EXPECT_EQ(f.raw->frees, 1);
    f.pool->set_max_cached_bytes(0);
    EXPECT_EQ(f.raw->frees, 2);
    EXPECT_FALSE(f.raw->wrong_device);
}
TEST(AllocationCache, ReleaseEnforcesLimitAndClearNeverFreesLiveRecords) {
    Fixture f;
    f.pool->set_max_cached_bytes(64);
    auto live = f.cpu->allocate(65);
    auto cached = f.cpu->allocate(64); cached.reset();
    f.pool->clear_cache();
    EXPECT_EQ(f.raw->frees, 1);
    EXPECT_EQ(f.pool->stats().current_bytes, 65u);
    EXPECT_EQ(f.pool->stats().reserved_bytes, 128u);
    std::memset(live.data(), 7, 65);
    live.reset();
    EXPECT_EQ(f.raw->frees, 2);
    EXPECT_EQ(f.pool->stats().cached_bytes, 0u);
}
TEST(AllocationCache, ResetPreservesLiveAndCacheGaugesAndResetsPeakToLive) {
    Fixture f;
    auto live = f.cpu->allocate(17);
    auto cached = f.cpu->allocate(65); cached.reset();
    f.pool->reset_stats();
    auto stats = f.pool->stats();
    EXPECT_EQ(stats.total_allocations, 0u);
    EXPECT_EQ(stats.raw_allocations, 0u);
    EXPECT_EQ(stats.reused_allocations, 0u);
    EXPECT_EQ(stats.current_bytes, 17u);
    EXPECT_EQ(stats.peak_bytes, 17u);
    EXPECT_EQ(stats.cached_bytes, 128u);
    EXPECT_EQ(stats.reserved_bytes, 192u);
    live.reset();
    stats = f.pool->stats();
    EXPECT_EQ(stats.current_bytes, 0u);
    EXPECT_EQ(stats.cached_bytes, 192u);
    EXPECT_EQ(stats.reserved_bytes, 192u);
    f.pool->clear_cache();
    EXPECT_EQ(f.pool->stats().reserved_bytes, 0u);
}
TEST(AllocationCache, ResetThenUncachedReleaseCannotUnderflow) {
    Fixture f;
    f.pool->set_enabled(false);
    auto live = f.cpu->allocate(17);
    f.pool->reset_stats();
    live.reset();
    EXPECT_EQ(f.pool->stats().current_bytes, 0u);
    EXPECT_EQ(f.pool->stats().reserved_bytes, 0u);
    EXPECT_EQ(f.pool->stats().peak_bytes, 17u);
}
TEST(AllocationCache, FailedMissPreservesExistingCacheAndAccounting) {
    Fixture f;
    auto first = f.cpu->allocate(64);
    void* pointer = first.data(); first.reset();
    const auto before = f.pool->stats();
    f.raw->fail_on = 2;
    EXPECT_THROW(f.cpu->allocate(128), std::bad_alloc);
    const auto after = f.pool->stats();
    EXPECT_EQ(after.current_bytes, before.current_bytes);
    EXPECT_EQ(after.reserved_bytes, before.reserved_bytes);
    EXPECT_EQ(after.cached_bytes, before.cached_bytes);
    EXPECT_EQ(after.total_allocations, before.total_allocations);
    auto reused = f.cpu->allocate(64);
    EXPECT_EQ(reused.data(), pointer);
}
TEST(AllocationCache, OverflowFailsBeforeRawAllocation) {
    Fixture f;
    EXPECT_THROW(f.cpu->allocate((std::numeric_limits<size_t>::max)()), std::bad_alloc);
    EXPECT_EQ(f.raw->attempts, 0);
    EXPECT_EQ(f.pool->stats().reserved_bytes, 0u);
}
TEST(AllocationScratch, ActiveLeasesCannotReuseEachOthersRegions) {
    auto raw = std::make_shared<TestRaw>();
    auto arena = std::make_shared<ScratchArena>(Device{}, raw);
    auto first = arena->acquire(128);
    void* p = first.allocate(100);
    auto second = arena->acquire(128);
    void* q = second.allocate(100);
    EXPECT_NE(p, q);
    EXPECT_EQ(arena->stats().used_bytes, 256u);
    EXPECT_EQ(arena->stats().reserved_bytes, 2u << 20);
    first = ScratchLease{};
    auto next = arena->acquire(128);
    EXPECT_EQ(next.allocate(100), p);
    EXPECT_EQ(arena->stats().used_bytes, 256u);
}
TEST(AllocationScratch, GrowthDoesNotInvalidateAnActiveLease) {
    auto arena = std::make_shared<ScratchArena>(Device{});
    auto first = arena->acquire(64);
    auto* pointer = static_cast<unsigned char*>(first.allocate(64));
    std::memset(pointer, 9, 64);
    auto larger = arena->acquire((1 << 20) + 64);
    EXPECT_NE(larger.allocate(64), pointer);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(pointer[i], 9);
    EXPECT_EQ(arena->stats().reserved_bytes, 3u << 20);
}
TEST(AllocationScratch, ExceptionAndMoveReleaseLeaseExactlyOnce) {
    auto raw = std::make_shared<TestRaw>();
    auto arena = std::make_shared<ScratchArena>(Device{}, raw);
    void* pointer = nullptr;
    EXPECT_THROW({
        auto lease = arena->acquire(64);
        pointer = lease.allocate(64);
        ScratchLease moved(std::move(lease));
        throw std::runtime_error("injected operation failure");
    }, std::runtime_error);
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    { auto lease = arena->acquire(64); EXPECT_EQ(lease.allocate(64), pointer); }
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    arena.reset();
    EXPECT_EQ(raw->frees, 1);
}
TEST(AllocationScratch, LeaseRetainsArenaAfterExternalOwnerDisappears) {
    auto raw = std::make_shared<TestRaw>();
    ScratchLease lease;
    std::weak_ptr<ScratchArena> weak;
    {
        auto arena = std::make_shared<ScratchArena>(Device{}, raw);
        weak = arena;
        lease = arena->acquire(64);
    }
    EXPECT_FALSE(weak.expired());
    EXPECT_NE(lease.allocate(64), nullptr);
    lease = ScratchLease{};
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(raw->frees, 1);
}
TEST(AllocationScratch, FailedGrowthLeavesActiveAndIdleRegionsIntact) {
    auto raw = std::make_shared<TestRaw>();
    auto arena = std::make_shared<ScratchArena>(Device{}, raw);
    auto first = arena->acquire(64);
    void* pointer = first.allocate(64);
    raw->fail_on = 2;
    EXPECT_THROW(arena->acquire(2 << 20), std::bad_alloc);
    EXPECT_EQ(arena->stats().reserved_bytes, 1u << 20);
    EXPECT_EQ(arena->stats().used_bytes, 64u);
    first = ScratchLease{};
    auto reused = arena->acquire(64);
    EXPECT_EQ(reused.allocate(64), pointer);
}
TEST(AllocationScratch, ExhaustionDoesNotAdvanceUsedGauge) {
    auto arena = std::make_shared<ScratchArena>(Device{});
    auto lease = arena->acquire(64);
    lease.allocate(lease.capacity());
    EXPECT_THROW(lease.allocate(1), std::runtime_error);
    EXPECT_EQ(arena->stats().used_bytes, lease.capacity());
    EXPECT_EQ(lease.allocate(0), nullptr);
}
TEST(AllocationScratch, LegacyCursorAndScopedLeaseUseOneArenaSafely) {
    auto arena = std::make_shared<ScratchArena>(Device{});
    void* first = arena->legacy_allocate(100);
    void* second = arena->legacy_allocate(200);
    EXPECT_EQ(second, static_cast<char*>(first) + 128);
    auto lease = arena->acquire(64);
    EXPECT_NE(lease.allocate(64), first);
    arena->legacy_reset();
    EXPECT_EQ(arena->stats().used_bytes, 64u);
    EXPECT_EQ(arena->legacy_allocate(64), first);
    EXPECT_EQ(arena->stats().used_bytes, 128u);
    EXPECT_EQ(arena->stats().peak_used_bytes, 448u);
}
TEST(AllocationScratch, LegacyGrowthRequiresResetAndDoesNotDisturbActiveLease) {
    auto arena = std::make_shared<ScratchArena>(Device{});
    auto lease = arena->acquire(64);
    auto* active = static_cast<char*>(lease.allocate(64));
    active[0] = 7;
    arena->legacy_allocate(64);
    EXPECT_THROW(arena->legacy_allocate(2 << 20), std::runtime_error);
    arena->legacy_reset();
    EXPECT_NE(arena->legacy_allocate(2 << 20), nullptr);
    EXPECT_EQ(active[0], 7);
}
TEST(AllocationScratch, DeviceKeysAreIsolatedWithFakeBackend) {
    auto raw = std::make_shared<TestRaw>();
    auto cpu = std::make_shared<ScratchArena>(Device{}, raw);
    auto gpu0 = std::make_shared<ScratchArena>(Device(DeviceType::CUDA, 0), raw, host_test_stream(Device(DeviceType::CUDA, 0)));
    auto gpu1 = std::make_shared<ScratchArena>(Device(DeviceType::CUDA, 1), raw, host_test_stream(Device(DeviceType::CUDA, 1)));
    auto a = cpu->acquire(64); auto b = gpu0->acquire(64); auto c = gpu1->acquire(64);
    void* p = a.allocate(64); void* q = b.allocate(64); void* r = c.allocate(64);
    EXPECT_NE(p, q); EXPECT_NE(q, r);
    EXPECT_EQ(c.device(), Device(DeviceType::CUDA, 1));
    a = ScratchLease{}; b = ScratchLease{}; c = ScratchLease{};
    cpu.reset(); gpu0.reset(); gpu1.reset();
    EXPECT_EQ(raw->frees, 3);
    EXPECT_FALSE(raw->wrong_device);
}
TEST(AllocationScratch, ScratchAccountingIsSeparateFromTensorPool) {
    const auto before = default_runtime().memory_pool()->stats();
    auto arena = std::make_shared<ScratchArena>(Device{});
    { auto lease = arena->acquire(64); lease.allocate(64); }
    const auto after = default_runtime().memory_pool()->stats();
    EXPECT_EQ(after.reserved_bytes, before.reserved_bytes);
    EXPECT_EQ(after.total_allocations, before.total_allocations);
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    EXPECT_EQ(arena->stats().peak_used_bytes, 64u);
}

#ifdef WITH_CUDA
TEST(AllocationCUDA, BufferUsesCUDAAllocatorAndReportsExactOrdinal) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) GTEST_SKIP() << "no physical CUDA device";
    int ambient = 0; ASSERT_EQ(cudaGetDevice(&ambient), cudaSuccess);
    Buffer buffer(17, Device(DeviceType::CUDA, ambient));
    EXPECT_NE(dynamic_cast<CUDAAllocator*>(buffer.allocation().allocator().get()), nullptr);
    EXPECT_EQ(buffer.device(), Device(DeviceType::CUDA, ambient));
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, buffer.data()), cudaSuccess);
    EXPECT_EQ(attributes.device, ambient);
}
TEST(AllocationCUDA, RawMissHitFreeAndFailureRestoreAmbientDevice) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) GTEST_SKIP() << "no physical CUDA device";
    int ambient = 0; ASSERT_EQ(cudaGetDevice(&ambient), cudaSuccess);
    const int target = count > 1 ? (ambient + 1) % count : ambient;
    auto pool = std::make_shared<MemoryPool>();
    auto allocator = std::make_shared<CUDAAllocator>(Device(DeviceType::CUDA, target), pool);
    auto assert_ambient = [&] { int current = -1; EXPECT_EQ(cudaGetDevice(&current), cudaSuccess); EXPECT_EQ(current, ambient); };
    auto first = allocator->allocate(64); assert_ambient();
    void* pointer = first.data(); first.reset(); assert_ambient();
    auto reused = allocator->allocate(64); assert_ambient();
    EXPECT_EQ(reused.data(), pointer);
    reused.reset(); pool->clear_cache(); assert_ambient();
    pool->set_enabled(false);
    auto raw = allocator->allocate(64); raw.reset(); assert_ambient();
    auto invalid = std::make_shared<CUDAAllocator>(Device(DeviceType::CUDA, count), pool);
    EXPECT_THROW(invalid->allocate(64), std::runtime_error);
    assert_ambient();
    EXPECT_EQ(pool->stats().reserved_bytes, 0u);
}
TEST(AllocationCUDA, ScopedScratchReductionAndGrowthUseRealDeviceMemory) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) GTEST_SKIP() << "no physical CUDA device";
    auto arena = default_runtime().cuda(0)->scratch();
    auto active = arena->acquire(64);
    void* pointer = active.allocate(64);
    int expected = 19, actual = 0;
    ASSERT_EQ(cudaMemcpy(pointer, &expected, sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    { auto larger = arena->acquire(2 << 20); larger.allocate(64); }
    Tensor input = ones({4096}).to(Device(DeviceType::CUDA, 0));
    Tensor result = sum(input).to(Device{});
    EXPECT_EQ(static_cast<float*>(result.data())[0], 4096.f);
    ASSERT_EQ(cudaMemcpy(&actual, pointer, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(arena->stats().used_bytes, 64u);
}
#endif
