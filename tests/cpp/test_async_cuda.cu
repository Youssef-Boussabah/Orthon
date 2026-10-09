#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <orthon/backends/cuda/unary_kernels.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/pool.h>
#include <orthon/memory/scratch.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/matmul_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/ops/tensor_access.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/runtime/completion.h>
#include <orthon/runtime/runtime.h>
using namespace orthon;
namespace {
bool available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
cudaStream_t native(const CudaStream &s) {
    return static_cast<cudaStream_t>(s.native_handle());
}
__global__ void delayed_write(float *p, float value, unsigned long long cycles) {
    auto start = clock64();
    while (clock64() - start < cycles) {
    }
    *p = value;
}
__global__ void delayed_consumer(float *p, float *witness, unsigned long long cycles) {
    auto start = clock64();
    while (clock64() - start < cycles) {
    }
    *witness = *p;
    *p = 11;
}
constexpr unsigned long long delay = 150000000ULL;
constexpr unsigned long long stress_delay = 5000000ULL;
} // namespace
TEST(AsyncCuda, PhysicalStreamCublasAndAmbientRestoration) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    int ambient = -1;
    ASSERT_EQ(cudaGetDevice(&ambient), cudaSuccess);
    for (int i = 0; i < 3; ++i) {
        Runtime r;
        auto c = r.cuda(ambient);
        auto stream = c->compute_stream();
        EXPECT_NE(native(stream), nullptr);
        unsigned int flags = 0;
        ASSERT_EQ(cudaStreamGetFlags(native(stream), &flags), cudaSuccess);
        EXPECT_EQ(flags, cudaStreamNonBlocking);
        auto aux = c->create_stream();
        EXPECT_NE(native(aux), native(stream));
        {
            auto lease = c->acquire_cublas();
            cudaStream_t bound = nullptr;
            ASSERT_EQ(cublasGetStream(static_cast<cublasHandle_t>(lease.handle()), &bound),
                      CUBLAS_STATUS_SUCCESS);
            EXPECT_EQ(bound, native(stream));
        }
        c->synchronize_compute();
    }
    int after = -1;
    ASSERT_EQ(cudaGetDevice(&after), cudaSuccess);
    EXPECT_EQ(after, ambient);
}
TEST(AsyncCuda, EnqueueReturnsWhileEventNotReadyThenCompletes) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto c = r.cuda(0);
    auto s = c->compute_stream();
    auto x = c->allocator()->allocate(sizeof(float));
    DeviceGuard guard(c->device());
    delayed_write<<<1, 1, 0, native(s)>>>(static_cast<float *>(x.data()), 7, delay);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto t = s.record_completion();
    EXPECT_FALSE(t.ready());
    int host_progress = 0;
    for (int i = 0; i < 100; ++i)
        host_progress += i;
    EXPECT_EQ(host_progress, 4950);
    c->synchronize_compute();
    EXPECT_TRUE(t.ready());
    float value = 0;
    c->copy_d2h_blocking(&value, x.data(), sizeof(value));
    EXPECT_FLOAT_EQ(value, 7);
}
TEST(AsyncCuda, OrdinaryAddReturnsBeforeDelayedInputCompletes) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    const Device device(DeviceType::CUDA, 0);
    auto c = default_runtime().cuda(0);
    auto a = full({1}, 1).to(device), b = full({1}, 2).to(device);
    // Warm output size class; cudaMalloc on a miss may itself block.
    {
        auto warm = add(a, b);
        c->synchronize_compute();
    }
    DeviceGuard guard(device);
    delayed_write<<<1, 1, 0, native(c->compute_stream())>>>(static_cast<float *>(a.data()), 5,
                                                            delay);
    auto result = add(a, b);
    auto token = c->compute_stream().record_completion();
    EXPECT_FALSE(token.ready());
    EXPECT_FLOAT_EQ(scalar_read_value(result, 0, "async test"), 7);
    EXPECT_TRUE(token.ready());
}
TEST(AsyncCuda, ThreeStreamEventOrderingRepeated) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto c = r.cuda(0);
    auto a = c->create_stream(), b = c->create_stream(), compute = c->compute_stream();
    auto x = c->allocator()->allocate(sizeof(float));
    DeviceGuard guard(c->device());
    for (int i = 0; i < 24; ++i) {
        delayed_write<<<1, 1, 0, native(a)>>>(static_cast<float *>(x.data()), 1, stress_delay);
        auto first = a.record_completion();
        first.enqueue_wait_on(b);
        delayed_write<<<1, 1, 0, native(b)>>>(static_cast<float *>(x.data()), 2, 0);
        auto second = b.record_completion();
        second.enqueue_wait_on(compute);
        float value = 0;
        c->copy_d2h_blocking(&value, x.data(), sizeof(value));
        EXPECT_FLOAT_EQ(value, 2);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}
TEST(AsyncCuda, PoolCrossStreamDelayedConsumerReuseStress) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto c = r.cuda(0);
    auto a = c->create_stream(), b = c->create_stream();
    auto pool = r.memory_pool();
    auto aa = std::make_shared<CUDAAllocator>(c->device(), pool, a),
         bb = std::make_shared<CUDAAllocator>(c->device(), pool, b);
    auto observer = c->allocator()->allocate(256);
    DeviceGuard guard(c->device());
    for (int i = 0; i < 32; ++i) {
        auto x = aa->allocate(sizeof(float));
        auto p = x.data();
        const float initial = 7;
        ASSERT_EQ(cudaMemcpyAsync(p, &initial, sizeof(float), cudaMemcpyHostToDevice, native(a)),
                  cudaSuccess);
        a.synchronize();
        delayed_consumer<<<1, 1, 0, native(a)>>>(
            static_cast<float *>(p), static_cast<float *>(observer.data()), stress_delay);
        x.reset();
        auto y = bb->allocate(sizeof(float));
        EXPECT_EQ(y.data(), p);
        delayed_write<<<1, 1, 0, native(b)>>>(static_cast<float *>(y.data()), 22, 0);
        float observed = 0, value = 0;
        ASSERT_EQ(cudaMemcpyAsync(&observed, observer.data(), sizeof(float), cudaMemcpyDeviceToHost,
                                  native(b)),
                  cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(&value, y.data(), sizeof(float), cudaMemcpyDeviceToHost, native(b)),
            cudaSuccess);
        b.synchronize();
        EXPECT_FLOAT_EQ(observed, 7);
        EXPECT_FLOAT_EQ(value, 22);
    }
    EXPECT_EQ(pool->stats().raw_allocations, 2u);
    EXPECT_EQ(pool->stats().reused_allocations, 63u);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}
TEST(AsyncCuda, ScratchCrossStreamDelayedConsumerReuseStress) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto c = r.cuda(0);
    auto a = c->create_stream(), b = c->create_stream();
    auto arena = c->scratch();
    auto observer = c->allocator()->allocate(sizeof(float));
    DeviceGuard guard(c->device());
    size_t reserved = 0;
    for (int i = 0; i < 32; ++i) {
        void *p;
        {
            auto lease = arena->acquire(sizeof(float), a);
            p = lease.allocate(sizeof(float));
            float initial = 7;
            ASSERT_EQ(
                cudaMemcpyAsync(p, &initial, sizeof(float), cudaMemcpyHostToDevice, native(a)),
                cudaSuccess);
            a.synchronize();
            delayed_consumer<<<1, 1, 0, native(a)>>>(
                static_cast<float *>(p), static_cast<float *>(observer.data()), stress_delay);
        }
        EXPECT_EQ(arena->stats().used_bytes, 0u);
        {
            auto lease = arena->acquire(sizeof(float), b);
            EXPECT_EQ(lease.allocate(sizeof(float)), p);
            delayed_write<<<1, 1, 0, native(b)>>>(static_cast<float *>(p), 22, 0);
            float observed = 0, value = 0;
            ASSERT_EQ(cudaMemcpyAsync(&observed, observer.data(), sizeof(float),
                                      cudaMemcpyDeviceToHost, native(b)),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(&value, p, sizeof(float), cudaMemcpyDeviceToHost, native(b)),
                      cudaSuccess);
            b.synchronize();
            EXPECT_FLOAT_EQ(observed, 7);
            EXPECT_FLOAT_EQ(value, 22);
        }
        if (i == 0)
            reserved = arena->stats().reserved_bytes;
        EXPECT_EQ(arena->stats().reserved_bytes, reserved);
    }
    EXPECT_EQ(arena->stats().used_bytes, 0u);
    EXPECT_EQ(reserved, 1u << 20);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}
TEST(AsyncCuda, PendingClearCacheDisableAndEvictionAreSafe) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    for (int mode = 0; mode < 3; ++mode) {
        Runtime r;
        auto c = r.cuda(0);
        auto observer = c->allocator()->allocate(256);
        auto x = c->allocator()->allocate(sizeof(float));
        const float initial = 7;
        c->copy_h2d_snapshot(x.data(), &initial, sizeof(float));
        DeviceGuard guard(c->device());
        delayed_consumer<<<1, 1, 0, native(c->compute_stream())>>>(
            static_cast<float *>(x.data()), static_cast<float *>(observer.data()), delay);
        auto completion = c->compute_stream().record_completion();
        EXPECT_FALSE(completion.ready());
        if (mode == 1)
            r.memory_pool()->set_enabled(false);
        x.reset();
        if (mode == 0)
            r.memory_pool()->clear_cache();
        if (mode == 2)
            r.memory_pool()->set_max_cached_bytes(0);
        EXPECT_TRUE(completion.ready());
        EXPECT_EQ(r.memory_pool()->stats().cached_bytes, 0u);
        float value = 0;
        c->copy_d2h_blocking(&value, observer.data(), sizeof(float));
        EXPECT_FLOAT_EQ(value, 7);
    }
}
TEST(AsyncCuda, DelayedHostReadsWritesAndAmpFlagsAreOrdered) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    const Device d(DeviceType::CUDA, 0);
    auto c = default_runtime().cuda(0);
    auto x = full({1}, 1).to(d);
    DeviceGuard guard(d);
    delayed_write<<<1, 1, 0, native(c->compute_stream())>>>(static_cast<float *>(x.data()), 9,
                                                            stress_delay);
    EXPECT_FLOAT_EQ(scalar_read_value(x, 0, "read"), 9);
    delayed_write<<<1, 1, 0, native(c->compute_stream())>>>(static_cast<float *>(x.data()), 10,
                                                            stress_delay);
    auto host = x.to(Device{});
    EXPECT_FLOAT_EQ(*static_cast<float *>(host.data()), 10);
    scalar_write_value(x, 0, 12, "write");
    EXPECT_FLOAT_EQ(scalar_read_value(add(x, x), 0, "read"), 24);
    delayed_write<<<1, 1, 0, native(c->compute_stream())>>>(
        static_cast<float *>(x.data()), std::numeric_limits<float>::infinity(), stress_delay);
    EXPECT_FALSE(isfinite_all_values(x));
    delayed_write<<<1, 1, 0, native(c->compute_stream())>>>(
        static_cast<float *>(x.data()), std::numeric_limits<float>::infinity(), stress_delay);
    EXPECT_FALSE(amp_unscale_values(x, .5f).second);
    float source = 33;
    c->copy_h2d_snapshot(x.data(), &source, sizeof(source));
    source = 99;
    EXPECT_FLOAT_EQ(scalar_read_value(x, 0, "snapshot"), 33);
}
TEST(AsyncCuda, TokenAllocationAndScratchRetainPhysicalResources) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Allocation x;
    ScratchLease scratch;
    CompletionToken token;
    CudaStream stream;
    float *scratch_pointer = nullptr;
    {
        Runtime r;
        auto c = r.cuda(0);
        stream = c->compute_stream();
        x = c->allocator()->allocate(sizeof(float));
        scratch = c->scratch()->acquire(sizeof(float));
        scratch_pointer = static_cast<float *>(scratch.allocate(sizeof(float)));
        DeviceGuard guard(c->device());
        delayed_write<<<1, 1, 0, native(stream)>>>(static_cast<float *>(x.data()), 7, delay);
        delayed_write<<<1, 1, 0, native(stream)>>>(scratch_pointer, 8, 0);
        token = stream.record_completion();
    }
    EXPECT_FALSE(token.ready());
    token.wait_host();
    EXPECT_TRUE(token.ready());
    float values[2] = {};
    ASSERT_EQ(
        cudaMemcpyAsync(values, x.data(), sizeof(float), cudaMemcpyDeviceToHost, native(stream)),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(values + 1, scratch_pointer, sizeof(float), cudaMemcpyDeviceToHost,
                              native(stream)),
              cudaSuccess);
    stream.synchronize();
    EXPECT_FLOAT_EQ(values[0], 7);
    EXPECT_FLOAT_EQ(values[1], 8);
    scratch = {};
    x.reset();
    token = {};
    stream = {};
}
TEST(AsyncCuda, PerTokenEventRetirementCostIsMeasured) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto s = r.cuda(0)->compute_stream();
    s.synchronize();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        auto token = s.record_completion();
    }
    s.synchronize();
    auto elapsed =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin)
            .count() /
        1000;
    RecordProperty("event_create_record_destroy_us_per_token", std::to_string(elapsed));
    EXPECT_GT(elapsed, 0);
}
TEST(AsyncCuda, SameStreamPendingPoolReuseD2DAndMemsetStayAsync) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto c = r.cuda(0);
    auto s = c->compute_stream();
    auto observer = c->allocator()->allocate(256), source = c->allocator()->allocate(128),
         x = c->allocator()->allocate(sizeof(float));
    float initial = 7, next = 22;
    c->copy_h2d_snapshot(x.data(), &initial, sizeof(float));
    c->copy_h2d_snapshot(source.data(), &next, sizeof(float));
    void *p = x.data();
    DeviceGuard guard(c->device());
    delayed_consumer<<<1, 1, 0, native(s)>>>(static_cast<float *>(p),
                                             static_cast<float *>(observer.data()), delay);
    x.reset();
    auto y = c->allocator()->allocate(sizeof(float));
    EXPECT_EQ(y.data(), p);
    c->memset_async(y.data(), 0, sizeof(float));
    c->copy_d2d_async(y.data(), source.data(), sizeof(float));
    auto token = s.record_completion();
    EXPECT_FALSE(token.ready());
    float seen = 0, value = 0;
    c->copy_d2h_blocking(&seen, observer.data(), sizeof(float));
    c->copy_d2h_blocking(&value, y.data(), sizeof(float));
    EXPECT_FLOAT_EQ(seen, 7);
    EXPECT_FLOAT_EQ(value, 22);
    EXPECT_EQ(r.memory_pool()->stats().reused_allocations, 1u);
}
TEST(AsyncCuda, SameStreamPendingScratchReuseStaysAsync) {
    if (!available())
        GTEST_SKIP() << "No physical CUDA";
    Runtime r;
    auto c = r.cuda(0);
    auto s = c->compute_stream();
    auto arena = c->scratch();
    auto observer = c->allocator()->allocate(sizeof(float));
    void *p;
    DeviceGuard guard(c->device());
    {
        auto lease = arena->acquire(sizeof(float));
        p = lease.allocate(sizeof(float));
        float initial = 7;
        c->copy_h2d_snapshot(p, &initial, sizeof(float));
        delayed_consumer<<<1, 1, 0, native(s)>>>(static_cast<float *>(p),
                                                 static_cast<float *>(observer.data()), delay);
    }
    auto lease = arena->acquire(sizeof(float));
    EXPECT_EQ(lease.allocate(sizeof(float)), p);
    delayed_write<<<1, 1, 0, native(s)>>>(static_cast<float *>(p), 22, 0);
    auto token = s.record_completion();
    EXPECT_FALSE(token.ready());
    float seen = 0, value = 0;
    c->copy_d2h_blocking(&seen, observer.data(), sizeof(float));
    c->copy_d2h_blocking(&value, p, sizeof(float));
    EXPECT_FLOAT_EQ(seen, 7);
    EXPECT_FLOAT_EQ(value, 22);
    EXPECT_EQ(arena->stats().reserved_bytes, 1u << 20);
}
