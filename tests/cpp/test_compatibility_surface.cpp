// Initial-release native compatibility is exercised only here.
#include <gtest/gtest.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/cast_ops.h>
#include <orthon/ops/cpu/conv2d_ops.h>
#include <orthon/ops/cpu/cross_entropy_ops.h>
#include <orthon/ops/cpu/embedding_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/fusion_ops.h>
#include <orthon/ops/cpu/index_ops.h>
#include <orthon/ops/cpu/matmul_ops.h>
#include <orthon/ops/cpu/maxpool2d_ops.h>
#include <orthon/ops/cpu/nll_loss_ops.h>
#include <orthon/ops/cpu/reduce_ops.h>
#include <orthon/ops/cpu/scalar_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/slice_ops.h>
#include <orthon/ops/cpu/softmax_ops.h>
#include <orthon/ops/cpu/unary_ops.h>
#include <orthon/ops/cuda/binary_kernels.h>
#include <orthon/ops/cuda/cast_kernels.h>
#include <orthon/ops/cuda/conv2d_kernels.h>
#include <orthon/ops/cuda/cross_entropy_kernels.h>
#include <orthon/ops/cuda/embedding_kernels.h>
#include <orthon/ops/cuda/fusion_kernels.h>
#include <orthon/ops/cuda/matmul_kernels.h>
#include <orthon/ops/cuda/maxpool2d_kernels.h>
#include <orthon/ops/cuda/nll_loss_kernels.h>
#include <orthon/ops/cuda/reduce_kernels.h>
#include <orthon/ops/cuda/softmax_kernels.h>
#include <orthon/ops/cuda/unary_kernels.h>
#include <orthon/core/storage.h>
#include <orthon/core/memory_pool.h>
#include <orthon/core/execution_copy.h>
#include <orthon/memory/cpu_allocator.h>
#include <orthon/memory/cuda_allocator.h>
#include <orthon/memory/pool.h>
#include <orthon/memory/scratch.h>
#include <orthon/runtime/runtime.h>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif
#include <cstring>
#include <type_traits>
using namespace orthon;
struct Storage { std::int64_t sentinel; };
static_assert(!std::is_same<::Storage, orthon::Storage>::value, "Storage must remain namespaced");
TEST(AllocationCompatibility, StorageAndPointerAdapterShareRuntimeAccounting) {
    auto& runtime = default_runtime();
    auto pool = runtime.memory_pool();
    const auto before = pool->stats();
    EXPECT_EQ(allocator_for(Device{}), runtime.cpu()->allocator());
    EXPECT_EQ(get_allocator(Device{}), runtime.cpu()->allocator().get());
    EXPECT_EQ(default_memory_pool(), pool);
    {
        orthon::Storage storage(17, Device{});
        EXPECT_EQ(storage.nbytes(), 17u);
        EXPECT_EQ(storage.device(), Device{});
        ASSERT_NE(storage.data(), nullptr);
        std::memset(storage.data(), 0x5a, storage.nbytes());
        EXPECT_EQ(pool->stats().current_bytes, before.current_bytes + 17);
        void* pointer = memory_pool_allocate(8, Device{});
        ASSERT_NE(pointer, nullptr);
        std::memcpy(pointer, storage.data(), 8);
        EXPECT_EQ(static_cast<unsigned char*>(pointer)[7], 0x5a);
        EXPECT_EQ(pool->stats().current_bytes, before.current_bytes + 25);
        EXPECT_EQ(memory_pool_stats().current_bytes, pool->stats().current_bytes);
        // Release uses the recorded owner, even if the old API's caller passes
        // an unrelated device. It must not initialize that CUDA context.
        memory_pool_deallocate(pointer, Device(DeviceType::CUDA, 7));
        EXPECT_EQ(pool->stats().current_bytes, before.current_bytes + 17);
        unsigned char foreign = 0;
        EXPECT_THROW(memory_pool_deallocate(&foreign, Device{}), std::invalid_argument);
        EXPECT_EQ(pool->stats().current_bytes, before.current_bytes + 17);
    }
    EXPECT_EQ(pool->stats().current_bytes, before.current_bytes);
    EXPECT_EQ(pool->stats().total_allocations, before.total_allocations + 2);
}

TEST(RuntimeContext, CompatibilityFacadesReachDefaultServices) {
    auto& runtime = default_runtime();
    EXPECT_EQ(allocator_for(Device()), runtime.cpu()->allocator());
    EXPECT_EQ(get_allocator(Device()), runtime.cpu()->allocator().get());
    EXPECT_EQ(scratch_arena_for(Device()), runtime.cpu()->scratch());
    EXPECT_EQ(default_memory_pool(), runtime.memory_pool());
}
TEST(CompatibilitySurface, ForwardersAndNativeHelpersResolveCanonicalDeclarations) {
    auto input = ones({2, 3});
    auto historical = kernel_operand(input);
    auto canonical = execution_operand(input);
    EXPECT_EQ(historical.impl(), canonical.impl());
    EXPECT_EQ(add(input, input).shape(), input.shape());
#ifdef WITH_CUDA
    using Binary = Tensor (*)(const Tensor&, const Tensor&);
    Binary add_compatibility = &cuda_add, mul_compatibility = &cuda_mul;
    auto cross_entropy_compatibility = &cuda_cross_entropy;
    EXPECT_NE(add_compatibility, nullptr);
    EXPECT_NE(mul_compatibility, nullptr);
    EXPECT_NE(cross_entropy_compatibility, nullptr);
#endif
}
TEST(CompatibilitySurface, ConvenienceAllocatorsUseCanonicalPoolAndRuntime) {
    auto& runtime = default_runtime();
    auto before = runtime.memory_pool()->stats().current_bytes;
    auto cpu = std::make_shared<CPUAllocator>();
    auto allocation = cpu->allocate(17);
    EXPECT_EQ(runtime.memory_pool()->stats().current_bytes, before + 17);
    EXPECT_EQ(allocation.device(), Device());
    auto cuda = std::make_shared<CUDAAllocator>(Device(DeviceType::CUDA, 0));
    EXPECT_EQ(cuda->device(), Device(DeviceType::CUDA, 0));
    // Construction is lazy and does not require a physical device in CPU builds.
}

TEST(ReductionSubsystem, InitialCudaNativeSignaturesForwardToCanonicalExecution) {
#ifdef WITH_CUDA
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1) GTEST_SKIP()<<"no physical CUDA device";
    const Device device(DeviceType::CUDA,0);
    auto input=full({2,3},2).to(device);
    EXPECT_FLOAT_EQ(*static_cast<float*>(cuda_sum(input,{},0,true).to(Device{}).data()),12);
    EXPECT_FLOAT_EQ(*static_cast<float*>(cuda_mean(input,{2,1},1,false).to(Device{}).data()),2);
    auto result=cuda_max_with_indices(input,{2},1,false);
    EXPECT_EQ(*static_cast<int64_t*>(result.second.to(Device{}).data()),0);
    EXPECT_FALSE(result.first.has_grad_fn()); EXPECT_FALSE(result.second.has_grad_fn());
    auto grad=full({2},3).to(device);
    auto destination=full({2,3},0).to(device);
    cuda_max_scatter_grad(grad,result.second,destination,2,3,1);
    auto host=destination.to(Device{});
    const float* values=static_cast<const float*>(host.data());
    for (int64_t i=0;i<6;++i) EXPECT_FLOAT_EQ(values[i],i==0 || i==3 ? 3 : 0);
    EXPECT_THROW(cuda_sum(input,{3},0,true),std::invalid_argument);
#else
    GTEST_SKIP()<<"CPU-only build";
#endif
}
