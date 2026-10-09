#include <orthon/backends/cpu/kernels.h>
#include <orthon/reduction/reduction.h>
#include <orthon/reduction/extrema.h>
#include <orthon/autograd/max_node.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/runtime/runtime.h>
#include <orthon/memory/scratch.h>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#ifdef WITH_CUDA
#include <orthon/backends/cuda/reduce_kernels.h>
#include <cuda_runtime_api.h>
#endif

using namespace orthon;
namespace {
ReductionPlan plan(std::vector<int64_t> shape, std::optional<int64_t> dim = {}, bool keep = false) {
    return normalize_reduction(shape, Device{}, DType::Float32, ReductionKind::Sum, dim, keep);
}
}
TEST(ReductionSubsystem, GlobalShapeAndGeometry) {
    auto p = plan({2,3,4});
    EXPECT_TRUE(p.reduce_all); EXPECT_FALSE(p.dim);
    EXPECT_TRUE(p.output_shape.empty());
    EXPECT_EQ(p.outer,1); EXPECT_EQ(p.reduce,24); EXPECT_EQ(p.inner,1);
    EXPECT_EQ(p.input_numel,24); EXPECT_EQ(p.output_numel,1);
}
TEST(ReductionSubsystem, GlobalKeepdim) {
    EXPECT_EQ(plan({2,3,4},{},true).output_shape, (std::vector<int64_t>{1,1,1}));
}
TEST(ReductionSubsystem, DimensionShapeAndGeometry) {
    auto p = plan({2,3,4},1);
    EXPECT_EQ(p.output_shape,(std::vector<int64_t>{2,4}));
    EXPECT_EQ(p.broadcast_shape,(std::vector<int64_t>{2,1,4}));
    EXPECT_EQ(p.outer,2); EXPECT_EQ(p.reduce,3); EXPECT_EQ(p.inner,4);
    EXPECT_EQ(p.outer*p.reduce*p.inner,p.input_numel);
    EXPECT_EQ(p.outer*p.inner,p.output_numel);
}
TEST(ReductionSubsystem, NegativeDimensionNormalizesOnce) {
    auto p = plan({2,3,4},-2,true);
    EXPECT_EQ(p.dim,1); EXPECT_EQ(p.output_shape,(std::vector<int64_t>{2,1,4}));
}
TEST(ReductionSubsystem, ScalarGlobal) {
    for (bool keep : {false,true}) {
        auto p = plan({},{},keep);
        EXPECT_TRUE(p.output_shape.empty()); EXPECT_EQ(p.reduce,1);
    }
}
TEST(ReductionSubsystem, ScalarAxisRejection) {
    EXPECT_THROW(plan({},0), std::out_of_range);
    EXPECT_THROW(plan({},-1), std::out_of_range);
}
TEST(ReductionSubsystem, AxisBounds) {
    EXPECT_THROW(plan({2,3},2),std::out_of_range);
    EXPECT_THROW(plan({2,3},-3),std::out_of_range);
}
TEST(ReductionSubsystem, EmptyAndNegativeShapes) {
    EXPECT_THROW(plan({2,0,3},100),std::runtime_error);
    EXPECT_THROW(plan({2,-1}),std::invalid_argument);
}
TEST(ReductionSubsystem, CheckedMultiplication) {
    const auto limit=(std::numeric_limits<int64_t>::max)();
    EXPECT_EQ(plan({limit}).input_numel,limit);
    EXPECT_THROW(plan({limit,2}),std::overflow_error);
    EXPECT_THROW(plan({limit,2},1),std::overflow_error);
}
TEST(ReductionSubsystem, KindDtypeExactDeviceWithoutAllocation) {
    auto p=normalize_reduction({2,3},Device(DeviceType::CUDA,7),DType::BFloat16,ReductionKind::ArgMax,-1,true);
    EXPECT_EQ(p.kind,ReductionKind::ArgMax); EXPECT_EQ(p.device,Device(DeviceType::CUDA,7));
    EXPECT_EQ(p.dtype,DType::BFloat16);
    EXPECT_EQ(p.extrema_policy,ExtremaPolicy::FirstLogicalMaximumFirstNaN);
}
TEST(ReductionSubsystem, ImmutableValueOnly) {
    static_assert(std::is_copy_constructible_v<ReductionPlan>);
    static_assert(!std::is_copy_assignable_v<ReductionPlan>);
    static_assert(!std::is_default_constructible_v<ReductionPlan>);
    static_assert(std::is_const_v<decltype(ReductionPlan::outer)>);
    auto p=plan({2,3},1); auto copy=p;
    EXPECT_EQ(copy.input_shape,p.input_shape); EXPECT_EQ(copy.output_numel,p.output_numel);
}
TEST(ReductionSubsystem, OperandMismatchRejected) {
    auto t=backend::cpu::full({2,3},2);
    EXPECT_NO_THROW(validate_reduction_operand(t,plan({2,3}),ReductionKind::Sum));
    EXPECT_THROW(backend::cpu::cpu_sum(t,plan({3,2})),std::invalid_argument);
    EXPECT_THROW(backend::cpu::cpu_mean(t,plan({2,3})),std::invalid_argument);
    auto foreign=normalize_reduction({2,3},Device(DeviceType::CUDA,7),DType::Float32,ReductionKind::Sum);
    EXPECT_THROW(backend::cpu::cpu_sum(t,foreign),std::invalid_argument);
}
TEST(ReductionSubsystem, ExtremaPaddingNaNTieAndSignedZero) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    EXPECT_TRUE(extrema_candidate_wins(-inf,0,-inf,-1));
    EXPECT_FALSE(extrema_candidate_wins(inf,-1,0.,0));
    EXPECT_TRUE(extrema_candidate_wins(nan,3,inf,0));
    EXPECT_FALSE(extrema_candidate_wins(inf,0,nan,3));
    EXPECT_TRUE(extrema_candidate_wins(nan,2,nan,3));
    EXPECT_TRUE(extrema_candidate_wins(-0.,1,+0.,2));
    EXPECT_FALSE(extrema_candidate_wins(7.,8,7.,2));
}
TEST(ReductionSubsystem, ScatterMismatchRejected) {
    auto input=backend::cpu::full({2,3},2);
    auto p=normalize_reduction(input,ReductionKind::Max,1);
    auto result=backend::cpu::cpu_max(input,p);
    auto grad=backend::cpu::full({2},3);
    EXPECT_NO_THROW(backend::cpu::cpu_max_backward(input,grad,*result.indices,p));
    EXPECT_THROW(backend::cpu::cpu_max_backward(input,backend::cpu::full({1},3),*result.indices,p),std::invalid_argument);
}
TEST(ReductionSubsystem, SavedIndexMutationIsWatchedGlobalAndDimension) {
    std::vector<Device> devices{Device{}};
#ifdef WITH_CUDA
    int count=0;
    if (cudaGetDeviceCount(&count)==cudaSuccess && count>0)
        devices.emplace_back(DeviceType::CUDA,0);
#endif
    for (auto device : devices) {
        for (auto dim : {std::optional<int64_t>{}, std::optional<int64_t>{1}}) {
            auto input=backend::cpu::full({2,3},2).to(device);
            input.requires_grad_(true);
            auto output=reduce_max(input,dim);
            auto node=std::dynamic_pointer_cast<MaxNode>(output.grad_fn());
            ASSERT_NE(node,nullptr);
            // A real saved-payload write must be caught before scatter reads it.
            if (device.is_cpu()) *static_cast<int64_t*>(node->argmax_indices_.data())=99;
#ifdef WITH_CUDA
            else default_runtime().cuda(node->argmax_indices_.device().index)->memset_async(node->argmax_indices_.data(),0x7f,sizeof(int64_t));
#endif
            node->argmax_indices_.impl()->buffer()->bump_version();
            EXPECT_THROW(output.backward(),std::runtime_error);
        }
    }
}
TEST(ReductionSubsystem, SequentialCudaWorkspaceReleasesAndReuses) {
#ifdef WITH_CUDA
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1) GTEST_SKIP()<<"no physical CUDA device";
    const Device device(DeviceType::CUDA,0);
    auto input=backend::cpu::full({262145},1).to(device);
    auto arena=default_runtime().scratch(device);
    size_t reserved=0;
    for (int round=0;round<2;++round) {
        for (auto kind : {ReductionKind::Sum,ReductionKind::Mean,ReductionKind::Max,ReductionKind::ArgMax}) {
            Tensor output = kind==ReductionKind::Sum ? sum(input) : kind==ReductionKind::Mean ? mean(input)
                : kind==ReductionKind::Max ? reduce_max(input) : argmax(input);
            auto host=output.to(Device{});
            if (kind==ReductionKind::ArgMax) EXPECT_EQ(*static_cast<int64_t*>(host.data()),0);
            else EXPECT_FLOAT_EQ(*static_cast<float*>(host.data()),kind==ReductionKind::Sum ? 262145 : 1);
            auto stats=arena->stats();
            EXPECT_EQ(stats.used_bytes,0u);
            const size_t demand=kind==ReductionKind::Sum || kind==ReductionKind::Mean ? 1024*sizeof(float)
                : 1024*(sizeof(float)+sizeof(int64_t));
            EXPECT_GE(stats.peak_used_bytes,demand);
            EXPECT_GE(stats.reserved_bytes,demand);
            if (round) EXPECT_EQ(stats.reserved_bytes,reserved);
        }
        reserved=arena->stats().reserved_bytes;
    }
#else
    GTEST_SKIP()<<"CPU-only build";
#endif
}
