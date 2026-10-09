#include <gtest/gtest.h>
#include <orthon/backends/cpu/kernels.h>
#include <orthon/dispatch/dispatcher.h>
#include <orthon/core/execution_copy.h>
#include <orthon/core/float_formats.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/backward_ops.h>
#include <orthon/ops/conv2d_ops.h>
#include <orthon/planning/conv2d_validation.h>
#include <orthon/planning/maxpool2d_validation.h>
#include <orthon/ops/embedding_ops.h>
#include <orthon/ops/nll_loss_ops.h>
#include <orthon/ops/fusion_ops.h>
#include <orthon/ops/softmax_ops.h>
#include <orthon/ops/maxpool2d_ops.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/ops/tensor_access.h>
#include <orthon/runtime/runtime.h>
#include <orthon/autograd/node.h>
#include <orthon/autograd/maxpool2d_node.h>
#include <future>
#include <limits>
#include <cmath>
#ifdef WITH_CUDA
#include <orthon/backends/cuda/kernels.h>
#include <orthon/backends/cuda/embedding_kernels.h>
#include <orthon/backends/cuda/nll_loss_kernels.h>
#include <orthon/backends/cuda/maxpool2d_kernels.h>
#endif
using namespace orthon;
using namespace orthon::dispatch;
namespace cpu = orthon::backend::cpu;
namespace {
void plain(const Tensor& t) { EXPECT_FALSE(t.requires_grad()); EXPECT_FALSE(t.grad_fn()); }
Tensor indices(std::vector<int64_t> values) {
    Tensor t = empty({static_cast<int64_t>(values.size())}, DType::Int64);
    std::copy(values.begin(), values.end(), static_cast<int64_t*>(t.data()));
    return t;
}
}

TEST(ContractsR2, PartialAliasesRejectBeforeMutation) {
    for (auto dtype : {DType::Float32, DType::Float64}) {
        Tensor base = full({4}, 2, dtype);
        Tensor left = base.slice(0, 0, 3), right = base.slice(0, 1, 4);
        auto version = base.impl()->buffer()->version();
        for (float alpha : {0.f, 1.f, -.5f}) {
            EXPECT_THROW(add_(left, right, alpha), std::runtime_error);
            EXPECT_THROW(add_(right, left, alpha), std::runtime_error);
        }
        EXPECT_EQ(base.impl()->buffer()->version(), version);
        auto host = cast(base.to(Device{}), DType::Float32);
        for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(host.data())[i], 2.f);
    }
}

TEST(ContractsR2, ExactAndEmptyAliasesKeepVersionOwnership) {
    Tensor base = full({4}, 2), exact = reshape(base, {4});
    auto version = base.impl()->buffer()->version();
    add_(base, exact, -.5f);
    EXPECT_EQ(base.impl()->buffer()->version(), version + 1);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(base.data())[i], 1.f);
    Tensor endpoint = base.slice(0, 4, 4), start = base.slice(0, 0, 0);
    EXPECT_NO_THROW(add_(endpoint, start, 0));
    EXPECT_EQ(base.impl()->buffer()->version(), version + 2);
}

TEST(ContractsR2, StableLogSoftmaxFiniteShift) {
    Tensor x = full({2, 2}, 1e8);
    auto logs = log_softmax(x, 1), probabilities = softmax(reshape(x, {4}), 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(static_cast<float*>(logs.data())[i], -std::log(2.f), 1e-6);
        EXPECT_FLOAT_EQ(static_cast<float*>(probabilities.data())[i], .25f);
    }
}

TEST(ContractsR2, PoolNonfiniteAndEmptyWindows) {
    auto infinity = std::numeric_limits<float>::infinity();
    Tensor x = full({1, 1, 2, 2}, -infinity);
    auto y = maxpool2d(x, 2, 2, 2, 2);
    EXPECT_EQ(static_cast<float*>(y.data())[0], -infinity);
    auto grad = maxpool2d_backward(x, ones({1, 1, 1, 1}), {2, 2, 0, 0, 2, 2});
    EXPECT_FLOAT_EQ(static_cast<float*>(grad.data())[0], 1.f);
    for (int i = 1; i < 4; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(grad.data())[i], 0.f);
    Tensor empty_input = empty({1, 1, 0, 0});
    EXPECT_EQ(maxpool2d(empty_input, 2, 2, 1, 1, 1, 1).numel(), 1);
    EXPECT_EQ(static_cast<float*>(maxpool2d(empty_input, 2, 2, 1, 1, 1, 1).data())[0], -infinity);
    EXPECT_EQ(maxpool2d_backward(empty_input, ones({1, 1, 1, 1}), {1, 1, 1, 1, 2, 2}).numel(), 0);
}

TEST(PoolingGeometryR51, HistoricalOverflowRejectsEveryCpuEntry) {
    const auto m = (std::numeric_limits<int64_t>::max)();
    Tensor x = ones({1, 1, 2, 2}), grad = ones({1, 1, 1, 1});
    SpatialArgs a{m, m, m / 2, m / 2, m, m};
    const auto version = x.impl()->buffer()->version();
    EXPECT_THROW(maxpool2d(x, m, m, m, m, m / 2, m / 2), std::overflow_error);
    EXPECT_THROW(cpu::cpu_maxpool2d(x, a), std::overflow_error);
    EXPECT_THROW(maxpool2d_backward(x, grad, a), std::overflow_error);
    EXPECT_THROW(cpu::cpu_maxpool2d_backward(x, grad, a), std::overflow_error);
    EXPECT_EQ(x.impl()->buffer()->version(), version);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(x.data())[i], 1.f);
}

TEST(PoolingGeometryR51, AxisLimitsAndFloorAdmission) {
    const auto m = (std::numeric_limits<int64_t>::max)();
    EXPECT_EQ(checked_maxpool2d_extent(m, 1, 1, 0), m);
    EXPECT_EQ(checked_maxpool2d_extent(m, m, m, 0), 1);
    EXPECT_EQ(checked_maxpool2d_extent(1, m, m, m / 2), 1);
    EXPECT_THROW(checked_maxpool2d_extent(2, m, m, m / 2), std::overflow_error);
    EXPECT_THROW(checked_maxpool2d_extent(0, m, 1, m / 2 + 1), std::runtime_error);
    EXPECT_THROW(checked_maxpool2d_extent(1, 2, 2, 0), std::runtime_error);
    EXPECT_THROW(checked_maxpool2d_extent(0, 1, m, 0), std::runtime_error);
    EXPECT_EQ(checked_maxpool2d_extent(2, 2, m, 0), 1);
    EXPECT_EQ(checked_maxpool2d_extent(0, 2, 1, 1), 1);
    EXPECT_THROW(checked_maxpool2d_extent(-1, 1, 1, 0), std::invalid_argument);
    EXPECT_THROW(checked_maxpool2d_extent(2, 0, 1, 0), std::runtime_error);
    EXPECT_THROW(checked_maxpool2d_extent(2, 1, 0, 0), std::runtime_error);
    EXPECT_THROW(checked_maxpool2d_extent(2, 1, 1, -1), std::runtime_error);
}

TEST(PoolingGeometryR51, OutputProductAndBytesRejectWithoutAllocation) {
    const auto m = (std::numeric_limits<int64_t>::max)();
    // Zero input payload; padding-only output would have an impossible product.
    EXPECT_THROW(checked_maxpool2d_geometry({m, m, 0, 0}, {1, 1, 1, 1, 2, 2}), std::overflow_error);
    const auto n = (std::numeric_limits<ptrdiff_t>::max)() / sizeof(float) + 1;
    EXPECT_THROW(checked_maxpool2d_geometry({static_cast<int64_t>(n), 1, 0, 0}, {1, 1, 1, 1, 2, 2}), std::overflow_error);
}

TEST(PoolingGeometryR51, CpuNegativeNumeratorAndBackwardAdmission) {
    Tensor x = ones({1, 1, 1, 1});
    SpatialArgs invalid{2, 2, 0, 0, 2, 2};
    EXPECT_THROW(maxpool2d(x, 2, 2, 2, 2), std::runtime_error);
    EXPECT_THROW(cpu::cpu_maxpool2d(x, invalid), std::runtime_error);
    EXPECT_THROW(cpu::cpu_maxpool2d_backward(x, x, invalid), std::runtime_error);
    x = ones({1, 1, 2, 2});
    SpatialArgs a{2, 2, 0, 0, 2, 2};
    EXPECT_THROW(maxpool2d_backward(x, ones({1, 1, 2, 1}), a), std::invalid_argument);
    EXPECT_THROW(cpu::cpu_maxpool2d_backward(x, ones({1, 1, 1, 1}, DType::Float64), a), std::invalid_argument);
    // A nontrivial transposed output gradient is rejected by direct entries.
    Tensor input = ones({1, 1, 3, 3});
    Tensor transposed = transpose(ones({1, 1, 2, 2}), 2, 3);
    EXPECT_THROW(maxpool2d_backward(input, transposed, {1, 1, 0, 0, 2, 2}), std::invalid_argument);
    EXPECT_THROW(cpu::cpu_maxpool2d_backward(input, transposed, {1, 1, 0, 0, 2, 2}), std::invalid_argument);
}

TEST(PoolingGeometryR51, ZeroSpatialBatchAndChannelRemainValid) {
    for (const auto& shape : {std::vector<int64_t>{1, 1, 0, 0}, {1, 1, 0, 2},
                              {1, 1, 2, 0}, {0, 1, 2, 2}, {1, 0, 2, 2}}) {
        Tensor x = empty(shape);
        auto y = cpu::cpu_maxpool2d(x, {1, 1, 1, 1, 2, 2});
        for (int64_t i = 0; i < y.numel(); ++i)
            EXPECT_EQ(static_cast<float*>(y.data())[i], -std::numeric_limits<float>::infinity());
        auto gi = cpu::cpu_maxpool2d_backward(x, ones(y.shape()), {1, 1, 1, 1, 2, 2});
        EXPECT_EQ(gi.shape(), shape);
        EXPECT_EQ(gi.numel(), 0);
    }
    const auto m = (std::numeric_limits<int64_t>::max)();
    auto y = maxpool2d(empty({1, 1, 0, 1}), m - 1, 1, m, 1, m / 2, 0);
    ASSERT_EQ(y.numel(), 1);
    EXPECT_EQ(static_cast<float*>(y.data())[0], -std::numeric_limits<float>::infinity());
}

TEST(PoolingGeometryR51, CudaMetadataNarrowingAndGridAreChecked) {
    const int64_t i = (std::numeric_limits<int>::max)();
    EXPECT_THROW(checked_maxpool2d_geometry({0, 1, i + 1, 1}, {1, 1, 0, 0, 1, 1}, true), std::overflow_error);
    EXPECT_THROW(checked_maxpool2d_geometry({i + 1, 1, 0, 0}, {1, 1, 1, 1, 2, 2}, true), std::overflow_error);
    EXPECT_THROW(checked_maxpool2d_geometry({0, i + 1, 0, 0}, {1, 1, 1, 1, 2, 2}, true), std::overflow_error);
    EXPECT_THROW(checked_maxpool2d_geometry({1, 1, 2, 2}, {i + 1, 1, 0, 0, 1, 1}, true), std::overflow_error);
    EXPECT_THROW(checked_maxpool2d_geometry({1, 1, 2, 2}, {1, 1, (i + 1) / 2, 0, i + 1, 1}, true), std::overflow_error);
    // Each parameter fits int32, but an unclipped endpoint can exceed it.
    const auto g = checked_maxpool2d_geometry({0, 1, i, 1}, {1, 1, i / 2, 0, i, 1}, true);
    EXPECT_EQ(g.height, i);
    const int64_t base = (g.height - 1) - i / 2;
    EXPECT_GT(base + i, i);  // Requires the widened device intermediate.
    EXPECT_EQ(checked_maxpool2d_cuda_blocks(i * 256), i);
    EXPECT_THROW(checked_maxpool2d_cuda_blocks(i * 256 + 1), std::overflow_error);
    EXPECT_THROW(checked_maxpool2d_geometry({1, 1, 1000000, 1000000}, {1, 1, 0, 0, 1, 1}, true), std::overflow_error);
}

TEST(PoolingGeometryR51, NodeNormalizesStridedUpstreamGradient) {
    Tensor x = empty({1, 1, 3, 3}), grad = empty({1, 1, 2, 2});
    for (int i = 0; i < 9; ++i) static_cast<float*>(x.data())[i] = static_cast<float>(i);
    for (int i = 0; i < 4; ++i) static_cast<float*>(grad.data())[i] = static_cast<float>(i + 1);
    grad = transpose(grad, 2, 3);
    ASSERT_FALSE(grad.is_contiguous());
    MaxPool2DNode node(x, 2, 2, 1, 1, 0, 0);
    auto result = node.backward(grad);
    ASSERT_EQ(result.size(), 1u);
    const float expected[] = {0, 0, 0, 0, 1, 3, 0, 2, 4};
    for (int i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(result[0].data())[i], expected[i]);
}

#ifdef WITH_CUDA
TEST(PoolingGeometryR51, DirectCudaEntriesRejectBeforeContextOrLaunch) {
    // CPU operands prove direct CUDA entries guard their device before execution.
    Tensor x = ones({1, 1, 2, 2}), grad = ones({1, 1, 1, 1});
    EXPECT_THROW(cuda_maxpool2d_forward(x, 2, 2, 2, 2, 0, 0), std::invalid_argument);
    EXPECT_THROW(cuda_maxpool2d_backward(x, grad, 2, 2, 2, 2, 0, 0), std::invalid_argument);
    EXPECT_THROW(cuda_maxpool2d_backward(x, ones({1, 1, 2, 1}), 2, 2, 2, 2, 0, 0), std::invalid_argument);
    Tensor device_input = x.to(Device{DeviceType::CUDA, 0});
    Tensor device_grad = grad.to(Device{DeviceType::CUDA, 0});
    EXPECT_THROW(maxpool2d_backward(device_input, grad, {2, 2, 0, 0, 2, 2}), std::invalid_argument);
    EXPECT_THROW(cuda_maxpool2d_backward(device_input, grad, 2, 2, 2, 2, 0, 0), std::invalid_argument);
    EXPECT_THROW(cpu::cpu_maxpool2d_backward(device_input, device_grad, {2, 2, 0, 0, 2, 2}), std::invalid_argument);
    EXPECT_THROW(cuda_maxpool2d_backward(device_input, ones({1, 1, 2, 1}).to(Device{DeviceType::CUDA, 0}),
                                       2, 2, 2, 2, 0, 0), std::invalid_argument);
}
#endif

TEST(ContractsR2, FusedRootKeepsReleasedNonleafState) {
    Tensor x = ones({1, 1}), weight = full({1, 1}, 2), bias = zeros({1});
    x.requires_grad_(true);
    auto output = fused_linear_relu(x, weight, bias);
    EXPECT_TRUE(output.requires_grad());
    EXPECT_FALSE(output.is_leaf());
    EXPECT_TRUE(output.has_grad_fn());
    output.backward();
    EXPECT_FALSE(output.is_leaf());
    EXPECT_FALSE(output.has_grad_fn());
    EXPECT_THROW(output.backward(), std::runtime_error);
    EXPECT_FALSE(output.has_grad());
    EXPECT_FLOAT_EQ(static_cast<float*>(x.grad().data())[0], 2.f);
}

TEST(BackendsR1, CheckedConvolutionExtentBoundaries) {
    const auto limit = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(checked_conv2d_extent(7, 3, 2, 1), 4);
    EXPECT_EQ(checked_conv2d_extent(limit, 1, limit, 0), 1);
    EXPECT_EQ(checked_conv2d_extent(1, 1, limit, (limit - 1) / 2), 1);
    EXPECT_THROW(checked_conv2d_extent(limit, 1, 1, 1), std::overflow_error);
    EXPECT_THROW(checked_conv2d_extent(1, 1, 1, limit), std::overflow_error);
    EXPECT_THROW(checked_conv2d_extent(1, 2, 2, 0), std::invalid_argument);
    EXPECT_THROW(checked_conv2d_extent(1, 0, 1, 0), std::invalid_argument);
    EXPECT_THROW(checked_conv2d_extent(1, 1, 0, 0), std::invalid_argument);
}

TEST(BackendsR1, ConvolutionBackwardAdmissionPrecedesExecution) {
    Tensor x = ones({1, 1, 3, 4}), w = ones({2, 1, 2, 2}), b = zeros({2});
    Tensor grad = ones({1, 2, 2, 3});
    EXPECT_THROW(conv2d_backward(x, w, b, grad, {0, 1, 0, 0}), std::invalid_argument);
    EXPECT_THROW(conv2d_backward(x, w, b, ones({1, 2, 3, 3}), {1, 1, 0, 0}), std::invalid_argument);
    EXPECT_THROW(cpu::cpu_conv2d_direct(x, w, b, {1, -1, 0, 0}), std::invalid_argument);
    EXPECT_THROW(cpu::cpu_conv2d_im2col(x, w, b, {1, 1, -1, 0}), std::invalid_argument);
    auto gradients = conv2d_backward(x, w, b, grad, {1, 1, 0, 0});
    EXPECT_EQ(gradients[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(static_cast<float*>(gradients[1].data())[0], 6.f);
}

TEST(BackendsR1, CopyRangeRejectsExtremeMetadataBeforeAccess) {
    Tensor dst = full({3}, 0), src = full({2}, 7);
    auto version = dst.impl()->buffer()->version();
    const auto limit = std::numeric_limits<int64_t>::max();
    for (auto args : {CopyArgs{limit, 0, 2}, CopyArgs{0, limit, 2},
                      CopyArgs{-1, 0, 1}, CopyArgs{0, 0, -1}, CopyArgs{0, 0, limit}})
        EXPECT_THROW(copy_range_values(dst, src, args), std::out_of_range);
    copy_range_values(dst, src, {1, 0, 2});
    EXPECT_FLOAT_EQ(static_cast<float*>(dst.data())[1], 7.f);
    EXPECT_EQ(dst.impl()->buffer()->version(), version);
    EXPECT_NO_THROW(copy_range_values(dst, src, {3, 2, 0}));
}

TEST(BackendsR1, EmptyCpuMemoryPrimitivesPreserveLogicalMutation) {
    Tensor t = zeros({2, 0, 3});
    auto version = t.impl()->buffer()->version();
    cpu::cpu_zero_fill(t, 0);
    EXPECT_EQ(t.impl()->buffer()->version(), version);
    fill_(t, 0);
    EXPECT_EQ(t.impl()->buffer()->version(), version + 1);
    EXPECT_NO_THROW(t.to(Device{}));
    Tensor base = ones({2, 3});
    Tensor endpoint = base.slice(0, 2, 2);
    EXPECT_NO_THROW(cpu::cpu_zero_fill(endpoint, 0));
    EXPECT_NO_THROW(endpoint.to(Device{}));
    EXPECT_NO_THROW(cpu::cpu_index_backward(empty({0}), {{2, 0}, DType::Float32, Device{}, 1}));
    auto empty_embedding = cpu::cpu_embedding(indices({0}), empty({1, 0}), {});
    EXPECT_EQ(empty_embedding.numel(), 0);
    EXPECT_NO_THROW(cpu::cpu_embedding_backward(indices({0}), empty({1, 0}), {1, 0}, {}));
    Tensor values = full({3}, 7);
    cpu::cpu_zero_fill(values, 0);
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(values.data())[i], 0.f);
}

#ifdef WITH_CUDA
TEST(BackendsR1, CudaIndexBackwardChecksBeforeGatherOrScatter) {
    const Device device{DeviceType::CUDA, 0};
    Tensor weight = full({3, 2}, 1).to(device), grad = full({1, 2}, 1).to(device);
    Tensor bad = indices({-1}).to(device);
    EXPECT_THROW(cuda_embedding_backward(bad, grad, {3, 2}, {}), std::out_of_range);
    Tensor log_probs = full({1, 3}, -1).to(device), upstream = full({1}, 1).to(device);
    EXPECT_THROW(cuda_nll_loss_backward(log_probs, indices({3}).to(device), upstream, "mean"), std::out_of_range);
    Tensor good = indices({2}).to(device);
    auto accumulated = cuda_embedding_backward(good, grad, {3, 2}, -1).to(Device{});
    for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(static_cast<float*>(accumulated.data())[i], 0.f);
    EXPECT_THROW(embedding_backward(good, ones({1, 2}), {3, 2}, {}), std::runtime_error);
    EXPECT_THROW(cuda_embedding_backward(good, ones({1, 1}).to(device), {3, 2}, {}), std::runtime_error);
}
#endif

TEST(Backends, TrackedViewsProducePlainUnaryAndPhysicalCopies) {
    Tensor t = full({2,3}, 2); t.requires_grad_(true);
    Tensor view = transpose(t,0,1);
    Tensor copy = execution_operand(view);
    plain(copy); EXPECT_NE(copy.impl()->buffer(), t.impl()->buffer());
    EXPECT_EQ(execution_operand(t).impl(), t.impl());
    for (auto fn : {cpu::cpu_relu, cpu::cpu_sigmoid, cpu::cpu_tanh, cpu::cpu_sqrt, cpu::cpu_gelu, cpu::cpu_exp}) {
        Tensor out = fn(view); plain(out); EXPECT_EQ(out.shape(), view.shape());
    }
    plain(cpu::cpu_leaky_relu(view,.2f,0));
    plain(cpu::cpu_cast(view,DType::Float64));
    plain(cpu::cpu_to(t,Device{}));
}
TEST(Backends, SpatialExecutionAndBackwardOwnResultsWithoutGraph) {
    Tensor x=full({1,1,3,3},1), w=full({1,1,2,2},2), b=full({1},3);
    x.requires_grad_(true);w.requires_grad_(true);b.requires_grad_(true);
    SpatialArgs a{1,1,0,0};
    Tensor direct=cpu::cpu_conv2d_direct(x,w,b,a), packed=cpu::cpu_conv2d_im2col(x,w,b,a);
    plain(direct);plain(packed);
    EXPECT_FLOAT_EQ(static_cast<float*>(direct.data())[0],11);
    EXPECT_FLOAT_EQ(static_cast<float*>(packed.data())[0],11);
    auto grads=cpu::cpu_conv2d_backward(x,w,b,full({1,1,2,2},1),a);
    ASSERT_EQ(grads.size(),3u);for(const auto& g:grads)plain(g);
    EXPECT_FLOAT_EQ(static_cast<float*>(grads[2].data())[0],4);
    a.kh=2;a.kw=2;
    plain(cpu::cpu_maxpool2d(x,a));plain(cpu::cpu_maxpool2d_backward(x,full({1,1,2,2},1),a));
}
TEST(Backends, LossAndEmbeddingExecutionPreserveOwnershipAndIndexFailures) {
    Tensor x=full({2,2},1);x.requires_grad_(true);Tensor idx=indices({0,1});
    plain(cpu::cpu_cross_entropy(x,{0,1}));plain(cpu::cpu_cross_entropy_backward(x,{0,1},full({1},1)));
    plain(cpu::cpu_nll_loss(x,idx,"mean"));plain(cpu::cpu_nll_loss_backward(x,idx,full({1},1),"mean"));
    plain(cpu::cpu_embedding(idx,x,{}));plain(cpu::cpu_embedding_backward(idx,x,{2,2},{}));
    for(int i=0;i<32;++i) {
        EXPECT_THROW(cpu::cpu_embedding(indices({3}),x,{}),std::out_of_range);
        EXPECT_THROW(cpu::cpu_nll_loss(x,indices({0,3}),"mean"),std::runtime_error);
    }
    EXPECT_TRUE(x.requires_grad());EXPECT_FALSE(x.grad_fn());
}
TEST(Backends, DimensionAndFusionAdaptersAreGraphFree) {
    Tensor x=full({2,2},1);x.requires_grad_(true);
    plain(cpu::cpu_softmax(x,1));plain(cpu::cpu_log_softmax(x,1));
    plain(cpu::cpu_log_softmax_backward(x,full({2,2},1),1));
    plain(cpu::cpu_fused_linear_relu(x,x,full({2},1)));
    plain(cpu::cpu_sum(x,normalize_reduction(x,ReductionKind::Sum,1,false)).values);
}
TEST(Backends, MutationKernelsLeaveVersionPolicyToSemanticBoundary) {
    Tensor t=full({2},1), b=full({2},2);auto buffer=t.impl()->buffer();const auto version=buffer->version();
    cpu::cpu_fill(t,3);cpu::inplace_mul_(t,2);cpu::inplace_add_(t,b,.5f);
    EXPECT_EQ(buffer->version(),version);EXPECT_FLOAT_EQ(static_cast<float*>(t.data())[0],7);
    fill_(t,4);EXPECT_EQ(buffer->version(),version+1);
    mul_(t,2);EXPECT_EQ(buffer->version(),version+2);
    add_(t,b,1);EXPECT_EQ(buffer->version(),version+3);
}
TEST(Backends, DropoutMaskRetainsLogicalSeedMapping) {
    Tensor t=full({2,3},1);Tensor v=transpose(t,0,1);
    auto a=cpu::cpu_dropout_mask(v,.25f,42),b=cpu::cpu_dropout_mask(execution_operand(v),.25f,42);
    plain(a);plain(b);
    for(int i=0;i<6;++i)EXPECT_FLOAT_EQ(static_cast<float*>(a.data())[i],static_cast<float*>(b.data())[i]);
}
TEST(Backends, ViewScatterAndCopyOwnIndependentPayloads) {
    Tensor g=full({2},3);
    Tensor out=cpu::cpu_index_backward(g,{{2,2},DType::Float32,Device{},1});
    plain(out);EXPECT_FLOAT_EQ(static_cast<float*>(out.data())[2],3);
    Tensor dst=full({4},0);const auto version=dst.impl()->buffer()->version();
    cpu::cpu_copy_range(dst,g,{1,0,2});EXPECT_EQ(dst.impl()->buffer()->version(),version);
    EXPECT_FLOAT_EQ(static_cast<float*>(dst.data())[1],3);
    Tensor empty_grad=empty({2,0});
    auto scattered=cpu::cpu_slice_backward(empty_grad,{{2,3},DType::Float32,Device{},0,1,1,1});
    plain(scattered);for(int i=0;i<6;++i)EXPECT_FLOAT_EQ(static_cast<float*>(scattered.data())[i],0);
}
TEST(Backends, SemanticReluRetainsOriginalSavedViewAndMutationWatch) {
    Tensor base=full({2,3},2);base.requires_grad_(true);Tensor view=transpose(base,0,1);
    Tensor out=relu(view);ASSERT_TRUE(out.grad_fn());
    ASSERT_EQ(out.grad_fn()->inputs.size(),1u);EXPECT_EQ(out.grad_fn()->inputs[0].impl(),view.impl());
    fill_(base,3);EXPECT_THROW(out.backward(),std::runtime_error);
}
TEST(Backends, ExtendedTypedLookupIsCanonicalAndConcurrent) {
    const auto& d=default_dispatcher();auto request=unary_request(OperatorId::LogSoftmax,full({2,2},1));
    const auto* expected=&d.resolve<DimensionKernel>(request,default_runtime());
    EXPECT_THROW(d.resolve<UnaryKernel>(request,default_runtime()),std::logic_error);
    std::vector<std::future<bool>> workers;
    for(int i=0;i<8;++i)workers.push_back(std::async(std::launch::async,[&] {
        for(int n=0;n<100;++n)if(&d.resolve<DimensionKernel>(request,default_runtime())!=expected)return false;
        return true;
    }));
    for(auto& worker:workers)EXPECT_TRUE(worker.get());
}
TEST(Backends, CpuAmpCopyDispatchRoundsAndPreservesOwnerAndVersion) {
    Tensor master = full({4}, 0);
    const float values[] = {1.00048828125f, 1.00390625f, -2.25f, 0.f};
    std::copy(std::begin(values), std::end(values), static_cast<float*>(master.data()));
    for (auto dtype : {DType::Float16, DType::BFloat16, DType::Float32}) {
        Tensor model = empty({4}, dtype); model.requires_grad_(true);
        auto impl = model.impl(); auto buffer = impl->buffer(); auto data = model.data();
        auto version = buffer->version();
        const auto& kernel = default_dispatcher().resolve<CopyKernel>(
            unary_request(OperatorId::AmpCopy, model), default_runtime());
        EXPECT_EQ(kernel.call, cpu::cpu_amp_copy);
        amp_copy_values(model, master);
        EXPECT_EQ(model.impl(), impl); EXPECT_EQ(model.impl()->buffer(), buffer);
        EXPECT_EQ(model.data(), data); EXPECT_EQ(buffer->version(), version + 1);
        EXPECT_TRUE(model.requires_grad()); EXPECT_TRUE(model.is_leaf());
        for (int i = 0; i < 4; ++i) {
            if (dtype == DType::Float32) EXPECT_FLOAT_EQ(static_cast<float*>(data)[i], values[i]);
            else EXPECT_EQ(static_cast<uint16_t*>(data)[i], dtype == DType::Float16
                           ? f32_to_f16_bits(values[i]) : f32_to_bf16_bits(values[i]));
        }
        if (dtype != DType::Float32)
            EXPECT_EQ(static_cast<uint16_t*>(data)[0], dtype == DType::Float16 ? 0x3c00u : 0x3f80u);
    }
}
TEST(Backends, AmpCopyValidationPrecedesMutation) {
    Tensor model = full({2,2}, 1, DType::Float16), master = full({2,2}, 2);
    const auto version = model.impl()->buffer()->version();
    EXPECT_THROW(amp_copy_values(model, full({4}, 3)), std::runtime_error);
    EXPECT_THROW(amp_copy_values(model, full({2,2}, 3, DType::Float64)), std::runtime_error);
    Tensor integers = full({2,2}, 1, DType::Int32);
    EXPECT_THROW(amp_copy_values(integers, master), std::runtime_error);
    Tensor strided = transpose(model, 0, 1);
    EXPECT_THROW(amp_copy_values(strided, master), std::runtime_error);
    EXPECT_THROW(amp_copy_values(model, transpose(master, 0, 1)), std::runtime_error);
    EXPECT_EQ(model.impl()->buffer()->version(), version);
    EXPECT_EQ(static_cast<uint16_t*>(model.data())[0], f32_to_f16_bits(1));
}
TEST(Backends, AmpCopyInvalidatesSavedAliasesAndAcceptsEmptyStorage) {
    Tensor owner = full({2}, 1); owner.requires_grad_(true);
    Tensor view = reshape(owner, {1,2}); Tensor saved = mul(view, view);
    amp_copy_values(owner, full({2}, 3));
    EXPECT_THROW(saved.backward(), std::runtime_error);
    Tensor dst = empty({0}, DType::BFloat16), src = empty({0});
    const auto version = dst.impl()->buffer()->version();
    amp_copy_values(dst, src);
    EXPECT_EQ(dst.impl()->buffer()->version(), version + 1);
}
#ifdef WITH_CUDA
TEST(Backends, PhysicalCudaAmpCopyRetainsRouteAndExactDeviceValidation) {
    Tensor host = full({2}, 1.25f), src = host.to(Device{DeviceType::CUDA, 0});
    for (auto dtype : {DType::Float16, DType::BFloat16, DType::Float32}) {
        Tensor dst = full({2}, 0, dtype).to(src.device());
        auto buffer = dst.impl()->buffer(); auto version = buffer->version();
        const auto& k = default_dispatcher().resolve<CopyKernel>(
            unary_request(OperatorId::AmpCopy, dst), default_runtime());
        EXPECT_EQ(k.call, backend::cuda::cuda_amp_copy_adapter);
        EXPECT_THROW(amp_copy_values(dst, host), std::runtime_error);
        EXPECT_EQ(buffer->version(), version);
        amp_copy_values(dst, src); EXPECT_EQ(buffer->version(), version + 1);
        auto copied = dst.to(Device{});
        if (dtype == DType::Float32) EXPECT_FLOAT_EQ(static_cast<float*>(copied.data())[0], 1.25f);
        else EXPECT_EQ(static_cast<uint16_t*>(copied.data())[0], dtype == DType::Float16
                       ? f32_to_f16_bits(1.25f) : f32_to_bf16_bits(1.25f));
    }
}
TEST(Backends, PhysicalCudaExtendedAdaptersNeverAttachGraph) {
    Tensor host=full({2,2},1), x=host.to(Device{DeviceType::CUDA,0});x.requires_grad_(true);
    auto& runtime=default_runtime();const auto& d=default_dispatcher();
    for(auto id:{OperatorId::Relu,OperatorId::Sigmoid,OperatorId::Tanh,OperatorId::Gelu,OperatorId::Sqrt,OperatorId::Exp}) {
        const auto& k=d.resolve<UnaryKernel>(unary_request(id,x),runtime);plain(k.call(transpose(x,0,1)));
    }
    const auto& soft=d.resolve<DimensionKernel>(unary_request(OperatorId::LogSoftmax,x),runtime);plain(soft.call(x,1));
    plain(backend::cuda::cuda_fused_linear_relu_adapter(x,x,full({2},1).to(x.device())));
    plain(backend::cuda::cuda_cross_entropy_adapter(x,{0,1}));
    plain(backend::cuda::cuda_embedding_adapter(indices({0,1}).to(x.device()),x,{}));
    const auto* expected=&d.resolve<DimensionKernel>(unary_request(OperatorId::LogSoftmax,x),runtime);
    std::vector<std::future<bool>> workers;
    for(int i=0;i<8;++i)workers.push_back(std::async(std::launch::async,[&] {
        for(int n=0;n<100;++n)if(&d.resolve<DimensionKernel>(unary_request(OperatorId::LogSoftmax,x),runtime)!=expected)return false;
        return true;
    }));
    for(auto& worker:workers)EXPECT_TRUE(worker.get());
}
#endif
