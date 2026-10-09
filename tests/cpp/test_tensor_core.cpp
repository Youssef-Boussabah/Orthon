#include <gtest/gtest.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/tensor.h>
#include <orthon/core/shape.h>
#include <orthon/core/strides.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/node.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/shape_ops.h>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace orthon;

TEST(TensorCore, ViewsShareBufferAndHaveDistinctMetadata) {
    Tensor base = full({2, 3}, 2.0);
    Tensor views[] = {base[1], base.slice(1, 1, 3), reshape(base, {6}), transpose(base, 0, 1)};
    for (const Tensor& view : views) {
        EXPECT_EQ(base.impl()->buffer(), view.impl()->buffer());
        EXPECT_NE(&base.impl()->layout(), &view.impl()->layout());
        EXPECT_NE(&base.impl()->autograd_meta(), &view.impl()->autograd_meta());
        EXPECT_NE(&base.impl()->tensor_meta(), &view.impl()->tensor_meta());
    }
    EXPECT_EQ(base.shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(base.strides(), (std::vector<int64_t>{3, 1}));
}

TEST(TensorCore, ViewsStartWithFreshAutogradState) {
    Tensor base = ones({2, 3});
    base.requires_grad_(true);
    base.set_grad(ones({2, 3}));
    int hooks = 0;
    base.add_post_accumulate_grad_hook([&] { ++hooks; });
    Tensor view = [&] { NoGradGuard guard; return transpose(base, 0, 1); }();
    EXPECT_FALSE(view.requires_grad());
    EXPECT_TRUE(view.is_leaf());
    EXPECT_FALSE(view.has_grad());
    EXPECT_FALSE(view.has_grad_fn());
    view.impl()->run_post_accumulate_grad_hooks();
    EXPECT_EQ(hooks, 0);
    view.requires_grad_(true);
    view.set_grad(ones({3, 2}));
    view.zero_grad();
    EXPECT_TRUE(base.has_grad());
    view.set_is_leaf(false);
    EXPECT_TRUE(base.is_leaf());
    base.impl()->run_post_accumulate_grad_hooks();
    EXPECT_EQ(hooks, 1);
}

TEST(TensorCore, FrontendAttachesGraphToFreshViewMetadata) {
    Tensor base = ones({2, 3});
    base.requires_grad_(true);
    Tensor view = base[1];
    EXPECT_TRUE(view.requires_grad());
    EXPECT_FALSE(view.is_leaf());
    EXPECT_TRUE(view.has_grad_fn());
    EXPECT_FALSE(base.has_grad_fn());
    view.backward();
    const float* grad = static_cast<const float*>(base.grad().data());
    for (int i = 0; i < 6; ++i) EXPECT_EQ(grad[i], i < 3 ? 0.f : 1.f);
}

TEST(TensorCore, TensorMetaOwnsLogicalDtype) {
    auto buffer = std::make_shared<Buffer>(16, Device{});
    TensorImpl floats(buffer, Layout({4}), TensorMeta(DType::Float32));
    TensorImpl bytes(buffer, Layout({16}), TensorMeta(DType::UInt8));
    EXPECT_EQ(floats.dtype(), floats.tensor_meta().dtype());
    EXPECT_EQ(bytes.dtype(), bytes.tensor_meta().dtype());
    EXPECT_EQ(floats.dtype(), DType::Float32);
    EXPECT_EQ(bytes.dtype(), DType::UInt8);
    EXPECT_EQ(floats.buffer(), bytes.buffer());
    EXPECT_EQ(buffer->nbytes(), 16u);
    EXPECT_EQ(buffer->device(), Device{});
    EXPECT_EQ(floats.data(), bytes.data());
}

TEST(TensorCore, IndependentAllocationsAndCopiesHaveDifferentBuffers) {
    Tensor base = ones({2, 3});
    Tensor other = ones({2, 3});
    Tensor copy = base.to(base.device());
    Tensor strided = transpose(base, 0, 1);
    Tensor packed = contiguous(strided);
    EXPECT_NE(base.impl()->buffer(), other.impl()->buffer());
    EXPECT_NE(base.impl()->buffer(), copy.impl()->buffer());
    EXPECT_NE(base.impl()->buffer(), packed.impl()->buffer());
    EXPECT_EQ(base.impl(), contiguous(base).impl());
    EXPECT_TRUE(packed.is_contiguous());
}

TEST(TensorCore, AliasGenerationCountsWritesAndNotMetadata) {
    Tensor base = ones({2, 3});
    Tensor view = base.slice(0, 1, 2);
    EXPECT_EQ(base.impl()->buffer()->version(), 0u);
    view.requires_grad_(true);
    view.set_grad(ones({1, 3}));
    EXPECT_EQ(base.impl()->buffer()->version(), 0u);
    fill_(view, 1.f);  // same-value write
    EXPECT_EQ(base.impl()->buffer()->version(), 1u);
    mul_(base, 2.f);
    mul_(base, .5f);  // change and restore
    EXPECT_EQ(view.impl()->buffer()->version(), 3u);
    EXPECT_THROW(add_(base, view, 1.f), std::runtime_error);
    EXPECT_EQ(view.impl()->buffer()->version(), 3u);
    Tensor empty_view = base.slice(0, 2, 2);
    fill_(empty_view, 9.f);
    EXPECT_EQ(base.impl()->buffer()->version(), 4u);
}

TEST(TensorCore, BufferSurvivesBaseDestruction) {
    Tensor view = [] { Tensor base = full({2, 3}, 7.0); return base[1]; }();
    EXPECT_EQ(view.impl()->buffer()->nbytes(), 24u);
    EXPECT_EQ(static_cast<const float*>(view.data())[2], 7.f);
    fill_(view, 8.f);
    EXPECT_EQ(static_cast<const float*>(view.data())[0], 8.f);
}

TEST(TensorCore, SavedVersionRetainsBufferWithoutTensorIdentity) {
    struct WatchNode : Node {
        std::vector<Tensor> backward(const Tensor&) override { return {}; }
    };
    std::weak_ptr<Buffer> weak;
    {
        WatchNode node;
        {
            Tensor value = ones({2});
            weak = value.impl()->buffer();
            node.watch_version(value, "value");
        }
        ASSERT_FALSE(weak.expired());
        EXPECT_NO_THROW(node.validate_saved_versions());
        auto buffer = weak.lock();
        ASSERT_NE(buffer, nullptr);
        buffer->bump_version();
        EXPECT_THROW(node.validate_saved_versions(), std::runtime_error);
    }
    EXPECT_TRUE(weak.expired());
}

TEST(TensorCore, SavedAliasMutationRejectsBeforeLeafGradWrites) {
    Tensor a = full({2}, 2.0), b = full({2}, 3.0);
    a.requires_grad_(true);
    b.requires_grad_(true);
    a.set_grad(full({2}, 7.0));
    Tensor result = mul(a, b);
    Tensor alias = reshape(b, {1, 2});
    fill_(alias, 3.f);
    EXPECT_THROW(result.backward(), std::runtime_error);
    EXPECT_EQ(static_cast<const float*>(a.grad().data())[0], 7.f);
    EXPECT_FALSE(b.has_grad());
}

TEST(TensorCore, DistinctAliasLeavesHaveIndependentGradientBuffers) {
    Tensor base = ones({2, 3});
    Tensor alias = reshape(base, {2, 3});
    base.requires_grad_(true);
    alias.requires_grad_(true);
    add(base, alias).backward();
    EXPECT_EQ(base.impl()->buffer(), alias.impl()->buffer());
    EXPECT_NE(base.grad().impl()->buffer(), alias.grad().impl()->buffer());
    Tensor grad = base.grad();
    mul_(grad, 4.f);
    EXPECT_EQ(static_cast<const float*>(alias.grad().data())[0], 1.f);
}

TEST(TensorCore, MetadataOwnershipCannotBeCopiedIntoAnotherIdentity) {
    EXPECT_FALSE(std::is_copy_constructible<TensorImpl>::value);
    EXPECT_FALSE(std::is_copy_constructible<AutogradMeta>::value);
    EXPECT_FALSE(std::is_copy_constructible<Buffer>::value);
}

TEST(TensorCoreLayout, ScalarRangeIsOneElement) {
    Layout scalar({});
    EXPECT_EQ(scalar.numel(), 1);
    EXPECT_EQ(scalar.ndim(), 0);
    EXPECT_TRUE(scalar.strides().empty());
    EXPECT_TRUE(scalar.is_contiguous());
    EXPECT_EQ(scalar.storage_range(), (std::pair<int64_t, int64_t>{0, 1}));
    EXPECT_NO_THROW(scalar.validate_bounds(4, 4));
    EXPECT_THROW(scalar.validate_bounds(0, 4), std::out_of_range);
}

TEST(TensorCoreLayout, ContiguousRangeAndStrides) {
    Layout layout({2, 3, 4});
    EXPECT_EQ(layout.numel(), 24);
    EXPECT_EQ(layout.ndim(), 3);
    EXPECT_EQ(layout.strides(), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(layout.storage_range(), (std::pair<int64_t, int64_t>{0, 24}));
    EXPECT_EQ(layout.payload_bytes(4), 96u);
    EXPECT_TRUE(layout.is_contiguous());
    EXPECT_NO_THROW(layout.validate_bounds(96, 4));
}

TEST(TensorCoreLayout, StridedRangeIncludesOffsetAndHoles) {
    Layout layout({2, 2}, {3, 1}, 1);
    EXPECT_EQ(layout.numel(), 4);
    EXPECT_EQ(layout.storage_range(), (std::pair<int64_t, int64_t>{1, 6}));
    EXPECT_FALSE(layout.is_contiguous());
    EXPECT_EQ(layout.payload_bytes(4), 16u);
    EXPECT_EQ(layout.offset_bytes(4), 4u);
    EXPECT_NO_THROW(layout.validate_bounds(24, 4));
    EXPECT_THROW(layout.validate_bounds(20, 4), std::out_of_range);
}

TEST(TensorCoreLayout, SingletonStrideDoesNotChangeContiguity) {
    Layout layout({2, 1, 3}, {3, 99, 1});
    EXPECT_TRUE(layout.is_contiguous());
    EXPECT_EQ(layout.storage_range().second, 6);
}

TEST(TensorCoreLayout, EmptyLayoutHasNoAccessedRange) {
    Layout layout({2, 0, 3});
    EXPECT_EQ(layout.numel(), 0);
    EXPECT_EQ(layout.strides(), (std::vector<int64_t>{0, 3, 1}));
    EXPECT_EQ(layout.storage_range(), (std::pair<int64_t, int64_t>{0, 0}));
    EXPECT_EQ(layout.payload_bytes(8), 0u);
    EXPECT_NO_THROW(layout.validate_bounds(0, 8));
    Tensor empty = zeros({2, 0, 3});
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(empty.slice(2, 1, 2).impl()->storage_offset(), 1);
    EXPECT_EQ(empty.slice(2, 1, 2).data(), nullptr);
}

TEST(TensorCoreLayout, EmptyEndpointViewPreservesOffset) {
    Tensor base = ones({2, 3});
    Tensor endpoint = base.slice(0, 2, 2);
    EXPECT_EQ(endpoint.numel(), 0);
    EXPECT_EQ(endpoint.impl()->storage_offset(), 6);
    EXPECT_EQ(endpoint.impl()->buffer(), base.impl()->buffer());
    EXPECT_EQ(endpoint.data(), static_cast<char*>(base.data()) + 24);
    EXPECT_EQ(endpoint.impl()->layout().storage_range(),
              (std::pair<int64_t, int64_t>{6, 6}));
}

TEST(TensorCoreLayout, EmptyStridedEndpointDoesNotFormInvalidPointer) {
    Tensor base = ones({2, 3});
    Tensor endpoint = base.slice(1, 1, 2).slice(0, 2, 2);
    EXPECT_EQ(endpoint.impl()->storage_offset(), 7);
    EXPECT_EQ(endpoint.numel(), 0);
    EXPECT_EQ(endpoint.data(), nullptr);
    EXPECT_NO_THROW(fill_(endpoint, 2.f));
    EXPECT_EQ(base.impl()->buffer()->version(), 1u);
}

TEST(TensorCoreLayout, RejectsShapeProductOverflowWithoutAllocation) {
    const int64_t limit = (std::numeric_limits<int64_t>::max)();
    EXPECT_THROW(Layout({limit, 2}), std::overflow_error);
    EXPECT_THROW(numel({limit, 2}), std::overflow_error);
    EXPECT_THROW(TensorImpl((std::vector<int64_t>{limit, 2}), DType::UInt8, Device{}),
                 std::overflow_error);
}

TEST(TensorCoreLayout, RejectsContiguousStrideOverflowWithoutAllocation) {
    const int64_t limit = (std::numeric_limits<int64_t>::max)();
    EXPECT_THROW(Layout({0, limit, 2}), std::overflow_error);
    EXPECT_THROW(compute_contiguous_strides({0, limit, 2}), std::overflow_error);
}

TEST(TensorCoreLayout, ZeroProductIsIndependentOfAxisOrder) {
    const int64_t limit = (std::numeric_limits<int64_t>::max)();
    EXPECT_EQ(numel({limit, 2, 0}), 0);
    EXPECT_EQ(numel({0, limit, 2}), 0);
    EXPECT_NO_THROW(Layout({limit, 2, 0}));
    EXPECT_EQ(Layout({limit, 2, 0}).numel(), 0);
}

TEST(TensorCoreLayout, RejectsStrideMultiplicationAndSpanAdditionOverflow) {
    const int64_t limit = (std::numeric_limits<int64_t>::max)();
    EXPECT_THROW(Layout({3}, {limit / 2 + 1}), std::overflow_error);
    EXPECT_THROW(Layout({2, 2}, {limit / 2 + 1, limit / 2 + 1}), std::overflow_error);
    EXPECT_THROW(Layout({2}, {1}, limit), std::overflow_error);
}

TEST(TensorCoreLayout, RejectsByteOverflowBeforeAllocation) {
    const int64_t limit = (std::numeric_limits<ptrdiff_t>::max)();
    EXPECT_THROW(Layout({limit}).payload_bytes(8), std::overflow_error);
    EXPECT_THROW(TensorImpl((std::vector<int64_t>{limit}), DType::Float64, Device{}),
                 std::overflow_error);
    EXPECT_THROW(Layout({0}, {1}, limit).offset_bytes(8), std::overflow_error);
}

TEST(TensorCoreLayout, RejectsOutOfBufferViewAndNullBuffer) {
    auto buffer = std::make_shared<Buffer>(16, Device{});
    EXPECT_THROW(TensorImpl(buffer, Layout({5}), TensorMeta(DType::Float32)),
                 std::out_of_range);
    EXPECT_THROW(TensorImpl(buffer, Layout({2}, {3}, 1), TensorMeta(DType::Float32)),
                 std::out_of_range);
    EXPECT_THROW(TensorImpl(nullptr, Layout({0}), TensorMeta(DType::Float32)),
                 std::invalid_argument);
}

TEST(TensorCoreLayout, RejectsInvalidMetadata) {
    EXPECT_THROW(Layout({-1}), std::invalid_argument);
    EXPECT_THROW(Layout({0, -1}), std::invalid_argument);
    EXPECT_THROW(Layout({2}, {}), std::invalid_argument);
    EXPECT_THROW(Layout({2}, {-1}), std::invalid_argument);
    EXPECT_THROW(Layout({0}, {1}, -1), std::invalid_argument);
    EXPECT_THROW(Layout({1}).validate_bounds(4, 0), std::invalid_argument);
}

TEST(TensorCoreLayout, ShiftedOffsetsUseCheckedArithmetic) {
    const int64_t limit = (std::numeric_limits<int64_t>::max)();
    Layout empty({0}, {1}, 2);
    EXPECT_EQ(empty.shifted_offset(3, 4), 14);
    EXPECT_THROW(empty.shifted_offset(limit, 2), std::overflow_error);
    EXPECT_THROW(empty.shifted_offset(limit, 1), std::overflow_error);
}

#ifdef WITH_CUDA
TEST(TensorCoreCUDA, ViewsShareBufferAndCopiesMaterialize) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
        GTEST_SKIP() << "no CUDA device";
    const Device gpu(DeviceType::CUDA, 0);
    Tensor base = full({2, 3}, 7.0).to(gpu);
    Tensor views[] = {base[1], base.slice(1, 1, 3), reshape(base, {6}), transpose(base, 0, 1)};
    for (const Tensor& view : views) {
        EXPECT_EQ(base.impl()->buffer(), view.impl()->buffer());
        EXPECT_NE(&base.impl()->layout(), &view.impl()->layout());
        EXPECT_NE(&base.impl()->autograd_meta(), &view.impl()->autograd_meta());
        EXPECT_EQ(view.device(), gpu);
    }
    Tensor copy = base.to(gpu);
    Tensor packed = contiguous(views[3]);
    EXPECT_NE(base.impl()->buffer(), copy.impl()->buffer());
    EXPECT_NE(base.impl()->buffer(), packed.impl()->buffer());
    EXPECT_EQ(base.impl(), contiguous(base).impl());
    Tensor host = packed.to(Device{});
    for (int64_t i = 0; i < host.numel(); ++i)
        EXPECT_EQ(static_cast<const float*>(host.data())[i], 7.f);
}

TEST(TensorCoreCUDA, AliasMutationSharesGenerationAndInvalidatesSavedValue) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
        GTEST_SKIP() << "no CUDA device";
    Tensor base = full({2, 3}, 2.0).to(Device(DeviceType::CUDA, 0));
    base.requires_grad_(true);
    Tensor result = mul(base, base);
    Tensor alias = base[1];
    mul_(alias, 1.f);  // same-value write on CUDA
    EXPECT_EQ(base.impl()->buffer()->version(), 1u);
    EXPECT_EQ(alias.impl()->buffer()->version(), 1u);
    EXPECT_THROW(result.backward(), std::runtime_error);
    EXPECT_FALSE(base.has_grad());
    Tensor host = alias.to(Device{});
    EXPECT_EQ(static_cast<const float*>(host.data())[0], 2.f);
}
#endif
