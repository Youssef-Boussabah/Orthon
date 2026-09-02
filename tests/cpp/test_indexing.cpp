#include <gtest/gtest.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/cpu/index_ops.h>
#include <orthon/ops/cpu/slice_ops.h>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

// ---- operator[] ----

TEST(Indexing, Index1D) {
    auto t = full({4}, 0.0f);
    auto* p = static_cast<float*>(t.data());
    p[0] = 10.f; p[1] = 20.f; p[2] = 30.f; p[3] = 40.f;

    auto elem = t[2];
    EXPECT_EQ(elem.ndim(), 0);
    EXPECT_EQ(elem.numel(), 1);
    EXPECT_FLOAT_EQ(*static_cast<const float*>(elem.data()), 30.f);
}

TEST(Indexing, Index2D) {
    auto t = full({3, 4}, 0.0f);
    auto* p = static_cast<float*>(t.data());
    for (int i = 0; i < 12; ++i) p[i] = static_cast<float>(i);

    auto row = t[1];
    EXPECT_EQ(row.shape(), (std::vector<int64_t>{4}));
    const auto* rp = static_cast<const float*>(row.data());
    EXPECT_FLOAT_EQ(rp[0], 4.f);
    EXPECT_FLOAT_EQ(rp[3], 7.f);
}

TEST(Indexing, IndexNegative) {
    auto t = full({3, 4}, 0.0f);
    auto* p = static_cast<float*>(t.data());
    for (int i = 0; i < 12; ++i) p[i] = static_cast<float>(i);

    auto row = t[-1];
    const auto* rp = static_cast<const float*>(row.data());
    EXPECT_FLOAT_EQ(rp[0], 8.f);
}

TEST(Indexing, IndexOutOfRange) {
    auto t = full({3}, 1.0f);
    EXPECT_THROW(t[3],  std::out_of_range);
    EXPECT_THROW(t[-4], std::out_of_range);
}

TEST(Indexing, IndexScalarTensorThrows) {
    Tensor scalar(std::make_shared<TensorImpl>(
        std::vector<int64_t>{}, DType::Float32, Device{}));
    EXPECT_THROW(scalar[0], std::runtime_error);
}

TEST(Indexing, IndexSharesStorage) {
    auto t = full({3}, 1.0f);
    auto elem = t[0];
    *static_cast<float*>(elem.data()) = 99.f;
    EXPECT_FLOAT_EQ(*static_cast<const float*>(t.data()), 99.f);
}

// ---- operator[] autograd ----
//
// The Python suite is the authoritative validation for these gates on the
// current machine; these mirror the core contract for C++ callers and for
// platforms where the GoogleTest targets actually build.

TEST(IndexingAutograd, AttachesGraphWhenParentRequiresGrad) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    auto row = t[1];
    EXPECT_TRUE(row.requires_grad());
    EXPECT_TRUE(row.has_grad_fn());
    EXPECT_FALSE(row.is_leaf());
    // Still a view: shape/strides/offset are untouched by the attachment.
    EXPECT_EQ(row.shape(), (std::vector<int64_t>{4}));
    EXPECT_EQ(row.impl()->storage_offset(), 4);
}

TEST(IndexingAutograd, AttachesNothingWhenParentDoesNotRequireGrad) {
    auto t = full({3, 4}, 1.0f);
    auto row = t[1];
    EXPECT_FALSE(row.requires_grad());
    EXPECT_FALSE(row.has_grad_fn());
    EXPECT_TRUE(row.is_leaf());
}

TEST(IndexingAutograd, NoGradModeAttachesNothing) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    Tensor row = [&] {
        NoGradGuard guard;
        return t[1];
    }();
    EXPECT_FALSE(row.requires_grad());
    EXPECT_FALSE(row.has_grad_fn());
    EXPECT_EQ(row.impl()->storage_offset(), 4);
}

TEST(IndexingAutograd, BackwardScattersIntoTheSelectedRow) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    t[1].backward();

    ASSERT_TRUE(t.has_grad());
    auto g = t.grad();
    EXPECT_EQ(g.shape(), (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(g.dtype(), DType::Float32);
    const auto* gp = static_cast<const float*>(g.data());
    for (int64_t i = 0; i < 12; ++i)
        EXPECT_FLOAT_EQ(gp[i], (i >= 4 && i < 8) ? 1.0f : 0.0f);
}

TEST(IndexingAutograd, NegativeIndexScattersIntoTheResolvedRow) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    t[-1].backward();

    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t i = 0; i < 12; ++i)
        EXPECT_FLOAT_EQ(gp[i], (i >= 8) ? 1.0f : 0.0f);
}

TEST(IndexingAutograd, ChainedIndexScattersIntoOneElement) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    t[1][2].backward();

    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t i = 0; i < 12; ++i)
        EXPECT_FLOAT_EQ(gp[i], (i == 6) ? 1.0f : 0.0f);
}

TEST(IndexingAutograd, GradientCarriesNoGraph) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);
    t[0].backward();

    auto g = t.grad();
    EXPECT_FALSE(g.requires_grad());
    EXPECT_FALSE(g.has_grad_fn());
    EXPECT_TRUE(g.is_contiguous());
}

// ---- index_backward internal contract ----
//
// index_backward is deliberately not exposed to Python, so these guards are
// only reachable from C++. Both fire before the CPU/CUDA branch is chosen, so
// they are exercisable in a CPU-only build.

TEST(IndexBackwardContract, RejectsGradientOnADifferentDevice) {
    auto grad = full({4}, 1.0f);                       // lives on CPU
    EXPECT_THROW(index_backward(grad, {3, 4}, DType::Float32,
                                Device(DeviceType::CUDA, 0), 1),
                 std::runtime_error);
}

TEST(IndexBackwardContract, RejectsWrongShapeWithMatchingElementCount) {
    // Parent [3, 2, 4] drops dim 0, so the block is [2, 4]. [4, 2] holds the
    // same 8 elements and the same bytes, but scatters to different places.
    auto grad = full({4, 2}, 1.0f);
    EXPECT_THROW(index_backward(grad, {3, 2, 4}, DType::Float32, Device{}, 1),
                 std::runtime_error);
}

TEST(IndexBackwardContract, RejectsMismatchedDType) {
    auto grad = full({4}, 1.0f);                       // Float32
    EXPECT_THROW(index_backward(grad, {3, 4}, DType::Float64, Device{}, 1),
                 std::runtime_error);
}

TEST(IndexBackwardContract, RejectsOutOfRangeNormalisedIndex) {
    auto grad = full({4}, 1.0f);
    EXPECT_THROW(index_backward(grad, {3, 4}, DType::Float32, Device{}, 3),
                 std::out_of_range);
    EXPECT_THROW(index_backward(grad, {3, 4}, DType::Float32, Device{}, -1),
                 std::out_of_range);
}

TEST(IndexBackwardContract, AcceptsTheExactBlockShape) {
    auto grad = full({2, 4}, 0.0f);
    auto* gp = static_cast<float*>(grad.data());
    for (int i = 0; i < 8; ++i) gp[i] = static_cast<float>(i + 1);

    auto out = index_backward(grad, {3, 2, 4}, DType::Float32, Device{}, 1);
    EXPECT_EQ(out.shape(), (std::vector<int64_t>{3, 2, 4}));

    const auto* op = static_cast<const float*>(out.data());
    for (int64_t i = 0; i < 24; ++i) {
        const float want = (i >= 8 && i < 16) ? static_cast<float>(i - 8 + 1) : 0.0f;
        EXPECT_FLOAT_EQ(op[i], want);
    }
}

// ---- slice ----

TEST(Slice, Slice1D) {
    auto t = full({6}, 0.0f);
    auto* p = static_cast<float*>(t.data());
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i);

    auto s = t.slice(0, 2, 5);
    EXPECT_EQ(s.shape(), (std::vector<int64_t>{3}));
    const auto* sp = static_cast<const float*>(s.data());
    EXPECT_FLOAT_EQ(sp[0], 2.f);
    EXPECT_FLOAT_EQ(sp[2], 4.f);
}

TEST(Slice, SliceColumns2D) {
    auto t = full({3, 6}, 0.0f);
    auto* p = static_cast<float*>(t.data());
    for (int i = 0; i < 18; ++i) p[i] = static_cast<float>(i);

    auto s = t.slice(1, 2, 4);
    EXPECT_EQ(s.shape(), (std::vector<int64_t>{3, 2}));
    auto row0 = s[0];
    const auto* rp = static_cast<const float*>(row0.data());
    EXPECT_FLOAT_EQ(rp[0], 2.f);
    EXPECT_FLOAT_EQ(rp[1], 3.f);
}

TEST(Slice, SliceSharesStorage) {
    auto t = full({4}, 1.0f);
    auto s = t.slice(0, 1, 3);
    *static_cast<float*>(s.data()) = 42.f;
    const auto* tp = static_cast<const float*>(t.data());
    EXPECT_FLOAT_EQ(tp[1], 42.f);
}

TEST(Slice, SliceInvalidRange) {
    auto t = full({4}, 1.0f);
    EXPECT_THROW(t.slice(0, 3, 2), std::out_of_range);
    EXPECT_THROW(t.slice(0, 0, 5), std::out_of_range);
    EXPECT_THROW(t.slice(1, 0, 1), std::out_of_range);
}


// ---- slice autograd ----
//
// The Python suite is the authoritative validation for these gates on the
// current machine; these mirror the core contract for C++ callers and for
// platforms where the GoogleTest targets actually build.

TEST(SliceAutograd, AttachesGraphWhenParentRequiresGrad) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    auto s = t.slice(1, 1, 3);
    EXPECT_TRUE(s.requires_grad());
    EXPECT_TRUE(s.has_grad_fn());
    EXPECT_FALSE(s.is_leaf());
    // Still a view: shape, strides and offset are untouched by the attachment,
    // and the storage is the parent's own.
    EXPECT_EQ(s.shape(),   (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(s.strides(), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(s.impl()->storage_offset(), 1);
    EXPECT_EQ(s.impl()->storage().get(), t.impl()->storage().get());
}

TEST(SliceAutograd, AttachesNothingWhenParentDoesNotRequireGrad) {
    auto s = full({3, 4}, 1.0f).slice(1, 1, 3);
    EXPECT_FALSE(s.requires_grad());
    EXPECT_FALSE(s.has_grad_fn());
    EXPECT_TRUE(s.is_leaf());
}

TEST(SliceAutograd, NoGradModeAttachesNothingButKeepsTheView) {
    auto t = full({3, 4}, 1.0f);
    t.requires_grad_(true);

    Tensor s = [&] {
        NoGradGuard guard;
        return t.slice(1, 1, 3);
    }();
    EXPECT_FALSE(s.requires_grad());
    EXPECT_FALSE(s.has_grad_fn());
    EXPECT_TRUE(s.is_leaf());
    EXPECT_EQ(s.shape(), (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(s.impl()->storage_offset(), 1);
    EXPECT_EQ(s.impl()->storage().get(), t.impl()->storage().get());
}

TEST(SliceAutograd, TrackedSliceStillSharesStorage) {
    auto t = full({4}, 1.0f);
    t.requires_grad_(true);
    auto s = t.slice(0, 1, 3);
    *static_cast<float*>(s.data()) = 42.f;
    EXPECT_FLOAT_EQ(static_cast<const float*>(t.data())[1], 42.f);
}

TEST(SliceAutograd, LeadingDimensionBackward) {
    auto t = full({4, 3}, 1.0f);
    t.requires_grad_(true);

    t.slice(0, 1, 3).backward();

    ASSERT_TRUE(t.has_grad());
    auto g = t.grad();
    EXPECT_EQ(g.shape(), (std::vector<int64_t>{4, 3}));
    const auto* gp = static_cast<const float*>(g.data());
    for (int64_t i = 0; i < 12; ++i)
        EXPECT_FLOAT_EQ(gp[i], (i >= 3 && i < 9) ? 1.0f : 0.0f);
}

TEST(SliceAutograd, InnerDimensionBackwardScattersEveryPrefixRun) {
    // Parent [2, 5, 3] sliced on dim 1 over [1, 4): prefix = 2, suffix = 3,
    // slice_len = 3. Two runs of 9 elements land 15 elements apart, so a single
    // flat 18-element memcpy would put the second run at offset 9, not 18.
    auto t = full({2, 5, 3}, 1.0f);
    t.requires_grad_(true);

    t.slice(1, 1, 4).backward();

    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t p = 0; p < 2; ++p) {
        for (int64_t r = 0; r < 5; ++r) {
            for (int64_t c = 0; c < 3; ++c) {
                const float want = (r >= 1 && r < 4) ? 1.0f : 0.0f;
                EXPECT_FLOAT_EQ(gp[p * 15 + r * 3 + c], want)
                    << "prefix " << p << " row " << r << " col " << c;
            }
        }
    }
}

TEST(SliceAutograd, LastDimensionBackward) {
    auto t = full({2, 3, 6}, 1.0f);
    t.requires_grad_(true);

    t.slice(2, 2, 5).backward();

    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t i = 0; i < 36; ++i) {
        const int64_t c = i % 6;
        EXPECT_FLOAT_EQ(gp[i], (c >= 2 && c < 5) ? 1.0f : 0.0f);
    }
}

TEST(SliceAutograd, NegativeDimAndBoundsAreNormalisedOnce) {
    // [3, 5] with slice(-1, -4, -1) resolves to dim 1 over [1, 4).
    auto t = full({3, 5}, 1.0f);
    t.requires_grad_(true);

    auto s = t.slice(-1, -4, -1);
    EXPECT_EQ(s.shape(), (std::vector<int64_t>{3, 3}));
    EXPECT_EQ(s.impl()->storage_offset(), 1);

    s.backward();
    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t i = 0; i < 15; ++i) {
        const int64_t c = i % 5;
        EXPECT_FLOAT_EQ(gp[i], (c >= 1 && c < 4) ? 1.0f : 0.0f);
    }
}

TEST(SliceAutograd, OverlappingSlicesAccumulate) {
    auto t = full({4, 3}, 1.0f);
    t.requires_grad_(true);

    t.slice(0, 0, 3).backward();   // rows 0..2
    t.slice(0, 1, 4).backward();   // rows 1..3, accumulating into the overlap

    const auto* gp = static_cast<const float*>(t.grad().data());
    const float want[4] = {1.0f, 2.0f, 2.0f, 1.0f};
    for (int64_t r = 0; r < 4; ++r)
        for (int64_t c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(gp[r * 3 + c], want[r]);
}

TEST(SliceAutograd, ChainedSlicesComposeThroughTwoNodes) {
    auto t = full({2, 6, 3}, 1.0f);
    t.requires_grad_(true);

    auto mid  = t.slice(1, 1, 5);
    auto leaf = mid.slice(1, 1, 3);
    EXPECT_TRUE(mid.has_grad_fn());
    EXPECT_TRUE(leaf.has_grad_fn());
    EXPECT_EQ(leaf.shape(), (std::vector<int64_t>{2, 2, 3}));

    leaf.backward();
    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t p = 0; p < 2; ++p)
        for (int64_t r = 0; r < 6; ++r)
            for (int64_t c = 0; c < 3; ++c)
                EXPECT_FLOAT_EQ(gp[p * 18 + r * 3 + c],
                                (r >= 2 && r < 4) ? 1.0f : 0.0f);
}

TEST(SliceAutograd, SliceOfATransposedParent) {
    auto t = full({4, 3}, 1.0f);
    t.requires_grad_(true);

    auto tr = transpose(t, 0, 1);          // [3, 4], strided
    ASSERT_FALSE(tr.is_contiguous());
    tr.slice(0, 1, 3).backward();          // columns 1..2 of the base

    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t r = 0; r < 4; ++r)
        for (int64_t c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(gp[r * 3 + c], (c >= 1 && c < 3) ? 1.0f : 0.0f);
}

TEST(SliceAutograd, EmptySliceBackwardIsAZeroParentGradient) {
    auto t = full({2, 5, 3}, 1.0f);
    t.requires_grad_(true);

    auto s = t.slice(1, 2, 2);
    EXPECT_EQ(s.shape(), (std::vector<int64_t>{2, 0, 3}));
    EXPECT_EQ(s.numel(), 0);

    s.backward();
    ASSERT_TRUE(t.has_grad());
    EXPECT_EQ(t.grad().shape(), (std::vector<int64_t>{2, 5, 3}));
    const auto* gp = static_cast<const float*>(t.grad().data());
    for (int64_t i = 0; i < 30; ++i) EXPECT_FLOAT_EQ(gp[i], 0.0f);
}

TEST(SliceAutograd, GradientCarriesNoGraph) {
    auto t = full({2, 5, 3}, 1.0f);
    t.requires_grad_(true);
    t.slice(1, 1, 4).backward();

    auto g = t.grad();
    EXPECT_FALSE(g.requires_grad());
    EXPECT_FALSE(g.has_grad_fn());
    EXPECT_TRUE(g.is_leaf());
    EXPECT_TRUE(g.is_contiguous());
}

// ---- slice_backward internal contract ----
//
// slice_backward is deliberately not exposed to Python, so these guards are
// only reachable from C++. Every one of them fires before the CPU/CUDA branch
// is chosen, so they are exercisable in a CPU-only build.

TEST(SliceBackwardContract, RejectsGradientOnADifferentDevice) {
    auto grad = full({3, 2}, 1.0f);                    // lives on CPU
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float32,
                                Device(DeviceType::CUDA, 0), 1, 1, 3),
                 std::runtime_error);
}

TEST(SliceBackwardContract, RejectsMismatchedDType) {
    auto grad = full({3, 2}, 1.0f);                    // Float32
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float64, Device{}, 1, 1, 3),
                 std::runtime_error);
}

TEST(SliceBackwardContract, RejectsWrongShapeWithMatchingElementCount) {
    // Parent [2, 5, 3] sliced on dim 1 over [1, 4) expects [2, 3, 3]. A
    // [2, 5, 2] gradient (dim 2 over [1, 3)) holds 18 elements too, but
    // scatters entirely differently.
    auto grad = full({2, 5, 2}, 1.0f);
    EXPECT_THROW(slice_backward(grad, {2, 5, 3}, DType::Float32, Device{}, 1, 1, 4),
                 std::runtime_error);
}

TEST(SliceBackwardContract, RejectsUnnormalisedDim) {
    auto grad = full({3, 2}, 1.0f);
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float32, Device{}, -1, 1, 3),
                 std::out_of_range);
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float32, Device{}, 2, 1, 3),
                 std::out_of_range);
}

TEST(SliceBackwardContract, RejectsOutOfRangeNormalisedBounds) {
    auto grad = full({3, 2}, 1.0f);
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float32, Device{}, 1, -1, 2),
                 std::out_of_range);
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float32, Device{}, 1, 3, 2),
                 std::out_of_range);
    EXPECT_THROW(slice_backward(grad, {3, 4}, DType::Float32, Device{}, 1, 2, 5),
                 std::out_of_range);
}

TEST(SliceBackwardContract, RejectsAScalarParent) {
    auto grad = full({1}, 1.0f);
    EXPECT_THROW(slice_backward(grad, {}, DType::Float32, Device{}, 0, 0, 1),
                 std::runtime_error);
}

TEST(SliceBackwardContract, MaterialisesAStridedGradient) {
    // Parent [2, 5, 3] sliced on dim 1 over [1, 4). The gradient arrives as a
    // transposed [2, 3, 3] view, so reading it in storage order without
    // materialising first would scatter the wrong values.
    auto packed = full({2, 3, 3}, 0.0f);
    auto* pp = static_cast<float*>(packed.data());
    for (int i = 0; i < 18; ++i) pp[i] = static_cast<float>(i + 1);
    auto strided = transpose(packed, 1, 2);
    ASSERT_FALSE(strided.is_contiguous());

    auto out = slice_backward(strided, {2, 5, 3}, DType::Float32, Device{}, 1, 1, 4);
    EXPECT_EQ(out.shape(), (std::vector<int64_t>{2, 5, 3}));

    const auto* op = static_cast<const float*>(out.data());
    for (int64_t p = 0; p < 2; ++p) {
        for (int64_t r = 0; r < 3; ++r) {
            for (int64_t c = 0; c < 3; ++c) {
                // strided[p][r][c] == packed[p][c][r]
                const float want = static_cast<float>(p * 9 + c * 3 + r + 1);
                EXPECT_FLOAT_EQ(op[p * 15 + (r + 1) * 3 + c], want);
            }
        }
    }
    for (int64_t p = 0; p < 2; ++p) {
        for (int64_t c = 0; c < 3; ++c) {
            EXPECT_FLOAT_EQ(op[p * 15 + 0 * 3 + c], 0.0f);
            EXPECT_FLOAT_EQ(op[p * 15 + 4 * 3 + c], 0.0f);
        }
    }
}

TEST(SliceBackwardContract, EmptySliceReturnsAZeroedParentGradient) {
    auto grad = full({2, 0, 3}, 1.0f);
    auto out = slice_backward(grad, {2, 5, 3}, DType::Float32, Device{}, 1, 2, 2);
    EXPECT_EQ(out.shape(), (std::vector<int64_t>{2, 5, 3}));
    const auto* op = static_cast<const float*>(out.data());
    for (int64_t i = 0; i < 30; ++i) EXPECT_FLOAT_EQ(op[i], 0.0f);
}

TEST(SliceBackwardContract, EmptyParentReturnsAnEmptyGradient) {
    auto grad = full({2, 0, 1}, 1.0f);
    auto out = slice_backward(grad, {2, 0, 3}, DType::Float32, Device{}, 2, 1, 2);
    EXPECT_EQ(out.shape(), (std::vector<int64_t>{2, 0, 3}));
    EXPECT_EQ(out.numel(), 0);
}
// ---- broadcasting ----

TEST(Broadcasting, ScalarBroadcast) {
    auto a = full({1}, 3.0f);
    auto b = full({4}, 2.0f);
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{4}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(p[i], 5.0f);
}

TEST(Broadcasting, BroadcastRows) {
    auto a = full({1, 4}, 1.0f);
    auto b = full({3, 4}, 2.0f);
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{3, 4}));
    EXPECT_FLOAT_EQ(*static_cast<const float*>(c.data()), 3.0f);
}

TEST(Broadcasting, BroadcastBothDims) {
    auto a = full({3, 1}, 10.0f);
    auto b = full({1, 4}, 5.0f);
    auto c = mul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{3, 4}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < 12; ++i)
        EXPECT_FLOAT_EQ(p[i], 50.0f);
}

TEST(Broadcasting, SameShapeStillWorks) {
    auto a = full({2, 3}, 2.0f);
    auto b = full({2, 3}, 3.0f);
    auto c = mul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_FLOAT_EQ(*static_cast<const float*>(c.data()), 6.0f);
}

// ---- .to() ----

TEST(ToDevice, CPUToCPUIsNoop) {
    auto t = full({3}, 1.0f);
    auto t2 = t.to(Device{});
    EXPECT_FLOAT_EQ(*static_cast<const float*>(t2.data()), 1.0f);
}

// Cross-device transfer is asserted per build configuration: a CPU-only
// build rejects a CUDA target, while a CUDA build with a device present must
// succeed. One unconditional expectation would encode the wrong contract for
// whichever configuration it was not written for.

#ifndef WITH_CUDA

TEST(ToDevice, CPUOnlyBuildRejectsCudaTarget) {
    auto t = full({3}, 1.0f);
    EXPECT_THROW(t.to(Device(DeviceType::CUDA, 0)), std::runtime_error);
}

#else

TEST(ToDevice, CPUToCUDARoundTripsValues) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
        GTEST_SKIP() << "no CUDA device";

    auto t = full({3}, 1.0f);
    auto d = t.to(Device(DeviceType::CUDA, 0));
    EXPECT_TRUE(d.device().is_cuda());
    EXPECT_EQ(d.device().index, 0);

    auto back = d.to(Device{});
    EXPECT_EQ(back.shape(), t.shape());
    const auto* p = static_cast<const float*>(back.data());
    for (int64_t i = 0; i < back.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 1.0f);
}

#endif  // WITH_CUDA
