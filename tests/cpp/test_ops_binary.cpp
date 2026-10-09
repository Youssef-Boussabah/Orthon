#include <gtest/gtest.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/core/tensor_impl.h>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

TEST(BinaryOps, AddCorrectness) {
    auto a = full({4}, 2.0f);
    auto b = full({4}, 3.0f);
    auto c = add(a, b);
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 5.0f);
}

TEST(BinaryOps, MulCorrectness) {
    auto a = full({4}, 2.0f);
    auto b = full({4}, 3.0f);
    auto c = mul(a, b);
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 6.0f);
}

TEST(BinaryOps, AddDoesNotMutateInputs) {
    auto a = full({3}, 1.0f);
    auto b = full({3}, 2.0f);
    add(a, b);
    EXPECT_FLOAT_EQ(static_cast<const float*>(a.data())[0], 1.0f);
    EXPECT_FLOAT_EQ(static_cast<const float*>(b.data())[0], 2.0f);
}

TEST(BinaryOps, ShapeMismatch) {
    auto a = zeros({2, 3});
    auto b = zeros({3, 2});
    EXPECT_THROW(add(a, b), std::runtime_error);
    EXPECT_THROW(mul(a, b), std::runtime_error);
}

TEST(BinaryOps, DtypeMismatch) {
    Tensor a(std::make_shared<TensorImpl>(std::vector<int64_t>{3}, DType::Float32, Device{}));
    Tensor b(std::make_shared<TensorImpl>(std::vector<int64_t>{3}, DType::Int32,   Device{}));
    EXPECT_THROW(add(a, b), std::runtime_error);
}

TEST(BinaryOps, Add2D) {
    auto a = full({2, 3}, 1.0f);
    auto b = full({2, 3}, 1.0f);
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 2.0f);
}

// [2,3] + [3]  →  right-aligned broadcast, output [2,3]
TEST(BinaryOps, BroadcastAdd_2x3_plus_3) {
    auto a = full({2, 3}, 10.0f);
    auto b = full({3},    2.0f);
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 12.0f);
}

TEST(BinaryOps, BroadcastMul_2x3_times_3) {
    auto a = full({2, 3}, 3.0f);
    auto b = full({3},    4.0f);
    auto c = mul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 12.0f);
}

// [2,3] + [1,3]  →  broadcast along dim 0, output [2,3]
TEST(BinaryOps, BroadcastAdd_2x3_plus_1x3) {
    auto a = full({2, 3}, 5.0f);
    auto b = full({1, 3}, 1.0f);
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 6.0f);
}

TEST(BinaryOps, BroadcastMul_2x3_times_1x3) {
    auto a = full({2, 3}, 5.0f);
    auto b = full({1, 3}, 2.0f);
    auto c = mul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 10.0f);
}

// [2,3] + [2,1]  →  broadcast along dim 1, output [2,3]
TEST(BinaryOps, BroadcastAdd_2x3_plus_2x1) {
    auto a = full({2, 3}, 7.0f);
    auto b = full({2, 1}, 3.0f);
    auto c = add(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 10.0f);
}

TEST(BinaryOps, BroadcastMul_2x3_times_2x1) {
    auto a = full({2, 3}, 4.0f);
    auto b = full({2, 1}, 3.0f);
    auto c = mul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    const auto* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 12.0f);
}

// Incompatible shapes must throw.
TEST(BinaryOps, BroadcastMismatchThrows) {
    auto a = full({2, 3}, 1.0f);
    auto b = full({4},    1.0f);   // 3 vs 4 is not broadcastable
    EXPECT_THROW(add(a, b), std::runtime_error);
    EXPECT_THROW(mul(a, b), std::runtime_error);
}

// Leaf gradient ownership
//
// AddNode returns the same gradient object for both of its inputs, because
// both partial derivatives are the identity. That is correct arithmetic, but
// two distinct leaves must still end up owning two distinct gradient buffers:
// .grad is public mutable state (DDP rescales it, AMP unscales it), so a
// shared buffer turns `a.grad *= k` into a silent edit of b.grad.
TEST(BinaryOps, AddBackwardGivesEachLeafItsOwnGradStorage) {
    auto a = ones({2, 3});
    auto b = ones({2, 3});
    a.requires_grad_(true);
    b.requires_grad_(true);

    sum(add(a, b)).backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());

    // Distinct TensorImpls AND distinct Buffers: sharing the buffer is what
    // makes an in-place edit visible through the other leaf, so the buffer is
    // the thing that has to differ.
    EXPECT_NE(a.grad().impl().get(), b.grad().impl().get());
    EXPECT_NE(a.grad().impl()->buffer().get(),
              b.grad().impl()->buffer().get());

    // Values are d(sum(a+b))/da = d(.)/db = 1 everywhere.
    const auto* ga = static_cast<const float*>(a.grad().data());
    const auto* gb = static_cast<const float*>(b.grad().data());
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(ga[i], 1.0f);
        EXPECT_FLOAT_EQ(gb[i], 1.0f);
    }
}

// The behavioural half of the invariant: mutate one leaf's gradient in place
// and the other leaf must not move. Checked in both directions, because a
// one-directional check passes even if the copy went to the wrong leaf.
TEST(BinaryOps, MutatingOneLeafGradLeavesTheOtherAlone) {
    auto a = ones({2, 3});
    auto b = ones({2, 3});
    a.requires_grad_(true);
    b.requires_grad_(true);
    sum(add(a, b)).backward();

    auto ga = a.grad();
    mul_(ga, 2.0f);
    const auto* pb = static_cast<const float*>(b.grad().data());
    for (int64_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(pb[i], 1.0f);

    auto gb = b.grad();
    mul_(gb, 5.0f);
    const auto* pa = static_cast<const float*>(a.grad().data());
    for (int64_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(pa[i], 2.0f);
}

// A second backward accumulates into the leaf's own buffer and must keep the
// two leaves independent - the accumulation path allocates, so this is a
// regression guard rather than a second test of the same code.
TEST(BinaryOps, SecondBackwardAccumulatesAndStaysIndependent) {
    auto a = ones({2, 3});
    auto b = ones({2, 3});
    a.requires_grad_(true);
    b.requires_grad_(true);

    sum(add(a, b)).backward();
    sum(add(a, b)).backward();

    EXPECT_NE(a.grad().impl()->buffer().get(),
              b.grad().impl()->buffer().get());

    const auto* ga = static_cast<const float*>(a.grad().data());
    const auto* gb = static_cast<const float*>(b.grad().data());
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(ga[i], 2.0f);
        EXPECT_FLOAT_EQ(gb[i], 2.0f);
    }
}

// Broadcasting reduces one side, so unbroadcast allocates and the two leaves
// were never aliased. Pinned so the ownership fix cannot be "achieved" by
// changing what a broadcast add computes.
TEST(BinaryOps, BroadcastAddBackwardKeepsCorrectGradients) {
    auto a = ones({2, 3});
    auto b = ones({3});
    a.requires_grad_(true);
    b.requires_grad_(true);

    sum(add(a, b)).backward();

    EXPECT_EQ(a.grad().shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(b.grad().shape(), (std::vector<int64_t>{3}));
    EXPECT_NE(a.grad().impl()->buffer().get(),
              b.grad().impl()->buffer().get());

    const auto* ga = static_cast<const float*>(a.grad().data());
    for (int64_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(ga[i], 1.0f);
    const auto* gb = static_cast<const float*>(b.grad().data());
    for (int64_t i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(gb[i], 2.0f);   // summed over rows
}

// One leaf used twice in the same expression is a different situation: there
// is only one leaf, so its gradient must accumulate to 2, not be privatised
// into two halves. Guards against over-applying the ownership rule.
TEST(BinaryOps, SameLeafOnBothSidesAccumulates) {
    auto a = ones({2, 3});
    a.requires_grad_(true);

    sum(add(a, a)).backward();

    const auto* ga = static_cast<const float*>(a.grad().data());
    for (int64_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(ga[i], 2.0f);
}
