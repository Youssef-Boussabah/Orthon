#include <gtest/gtest.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

// Negation.

TEST(UnaryOps, NegZeros) {
    Tensor t = zeros({3, 4});
    Tensor r = neg(t);
    EXPECT_EQ(r.shape(), t.shape());
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 0.0f);
}

TEST(UnaryOps, NegPositive) {
    Tensor t = full({4}, 2.0f);
    Tensor r = neg(t);
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], -2.0f);
}

TEST(UnaryOps, NegNegative) {
    Tensor t = full({3}, -5.0f);
    Tensor r = neg(t);
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 5.0f);
}

TEST(UnaryOps, NegPreservesShape) {
    Tensor t = zeros({2, 3, 4});
    Tensor r = neg(t);
    EXPECT_EQ(r.shape(), t.shape());
    EXPECT_EQ(r.numel(), 24);
}

TEST(UnaryOps, NegDoesNotMutateInput) {
    Tensor t = full({3}, 7.0f);
    neg(t);
    const float* p = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 7.0f);
}

TEST(UnaryOps, NegNonContiguous) {
    Tensor t = full({2, 3}, 4.0f);
    Tensor tr = transpose(t, 0, 1);
    EXPECT_FALSE(tr.is_contiguous());
    Tensor r = neg(tr);
    EXPECT_TRUE(r.is_contiguous());
    EXPECT_EQ(r.shape(), tr.shape());
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], -4.0f);
}

// ReLU.

TEST(UnaryOps, ReluAllPositive) {
    Tensor t = full({4}, 3.0f);
    Tensor r = relu(t);
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 3.0f);
}

TEST(UnaryOps, ReluAllNegative) {
    Tensor t = full({4}, -3.0f);
    Tensor r = relu(t);
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 0.0f);
}

TEST(UnaryOps, ReluZero) {
    Tensor t = zeros({3});
    Tensor r = relu(t);
    const float* p = static_cast<const float*>(r.data());
    for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 0.0f);
}

TEST(UnaryOps, ReluPreservesShape) {
    Tensor t = zeros({2, 5});
    Tensor r = relu(t);
    EXPECT_EQ(r.shape(), t.shape());
}

TEST(UnaryOps, ReluDoesNotMutateInput) {
    Tensor t = full({3}, -1.0f);
    relu(t);
    const float* p = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) EXPECT_FLOAT_EQ(p[i], -1.0f);
}

TEST(UnaryOps, ReluNonContiguous) {
    Tensor base = zeros({2, 3});
    float* bp = static_cast<float*>(base.data());
    bp[0] = -1.0f; bp[1] =  2.0f; bp[2] = -3.0f;
    bp[3] =  4.0f; bp[4] = -5.0f; bp[5] =  6.0f;
    Tensor tr = transpose(base, 0, 1);  // shape (3,2), non-contiguous
    EXPECT_FALSE(tr.is_contiguous());
    Tensor r = relu(tr);
    EXPECT_TRUE(r.is_contiguous());
    // Logical order of tr: (0,0)=-1 (0,1)=4 (1,0)=2 (1,1)=-5 (2,0)=-3 (2,1)=6
    const float* p = static_cast<const float*>(r.data());
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 4.0f);
    EXPECT_FLOAT_EQ(p[2], 2.0f);
    EXPECT_FLOAT_EQ(p[3], 0.0f);
    EXPECT_FLOAT_EQ(p[4], 0.0f);
    EXPECT_FLOAT_EQ(p[5], 6.0f);
}
