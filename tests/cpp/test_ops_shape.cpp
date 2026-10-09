#include <gtest/gtest.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

TEST(ShapeOps, ReshapeMetadata) {
    auto t = zeros({2, 3});
    auto r = reshape(t, {3, 2});
    EXPECT_EQ(r.shape(),   (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(r.strides(), (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(r.numel(), 6);
}

TEST(ShapeOps, ReshapeSharesStorage) {
    auto t = full({6}, 7.0f);
    auto r = reshape(t, {2, 3});
    EXPECT_EQ(r.data(), t.data());
}

TEST(ShapeOps, ReshapeToFlat) {
    auto t = zeros({2, 3});
    auto r = reshape(t, {6});
    EXPECT_EQ(r.shape(),   (std::vector<int64_t>{6}));
    EXPECT_EQ(r.strides(), (std::vector<int64_t>{1}));
}

TEST(ShapeOps, ReshapeNumelMismatch) {
    auto t = zeros({2, 3});
    EXPECT_THROW(reshape(t, {2, 4}), std::invalid_argument);
}

TEST(ShapeOps, ReshapeNonContiguousThrows) {
    auto t  = zeros({2, 3});
    auto tr = transpose(t, 0, 1);   // non-contiguous view
    EXPECT_THROW(reshape(tr, {6}), std::runtime_error);
}

TEST(ShapeOps, TransposeMetadata) {
    auto t = zeros({2, 3});
    auto r = transpose(t, 0, 1);
    EXPECT_EQ(r.shape(),   (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(r.strides(), (std::vector<int64_t>{1, 3}));
}

TEST(ShapeOps, TransposeSharesStorage) {
    auto t = zeros({2, 3});
    auto r = transpose(t, 0, 1);
    EXPECT_EQ(r.data(), t.data());
}

TEST(ShapeOps, TransposeIsNotContiguous) {
    auto t = zeros({2, 3});
    EXPECT_FALSE(transpose(t, 0, 1).is_contiguous());
}

// Out of range means out of range AFTER normalisation, in both directions.
// transpose() resolves a negative dim against ndim before range-checking,
// exactly like Tensor::slice and the reductions' resolve_dim, so
// `transpose(t, -1, 0)` on a 2-D tensor is the valid call
// `transpose(t, 1, 0)` and not an error. Normalize before validating:
// functional/math.py builds attention scores with `k.transpose(-2, -1)`.
TEST(ShapeOps, TransposeInvalidDim) {
    auto t = zeros({2, 3});
    EXPECT_THROW(transpose(t,  0,  3), std::out_of_range);   // >= ndim
    EXPECT_THROW(transpose(t,  3,  0), std::out_of_range);
    EXPECT_THROW(transpose(t,  0, -3), std::out_of_range);   // < 0 after += ndim
    EXPECT_THROW(transpose(t, -3,  0), std::out_of_range);
}

TEST(ShapeOps, TransposeNormalisesNegativeDims) {
    auto t = zeros({2, 3});
    auto negative = transpose(t, -1, -2);   // == transpose(t, 1, 0)
    auto positive = transpose(t,  1,  0);
    EXPECT_EQ(negative.shape(),   positive.shape());
    EXPECT_EQ(negative.strides(), positive.strides());
    EXPECT_EQ(negative.shape(), (std::vector<int64_t>{3, 2}));

    auto mixed = transpose(t, -1, 0);       // == transpose(t, 1, 0)
    EXPECT_EQ(mixed.shape(), (std::vector<int64_t>{3, 2}));
}

TEST(ShapeOps, TransposeSameDimIsNoop) {
    auto t = zeros({2, 3});
    auto r = transpose(t, 1, 1);
    EXPECT_EQ(r.shape(),   t.shape());
    EXPECT_EQ(r.strides(), t.strides());
}
