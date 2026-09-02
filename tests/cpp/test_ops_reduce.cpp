#include <gtest/gtest.h>
#include <orthon/ops/cpu/reduce_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

static float scalar_value(const Tensor& t) {
    return static_cast<const float*>(t.data())[0];
}

// Sum.

TEST(ReduceOps, SumZeros) {
    Tensor t = zeros({3, 4});
    EXPECT_FLOAT_EQ(scalar_value(sum(t)), 0.0f);
}

TEST(ReduceOps, SumOnes) {
    Tensor t = ones({3, 4});
    EXPECT_FLOAT_EQ(scalar_value(sum(t)), 12.0f);
}

TEST(ReduceOps, SumFull) {
    Tensor t = full({5}, 2.5f);
    EXPECT_FLOAT_EQ(scalar_value(sum(t)), 12.5f);
}

TEST(ReduceOps, SumContiguous2D) {
    Tensor t = full({2, 3}, 1.0f);
    EXPECT_FLOAT_EQ(scalar_value(sum(t)), 6.0f);
}

TEST(ReduceOps, SumNonContiguous) {
    Tensor t = ones({2, 3});
    Tensor tr = transpose(t, 0, 1);
    EXPECT_FALSE(tr.is_contiguous());
    EXPECT_FLOAT_EQ(scalar_value(sum(tr)), 6.0f);
}

TEST(ReduceOps, SumNonContiguousMixed) {
    Tensor t = zeros({2, 2});
    static_cast<float*>(t.data())[0] = 1.0f;
    static_cast<float*>(t.data())[1] = 2.0f;
    static_cast<float*>(t.data())[2] = 3.0f;
    static_cast<float*>(t.data())[3] = 4.0f;
    Tensor tr = transpose(t, 0, 1);
    EXPECT_FLOAT_EQ(scalar_value(sum(tr)), 10.0f);
}

TEST(ReduceOps, SumEmptyThrows) {
    Tensor t = empty({0});
    EXPECT_THROW(sum(t), std::runtime_error);
}

// Mean.

TEST(ReduceOps, MeanFull) {
    Tensor t = full({4}, 3.0f);
    EXPECT_FLOAT_EQ(scalar_value(mean(t)), 3.0f);
}

TEST(ReduceOps, MeanOnes) {
    Tensor t = ones({2, 5});
    EXPECT_FLOAT_EQ(scalar_value(mean(t)), 1.0f);
}

TEST(ReduceOps, MeanNonContiguous) {
    Tensor t = full({3, 3}, 6.0f);
    Tensor tr = transpose(t, 0, 1);
    EXPECT_FALSE(tr.is_contiguous());
    EXPECT_FLOAT_EQ(scalar_value(mean(tr)), 6.0f);
}

TEST(ReduceOps, MeanEmptyThrows) {
    Tensor t = empty({0});
    EXPECT_THROW(mean(t), std::runtime_error);
}
