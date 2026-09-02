#include <gtest/gtest.h>
#include <cstring>
#include <orthon/core/tensor.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/matmul_ops.h>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

TEST(Matmul, Square2x2) {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    Tensor a = empty({2, 2});
    Tensor b = empty({2, 2});
    float da[] = {1, 2, 3, 4};
    float db[] = {5, 6, 7, 8};
    std::memcpy(a.data(), da, 4 * sizeof(float));
    std::memcpy(b.data(), db, 4 * sizeof(float));
    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 2}));
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 19.0f);
    EXPECT_FLOAT_EQ(p[1], 22.0f);
    EXPECT_FLOAT_EQ(p[2], 43.0f);
    EXPECT_FLOAT_EQ(p[3], 50.0f);
}

TEST(Matmul, Rect2x3by3x2) {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    Tensor a = empty({2, 3});
    Tensor b = empty({3, 2});
    float da[] = {1, 2, 3, 4, 5, 6};
    float db[] = {7, 8, 9, 10, 11, 12};
    std::memcpy(a.data(), da, 6 * sizeof(float));
    std::memcpy(b.data(), db, 6 * sizeof(float));
    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 2}));
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0],  58.0f);
    EXPECT_FLOAT_EQ(p[1],  64.0f);
    EXPECT_FLOAT_EQ(p[2], 139.0f);
    EXPECT_FLOAT_EQ(p[3], 154.0f);
}

TEST(Matmul, NonContiguousInput) {
    // transpose([[1,2],[3,4]]) = [[1,3],[2,4]]
    // [[1,3],[2,4]] @ I = [[1,3],[2,4]]
    Tensor base = empty({2, 2});
    float d[] = {1, 2, 3, 4};
    std::memcpy(base.data(), d, 4 * sizeof(float));
    Tensor tr = transpose(base, 0, 1);
    ASSERT_FALSE(tr.is_contiguous());
    Tensor eye = empty({2, 2});
    float de[] = {1, 0, 0, 1};
    std::memcpy(eye.data(), de, 4 * sizeof(float));
    Tensor c = matmul(tr, eye);
    EXPECT_TRUE(c.is_contiguous());
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 3.0f);
    EXPECT_FLOAT_EQ(p[2], 2.0f);
    EXPECT_FLOAT_EQ(p[3], 4.0f);
}

TEST(Matmul, ShapeMismatchThrows) {
    Tensor a = empty({2, 3});
    Tensor b = empty({4, 2});
    EXPECT_THROW(matmul(a, b), std::runtime_error);
}

TEST(Matmul, OutputIsContiguous) {
    Tensor a = ones({3, 3});
    Tensor b = ones({3, 3});
    Tensor c = matmul(a, b);
    EXPECT_TRUE(c.is_contiguous());
}
