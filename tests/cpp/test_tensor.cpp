#include <gtest/gtest.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/core/tensor_impl.h>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

TEST(TensorCreation, ShapeStridesNumel) {
    auto t = empty({2, 3});
    EXPECT_EQ(t.shape(),   (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(t.strides(), (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.ndim(),  2);
}

TEST(TensorCreation, ZerosValues) {
    auto t = zeros({3});
    const auto* p = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_EQ(p[i], 0.0f) << "at index " << i;
}

TEST(TensorCreation, OnesValues) {
    auto t = ones({4});
    const auto* p = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_EQ(p[i], 1.0f) << "at index " << i;
}

TEST(TensorCreation, FullValues) {
    auto t = full({5}, 3.14f);
    const auto* p = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 3.14f) << "at index " << i;
}

TEST(TensorCreation, Scalar) {
    auto t = zeros({});
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.ndim(),  0);
    EXPECT_EQ(t.strides(), std::vector<int64_t>{});
}

TEST(TensorCreation, DtypeAndDevice) {
    auto t = zeros({2, 2});
    EXPECT_EQ(t.dtype(), DType::Float32);
    EXPECT_TRUE(t.device().is_cpu());
    EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorCreation, CheapCopySharesStorage) {
    auto a = zeros({10});
    auto b = a;
    EXPECT_EQ(a.data(), b.data());
}
