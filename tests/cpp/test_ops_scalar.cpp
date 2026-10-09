// Native coverage: full() scalar precision, Tensor-Tensor sub/div, and the
// scalar arithmetic primitives.
//
// These exercise the C++ API directly, with no Python in the picture, so the
// contracts below stand on their own rather than through the bindings.
#include <gtest/gtest.h>

#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/node.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/ops/scalar_ops.h>
#include <orthon/ops/shape_ops.h>

#include <cmath>
#include <stdexcept>
#include <vector>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

static const float*  f32(const Tensor& t) { return static_cast<const float*>(t.data()); }
static const double* f64(const Tensor& t) { return static_cast<const double*>(t.data()); }

static Tensor from_f32(const std::vector<int64_t>& shape,
                       const std::vector<float>& values) {
    Tensor t = empty(shape, DType::Float32, Device{});
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < values.size(); ++i) p[i] = values[i];
    return t;
}

static Tensor from_f64(const std::vector<int64_t>& shape,
                       const std::vector<double>& values) {
    Tensor t = empty(shape, DType::Float64, Device{});
    double* p = static_cast<double*>(t.data());
    for (size_t i = 0; i < values.size(); ++i) p[i] = values[i];
    return t;
}

// full() no longer narrows its value through float32.
TEST(FullPrecision, Float64KeepsFullDoubleOneThird) {
    Tensor t = full({}, 1.0 / 3.0, DType::Float64, Device{});
    // Exact equality is the point: any float32 round-trip lands on
    // 0.3333333432674408 instead.
    EXPECT_EQ(f64(t)[0], 1.0 / 3.0);
    EXPECT_NE(f64(t)[0], static_cast<double>(static_cast<float>(1.0 / 3.0)));
}

TEST(FullPrecision, Float64KeepsFullDoublePi) {
    const double pi = 3.141592653589793;
    Tensor t = full({2}, pi, DType::Float64, Device{});
    EXPECT_EQ(f64(t)[0], pi);
    EXPECT_EQ(f64(t)[1], pi);
}

TEST(FullPrecision, Float64Rank0AndShapedAgree) {
    const double v = 0.1 + 0.2;   // 0.30000000000000004, not representable in f32
    EXPECT_EQ(f64(full({}, v, DType::Float64, Device{}))[0], v);
    EXPECT_EQ(f64(full({3}, v, DType::Float64, Device{}))[2], v);
}

TEST(FullPrecision, Float32MatchesExplicitFloatRounding) {
    const double v = 1.0 / 3.0;
    Tensor t = full({2}, v, DType::Float32, Device{});
    EXPECT_EQ(f32(t)[0], static_cast<float>(v));
    EXPECT_EQ(f32(t)[1], static_cast<float>(v));
}

TEST(FullPrecision, IntegerAndBoolConversionUnchanged) {
    Tensor i32 = full({2}, 3.9, DType::Int32, Device{});
    EXPECT_EQ(static_cast<const int32_t*>(i32.data())[0], 3);       // truncates
    Tensor i64 = full({2}, -2.5, DType::Int64, Device{});
    EXPECT_EQ(static_cast<const int64_t*>(i64.data())[0], -2);      // toward zero
    Tensor b_true  = full({1}, 0.5, DType::Bool, Device{});
    Tensor b_false = full({1}, 0.0, DType::Bool, Device{});
    EXPECT_EQ(static_cast<const uint8_t*>(b_true.data())[0],  1u);
    EXPECT_EQ(static_cast<const uint8_t*>(b_false.data())[0], 0u);
}

TEST(FullPrecision, OnesStillOnes) {
    Tensor t = ones({3}, DType::Float64, Device{});
    for (int64_t i = 0; i < t.numel(); ++i) EXPECT_EQ(f64(t)[i], 1.0);
}

// float64 mean backward scale is formed in double.
TEST(MeanBackwardPrecision, Float64ScaleIsExactDouble) {
    for (int64_t factor : {int64_t{3}, int64_t{6}, int64_t{7}, int64_t{1000}}) {
        Tensor t = full({factor}, 1.0, DType::Float64, Device{});
        t.requires_grad_(true);
        mean(t).backward();
        const double expected = 1.0 / static_cast<double>(factor);
        const double got = f64(t.grad())[0];
        EXPECT_DOUBLE_EQ(got, expected) << "factor " << factor;
        // And it is strictly better than what a float32 scale would give.
        EXPECT_NE(got, static_cast<double>(1.0f / static_cast<float>(factor)))
            << "factor " << factor;
    }
}

TEST(MeanBackwardPrecision, Float32ScaleUnchanged) {
    Tensor t = full({6}, 1.0f, DType::Float32, Device{});
    t.requires_grad_(true);
    mean(t).backward();
    EXPECT_FLOAT_EQ(f32(t.grad())[0], 1.0f / 6.0f);
}

// Tensor-Tensor sub and div.
TEST(SubDiv, SubForwardFloat32) {
    Tensor a = from_f32({3}, {5.0f, 2.0f, -1.0f});
    Tensor b = from_f32({3}, {1.0f, 4.0f, -3.0f});
    Tensor c = sub(a, b);
    EXPECT_FLOAT_EQ(f32(c)[0], 4.0f);
    EXPECT_FLOAT_EQ(f32(c)[1], -2.0f);
    EXPECT_FLOAT_EQ(f32(c)[2], 2.0f);
}

TEST(SubDiv, DivForwardFloat32) {
    Tensor a = from_f32({3}, {6.0f, 2.0f, -9.0f});
    Tensor b = from_f32({3}, {2.0f, 4.0f, 3.0f});
    Tensor c = div(a, b);
    EXPECT_FLOAT_EQ(f32(c)[0], 3.0f);
    EXPECT_FLOAT_EQ(f32(c)[1], 0.5f);
    EXPECT_FLOAT_EQ(f32(c)[2], -3.0f);
}

TEST(SubDiv, SubForwardFloat64) {
    Tensor a = from_f64({2}, {1.5, -0.25});
    Tensor b = from_f64({2}, {0.5, 0.25});
    Tensor c = sub(a, b);
    EXPECT_EQ(c.dtype(), DType::Float64);
    EXPECT_DOUBLE_EQ(f64(c)[0], 1.0);
    EXPECT_DOUBLE_EQ(f64(c)[1], -0.5);
}

TEST(SubDiv, DivForwardFloat64) {
    Tensor a = from_f64({2}, {1.0, 3.0});
    Tensor b = from_f64({2}, {4.0, 2.0});
    Tensor c = div(a, b);
    EXPECT_EQ(c.dtype(), DType::Float64);
    EXPECT_DOUBLE_EQ(f64(c)[0], 0.25);
    EXPECT_DOUBLE_EQ(f64(c)[1], 1.5);
}

TEST(SubDiv, SubBroadcasts) {
    Tensor a = from_f32({2, 1}, {10.0f, 20.0f});
    Tensor b = from_f32({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor c = sub(a, b);
    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 3}));
    EXPECT_FLOAT_EQ(f32(c)[0], 9.0f);
    EXPECT_FLOAT_EQ(f32(c)[5], 17.0f);
}

TEST(SubDiv, SubOnAView) {
    Tensor base = from_f32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor tr = transpose(base, 0, 1);
    Tensor c = sub(tr, full({}, 1.0, DType::Float32, Device{}));
    // Transposed logical order is 1, 3, 2, 4.
    EXPECT_FLOAT_EQ(f32(c)[0], 0.0f);
    EXPECT_FLOAT_EQ(f32(c)[1], 2.0f);
    EXPECT_FLOAT_EQ(f32(c)[2], 1.0f);
    EXPECT_FLOAT_EQ(f32(c)[3], 3.0f);
}

TEST(SubDiv, SubBackwardSignsMatchAnalytic) {
    Tensor a = from_f32({3}, {5.0f, 2.0f, 1.0f});
    Tensor b = from_f32({3}, {1.0f, 1.0f, 1.0f});
    a.requires_grad_(true);
    b.requires_grad_(true);
    sum(sub(a, b)).backward();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(f32(a.grad())[i], 1.0f);
        EXPECT_FLOAT_EQ(f32(b.grad())[i], -1.0f);
    }
}

TEST(SubDiv, DivBackwardMatchesAnalytic) {
    Tensor a = from_f32({2}, {6.0f, 3.0f});
    Tensor b = from_f32({2}, {2.0f, 4.0f});
    a.requires_grad_(true);
    b.requires_grad_(true);
    sum(div(a, b)).backward();
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    EXPECT_FLOAT_EQ(f32(a.grad())[0], 0.5f);
    EXPECT_FLOAT_EQ(f32(a.grad())[1], 0.25f);
    EXPECT_FLOAT_EQ(f32(b.grad())[0], -6.0f / 4.0f);
    EXPECT_FLOAT_EQ(f32(b.grad())[1], -3.0f / 16.0f);
}

TEST(SubDiv, DtypeAndDeviceValidationInherited) {
    Tensor f = zeros({3}, DType::Float32, Device{});
    Tensor d = zeros({3}, DType::Float64, Device{});
    EXPECT_THROW(sub(f, d), std::runtime_error);
    EXPECT_THROW(div(f, d), std::runtime_error);
    Tensor a = zeros({2, 3}, DType::Float32, Device{});
    Tensor b = zeros({4, 5}, DType::Float32, Device{});
    EXPECT_THROW(sub(a, b), std::runtime_error);
    EXPECT_THROW(div(a, b), std::runtime_error);
}

TEST(SubDiv, DivByZeroPreservesFloatingSemantics) {
    Tensor a = from_f32({3}, {1.0f, -1.0f, 0.0f});
    Tensor b = from_f32({3}, {0.0f, 0.0f, 0.0f});
    Tensor c = div(a, b);
    EXPECT_TRUE(std::isinf(f32(c)[0]) && f32(c)[0] > 0.0f);
    EXPECT_TRUE(std::isinf(f32(c)[1]) && f32(c)[1] < 0.0f);
    EXPECT_TRUE(std::isnan(f32(c)[2]));
}

// the scalar representation contract.
TEST(ScalarOps, ScalarLikeIsRank0AndUntracked) {
    Tensor ref = zeros({64, 64}, DType::Float32, Device{});
    ref.requires_grad_(true);
    Tensor s = scalar_like(ref, 2.5);
    EXPECT_TRUE(s.shape().empty());          // rank 0, not ref-shaped
    EXPECT_EQ(s.numel(), 1);                 // O(1) storage
    EXPECT_EQ(s.dtype(), ref.dtype());
    EXPECT_EQ(s.device(), ref.device());
    EXPECT_FALSE(s.requires_grad());
    EXPECT_FALSE(s.has_grad_fn());
    EXPECT_TRUE(s.is_leaf());
    EXPECT_FLOAT_EQ(f32(s)[0], 2.5f);
}

TEST(ScalarOps, ScalarLikeFollowsTensorDtypeNotPythonDouble) {
    Tensor f64_ref = zeros({2}, DType::Float64, Device{});
    Tensor sd = scalar_like(f64_ref, 1.0 / 3.0);
    EXPECT_EQ(sd.dtype(), DType::Float64);
    EXPECT_EQ(f64(sd)[0], 1.0 / 3.0);        // full double survives

    Tensor f32_ref = zeros({2}, DType::Float32, Device{});
    Tensor sf = scalar_like(f32_ref, 1.0 / 3.0);
    EXPECT_EQ(sf.dtype(), DType::Float32);   // NOT promoted to float64
    EXPECT_EQ(f32(sf)[0], static_cast<float>(1.0 / 3.0));
}

// scalar forward and backward.
TEST(ScalarOps, ForwardValues) {
    Tensor x = from_f32({3}, {1.0f, 2.0f, 4.0f});
    EXPECT_FLOAT_EQ(f32(scalar_add(x, 3.0))[0], 4.0f);
    EXPECT_FLOAT_EQ(f32(scalar_sub(x, 3.0))[1], -1.0f);
    EXPECT_FLOAT_EQ(f32(scalar_rsub(3.0, x))[1], 1.0f);
    EXPECT_FLOAT_EQ(f32(scalar_mul(x, 3.0))[2], 12.0f);
    EXPECT_FLOAT_EQ(f32(scalar_div(x, 4.0))[2], 1.0f);
    EXPECT_FLOAT_EQ(f32(scalar_rdiv(4.0, x))[2], 1.0f);
}

TEST(ScalarOps, ForwardShapeIsUnchangedByRank0Operand) {
    Tensor x = zeros({2, 3, 4}, DType::Float32, Device{});
    EXPECT_EQ(scalar_add(x, 1.0).shape(), x.shape());
    EXPECT_EQ(scalar_mul(x, 1.0).shape(), x.shape());
}

TEST(ScalarOps, BackwardMatchesAnalyticDerivatives) {
    auto grad_of = [](Tensor (*build)(const Tensor&)) {
        Tensor x = from_f32({2}, {2.0f, 4.0f});
        x.requires_grad_(true);
        sum(build(x)).backward();
        return std::vector<float>{f32(x.grad())[0], f32(x.grad())[1]};
    };

    auto g_mul  = grad_of([](const Tensor& t) { return scalar_mul(t, 3.0); });
    EXPECT_FLOAT_EQ(g_mul[0], 3.0f);

    auto g_add  = grad_of([](const Tensor& t) { return scalar_add(t, 3.0); });
    EXPECT_FLOAT_EQ(g_add[0], 1.0f);

    auto g_sub  = grad_of([](const Tensor& t) { return scalar_sub(t, 3.0); });
    EXPECT_FLOAT_EQ(g_sub[0], 1.0f);

    auto g_rsub = grad_of([](const Tensor& t) { return scalar_rsub(3.0, t); });
    EXPECT_FLOAT_EQ(g_rsub[0], -1.0f);

    auto g_div  = grad_of([](const Tensor& t) { return scalar_div(t, 4.0); });
    EXPECT_FLOAT_EQ(g_div[0], 0.25f);

    // d(4/x)/dx = -4/x^2 -> -1.0 at x=2, -0.25 at x=4
    auto g_rdiv = grad_of([](const Tensor& t) { return scalar_rdiv(4.0, t); });
    EXPECT_FLOAT_EQ(g_rdiv[0], -1.0f);
    EXPECT_FLOAT_EQ(g_rdiv[1], -0.25f);
}

TEST(ScalarOps, ScalarConstantNeverReceivesGradient) {
    // The rank-0 operand is not reachable from the graph at all: the node holds
    // exactly one input, the tensor.
    Tensor x = from_f32({2}, {1.0f, 2.0f});
    x.requires_grad_(true);
    Tensor y = scalar_mul(x, 3.0);
    ASSERT_TRUE(y.has_grad_fn());
    EXPECT_EQ(y.grad_fn()->inputs.size(), 1u);
    EXPECT_EQ(y.grad_fn()->inputs[0].impl().get(), x.impl().get());
}

// mutation selectivity. Watch only what backward actually reads.
static void mutate(Tensor& t) {
    static_cast<float*>(t.data())[0] = 99.0f;
    t.impl()->buffer()->bump_version();
}

TEST(ScalarVersioning, AddMulSubDivDoNotWatchTheInput) {
    struct Case { const char* name; Tensor (*build)(const Tensor&); };
    const Case cases[] = {
        {"x + s", [](const Tensor& t) { return scalar_add(t, 3.0); }},
        {"x * s", [](const Tensor& t) { return scalar_mul(t, 3.0); }},
        {"x - s", [](const Tensor& t) { return scalar_sub(t, 3.0); }},
        {"s - x", [](const Tensor& t) { return scalar_rsub(3.0, t); }},
        {"x / s", [](const Tensor& t) { return scalar_div(t, 4.0); }},
    };
    for (const Case& c : cases) {
        Tensor x = from_f32({2}, {1.0f, 2.0f});
        x.requires_grad_(true);
        Tensor y = c.build(x);
        mutate(x);
        // Backward reads no forward-time payload of x, so this must succeed.
        EXPECT_NO_THROW(sum(y).backward()) << c.name;
    }
}

TEST(ScalarVersioning, ReverseDivisionDoesWatchTheInput) {
    // d(s/x)/dx = -s/x^2 really does need x's forward values.
    Tensor x = from_f32({2}, {1.0f, 2.0f});
    x.requires_grad_(true);
    Tensor y = scalar_rdiv(4.0, x);
    mutate(x);
    EXPECT_THROW(sum(y).backward(), std::runtime_error);
}

// scalar arithmetic follows normal grad mode.
TEST(ScalarOps, NoGradProducesNoGraph) {
    Tensor x = from_f32({2}, {1.0f, 2.0f});
    x.requires_grad_(true);
    NoGradGuard guard;
    Tensor y = scalar_mul(x, 3.0);
    EXPECT_FALSE(y.requires_grad());
    EXPECT_FALSE(y.has_grad_fn());
    EXPECT_TRUE(y.is_leaf());
}

TEST(ScalarOps, UntrackedInputProducesUntrackedOutput) {
    Tensor x = from_f32({2}, {1.0f, 2.0f});
    Tensor y = scalar_mul(x, 3.0);
    EXPECT_FALSE(y.requires_grad());
    EXPECT_FALSE(y.has_grad_fn());
}

// unbroadcast is typed on the gradient's dtype (float64 broadcast backward).
TEST(BroadcastBackward, Float64GradientKeepsItsDtypeAndValues) {
    Tensor a = from_f64({1, 3}, {1.0, 2.0, 3.0});
    Tensor b = from_f64({2, 1}, {10.0, 20.0});
    a.requires_grad_(true);
    b.requires_grad_(true);
    sum(add(a, b)).backward();
    EXPECT_EQ(a.grad().dtype(), DType::Float64);
    EXPECT_EQ(b.grad().dtype(), DType::Float64);
    for (int64_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(f64(a.grad())[i], 2.0);
    for (int64_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(f64(b.grad())[i], 3.0);
}
