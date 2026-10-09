// Native coverage for the canonical low-precision conversions.
//
// These run with no Python and no NumPy in the picture, so the IEEE-754
// round-to-nearest-even contract in include/orthon/core/float_formats.h
// stands on its own. The expected bit patterns below are the IEEE results,
// which are also what NumPy float16 and PyTorch bfloat16 produce - the Python
// suite (tests/python/test_low_precision.py) asserts that agreement directly.
#include <gtest/gtest.h>

#include <orthon/core/dtype.h>
#include <orthon/core/float_formats.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/ops/fill_ops.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

static float bits_to_f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t f32_to_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// float32 -> float16
TEST(FloatFormats, F16SignedZero) {
    EXPECT_EQ(f32_to_f16_bits(0.0f), 0x0000u);
    EXPECT_EQ(f32_to_f16_bits(-0.0f), 0x8000u);
}

TEST(FloatFormats, F16Normals) {
    EXPECT_EQ(f32_to_f16_bits(1.0f), 0x3C00u);
    EXPECT_EQ(f32_to_f16_bits(-1.0f), 0xBC00u);
    EXPECT_EQ(f32_to_f16_bits(0.5f), 0x3800u);
    EXPECT_EQ(f32_to_f16_bits(2.0f), 0x4000u);
    EXPECT_EQ(f32_to_f16_bits(65504.0f), 0x7BFFu);   // largest finite half
}

TEST(FloatFormats, F16SmallestNormalAndTheBoundary) {
    EXPECT_EQ(f32_to_f16_bits(6.103515625e-05f), 0x0400u);   // 2^-14
    EXPECT_EQ(f32_to_f16_bits(6.0975552e-05f), 0x03FFu);     // largest subnormal
}

TEST(FloatFormats, F16Subnormals) {
    // A subnormal shift one bit too large returns 1e-5 as
    // 4.947185516357422e-06 (0x0053) instead of 0x00A8, so these pin the
    // boundary explicitly.
    EXPECT_EQ(f32_to_f16_bits(1e-5f), 0x00A8u);
    EXPECT_EQ(f32_to_f16_bits(1e-6f), 0x0011u);
    EXPECT_EQ(f32_to_f16_bits(1e-7f), 0x0002u);
    // 2^-24, the smallest representable half. Used to underflow to zero.
    EXPECT_EQ(f32_to_f16_bits(5.960464477539063e-08f), 0x0001u);
    EXPECT_EQ(f32_to_f16_bits(6e-8f), 0x0001u);
    EXPECT_EQ(f32_to_f16_bits(-6e-8f), 0x8001u);
}

TEST(FloatFormats, F16UnderflowAndItsTie) {
    // Exactly 2^-25 is halfway between 0 and 2^-24; ties-to-even picks zero.
    EXPECT_EQ(f32_to_f16_bits(2.980232238769531e-08f), 0x0000u);
    // Anything above that tie rounds up to the smallest subnormal.
    EXPECT_EQ(f32_to_f16_bits(2.9802327e-08f), 0x0001u);
    // Well below the tie, still zero, sign preserved.
    EXPECT_EQ(f32_to_f16_bits(1e-9f), 0x0000u);
    EXPECT_EQ(f32_to_f16_bits(-1e-9f), 0x8000u);
}

TEST(FloatFormats, F16RoundsToNearestEvenNotTowardZero) {
    // Normals used to TRUNCATE the mantissa. In the [2048, 4096) binade the
    // half spacing is 2, so:
    EXPECT_EQ(f32_to_f16_bits(2050.0f), 0x6801u);   // representable exactly
    EXPECT_EQ(f32_to_f16_bits(2051.0f), 0x6802u);   // rounds UP to 2052
    EXPECT_EQ(f32_to_f16_bits(2049.0f), 0x6800u);   // exact tie -> even (2048)
    EXPECT_EQ(f32_to_f16_bits(2053.0f), 0x6802u);   // exact tie -> even (2052)
}

TEST(FloatFormats, F16OverflowBoundary) {
    // 65504 is the largest finite half; the next step is 32, so the rounding
    // boundary to infinity is 65520.
    EXPECT_EQ(f32_to_f16_bits(65504.0f), 0x7BFFu);
    EXPECT_EQ(f32_to_f16_bits(65519.0f), 0x7BFFu);
    EXPECT_EQ(f32_to_f16_bits(65520.0f), 0x7C00u);   // exact tie -> even -> inf
    EXPECT_EQ(f32_to_f16_bits(70000.0f), 0x7C00u);
    EXPECT_EQ(f32_to_f16_bits(1e30f), 0x7C00u);
    EXPECT_EQ(f32_to_f16_bits(-1e30f), 0xFC00u);
    // A large finite float must never wrap into the sign bit.
    EXPECT_EQ(f32_to_f16_bits(std::numeric_limits<float>::max()), 0x7C00u);
    EXPECT_EQ(f32_to_f16_bits(-std::numeric_limits<float>::max()), 0xFC00u);
}

TEST(FloatFormats, F16Infinity) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(f32_to_f16_bits(inf), 0x7C00u);
    EXPECT_EQ(f32_to_f16_bits(-inf), 0xFC00u);
}

TEST(FloatFormats, F16NaNStaysNaN) {
    const uint16_t quiet = f32_to_f16_bits(bits_to_f32(0x7FC00000u));
    EXPECT_EQ(quiet & 0x7C00u, 0x7C00u);
    EXPECT_NE(quiet & 0x03FFu, 0u);          // a NaN, never an infinity

    // A NaN whose payload lives entirely in the bits float16 discards must
    // still come back as a NaN rather than collapsing to infinity.
    const uint16_t tiny = f32_to_f16_bits(bits_to_f32(0x7F800001u));
    EXPECT_EQ(tiny & 0x7C00u, 0x7C00u);
    EXPECT_NE(tiny & 0x03FFu, 0u);

    const uint16_t negative = f32_to_f16_bits(bits_to_f32(0xFFC00000u));
    EXPECT_EQ(negative & 0x8000u, 0x8000u);
    EXPECT_NE(negative & 0x03FFu, 0u);
}

// float16 -> float32
TEST(FloatFormats, F16DecodeRoundTripsEveryHalf) {
    // Every one of the 65536 half patterns must widen and narrow back to
    // itself: float32 represents all of them exactly.
    for (uint32_t i = 0; i < 65536u; ++i) {
        const uint16_t bits = static_cast<uint16_t>(i);
        const uint32_t exponent = (bits >> 10) & 0x1Fu;
        const uint32_t mantissa = bits & 0x3FFu;
        if (exponent == 31u && mantissa != 0u) continue;   // NaN: compare below
        EXPECT_EQ(f32_to_f16_bits(f16_bits_to_f32(bits)), bits)
            << "half pattern 0x" << std::hex << i;
    }
}

TEST(FloatFormats, F16DecodeKnownValues) {
    EXPECT_FLOAT_EQ(f16_bits_to_f32(0x3C00u), 1.0f);
    EXPECT_FLOAT_EQ(f16_bits_to_f32(0xBC00u), -1.0f);
    EXPECT_FLOAT_EQ(f16_bits_to_f32(0x0000u), 0.0f);
    EXPECT_TRUE(std::signbit(f16_bits_to_f32(0x8000u)));
    EXPECT_FLOAT_EQ(f16_bits_to_f32(0x7BFFu), 65504.0f);
    EXPECT_FLOAT_EQ(f16_bits_to_f32(0x0001u), 5.960464477539063e-08f);
    EXPECT_FLOAT_EQ(f16_bits_to_f32(0x00A8u), 1.0013580322265625e-05f);
    EXPECT_TRUE(std::isinf(f16_bits_to_f32(0x7C00u)));
    EXPECT_TRUE(std::isnan(f16_bits_to_f32(0x7E00u)));
}

// float32 -> bfloat16
TEST(FloatFormats, BF16SignedZeroAndNormals) {
    EXPECT_EQ(f32_to_bf16_bits(0.0f), 0x0000u);
    EXPECT_EQ(f32_to_bf16_bits(-0.0f), 0x8000u);
    EXPECT_EQ(f32_to_bf16_bits(1.0f), 0x3F80u);
    EXPECT_EQ(f32_to_bf16_bits(-1.0f), 0xBF80u);
    EXPECT_EQ(f32_to_bf16_bits(0.5f), 0x3F00u);
}

TEST(FloatFormats, BF16RoundsToNearestEven) {
    // fill_ops used to TRUNCATE here, giving 0x477F (65280) for this value.
    EXPECT_EQ(f32_to_bf16_bits(65504.0f), 0x4780u);
    // Exact ties: these two float32 values sit halfway between adjacent
    // bfloat16 values, which is where ties-to-even and the old round-half-up
    // (bits + 0x8000) >> 16 disagree.
    EXPECT_EQ(f32_to_bf16_bits(bits_to_f32(0x3F808000u)), 0x3F80u);  // -> even
    EXPECT_EQ(f32_to_bf16_bits(bits_to_f32(0x3F818000u)), 0x3F82u);  // -> even
    // Just above and below a tie.
    EXPECT_EQ(f32_to_bf16_bits(bits_to_f32(0x3F808001u)), 0x3F81u);
    EXPECT_EQ(f32_to_bf16_bits(bits_to_f32(0x3F807FFFu)), 0x3F80u);
}

TEST(FloatFormats, BF16OverflowAndInfinity) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(f32_to_bf16_bits(inf), 0x7F80u);
    EXPECT_EQ(f32_to_bf16_bits(-inf), 0xFF80u);
    // The largest finite float rounds up past the largest finite bfloat16.
    EXPECT_EQ(f32_to_bf16_bits(std::numeric_limits<float>::max()), 0x7F80u);
    EXPECT_EQ(f32_to_bf16_bits(-std::numeric_limits<float>::max()), 0xFF80u);
}

TEST(FloatFormats, BF16NaNStaysNaN) {
    // Rounding a NaN with an all-ones mantissa would carry into the exponent
    // and produce an infinity, so NaN is handled before the rounding step.
    const uint16_t worst = f32_to_bf16_bits(bits_to_f32(0x7FFFFFFFu));
    EXPECT_EQ(worst & 0x7F80u, 0x7F80u);
    EXPECT_NE(worst & 0x007Fu, 0u);

    const uint16_t quiet = f32_to_bf16_bits(bits_to_f32(0x7FC00000u));
    EXPECT_EQ(quiet & 0x7F80u, 0x7F80u);
    EXPECT_NE(quiet & 0x007Fu, 0u);

    const uint16_t negative = f32_to_bf16_bits(bits_to_f32(0xFFFFFFFFu));
    EXPECT_EQ(negative & 0x8000u, 0x8000u);
    EXPECT_NE(negative & 0x007Fu, 0u);
}

TEST(FloatFormats, BF16DecodeIsATruncatedFloat) {
    EXPECT_EQ(f32_to_bits(bf16_bits_to_f32(0x3F80u)), 0x3F800000u);
    EXPECT_EQ(f32_to_bits(bf16_bits_to_f32(0x8000u)), 0x80000000u);
    EXPECT_TRUE(std::isinf(bf16_bits_to_f32(0x7F80u)));
}

TEST(FloatFormats, BF16DecodeRoundTripsEveryPattern) {
    for (uint32_t i = 0; i < 65536u; ++i) {
        const uint16_t bits = static_cast<uint16_t>(i);
        if ((bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0u) continue;
        EXPECT_EQ(f32_to_bf16_bits(bf16_bits_to_f32(bits)), bits)
            << "bfloat16 pattern 0x" << std::hex << i;
    }
}

// The three host entry points must all use these helpers.
static uint16_t stored_bits(const Tensor& t) {
    return static_cast<const uint16_t*>(t.data())[0];
}

TEST(FloatFormats, FullAndCastAgreeOnFloat16) {
    const double values[] = {1e-5, 6e-8, 1.0 / 3.0, 65504.0, 2051.0, -2.25};
    for (double value : values) {
        Tensor filled = full({1}, value, DType::Float16, Device{});
        Tensor source = full({1}, value, DType::Float32, Device{});
        Tensor converted = cast(source, DType::Float16);
        const uint16_t expected = f32_to_f16_bits(static_cast<float>(value));
        EXPECT_EQ(stored_bits(filled), expected) << "full " << value;
        EXPECT_EQ(stored_bits(converted), expected) << "cast " << value;
    }
}

TEST(FloatFormats, FullAndCastAgreeOnBFloat16) {
    const double values[] = {1e-5, 6e-8, 1.0 / 3.0, 65504.0, 1.00390625, -2.25};
    for (double value : values) {
        Tensor filled = full({1}, value, DType::BFloat16, Device{});
        Tensor source = full({1}, value, DType::Float32, Device{});
        Tensor converted = cast(source, DType::BFloat16);
        const uint16_t expected = f32_to_bf16_bits(static_cast<float>(value));
        EXPECT_EQ(stored_bits(filled), expected) << "full " << value;
        EXPECT_EQ(stored_bits(converted), expected) << "cast " << value;
    }
}

TEST(FloatFormats, CastWidensLowPrecisionExactly) {
    for (uint32_t i = 0; i < 65536u; i += 337u) {      // sample the space
        const uint16_t bits = static_cast<uint16_t>(i);
        if ((bits & 0x7C00u) == 0x7C00u) continue;      // skip inf/NaN
        Tensor half(std::make_shared<TensorImpl>(
            std::vector<int64_t>{1}, DType::Float16, Device{}));
        static_cast<uint16_t*>(half.data())[0] = bits;
        Tensor wide = cast(half, DType::Float32);
        EXPECT_EQ(static_cast<const float*>(wide.data())[0],
                  f16_bits_to_f32(bits));
        EXPECT_EQ(stored_bits(cast(wide, DType::Float16)), bits);
    }
}
