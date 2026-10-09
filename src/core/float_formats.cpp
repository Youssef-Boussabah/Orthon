#include <orthon/core/float_formats.h>
#include <cstring>

namespace orthon {

// See float_formats.h for why this is the single canonical implementation.

uint16_t f32_to_f16_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;   // already in half position
    const uint32_t mag  = x & 0x7FFFFFFFu;       // magnitude, sign stripped

    // Inf / NaN
    if (mag >= 0x7F800000u) {
        if (mag > 0x7F800000u) {
            // NaN. Keep the top payload bits so a NaN stays distinguishable,
            // and force at least one mantissa bit so it can never collapse
            // into an infinity.
            uint32_t payload = (mag >> 13) & 0x03FFu;
            if (payload == 0u) payload = 1u;
            return static_cast<uint16_t>(sign | 0x7C00u | payload);
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Magnitudes at or above 2^16 always overflow
    //
    // Not merely an optimisation: the normal path below subtracts a bias and
    // shifts, and for a float32 this large the result would run past bit 15
    // and corrupt the sign. Values just under 2^16 are not short-circuited
    // here - 65504 must stay finite, and 65520 must round up to infinity -
    // so that boundary is left to the rounding step, which handles it by
    // carrying the mantissa into the exponent.
    if (mag >= 0x47800000u)
        return static_cast<uint16_t>(sign | 0x7C00u);

    // Normal half range: |x| >= 2^-14
    if (mag >= 0x38800000u) {
        // Re-bias 127 -> 15 and drop 13 mantissa bits in one step. The
        // addition wraps modulo 2^32, which is exactly mag - (112 << 23);
        // that subtrahend has 13 zero low bits, so the shift stays exact.
        uint32_t bits = (mag + 0xC8000000u) >> 13;
        const uint32_t rem = mag & 0x1FFFu;          // the 13 discarded bits
        if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u)))
            ++bits;                                   // may carry into the
                                                      // exponent, and at the
                                                      // top into infinity
        return static_cast<uint16_t>(sign | bits);
    }

    // Below the smallest representable magnitude
    // The smallest half subnormal is 2^-24; anything under 2^-25 rounds to
    // zero, and exactly 2^-25 is a tie that ties-to-even also sends to zero
    // (handled by the subnormal path below, which is why the cutoff is
    // strictly less-than).
    if (mag < 0x33000000u)
        return static_cast<uint16_t>(sign);

    // Subnormal half
    // A half subnormal is k * 2^-24 for integer k in [0, 1023]. With the
    // implicit leading 1 restored, k = mant24 >> (126 - biased_exp32).
    const uint32_t exp   = mag >> 23;                       // 102..112
    const uint32_t shift = 126u - exp;                      // 14..24
    const uint32_t mant  = (mag & 0x7FFFFFu) | 0x800000u;   // implicit 1
    uint32_t bits = mant >> shift;
    const uint32_t rem  = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (bits & 1u)))
        ++bits;   // k == 1024 is correct: it becomes the smallest normal
    return static_cast<uint16_t>(sign | bits);
}

float f16_bits_to_f32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
    const uint32_t exp5 = (bits >> 10) & 0x1Fu;
    const uint32_t mant = bits & 0x3FFu;
    uint32_t out;
    if (exp5 == 0u) {
        if (mant == 0u) {
            out = sign;                                  // +/-0
        } else {
            // Subnormal half -> normalised float: shift the mantissa up until
            // the implicit bit appears, decrementing the exponent to match.
            uint32_t m = mant, e = 127u - 14u;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            out = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp5 == 31u) {
        out = sign | 0x7F800000u | (mant << 13);         // +/-inf or NaN
    } else {
        out = sign | ((exp5 + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t f32_to_bf16_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));

    // NaN is handled before rounding: adding the rounding constant to a NaN
    // whose payload is all ones would carry into the exponent and produce an
    // infinity. Quieting it keeps it a NaN under every payload.
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x007FFFFFu) != 0u)
        return static_cast<uint16_t>((x >> 16) | 0x0040u);

    // Round-to-nearest, ties-to-even on the 16 discarded bits. The tie bias
    // depends on the low bit of the surviving mantissa, which is what makes
    // this ties-to-EVEN rather than the ties-away that a fixed +0x8000 gives.
    // A carry out of the mantissa correctly walks into the exponent, and at
    // the top into infinity - which is why the largest finite floats become
    // bfloat16 infinity, matching NumPy and PyTorch.
    const uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<uint16_t>((x + rounding) >> 16);
}

float bf16_bits_to_f32(uint16_t bits) {
    const uint32_t out = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

}  // namespace orthon
