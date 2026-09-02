#pragma once
#include <cstdint>

namespace orthon {

// Canonical HOST-side conversions between float32 and the two 16-bit formats.
//
// WHY THIS FILE EXISTS
// --------------------
// `full`, `cast` and `from_numpy` all narrow a value to float16 or bfloat16.
// Given a per-call-site conversion they drift apart, and each hands back a
// different bit pattern for the same Python value - truncation in one place,
// round-half-up in another, round-to-nearest-even in a third, plus a subnormal
// shift that is easy to get wrong:
//
//   1e-5   -> f16    truncating: 4.947185516357422e-06
//                    correct:    1.0013580322265625e-05
//   6e-8   -> f16    a bad subnormal shift: 0.0
//                    correct:    5.960464477539063e-08
//   65504  -> bf16   truncating: 65280.0    ties-to-even: 65536.0
//
// One definition, here, is what keeps the three entry points in agreement.
//
// THE ONE RULE
// ------------
// IEEE-754 round-to-nearest, ties-to-even, for both formats. That is what
// NumPy's float16, PyTorch's float16/bfloat16 and CUDA's __float2half /
// __float2bfloat16 intrinsics all implement, so adopting it makes the host
// agree with the device and with the reference frameworks simultaneously.
//
// CUDA device kernels keep calling the CUDA intrinsics; they are not routed
// through these host functions. Parity between the two is a TEST obligation,
// not an implementation coupling.
//
// SERIALIZATION IS DELIBERATELY NOT A CALLER. The checkpoint
// codec moves raw bytes, so a stored float16/bfloat16 payload is restored
// bit-for-bit and never re-rounded. Checkpoint fidelity is therefore
// independent of any future change to the rounding policy here.

// float32 -> IEEE 754 binary16 bit pattern. Round-to-nearest, ties-to-even.
// Handles signed zero, subnormals, the subnormal/normal boundary, overflow to
// infinity, and NaN (which stays a NaN and keeps its top payload bits).
uint16_t f32_to_f16_bits(float value);

// IEEE 754 binary16 bit pattern -> float32. Exact; every half is representable.
float f16_bits_to_f32(uint16_t bits);

// float32 -> bfloat16 bit pattern. Round-to-nearest, ties-to-even.
// bfloat16 shares float32's exponent field, so this is a mantissa rounding
// only; NaN is quieted rather than allowed to round into infinity.
uint16_t f32_to_bf16_bits(float value);

// bfloat16 bit pattern -> float32. Exact; a bfloat16 is a truncated float32.
float bf16_bits_to_f32(uint16_t bits);

}  // namespace orthon
