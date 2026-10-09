#pragma once
#include <cstdint>

namespace orthon {

// Host Float16/BFloat16 conversions share IEEE-754 round-to-nearest,
// ties-to-even rules across full(), cast() and from_numpy(). CUDA kernels
// use native intrinsics; tests establish host/device parity. Checkpoints
// copy raw bytes and never re-round stored payloads.

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
