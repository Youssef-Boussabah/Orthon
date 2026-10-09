#pragma once
#include <orthon/core/tensor.h>
#include <vector>

namespace orthon {

// CUDA add with NumPy-style broadcasting.
// Both tensors must be contiguous float32/float16/bfloat16 of matching dtype
// on the same CUDA device (validated here; throws on violation).
// Handles shape broadcasting; throws on shape mismatch.
Tensor cuda_add(const Tensor& a, const Tensor& b);

// CUDA element-wise multiply with NumPy-style broadcasting.
// Same preconditions as cuda_add.
Tensor cuda_mul(const Tensor& a, const Tensor& b);

// Internal validated execution overloads used by registered adapters. Logical
// broadcast validation/output shape are supplied by the semantic operator;
// the two-argument compatibility entry points retain their checked contract.
Tensor cuda_add(const Tensor& a, const Tensor& b, const std::vector<int64_t>& out_shape);
Tensor cuda_mul(const Tensor& a, const Tensor& b, const std::vector<int64_t>& out_shape);

// Reduce `grad` to `target_shape` by summing over broadcast dimensions.
// Returns grad unchanged (zero-copy) when shapes already match.
// Accepts float32, float16, or bfloat16; half-precision inputs accumulate in
// float32 internally.
Tensor cuda_unbroadcast(const Tensor& grad,
                        const std::vector<int64_t>& target_shape);

// In-place: a[i] += alpha * b[i].  Both tensors must be contiguous float32
// on the same CUDA device with identical shapes (no broadcasting).
void cuda_add_(Tensor& a, const Tensor& b, float alpha);

// In-place: a[i] *= alpha.  Tensor must be contiguous float32 on CUDA.
void cuda_mul_(Tensor& a, float alpha);

}  // namespace orthon
