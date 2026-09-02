#pragma once
#include <cstdint>
#include <vector>
#include <orthon/core/tensor.h>

namespace orthon {

// Throw std::runtime_error on mismatch.
void check_same_dtype (const Tensor& a, const Tensor& b);
void check_same_device(const Tensor& a, const Tensor& b);

// Returns the broadcasted output shape (NumPy/PyTorch semantics).
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b);

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

// Both are COMPOSITIONS of ops that are already correct on every device and
// dtype the runtime supports, not new kernels:
//
//     sub(a, b) = add(a, neg(b))
//     div(a, b) = mul(a, reciprocal(b))
//
// A dedicated subtraction or division kernel would buy nothing. Composition
// is not just cheaper to maintain, it inherits - for free and by construction
// - broadcasting, strided/view operands on CPU and CUDA, the float16/bfloat16
// CUDA paths, and the correct derivative chain (NegNode / Reciprocal
// ActivationNode composed with AddNode / MulNode). Saved-version safety also
// lands in the right place automatically: reciprocal is value-sensitive and
// watches its operand, add and neg are not and do not.
//
// Validation is inherited too, so the existing device-mismatch, dtype-mismatch
// and non-broadcastable-shape errors are raised unchanged.
Tensor sub(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// In-place: a += alpha * b.  Does NOT touch autograd metadata.
// Requires identical shapes and contiguous tensors.
// CPU supports float32/float64; CUDA supports float32.
void add_(Tensor& a, const Tensor& b, float alpha = 1.0f);

// In-place: a *= alpha.  Does NOT touch autograd metadata.
// Requires a contiguous tensor.
// CPU supports float32/float64; CUDA supports float32.
void mul_(Tensor& a, float alpha);

}  // namespace orthon
