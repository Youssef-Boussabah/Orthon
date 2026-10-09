#pragma once
#include <orthon/core/tensor.h>

namespace orthon {

// The Tensor supplies scalar dtype and exact device; no promotion occurs.
// Rank-0 constants use O(1) storage and existing broadcasting. Internal CUDA
// materialization runs under NoGradGuard.

// Rank-0 constant carrying `reference`'s dtype and exact Device.
//
// Internal native constant construction; Python uses public scalar operators.
Tensor scalar_like(const Tensor& reference, double value);

// The scalar forward with NO autograd node attached. Used by ScalarMulNode's
// backward, which must not build a graph regardless of ambient grad mode.
Tensor scalar_mul_untracked(const Tensor& t, double value);

// Public scalar arithmetic. Each follows normal grad mode: a tracked input
// under grad mode produces a tracked output, and everything inside no_grad()
// stays graph-free.
Tensor scalar_add (const Tensor& t, double value);   // t + s   /   s + t
Tensor scalar_mul (const Tensor& t, double value);   // t * s   /   s * t
Tensor scalar_sub (const Tensor& t, double value);   // t - s
Tensor scalar_rsub(double value, const Tensor& t);   // s - t
Tensor scalar_div (const Tensor& t, double value);   // t / s
Tensor scalar_rdiv(double value, const Tensor& t);   // s / t

}  // namespace orthon
