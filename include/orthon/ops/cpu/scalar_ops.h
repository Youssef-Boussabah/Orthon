#pragma once
#include <orthon/core/tensor.h>

namespace orthon {

// Arithmetic between a Tensor and a Python numeric scalar.
//
// SCALAR SEMANTIC CONTRACT
// ------------------------
// The TENSOR is authoritative. A Python int/float is coerced to the tensor's
// own dtype and its exact Device - including cuda:N - and nothing else happens:
// there is no NumPy/PyTorch-style promotion lattice here. A float32 tensor is
// never promoted to float64 just because a Python float is a C double, and
// integer-tensor arithmetic is not widened either.
//
// SCALAR REPRESENTATION
// ---------------------
// The scalar becomes a RANK-0 tensor (shape == []) and rides the existing
// broadcasting machinery. It is emphatically NOT full(list(tensor.shape), v):
// storage is O(1) rather than O(tensor.numel()), which is what makes the
// optimizer cleanup worth doing at all - Adam used to allocate two
// PARAMETER-SHAPED constants per parameter per step.
//
// The scalar tensor is untracked (requires_grad false, no grad_fn, leaf) and,
// when it has to cross to a CUDA device, is built under NoGradGuard.

// Rank-0 constant carrying `reference`'s dtype and exact Device.
//
// Kept out of the public Python API (bound only as the underscore-private
// diagnostic `orthon._scalar_like`, which exists so tests can pin the
// contract above rather than infer it).
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
