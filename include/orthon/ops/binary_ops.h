#pragma once
#include <orthon/core/broadcast.h>
#include <cstdint>
#include <vector>
#include <orthon/core/tensor.h>

namespace orthon {
class Runtime;

// Throw std::runtime_error on mismatch.
void check_same_dtype (const Tensor& a, const Tensor& b);
void check_same_device(const Tensor& a, const Tensor& b);

// Returns the broadcasted output shape (NumPy/PyTorch semantics).
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b);

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

// Subtraction and division compose add/neg and mul/reciprocal. This preserves
// broadcasting, exact-device/dtype checks and selective autograd watches.
Tensor sub(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// In-place: a += alpha * b.  Does not touch autograd metadata.
// Requires identical shapes and contiguous tensors.
// CPU supports float32/float64; CUDA supports float32.
void add_(Tensor& a, const Tensor& b, float alpha = 1.0f);

// In-place: a *= alpha.  Does not touch autograd metadata.
// Requires a contiguous tensor.
// CPU supports float32/float64; CUDA supports float32.
void mul_(Tensor& a, float alpha);

}  // namespace orthon

namespace orthon {
namespace dispatch {
class Dispatcher; enum class OperatorId;
Tensor binary_semantic(OperatorId, const Tensor&, const Tensor&, const Dispatcher&, Runtime&);
} // namespace dispatch
} // namespace orthon
