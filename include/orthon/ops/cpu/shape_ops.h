#pragma once
#include <cstdint>
#include <vector>

#include <orthon/core/tensor.h>

namespace orthon {

Tensor reshape    (const Tensor& input, const std::vector<int64_t>& new_shape);
Tensor transpose  (const Tensor& input, int64_t dim0, int64_t dim1);
Tensor contiguous (const Tensor& input);

// The one internal materialisation used to feed a contiguous-only kernel from
// a possibly-strided public operand.
//
// Returns `t` itself when it is already contiguous; otherwise a fresh
// contiguous physical copy on t's OWN device, produced under an explicit
// NoGradGuard. The result is a KERNEL INPUT, never a public value: the public
// autograd Node must still be attached to the ORIGINAL tensor, so the graph
// shape, the saved-version watches and the view backward routing all stay
// pinned to what the user actually passed and no hidden ToNode appears.
//
// It is deliberately NOT contiguous(): contiguous() is a public differentiable
// operation that attaches a ContiguousNode.
Tensor kernel_operand(const Tensor& t);

}  // namespace orthon
