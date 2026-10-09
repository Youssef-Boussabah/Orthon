#include <orthon/core/execution_copy.h>
#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/reshape_node.h>
#include <orthon/autograd/transpose_node.h>
#include <orthon/autograd/contiguous_node.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/shape.h>
#include <orthon/core/strides.h>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <utility>

namespace orthon {

static std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) r += ", "; r += std::to_string(s[i]); }
    return r + "]";
}

Tensor reshape(const Tensor& input, const std::vector<int64_t>& new_shape) {
    validate_shape(new_shape);
    if (numel(new_shape) != input.numel())
        throw std::invalid_argument(
            "reshape: cannot reshape tensor of shape " + shape_str(input.shape()) +
            " (" + std::to_string(input.numel()) + " elements) into shape " +
            shape_str(new_shape) + " (" + std::to_string(numel(new_shape)) + " elements)");
    if (!input.is_contiguous())
        throw std::runtime_error("reshape: only contiguous tensors supported");

    Tensor out(std::make_shared<TensorImpl>(
        input.impl()->buffer(),
        Layout(new_shape, compute_contiguous_strides(new_shape), input.impl()->storage_offset()),
        input.impl()->tensor_meta()
    ));

    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ReshapeNode>(input));
    }

    return out;
}

Tensor contiguous(const Tensor& input) {
    if (input.is_contiguous()) {
        // Already contiguous: return tensor as-is, preserving its existing grad_fn.
        return input;
    }

    // Materialize a physical contiguous copy via Tensor::to(), which reuses
    // make_contiguous_copy (CPU) or cuda_make_contiguous (CUDA) internally.
    // Explicit no-grad: public to() is differentiable, but this call is
    // contiguous()'s own materialisation step - the public graph contract for
    // contiguous(x) is exactly one ContiguousNode, attached below, never a
    // hidden ToNode underneath it.
    Tensor out = [&] {
        NoGradGuard guard;
        return input.to(input.device());
    }();

    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ContiguousNode>(input));
    }
    return out;
}

Tensor kernel_operand(const Tensor& t) { return execution_operand(t); }

Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
    const int64_t ndim = input.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
        throw std::out_of_range("transpose: dimension index out of range");

    auto new_shape   = input.shape();
    auto new_strides = input.strides();
    std::swap(new_shape[dim0],   new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    Tensor out(std::make_shared<TensorImpl>(
        input.impl()->buffer(),
        Layout(std::move(new_shape), std::move(new_strides), input.impl()->storage_offset()),
        input.impl()->tensor_meta()
    ));

    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<TransposeNode>(input, dim0, dim1));
    }

    return out;
}

}  // namespace orthon
