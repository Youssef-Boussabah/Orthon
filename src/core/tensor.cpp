#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/core/tensor.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/dtype.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/autograd/engine.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/index_node.h>
#include <orthon/autograd/slice_node.h>
#include <orthon/autograd/to_node.h>
#include <cstring>
#include <stdexcept>
#include <vector>


namespace orthon {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

const std::vector<int64_t>& Tensor::shape()   const { return impl_->shape(); }
const std::vector<int64_t>& Tensor::strides() const { return impl_->strides(); }
int64_t Tensor::numel()         const { return impl_->numel(); }
int64_t Tensor::ndim()          const { return impl_->ndim(); }
DType   Tensor::dtype()         const { return impl_->dtype(); }
Device  Tensor::device()        const { return impl_->device(); }
void*   Tensor::data()          const { return impl_->data(); }
bool    Tensor::is_contiguous() const { return impl_->is_contiguous(); }

Tensor Tensor::operator[](int64_t i) const {
    if (impl_->ndim() == 0)
        throw std::runtime_error("operator[]: cannot index a 0-d tensor");
    const auto& sh = impl_->shape();
    const auto& st = impl_->strides();
    if (i < 0) i += sh[0];
    if (i < 0 || i >= sh[0])
        throw std::out_of_range("operator[]: index out of range");

    int64_t new_offset = impl_->layout().shifted_offset(i, st[0]);
    std::vector<int64_t> new_shape (sh.begin() + 1, sh.end());
    std::vector<int64_t> new_strides(st.begin() + 1, st.end());
    Tensor out(std::make_shared<TensorImpl>(
        impl_->buffer(), Layout(std::move(new_shape), std::move(new_strides), new_offset),
        impl_->tensor_meta()));

    // Graph metadata is the only thing autograd adds here: `out` keeps sharing
    // this tensor's storage, shape, strides and offset either way. `i` has been
    // normalised above, so IndexNode never re-resolves a negative subscript.
    if (is_grad_enabled() && requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<IndexNode>(*this, i));
    }

    return out;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
    const int64_t nd = impl_->ndim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd)
        throw std::out_of_range("slice: dim out of range");

    const auto& sh = impl_->shape();
    const auto& st = impl_->strides();
    if (start < 0) start += sh[dim];
    if (end   < 0) end   += sh[dim];
    if (start < 0 || end > sh[dim] || start > end)
        throw std::out_of_range("slice: invalid start/end");

    int64_t new_offset = impl_->layout().shifted_offset(start, st[dim]);
    std::vector<int64_t> new_shape  = sh;
    std::vector<int64_t> new_strides = st;
    new_shape[dim] = end - start;
    Tensor out(std::make_shared<TensorImpl>(
        impl_->buffer(), Layout(std::move(new_shape), std::move(new_strides), new_offset),
        impl_->tensor_meta()));

    // Graph metadata is the only thing autograd adds here: `out` keeps sharing
    // this tensor's storage, and keeps the same shape, strides and offset either
    // way. dim/start/end have all been normalised above, so SliceNode stores
    // resolved values and never re-interprets a negative argument at backward.
    if (is_grad_enabled() && requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<SliceNode>(*this, dim, start, end));
    }

    return out;
}

// Linear (contiguous) copy of this tensor on the same device.





Tensor Tensor::to(Device target) const {
    // Device values are mutable for native consumers; validate the explicit
    // target before dispatch, without allocating or changing current device.
    Device(target.type, target.index);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TransferKernel>(
        transfer_request(*this, target), default_runtime());
    Tensor out = kernel.call(*this, target);

    // Public to() participates in autograd. The forward stays the physical
    // copy above; only graph metadata is added, and only when it means
    // something: grad mode on, the source tracked, and a FLOATING dtype
    // (integer/bool transfers never join a graph, matching the cast
    // contract). Under NoGradGuard - which includes every internal
    // materialisation that wraps itself in one, and all of Tensor::backward -
    // this attaches nothing, which is what internal snapshot/detach call
    // sites rely on EXPLICITLY rather than on a detach side effect.
    if (is_grad_enabled() && requires_grad() && dtype_is_floating(dtype())) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ToNode>(*this));
    }
    return out;
}

const std::shared_ptr<TensorImpl>& Tensor::impl() const { return impl_; }

bool    Tensor::requires_grad() const { return impl_->requires_grad(); }
Tensor& Tensor::requires_grad_(bool value) {
    impl_->set_requires_grad(value);
    return *this;
}
bool Tensor::is_leaf()  const { return impl_->is_leaf(); }
void Tensor::set_is_leaf(bool value) { impl_->set_is_leaf(value); }
bool Tensor::has_grad() const { return impl_->has_grad(); }

Tensor Tensor::grad() const {
    if (!impl_->has_grad())
        throw std::runtime_error("grad(): tensor has no gradient");
    return Tensor(impl_->grad_impl());
}

void Tensor::set_grad(const Tensor& grad) {
    impl_->set_grad_impl(grad.impl());
}

void Tensor::zero_grad() { impl_->zero_grad(); }

void Tensor::add_post_accumulate_grad_hook(std::function<void()> hook) {
    impl_->add_post_accumulate_grad_hook(std::move(hook));
}
void Tensor::register_backward_finalizer(uintptr_t key, std::function<void()> callback) {
    impl_->register_backward_finalizer(key, std::move(callback));
}

bool                  Tensor::has_grad_fn() const { return impl_->has_grad_fn(); }
std::shared_ptr<Node> Tensor::grad_fn()     const { return impl_->grad_fn(); }
void                  Tensor::set_grad_fn(std::shared_ptr<Node> fn) { impl_->set_grad_fn(std::move(fn)); }
void                  Tensor::clear_grad_fn() { impl_->clear_grad_fn(); }

void Tensor::backward(bool retain_graph) {
    default_autograd_engine().backward(*this, retain_graph);
}

}  // namespace orthon
