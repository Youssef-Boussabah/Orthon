#include <orthon/core/tensor_impl.h>
#include <stdexcept>
#include <utility>

namespace orthon {

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device)
    : layout_(shape), tensor_meta_(dtype) {
    // All shape/stride/byte arithmetic must succeed BEFORE any allocation.
    const size_t bytes = layout_.payload_bytes(dtype_size(dtype));
    layout_.validate_bounds(bytes, dtype_size(dtype));
    buffer_ = std::make_shared<Buffer>(bytes, device);
}

TensorImpl::TensorImpl(std::shared_ptr<Buffer> buffer, Layout layout, TensorMeta meta)
    : buffer_(std::move(buffer)), layout_(std::move(layout)), tensor_meta_(meta) {
    if (!buffer_) throw std::invalid_argument("TensorImpl: Buffer must not be null");
    layout_.validate_bounds(buffer_->nbytes(), dtype_size(dtype()));
}

const std::vector<int64_t>& TensorImpl::shape() const { return layout_.shape(); }
const std::vector<int64_t>& TensorImpl::strides() const { return layout_.strides(); }
int64_t TensorImpl::numel() const { return layout_.numel(); }
int64_t TensorImpl::ndim() const { return layout_.ndim(); }
DType TensorImpl::dtype() const { return tensor_meta_.dtype(); }
Device TensorImpl::device() const { return buffer_->device(); }
bool TensorImpl::is_contiguous() const { return layout_.is_contiguous(); }
const std::shared_ptr<Buffer>& TensorImpl::buffer() const { return buffer_; }
const Layout& TensorImpl::layout() const { return layout_; }
const TensorMeta& TensorImpl::tensor_meta() const { return tensor_meta_; }
const AutogradMeta& TensorImpl::autograd_meta() const { return autograd_meta_; }
int64_t TensorImpl::storage_offset() const { return layout_.storage_offset(); }

void* TensorImpl::data() const {
    void* base = buffer_->data();
    const size_t offset = layout_.offset_bytes(dtype_size(dtype()));
    // Empty views can have a logical endpoint beyond the allocation (e.g. a
    // sliced column followed by an empty row slice). Preserve metadata without
    // forming an out-of-object pointer or doing arithmetic on nullptr.
    if (!base || (numel() == 0 && offset > buffer_->nbytes())) return nullptr;
    return static_cast<char*>(base) + offset;
}

bool TensorImpl::requires_grad() const { return autograd_meta_.requires_grad(); }
void TensorImpl::set_requires_grad(bool value) { autograd_meta_.set_requires_grad(value); }
bool TensorImpl::is_leaf() const { return autograd_meta_.is_leaf(); }
void TensorImpl::set_is_leaf(bool value) { autograd_meta_.set_is_leaf(value); }
bool TensorImpl::has_grad() const { return autograd_meta_.has_grad(); }
std::shared_ptr<TensorImpl> TensorImpl::grad_impl() const { return autograd_meta_.grad_impl(); }
void TensorImpl::set_grad_impl(std::shared_ptr<TensorImpl> grad) {
    autograd_meta_.set_grad_impl(std::move(grad));
}
void TensorImpl::zero_grad() { autograd_meta_.zero_grad(); }
void TensorImpl::add_post_accumulate_grad_hook(std::function<void()> hook) {
    autograd_meta_.add_post_accumulate_grad_hook(std::move(hook));
}
void TensorImpl::register_backward_finalizer(uintptr_t key, std::function<void()> callback) {
    autograd_meta_.register_backward_finalizer(key, std::move(callback));
}
void TensorImpl::run_post_accumulate_grad_hooks() { autograd_meta_.run_post_accumulate_grad_hooks(); }
bool TensorImpl::has_grad_fn() const { return autograd_meta_.has_grad_fn(); }
std::shared_ptr<Node> TensorImpl::grad_fn() const { return autograd_meta_.grad_fn(); }
void TensorImpl::set_grad_fn(std::shared_ptr<Node> fn) { autograd_meta_.set_grad_fn(std::move(fn)); }
void TensorImpl::clear_grad_fn() { autograd_meta_.clear_grad_fn(); }

}  // namespace orthon
