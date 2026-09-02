#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/core/shape.h>
#include <orthon/core/strides.h>
#include <orthon/core/storage.h>
#include <orthon/autograd/node.h>

namespace orthon {

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device)
    : shape_(shape), storage_offset_(0)
{
    validate_shape(shape_);
    // Qualified so it resolves to the free function in orthon rather than the
    // member of the same name being defined here.
    const int64_t n = orthon::numel(shape_);
    const size_t  n_bytes = static_cast<size_t>(n) * dtype_size(dtype);
    storage_ = std::make_shared<Storage>(n_bytes, dtype, device);
    strides_ = compute_contiguous_strides(shape_);
}

const std::vector<int64_t>& TensorImpl::shape() const { return shape_; }
const std::vector<int64_t>& TensorImpl::strides() const { return strides_; }

int64_t TensorImpl::numel() const { return orthon::numel(shape_); }
int64_t TensorImpl::ndim() const { return static_cast<int64_t>(shape_.size()); }

DType  TensorImpl::dtype() const { return storage_->datatype(); }
Device TensorImpl::device() const { return storage_->device(); }

// storage_offset_ is in elements; convert to bytes before advancing the pointer.
void* TensorImpl::data() const {
    return static_cast<char*>(storage_->data())
         + storage_offset_ * static_cast<int64_t>(dtype_size(dtype()));
}

bool TensorImpl::is_contiguous() const {
    return orthon::is_contiguous(shape_, strides_);
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
                       std::vector<int64_t>     shape,
                       std::vector<int64_t>     strides,
                       int64_t                  storage_offset)
    : storage_(std::move(storage))
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , storage_offset_(storage_offset)
{}

std::shared_ptr<Storage> TensorImpl::storage()        const { return storage_; }
int64_t                  TensorImpl::storage_offset() const { return storage_offset_; }

bool TensorImpl::requires_grad() const { return requires_grad_; }
void TensorImpl::set_requires_grad(bool value) { requires_grad_ = value; }

bool TensorImpl::is_leaf() const { return is_leaf_; }
void TensorImpl::set_is_leaf(bool value) { is_leaf_ = value; }

bool TensorImpl::has_grad() const { return grad_ != nullptr; }

std::shared_ptr<TensorImpl> TensorImpl::grad_impl() const { return grad_; }
void TensorImpl::set_grad_impl(std::shared_ptr<TensorImpl> grad) { grad_ = std::move(grad); }

void TensorImpl::zero_grad() { grad_ = nullptr; }

void TensorImpl::add_post_accumulate_grad_hook(std::function<void()> hook) {
    post_accumulate_grad_hooks_.push_back(std::move(hook));
}

void TensorImpl::run_post_accumulate_grad_hooks() {
    for (auto& hook : post_accumulate_grad_hooks_)
        hook();
}

bool                  TensorImpl::has_grad_fn() const { return grad_fn_ != nullptr; }
std::shared_ptr<Node> TensorImpl::grad_fn()     const { return grad_fn_; }
void                  TensorImpl::set_grad_fn(std::shared_ptr<Node> fn) { grad_fn_ = std::move(fn); }
void                  TensorImpl::clear_grad_fn() { grad_fn_ = nullptr; }

}  // namespace orthon
