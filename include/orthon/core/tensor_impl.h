#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <orthon/device/device.h>
#include <orthon/core/dtype.h>
#include <orthon/core/shape.h>
#include <orthon/core/strides.h>
#include <orthon/core/buffer.h>
#include <orthon/core/layout.h>
#include <orthon/core/tensor_meta.h>
#include <orthon/core/autograd_meta.h>

namespace orthon {

class Node;  // forward-declare; full type only needed in .cpp files

class TensorImpl {
public:
    // Compatibility allocation entry point, building the canonical components.
    TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device);

    // Internal/view entry point: shared physical bytes, independent logical and
    // autograd identity. Bounds are validated against Buffer before use.
    TensorImpl(std::shared_ptr<Buffer> buffer, Layout layout, TensorMeta meta);
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;

    const std::vector<int64_t>& shape()   const;
    const std::vector<int64_t>& strides() const;

    int64_t numel()  const;
    int64_t ndim()   const;

    DType   dtype()  const;
    Device  device() const;
    void*   data()   const;

    bool is_contiguous() const;

    const std::shared_ptr<Buffer>& buffer() const;
    const Layout& layout() const;
    const TensorMeta& tensor_meta() const;
    const AutogradMeta& autograd_meta() const;
    int64_t                  storage_offset() const;

    // autograd metadata
    bool requires_grad() const;
    void set_requires_grad(bool value);
    bool is_leaf() const;
    void set_is_leaf(bool value);
    bool has_grad() const;
    std::shared_ptr<TensorImpl> grad_impl() const;
    void set_grad_impl(std::shared_ptr<TensorImpl> grad);
    void zero_grad();
    void add_post_accumulate_grad_hook(std::function<void()> hook);
    void run_post_accumulate_grad_hooks();
    void register_backward_finalizer(uintptr_t key, std::function<void()> callback);

    // computation graph
    bool                 has_grad_fn() const;
    std::shared_ptr<Node> grad_fn() const;
    void                  set_grad_fn(std::shared_ptr<Node> fn);
    void                  clear_grad_fn();

private:
    std::shared_ptr<Buffer> buffer_;
    Layout layout_;
    TensorMeta tensor_meta_;
    AutogradMeta autograd_meta_;
};

}  // namespace orthon
