#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <orthon/device/device.h>
#include <orthon/core/dtype.h>

namespace orthon {

// Forward-declare to keep this header lightweight.
class TensorImpl;
class Node;

class Tensor {
public:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    const std::vector<int64_t>& shape()   const;
    const std::vector<int64_t>& strides() const;
    int64_t numel()         const;
    int64_t ndim()          const;
    DType   dtype()         const;
    Device  device()        const;
    void*   data()          const;
    bool    is_contiguous() const;

    // Returns a view with dimension 0 dropped (index into first dim).
    Tensor operator[](int64_t i) const;

    // Returns a view along `dim` covering [start, end).
    Tensor slice(int64_t dim, int64_t start, int64_t end) const;

    // Returns a contiguous physical copy on `target`. Same-device calls still
    // copy; host<->device and cross-GPU transfers (explicit ordinals) are
    // supported. CUDA targets throw in CPU-only builds. Differentiable for
    // floating dtypes.
    Tensor to(Device target) const;

    // Exposes implementation for ops that need to build views or access storage.
    const std::shared_ptr<TensorImpl>& impl() const;

    // autograd metadata
    bool     requires_grad() const;
    Tensor&  requires_grad_(bool value = true);
    bool     is_leaf() const;
    void     set_is_leaf(bool value);
    bool     has_grad() const;
    Tensor   grad() const;  // throws std::runtime_error if !has_grad()
    void     set_grad(const Tensor& grad);
    void     zero_grad();
    void     add_post_accumulate_grad_hook(std::function<void()> hook);
    void     register_backward_finalizer(uintptr_t key, std::function<void()> callback);

    // computation graph
    bool                  has_grad_fn() const;
    std::shared_ptr<Node> grad_fn() const;
    void                  set_grad_fn(std::shared_ptr<Node> fn);
    void                  clear_grad_fn();

    // backward pass
    void backward(bool retain_graph = false);

private:
    std::shared_ptr<TensorImpl> impl_;
};

}  // namespace orthon
