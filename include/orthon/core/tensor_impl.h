#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <orthon/device/device.h>
#include <orthon/core/dtype.h>
#include <orthon/core/shape.h>
#include <orthon/core/strides.h>
#include <orthon/core/storage.h>

namespace orthon {

class Node;  // forward-declare; full type only needed in .cpp files

class TensorImpl {
public:
    // Allocating constructor: creates new Storage.
    TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device);

    // View constructor: shares existing Storage (for reshape/transpose).
    TensorImpl(std::shared_ptr<Storage>  storage,
               std::vector<int64_t>      shape,
               std::vector<int64_t>      strides,
               int64_t                   storage_offset);

    const std::vector<int64_t>& shape()   const;
    const std::vector<int64_t>& strides() const;

    int64_t numel()  const;
    int64_t ndim()   const;

    DType   dtype()  const;
    Device  device() const;
    void*   data()   const;

    bool is_contiguous() const;

    std::shared_ptr<Storage> storage()        const;
    int64_t                  storage_offset() const;

    // ---- autograd metadata ----
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

    // ---- computation graph ----
    bool                 has_grad_fn() const;
    std::shared_ptr<Node> grad_fn() const;
    void                  set_grad_fn(std::shared_ptr<Node> fn);
    void                  clear_grad_fn();

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t>     shape_;
    std::vector<int64_t>     strides_;
    int64_t                  storage_offset_;

    bool requires_grad_ = false;
    bool is_leaf_       = true;
    std::shared_ptr<TensorImpl> grad_     = nullptr;
    std::shared_ptr<Node>       grad_fn_  = nullptr;
    std::vector<std::function<void()>> post_accumulate_grad_hooks_;
};

}  // namespace orthon
