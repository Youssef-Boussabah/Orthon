#pragma once
#include <functional>
#include <cstdint>
#include <memory>
#include <vector>

namespace orthon {

class Node;
class TensorImpl;

// One per TensorImpl, never shared or copied from a view's parent.
class AutogradMeta {
public:
    enum class History { None, Live, Released };
    AutogradMeta() = default;
    AutogradMeta(const AutogradMeta&) = delete;
    AutogradMeta& operator=(const AutogradMeta&) = delete;

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool value) { requires_grad_ = value; }
    bool is_leaf() const { return is_leaf_; }
    void set_is_leaf(bool value) { is_leaf_ = value; }
    bool has_grad() const { return grad_ != nullptr; }
    std::shared_ptr<TensorImpl> grad_impl() const { return grad_; }
    void set_grad_impl(std::shared_ptr<TensorImpl> grad);
    void zero_grad();
    bool has_grad_fn() const { return grad_fn_ != nullptr; }
    std::shared_ptr<Node> grad_fn() const { return grad_fn_; }
    History history() const { return history_; }
    void set_grad_fn(std::shared_ptr<Node> fn);
    void clear_grad_fn();
    void add_post_accumulate_grad_hook(std::function<void()> hook);
    void run_post_accumulate_grad_hooks();
    using Finalizer = std::pair<uintptr_t, std::function<void()>>;
    void register_backward_finalizer(uintptr_t key, std::function<void()> callback);
    const std::vector<Finalizer>& backward_finalizers() const { return backward_finalizers_; }

private:
    bool requires_grad_ = false;
    bool is_leaf_ = true;
    std::shared_ptr<TensorImpl> grad_;
    std::shared_ptr<Node> grad_fn_;
    History history_ = History::None;
    std::vector<std::function<void()>> post_accumulate_grad_hooks_;
    std::vector<Finalizer> backward_finalizers_;
};

}  // namespace orthon
