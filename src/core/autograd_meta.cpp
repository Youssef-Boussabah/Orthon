#include <orthon/core/autograd_meta.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/autograd/node.h>
#include <utility>

namespace orthon {

void AutogradMeta::set_grad_impl(std::shared_ptr<TensorImpl> grad) { grad_ = std::move(grad); }
void AutogradMeta::zero_grad() { grad_.reset(); }
void AutogradMeta::set_grad_fn(std::shared_ptr<Node> fn) {
    if (!fn) { clear_grad_fn(); return; }
    fn->seal_input_edges();
    grad_fn_ = std::move(fn);
    history_ = History::Live;
}
void AutogradMeta::clear_grad_fn() {
    if (history_ == History::Live) history_ = History::Released;
    grad_fn_.reset();
}
void AutogradMeta::add_post_accumulate_grad_hook(std::function<void()> hook) {
    post_accumulate_grad_hooks_.push_back(std::move(hook));
}
void AutogradMeta::run_post_accumulate_grad_hooks() {
    for (auto& hook : post_accumulate_grad_hooks_) hook();
}
void AutogradMeta::register_backward_finalizer(uintptr_t key, std::function<void()> callback) {
    for (auto& entry : backward_finalizers_) {
        if (entry.first == key) { entry.second = std::move(callback); return; }
    }
    backward_finalizers_.emplace_back(key, std::move(callback));
}

}  // namespace orthon
