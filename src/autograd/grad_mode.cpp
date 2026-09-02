#include <orthon/autograd/grad_mode.h>

namespace orthon {

namespace {
thread_local bool grad_enabled = true;
}

bool is_grad_enabled() {
    return grad_enabled;
}

bool set_grad_enabled(bool enabled) {
    const bool previous = grad_enabled;
    grad_enabled = enabled;
    return previous;
}

GradModeGuard::GradModeGuard(bool enabled)
    : previous_(set_grad_enabled(enabled)) {}

GradModeGuard::~GradModeGuard() {
    set_grad_enabled(previous_);
}

NoGradGuard::NoGradGuard()
    : GradModeGuard(false) {}

}  // namespace orthon
