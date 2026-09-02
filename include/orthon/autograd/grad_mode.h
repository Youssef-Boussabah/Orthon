#pragma once

namespace orthon {

bool is_grad_enabled();
bool set_grad_enabled(bool enabled);

class GradModeGuard {
public:
    explicit GradModeGuard(bool enabled);
    ~GradModeGuard();

    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool previous_;
};

class NoGradGuard : public GradModeGuard {
public:
    NoGradGuard();
};

}  // namespace orthon
