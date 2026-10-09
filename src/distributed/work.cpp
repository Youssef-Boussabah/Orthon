#include <orthon/distributed/work.h>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <thread>
namespace orthon {
namespace distributed {
Work::Work(CompletionToken completion, std::shared_ptr<GroupLifetime> lifetime,
           std::shared_ptr<void> storage, std::function<void()> check)
    : completion_(std::move(completion)), lifetime_(std::move(lifetime)), storage_(std::move(storage)),
      check_(std::move(check)) {}
bool Work::wait() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!poll()) std::this_thread::sleep_for(std::chrono::microseconds(50));
    return true;
}
bool Work::is_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    return poll();
}
void Work::check_status() {
    if (failure_) std::rethrow_exception(failure_);
    try { if (check_) check_(); }
    catch (...) {
        failure_ = std::current_exception();
        lifetime_->failed = true;
        throw;
    }
}
bool Work::poll() {
    if (failure_) std::rethrow_exception(failure_);
    if (completed_) return true;
    check_status();
    try {
        if (!completion_.ready()) return false;
    } catch (...) { lifetime_->failed = true; throw; }
    // Readiness alone cannot publish success: an asynchronous transport error
    // can become visible between the first status check and the event query.
    check_status();
    completed_ = true;
    completion_ = {};
    storage_.reset();
    check_ = {};
    return true;
}
void Work::complete_after_drain(std::exception_ptr failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!completed_ && !failure_) failure_ = std::move(failure);
    completed_ = !failure_;
    completion_ = {};
    storage_.reset();
    check_ = {};
}
} // namespace distributed
} // namespace orthon
