#pragma once
#include <atomic>
#include <functional>
#include <exception>
#include <memory>
#include <mutex>
#include <orthon/runtime/completion.h>

namespace orthon {
namespace distributed {
struct GroupLifetime {
    std::atomic<bool> closed{false}, failed{false};
};

class Work {
public:
    Work(CompletionToken completion, std::shared_ptr<GroupLifetime> lifetime,
         std::shared_ptr<void> storage = {}, std::function<void()> check = {});
    ~Work() = default; // Dropping work never performs a host wait.
    bool wait();
    bool is_completed();
    void complete_after_drain(std::exception_ptr failure = {});
private:
    std::mutex mutex_;
    CompletionToken completion_;
    std::shared_ptr<GroupLifetime> lifetime_;
    std::shared_ptr<void> storage_;
    bool completed_ = false;
    std::function<void()> check_;
    std::exception_ptr failure_;
    void check_status();
    bool poll();
};
} // namespace distributed
} // namespace orthon
