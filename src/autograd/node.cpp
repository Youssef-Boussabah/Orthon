#include <orthon/autograd/node.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/storage.h>
#include <stdexcept>

namespace orthon {

void Node::watch_version(const Tensor& t, std::string role) {
    std::shared_ptr<Storage> storage = t.impl()->storage();
    const uint64_t expected = storage->version();
    saved_versions_.push_back({std::move(storage), expected, std::move(role)});
}

void Node::validate_saved_versions() const {
    for (const SavedTensorVersion& saved : saved_versions_) {
        const uint64_t current = saved.storage->version();
        if (current != saved.expected_version) {
            throw std::runtime_error(
                std::string("backward: tensor saved by ") + op_name() +
                " as '" + saved.role +
                "' was modified in-place after it was saved for backward"
                " (expected version " + std::to_string(saved.expected_version) +
                ", found " + std::to_string(current) + ")");
        }
    }
}

}  // namespace orthon
