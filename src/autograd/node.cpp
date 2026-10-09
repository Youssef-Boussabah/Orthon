#include <orthon/autograd/node.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/buffer.h>
#include <stdexcept>

namespace orthon {

GraphEdge GraphEdge::capture(const Tensor& input) {
    if (!input.requires_grad()) return {};
    if (auto producer = input.grad_fn())
        return {Kind::Producer, std::move(producer)};
    if (input.is_leaf()) return {Kind::Leaf, {}};
    return {Kind::Released, {}};
}

void Node::seal_input_edges() {
    if (edges_sealed_) return;
    std::vector<GraphEdge> edges;
    edges.reserve(inputs.size());
    for (const auto& input : inputs) edges.push_back(GraphEdge::capture(input));
    input_edges_ = std::move(edges);
    edges_sealed_ = true;
}

void Node::watch_version(const Tensor& t, std::string role) {
    std::shared_ptr<Buffer> buffer = t.impl()->buffer();
    const uint64_t expected = buffer->version();
    saved_versions_.push_back({std::move(buffer), expected, std::move(role)});
}

void Node::validate_saved_versions() const {
    for (const SavedTensorVersion& saved : saved_versions_) {
        const uint64_t current = saved.buffer->version();
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
