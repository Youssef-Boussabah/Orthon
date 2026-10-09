#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <orthon/core/tensor.h>

namespace orthon {

class Buffer;
class Node;

// Forward-time topology, independent of an input Tensor's direct history.
// Only producer edges own Nodes; leaves and released histories are distinct.
struct GraphEdge {
    enum class Kind { None, Leaf, Producer, Released };
    Kind kind = Kind::None;
    std::shared_ptr<Node> producer;
    static GraphEdge capture(const Tensor& input);
};

// One saved-value dependency of a Node's backward.
//
// Records the Buffer a saved tensor lives in and the mutation generation it
// had when the Node captured it. Holding the shared_ptr keeps the counter
// alive exactly as long as the graph does; no side registry, no raw-pointer
// keys, no lifetime problem beyond the one Buffer ownership already solves.
struct SavedTensorVersion {
    std::shared_ptr<Buffer> buffer;
    uint64_t expected_version;
    std::string role;
};

class Node {
public:
    std::vector<Tensor> inputs;

    // Called once by central graph attachment, after the formula constructor.
    void seal_input_edges();
    const std::vector<GraphEdge>& input_edges() const { return input_edges_; }
    bool edges_sealed() const { return edges_sealed_; }

    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;

    // Operation name used in stale-tensor diagnostics ("MulNode", ...).
    virtual const char* op_name() const { return "Node"; }

    // Declare that backward READS the current VALUES of `t`. Call from the
    // Node constructor for every saved tensor whose payload the backward
    // formula consumes - graph inputs and privately saved tensors alike
    // (EmbeddingNode's indices, MaxNode's argmax buffer). Do not call for
    // tensors backward uses only for shape/dtype/device metadata: that is
    // the whole point of selective tracking.
    void watch_version(const Tensor& t, std::string role);

    // Throws std::runtime_error if any watched buffer was mutated after the
    // version was captured. Run once as a preflight over the whole graph and
    // again immediately before this Node's backward executes.
    void validate_saved_versions() const;

    virtual ~Node() = default;

private:
    bool edges_sealed_ = false;
    std::vector<GraphEdge> input_edges_;
    std::vector<SavedTensorVersion> saved_versions_;
};

}  // namespace orthon
