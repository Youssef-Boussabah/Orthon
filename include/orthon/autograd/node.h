#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <orthon/core/tensor.h>

namespace orthon {

class Storage;

// One saved-value dependency of a Node's backward.
//
// Records the Storage a saved tensor lives in and the mutation generation it
// had when the Node captured it. Holding the shared_ptr keeps the counter
// alive exactly as long as the graph does; no side registry, no raw-pointer
// keys, no lifetime problem beyond the one Storage ownership already solves.
struct SavedTensorVersion {
    std::shared_ptr<Storage> storage;
    uint64_t expected_version;
    std::string role;
};

class Node {
public:
    std::vector<Tensor> inputs;

    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;

    // Operation name used in stale-tensor diagnostics ("MulNode", ...).
    virtual const char* op_name() const { return "Node"; }

    // Declare that backward READS the current VALUES of `t`. Call from the
    // Node constructor for every saved tensor whose payload the backward
    // formula consumes - graph inputs and privately saved tensors alike
    // (EmbeddingNode's indices, MaxNode's argmax buffer). Do NOT call for
    // tensors backward uses only for shape/dtype/device metadata: that is
    // the whole point of selective tracking.
    void watch_version(const Tensor& t, std::string role);

    // Throws std::runtime_error if any watched storage was mutated after the
    // version was captured. Run once as a preflight over the whole graph and
    // again immediately before this Node's backward executes.
    void validate_saved_versions() const;

    virtual ~Node() = default;

private:
    std::vector<SavedTensorVersion> saved_versions_;
};

}  // namespace orthon
