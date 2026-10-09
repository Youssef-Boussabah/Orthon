#pragma once

namespace orthon {
class Tensor;

// Stateless, synchronous first-order execution. Each call owns a private task.
class AutogradEngine {
public:
    void backward(const Tensor& root, bool retain_graph = false) const;
};

const AutogradEngine& default_autograd_engine();
}  // namespace orthon
