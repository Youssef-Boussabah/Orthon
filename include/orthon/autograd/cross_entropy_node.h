#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/backward_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/softmax_ops.h>
#include <stdexcept>
#include <vector>
#include <cstdint>



namespace orthon {

class CrossEntropyNode : public Node {
public:
    std::vector<int64_t> targets_;

    CrossEntropyNode(const Tensor& logits, std::vector<int64_t> targets)
        : targets_(std::move(targets)) {
        inputs = {logits};
        // Backward recomputes softmax(logits) from forward-time values.
        // targets_ is an owned int64 vector copied at construction, so no
        // tensor mutation can invalidate it - nothing to watch there.
        watch_version(logits, "logits");
    }

    const char* op_name() const override { return "CrossEntropyNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        const Tensor& logits = inputs[0];

        NoGradGuard guard;
        return {cross_entropy_backward(logits, targets_, grad_output)};
    }
};

}  // namespace orthon
