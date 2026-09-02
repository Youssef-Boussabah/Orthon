#pragma once

#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/nll_loss_ops.h>
#include <string>
#include <vector>

namespace orthon {

class NllLossNode : public Node {
public:
    std::string reduction_;

    NllLossNode(const Tensor& log_probs,
                const Tensor& targets,
                std::string reduction)
        : reduction_(std::move(reduction)) {
        inputs = {log_probs, targets};
        // nll_loss_backward writes -g/N at [i, targets[i]]: it reads the
        // targets' VALUES but touches log_probs only for shape/dtype/device
        // (both CPU and CUDA paths), so log_probs is deliberately not watched.
        watch_version(targets, "targets");
    }

    const char* op_name() const override { return "NllLossNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        return {nll_loss_backward(inputs[0], inputs[1], grad_output, reduction_)};
    }
};

}  // namespace orthon
