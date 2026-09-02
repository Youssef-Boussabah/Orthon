#pragma once
#include <orthon/autograd/node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/cpu/softmax_ops.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/cross_entropy_kernels.h>
#endif

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

#ifdef WITH_CUDA
        if (logits.device().is_cuda())
            return {cuda_cross_entropy_backward(logits, targets_, grad_output)};
#endif

        if (!logits.device().is_cpu())
            throw std::runtime_error(
                "CrossEntropyNode: backward only supported on CPU and CUDA");

        const float g = static_cast<const float*>(grad_output.data())[0];

        const int64_t B   = logits.shape()[0];
        const int64_t C   = logits.shape()[1];
        const float inv_B = 1.0f / static_cast<float>(B);

        // Numerically-stable softmax probabilities in a fresh contiguous tensor.
        //
        // Tensor::backward already holds a structural NoGradGuard for the whole
        // traversal; this local guard keeps the invariant if this backward is
        // ever invoked directly. `logits` requires grad by definition here, so
        // an unguarded softmax() would attach a SoftmaxNode to `probs` - which
        // is mutated in place below and returned AS the gradient. The gradient
        // would then carry a graph, which is exactly the higher-order autograd
        // this engine does not implement.
        NoGradGuard guard;
        Tensor probs = softmax(logits, -1);
        float* p = static_cast<float*>(probs.data());

        // grad[i,j] = g * (softmax[i,j] - one_hot[i,j]) / B
        for (int64_t i = 0; i < B; ++i) {
            for (int64_t j = 0; j < C; ++j) {
                const float oh = (j == targets_[i]) ? 1.0f : 0.0f;
                p[i * C + j] = g * (p[i * C + j] - oh) * inv_B;
            }
        }

        return {probs};
    }
};

}  // namespace orthon
