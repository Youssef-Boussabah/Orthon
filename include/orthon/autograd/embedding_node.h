#pragma once

#include <optional>
#include <vector>

#include <orthon/autograd/node.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/embedding_ops.h>

namespace orthon {

class EmbeddingNode : public Node {
public:
    Tensor indices_;
    std::vector<int64_t> weight_shape_;
    std::optional<int64_t> padding_idx_;

    EmbeddingNode(const Tensor& indices,
                  const Tensor& weight,
                  std::optional<int64_t> padding_idx)
        : indices_(indices)
        , weight_shape_(weight.shape())
        , padding_idx_(padding_idx) {
        inputs = {weight};
        // grad_weight scatters grad_output rows by the saved indices' VALUES.
        // indices_ is not a graph input, which is exactly why saved-version
        // tracking cannot be limited to Node::inputs. The weight payload is
        // never read (only weight_shape_, copied above), so weight is
        // deliberately not watched.
        watch_version(indices_, "indices");
    }

    const char* op_name() const override { return "EmbeddingNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        return {embedding_backward(indices_, grad, weight_shape_, padding_idx_)};
    }
};

}  // namespace orthon
