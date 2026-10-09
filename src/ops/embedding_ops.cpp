#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/embedding_ops.h>
#include <orthon/planning/index_validation.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/embedding_node.h>
#include <cstring>
#include <stdexcept>
#include <string>


namespace orthon {

Tensor embedding(const Tensor& indices,
                 const Tensor& weight,
                 std::optional<int64_t> padding_idx) {
    validate_embedding_inputs(indices, weight);
    const auto pad = normalize_embedding_padding(padding_idx, weight.shape()[0]);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<EmbeddingKernel>(unary_request(OperatorId::Embedding, weight), default_runtime());
    Tensor out = kernel.call(indices, weight, pad >= 0 ? std::optional<int64_t>(pad) : std::nullopt);
    if (is_grad_enabled() && weight.requires_grad()) {
        out.requires_grad_(true); out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<EmbeddingNode>(indices, weight, pad >= 0 ? std::optional<int64_t>(pad) : std::nullopt));
    }
    return out;
}
Tensor embedding_backward(const Tensor& indices,
                          const Tensor& grad_output,
                          const std::vector<int64_t>& weight_shape,
                          std::optional<int64_t> padding_idx) {
    validate_embedding_backward(indices, grad_output, weight_shape);
    const auto pad = normalize_embedding_padding(padding_idx, weight_shape[0]);

    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<EmbeddingBackwardKernel>(unary_request(OperatorId::EmbeddingBackward, grad_output), default_runtime());
    return kernel.call(indices, grad_output, weight_shape, pad >= 0 ? std::optional<int64_t>(pad) : std::nullopt);
}
} // namespace orthon
