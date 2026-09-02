#include <orthon/ops/cpu/embedding_ops.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/embedding_node.h>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/embedding_kernels.h>
#endif

namespace orthon {

static bool is_index_dtype(DType dtype) {
    return dtype == DType::Int32 || dtype == DType::Int64;
}

static int64_t normalize_padding_idx(std::optional<int64_t> padding_idx,
                                     int64_t num_embeddings) {
    if (!padding_idx.has_value()) return -1;
    int64_t pad = *padding_idx;
    if (pad < 0) pad += num_embeddings;
    if (pad < 0 || pad >= num_embeddings)
        throw std::out_of_range("embedding: padding_idx out of range");
    return pad;
}

static int64_t read_index(const Tensor& indices, int64_t logical_index) {
    if (indices.dtype() == DType::Int64) {
        return static_cast<const int64_t*>(indices.data())[logical_index];
    }
    return static_cast<int64_t>(
        static_cast<const int32_t*>(indices.data())[logical_index]);
}

static void validate_embedding_inputs(const Tensor& indices,
                                      const Tensor& weight) {
    if (!is_index_dtype(indices.dtype()))
        throw std::runtime_error("embedding: indices must be int32 or int64");
    if (weight.dtype() != DType::Float32)
        throw std::runtime_error("embedding: weight must be float32");
    if (weight.ndim() != 2)
        throw std::runtime_error("embedding: weight must have shape [num_embeddings, embedding_dim]");
    if (indices.device() != weight.device())
        throw std::runtime_error("embedding: indices and weight must be on the same device");
    // Both preconditions are enforced identically on CPU and CUDA. Enforcing
    // them on only one device produces wrong answers rather than errors:
    //
    // indices: the CUDA kernel reads indices[token_pos] linearly, so a
    // strided index tensor would be accepted and silently wrong (a
    // transposed 2x3 index view yields rows [0,1,2,3,0,1] instead of
    // [0,3,1,0,2,1]). Checked here so both devices reject identically.
    //
    // weight: read as w[token * embedding_dim + j] by the CPU loop AND by the
    // CUDA kernel, i.e. both assume a row-major layout that a view need not
    // have. Neither device checked, so both returned storage-order rows for a
    // strided weight. This is a shared, previously undocumented precondition,
    // now stated once - it makes an existing silent-wrong path loud on both
    // devices rather than widening any contract.
    if (!indices.is_contiguous())
        throw std::runtime_error("embedding: indices must be contiguous");
    if (!weight.is_contiguous())
        throw std::runtime_error("embedding: weight must be contiguous");
}

Tensor embedding(const Tensor& indices,
                 const Tensor& weight,
                 std::optional<int64_t> padding_idx) {
    validate_embedding_inputs(indices, weight);
    const int64_t num_embeddings = weight.shape()[0];
    const int64_t embedding_dim = weight.shape()[1];
    const int64_t pad = normalize_padding_idx(padding_idx, num_embeddings);

#ifdef WITH_CUDA
    if (weight.device().is_cuda()) {
        Tensor out = cuda_embedding_forward(indices, weight, pad >= 0 ? std::optional<int64_t>(pad)
                                                                      : std::nullopt);
        if (is_grad_enabled() && weight.requires_grad()) {
            out.requires_grad_(true);
            out.set_is_leaf(false);
            out.set_grad_fn(std::make_shared<EmbeddingNode>(
                indices, weight, pad >= 0 ? std::optional<int64_t>(pad) : std::nullopt));
        }
        return out;
    }
#endif
    if (!indices.device().is_cpu())
        throw std::runtime_error("embedding: CUDA support not compiled");

    std::vector<int64_t> out_shape = indices.shape();
    out_shape.push_back(embedding_dim);
    Tensor out(std::make_shared<TensorImpl>(out_shape, DType::Float32, weight.device()));
    const float* w = static_cast<const float*>(weight.data());
    float* y = static_cast<float*>(out.data());
    const int64_t index_count = indices.numel();

    for (int64_t i = 0; i < index_count; ++i) {
        const int64_t token = read_index(indices, i);
        if (token < 0 || token >= num_embeddings)
            throw std::out_of_range(
                "embedding: index " + std::to_string(token) + " out of range");
        const float* row = w + token * embedding_dim;
        float* dst = y + i * embedding_dim;
        std::memcpy(dst, row, static_cast<size_t>(embedding_dim) * sizeof(float));
    }

    if (is_grad_enabled() && weight.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<EmbeddingNode>(
            indices, weight, pad >= 0 ? std::optional<int64_t>(pad) : std::nullopt));
    }
    return out;
}

Tensor embedding_backward(const Tensor& indices,
                          const Tensor& grad_output,
                          const std::vector<int64_t>& weight_shape,
                          std::optional<int64_t> padding_idx) {
    if (weight_shape.size() != 2)
        throw std::runtime_error("embedding backward: weight_shape must be rank 2");
    const int64_t num_embeddings = weight_shape[0];
    const int64_t embedding_dim = weight_shape[1];
    const int64_t pad = normalize_padding_idx(padding_idx, num_embeddings);
    // Same hoist as the forward: checked here so both devices reject strided
    // indices identically (the CUDA scatter kernel would otherwise read them in
    // storage order). EmbeddingNode already normalises grad_output at the Node
    // boundary, so only indices needs a check.
    if (!indices.is_contiguous())
        throw std::runtime_error("embedding backward: indices must be contiguous");

#ifdef WITH_CUDA
    if (grad_output.device().is_cuda()) {
        return cuda_embedding_backward(indices, grad_output, weight_shape,
                                       pad >= 0 ? std::optional<int64_t>(pad)
                                                : std::nullopt);
    }
#endif
    if (!grad_output.device().is_cpu())
        throw std::runtime_error("embedding backward: CUDA support not compiled");

    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight_shape, DType::Float32, grad_output.device()));
    std::memset(grad_weight.data(), 0,
                static_cast<size_t>(grad_weight.numel()) * sizeof(float));

    const float* go = static_cast<const float*>(grad_output.data());
    float* gw = static_cast<float*>(grad_weight.data());
    const int64_t index_count = indices.numel();

    for (int64_t i = 0; i < index_count; ++i) {
        const int64_t token = read_index(indices, i);
        if (token == pad) continue;
        if (token < 0 || token >= num_embeddings)
            throw std::out_of_range(
                "embedding backward: index " + std::to_string(token) + " out of range");
        const float* src = go + i * embedding_dim;
        float* dst = gw + token * embedding_dim;
        for (int64_t j = 0; j < embedding_dim; ++j) {
            dst[j] += src[j];
        }
    }
    return grad_weight;
}

}  // namespace orthon
