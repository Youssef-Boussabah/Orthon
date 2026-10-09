#include <orthon/planning/index_validation.h>
#include <orthon/dispatch/operator.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/float_formats.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include <orthon/backends/cpu/kernels.h>

namespace orthon { namespace backend { namespace cpu {
using namespace dispatch;
static int64_t read_index(const Tensor& indices, int64_t logical_index) {
    if (indices.dtype() == DType::Int64) {
        return static_cast<const int64_t*>(indices.data())[logical_index];
    }
    return static_cast<int64_t>(
        static_cast<const int32_t*>(indices.data())[logical_index]);
}
Tensor cpu_embedding(const Tensor& indices, const Tensor& weight, std::optional<int64_t> padding_idx) {
    validate_embedding_inputs(indices, weight);
    normalize_embedding_padding(padding_idx, weight.shape()[0]);
    const auto num_embeddings = weight.shape()[0], embedding_dim = weight.shape()[1];
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
        if (embedding_dim == 0) continue;
        const float* row = w + token * embedding_dim;
        float* dst = y + i * embedding_dim;
        std::memcpy(dst, row, static_cast<size_t>(embedding_dim) * sizeof(float));
    }

    return out;
}
Tensor cpu_embedding_backward(const Tensor& indices, const Tensor& grad_output, const std::vector<int64_t>& weight_shape, std::optional<int64_t> padding_idx) {
    validate_embedding_backward(indices, grad_output, weight_shape);
    const auto num_embeddings = weight_shape[0], embedding_dim = weight_shape[1];
    const auto pad = normalize_embedding_padding(padding_idx, num_embeddings);
    Tensor grad_weight(std::make_shared<TensorImpl>(
        weight_shape, DType::Float32, grad_output.device()));
    if (grad_weight.numel() > 0)
        std::memset(grad_weight.data(), 0,
                    static_cast<size_t>(grad_weight.numel()) * sizeof(float));

    const float* go = static_cast<const float*>(grad_output.data());
    float* gw = static_cast<float*>(grad_weight.data());
    const int64_t index_count = indices.numel();

    for (int64_t i = 0; i < index_count; ++i) {
        const int64_t token = read_index(indices, i);
        if (token < 0 || token >= num_embeddings)
            throw std::out_of_range(
                "embedding backward: index " + std::to_string(token) + " out of range");
        if (token == pad || embedding_dim == 0) continue;
        const float* src = go + i * embedding_dim;
        float* dst = gw + token * embedding_dim;
        for (int64_t j = 0; j < embedding_dim; ++j) {
            dst[j] += src[j];
        }
    }
    return grad_weight;

}

} } }
