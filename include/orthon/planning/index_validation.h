#pragma once
#include <orthon/core/tensor.h>
#include <orthon/core/layout.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace orthon {

inline bool is_index_dtype(DType dtype) {
    return dtype == DType::Int32 || dtype == DType::Int64;
}

inline int64_t normalize_embedding_padding(std::optional<int64_t> padding_idx,
                                           int64_t num_embeddings) {
    if (!padding_idx) return -1;
    auto pad = *padding_idx;
    if (pad < 0) pad += num_embeddings;
    if (pad < 0 || pad >= num_embeddings)
        throw std::out_of_range("embedding: padding_idx out of range");
    return pad;
}

inline void validate_embedding_inputs(const Tensor& indices, const Tensor& weight) {
    if (!is_index_dtype(indices.dtype()))
        throw std::runtime_error("embedding: indices must be int32 or int64");
    if (weight.dtype() != DType::Float32 || weight.ndim() != 2)
        throw std::runtime_error("embedding: weight must be float32 [num_embeddings, embedding_dim]");
    if (indices.device() != weight.device())
        throw std::runtime_error("embedding: indices and weight must be on the same device");
    if (!indices.is_contiguous())
        throw std::runtime_error("embedding: indices must be contiguous");
    if (!weight.is_contiguous())
        throw std::runtime_error("embedding: weight must be contiguous");
}

inline void validate_embedding_backward(const Tensor& indices, const Tensor& grad,
                                        const std::vector<int64_t>& weight_shape) {
    if (weight_shape.size() != 2)
        throw std::runtime_error("embedding backward: weight_shape must be rank 2");
    Layout(weight_shape).payload_bytes(sizeof(float));
    if (!is_index_dtype(indices.dtype()) || grad.dtype() != DType::Float32)
        throw std::runtime_error("embedding backward: expected integer indices and float32 gradient");
    if (indices.device() != grad.device())
        throw std::runtime_error("embedding backward: device mismatch");
    if (!indices.is_contiguous() || !grad.is_contiguous())
        throw std::runtime_error("embedding backward: tensors must be contiguous");
    auto expected = indices.shape();
    expected.push_back(weight_shape[1]);
    if (grad.shape() != expected)
        throw std::runtime_error("embedding backward: grad_output shape mismatch");
}

inline void validate_nll_reduction(const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none")
        throw std::runtime_error(
            "nll_loss: reduction must be 'mean', 'sum', or 'none'");
}

inline void validate_nll_forward(const Tensor& log_probs,
                              const Tensor& targets,
                              const std::string& reduction) {
    validate_nll_reduction(reduction);
    if (log_probs.ndim() != 2)
        throw std::runtime_error("nll_loss: log_probs must be 2D [N, C]");
    if (log_probs.dtype() != DType::Float32)
        throw std::runtime_error("nll_loss: log_probs must be float32");
    if (!is_index_dtype(targets.dtype()))
        throw std::runtime_error("nll_loss: targets must be int32 or int64");
    if (targets.ndim() != 1)
        throw std::runtime_error("nll_loss: targets must be 1D [N]");
    if (log_probs.device() != targets.device())
        throw std::runtime_error("nll_loss: log_probs and targets must be on the same device");
    if (!log_probs.is_contiguous())
        throw std::runtime_error("nll_loss: log_probs must be contiguous");
    if (!targets.is_contiguous())
        throw std::runtime_error("nll_loss: targets must be contiguous");

    const int64_t N = log_probs.shape()[0];
    const int64_t C = log_probs.shape()[1];
    if (N == 0)
        throw std::runtime_error("nll_loss: empty batch");
    if (targets.shape()[0] != N)
        throw std::runtime_error(
            "nll_loss: targets length (" + std::to_string(targets.shape()[0]) +
            ") does not match batch_size (" + std::to_string(N) + ")");
    if (C <= 0)
        throw std::runtime_error("nll_loss: class dimension must be non-empty");
}

inline void validate_nll_inputs(const Tensor& log_probs, const Tensor& targets,
                                const std::string& reduction, const Tensor* grad = nullptr) {
    validate_nll_forward(log_probs, targets, reduction);
    if (!grad) return;
    if (grad->dtype() != DType::Float32)
        throw std::runtime_error("nll_loss backward: grad_output must be float32");
    if (grad->device() != log_probs.device())
        throw std::runtime_error("nll_loss backward: device mismatch");
    if (!grad->is_contiguous())
        throw std::runtime_error("nll_loss backward: grad_output must be contiguous");
    if (reduction == "none" && (grad->ndim() != 1 || grad->shape()[0] != log_probs.shape()[0]))
        throw std::runtime_error("nll_loss backward: grad_output shape mismatch for reduction='none'");
    if (reduction != "none" && grad->numel() != 1)
        throw std::runtime_error("nll_loss backward: scalar grad_output expected");

}

} // namespace orthon
