#include <orthon/ops/cpu/nll_loss_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/nll_loss_node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/nll_loss_kernels.h>
#endif

namespace orthon {

static bool is_index_dtype(DType dtype) {
    return dtype == DType::Int32 || dtype == DType::Int64;
}

static int64_t read_target(const Tensor& targets, int64_t i) {
    if (targets.dtype() == DType::Int64)
        return static_cast<const int64_t*>(targets.data())[i];
    return static_cast<int64_t>(static_cast<const int32_t*>(targets.data())[i]);
}

static void validate_reduction(const std::string& reduction) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none")
        throw std::runtime_error(
            "nll_loss: reduction must be 'mean', 'sum', or 'none'");
}

static void validate_nll_loss(const Tensor& log_probs,
                              const Tensor& targets,
                              const std::string& reduction) {
    validate_reduction(reduction);
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

static void validate_targets_cpu(const Tensor& log_probs, const Tensor& targets) {
    const int64_t N = log_probs.shape()[0];
    const int64_t C = log_probs.shape()[1];
    for (int64_t i = 0; i < N; ++i) {
        const int64_t target = read_target(targets, i);
        if (target < 0 || target >= C)
            throw std::runtime_error(
                "nll_loss: target " + std::to_string(target) +
                " at index " + std::to_string(i) +
                " is out of range [0, " + std::to_string(C) + ")");
    }
}

Tensor nll_loss(const Tensor& log_probs,
                const Tensor& targets,
                const std::string& reduction) {
    validate_nll_loss(log_probs, targets, reduction);

#ifdef WITH_CUDA
    if (log_probs.device().is_cuda()) {
        Tensor out = cuda_nll_loss_forward(log_probs, targets, reduction);
        if (is_grad_enabled() && log_probs.requires_grad()) {
            out.requires_grad_(true);
            out.set_is_leaf(false);
            out.set_grad_fn(std::make_shared<NllLossNode>(log_probs, targets, reduction));
        }
        return out;
    }
#endif

    if (!log_probs.device().is_cpu())
        throw std::runtime_error("nll_loss: unsupported device");
    validate_targets_cpu(log_probs, targets);

    const int64_t N = log_probs.shape()[0];
    const int64_t C = log_probs.shape()[1];
    const float* src = static_cast<const float*>(log_probs.data());

    Tensor out(std::make_shared<TensorImpl>(
        reduction == "none" ? std::vector<int64_t>{N} : std::vector<int64_t>{1},
        DType::Float32,
        log_probs.device()));
    float* dst = static_cast<float*>(out.data());

    if (reduction == "none") {
        for (int64_t i = 0; i < N; ++i)
            dst[i] = -src[i * C + read_target(targets, i)];
    } else {
        double total = 0.0;
        for (int64_t i = 0; i < N; ++i)
            total += -static_cast<double>(src[i * C + read_target(targets, i)]);
        if (reduction == "mean")
            total /= static_cast<double>(N);
        dst[0] = static_cast<float>(total);
    }

    if (is_grad_enabled() && log_probs.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<NllLossNode>(log_probs, targets, reduction));
    }
    return out;
}

Tensor nll_loss_backward(const Tensor& log_probs,
                         const Tensor& targets,
                         const Tensor& grad_output,
                         const std::string& reduction) {
    validate_nll_loss(log_probs, targets, reduction);
    if (grad_output.dtype() != DType::Float32)
        throw std::runtime_error("nll_loss backward: grad_output must be float32");
    if (grad_output.device() != log_probs.device())
        throw std::runtime_error("nll_loss backward: device mismatch");
    if (!grad_output.is_contiguous())
        throw std::runtime_error("nll_loss backward: grad_output must be contiguous");
    if (reduction == "none" && (grad_output.ndim() != 1 || grad_output.shape()[0] != log_probs.shape()[0]))
        throw std::runtime_error("nll_loss backward: grad_output shape mismatch for reduction='none'");
    if (reduction != "none" && grad_output.numel() != 1)
        throw std::runtime_error("nll_loss backward: scalar grad_output expected");

#ifdef WITH_CUDA
    if (log_probs.device().is_cuda())
        return cuda_nll_loss_backward(log_probs, targets, grad_output, reduction);
#endif

    if (!log_probs.device().is_cpu())
        throw std::runtime_error("nll_loss backward: unsupported device");
    validate_targets_cpu(log_probs, targets);

    const int64_t N = log_probs.shape()[0];
    const int64_t C = log_probs.shape()[1];
    Tensor grad(std::make_shared<TensorImpl>(
        log_probs.shape(), DType::Float32, log_probs.device()));
    std::memset(grad.data(), 0, static_cast<size_t>(grad.numel()) * sizeof(float));

    const float* gout = static_cast<const float*>(grad_output.data());
    float* dst = static_cast<float*>(grad.data());
    const float mean_scale = reduction == "mean"
        ? 1.0f / static_cast<float>(N)
        : 1.0f;

    for (int64_t i = 0; i < N; ++i) {
        const float upstream = reduction == "none" ? gout[i] : gout[0];
        dst[i * C + read_target(targets, i)] = -upstream * mean_scale;
    }
    return grad;
}

}  // namespace orthon
