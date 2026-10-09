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
static int64_t read_target(const Tensor& targets, int64_t i) {
    if (targets.dtype() == DType::Int64)
        return static_cast<const int64_t*>(targets.data())[i];
    return static_cast<int64_t>(static_cast<const int32_t*>(targets.data())[i]);
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
Tensor cpu_nll_loss(const Tensor& log_probs,
                const Tensor& targets,
                const std::string& reduction) {
    validate_nll_inputs(log_probs, targets, reduction);
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

    return out;
}
Tensor cpu_nll_loss_backward(const Tensor& log_probs,
                         const Tensor& targets,
                         const Tensor& grad_output,
                         const std::string& reduction) {
    validate_nll_inputs(log_probs, targets, reduction, &grad_output);
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

} } }
