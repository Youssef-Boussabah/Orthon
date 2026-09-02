#include <orthon/ops/cpu/cross_entropy_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/autograd/cross_entropy_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/dtype.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/cross_entropy_kernels.h>
#endif

namespace orthon {

static void validate_cross_entropy(const Tensor& logits,
                                    const std::vector<int64_t>& targets) {
    if (logits.ndim() != 2)
        throw std::runtime_error("cross_entropy: logits must be 2D");
    if (logits.dtype() != DType::Float32)
        throw std::runtime_error("cross_entropy: logits must be float32");

    const int64_t B = logits.shape()[0];
    const int64_t C = logits.shape()[1];

    if (B == 0)
        throw std::runtime_error("cross_entropy: empty batch");
    if (static_cast<int64_t>(targets.size()) != B)
        throw std::runtime_error(
            "cross_entropy: targets length (" +
            std::to_string(targets.size()) + ") does not match batch_size (" +
            std::to_string(B) + ")");

    for (int64_t i = 0; i < static_cast<int64_t>(targets.size()); ++i) {
        if (targets[i] < 0 || targets[i] >= C)
            throw std::runtime_error(
                "cross_entropy: target " + std::to_string(targets[i]) +
                " at index " + std::to_string(i) +
                " is out of range [0, " + std::to_string(C) + ")");
    }
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets) {
    validate_cross_entropy(logits, targets);

    float loss_val;

#ifdef WITH_CUDA
    if (logits.device().is_cuda()) {
        if (!logits.is_contiguous())
            throw std::runtime_error(
                "cross_entropy: CUDA path requires contiguous logits");
        Tensor out = cuda_cross_entropy_loss(logits, targets);
        if (is_grad_enabled() && logits.requires_grad()) {
            out.requires_grad_(true);
            out.set_is_leaf(false);
            out.set_grad_fn(std::make_shared<CrossEntropyNode>(logits, targets));
        }
        return out;
    } else
#endif
    {
        if (!logits.device().is_cpu())
            throw std::runtime_error("cross_entropy: unsupported device");
        if (!logits.is_contiguous())
            throw std::runtime_error(
                "cross_entropy: CPU path requires contiguous logits");

        const int64_t B   = logits.shape()[0];
        const int64_t C   = logits.shape()[1];
        const float*  src = static_cast<const float*>(logits.data());

        double total = 0.0;
        for (int64_t i = 0; i < B; ++i) {
            const float* row  = src + i * C;
            float        rmax = *std::max_element(row, row + C);

            float sum_exp = 0.0f;
            for (int64_t j = 0; j < C; ++j)
                sum_exp += std::exp(row[j] - rmax);

            total += -static_cast<double>(row[targets[i]]) + rmax +
                     std::log(static_cast<double>(sum_exp));
        }
        loss_val = static_cast<float>(total / B);
    }

    Tensor out = full({1}, loss_val);
    if (is_grad_enabled() && logits.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<CrossEntropyNode>(logits, targets));
    }
    return out;
}

}  // namespace orthon
