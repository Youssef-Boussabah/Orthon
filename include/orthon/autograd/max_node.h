#pragma once
#include <orthon/autograd/node.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <cstdint>
#include <optional>
#include <vector>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/reduce_kernels.h>
#endif

namespace orthon {

class MaxNode : public Node {
public:
    std::optional<int64_t> dim_;
    bool keepdim_;
    std::vector<int64_t> input_shape_;
    Tensor argmax_indices_;
    int64_t outer_, inner_, reduce_;

    MaxNode(const Tensor& input,
            const Tensor& argmax_indices,
            std::optional<int64_t> dim,
            bool keepdim,
            int64_t outer,
            int64_t inner,
            int64_t reduce)
        : dim_(dim), keepdim_(keepdim),
          input_shape_(input.shape()),
          argmax_indices_(argmax_indices),
          outer_(outer), inner_(inner), reduce_(reduce) {
        inputs = {input};
        // Backward scatters grad_output by the SAVED argmax buffer's VALUES;
        // the input payload is never re-read (only input_shape_, dtype and
        // device), so mutating the input after max() stays legal while the
        // saved index buffer itself is watched.
        watch_version(argmax_indices_, "argmax_indices");
    }

    const char* op_name() const override { return "MaxNode"; }

    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Backward kernels index grad_output linearly: CPU variants reject a strided
        // gradient, CUDA variants read storage order and corrupt silently. Normalise
        // once at the Node boundary; no-op when already contiguous.
        const Tensor grad = contiguous(grad_output);
        const Tensor& input = inputs[0];

        const DType dt = input.dtype();

        if (!dim_.has_value()) {
            // Full reduction: argmax_indices_ holds a single flat index.
            Tensor idx_cpu = argmax_indices_.device().is_cuda()
                ? argmax_indices_.to(Device{DeviceType::CPU, 0})
                : argmax_indices_;
            Tensor g_cpu = grad.device().is_cuda()
                ? grad.to(Device{DeviceType::CPU, 0})
                : grad;
            int64_t flat_idx = *static_cast<const int64_t*>(idx_cpu.data());

            Tensor grad_in = zeros(input_shape_, dt, Device{});
            if (dt == DType::Float64) {
                double g = *static_cast<const double*>(g_cpu.data());
                static_cast<double*>(grad_in.data())[flat_idx] = g;
            } else if (dt == DType::Float16 || dt == DType::BFloat16) {
                uint16_t raw = *static_cast<const uint16_t*>(g_cpu.data());
                static_cast<uint16_t*>(grad_in.data())[flat_idx] = raw;
            } else {
                float g = *static_cast<const float*>(g_cpu.data());
                static_cast<float*>(grad_in.data())[flat_idx] = g;
            }
            if (input.device().is_cuda())
                grad_in = grad_in.to(input.device());
            return {grad_in};
        }

#ifdef WITH_CUDA
        if (input.device().is_cuda()) {
            Tensor grad_in = zeros(input_shape_, dt, Device{});
            grad_in = grad_in.to(input.device());
            cuda_max_scatter_grad(grad, argmax_indices_, grad_in,
                                  outer_, reduce_, inner_);
            return {grad_in};
        }
#endif
        // CPU dim-reduction scatter.
        Tensor grad_in = zeros(input_shape_, dt, Device{});
        const int64_t* idx_ptr = static_cast<const int64_t*>(argmax_indices_.data());
        const int64_t  n_out   = outer_ * inner_;
        if (dt == DType::Float64) {
            const double* go_ptr = static_cast<const double*>(grad.data());
            double*       gi_ptr = static_cast<double*>(grad_in.data());
            for (int64_t i = 0; i < n_out; ++i) {
                int64_t outer_i = i / inner_;
                int64_t inner_i = i % inner_;
                int64_t r       = idx_ptr[i];
                gi_ptr[outer_i * reduce_ * inner_ + r * inner_ + inner_i] = go_ptr[i];
            }
        } else {
            const float* go_ptr = static_cast<const float*>(grad.data());
            float*       gi_ptr = static_cast<float*>(grad_in.data());
            for (int64_t i = 0; i < n_out; ++i) {
                int64_t outer_i = i / inner_;
                int64_t inner_i = i % inner_;
                int64_t r       = idx_ptr[i];
                gi_ptr[outer_i * reduce_ * inner_ + r * inner_ + inner_i] = go_ptr[i];
            }
        }
        return {grad_in};
    }
};

}  // namespace orthon
